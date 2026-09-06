"""
Wall-clock benchmark for examples/run_hmc_rosenbrock.py: sequential HMC vs. Picard vs.
DEER (damped Newton, the solver run_hmc_rosenbrock.py uses) vs. quasi-DEER (diagonal Jacobian).

Reports (a) time to convergence and (b) time of a single iteration, and saves a figure to
figures/hmc_rosenbrock_timing.png plus the raw numbers to figures/hmc_rosenbrock_timing.json.
Run from the repo root on a GPU.
"""
import jax
jax.config.update('jax_default_matmul_precision', 'highest')

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import json, os, platform, subprocess, time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import samplers, picard, deer
from inference_gym import using_jax as gym

# ---------------------------------------------------------------- same setup as run_hmc_rosenbrock.py
target = gym.targets.VectorModel(gym.targets.Banana(), flatten_sample_transformations=True)
D = target.event_shape[0]

def target_log_prob(x):
    y = target.default_event_space_bijector(x)
    fldj = target.default_event_space_bijector.forward_log_det_jacobian(x)
    return target.unnormalized_log_prob(y) + fldj

chain_length = 100000
key = jr.PRNGKey(1313)
key, skey = jr.split(key)
initial_state = 0. + 10. * jr.normal(skey, (D,))
max_iter = chain_length
damp_factor = 0.55
params = {"epsilon": 0.5, "num_leapfrog_steps": 8}
yinit_guess = initial_state[None, :] * jnp.ones((chain_length, D))
drivers = jr.split(key, (chain_length,))

sampler = samplers.ParallelHMC(target_log_prob, D, chain_length, max_iter,
                               full_trace=False, damp_factor=damp_factor)
run_sequential = jax.jit(sampler.run_sequential_hmc)
run_parallel = jax.jit(sampler.run_parallel_hmc)
run_picard = jax.jit(sampler.run_picard_hmc)

def run_quasi(key, initial_state, yinit_guess, params):
    drivers = jr.split(key, (chain_length,))
    return deer.seq1d(sampler.hmc_fxn_for_deer, initial_state, drivers, params,
                      yinit_guess=yinit_guess, max_iter=max_iter, quasi=True, qmem_efficient=False,
                      clip_val=sampler.clip_val, full_trace=False, damp_factor=damp_factor)
run_quasi = jax.jit(run_quasi)

# ---------------------------------------------------------------- timing helpers
N_REPEATS = 5

def timeit(fn, *args):
    """Compile once, then return the median wall-clock of N_REPEATS runs (seconds) and the last output."""
    out = fn(*args); jax.block_until_ready(out)
    ts = []
    for _ in range(N_REPEATS):
        t0 = time.perf_counter(); out = fn(*args); jax.block_until_ready(out)
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts)), out

dev = jax.devices()[0]
gpu_name = dev.device_kind
print("Device:", dev, "|", gpu_name)

results = {"gpu": gpu_name, "jax": jax.__version__, "chain_length": chain_length,
           "num_leapfrog_steps": params["num_leapfrog_steps"], "epsilon": params["epsilon"],
           "damp_factor": damp_factor, "n_repeats": N_REPEATS, "methods": {}}

# --- sequential
t_seq, states_seq = timeit(run_sequential, key, initial_state, params)
# one HMC transition on its own (kernel-launch bound); amortized = t_seq / chain_length
one_step = jax.jit(lambda s, d: sampler.hmc_fxn_for_deer(s, d, params))
t_seq_step, _ = timeit(one_step, initial_state, drivers[0])
results["methods"]["sequential"] = dict(time_to_convergence=t_seq, iters=chain_length,
    time_per_iter=t_seq / chain_length, time_per_iter_standalone=t_seq_step, converged=True, max_err=0.0)
print(f"Sequential: {t_seq:.3f}s total, {t_seq/chain_length*1e6:.1f} us per HMC step (amortized), "
      f"{t_seq_step*1e3:.3f} ms per HMC step (standalone jit)")

# --- DEER (damped Newton, full Jacobian) and quasi-DEER
def single_newton_iter_time(quasi):
    """Time of one Newton iteration via the slope of a full_trace scan with 1 vs. 1+k iterations."""
    k = 5
    def make(n):
        def f(key, initial_state, yinit_guess, params):
            drivers = jr.split(key, (chain_length,))
            return deer.seq1d(sampler.hmc_fxn_for_deer, initial_state, drivers, params,
                              yinit_guess=yinit_guess, max_iter=n, quasi=quasi, qmem_efficient=False,
                              clip_val=sampler.clip_val, full_trace=True, damp_factor=damp_factor)[0]
        return jax.jit(f)
    t1, _ = timeit(make(1), key, initial_state, yinit_guess, params)
    t2, _ = timeit(make(1 + k), key, initial_state, yinit_guess, params)
    return (t2 - t1) / k

for name, fn, quasi in [("deer", run_parallel, False), ("quasi_deer", run_quasi, True)]:
    t, (states, iters) = timeit(fn, key, initial_state, yinit_guess, params)
    iters = int(iters)
    err = float(jnp.max(jnp.abs(states - states_seq)))
    t_iter = single_newton_iter_time(quasi)
    results["methods"][name] = dict(time_to_convergence=t, iters=iters, time_per_iter=t_iter,
        time_per_iter_amortized=t / iters, converged=iters < max_iter, max_err=err)
    print(f"{name}: {t:.3f}s, {iters} iters (converged: {iters < max_iter}), "
          f"{t_iter*1e3:.2f} ms per Newton iteration, max err vs. sequential {err:.3e}")

# --- Picard
t_pic, (states_pic, iters_pic) = timeit(run_picard, key, initial_state, yinit_guess, params)
iters_pic = int(iters_pic)
err_pic = float(jnp.max(jnp.abs(states_pic - states_seq)))
pic_step = jax.jit(lambda yt: picard.picard_step(sampler.hmc_fxn_for_deer, initial_state, yt, drivers, params))
t_pic_step, _ = timeit(pic_step, yinit_guess)
results["methods"]["picard"] = dict(time_to_convergence=t_pic, iters=iters_pic, time_per_iter=t_pic_step,
    time_per_iter_amortized=t_pic / iters_pic, converged=iters_pic < max_iter, max_err=err_pic)
print(f"Picard: {t_pic:.3f}s, {iters_pic} iters (converged: {iters_pic < max_iter}), "
      f"{t_pic_step*1e3:.2f} ms per Picard sweep, max err vs. sequential {err_pic:.3e}")

os.makedirs("figures", exist_ok=True)
with open("figures/hmc_rosenbrock_timing.json", "w") as f:
    json.dump(results, f, indent=2)

# ---------------------------------------------------------------- plot
order = ["sequential", "picard", "deer", "quasi_deer"]
labels = {"sequential": "Sequential HMC\n(lax.scan)",
          "picard": "Picard\n(Jacobian = I)",
          "deer": "DEER, damped Newton\n(full Jacobian; run_hmc_rosenbrock.py)",
          "quasi_deer": "quasi-DEER\n(diagonal Jacobian)"}
colors = {"sequential": "#2a78d6", "picard": "#eb6834", "deer": "#1baf7a", "quasi_deer": "#eda100"}
text_primary, text_secondary, grid = "#0b0b0b", "#52514e", "#e6e5e1"

def fmt_time(t):
    if t < 1e-3: return f"{t*1e6:.0f} µs"
    if t < 1: return f"{t*1e3:.1f} ms"
    return f"{t:.2f} s"

fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))
fig.patch.set_facecolor("#fcfcfb")
y = np.arange(len(order))[::-1]

for ax, key_, title, xlabel in [
        (axes[0], "time_to_convergence", "Time to convergence", "seconds (log scale)"),
        (axes[1], "time_per_iter", "Time of a single iteration", "seconds (log scale)")]:
    vals = [results["methods"][m][key_] for m in order]
    bars = ax.barh(y, vals, height=0.55, color=[colors[m] for m in order], edgecolor="#fcfcfb", linewidth=2)
    ax.set_xscale("log")
    ax.set_yticks(y); ax.set_yticklabels([labels[m] for m in order], fontsize=9, color=text_primary)
    ax.set_title(title, fontsize=12, color=text_primary, loc="left", fontweight="bold")
    ax.set_xlabel(xlabel, color=text_secondary)
    ax.set_facecolor("#fcfcfb")
    ax.grid(axis="x", color=grid, linewidth=0.8); ax.set_axisbelow(True)
    for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(grid); ax.tick_params(colors=text_secondary)
    xmax = max(vals)
    ax.set_xlim(min(vals) / 3, xmax * 12)
    for yi, m, v in zip(y, order, vals):
        r = results["methods"][m]
        if key_ == "time_to_convergence":
            n = r["iters"]
            unit = "HMC steps" if m == "sequential" else "iterations"
            note = f"{fmt_time(v)}  ({n:,} {unit}" + ("" if r["converged"] else ", hit max_iter: NOT converged") + ")"
        else:
            unit = {"sequential": "per HMC step (amortized)", "picard": "per Picard sweep",
                    "deer": "per Newton iteration", "quasi_deer": "per Newton iteration"}[m]
            note = f"{fmt_time(v)}  {unit}"
        ax.text(v * 1.15, yi, note, va="center", ha="left", fontsize=8.5, color=text_primary)

fig.suptitle(f"Parallel HMC on the Rosenbrock/Banana target: {chain_length:,} samples, "
             f"{params['num_leapfrog_steps']} leapfrog steps, D={D}   |   {gpu_name}, JAX {jax.__version__}",
             fontsize=10.5, color=text_secondary, x=0.01, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.94), w_pad=3)
fig.savefig("figures/hmc_rosenbrock_timing.png", dpi=160, facecolor=fig.get_facecolor())
print("saved figures/hmc_rosenbrock_timing.png")
