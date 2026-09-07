"""
Parallel HMC on the Rosenbrock (Banana) target: damped full-Jacobian DEER (Newton) vs. Picard
(Jacobian damp_factor * I, with damp_factor chosen by a sweep) vs. Jacobi (Jacobian 0), all
against the sequential chain.
Writes figures/hmc_rosenbrock_{trace,error,iterations}.png and, in the style of
static/rosenbrock.gif, figures/rosenbrock_{deer,picard,jacobi,sequential}.gif.
"""
import jax
jax.config.update('jax_default_matmul_precision', 'highest')

import jax.numpy as jnp
import jax.random as jr
import numpy as np

from src import samplers

import os
import time

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from PIL import Image

from inference_gym import using_jax as gym

from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions

target = gym.targets.VectorModel(gym.targets.Banana(),
                                 flatten_sample_transformations=True)
D = target.event_shape[0]

def target_log_prob(x):
    """Unnormalized, unconstrained target density.
    This is a thin wrapper that applies the default bijectors so that we can
    ignore any constraints.
    """
    y = target.default_event_space_bijector(x)
    fldj = target.default_event_space_bijector.forward_log_det_jacobian(x)
    return target.unnormalized_log_prob(y) + fldj

# define chain
chain_length = 1000
key = jr.PRNGKey(1313)
key, skey = jr.split(key)
initial_state = 0. + 10. * jr.normal(skey, (D,))
max_iter = chain_length # max number of parallel iters (Picard and Jacobi need at most chain_length)
damp_factor = 0.55 # scales the Jacobian for Newton (DEER)

params = {}
params["epsilon"] = 0.5
params["num_leapfrog_steps"] = 8

def make_sampler(damp, max_iter=max_iter, full_trace=False):
    return samplers.ParallelHMC(target_log_prob, D, chain_length, max_iter,
        full_trace=full_trace, damp_factor=damp)
sampler = make_sampler(damp_factor)

run_sequential = jax.jit(sampler.run_sequential_hmc)
states_seq = run_sequential(key, initial_state, params)

accept_ratio = 1.0 - jnp.mean(states_seq[1:,0]==states_seq[:-1,0])
print("Accept ratio: ", accept_ratio)

# Jacobians of the transition along the sequential chain, J_t = df/dy at (y_{t-1}, driver_t): what DEER
# linearizes with at convergence, and what Picard's single scalar a*I has to stand in for
drivers = jr.split(key, (chain_length,))
y_prev = jnp.concatenate((initial_state[None], states_seq[:-1]), axis=0)
jacs = np.asarray(jax.vmap(jax.jacfwd(sampler.hmc_fxn_for_deer), in_axes=(0, 0, None))(y_prev, drivers, params))
eigs = np.sort(np.linalg.eigvals(jacs).real, axis=1)  # (chain_length, D), across-banana (stiff) eigenvalue first
print(f"Jacobians along the chain: mean spectral norm {np.linalg.norm(jacs, ord=2, axis=(1, 2)).mean():.3f}, "
      f"mean trace/D {np.trace(jacs, axis1=1, axis2=2).mean() / D:.3f}, "
      f"median eigenvalues {np.median(eigs, axis=0).round(2)}, 10th pct of the smaller one {np.percentile(eigs[:, 0], 10):.2f}")

# every parallel solver starts from the initial state broadcast along the chain
yinit_guess = initial_state[None, :] * jnp.ones((chain_length, D))

# Picard's damping. The mean spectral norm (~1.03) is set by the soft direction along the banana (eigenvalue
# ~0.9) and would not damp at all, while across the banana the eigenvalue is *negative* (~-0.3, -0.6 in the
# tail): the HMC trajectory rotates that coordinate by more than a quarter period. Linearizing with a*I, the
# error in a direction with eigenvalue lam contracts by |lam - a| / (1 - |a|) per sweep (the 1/(1-|a|) is
# Picard's geometric memory of earlier mismatches), so a > 0 helps the soft direction a little and hurts the
# stiff direction a lot; above a ~ 0.2-0.35 the stiff direction stops contracting and Picard only finishes by
# its one-index-per-sweep exact front. Sweep a and keep the best (a = 0 is Jacobi).
print("Picard damp factor sweep:")
picard_iters = {}
for a in [0.0, 0.1, 0.2, 0.3, 0.4, damp_factor]:
    picard_iters[a] = int(jax.jit(make_sampler(a).run_picard_hmc)(key, initial_state, yinit_guess, params)[1])
    print(f"  damp_factor = {a:.2f}: {picard_iters[a]} iters")
picard_damp = min(picard_iters, key=picard_iters.get)

solvers = {  # name: (ParallelHMC method, damp_factor, plot color)
    "DEER": ("run_parallel_hmc", damp_factor, 'tab:blue'),
    "Picard": ("run_picard_hmc", picard_damp, 'tab:brown'),
    "Jacobi": ("run_jacobi_hmc", damp_factor, 'tab:green'),  # damping is irrelevant for Jacobi
}
label = {"DEER": f"DEER (damp {damp_factor})", "Picard": f"Picard (damp {picard_damp})", "Jacobi": "Jacobi"}
states, iters = {}, {}
for name, (method, damp, _) in solvers.items():
    run = jax.jit(getattr(make_sampler(damp), method))
    states[name], iters[name] = run(key, initial_state, yinit_guess, params)
    # wall-clock (post-compilation) and max error vs. sequential
    t0 = time.time(); run(key, initial_state, yinit_guess, params)[0].block_until_ready()
    print(f"{label[name]}: {iters[name]} iters (converged: {iters[name] < max_iter}), {time.time()-t0:.3f}s, "
          f"max error vs. sequential: {jnp.max(jnp.abs(states[name] - states_seq)):.3e}")
os.makedirs("figures", exist_ok=True)

# difference from the sequential chain at convergence, Newton (DEER) vs. Picard vs. Jacobi
dim = 1
fig, axes = plt.subplots(1, 3, figsize=[18, 4], sharex=True, sharey=True)
for ax, (name, (_, _, color)) in zip(axes, solvers.items()):
    ax.plot(states[name][:,dim] - states_seq[:,dim], color=color, label=label[name])
    ax.set_title(f"{label[name]}: {iters[name]} iterations")
    ax.set_xlabel("sample iteration")
    ax.set_xlim([0, chain_length])
    ax.legend()
axes[0].set_ylabel(f"parallel - sequential ($x_{dim+1}$)")
fig.suptitle("Difference from sequential samples at convergence")
plt.tight_layout()
plt.savefig("figures/hmc_rosenbrock_trace.png", dpi=150)
plt.show()

# every iterate of each solver up to its convergence, initial guess first: (iters + 1, chain_length, D).
# max_iter and full_trace are constructor arguments, so build one sampler per solver
traces = {}
for name, (method, damp, _) in solvers.items():
    tracer = make_sampler(damp, max_iter=int(iters[name]), full_trace=True)
    traces[name], _ = jax.jit(getattr(tracer, method))(key, initial_state, yinit_guess, params)
# max abs error vs. sequential at every iteration
errs = {name: jnp.max(jnp.abs(tr - states_seq[None]), axis=(1, 2)) for name, tr in traces.items()}

plt.figure()
for name, (_, _, color) in solvers.items():
    plt.loglog(jnp.arange(1, len(errs[name])), errs[name][1:], color=color, label=label[name])
plt.xlabel("iteration")
plt.ylabel("max abs error vs. sequential")
plt.title("Convergence of parallel HMC solvers")
plt.legend()
plt.savefig("figures/hmc_rosenbrock_error.png", dpi=150)
plt.show()

plt.figure(figsize=[8,8])
for i, itr in enumerate([1, 10, 25, max(int(v) for v in iters.values())]):
    plt.subplot(2, 2, i+1)
    plt.plot(states_seq[:,0], states_seq[:,1], 'k', rasterized=True)
    for name, (_, _, color) in solvers.items():
        tr = traces[name][min(itr, len(traces[name]) - 1)]  # a solver that has converged holds its last iterate
        plt.plot(tr[:,0], tr[:,1], color=color, alpha=0.75, rasterized=True)
        if name == "DEER":  # keep the axis limits set by sequential + DEER; Picard may be off the chart
            xl, yl = plt.xlim(), plt.ylim()
    plt.xlim(xl); plt.ylim(yl)
    plt.xlabel("$x_1$", fontsize=16)
    plt.ylabel("$x_2$", fontsize=16)
    plt.title(f"Parallel Iteration {itr}", fontsize=24)
    if i == 0:
        plt.legend(['Sequential'] + [label[name] for name in solvers])
plt.suptitle(f'{chain_length:,} HMC Samples', fontsize=16, fontweight="bold")
plt.tight_layout()
plt.savefig("figures/hmc_rosenbrock_iterations.png", dpi=150)
plt.show()

# animate each solver's iterates against the sequential chain, and the sequential chain revealed one
# sample at a time
seq = np.asarray(states_seq)
t = np.arange(chain_length)
NUM_FRAMES = 140  # max distinct iterates / sample counts per gif
ONE_BY_ONE, PAUSE = 50, 1.0  # sequential: leading samples added one per frame, and seconds each is held

def _lims(x, pad=0.05):
    lo, hi = x.min(), x.max(); m = pad * (hi - lo)
    return lo - m, hi + m

def make_gif(path, snapshots, durations, titles, color, label, sequential=False):
    """
    One frame per snapshot (array of states), shown for the matching duration in seconds.
    Parallel solvers: x1 / x2 traces and the banana, with the sequential chain in gray behind the
    iterate. Sequential: banana only, marking the previous and new sample while the chain is still
    being revealed one sample at a time (identical when the proposal was rejected). Axis limits
    come from the sequential chain, so a diverged iterate is simply off-chart.
    """
    with plt.rc_context({"font.size": 18, "lines.linewidth": 2}):
        if sequential:
            fig = Figure(figsize=(9, 8.5), dpi=100)
            ax3 = fig.add_subplot()
            trace_axes = ()
        else:
            fig = Figure(figsize=(24, 8), dpi=100)
            gs = fig.add_gridspec(2, 2, width_ratios=[2.2, 1])
            ax1, ax2, ax3 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[:, 1])
            trace_axes = ((ax1, 0), (ax2, 1))
            for ax, d in trace_axes:
                ax.plot(t, seq[:, d], color='gray')
                ax.set_xlim(0, chain_length); ax.set_ylim(*_lims(seq[:, d]))
            ax3.plot(seq[:, 0], seq[:, 1], color='gray', label='sequential', rasterized=True)
            ax1.set_ylabel("$x_1$"); ax2.set_ylabel("$x_2$"); ax2.set_xlabel("sample")
        canvas = FigureCanvasAgg(fig)
        trace_lines = [ax.plot([], [], color=color)[0] for ax, _ in trace_axes]
        (l3,) = ax3.plot([], [], color=color, label=label, rasterized=True)
        if sequential:
            (m_prev,) = ax3.plot([], [], 'o', mfc='none', mec='k', mew=2, ms=12, label='previous sample')
            (m_new,) = ax3.plot([], [], 'o', color='k', ms=10, label='new sample')
        ax3.set_xlabel("$x_1$"); ax3.set_ylabel("$x_2$")
        ax3.set_xlim(*_lims(seq[:, 0])); ax3.set_ylim(*_lims(seq[:, 1]))
        ax3.legend(loc="upper center" if sequential else "upper left")
        title = fig.suptitle("", fontsize=24)
        fig.tight_layout(rect=[0, 0, 1, 0.94])

        frames = []
        for y, txt in zip(snapshots, titles):
            for (ax, d), l in zip(trace_axes, trace_lines):
                l.set_data(t, y[:, d])
            l3.set_data(y[:, 0], y[:, 1])
            if sequential:
                prev = y[-2:-1] if len(y) <= ONE_BY_ONE else y[:0]  # markers only while adding one sample at a time
                m_prev.set_data(prev[:, 0], prev[:, 1]); m_new.set_data(y[-1:, 0], y[-1:, 1])
            title.set_text(txt)
            canvas.draw()
            frames.append(Image.fromarray(np.asarray(canvas.buffer_rgba())).convert("RGB"))
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=[int(1000 * d) for d in durations], loop=0)
    print(f"wrote {path}: {len(frames)} frames, {sum(durations):.1f}s of animation")

def hold_ends(durations):  # hold the first / last frame a little longer
    durations[0] += 0.4; durations[-1] += 1.6
    return durations

for name, (_, _, color) in solvers.items():  # evenly spaced iterates, initial guess first
    tr = np.asarray(traces[name])
    ks = np.unique(np.round(np.linspace(0, len(tr) - 1, min(NUM_FRAMES, len(tr)))).astype(int))
    make_gif(f"figures/rosenbrock_{name.lower()}.gif", [tr[k] for k in ks], hold_ends([0.04] * len(ks)),
             [f"{chain_length:,} Samples, {label[name]} Iteration {k}" for k in ks], color, name.lower())
# one sample per frame for the first ONE_BY_ONE samples, then log-spaced sample counts up to the full chain
ns = list(range(1, ONE_BY_ONE + 1)) + [int(n) for n in np.unique(np.round(np.geomspace(ONE_BY_ONE + 1, chain_length, NUM_FRAMES)).astype(int))]
make_gif("figures/rosenbrock_sequential.gif", [seq[:n] for n in ns],
         hold_ends([PAUSE] * ONE_BY_ONE + [0.08] * (len(ns) - ONE_BY_ONE)),
         [f"{chain_length:,} Samples, Sequential Sample {n}" for n in ns], 'r', 'sequential', sequential=True)
