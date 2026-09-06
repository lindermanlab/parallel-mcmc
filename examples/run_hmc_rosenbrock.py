import jax
jax.config.update('jax_default_matmul_precision', 'highest')

import jax.numpy as jnp
import jax.random as jr

from src import samplers, picard

from functools import partial
import os
import time

import matplotlib.pyplot as plt 

from inference_gym import using_jax as gym

from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions

import matplotlib.pyplot as plt 

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
chain_length = 100000
key = jr.PRNGKey(1313)
key, skey = jr.split(key)
initial_state = 0. + 10. * jr.normal(skey, (D,))
max_iter = chain_length # max number of parallel iters (Picard needs at most chain_length)
damp_factor = 0.55

params = {}
params["epsilon"] = 0.5
params["num_leapfrog_steps"] = 8

sampler = samplers.ParallelHMC(target_log_prob, D, chain_length, max_iter, 
    full_trace=False, damp_factor=damp_factor)

run_sequential = jax.jit(sampler.run_sequential_hmc)
run_parallel = jax.jit(sampler.run_parallel_hmc)
run_picard = jax.jit(sampler.run_picard_hmc)

states_seq = run_sequential(key, initial_state, params)

accept_ratio = 1.0 - jnp.mean(states_seq[1:,0]==states_seq[:-1,0])
print("Accept ratio: ", accept_ratio)

yinit_guess = initial_state[None, :] * jnp.ones((chain_length, D))
states_par, iters = run_parallel(key, initial_state, yinit_guess, params)
print(f"Parallel (DEER) sampler: {iters} iters (converged: {iters < max_iter})")
states_pic, iters_pic = run_picard(key, initial_state, yinit_guess, params)
print(f"Picard sampler: {iters_pic} iters (converged: {iters_pic < max_iter})")

# compare parallel methods: wall-clock (post-compilation) and max error vs. sequential
t0 = time.time(); run_parallel(key, initial_state, yinit_guess, params)[0].block_until_ready()
print(f"DEER time: {time.time()-t0:.3f}s, max error vs. sequential: {jnp.max(jnp.abs(states_par - states_seq)):.3e}")
t0 = time.time(); run_picard(key, initial_state, yinit_guess, params)[0].block_until_ready()
print(f"Picard time: {time.time()-t0:.3f}s, max error vs. sequential: {jnp.max(jnp.abs(states_pic - states_seq)):.3e}")
os.makedirs("figures", exist_ok=True)

# visualize last 10K
dim = 1
plt.figure()
plt.plot(states_seq[:,dim], 'r', label="sequential", alpha=0.8)
plt.plot(states_par[:,dim], 'b:', label="parallel (DEER)", alpha=0.8)
plt.plot(states_pic[:,dim], '--', color='tab:brown', label="picard", alpha=0.8)
plt.xlabel("sample iteration")
plt.ylabel("states")
plt.xlim([chain_length-10010, chain_length+10])
plt.ylim([states_seq[:,dim].min()-1, states_seq[:,dim].max()+1])  # picard may diverge off-axis
plt.title("Parallel (DEER) and Picard samples vs. sequential")
plt.legend()
plt.savefig("figures/hmc_rosenbrock_trace.png", dpi=150)
plt.show()
# get full sample trace (DEER and Picard) for the first iters+1 iterations
max_iter = iters+1
sampler = samplers.ParallelHMC(target_log_prob, D, chain_length, max_iter, 
    full_trace=True, damp_factor=damp_factor)
run_parallel = jax.jit(sampler.run_parallel_hmc)
run_picard = jax.jit(sampler.run_picard_hmc)
states_par, iters = run_parallel(key, initial_state, yinit_guess, params)
states_pic_trace, _ = run_picard(key, initial_state, yinit_guess, params)

# max abs error vs. sequential at every iteration. DEER: read off the full trace above.
# Picard: scan over picard_step for the number of iterations it needed, keeping only the error
err_deer = jnp.max(jnp.abs(states_par - states_seq[None]), axis=(1, 2))
drivers = jr.split(key, (chain_length,))
def _picard_err_step(yt, _):
    yt = picard.picard_step(sampler.hmc_fxn_for_deer, initial_state, yt, drivers, params)
    return yt, jnp.max(jnp.abs(yt - states_seq))
_, err_pic = jax.jit(lambda: jax.lax.scan(_picard_err_step, yinit_guess, None, length=int(iters_pic)))()

plt.figure()
plt.loglog(jnp.arange(1, len(err_deer)), err_deer[1:], color='tab:blue', label="DEER")
plt.loglog(jnp.arange(1, len(err_pic)+1), err_pic, color='tab:brown', label="Picard")
plt.xlabel("iteration")
plt.ylabel("max abs error vs. sequential")
plt.title("Convergence of parallel HMC solvers")
plt.legend()
plt.savefig("figures/hmc_rosenbrock_error.png", dpi=150)
plt.show()

plt.figure(figsize=[8,8])
for i, itr in enumerate([1, 10, 25, max_iter]):
    plt.subplot(2, 2, i+1)
    plt.plot(states_seq[:,0], states_seq[:,1], 'k', rasterized=True)
    plt.plot(states_par[itr][:,0], states_par[itr][:,1], color='tab:blue', alpha=0.75, rasterized=True)
    # keep the axis limits set by sequential + DEER; Picard may be off the chart
    xl, yl = plt.xlim(), plt.ylim()
    plt.plot(states_pic_trace[itr][:,0], states_pic_trace[itr][:,1], color='tab:brown', alpha=0.75, rasterized=True)
    plt.xlim(xl); plt.ylim(yl)
    plt.xlabel("$x_1$", fontsize=16)
    plt.ylabel("$x_2$", fontsize=16)
    plt.title(f"Parallel Iteration {itr}", fontsize=24)
    if i == 0:
        plt.legend(['Sequential', 'DEER', 'Picard'])
plt.suptitle('100K HMC Samples', fontsize=16, fontweight="bold")
plt.tight_layout()
plt.savefig("figures/hmc_rosenbrock_iterations.png", dpi=150)
plt.show()
