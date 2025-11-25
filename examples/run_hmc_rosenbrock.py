import jax
jax.config.update('jax_default_matmul_precision', 'highest')

import jax.numpy as jnp
import jax.random as jr

from src import samplers

from functools import partial
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
max_iter = 200 # max number of parallel iters
damp_factor = 0.55

params = {}
params["epsilon"] = 0.5
params["num_leapfrog_steps"] = 8

sampler = samplers.ParallelHMC(target_log_prob, D, chain_length, max_iter, 
    full_trace=False, damp_factor=damp_factor)

run_sequential = jax.jit(sampler.run_sequential_hmc)
run_parallel = jax.jit(sampler.run_parallel_hmc)

states_seq = run_sequential(key, initial_state, params)

accept_ratio = 1.0 - jnp.mean(states_seq[1:,0]==states_seq[:-1,0])
print("Accept ratio: ", accept_ratio)

yinit_guess = initial_state[None, :] * jnp.ones((chain_length, D))
states_par, iters = run_parallel(key, initial_state, yinit_guess, params)
print(f"Parallel samplers converged in {iters} iters")

# visualize last 10K
dim = 1
plt.figure()
plt.plot(states_seq[:,dim], 'r', label="sequential", alpha=0.8)
plt.plot(states_par[:,dim], 'b:', label="parallel", alpha=0.8)
plt.xlabel("sample iteration")
plt.ylabel("states")
plt.xlim([chain_length-10010, chain_length+10])
plt.title("Parallel samples at convergence vs. sequential samples")
plt.legend()
plt.show()
# get full sample trace 
max_iter = iters+1
sampler = samplers.ParallelHMC(target_log_prob, D, chain_length, max_iter, 
    full_trace=True, damp_factor=damp_factor)
run_parallel = jax.jit(sampler.run_parallel_hmc)
states_par, iters = run_parallel(key, initial_state, yinit_guess, params)

plt.figure(figsize=[8,8])
plt.subplot(221)
itr = 1
plt.plot(states_seq[:,0], states_seq[:,1], 'k', rasterized=True)
plt.plot(states_par[itr][:,0], states_par[itr][:,1], alpha=0.75, rasterized=True)
plt.xlabel("$x_1$", fontsize=16)
plt.ylabel("$x_2$", fontsize=16)
plt.title("Parallel Iteration 1", fontsize=24)
plt.legend(['Sequential', 'Parallel'])
itr = 10
plt.subplot(222)
plt.plot(states_seq[:,0], states_seq[:,1], 'k', rasterized=True)
plt.plot(states_par[itr][:,0], states_par[itr][:,1], alpha=0.75, rasterized=True)
plt.xlabel("$x_1$", fontsize=16)
plt.ylabel("$x_2$", fontsize=16)
plt.title("Parallel Iteration 10", fontsize=24)
itr = 25
plt.subplot(223)
plt.plot(states_seq[:,0], states_seq[:,1], 'k', rasterized=True)
plt.plot(states_par[itr][:,0], states_par[itr][:,1], alpha=0.75, rasterized=True)
plt.xlabel("$x_1$", fontsize=16)
plt.ylabel("$x_2$", fontsize=16)
plt.title("Parallel Iteration 25", fontsize=24)
itr = max_iter
plt.subplot(224)
plt.plot(states_seq[:,0], states_seq[:,1], 'k', rasterized=True)
plt.plot(states_par[itr][:,0], states_par[itr][:,1], alpha=0.75, rasterized=True)
plt.xlabel("$x_1$", fontsize=16)
plt.ylabel("$x_2$", fontsize=16)
plt.title(f"Parallel Iteration {max_iter}", fontsize=24)
plt.suptitle('100K HMC Samples', fontsize=16, fontweight="bold")
plt.tight_layout()
plt.show()