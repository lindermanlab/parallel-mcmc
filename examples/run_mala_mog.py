import jax
jax.config.update('jax_default_matmul_precision', 'highest')
import jax.numpy as jnp
import jax.random as jr
from jax.scipy.special import logsumexp

from src import samplers

from functools import partial
import time

import matplotlib.pyplot as plt 

from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions

# define simple 2D gaussian mixture distribution
num_components=4
D=2
grid_points = jnp.array([-2.,2.])
means = jnp.array(jnp.meshgrid(grid_points, grid_points)).T.reshape(num_components, D)

def log_prob(x, params):
  num_components=4
  D=2
  grid_points = jnp.array([-2.,2.])
  means = jnp.array(jnp.meshgrid(grid_points, grid_points)).T.reshape(num_components, D)
  logps = tfd.MultivariateNormalDiag(loc=means, scale_diag=1.0*jnp.ones((D,))).log_prob(x) + jnp.log(1.0/num_components)
  logp = logsumexp(logps)
  return logp

# Define chain
key = jr.PRNGKey(13)
chain_length = 100000
step_size = 0.5
max_iter = 200 # max number of parallel iters
key, *skeys = jr.split(key, 3)
initial_state = jr.normal(skeys[0], (D,))
sample_key = skeys[1]

params = {}
params["step_size"] = step_size
key, skey = jr.split(key)
params["key"] = skey # random key for stochastic quasi deer
params["basis"] = jnp.eye(D)
params["target_params"] = {}

sampler = samplers.ParallelMALA(log_prob, D, chain_length, max_iter, 
    full_trace=False, basis_transformation=False)

run_sequential = jax.jit(sampler.run_sequential_mala)
run_parallel = jax.jit(sampler.run_parallel_mala)

states_seq = run_sequential(sample_key, initial_state, params)

accept_ratio = 1.0 - jnp.mean(states_seq[1:,0]==states_seq[:-1,0])
print("Accept ratio: ", accept_ratio)

key, skey = jr.split(key)
yinit_guess = jr.normal(skey, (chain_length, D))
states_par, iters = run_parallel(sample_key, initial_state, yinit_guess, params)
print(f"Parallel samplers converged in {iters} iters")

from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec

plt.style.use('seaborn-v0_8-whitegrid')
fig = plt.figure(figsize=(9, 3))
gs = gridspec.GridSpec(1,3)

ax1 = fig.add_subplot(gs[0])
ax1.plot(states_par[:, 0], states_par[:, 1], 'b.', markersize=2, alpha=0.01, rasterized=True)
ax1.set_xlim([-5,5])
ax1.set_ylim([-5,5])
ax1.set_xlabel("coordinate 1", fontsize=16)
ax1.set_ylabel("coordinate 2", fontsize=16)
ax1.set_title("100K Parallel Samples (Iter 50)", fontsize=16)
ax1.legend()
ax2 = fig.add_subplot(gs[1:])
plot_range = jnp.arange(90000, 100000-1)
ax2.plot(plot_range, states_par[1+plot_range, 0], 'b', label="parallel, iter 50", alpha=0.5, rasterized=True)
ax2.plot(plot_range, states_seq[plot_range, 0], 'r', label="seq", alpha=0.5, rasterized=True)
ax2.set_ylim([-6,6])
ax2.set_title("Last 10K Samples", fontsize=16)
ax2.set_ylabel("coordinate 1", fontsize=16)
ax2.set_xlabel("iteration", fontsize=16)
ax2.legend(loc='best')
plt.tight_layout()
plt.subplots_adjust(bottom=0.15)  # This creates space at the top for the suptitle
plt.show()