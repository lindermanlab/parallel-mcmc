import jax
jax.config.update('jax_default_matmul_precision', 'highest')
import jax.numpy as jnp
import jax.random as jr
from jax.scipy.special import logsumexp

from src import samplers

from functools import partial
import time

import matplotlib.pyplot as plt 

import tensorflow_datasets as tfds
from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions

import matplotlib.pyplot as plt 
plt.ion()

key = jr.PRNGKey(123)
import numpy as np 
d_data = np.load("../static/imdb1024.npz")
y = jnp.array(d_data['train_labels'])
X = jnp.array(d_data['train_emb'])

def joint_log_prob(x, y, beta, params):
    ridge_coef = params['ridge_coef']
    lp = tfd.Normal(0., ridge_coef).log_prob(beta).sum()
    logits = x @ beta
    lp += tfd.Bernoulli(logits).log_prob(y).sum()
    return lp
logp = partial(joint_log_prob, X, y)
D = X.shape[-1]
target_log_prob_and_grad = jax.value_and_grad(logp)

# compute basis using feature matrix
def compute_P(X):
    return jnp.linalg.svd(X.T @ X)[0]
P = compute_P(X)

# Define chain
batch_size = 1
chain_length = 1024
step_size = 0.015
max_iter = 1024 # max number of parallel iters
window_size = 256 #256
key, *skeys = jr.split(key, 3)
initial_state = 1.0 / jnp.sqrt(D) * jr.normal(skeys[0], (batch_size, D,))
batch_keys = jr.split(skeys[0], batch_size)

params = {}
params["step_size"] = step_size
key, skey = jr.split(key)
params["key"] = skey # random key for stochastic quasi deer
params["basis"] = jnp.eye(D)
params["target_params"] = {'ridge_coef':1.0}

sampler = samplers.ParallelMALA(logp, D, chain_length, max_iter, 
    full_trace=False, basis_transformation=True, window_size=window_size)

batch_sequential = jax.jit(jax.vmap(sampler.run_sequential_mala, in_axes=(0,0,None)))
batch_parallel = jax.jit(jax.vmap(sampler.run_parallel_mala_window, in_axes=(0,0,0,None)))

states_seq = batch_sequential(batch_keys, initial_state, params)

accept_ratio = 1.0 - jnp.mean(states_seq[:,1:,0]==states_seq[:,:-1,0])
print("Accept ratio: ", accept_ratio)

params["basis"] = P
yinit_guess = initial_state[:, None, :] * jnp.ones((batch_size, chain_length, D))

states_par, iters = batch_parallel(batch_keys, initial_state, yinit_guess, params)
print(f"Parallel samplers converged in {iters} iters")

batch_idx = 0
dim = 0
plt.figure()
plt.plot(states_seq[batch_idx][:,dim], 'r', label="sequential", alpha=0.8)
plt.plot(states_par[batch_idx][:,dim], 'b:', label="parallel", alpha=0.8)
plt.xlabel("sample iteration")
plt.ylabel("dim " + str(dim))
plt.title("Parallel samples at convergence vs. sequential samples")
plt.legend()
plt.show()

mae = jnp.max(jnp.abs(states_par-states_seq))
print("Max Abs Error: ", mae)