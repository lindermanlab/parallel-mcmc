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

key = jr.PRNGKey(13)

def load_dataset():
    import numpy as np
    """
    Dataset loading copied from inference_gym package
    https://github.com/tensorflow/probability/blob/main/spinoffs/inference_gym/inference_gym/internal/data.py
    """
    def _normalize_zero_mean_one_std(train, test):
        train = np.asarray(train)
        test = np.asarray(test)
        train_mean = train.mean(0, keepdims=True)
        train_std = train.std(0, keepdims=True)
        return (train - train_mean) / train_std, (test - train_mean) / train_std

    train_fraction=1.
    num_points = 1000
    num_train = int(num_points * train_fraction)
    num_test = num_points - num_train
    num_features = 24

    dataset = tfds.load('german_credit_numeric:1.*.*')
    features = []
    labels = []
    for entry in tfds.as_numpy(dataset)['train']:
        features.append(entry['features'])
        # We're reversing the labels to match what's in the original dataset,
        # rather the TFDS encoding.
        labels.append(1 - entry['label'])
    features = np.stack(features, axis=0)
    labels = np.stack(labels, axis=0)

    train_features = features[:num_train]
    test_features = features[num_train:]
    train_features, test_features = _normalize_zero_mean_one_std(
        train_features, test_features)
    train_labels = labels[:num_train].astype(np.int32)
    return train_features, train_labels

train_features, train_labels = load_dataset()
X = jnp.asarray(train_features)
y = jnp.asarray(train_labels)
# add intercept
X = jnp.hstack((X, jnp.ones((1000,1))))

def joint_log_prob(x, y, beta, params):
    lp = tfd.Normal(0., params["ridge_coef"]).log_prob(beta).sum()
    logits = x @ beta
    lp += tfd.Bernoulli(logits).log_prob(y).sum()
    return lp
logp = partial(joint_log_prob, X, y)
D = X.shape[-1]

# Define chain
batch_size = 2
chain_length = 2048
step_size = 0.0015
max_iter = 40 # max number of parallel iters
key, *skeys = jr.split(key, 3)
initial_state = 0.1 * jr.normal(skeys[0], (batch_size, D,))
batch_keys = jr.split(skeys[0], batch_size)

params = {}
params["step_size"] = step_size
key, skey = jr.split(key)
params["key"] = skey # random key for stochastic quasi deer
params["basis"] = jnp.eye(D)
params["target_params"] = {'ridge_coef':1.0}

sampler = samplers.ParallelMALA(logp, D, chain_length, max_iter, 
    full_trace=False, basis_transformation=True)

batch_sequential = jax.jit(jax.vmap(sampler.run_sequential_mala, in_axes=(0,0,None)))
batch_parallel = jax.jit(jax.vmap(sampler.run_parallel_mala, in_axes=(0,0,0,None)))

states_seq = batch_sequential(batch_keys, initial_state, params)

accept_ratio = 1.0 - jnp.mean(states_seq[:,1:,0]==states_seq[:,:-1,0])
print("Accept ratio: ", accept_ratio)

# NOTE: we estimate basis assuming we have access to a few sequential steps. 
# This is reasonable, as we also pre-tuned the step size. 
# One could imagine a previous short chain was used to tune the step size and sequential 
n_warmup = 5
states_warmup = states_seq[:, n_warmup, :]
yinit_guess = initial_state[:, None, :] * jnp.ones((batch_size, chain_length, D))
hess = jax.hessian(logp)
H = hess(jnp.mean(states_warmup, axis=0), params['target_params'])
mean_J = jnp.eye(D) + step_size * H 
P = jnp.linalg.svd(mean_J)[0]
params["basis"] = P

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