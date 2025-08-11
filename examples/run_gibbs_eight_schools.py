import jax
jax.config.update('jax_default_matmul_precision', 'highest')

import jax.numpy as jnp
import jax.random as jr
from jax.scipy.special import logsumexp

from functools import partial

import argparse
import time
import os
import numpy as np

from src.deer import seq1d
from src.elk import elk_alg, quasi_elk

from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions

import matplotlib.pyplot as plt 

# batch size 
B = 2
L = 10000

## Sample Data (modified 8 schools)
S = 8                   # number of schools
N_s = 20  # number of students per school
mu_0 = 0.0              # prior mean of the global effect
kappa_0 = 0.1           # prior concentration in the NIX prior
nu_0 = 0.1              # degrees of freedom for the prior on \tau^2
tausq_0 = 100.0         # prior mean of the variance \tau^2
alpha_0 = 0.1           # degrees of freedom for the prior on \sigma_s^2
sigmasq_0 = 10.0        # scale of the prior on \sigma_s^2

# Sample data
data_key = jr.PRNGKey(13) # fix this for data
x_bars = jnp.array([28., 8., -3., 7., -1., 1., 18., 12.])
sigma_bars = jnp.array([15., 10., 16., 11., 9., 11., 10., 18.])
xs = tfd.Normal(x_bars, jnp.sqrt(N_s) * sigma_bars).sample((N_s,), seed=data_key)

# z-score the samples
zs = (xs - xs.mean(axis=0)) / xs.std(axis=0)

# Rescale so they have the desired variance
xs = x_bars + jnp.sqrt(N_s) * sigma_bars * zs

## Initialize the Gibbs sampler with a draw from the prior
nu_N = nu_0 + S + 1
alpha_sig = alpha_0 + N_s

def sample_prior(key):
    ## Initialize the Gibbs sampler with a draw from the prior
    key, *skeys = jr.split(key, 5)
    nu_N = nu_0 + S + 1
    tausq = tfd.InverseGamma(0.5 * nu_N, nu_N * tausq_0 * 0.5).sample(seed=skeys[3])
    mu = tfd.Normal(mu_0, jnp.sqrt(tausq / kappa_0)).sample(seed=skeys[2])
    thetas = tfd.Normal(mu, jnp.sqrt(tausq)).sample((S,), seed=skeys[0])
    alpha_sig = alpha_0 + N_s
    sigmasq = tfd.InverseGamma(0.5 * alpha_sig, 0.5 * alpha_sig * sigmasq_0).sample((S,), seed=skeys[1])
    return jnp.concatenate((thetas, sigmasq, jnp.array([mu]), jnp.array([tausq])))

# define gibbs steps
# S = len(thetas)
alpha_N = alpha_0 + N_s
nu_N = nu_0 + S + 1

def gibbs_sample_thetas(tausq, mu, sigmasqs, xs, zs):
    # zs are standard normal
    v_theta = 1. / ((N_s / sigmasqs) + (1 / tausq))
    theta_hat = v_theta * ((xs.sum(axis=0) / sigmasqs) + mu / tausq)
    # thetas = Normal(theta_hat, torch.sqrt(v_theta)).sample()
    thetas = theta_hat + jnp.sqrt(v_theta) * zs
    return thetas

def gibbs_sample_sigmasq(alpha_0, sigmasq_0, thetas, xs, zs):
    # zs are inverse Gamma (0.5* alpha_N, 1)
    sigmasq_N = 1. / alpha_N * (alpha_0 * sigmasq_0
                               + jnp.sum((xs - thetas)**2, axis=0))
    # ScaledInvChiSq(alpha_N, sigmasq_N).sample()
    return 0.5 * alpha_N * sigmasq_N * zs

def gibbs_sample_mu(mu_0, kappa_0, tausq, thetas, zs):
    # zs are standard normal
    v_mu = tausq / (kappa_0 + S)
    mu_hat = (mu_0 * kappa_0 + thetas.sum()) / (kappa_0 + S)
    # return Normal(mu_hat, torch.sqrt(v_mu)).sample()
    return mu_hat + jnp.sqrt(v_mu) * zs

def gibbs_sample_tausq(nu_0, tausq_0, mu_0, kappa_0, mu, thetas, zs):
    # zs are inverse Gamma (0.5* nu_N, 1)
    tausq_N = 1. / nu_N * (nu_0 * tausq_0 + kappa_0 * (mu - mu_0)**2
                        + jnp.sum((thetas - mu)**2))
    # return ScaledInvChiSq(nu_N, tausq_N).sample()
    return 0.5 * nu_N * tausq_N * zs

def fxn_for_deer(state, driver, params):
    thetas, sigmasq, mu, tausq = jnp.split(state, (S, 2*S, 2*S+1))

    key, *skeys = jr.split(driver, 5)
    zs_thetas = jr.normal(skeys[0], (S,))
    zs_sigmasq = tfd.InverseGamma(0.5 * alpha_N, 1).sample((S,), seed=skeys[1])
    zs_mu = jr.normal(skeys[2], (1,))
    zs_tausq = tfd.InverseGamma(0.5 * nu_N, 1).sample((1,), seed=skeys[3])

    tausq = gibbs_sample_tausq(nu_0, tausq_0, mu_0, kappa_0, mu, thetas, zs_tausq)
    mu = gibbs_sample_mu(mu_0, kappa_0, tausq, thetas, zs_mu)
    thetas = gibbs_sample_thetas(tausq, mu, sigmasq, xs, zs_thetas)
    sigmasq = gibbs_sample_sigmasq(alpha_0, sigmasq_0, thetas, xs, zs_sigmasq)

    state = jnp.concatenate((thetas, sigmasq, mu, tausq))

    return state

# sample
key = jr.PRNGKey(1313)
key, *skeys = jr.split(key, 3)
initial_state = jax.vmap(sample_prior)(jr.split(skeys[0], (B,)))
drivers = jr.split(skeys[1], (B, L))
params = {}

# warm-up sampler
n_warmup = 3
for i in range(n_warmup):
    key, skey = jr.split(key)
    initial_state = jax.vmap(fxn_for_deer, in_axes=(0,0,None))(initial_state, jr.split(skey, (B,)), params)

# timing
def fxn_for_scan(state, driver):
  state = fxn_for_deer(state, driver, params)
  return state, state,

@jax.jit
def _run_model(initial_state, drivers):
  _, out_states = jax.lax.scan(fxn_for_scan, initial_state, drivers[1:])
  return out_states
run_model = jax.jit(jax.vmap(_run_model))
out_states = run_model(initial_state, drivers)

D_dim = initial_state.shape[-1]
yinit_guess = initial_state[:, None, :] * jnp.ones((B, L-1, D_dim))
max_deer_iter = 400
max_elk_iter = 400

batch_deer = jax.jit(jax.vmap(lambda init_state, drivers, yinit_guess : seq1d(
    fxn_for_deer, init_state, drivers[1:], params, yinit_guess=yinit_guess, 
    max_iter=max_deer_iter, quasi=False, qmem_efficient=False, full_trace=False)))
outputs_deer = batch_deer(initial_state, drivers, yinit_guess)

preconditioner = jnp.concatenate((10.0*jnp.ones((8,)), 1000*jnp.ones((8,)), 10*jnp.ones((2,))))
params['key']=jr.PRNGKey(321)
batch_qdeer = jax.jit(jax.vmap(lambda init_state, drivers, yinit_guess : seq1d(
    fxn_for_deer, init_state, drivers[1:], params, yinit_guess=yinit_guess, 
    max_iter=max_deer_iter, quasi=True, qmem_efficient=True, full_trace=False, 
    preconditioner=preconditioner, clip_val=1.0)))
outputs_qdeer = batch_qdeer(initial_state, drivers, yinit_guess)

print("qDEER iters: ", outputs_qdeer[-1])
print("DEER iters: ", outputs_deer[-1])

N_samples = out_states.shape[1]
b_idx = 0
thetas, sigmasq, mu, tausq = jnp.split(out_states[b_idx], (S, 2*S, 2*S+1), axis=1)
_, _, _, tausq_par = jnp.split(outputs_deer[0][b_idx], (S, 2*S, 2*S+1), axis=1)
burnin=0
plt.style.use('seaborn-v0_8-whitegrid')
fig, axs = plt.subplots(1, 2, figsize=(12, 4))
from matplotlib.lines import Line2D
axs[0].plot(np.arange(burnin, N_samples), 
            np.sqrt(tausq[burnin:]), label="seq", color='r', alpha=0.5)
axs[0].plot(np.arange(burnin, N_samples), 
            np.sqrt(tausq_par[burnin:]), label="par", color='b', alpha=0.5)
axs[0].set_xlabel("Iteration", fontsize=16)
# axs[0].legend(fontsize=16)
axs[0].set_ylabel(r"$\tau$ samples", fontsize=16)
axs[1].hist([np.sqrt(tausq[burnin:,0]), np.sqrt(tausq_par[burnin:,0])], 50, label=['seq', 'par'], alpha=0.75, ec='k', density=True, color=['r', 'b'])
axs[1].set_xlabel(r"$\tau$", fontsize=16)
axs[1].set_ylabel(r"$p(\tau \mid \mathbf{X}, \phi)$", fontsize=16)
# custom legend
custom_lines = [Line2D([], [], label="Parallel Gibbs", color='b', lw=2.0),
                Line2D([], [], label="Sequential Gibbs", color='r', lw=2.0)]
fig.legend(handles=custom_lines, bbox_to_anchor=(0.6, 0.1), ncol=2)
# After tight_layout, adjust the subplots to make room for the suptitle
plt.tight_layout()
plt.subplots_adjust(bottom=0.15)  # This creates space at the top for the suptitle
plt.show()