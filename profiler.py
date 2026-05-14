import jax # note that we are using float32 by default.
jax.config.update("jax_default_matmul_precision", "highest") # removing this can 2x iters needed for quasi-Newton convergence.
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Float, Int, Bool, UInt32
from functools import partial
import numpy as onp

# let's give Zoe + NVIDIA the jit-compilable function itself, not the wrapper. Easier access.
from parallel_samplers import _sample_multiple_chains_jit, fhat_t_mala, hutchinson

# miscellaneous
import sys, os
from tqdm.autonotebook import tqdm
import time

# for command-line arguments
import argparse

'''
For initial wall-clock profiling for Zoe + NVIDIA, let's not use a sliding-window: i.e., window_size = chain_length.
- Let's also try to get them as much profiling results as possible.

Settings: batch-size (# of chains in parallel) and chain_length (# of samples per chain).
b - 1, 2, 4, 8, 16, 32, 64
chain_length - 1024, 2048, 4096, 8192, 16384, 32768, 65536
'''
# command-line arguments: how many chains do I want and how long is each chain? dimensions?
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--chain_length", type=int, default=1024)
args = parser.parse_args(); b, chain_length = args.batch_size, args.chain_length
max_iters, window_size = chain_length, chain_length

# directory to store HLO logs for (b, chain_length)
PATH = f"hlo_logs/b={b}_chain-length={chain_length}"
os.makedirs(PATH, exist_ok=True)


# load in the whitened data for bayesian logistic regression (BLR, german credit)
X, y = jnp.asarray(onp.loadtxt("data/X.txt")), jnp.asarray(onp.loadtxt("data/y.txt"))

# BLR prior variance + dimensionality of our data, orthogonal basis transformation Q
sigma_blr, d, Q = 1.0, 25, jnp.load("Q.npy")

# our target logp function
def logp(beta: Float[Array, "d"]) -> Float[Array, ""]:
    d = beta.shape[0]
    logits = X @ beta

    lp = (
        -0.5 * jnp.sum((beta / sigma_blr) ** 2)
        - d * jnp.log(sigma_blr)
        - 0.5 * d * jnp.log(2.0 * jnp.pi)
    )
    lp += jnp.sum(y * logits - jnp.logaddexp(0.0, logits))
    return lp

# what is the stepsize for MALA
epsilon = 0.0015

# sample some initial conditions + our RNG keys
key = jr.PRNGKey(858); k0, kn, ku, kh = jr.split(key, 4)
x0s = 0.1 * jr.normal(k0, shape=(b, d))
xs_init = jnp.tile(x0s[:,None,:], reps=(1, chain_length, 1))
keys_norm = jr.split(kn, b*chain_length).reshape((b, chain_length, 2))
keys_unif = jr.split(ku, b*chain_length).reshape((b, chain_length, 2))
keys_hutchinson = jr.split(kh, b*chain_length).reshape((b, chain_length, 2))


# get the logp + gradient eval, batch via vmap
logp_and_grad = jax.value_and_grad(logp)
logp_and_grad_batched = jax.vmap(fun=logp_and_grad, in_axes=0, out_axes=0)


# 1. start by getting StableHLO outputs (not exclusive to our current device! platform independent)
lowered = _sample_multiple_chains_jit.lower(
    x0s=x0s, xs_init=xs_init, 
    keys_norm=keys_norm, keys_unif=keys_unif, keys_hutchinson=keys_hutchinson, 
    logp_and_grad_batched=logp_and_grad_batched,
    epsilon=epsilon, beta=1.0, damp_factor=1.0, clip_val=1e8, rtol=1e-3, atol=1e-4,
    fhat_t_mcmc=fhat_t_mala, use_Q=True, Q=Q, chain_length=chain_length, W=window_size, n_probes=2, d=d, 
    transition_and_jacobian_estimator=hutchinson, full_trace=False, max_iters=max_iters)
with open(f"{PATH}/stablehlo.txt", "wt") as file:
    file.write(lowered.as_text())

'''
Question for Zoe + NVIDIA - there seems to be a lot of options for .compile(). How to use them?

Below, we save compiled results to text as HLO. 

Also, from memory_analysis():
    a. generated_code_size_in_bytes - compiled code on GPU size.
    b. argument_size_in_bytes - total size of inputs to my function after compilation on GPU.
    c. output_size_in_bytes - total size of all outputs from my function on GPU.
    d. temp_size_in_bytes - max. total size of all intermediate buffers / temp files
    e. alias_size_in_bytes - memory shared between inputs and outputs (to be negated/subtracted).

And, from cost_analysis(), intensity = cost_analysis["flops"] / cost_analysis["bytes accessed"]

*we'll write all of this to a .csv over all settings later.
'''
# 2. see how StableHLO is compiled to HLO for our NVIDIA H100 target device (80 GB HBM3)
compiled = lowered.compile()
with open(f"{PATH}/hlo.txt", "wt") as file:
    file.write(compiled.as_text())
mem_analysis, cost_analysis = compiled.memory_analysis(), compiled.cost_analysis()


# no. of warm-up + timing runs, each.
n_trials = 5

# It's very possible it won't work out due to memory explosion
try:

    # a. warm-up runs
    for _ in tqdm(range(n_trials)):
        
        # get our outputs, but make sure we block until ready.
        outputs_par = _sample_multiple_chains_jit(
            x0s=x0s, xs_init=xs_init, 
            keys_norm=keys_norm, keys_unif=keys_unif, keys_hutchinson=keys_hutchinson, 
            logp_and_grad_batched=logp_and_grad_batched,
            epsilon=epsilon, beta=1.0, damp_factor=1.0, clip_val=1e8, rtol=1e-3, atol=1e-4,
            fhat_t_mcmc=fhat_t_mala, use_Q=True, Q=Q, chain_length=chain_length, W=window_size, n_probes=2, d=d, 
            transition_and_jacobian_estimator=hutchinson, full_trace=False, max_iters=max_iters)
        samples_par, iters_par = jax.block_until_ready(outputs_par)

    # b. actual timing runs
    start = time.time()
    for _ in tqdm(range(n_trials)):
        # get our outputs, but make sure we block until ready.
        outputs_par = _sample_multiple_chains_jit(
            x0s=x0s, xs_init=xs_init, 
            keys_norm=keys_norm, keys_unif=keys_unif, keys_hutchinson=keys_hutchinson, 
            logp_and_grad_batched=logp_and_grad_batched,
            epsilon=epsilon, beta=1.0, damp_factor=1.0, clip_val=1e8, rtol=1e-3, atol=1e-4,
            fhat_t_mcmc=fhat_t_mala, use_Q=True, Q=Q, chain_length=chain_length, W=window_size, n_probes=2, d=d, 
            transition_and_jacobian_estimator=hutchinson, full_trace=False, max_iters=max_iters)
        samples_par, iters_par = jax.block_until_ready(outputs_par)
    end = time.time()

    # how long did each trial take on average?
    avg_time = (end - start) / n_trials
    max_iters_needed = int(iters_par.max())

except:

    # memory explosion, couldn't get timing results.
    avg_time = onp.nan
    max_iters_needed = onp.nan


# writing to our overall logs file
cols = ["b", "chain_length", "avg_time", "max_iters_needed", "code", "arguments", "outputs", "intermediates", "aliases", "intensity"]
if "summary.csv" not in os.listdir():
    with open("summary.csv", "wt") as file:
        file.write(",".join(cols) + "\n")
with open("summary.csv", "at") as file:
    file.write(
        ",".join(map(str, [b, chain_length, avg_time, max_iters_needed,
         mem_analysis.generated_code_size_in_bytes / 1e9, 
         mem_analysis.argument_size_in_bytes / 1e9, 
         mem_analysis.output_size_in_bytes / 1e9, 
         mem_analysis.temp_size_in_bytes / 1e9, 
         -mem_analysis.alias_size_in_bytes / 1e9, 
         cost_analysis["flops"] / cost_analysis["bytes accessed"]
        ])) + "\n")