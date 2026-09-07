# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

JAX code for the NeurIPS 2025 paper "Parallelizing MCMC Across the Sequence Length" (arXiv 2508.18413).
An MCMC chain is treated as a nonlinear recurrence `y[t+1] = f(y[t], driver[t], params)` where
`driver[t]` is a PRNG key. All `y[t]` are solved for simultaneously with (quasi-)Newton iterations
(DEER): linearize around the current guess, solve the resulting linear recurrence with
`jax.lax.associative_scan`, repeat until converged. GPU is needed for speedups; CPU works for testing.

## Setup and commands

```bash
pip install -U pip
# install JAX first (tested on 0.5.3 and 0.6.2), e.g. pip install -U "jax[cuda12]==0.6.2" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install -e .
```

Run examples from the repo root:
```bash
python examples/run_mala_mog.py             # parallel MALA, 2D mixture of Gaussians (quasi-DEER)
python examples/run_mala_german_credit.py   # parallel MALA, logistic regression (downloads dataset via tfds on first run)
python examples/run_hmc_rosenbrock.py       # parallel HMC (1K samples): damped DEER vs. Picard vs. Jacobi; also writes figures/*.png and rosenbrock_*.gif
python examples/run_gibbs_eight_schools.py  # parallel Gibbs, calls deer.seq1d directly
cd examples && python run_mala_sentiment.py # windowed quasi-DEER; MUST run from examples/ (loads ../static/imdb1024.npz)
```

There is no test suite and no linter configured. The verification loop is: run an example and
compare the parallel output against the sequential baseline (each script prints accept ratio,
Newton iteration count, and in some cases max abs error, then opens matplotlib plots; all but the
sentiment script block on `plt.show()`).

`nbs/rosenbrock.ipynb` draws a 1K-sample HMC chain on the Banana target and computes the Jacobian of
`hmc_fxn_for_deer` at every step for manual inspection (entries, eigenvalues, across/along-banana
projection). Jupyter is not a dependency of the package; the notebook only needs the example deps.

## Package layout gotcha

The importable package is literally named `src` (`pyproject.toml` finds packages in `.`), so code
does `from src import samplers`. Example scripts live in `examples/`, so `python examples/x.py`
puts `examples/` on `sys.path`, not the repo root; the editable install is what makes `src` resolve.

## Architecture

### Solver variants (`src/`)
All expose `seq1d(func, y0, xinp, params, ...) -> (states, iters)` with the same core knobs:
`yinit_guess` (shape `(chain_length, D)`), `max_iter`, `damp_factor` (scales the Jacobian),
`clip_val` (clips Jacobian entries), `preconditioner` (diagonal), `full_trace`.
Convergence is a relative tolerance (`tol=1e-4, rtol=1e-3` in float32; `1e-7, 1e-4` in float64).

- `deer.py` - reference implementation adapted from Lim et al.'s DEER (BSD-3). `quasi=False` uses
  the full Jacobian (`jax.jacfwd`) with a matrix associative scan; `quasi=True` uses only the
  diagonal. `qmem_efficient=True` estimates the diagonal with a Hutchinson/Rademacher JVP probe,
  `False` materializes the Jacobian and takes `jnp.diag`.
- `qdeer.py` - stripped-down stochastic quasi-DEER only (Hutchinson JVP estimator). This is what
  `ParallelMALA` uses. Requires `params["key"]` for the Rademacher probes.
- `windowed_qdeer.py` - quasi-DEER over a sliding window: `seq1d(func, y0, xinp, params, window, ...)`
  (note the extra positional `window` arg). Each iteration solves one window, then advances the
  window start past the first non-converged index; the loop ends when the window reaches the end
  of the chain, so `max_iter` bounds window iterations, not full-sequence sweeps.
- `elk.py` - Levenberg-Marquardt damped Newton solved with a parallel Kalman filter (`elk_alg`,
  `quasi_elk`). `sigmasq = 1/lambda`; large `sigmasq` recovers plain DEER. Imported in a few
  places but not currently called by the samplers or examples.
- `qdeer_leapfrog.py` - block-diagonal quasi-DEER for parallelizing the leapfrog steps inside a
  single HMC transition. Not wired into `samplers.py`.
- `picard.py` - Picard iteration ported from lindermanlab/micro_deer: Jacobian replaced by
  `damp_factor * I`. With `damp_factor=1` each sweep is a cumsum of `f(y[t-1]) - y[t-1]`; the
  recurrence `y[t] = a*y[t-1] + f(y_old[t-1]) - a*y_old[t-1]` is solved with
  `qdeer.diagonal_matmul_recursive`. Only converges when the transition is near `damp_factor * I`
  (e.g. HMC with tiny `epsilon`). Error in a direction with Jacobian eigenvalue `lam` contracts by
  `|lam - a| / (1 - |a|)` per sweep. On Rosenbrock HMC the eigenvalue along the banana is ~0.9 but
  across it is *negative* (~-0.3, -0.6 in the tail), so no scalar fits: the mean spectral norm (1.03)
  is useless as `a`, and `a >= ~0.35` (incl. DEER's 0.55, or undamped) stops contracting and only
  finishes via the one-index-per-sweep exact front (~`chain_length` sweeps). The example sweeps `a`
  and picks the best (0.2: 117 sweeps at 1K vs. Jacobi 154, DEER 74). Initial guess is irrelevant.
- `jacobi.py` - Jacobi iteration (also from micro_deer): Jacobian replaced by zero, so each sweep is
  just `vmap(f)` over the previous iterate, `y[t] <- f(y[t-1])`. Converges at the rate the chain
  forgets its past under common random numbers. `picard.iterate` holds the while_loop / full_trace
  scaffolding shared by both `seq1d`s.

Gotcha for any new solver: replace NaNs (and clip) in `yt_next` *before* computing `err`. A NaN
`err` makes `err > tol` false and silently exits the `while_loop`, reporting a false convergence.

`full_trace=True` switches from `lax.while_loop` (returns final iterate + iteration count) to
`lax.scan` over exactly `max_iter` steps and returns every Newton iterate with the initial guess
prepended; `iters` is then just `max_iter`. Examples use this to plot intermediate iterates.

### Samplers (`src/samplers.py`)
`ParallelMALA` and `ParallelHMC` wrap a `log_prob` into a one-step transition
`*_fxn_for_deer(state, driver, params)`. Key design points:
- Metropolis accept/reject uses `sigmoid_accept`, a straight-through estimator (hard step on the
  forward pass, sigmoid gradient), so the transition is differentiable and DEER can take Jacobians.
- Each class has `run_sequential_*` (a `lax.scan` baseline) and `run_parallel_*`. Drivers are
  `jr.split(key, chain_length)`; the same key gives identical sequential and parallel chains at
  convergence, which is how correctness is checked.
- `ParallelMALA` uses `qdeer.seq1d`, or `windowed_qdeer.seq1d` via `run_parallel_mala_window`
  when `window_size` is set. With `basis_transformation=True` the state is rotated by the
  orthogonal `params["basis"]` so the diagonal-Jacobian approximation is better; examples compute
  the basis from the SVD of the Hessian at a warm-up point or of `X.T @ X`.
- `ParallelHMC` uses `deer.seq1d(quasi=False)` (full Jacobian) with `damp_factor`; it also has
  `run_picard_hmc` and `run_jacobi_hmc` (same signature as `run_parallel_hmc`) built on
  `picard.seq1d` / `jacobi.seq1d`. Picard uses the sampler's `damp_factor` as its scalar Jacobian.
- The `alg` constructor argument is stored but unused. The `ParallelHMC` docstring is a stale copy
  of the MALA one.

### Conventions in examples
- `params` is a plain dict. MALA expects `step_size`, `key`, `basis`, `target_params`;
  HMC expects `epsilon`, `num_leapfrog_steps`. `log_prob` for MALA takes `(state, target_params)`;
  for HMC it takes `(state)` only.
- Batching over chains is `jax.jit(jax.vmap(sampler.run_parallel_*, in_axes=(0,0,0,None)))`.
- `jax_default_matmul_precision` is set to `"highest"` at import of `samplers.py` and at the top
  of every example; keep this when adding new scripts.
- A typical `yinit_guess` is the initial state broadcast along the chain.
