"""
Picard iteration for solving y[i + 1] = func(y[i], x[i], params) in parallel.

Ported from micro_deer (https://github.com/lindermanlab/micro_deer). Picard is the
special case of DEER where every Jacobian is replaced by the identity, so each
iteration reduces to a parallel cumulative sum instead of a linear solve. With
damp_factor < 1 the Jacobian is damp_factor * I instead (the same damping as
quasi-DEER), and the linear recurrence is solved with quasi-DEER's associative scan.
"""
from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp

from src.qdeer import diagonal_matmul_recursive


def picard_step(func, y0, yt, xinp, params, damp_factor=1.0):
    """
    One Picard step with Jacobian a * I (a = damp_factor):
    y_t = a * y_{t-1} + b_t with b_t = f(y^k_{t-1}) - a * y^k_{t-1}, and y_0 = f(y0) exactly.
    For a = 1 this is the cumulative sum of f(y^k_{t-1}) - y^k_{t-1}.
    """
    yt_prev = jnp.concatenate((y0[None, :], yt[:-1]), axis=0)  # (nsamples, ny)
    fs = jax.vmap(func, in_axes=(0, 0, None))(yt_prev, xinp, params)
    bs = fs[1:] - damp_factor * yt_prev[1:]  # (nsamples - 1, ny)
    mats = jnp.full_like(bs, damp_factor)
    yt_next = diagonal_matmul_recursive(mats, bs, fs[0])  # y[i + 1] = mats[i] * y[i] + bs[i]
    # same clipping / nan handling as deer.deer_iteration_helper (clip_ytnext=True)
    yt_next = jnp.clip(yt_next, a_min=-1e8, a_max=1e8)
    return jnp.where(jnp.isnan(yt_next), 0.0, yt_next)


def iterate(
    step: Callable[[jnp.ndarray], jnp.ndarray],
    yinit_guess: jnp.ndarray,
    max_iter: int,
    full_trace: bool,
):
    """
    Run yt <- step(yt) from yinit_guess until the relative change is below tolerance
    (lax.while_loop) or, if full_trace, for exactly max_iter steps (lax.scan).

    Shared by picard.seq1d and jacobi.seq1d. step must already have replaced nans
    in its output (see picard_step), otherwise a nan err ends the while_loop early.
    Returns (yt, iters) with the same convention as deer.seq1d.
    """
    dtype = yinit_guess.dtype
    # same tolerances as deer.deer_iteration_helper
    tol = 1e-7 if dtype == jnp.float64 else 1e-4
    rtol = 1e-4 if dtype == jnp.float64 else 1e-3

    def iter_func(carry):
        _, yt, iiter = carry
        yt_next = step(yt)
        err = jnp.max(jnp.abs(yt_next - yt) - rtol * jnp.abs(yt))
        return err, yt_next, iiter + 1

    def cond_func(carry):
        err, _, iiter = carry
        return jnp.logical_and(err > tol, iiter < max_iter)

    def scan_func(carry, _):
        new_carry = iter_func(carry)
        return new_carry, new_carry[1]

    init = (jnp.array(1e10, dtype=dtype), yinit_guess, jnp.array(0, dtype=jnp.int32))
    if full_trace:
        _, yt = jax.lax.scan(scan_func, init, None, length=max_iter)
        return jnp.vstack((yinit_guess[None, ...], yt)), max_iter
    else:
        _, yt, samp_iters = jax.lax.while_loop(cond_func, iter_func, init)
        return yt, samp_iters


def seq1d(
    func: Callable[[jnp.ndarray, Any, Any], jnp.ndarray],
    y0: jnp.ndarray,
    xinp: Any,
    params: Any,
    yinit_guess: Optional[jnp.ndarray] = None,
    max_iter: int = 10000,
    full_trace: bool = False,
    damp_factor: float = 1.0,
):
    """
    Solve y[i + 1] = func(y[i], x[i], params) with Picard iteration, using the
    Jacobian damp_factor * I (damp_factor=1 is plain Picard).

    Same interface and return convention as deer.seq1d: returns (y, iters), where y
    excludes the initial state and has shape (nsamples, ny). If full_trace, y has
    shape (max_iter + 1, nsamples, ny) with the initial guess prepended.
    """
    xinp_flat = jax.tree_util.tree_flatten(xinp)[0][0]
    if yinit_guess is None:
        yinit_guess = jnp.zeros(
            (xinp_flat.shape[0], y0.shape[-1]), dtype=xinp_flat.dtype
        )
    step = lambda yt: picard_step(func, y0, yt, xinp, params, damp_factor)
    return iterate(step, yinit_guess, max_iter, full_trace)
