"""
Picard iteration for solving y[i + 1] = func(y[i], x[i], params) in parallel.

Ported from micro_deer (https://github.com/lindermanlab/micro_deer). Picard is the
special case of DEER where every Jacobian is replaced by the identity, so each
iteration reduces to a parallel cumulative sum instead of a linear solve.
"""
from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp


def picard_step(func, y0, yt, xinp, params):
    """One Picard step: b_t = f(y_{t-1}) - y_{t-1}, then y_t = y_0 + cumsum(b)."""
    yt_prev = jnp.concatenate((y0[None, :], yt[:-1]), axis=0)  # (nsamples, ny)
    fs = jax.vmap(func, in_axes=(0, 0, None))(yt_prev, xinp, params)
    bs = fs - yt_prev
    bs = bs.at[0].set(fs[0])  # first element already includes y0
    return jnp.cumsum(bs, axis=0)


def seq1d(
    func: Callable[[jnp.ndarray, Any, Any], jnp.ndarray],
    y0: jnp.ndarray,
    xinp: Any,
    params: Any,
    yinit_guess: Optional[jnp.ndarray] = None,
    max_iter: int = 10000,
    full_trace: bool = False,
):
    """
    Solve y[i + 1] = func(y[i], x[i], params) with Picard iteration.

    Same interface and return convention as deer.seq1d: returns (y, iters), where y
    excludes the initial state and has shape (nsamples, ny). If full_trace, y has
    shape (max_iter + 1, nsamples, ny) with the initial guess prepended.
    """
    xinp_flat = jax.tree_util.tree_flatten(xinp)[0][0]
    if yinit_guess is None:
        yinit_guess = jnp.zeros(
            (xinp_flat.shape[0], y0.shape[-1]), dtype=xinp_flat.dtype
        )
    dtype = yinit_guess.dtype
    # same tolerances as deer.deer_iteration_helper
    tol = 1e-7 if dtype == jnp.float64 else 1e-4
    rtol = 1e-4 if dtype == jnp.float64 else 1e-3

    def iter_func(carry):
        _, yt, iiter = carry
        yt_next = picard_step(func, y0, yt, xinp, params)
        # same clipping / nan handling as deer.deer_iteration_helper (clip_ytnext=True),
        # applied before computing err so a nan cannot terminate the while_loop early
        yt_next = jnp.clip(yt_next, a_min=-1e8, a_max=1e8)
        yt_next = jnp.where(jnp.isnan(yt_next), 0.0, yt_next)
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
