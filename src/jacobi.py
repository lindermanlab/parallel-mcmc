"""
Jacobi iteration for solving y[i + 1] = func(y[i], x[i], params) in parallel.

Ported from micro_deer (https://github.com/lindermanlab/micro_deer). Jacobi is the
special case of DEER where every Jacobian is replaced by zero: each iteration just
evaluates the transition at every index of the previous iterate in parallel,
y[t] <- func(y[t-1], x[t-1], params), with no linear solve at all. After k sweeps the
first k states are exact, so it needs at most chain_length iterations.
"""
from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp

from src.picard import iterate


def jacobi_step(func, y0, yt, xinp, params):
    """One Jacobi step: y_t = f(y_{t-1}) for every t, using the previous iterate."""
    yt_prev = jnp.concatenate((y0[None, :], yt[:-1]), axis=0)  # (nsamples, ny)
    yt_next = jax.vmap(func, in_axes=(0, 0, None))(yt_prev, xinp, params)
    # same clipping / nan handling as picard_step
    yt_next = jnp.clip(yt_next, a_min=-1e8, a_max=1e8)
    return jnp.where(jnp.isnan(yt_next), 0.0, yt_next)


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
    Solve y[i + 1] = func(y[i], x[i], params) with Jacobi iteration.

    Same interface and return convention as picard.seq1d / deer.seq1d.
    """
    xinp_flat = jax.tree_util.tree_flatten(xinp)[0][0]
    if yinit_guess is None:
        yinit_guess = jnp.zeros(
            (xinp_flat.shape[0], y0.shape[-1]), dtype=xinp_flat.dtype
        )
    step = lambda yt: jacobi_step(func, y0, yt, xinp, params)
    return iterate(step, yinit_guess, max_iter, full_trace)
