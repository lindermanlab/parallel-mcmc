""" deer.py
Code adapted from the original DEER codebase by Lim et al. (2024): https://github.com/machine-discovery/deer 
Based on commit: 17b0b625d3413cb3251418980fb78916e5dacfaa (1/18/24) 
Copyright (c) 2023, Machine Discovery Ltd 
Licensed under the BSD 3-Clause License (see LICENSE file for details).

Modifications for benchmarking and quasi-DEER by Xavier Gonzalez (2024). """

from typing import Callable, Any, Tuple, List, Optional

import jax
import jax.numpy as jnp
import jax.random as jr

from functools import partial


def seq1d(
    func: Callable[[jnp.ndarray, Any, Any], jnp.ndarray],
    y0: jnp.ndarray,
    xinp: Any,
    params: Any,
    window: int,
    yinit_guess: Optional[jnp.ndarray] = None,
    max_iter: int = 10000,
    full_trace: bool = False,  # XG addition
    damp_factor: float=1.0, # Damping 
    preconditioner: Any=None, # Diagonal preconditioner,
    clip_val: float=1e8,
):
    # set the default initial guess
    xinp_flat = jax.tree_util.tree_flatten(xinp)[0][0]
    if yinit_guess is None:
        yinit_guess = jnp.zeros(
            (xinp_flat.shape[0], y0.shape[-1]), dtype=xinp_flat.dtype
        )  # (nsamples, ny)

    def func2(y: jnp.ndarray, x: Any, params: Any) -> jnp.ndarray:
        # ylist: (ny,)
        return func(y, x, params)

    def shifter_func(y: jnp.ndarray, y0: jnp.ndarray) -> jnp.ndarray:
        y = jnp.concatenate((y0[None, :], y[:-1, :]), axis=0)  # (nsamples, ny)
        return y

    yt, samp_iters = diagonal_deer_iteration_helper(
        inv_lin=diagonal_seq1d_inv_lin,
        func=func2,
        shifter_func=shifter_func,
        dyn_func=func2,
        p_num=1,
        params=params,
        xinput=xinp,
        y0=y0,
        yinit_guess=yinit_guess,
        window=window,
        max_iter=max_iter,
        clip_ytnext=True,
        full_trace=full_trace,
        damp_factor=damp_factor,
        preconditioner=preconditioner, 
        clip_val=clip_val,
    )

    if full_trace:
        return (jnp.vstack((yinit_guess[None, ...], yt)), samp_iters)
    else:
        return (yt, samp_iters)


def diagonal_binary_operator(
    element_i: Tuple[jnp.ndarray, jnp.ndarray],
    element_j: Tuple[jnp.ndarray, jnp.ndarray],
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    XG addition to make the matrix multiplication diagonal
    """
    # associative operator for the scan
    gti, hti = element_i
    gtj, htj = element_j
    a = gtj * gti
    b = gtj * hti + htj
    return a, b


def diagonal_matmul_recursive(
    mats: jnp.ndarray, vecs: jnp.ndarray, y0: jnp.ndarray
) -> jnp.ndarray:
    """
    XG addition to make the matrix multiplication diagonal

    Perform the matrix multiplication recursively, y[i + 1] = mats[i] @ y[i] + vec[i].

    Arguments
    ---------
    mats: jnp.ndarray
        The matrices to be multiplied, shape (nsamples - 1, ny) # changed to make the matrices diagonal
    vecs: jnp.ndarray
        The vector to be multiplied, shape (nsamples - 1, ny)
    y0: jnp.ndarray
        The initial condition, shape (ny,)

    Returns
    -------
    result: jnp.ndarray
        The result of the matrix multiplication, shape (nsamples, ny)
    """
    # shift the elements by one index
    eye = jnp.ones(mats.shape[-1], dtype=mats.dtype)[None]  # (1, ny)
    first_elem = jnp.concatenate((eye, mats), axis=0)  # (nsamples, ny)
    second_elem = jnp.concatenate((y0[None], vecs), axis=0)  # (nsamples, ny)

    # perform the scan
    elems = (first_elem, second_elem)
    _, yt = jax.lax.associative_scan(diagonal_binary_operator, elems)
    return yt  # (nsamples, ny)


def diagonal_seq1d_inv_lin(
    gmat: jnp.ndarray, rhs: jnp.ndarray, y0: jnp.ndarray
) -> jnp.ndarray:
    """
    Inverse of the linear operator for solving the discrete sequential equation.
    y[i + 1] + G[i] y[i] = rhs[i], y[0] = y0.

    Arguments
    ---------
    gmat: jnp.ndarray
        The list of 1 G-matrix of shape (nsamples, ny). NOTE: these G-matrices must be diagonal (XG addition)
    rhs: jnp.ndarray
        The right hand side of the equation of shape (nsamples, ny).
    inv_lin_params: Tuple[jnp.ndarray]
        The parameters of the linear operator.
        The first element is the initial condition (ny,).

    Returns
    -------
    y: jnp.ndarray
        The solution of the linear equation of shape (nsamples, ny).
    """
    # compute the recursive matrix multiplication and drop the first element
    yt = diagonal_matmul_recursive(-gmat, rhs, y0)[1:]  # (nsamples, ny)
    return yt

def set_after_index_2d(arr, t, value):
    indices = jnp.arange(arr.shape[0])[:, None]
    # Only update if t is valid AND index is strictly greater than t
    valid_update = (t > 0) & (t < arr.shape[0] - 1)  # t must allow at least one element after
    mask = (indices > t) & valid_update
    return jnp.where(mask, value, arr)

def diagonal_deer_iteration_helper(
    inv_lin: Callable[[List[jnp.ndarray], jnp.ndarray, Any], jnp.ndarray],
    func: Callable[[List[jnp.ndarray], Any, Any], jnp.ndarray],
    shifter_func: Callable[[jnp.ndarray, Any], List[jnp.ndarray]],
    dyn_func: Callable[[jnp.ndarray, Any, Any], jnp.ndarray],
    p_num: int,
    params: Any,  # gradable
    xinput: Any,  # gradable
    y0: jnp.ndarray,
    yinit_guess: jnp.ndarray,
    window: int,
    max_iter: int = 100,
    clip_ytnext: bool = False,
    full_trace: bool = False,  # XG addition
    damp_factor: float=1.0, # Damping 
    preconditioner: Any=None, # Diagonal preconditioner
    clip_val: float=1e8,
) -> Tuple[jnp.ndarray, Optional[List[jnp.ndarray]], Callable]:

    precond = preconditioner if preconditioner is not None else jnp.ones((yinit_guess.shape[-1]))
    keys = jr.split(params['key'], (window))

    T = yinit_guess.shape[0] #TODO: check if this is right

    def deer_jvp(z, driver, params, v):
        return jax.jvp(lambda z : dyn_func(z, driver, params), (z,), (v, ))

    dtype = yinit_guess.dtype
    # set the tolerance to be 1e-4 if dtype is float32, else 1e-7 for float64
    tol = 1e-7 if dtype == jnp.float64 else 1e-4
    rtol = 1e-4 if dtype == jnp.float64 else 1e-3

    def iter_func(
        iter_inp: Tuple[jnp.ndarray, jnp.ndarray, List[jnp.ndarray], jnp.ndarray]
    ) -> Tuple[jnp.ndarray, jnp.ndarray, List[jnp.ndarray], jnp.ndarray]:
        t, yt, y_init, iiter = iter_inp # orig

        t = jnp.minimum(t, T-window)
        yt_win = jax.lax.dynamic_slice(yt, (t, 0), (window, yt.shape[-1])) #yt[t:(t+window)]
        xinp_win = jax.lax.dynamic_slice(xinput, (t, 0), (window, xinput.shape[-1]))

        ytparams = shifter_func(yt_win, y_init)

        rhs, gts = jax.vmap(quasi_diag_estimator, in_axes=(0, 0, None, None, 0))(
            ytparams, xinp_win, params, deer_jvp, keys)
        gts = -jnp.clip(damp_factor / precond[None,:] * gts, -clip_val, clip_val)
        rhs += gts * ytparams 
        yt_next_win = inv_lin(gts, rhs, y_init)  # (nsamples, ny)

        if clip_ytnext:
            clip = 1e8
            yt_next_win = jnp.clip(yt_next_win, a_min=-clip, a_max=clip)
            yt_next_win = jnp.where(jnp.isnan(yt_next_win), 0.0, yt_next_win)

        # relative tolerance
        err = jnp.max( jnp.abs(yt_next_win - yt_win) - rtol * jnp.abs(yt_win), axis=-1)
        mask = err > tol                    # shape: (window,)
        first_violation = jnp.argmax(mask)  # 0 if none; that's why we fix below
        stride = jnp.where(mask.any(), first_violation, window)

        # update yt 
        yt = jax.lax.dynamic_update_slice(yt, yt_next_win, (t,0))

        # stride - compute minimum index in the current window that has not converged
        t = t + stride

        new_t = jnp.minimum(t, T - window)
        y_init_index = new_t - 1
        y_init = jnp.where(new_t > 0, yt[y_init_index], y_init)

        #TODO: warm start rest of sequence?

        return t, yt, y_init, iiter + 1

    def scan_func(iter_inp, args):
        t, yt, y_init, iiter = iter_inp # orig

        t = jnp.minimum(t, T-window)
        yt_win = jax.lax.dynamic_slice(yt, (t, 0), (window, yt.shape[-1])) #yt[t:(t+window)]
        xinp_win = jax.lax.dynamic_slice(xinput, (t, 0), (window, xinput.shape[-1]))

        ytparams = shifter_func(yt_win, y_init)

        rhs, gts = jax.vmap(quasi_diag_estimator, in_axes=(0, 0, None, None, 0))(
            ytparams, xinp_win, params, deer_jvp, keys)
        gts = -jnp.clip(damp_factor / precond[None,:] * gts, -clip_val, clip_val)
        rhs += gts * ytparams 
        yt_next_win = inv_lin(gts, rhs, y_init)  # (nsamples, ny)

        if clip_ytnext:
            clip = 1e8
            yt_next_win = jnp.clip(yt_next_win, a_min=-clip, a_max=clip)
            yt_next_win = jnp.where(jnp.isnan(yt_next_win), 0.0, yt_next_win)

        # relative tolerance
        err = jnp.max( jnp.abs(yt_next_win - yt_win) - rtol * jnp.abs(yt_win), axis=-1)
        mask = err > tol                    # shape: (window,)
        first_violation = jnp.argmax(mask)  # 0 if none; that's why we fix below
        stride = jnp.where(mask.any(), first_violation, window)

        # update yt 
        yt = jax.lax.dynamic_update_slice(yt, yt_next_win, (t,0))

        # stride - compute minimum index in the current window that has not converged
        t = t + stride

        new_t = jnp.minimum(t, T - window)
        y_init_index = new_t - 1
        y_init = jnp.where(new_t > 0, yt[y_init_index], y_init)

        #TODO: warm start rest

        new_carry = t, yt, y_init, iiter + 1
        return new_carry, yt

    def cond_func(
        iter_inp: Tuple[jnp.ndarray, jnp.ndarray, List[jnp.ndarray], jnp.ndarray]
    ) -> bool:
        t, _, _, iiter = iter_inp
        return jnp.logical_and(t < T, iiter < max_iter)

    y_init = y0
    t = 0  # initial error should be very high
    iiter = jnp.array(0, dtype=jnp.int32)
    # decide whether to record full trace or not
    if full_trace:
        _, yt = jax.lax.scan(
            scan_func, (t, yinit_guess, y_init, iiter), None, length=max_iter
        )
        samp_iters = max_iter
    else:
        _, yt, _, samp_iters = jax.lax.while_loop(
            cond_func, iter_func, (t, yinit_guess, y_init, iiter)
        )
    return yt, samp_iters

def quasi_diag_estimator(state, inputs, params, deer_jvp, key, num_samples=1):
    z_rad = jr.rademacher(key, (num_samples, state.shape[0])).astype(float)
    vmap_jvp = jax.vmap(deer_jvp, in_axes=(None, None, None, 0))
    f_vals, jac_vals = vmap_jvp(state, inputs, params, z_rad)
    jac_diag = jnp.mean(z_rad * jac_vals, axis=0)
    return f_vals[0], jac_diag