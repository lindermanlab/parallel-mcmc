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
    yinit_guess: Optional[jnp.ndarray] = None,
    max_iter: int = 10000,
    full_trace: bool = False,  # XG addition
    damp_factor: float=1.0, # Damping 
    preconditioner: Any=None, # Diagonal preconditioner,
    clip_val: float=1e8
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

    def shifter_func(y: jnp.ndarray, shifter_params: Any) -> jnp.ndarray:
        # y: (nsamples, ny)
        # shifter_params = (y0,)
        (y0,) = shifter_params
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
        inv_lin_params=(y0,),
        shifter_func_params=(y0,),
        yinit_guess=yinit_guess,
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
    gmat: jnp.ndarray, rhs: jnp.ndarray, inv_lin_params: Tuple[jnp.ndarray]
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
    # extract the parameters
    (y0,) = inv_lin_params

    # compute the recursive matrix multiplication and drop the first element
    yt = diagonal_matmul_recursive(-gmat, rhs, y0)[1:]  # (nsamples, ny)
    return yt


def diagonal_deer_iteration_helper(
    inv_lin: Callable[[List[jnp.ndarray], jnp.ndarray, Any], jnp.ndarray],
    func: Callable[[List[jnp.ndarray], Any, Any], jnp.ndarray],
    shifter_func: Callable[[jnp.ndarray, Any], List[jnp.ndarray]],
    dyn_func: Callable[[jnp.ndarray, Any, Any], jnp.ndarray],
    p_num: int,
    params: Any,  # gradable
    xinput: Any,  # gradable
    inv_lin_params: Any,  # gradable
    shifter_func_params: Any,  # gradable
    yinit_guess: jnp.ndarray,
    max_iter: int = 100,
    clip_ytnext: bool = False,
    full_trace: bool = False,  # XG addition
    damp_factor: float=1.0, # Damping 
    preconditioner: Any=None, # Diagonal preconditioner
    clip_val: float=1e8,
) -> Tuple[jnp.ndarray, Optional[List[jnp.ndarray]], Callable]:

    precond = preconditioner if preconditioner is not None else jnp.ones((yinit_guess.shape[-1]))
    keys = jr.split(params['key'], (xinput.shape[0]))

    def deer_jvp(z, driver, params, v):
        return jax.jvp(lambda z : dyn_func(z, driver, params), (z,), (v, ))

    dtype = yinit_guess.dtype
    # set the tolerance to be 1e-4 if dtype is float32, else 1e-7 for float64
    tol = 1e-7 if dtype == jnp.float64 else 1e-4
    rtol = 1e-4 if dtype == jnp.float64 else 1e-3

    def iter_func(
        iter_inp: Tuple[jnp.ndarray, jnp.ndarray, List[jnp.ndarray], jnp.ndarray]
    ) -> Tuple[jnp.ndarray, jnp.ndarray, List[jnp.ndarray], jnp.ndarray]:
        err, yt, iiter = iter_inp # orig
        # yt: (nsamples, ny)
        ytparams = shifter_func(yt, shifter_func_params)

        rhs, gts = jax.vmap(quasi_diag_estimator, in_axes=(0, 0, None, None, 0))(
            ytparams, xinput, params, deer_jvp, keys)
        gts = -jnp.clip(damp_factor / precond[None,:] * gts, -clip_val, clip_val)
        rhs += gts * ytparams 
        yt_next = inv_lin(gts, rhs, inv_lin_params)  # (nsamples, ny)

        if clip_ytnext:
            clip = 1e8
            yt_next = jnp.clip(yt_next, a_min=-clip, a_max=clip)
            yt_next = jnp.where(jnp.isnan(yt_next), 0.0, yt_next)

        # relative tolerance
        err = jnp.max( jnp.abs(yt_next - yt) - rtol * jnp.abs(yt) )

        return err, yt_next, iiter + 1

    def scan_func(iter_inp, args):
        err, yt, iiter = iter_inp
        # yt: (nsamples, ny)
        ytparams = shifter_func(yt, shifter_func_params)

        rhs, gts = jax.vmap(quasi_diag_estimator, in_axes=(0, 0, None, None, 0))(
            ytparams, xinput, params, deer_jvp, keys)
        # gts = jnp.nan_to_num(gts, nan=0.0)
        # rhs = jnp.nan_to_num(rhs, nan=jnp.nanmean(rhs))
        gts = -jnp.clip(damp_factor / precond[None,:] * gts, -clip_val, clip_val)
        # jax.debug.print("gts nan: {x}", x=jnp.sum(jnp.isnan(gts)))
        # jax.debug.print("rhs nan: {x}", x=jnp.sum(jnp.isnan(rhs)))
        rhs += gts * ytparams 
        yt_next = inv_lin(gts, rhs, inv_lin_params)  # (nsamples, ny)
        
        err = jnp.max( jnp.abs(yt_next - yt) - rtol * jnp.abs(yt) )

        yt_next = jnp.nan_to_num(yt_next)  # XG addition, avoid nans
        new_carry = err, yt_next, iiter + 1
        return new_carry, yt_next

    def cond_func(
        iter_inp: Tuple[jnp.ndarray, jnp.ndarray, List[jnp.ndarray], jnp.ndarray]
    ) -> bool:
        err, _, iiter = iter_inp
        return jnp.logical_and(err > tol, iiter < max_iter)

    err = jnp.array(1e10, dtype=dtype)  # initial error should be very high
    iiter = jnp.array(0, dtype=jnp.int32)
    # decide whether to record full trace or not
    if full_trace:
        _, yt = jax.lax.scan(
            scan_func, (err, yinit_guess, iiter), None, length=max_iter
        )
        samp_iters = max_iter
    else:
        _, yt, samp_iters = jax.lax.while_loop(
            cond_func, iter_func, (err, yinit_guess, iiter)
        )
    return yt, samp_iters

def quasi_diag_estimator(state, inputs, params, deer_jvp, key, num_samples=1):
    z_rad = jr.rademacher(key, (num_samples, state.shape[0])).astype(float)
    vmap_jvp = jax.vmap(deer_jvp, in_axes=(None, None, None, 0))
    f_vals, jac_vals = vmap_jvp(state, inputs, params, z_rad)
    jac_diag = jnp.mean(z_rad * jac_vals, axis=0)
    return f_vals[0], jac_diag