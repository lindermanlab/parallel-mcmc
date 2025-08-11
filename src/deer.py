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
    memory_efficient: bool = False,
    quasi: bool = False,
    qmem_efficient: bool = True,  # XG addition
    full_trace: bool = False,  # XG addition
    damp_factor: float=1.0, # Damping 
    preconditioner: Any=None, # Diagonal preconditioner,
    clip_val: float=1e8
):
    """
    Solve the discrete sequential equation, y[i + 1] = func(y[i], x[i], params) with the DEER framework.

    Arguments
    ---------
    func: Callable[[jnp.ndarray, Any, Any], jnp.ndarray]
        Function to evaluate the next output signal y[i + 1] from the current output signal y[i].
        The arguments are: output signal y (ny,), input signal x (*nx,) in a pytree, and parameters.
        The return value is the next output signal y[i + 1] (ny,).
    y0: jnp.ndarray
        Initial condition on y (ny,).
    xinp: Any
        The external input signal in a pytree of shape (nsamples, *nx)
    params: Any
        The parameters of the function ``func``.
    yinit_guess: jnp.ndarray or None
        The initial guess of the output signal (nsamples, ny).
        If None, it will be initialized as 0s.
    max_iter: int
        The maximum number of iterations to perform.
    memory_efficient: bool
        If True, then use the memory efficient algorithm for the DEER iteration.
    quasi: bool
        If True, then make all the Jacobians diagonal. (XG addition)
    qmem_efficient: bool
        If True, use the memory efficient of quasi; if false, call jnp.diag on jac func
        Note: need to use false for eigenworms
    full_trace: bool
        If True, return the full trace of all the Newton iterates for a fixed specification of max_iter (uses a scan)
        if False, return only the final iterate (uses a jax.lax.while_loop)

    Returns
    -------
    y: jnp.ndarray
        The output signal as the solution of the discrete difference equation (nsamples, ny),
        excluding the initial states.
    """
    # set the default initial guess
    xinp_flat = jax.tree_util.tree_flatten(xinp)[0][0]
    if yinit_guess is None:
        yinit_guess = jnp.zeros(
            (xinp_flat.shape[0], y0.shape[-1]), dtype=xinp_flat.dtype
        )  # (nsamples, ny)

    def func2(ylist: List[jnp.ndarray], x: Any, params: Any) -> jnp.ndarray:
        # ylist: (ny,)
        return func(ylist[0], x, params)

    def shifter_func(y: jnp.ndarray, shifter_params: Any) -> List[jnp.ndarray]:
        # y: (nsamples, ny)
        # shifter_params = (y0,)
        (y0,) = shifter_params
        y = jnp.concatenate((y0[None, :], y[:-1, :]), axis=0)  # (nsamples, ny)
        return [y]

    # perform the deer iteration
    if quasi:
        yt, samp_iters = deer_iteration(
            inv_lin=diagonal_seq1d_inv_lin,
            p_num=1,
            func=func2,
            dyn_func=func,
            shifter_func=shifter_func,
            params=params,
            xinput=xinp,
            inv_lin_params=(y0,),
            shifter_func_params=(y0,),
            yinit_guess=yinit_guess,
            max_iter=max_iter,
            memory_efficient=memory_efficient,
            clip_ytnext=True,
            quasi=quasi,
            full_trace=full_trace,
            qmem_efficient=qmem_efficient,
            damp_factor=damp_factor,
            preconditioner=preconditioner, 
            clip_val=clip_val,
        )
    else:
        yt, samp_iters = deer_iteration(
            inv_lin=seq1d_inv_lin,
            p_num=1,
            func=func2,
            dyn_func=func,
            shifter_func=shifter_func,
            params=params,
            xinput=xinp,
            inv_lin_params=(y0,),
            shifter_func_params=(y0,),
            yinit_guess=yinit_guess,
            max_iter=max_iter,
            memory_efficient=memory_efficient,
            clip_ytnext=True,
            quasi=quasi,
            full_trace=full_trace,
            qmem_efficient=qmem_efficient,
            damp_factor=damp_factor,
            preconditioner=preconditioner, 
            clip_val=clip_val,
        )
    if full_trace:
        return (jnp.vstack((yinit_guess[None, ...], yt)), samp_iters)
    else:
        return (yt, samp_iters)


#@partial(jax.custom_vjp, nondiff_argnums=(0, 1, 2, 3, 9, 10, 11, 12, 13, 14))

def deer_iteration(
    inv_lin: Callable[[List[jnp.ndarray], jnp.ndarray, Any], jnp.ndarray],
    func: Callable[[List[jnp.ndarray], Any, Any], jnp.ndarray],
    shifter_func: Callable[[jnp.ndarray], List[jnp.ndarray]],
    dyn_func: Callable[[jnp.ndarray, Any, Any], jnp.ndarray],
    p_num: int,
    params: Any,  # gradable
    xinput: Any,  # gradable
    inv_lin_params: Any,  # gradable
    shifter_func_params: Any,  # gradable
    yinit_guess: jnp.ndarray,  # gradable as 0
    max_iter: int = 100,
    memory_efficient: bool = False,
    clip_ytnext: bool = False,
    quasi: bool = False,  # XG addition
    qmem_efficient: bool = True,  # XG addition
    full_trace: bool = False,  # XG addition
    damp_factor: float=1.0, # Damping 
    preconditioner: Any=None, # Diagonal preconditioner
    clip_val: float=1e8
) -> jnp.ndarray:
    """
    Perform the iteration from the DEER framework.

    Arguments
    ---------
    inv_lin: Callable[[List[jnp.ndarray], jnp.ndarray, Any], jnp.ndarray]
        Inverse of the linear operator.
        Takes the list of G-matrix (nsamples, ny, ny) (p-elements),
        the right hand side of the equation (nsamples, ny), and the inv_lin parameters in a tree.
        Returns the results of the inverse linear operator (nsamples, ny).
    func: Callable[[List[jnp.ndarray], Any, Any], jnp.ndarray]
        The non-linear function.
        Function that takes the list of y [output: (ny,)] (p elements), x [input: (*nx)] (in a pytree),
        and parameters (any structure of pytree).
        Returns the output of the function.
    shifter_func: Callable[[jnp.ndarray, Any], List[jnp.ndarray]]
        The function that shifts the input signal.
        It takes the signal of shape (nsamples, ny) and produces a list of shifted signals of shape (nsamples, ny).
    p_num: int
        Number of how many dependency on values of ``y`` at different places the function ``func`` has
    params: Any
        The parameters of the function ``func``.
    xinput: Any
        The external input signal of in a pytree with shape (nsamples, *nx).
    inv_lin_params: tree structure of jnp.ndarray
        The parameters of the function ``inv_lin``.
    shifter_func_params: tree structure of jnp.ndarray
        The parameters of the function ``shifter_func``.
    yinit_guess: jnp.ndarray or None
        The initial guess of the output signal (nsamples, ny).
        If None, it will be initialized as 0s.
    max_iter: int
        The maximum number of iterations to perform.
    memory_efficient: bool
        If True, then do not save the Jacobian matrix for the backward pass.
        This can save memory, but the backward pass will be slower due to recomputation of
        the Jacobian matrix.
    quasi: bool (XG addition)
        If True, then make all the Jacobians diagonal
    full_trace: bool (XG addition)
        If True, then return all the intermediate y values (up to length max_iter)
        If False, then return only the final y value (which may be decided by early stopping up to the tolerance)

    Returns
    -------
    y: jnp.ndarray
        The output signal as the solution of the non-linear differential equations (nsamples, ny).
    """
    if quasi:
        yt, _, _, _, samp_iters = diagonal_deer_iteration_helper(
            inv_lin=inv_lin,
            func=func,
            shifter_func=shifter_func,
            dyn_func=dyn_func,
            p_num=p_num,
            params=params,
            xinput=xinput,
            inv_lin_params=inv_lin_params,
            shifter_func_params=shifter_func_params,
            yinit_guess=yinit_guess,
            max_iter=max_iter,
            memory_efficient=memory_efficient,
            clip_ytnext=clip_ytnext,
            full_trace=full_trace,
            qmem_efficient=qmem_efficient,
            damp_factor=damp_factor,
            preconditioner=preconditioner, 
            clip_val=clip_val,
        )
        return (yt, samp_iters)
    else:
        yt, _, _, _, samp_iters = deer_iteration_helper(
            inv_lin=inv_lin,
            func=func,
            shifter_func=shifter_func,
            p_num=p_num,
            params=params,
            xinput=xinput,
            inv_lin_params=inv_lin_params,
            shifter_func_params=shifter_func_params,
            yinit_guess=yinit_guess,
            max_iter=max_iter,
            memory_efficient=memory_efficient,
            clip_ytnext=clip_ytnext,
            full_trace=full_trace,
            damp_factor=damp_factor,
            preconditioner=preconditioner, 
            clip_val=clip_val,
        )
        return (yt, samp_iters)


def deer_iteration_helper(
    inv_lin: Callable[[List[jnp.ndarray], jnp.ndarray, Any], jnp.ndarray],
    func: Callable[[List[jnp.ndarray], Any, Any], jnp.ndarray],
    shifter_func: Callable[[jnp.ndarray, Any], List[jnp.ndarray]],
    p_num: int,
    params: Any,  # gradable
    xinput: Any,  # gradable
    inv_lin_params: Any,  # gradable
    shifter_func_params: Any,  # gradable
    yinit_guess: jnp.ndarray,
    max_iter: int = 100,
    memory_efficient: bool = False,
    clip_ytnext: bool = False,
    full_trace: bool = False, # XG addition
    damp_factor: float=1.0, # Damping 
    preconditioner: Any=None, # Diagonal preconditioner for Quasi
    clip_val: float=1e8,
) -> Tuple[jnp.ndarray, Optional[List[jnp.ndarray]], Callable]:
    """
    Notes:
        - XG addition: full_trace, to return all the intermediate y values (up to length max_iter)
    """
    # obtain the functions to compute the jacobians and the function
    jacfunc = jax.vmap(jax.jacfwd(func, argnums=0), in_axes=(0, 0, None))
    func2 = jax.vmap(func, in_axes=(0, 0, None))

    dtype = yinit_guess.dtype
    # set the tolerance to be 1e-4 if dtype is float32, else 1e-7 for float64
    tol = 1e-7 if dtype == jnp.float64 else 1e-4
    rtol = 1e-4 if dtype == jnp.float64 else 1e-3

    # use the iter function if doing early stopping
    def iter_func(
        iter_inp: Tuple[jnp.ndarray, jnp.ndarray, List[jnp.ndarray], jnp.ndarray]
    ) -> Tuple[jnp.ndarray, jnp.ndarray, List[jnp.ndarray], jnp.ndarray]:
        err, yt, gt_, iiter = iter_inp
        # gt_ is not used, but it is needed to return at the end of scan iteration
        # yt: (nsamples, ny)
        ytparams = shifter_func(yt, shifter_func_params)
        gts = [
            -jnp.clip(damp_factor*gt, -clip_val, clip_val) for gt in jacfunc(ytparams, xinput, params)
        ]  # [p_num] + (nsamples, ny, ny), meaning its a list of length p num
        # rhs: (nsamples, ny)
        rhs = func2(ytparams, xinput, params)  # (carry, input, params) 
        rhs += sum(
            [jnp.einsum("...ij,...j->...i", gt, ytp) for gt, ytp in zip(gts, ytparams)]
        )
        yt_next = inv_lin(gts, rhs, inv_lin_params)  # (nsamples, ny)

        if clip_ytnext:
            clip = 1e8
            yt_next = jnp.clip(yt_next, a_min=-clip, a_max=clip)
            yt_next = jnp.where(jnp.isnan(yt_next), 0.0, yt_next)

        err = jnp.max( jnp.abs(yt_next - yt) - rtol * jnp.abs(yt) )

        return err, yt_next, gts, iiter + 1

    # key = jr.PRNGKey(13)
    # def scale_matrices(matrix, key, threshold=1.0):
    #     safety_factor = 100
    #     rv = jr.normal(key, matrix.shape[0])
    #     norm_rv = rv / jnp.linalg.norm(rv)
    #     amp = matrix @ norm_rv 
    #     # amp = jnp.linalg.norm(amp)
    #     norm_rv = amp / jnp.linalg.norm(amp)
    #     amp = matrix @ norm_rv
    #     amp = jnp.linalg.norm(amp)
    #     needs_scaling = amp > threshold
    #     scale_factor = jnp.where(needs_scaling, 
    #                         amp * safety_factor,
    #                         1.0)
    #     return matrix / scale_factor
    # # def scale_matrices(matrix, key):
    # #     scale_factor = jnp.linalg.norm(matrix)
    # #     return matrix / scale_factor
    # batch_scale_matrices = jax.vmap(scale_matrices)

    # use the scan function to get the full trace
    def scan_func(iter_inp, args):
        err, yt, gt_, iiter = iter_inp
        # gt_ is not used, but it is needed to return at the end of scan iteration
        # yt: (nsamples, ny)
        ytparams = shifter_func(yt, shifter_func_params)
        gts = [
            -jnp.clip(damp_factor*gt, -clip_val, clip_val) for gt in jacfunc(ytparams, xinput, params)
        ]  # [p_num] + (nsamples, ny, ny), meaning its a list of length p num
        # rhs: (nsamples, ny)
        rhs = func2(ytparams, xinput, params)  # (carry, input, params)
        rhs += sum(
            [jnp.einsum("...ij,...j->...i", gt, ytp) for gt, ytp in zip(gts, ytparams)]
        )
        yt_next = inv_lin(gts, rhs, inv_lin_params)  # (nsamples, ny)

        # err = jnp.max(jnp.abs(yt_next - yt))  # checking convergence
        err = jnp.max( jnp.abs(yt_next - yt) - rtol * jnp.abs(yt) )

        yt_next = jnp.nan_to_num(yt_next)  # XG addition, avoid nans
        new_carry = err, yt_next, gts, iiter + 1
        return new_carry, yt_next

    def cond_func(
        iter_inp: Tuple[jnp.ndarray, jnp.ndarray, List[jnp.ndarray], jnp.ndarray]
    ) -> bool:
        err, _, _, iiter = iter_inp
        return jnp.logical_and(err > tol, iiter < max_iter)

    err = jnp.array(1e10, dtype=dtype)  # initial error should be very high
    gt = jnp.zeros(
        (yinit_guess.shape[0], yinit_guess.shape[-1], yinit_guess.shape[-1]),
        dtype=dtype,
    )
    gts = [gt] * p_num

    iiter = jnp.array(0, dtype=jnp.int32)
    # decide whether to record full trace or not
    if full_trace:
        _, yt = jax.lax.scan(
            scan_func, (err, yinit_guess, gts, iiter), None, length=max_iter
        )
        samp_iters = max_iter
    else:
        _, yt, gts, samp_iters = jax.lax.while_loop(
            cond_func, iter_func, (err, yinit_guess, gts, iiter)
        )
    if memory_efficient:
        gts = None
    rhs = jnp.zeros_like(gts[0][..., 0])  # (nsamples, ny)
    return yt, gts, rhs, func, samp_iters


def binary_operator(
    element_i: Tuple[jnp.ndarray, jnp.ndarray],
    element_j: Tuple[jnp.ndarray, jnp.ndarray],
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    # associative operator for the scan
    gti, hti = element_i
    gtj, htj = element_j
    a = gtj @ gti
    b = jnp.einsum("...ij,...j->...i", gtj, hti) + htj
    return a, b


def matmul_recursive(
    mats: jnp.ndarray, vecs: jnp.ndarray, y0: jnp.ndarray
) -> jnp.ndarray:
    """
    Perform the matrix multiplication recursively, y[i + 1] = mats[i] @ y[i] + vec[i].

    Arguments
    ---------
    mats: jnp.ndarray
        The matrices to be multiplied, shape (nsamples - 1, ny, ny)
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
    eye = jnp.eye(mats.shape[-1], dtype=mats.dtype)[None]  # (1, ny, ny)
    first_elem = jnp.concatenate((eye, mats), axis=0)  # (nsamples, ny, ny)
    second_elem = jnp.concatenate((y0[None], vecs), axis=0)  # (nsamples, ny)

    # perform the scan
    elems = (first_elem, second_elem)
    _, yt = jax.lax.associative_scan(binary_operator, elems)
    return yt  # (nsamples, ny)


def seq1d_inv_lin(
    gmat: List[jnp.ndarray], rhs: jnp.ndarray, inv_lin_params: Tuple[jnp.ndarray]
) -> jnp.ndarray:
    """
    Inverse of the linear operator for solving the discrete sequential equation.
    y[i + 1] + G[i] y[i] = rhs[i], y[0] = y0.

    Arguments
    ---------
    gmat: jnp.ndarray
        The list of 1 G-matrix of shape (nsamples, ny, ny).
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
    gmat = gmat[0]

    # compute the recursive matrix multiplication and drop the first element
    yt = matmul_recursive(-gmat, rhs, y0)[1:]  # (nsamples, ny)
    return yt

# ---------------------------------------------------------------------------#
#                                Quasi
#                                  XG addition to do quasi-Newton (i.e. just use diagonalized Jacobians)
# ---------------------------------------------------------------------------#


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
    gmat: List[jnp.ndarray], rhs: jnp.ndarray, inv_lin_params: Tuple[jnp.ndarray]
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
    gmat = gmat[0]

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
    memory_efficient: bool = False,
    clip_ytnext: bool = False,
    full_trace: bool = False,  # XG addition
    qmem_efficient: bool = True,  # XG addition
    damp_factor: float=1.0, # Damping 
    preconditioner: Any=None, # Diagonal preconditioner
    clip_val: float=1e8,
) -> Tuple[jnp.ndarray, Optional[List[jnp.ndarray]], Callable]:
    # obtain the functions to compute the jacobians and the function
    jacfunc = jax.vmap(
        jax.jacfwd(func, argnums=0), in_axes=(0, 0, None)
    )  # bunch of dense matrices

    precond = preconditioner if preconditioner is not None else jnp.ones((yinit_guess.shape[-1]))

    if qmem_efficient:
        def deer_jvp(z, driver, params, v):
            return jax.jvp(lambda z : dyn_func(z, driver, params), (z,), (v, ))[1]
        keys = jr.split(params['key'], (xinput.shape[0]))

    func2 = jax.vmap(func, in_axes=(0, 0, None))

    dtype = yinit_guess.dtype
    # set the tolerance to be 1e-4 if dtype is float32, else 1e-7 for float64
    tol = 1e-7 if dtype == jnp.float64 else 5e-4
    rtol = 1e-4 if dtype == jnp.float64 else 1e-3

    def iter_func(
        iter_inp: Tuple[jnp.ndarray, jnp.ndarray, List[jnp.ndarray], jnp.ndarray]
    ) -> Tuple[jnp.ndarray, jnp.ndarray, List[jnp.ndarray], jnp.ndarray]:
        err, yt, gt_, iiter = iter_inp
        # gt_ is not used, but it is needed to return at the end of scan iteration
        # yt: (nsamples, ny)
        ytparams = shifter_func(yt, shifter_func_params)
        # XG change to be more memory efficient
        if qmem_efficient:
            gts = [
               -jnp.clip(damp_factor / precond[None,:] *jax.vmap(quasi_diag_estimator, in_axes=(0, 0, None, None, 0))(
                   ytparams[0], xinput, params, deer_jvp, keys), -clip_val, clip_val)
            ]
        else:
            gts = [
                -jnp.clip(damp_factor / precond[None,:] * jax.vmap(jnp.diag)(gt), -clip_val, clip_val)
                for gt in jacfunc(
                    ytparams, xinput, params
                )  # adjusted to deal with scalars
            ]  # [p_num] + (nsamples, ny)
        # rhs: (nsamples, ny)
        rhs = func2(ytparams, xinput, params)  # (carry, input, params) 
        rhs += sum(
            [
                gt * ytp for gt, ytp in zip(gts, ytparams)
            ]  # adjusted to deal with scalars
        )
        yt_next = inv_lin(gts, rhs, inv_lin_params)  # (nsamples, ny)

        if clip_ytnext:
            clip = 1e8
            yt_next = jnp.clip(yt_next, a_min=-clip, a_max=clip)
            yt_next = jnp.where(jnp.isnan(yt_next), 0.0, yt_next)

        # err = jnp.max(jnp.abs(yt_next - yt))  # checking convergence
        # relative tolerance
        err = jnp.max( jnp.abs(yt_next - yt) - rtol * jnp.abs(yt) )

        return err, yt_next, gts, iiter + 1

    def scan_func(iter_inp, args):
        err, yt, gt_, iiter = iter_inp
        # gt_ is not used, but it is needed to return at the end of scan iteration
        # yt: (nsamples, ny)
        ytparams = shifter_func(yt, shifter_func_params)
        if qmem_efficient:
            gts = [
               -jnp.clip(damp_factor / precond[None,:] *jax.vmap(quasi_diag_estimator, in_axes=(0, 0, None, None, 0))(
                   ytparams[0], xinput, params, deer_jvp, keys), -clip_val, clip_val)
            ]
        else:
            gts = [
                -jnp.clip(damp_factor / precond[None,:] * jax.vmap(jnp.diag)(gt), -clip_val, clip_val)
                for gt in jacfunc(
                    ytparams, xinput, params
                )  # adjusted to deal with scalars
            ]  # [p_num] + (nsamples, ny)
        # rhs: (nsamples, ny)
        rhs = func2(ytparams, xinput, params)  # (carry, input, params) 
        rhs += sum(
            [
                gt * ytp for gt, ytp in zip(gts, ytparams)
            ]  # adjusted to deal with scalars
        )
        yt_next = inv_lin(gts, rhs, inv_lin_params)  # (nsamples, ny)

        # err = jnp.max(jnp.abs(yt_next - yt))  # checking convergence
        err = jnp.max( jnp.abs(yt_next - yt) - rtol * jnp.abs(yt) )

        yt_next = jnp.nan_to_num(yt_next)  # XG addition, avoid nans
        new_carry = err, yt_next, gts, iiter + 1
        return new_carry, yt_next

    def cond_func(
        iter_inp: Tuple[jnp.ndarray, jnp.ndarray, List[jnp.ndarray], jnp.ndarray]
    ) -> bool:
        err, _, _, iiter = iter_inp
        return jnp.logical_and(err > tol, iiter < max_iter)

    err = jnp.array(1e10, dtype=dtype)  # initial error should be very high
    gt = jnp.zeros(
        (yinit_guess.shape[0], yinit_guess.shape[-1]),
        dtype=dtype,
    )
    gts = [gt] * p_num
    iiter = jnp.array(0, dtype=jnp.int32)
    # decide whether to record full trace or not
    if full_trace:
        _, yt = jax.lax.scan(
            scan_func, (err, yinit_guess, gts, iiter), None, length=max_iter
        )
        samp_iters = max_iter
    else:
        _, yt, gts, samp_iters = jax.lax.while_loop(
            cond_func, iter_func, (err, yinit_guess, gts, iiter)
        )
    if memory_efficient:
        gts = None
    rhs = jnp.zeros_like(gts[0])  # (nsamples, ny)
    return yt, gts, rhs, func, samp_iters

def quasi_diag_estimator(state, inputs, params, deer_jvp, key, num_samples=1):
    z_rad = jr.rademacher(key, (num_samples, state.shape[0])).astype(float)
    vmap_jvp = jax.vmap(deer_jvp, in_axes=(None, None, None, 0))
    jac_diag = jnp.mean(z_rad * vmap_jvp(state, inputs, params, z_rad), axis=0)
    return jac_diag
