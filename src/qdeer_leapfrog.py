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
    logp: Callable[jnp.ndarray, jnp.ndarray],
    yinit_guess: Optional[jnp.ndarray] = None,
    max_iter: int = 10000,
    memory_efficient: bool = True,
    qmem_efficient: bool = False,  # XG addition
    full_trace: bool = False,  # XG addition
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
        The parameters of the function ``func``. This should contain the step size epsilon. 
    logp: Callable
        Function to compute target density given a state in leapfrog. 
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
    yt, samp_iters = deer_iteration(
        inv_lin=block_diagonal_seq1d_inv_lin,
        p_num=1,
        func=func2,
        logp=logp,
        shifter_func=shifter_func,
        params=params,
        xinput=xinp,
        inv_lin_params=(y0,),
        shifter_func_params=(y0,),
        yinit_guess=yinit_guess,
        max_iter=max_iter,
        memory_efficient=memory_efficient,
        clip_ytnext=True,
        full_trace=full_trace,
        qmem_efficient=qmem_efficient,
    )

    if full_trace:
        return (jnp.vstack((yinit_guess[None, ...], yt)), samp_iters)
    else:
        return (yt, samp_iters)


def deer_iteration(
    inv_lin: Callable[[List[jnp.ndarray], jnp.ndarray, Any], jnp.ndarray],
    func: Callable[[List[jnp.ndarray], Any, Any], jnp.ndarray],
    logp: Callable[jnp.ndarray, jnp.ndarray],
    shifter_func: Callable[[jnp.ndarray], List[jnp.ndarray]],
    p_num: int,
    params: Any,  # gradable
    xinput: Any,  # gradable
    inv_lin_params: Any,  # gradable
    shifter_func_params: Any,  # gradable
    yinit_guess: jnp.ndarray,  # gradable as 0
    max_iter: int = 100,
    memory_efficient: bool = True,
    clip_ytnext: bool = False,
    qmem_efficient: bool = False,  # XG addition
    full_trace: bool = False,  # XG addition
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
    yt, _, _, _, samp_iters = block_diagonal_deer_iteration_helper(
        inv_lin=inv_lin,
        func=func,
        logp=logp,
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
        qmem_efficient=qmem_efficient,
    )
    return (yt, samp_iters)


def diagonal_block_binary_operator(
    element_i: Tuple[jnp.ndarray, jnp.ndarray],
    element_j: Tuple[jnp.ndarray, jnp.ndarray],
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Function assumes leading batch dimension per call by associative scan.
    """
    # associative operator for the scan
    gti, hti = element_i
    gtj, htj = element_j
    # unpack gts into 4 diags
    gti_11, gti_12, gti_21, gti_22 = jnp.split(gti, 4, axis=1)
    gtj_11, gtj_12, gtj_21, gtj_22 = jnp.split(gtj, 4, axis=1)
    # multiply and sum diagonals to get new diagonals
    a11 = gtj_11 * gti_11 + gtj_12 * gti_21
    a12 = gtj_11 * gti_12 + gtj_12 * gti_22
    a21 = gtj_21 * gti_11 + gtj_22 * gti_21
    a22 = gtj_21 * gti_12 + gtj_22 * gti_22
    a = jnp.hstack((a11, a12, a21, a22))
    # multiply by vector
    h1, h2 = jnp.split(hti, 2, axis=-1)
    b1 = gtj_11[:, 0, :] * h1 + gtj_12[:, 0, :] * h2 
    b2 = gtj_21[:, 0, :] * h1 + gtj_22[:, 0, :] * h2 
    b = jnp.concatenate((b1, b2), axis=-1) + htj
    return a, b

# def diagonal_block_binary_operator(
#     element_i: Tuple[jnp.ndarray, jnp.ndarray],
#     element_j: Tuple[jnp.ndarray, jnp.ndarray],
# ) -> Tuple[jnp.ndarray, jnp.ndarray]:
#     """
#     Function assumes leading batch dimension per call by associative scan.
#     """
#     # associative operator for the scan
#     gti, hti = element_i
#     gtj, htj = element_j
#     # unpack gts into 4 diags
#     # gti_11, gti_12, gti_21, gti_22 = jnp.split(gti, (1,2,3), axis=1)
#     # gtj_11, gtj_12, gtj_21, gtj_22 = jnp.split(gtj, (1,2,3), axis=1)
#     gti_11, gti_12, gti_21, gti_22 = jnp.split(gti, 4, axis=-2)
#     gtj_11, gtj_12, gtj_21, gtj_22 = jnp.split(gtj, 4, axis=-2)
#     # multiple diagonals
#     a11 = gtj_11 * gti_11 + gtj_12 * gti_21
#     a12 = gtj_11 * gti_12 + gtj_12 * gti_22
#     a21 = gtj_21 * gti_11 + gtj_22 * gti_21
#     a22 = gtj_21 * gti_12 + gtj_22 * gti_22
#     a = jnp.concatenate((a11, a12, a21, a22), axis=-2)
#     # multiply by vector
#     h1, h2 = jnp.split(hti, 2, axis=-1)

#     # v1
#     # b1 = gtj_11[:, 0, :] * h1 + gtj_12[:, 0, :] * h2 
#     # b2 = gtj_21[:, 0, :] * h1 + gtj_22[:, 0, :] * h2 
#     # b1 = gtj_11[:, 0] * h1 + gtj_12[:, 0] * h2 
#     # b2 = gtj_21[:, 0] * h1 + gtj_22[:, 0] * h2 

#     # v2
#     # b1 = jnp.squeeze(gtj_11) * h1 + jnp.squeeze(gtj_12) * h2 
#     # b2 = jnp.squeeze(gtj_21) * h1 + jnp.squeeze(gtj_22) * h2 

#     # v3
#     # Apply matrix blocks to vector halves
#     # Make sure we're getting the right dimensions by using explicit shapes
#     gtj_11_flat = jnp.squeeze(gtj_11, axis=-2)
#     gtj_12_flat = jnp.squeeze(gtj_12, axis=-2)
#     gtj_21_flat = jnp.squeeze(gtj_21, axis=-2)
#     gtj_22_flat = jnp.squeeze(gtj_22, axis=-2)
#     # Compute the transformed vector parts
#     b1 = gtj_11_flat * h1 + gtj_12_flat * h2
#     b2 = gtj_21_flat * h1 + gtj_22_flat * h2

#     b = jnp.concatenate((b1, b2), axis=-1) + htj
#     return a, b


def multiply_diags_vec(gts, yts):
    gt_11, gt_12, gt_21, gt_22 = jnp.split(gts, (1,2,3), axis=1)
    h1, h2 = jnp.split(yts, 2, axis=-1)
    b1 = gt_11[:, 0, :] * h1 + gt_12[:, 0, :] * h2 
    b2 = gt_21[:, 0, :] * h1 + gt_22[:, 0, :] * h2 
    b = jnp.concatenate((b1, b2), axis=-1)
    return b


def block_diagonal_matmul_recursive(
    mats: jnp.ndarray, vecs: jnp.ndarray, y0: jnp.ndarray
) -> jnp.ndarray:
    """

    Perform the matrix multiplication recursively, y[i + 1] = mats[i] @ y[i] + vec[i].

    Arguments
    ---------
    mats: jnp.ndarray
        The diagonals of the block diagonal components of the matrices to be multiplied, shape (nsamples - 1, 4, ny/2)
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
    # diags of identity matrix
    half_D = mats.shape[-1]
    eye = jnp.vstack(( jnp.ones((1, half_D)), jnp.zeros((2, half_D)), jnp.ones((1, half_D))))[None, ...] # (1, 4, ny/2)
    first_elem = jnp.concatenate((eye, mats), axis=0)  # (nsamples, 4, ny/2)
    second_elem = jnp.concatenate((y0[None], vecs), axis=0)  # (nsamples, ny)

    # perform the scan
    elems = (first_elem, second_elem)
    _, yt = jax.lax.associative_scan(diagonal_block_binary_operator, elems)
    return yt  # (nsamples, ny)


def block_diagonal_seq1d_inv_lin(
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
    yt = block_diagonal_matmul_recursive(-gmat, rhs, y0)[1:]  # (nsamples, ny)
    return yt


def block_diagonal_deer_iteration_helper(
    inv_lin: Callable[[List[jnp.ndarray], jnp.ndarray, Any], jnp.ndarray],
    func: Callable[[List[jnp.ndarray], Any, Any], jnp.ndarray],
    logp: Callable[jnp.ndarray, jnp.ndarray],
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
    full_trace: bool = False,  # XG addition
    qmem_efficient: bool = False,  # XG addition
) -> Tuple[jnp.ndarray, Optional[List[jnp.ndarray]], Callable]:
    # obtain the functions to compute the jacobians and the function
    func2 = jax.vmap(func, in_axes=(0, 0, None))

    dtype = yinit_guess.dtype
    # set the tolerance to be 1e-4 if dtype is float32, else 1e-7 for float64
    tol = 1e-7 if dtype == jnp.float64 else 1e-4

    if qmem_efficient:
        def hvp(x, v):
            # return jax.jvp(jax.grad(logp),(x,), (v,))[1]
            grad_x = lambda x : params["basis"].T @ jax.grad(logp)(params["basis"] @ x)
            return jax.jvp(grad_x,(x,), (v,))[1]
        # keys = jr.split(jr.PRNGKey(0), (xinput.shape[0]))
        keys = jr.split(params['key'], (xinput.shape[0]))

    def iter_func(
        iter_inp: Tuple[jnp.ndarray, jnp.ndarray, List[jnp.ndarray], jnp.ndarray]
    ) -> Tuple[jnp.ndarray, jnp.ndarray, List[jnp.ndarray], jnp.ndarray]:
        err, yt, gt_, iiter = iter_inp
        # gt_ is not used, but it is needed to return at the end of scan iteration
        # yt: (nsamples, ny)
        ytparams = shifter_func(yt, shifter_func_params)
        # XG change to be more memory efficient
        if qmem_efficient:
            # import pdb; pdb.set_trace()
            gts = [
               -jax.vmap(leapfrog_derivative_diag_blocks_mem, in_axes=(0, 0, None, None, 0))(
                   ytparams[0], xinput, params, hvp, keys
               )
            ]
        else:
            gts = [
               -jax.vmap(leapfrog_derivative_diag_blocks, in_axes=(0, 0, None, None))(
                   ytparams[0], xinput, params, logp
               )
            ]
        # rhs: (nsamples, ny)
        rhs = func2(ytparams, xinput, params)  # (carry, input, params) 
        rhs += sum(
            [
                multiply_diags_vec(gt, ytp) for gt, ytp in zip(gts, ytparams)
            ]  # adjusted to deal with scalars
        )
        yt_next = inv_lin(gts, rhs, inv_lin_params)  # (nsamples, ny)

        if clip_ytnext:
            clip = 1e8
            yt_next = jnp.clip(yt_next, a_min=-clip, a_max=clip)
            yt_next = jnp.where(jnp.isnan(yt_next), 0.0, yt_next)

        err = jnp.max(jnp.abs(yt_next - yt))  # checking convergence
        return err, yt_next, gts, iiter + 1

    def scan_func(iter_inp, args):
        err, yt, gt_, iiter = iter_inp
        # gt_ is not used, but it is needed to return at the end of scan iteration
        # yt: (nsamples, ny)
        ytparams = shifter_func(yt, shifter_func_params)
        if qmem_efficient:
            print("efficient")
            gts = [
               -jax.vmap(leapfrog_derivative_diag_blocks_mem, in_axes=(0, 0, None, None, 0))(
                   ytparams[0], xinput, params, hvp, keys
               )
            ]
        else:
            gts = [
               -jax.vmap(leapfrog_derivative_diag_blocks, in_axes=(0, 0, None, None))(
                   ytparams[0], xinput, params, logp
               )
            ]
        # rhs: (nsamples, ny)
        rhs = func2(ytparams, xinput, params)  # (carry, input, params) 
        rhs += sum(
            [
                multiply_diags_vec(gt, ytp) for gt, ytp in zip(gts, ytparams)
            ]  # adjusted to deal with scalars
        )
        yt_next = inv_lin(gts, rhs, inv_lin_params)  # (nsamples, ny)

        err = jnp.max(jnp.abs(yt_next - yt))  # checking convergence
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
        (yinit_guess.shape[0], 4, int(yinit_guess.shape[-1]/2)),
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
        rhs = None 
    else:
        rhs = jnp.zeros_like(gts[0])  # (nsamples, ny)
    return yt, gts, rhs, func, samp_iters


def leapfrog_derivative_diag_blocks(state, inputs, params, logp):
    step_size = params["epsilon"]
    z, m = jnp.split(state, 2)
    half_D = z.shape[0] 
    diag11 = jnp.ones((1, half_D))
    diag12 = step_size * jnp.ones((1, half_D))
    hess_diag = jnp.diag(jax.hessian(logp)(z+step_size*m))
    diag21 = step_size * hess_diag[None, :]
    diag22 = jnp.ones((1, half_D)) + step_size**2 * hess_diag
    out = jnp.vstack(( diag11, diag12, diag21, diag22 ))
    return out


"""
def leapfrog_derivative_diag_blocks_mem(state, inputs, params, hvp, key):
    step_size = params["epsilon"]
    z, m = jnp.split(state, 2)
    if params["mass_diag"] is None:
        mass_diag = jnp.ones((m.shape[0]))
    else:
        mass_diag = params["mass_diag"]
    half_D = z.shape[0] 
    diag11 = jnp.ones((1, half_D))
    diag12 = step_size * jnp.ones((1, half_D)) / mass_diag[None, :]
    z_rad = jr.rademacher(key, (z.shape)).astype(float)
    hess_diag = z_rad * hvp(z+step_size*m/mass_diag, z_rad)
    diag21 = step_size * hess_diag[None, :]
    diag22 = jnp.ones((1, half_D)) + step_size**2 * hess_diag / mass_diag
    out = jnp.vstack(( diag11, diag12, diag21, diag22 ))
    return out
"""
def leapfrog_derivative_diag_blocks_mem(state, inputs, params, hvp, key):
    step_size = params["epsilon"]
    z, m = jnp.split(state, 2)
    # mass_diag = params["mass_diag"]
    mass_diag = jnp.ones_like(z) #TODO: change
    half_D = z.shape[0] 
    diag11 = jnp.ones((1, half_D))
    diag12 = step_size * jnp.ones((1, half_D)) / mass_diag[None, :]
    z_rad = jr.rademacher(key, (z.shape)).astype(float)
    hess_diag = z_rad * hvp(z+step_size*m/mass_diag, z_rad)
    diag21 = step_size * hess_diag[None, :]
    diag22 = jnp.ones((1, half_D)) + step_size**2 * hess_diag / mass_diag
    out = jnp.vstack(( diag11, diag12, diag21, diag22 ))
    return out


# def leapfrog_derivative_diag_blocks_mem(state, inputs, params, hvp, key):
#     step_size = params["epsilon"]
#     z, m = jnp.split(state, 2)
#     mass_diag = params["mass_diag"]
#     half_D = z.shape[0] 
#     diag11 = jnp.ones((1, half_D))
#     diag12 = step_size * jnp.ones((1, half_D)) / mass_diag[None, :]
#     z_rad = jr.rademacher(key, (z.shape)).astype(float)
#     hess_diag = z_rad * params["basis"].T @ hvp(z+step_size*m/mass_diag, params["basis"] @ z_rad)
#     diag21 = step_size * hess_diag[None, :]
#     diag22 = jnp.ones((1, half_D)) + step_size**2 * hess_diag / mass_diag
#     out = jnp.vstack(( diag11, diag12, diag21, diag22 ))
#     return out