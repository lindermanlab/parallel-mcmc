"""
elk.py
code to write Evaluating Levenberg-Marquardt with Kalman (ELK) code

Note that lambda = 1 / sigmasq. we use lambda in the paper, but sigmasq in the code.
"""

import jax
from jax import lax, vmap
from jax.lax import scan

import jax.numpy as jnp

from jaxtyping import Array, Float
from typing import NamedTuple, Union, Optional
from functools import partial

class ScalarParams(NamedTuple):
    """
    Class to hold the params we are going to use in ELK
    We hold for both the sequence length and the state dimension
    """

    initial_mean: Float[Array, "state_dim"]
    dynamics_weights: Float[Array, "ntime state_dim"]
    dynamics_bias: Float[Array, "ntime state_dim"]
    emission_noises: Float[Array, "ntime state_dim"]


def make_scalar_params(initial_mean, dynamics_weights, dynamics_bias, emission_noises):
    return ScalarParams(initial_mean, dynamics_weights, dynamics_bias, emission_noises)

class PosteriorScalarFilter(NamedTuple):
    r"""Marginals of the Gaussian filtering posterior.
    :param filtered_means: array of filtered means $\mathbb{E}[z_t \mid y_{1:t}, u_{1:t}]$
    :param filtered_covariances: array of filtered covariances $\mathrm{Cov}[z_t \mid y_{1:t}, u_{1:t}]$
    """

    filtered_means: Optional[Float[Array, "ntime"]] = None
    filtered_covariances: Optional[Float[Array, "ntime"]] = None


class FilterMessageScalar(NamedTuple):
    """
    Filtering associative scan elements.

    Note that every one of these is a scalar in our formulation
    but tha they span the sequence length

    Attributes:
        A: P(z_j | y_{i:j}, z_{i-1}) weights.
        b: P(z_j | y_{i:j}, z_{i-1}) bias.
        C: P(z_j | y_{i:j}, z_{i-1}) covariance.
        J:   P(z_{i-1} | y_{i:j}) covariance.
        eta: P(z_{i-1} | y_{i:j}) mean.
    """

    A: Float[Array, "ntime"]
    b: Float[Array, "ntime"]
    C: Float[Array, "ntime"]
    J: Float[Array, "ntime"]
    eta: Float[Array, "ntime"]


def _initialize_filtering_messages(params, emissions):
    """Preprocess observations to construct input for filtering assocative scan."""

    num_timesteps = emissions.shape[0]

    def _first_message(params, y):
        m = params.initial_mean
        sigma2 = params.emission_noises[0]

        S = jnp.ones(1) + sigma2

        A = jnp.zeros(1)
        b = m + (y - m) / S
        C = jnp.ones(1) - (S**-1)
        eta = jnp.zeros(1)
        J = jnp.ones(1)

        return A, b, C, J, eta

    @partial(vmap, in_axes=(None, 0, 0))
    def _generic_message(params, y, t):
        """
        Notes:
            * y are the observations
            * note that the dynamics params and the emissions params are shifted by 1 in a sense
        """
        F = params.dynamics_weights[t]
        b = params.dynamics_bias[t]
        sigma2 = params.emission_noises[t + 1]  # shift

        K = 1 / (1 + sigma2)

        eta = F * K * (y - b)
        J = (F**2) * K

        A = F - K * F
        b = b + K * (y - b)
        C = 1 - K

        return A, b, C, J, eta

    A0, b0, C0, J0, eta0 = _first_message(params, emissions[0])
    At, bt, Ct, Jt, etat = _generic_message(
        params,
        emissions[1:],
        jnp.arange(
            len(emissions) - 1
        ),  # the dynamics params at step 0 generated step 1
    )

    return FilterMessageScalar(
        A=jnp.concatenate([A0, At]),
        b=jnp.concatenate([b0, bt]),
        C=jnp.concatenate([C0, Ct]),
        J=jnp.concatenate([J0, Jt]),
        eta=jnp.concatenate([eta0, etat]),
    )


def parallel_scalar_filter(
    params: ScalarParams,
    emissions: Float[Array, "ntime"],
):
    """
    Notes:
        * really meant to solve the scalar problem
    """

    @vmap
    def _operator(elem1, elem2):
        A1, b1, C1, J1, eta1 = elem1
        A2, b2, C2, J2, eta2 = elem2

        denom = C1 * J2 + 1

        A = (A1 * A2) / denom
        b = A2 * (C1 * eta2 + b1) / denom + b2
        C = C1 * (A2**2) / denom + C2

        eta = A1 * (eta2 - J2 * b1) / denom + eta1
        J = J2 * (A1**2) / denom + J1

        return FilterMessageScalar(A, b, C, J, eta)

    initial_messages = _initialize_filtering_messages(params, emissions)
    final_messages = lax.associative_scan(_operator, initial_messages)

    return PosteriorScalarFilter(
        filtered_means=final_messages.b,
        filtered_covariances=final_messages.C,
    )

def elk_alg(
    f,
    initial_state,
    states_guess,
    drivers,
    sigmasq=1e8,
    num_iters=10,
    quasi=False,
    AR=False,
    deer=False,
    full_trace=True,
    max_iter=10000,
):
    """
    Run ELK to evaluate the model. Uses a Kalman filter.

    Args:
      f: a forward fxn that takes in a full state and a driver, and outputs the next full state.
          In the context of a GRU, f is a GRU cell, the full state is the hidden state, and the driver is the input
      initial_state: packed_state, jax.Array (DIM,)
      states_guess, jax.Array, (L-1, DIM)
      drivers, jax.Array, (L-1,N_noise)
      sigmasq: float, controls regularization (high sigmasq -> low regularization)
      num_iters: number of iterations to run
      quasi: bool, whether to use quasi-newton or not
      AR: bool, basically evaluate autoregressively (Jacobi iterations, zeroth order) 
      deer: bool, whether to use deer or not (equivalent to sigmasq=infinity, but more numerically stable)
    Notes:
    - The initial_state is NOT the same as the initial mean we give to dynamax
    - The initial_mean is something on which we do inference
    - The initial_state is the fixed starting point.

    The structure looks like the following.
    Let h0 be the initial_state (fixed), h[1:L-1] be the states, and e[0:L-2] be the drivers

    Then our graph looks like

    h0 -----> h1 ---> h2 ---> ..... h_{L-2} ----> h_{L-1}
              |       |                   |          |
              e1      e2       ..... e_{L-2}      e_{L-1}
    """
    DIM = len(initial_state)
    L = len(drivers)

    @jax.vmap
    def full_mat_operator(q_i, q_j):
        """Binary operator for parallel scan of linear recurrence. Assumes a full Jacobian matrix A
        Args:
            q_i: tuple containing J_i and b_i at position i       (P,P), (P,)
            q_j: tuple containing J_j and b_j at position j       (P,P), (P,)
        Returns:
            new element ( A_out, Bu_out )
        """
        A_i, b_i = q_i
        A_j, b_j = q_j
        return A_j @ A_i, A_j @ b_i + b_j

    @jax.vmap
    def diag_mat_operator(q_i, q_j):
        """Binary operator for parallel scan of linear recurrence. Assumes a DIAGONAL Jacobian matrix A
        Args:
            q_i: tuple containing J_i and b_i at position i       (P,P), (P,)
            q_j: tuple containing J_j and b_j at position j       (P,P), (P,)
        Returns:
            new element ( A_out, Bu_out )
        """
        A_i, b_i = q_i
        A_j, b_j = q_j
        return A_j * A_i, A_j * b_i + b_j

    @jax.jit
    def _step(states, args):
        # Evaluate f and its Jacobian in parallel across timesteps 1,..,T-1
        fs = vmap(f)(states[:-1], drivers[1:])
        Jfs = vmap(jax.jacrev(f, argnums=0))(
            states[:-1], drivers[1:]
        )  

        # Compute the As and bs from fs and Jfs
        if quasi:
            As = vmap(lambda Jf: jnp.diag(Jf))(Jfs)
            bs = fs - As * states[:-1]
        elif AR:
            As = jnp.zeros_like(Jfs)
            bs = fs
        else:
            As = Jfs
            bs = fs - jnp.einsum("tij,tj->ti", As, states[:-1])

        if quasi and not deer:
            params = make_scalar_params(
                initial_mean=f(initial_state, drivers[0]),
                dynamics_weights=As,
                dynamics_bias=bs,
                emission_noises=jnp.ones(L) * sigmasq,
            )
        elif deer:
            # initial_state is h0
            b0 = f(initial_state, drivers[0])  # h1
            A0 = jnp.zeros_like(As[0])
            A = jnp.concatenate(
                [A0[jnp.newaxis, :], As]
            )  # (L, D, D) [or (L, D) for quasi]
            b = jnp.concatenate([b0[jnp.newaxis, :], bs])  # (L, D)
            if quasi:
                binary_op = diag_mat_operator
            else:
                binary_op = full_mat_operator
        else:
            params = make_lgssm_params(
                initial_mean=f(initial_state, drivers[0]),
                initial_cov=jnp.eye(DIM),
                dynamics_weights=As,
                dynamics_bias=bs,
                dynamics_cov=jnp.eye(DIM),
                emissions_weights=jnp.eye(DIM),
                emissions_cov=jnp.eye(DIM) * sigmasq,
                emissions_bias=jnp.zeros(DIM),
            )
        # run appropriate parallel alg
        if deer:
            _, new_states = jax.lax.associative_scan(binary_op, (A, b))
        elif quasi:
            post = jax.vmap(
                parallel_scalar_filter,
                in_axes=(
                    ScalarParams(0, 1, 1, None),
                    1,
                ),
                out_axes=1,
            )(params, states)
            new_states = post.filtered_means
        else:
            post = lgssm.parallel_inference.lgssm_filter(params, states)
            new_states = post.filtered_means
        return new_states, new_states

    if full_trace:
        _, states_iters = scan(_step, states_guess, None, length=num_iters)
        missing_init_state = jnp.vstack((states_guess[None, ...], states_iters))
        everything = jnp.concatenate(
            (
                jnp.broadcast_to(
                    initial_state,
                    (missing_init_state.shape[0], 1, missing_init_state.shape[-1]),
                ),
                missing_init_state,
            ),
            axis=1,
        )
        return everything
    else:
        dtype = states_guess.dtype
        err = jnp.array(1e10, dtype=dtype)  # initial error should be very high
        tol = 1e-7 if dtype == jnp.float64 else 1e-4
        iiter = jnp.array(0, dtype=jnp.int32)
        def cond_func(iter_inp):
            err, _, iiter = iter_inp
            return jnp.logical_and(err > tol, iiter < max_iter)

        def iter_func(iter_inp):
            err, yt, iiter = iter_inp
            yt_next, _ = _step(yt, None)
            err = jnp.max(jnp.abs(yt_next - yt))  # checking convergence
            return err, yt_next, iiter + 1

        _, yt, samp_iters = jax.lax.while_loop(
            cond_func, iter_func, (err, states_guess, iiter)
        )
        return yt, samp_iters


@jax.vmap
def full_mat_operator(q_i, q_j):
    """Binary operator for parallel scan of linear recurrence. Assumes a full Jacobian matrix A
    Args:
        q_i: tuple containing J_i and b_i at position i       (P,P), (P,)
        q_j: tuple containing J_j and b_j at position j       (P,P), (P,)
    Returns:
        new element ( A_out, Bu_out )
    """
    A_i, b_i = q_i
    A_j, b_j = q_j
    return A_j @ A_i, A_j @ b_i + b_j

@jax.vmap
def diag_mat_operator(q_i, q_j):
    """Binary operator for parallel scan of linear recurrence. Assumes a DIAGONAL Jacobian matrix A
    Args:
        q_i: tuple containing J_i and b_i at position i       (P,P), (P,)
        q_j: tuple containing J_j and b_j at position j       (P,P), (P,)
    Returns:
        new element ( A_out, Bu_out )
    """
    A_i, b_i = q_i
    A_j, b_j = q_j
    return A_j * A_i, A_j * b_i + b_j


def quasi_elk(
    f,
    initial_state,
    states_guess,
    drivers,
    sigmasq=1e8,
    num_iters=10,
    full_trace=True,
    max_iter=10000,
):
    """
    Run ELK to evaluate the model. Uses a Kalman filter.

    Args:
      f: a forward fxn that takes in a full state and a driver, and outputs the next full state.
          In the context of a GRU, f is a GRU cell, the full state is the hidden state, and the driver is the input
      initial_state: packed_state, jax.Array (DIM,)
      states_guess, jax.Array, (L-1, DIM)
      drivers, jax.Array, (L-1,N_noise)
      sigmasq: float, controls regularization (high sigmasq -> low regularization)
      num_iters: number of iterations to run
      quasi: bool, whether to use quasi-newton or not
      AR: bool, basically evaluate autoregressively (Jacobi iterations, zeroth order) 
      deer: bool, whether to use deer or not (equivalent to sigmasq=infinity, but more numerically stable)
    Notes:
    - The initial_state is NOT the same as the initial mean we give to dynamax
    - The initial_mean is something on which we do inference
    - The initial_state is the fixed starting point.

    The structure looks like the following.
    Let h0 be the initial_state (fixed), h[1:L-1] be the states, and e[0:L-2] be the drivers

    Then our graph looks like

    h0 -----> h1 ---> h2 ---> ..... h_{L-2} ----> h_{L-1}
              |       |                   |          |
              e1      e2       ..... e_{L-2}      e_{L-1}
    """
    DIM = len(initial_state)
    L = len(drivers)

    # obtain the functions to compute the jacobians and the function
    jacfunc = jax.vmap(jax.jacrev(f, argnums=0), in_axes=(0, 0))
    func2 = jax.vmap(f, in_axes=(0, 0))

    @jax.jit
    def _step(states, args):
        # fs = func2(states[:-1], drivers[1:])
        # Jfs = jacfunc(states[:-1], drivers[1:])
        # # Compute the As and bs from fs and Jfs
        # As = vmap(lambda Jf: jnp.diag(Jf))(Jfs)
        def elk_jvp(z, driver, v):
            return jax.jvp(lambda z : f(z, driver), (z,), (v, ))
        z_rad = jax.random.rademacher(jax.random.PRNGKey(13), (states.shape[0]-1, states.shape[1])).astype(float) # 1 sample estimate
        fs, jvp_vals = jax.vmap(elk_jvp)(states[:-1], drivers[1:], z_rad)
        As = z_rad * jvp_vals 
        bs = fs - As * states[:-1]

        params = make_scalar_params(
            initial_mean=f(initial_state, drivers[0]),
            dynamics_weights=As,
            dynamics_bias=bs,
            emission_noises=jnp.ones(L) * sigmasq,
        )

        # run appropriate parallel alg
        post = jax.vmap(
            parallel_scalar_filter,
            in_axes=(
                ScalarParams(0, 1, 1, None),
                1,
            ),
            out_axes=1,
        )(params, states)
        new_states = post.filtered_means

        return new_states, new_states

    if full_trace:
        _, states_iters = scan(_step, states_guess, None, length=num_iters)
        missing_init_state = jnp.vstack((states_guess[None, ...], states_iters))
        everything = jnp.concatenate(
            (
                jnp.broadcast_to(
                    initial_state,
                    (missing_init_state.shape[0], 1, missing_init_state.shape[-1]),
                ),
                missing_init_state,
            ),
            axis=1,
        )
        return everything
    else:
        dtype = states_guess.dtype
        err = jnp.array(1e10, dtype=dtype)  # initial error should be very high
        tol = 1e-7 if dtype == jnp.float64 else 1e-4
        iiter = jnp.array(0, dtype=jnp.int32)
        def cond_func(iter_inp):
            err, _, iiter = iter_inp
            return jnp.logical_and(err > tol, iiter < max_iter)

        def iter_func(iter_inp):
            err, yt, iiter = iter_inp
            yt_next, _ = _step(yt, None)
            err = jnp.max(jnp.abs(yt_next - yt))  # checking convergence
            return err, yt_next, iiter + 1

        _, yt, samp_iters = jax.lax.while_loop(
            cond_func, iter_func, (err, states_guess, iiter)
        )
        return yt, samp_iters
