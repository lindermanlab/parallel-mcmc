import jax
jax.config.update("jax_default_matmul_precision", "highest")
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Float, Int, Bool, UInt32
from typing import Union, Optional, Tuple, TypeAlias
from collections.abc import Callable
from functools import partial

'''
Written by Skyler Wu, implementing Zoltowski & Wu et al.'s "Parallelizing MCMC Across the Sequence Length."

The following is a modernized, modular version of Parallel MALA, with memory-efficient quasi-DEER.
'''

#### HELPER FUNCTIONS ####


# helper function to implement the stop-gradient trick for accept/reject differentiability
def sigmoid_accept(
    x : Float[Array, "batch 1"], 
    beta : float = 1.0) -> Float[Array, "batch 1"]:
    '''
    On forward pass, returns 1.0 if x > 0, else 0.0 (i.e., hard accept/reject)
    On the backwards pass, uses d/dx sigmoid(beta*x) as a surrogate gradient (smooth, nonzero near 0).

    Args:
        x - input drivers (i.e., Uniform r.v. draws) for implementing accept/reject.
        beta - tempering constant on the sigmoid. Larger beta means closer to hard accept/reject.

    Returns:
        1.0 if x > 0 else 0.0.
    '''
    soft = jax.nn.sigmoid(x * beta) # what gets used for the backwards pass.
    hard = (x > 0).astype(soft.dtype) # what gets used for the forward pass. Need to cast so subtraction dtypes work out.
    return soft + jax.lax.stop_gradient(hard - soft)


'''
Hutchinson's diagonal Jacobian estimator: mathbb{E}[r otimes (Jr)] = diag(J)
- Idea is that JVPs Jr are much easier to compute than the actual Jacobians themselves.
- Here, r \in R^d is comprised of i.i.d. Rademachers (random signs). Could also use Normals, etc.
'''
# function to use Hutchinson's estimator to estimate diag(Jacobians)
def hutchinson(
    fhat_t : Callable[[Float[Array, "n d"], UInt32[Array, "n 2"], UInt32[Array, "n 2"]], Float[Array, "n d"]],
    z_tm1s : Float[Array, "n d"],
    keys_norm : UInt32[Array, "n 2"],
    keys_unif : UInt32[Array, "n 2"],
    keys_hutchinson : UInt32[Array, "n 2"],
    n_probes : int,
    d : int) -> tuple[Float[Array, "n d"], Float[Array, "n d"]]:
    '''
    Hutchinson's estimator to estimate diagonal Jacobians.

    Args:
        fhat_t - function that outputs next MCMC sample, given current sample. Only takes as input (current values, keys_norm, keys_unif)
        z_tm1s - current samples in the transformed space.
        keys_norm - PRNG keys for generating the MVN(0, I_d) for the proposal.
        keys_unif - PRNG keys for generating the Unif(0, 1) for the accept/reject implementation.
        keys_hutchinson - PRNG keys for generating the n_probes Rademacher probes for each Jacobian estimate.
        n_probes - how many Rademacher vectors are we using to estimate the diag(Jacobian) via Hutchinson's?
        d - how many dimensions is our parameter space for sampling?

    Returns:
        fhat_t_vals - function evals at the given z_tm1s.
        hutch - a matrix, where each row is the Hutchinson's estimator of the diag(J_tm1) for each z_tm1.

    *Remark: (n_probes, d) will be static arguments in upstream jit-compilation.
    '''

    # a quick helper function to close out the keys_norm and keys_unif when calling fhat_t (in transformed space)
    fhat_t_closed = partial(fhat_t, keys_norm=keys_norm, keys_unif=keys_unif) # (n, d) -> (n, d)

    '''
    Remark: in the future, if self.d is too large, we can compute the probes as needed, as opposed to at once for full window.
    '''
    # sample our rademacher probes + make sure they're floats for compatibility (1 set of n_probe d-dim Rademacher vectors per key).
    r = jax.vmap(
        fun=partial(jr.rademacher, shape=(n_probes, d), dtype=z_tm1s.dtype), 
        in_axes=0, out_axes=1)(keys_hutchinson) # (n_probes, n, d)

    # go with vmap + jvp for all settings - gets us the batched forward-pass values. Need primals & tangents as tuples!
    fhat_t_vals_all, jvps = jax.vmap(
        fun=lambda rs : jax.jvp(fhat_t_closed, (z_tm1s,), (rs,)), # rs is (n, d), i.e., 1 probe per primal point.
        in_axes=0, out_axes=0)(r) # (n_probes, n, d), (n_probes, n, d)

    # extract out the primal outputs, no need so many copies
    fhat_t_vals = fhat_t_vals_all[0] # (n, d)

    # return the fhat_t_vals and the hutchinson estimators of diag(J_t): mean of r \otimes (Jr) 
    return fhat_t_vals, (r * jvps).mean(axis=0) # (n, d), (n, d)


# function to compose A_1 x + b_1 and A_2 x + b_2, i.e. A_2 (A_1 x + b_1) + b_2 = (A_2 A_1) x + (A_2 b_1 + b_2), w/t diag A_i.
def compose_diag(
    left : Tuple[Float[Array, "d"], Float[Array, "d"]], 
    right : Tuple[Float[Array, "d"], Float[Array, "d"]]) -> Tuple[Float[Array, "d"], Float[Array, "d"]]:
    '''
    Takes two parallel-scan elements and returns their composition.
    
    Args:
        left - (A1, b1), where A1 is a diagonal matrix, only storing its diagonal entries as a vector, and b1 is a vector.
        right - (A2, b2), where A2 is a diagonal matrix, only storing its diagonal entries as a vector, and b2 is a vector.

    Returns:
        (A2 A1, A2 b1 + b2), because A2 (A1 x + b1) + b2 = (A2 A1) x + (A2 b1 + b2).
    '''
    
    # unpack the (a_i, b_i) involved. a_i here is diagonal entries of the diagonal A_i matrix.
    a1, b1 = left
    a2, b2 = right

    # because diagonal matrices, can work as vectors with element-wise multiplication
    return ( a2 * a1, (a2 * b1) + b2 )

# helper return types for .sample_single_chain() output: full-trace vs. not. FOR SINGLE CHAINS.
SampleFinalType : TypeAlias = Tuple[Float[Array, "chain_length_plus_1 d"], Int[Array, ""]]
SampleTraceType : TypeAlias = Float[Array, "max_iters chain_length_plus_1 d"]

# helper return types for .sample() output: full-trace vs. not. FOR MULTIPLE CHAINS.
BatchedSampleFinalType : TypeAlias = Tuple[Float[Array, "b chain_length_plus_1 d"], Int[Array, "b"]]
BatchedSampleTraceType : TypeAlias = Float[Array, "b max_iters chain_length_plus_1 d"]


#### MCMC KERNEL FUNCTIONS, CLASS FREE ####


# function that computes transition function for MALA updates, potentially with orthog. transformation (Eq. 11 in Zoltowski & Wu et al.)
def fhat_t_mala(
    z_tm1s : Float[Array, "n d"], 
    epsilon : float,
    keys_norm : UInt32[Array, "n 2"], 
    keys_unif : UInt32[Array, "n 2"],
    logp_and_grad_batched : Callable[[Float[Array, "n d"]], tuple[Float[Array, "n"], Float[Array, "n d"]]],
    d : int,
    use_Q : bool,
    Q : Float[Array, "d d"],
    beta : float) -> Float[Array, "n d"]:
    '''
    For math notation, let z_t and s_t be COLUMN vectors. We store them as row vectors in (n, d) structures!
    Our MALA update function in the new coordinate system z_t = Q^T s_t.
    z_t = Q^T f_t(Q z_tm1) =(defined) fhat_t(z_tm1)

    Args:
        z_tm1s - (potentially) orthogonal-transformed coordinates of current samples.
        epsilon - MALA stepsize for the proposal.
        keys_norm - PRNG keys for generating the MVN(0, I_d) for the proposal.
        keys_unif - PRNG keys for generating the Unif(0, 1) for the accept/reject implementation.
        logp_and_grad_batched - function that returns (the target log-density and the gradient) at each input.
        d - dimensionality of our sample space.
        use_Q - Boolean of whether we are doing an orthogonal transformation or not.
        Q - dxd orthogonal transformation matrix.
        beta - tempering strength of the sigmoid accept/reject gate.

    Returns:
        z_ts - next sampler steps in the transformed coordinates.

    *Note that because z_tm1s is (n, d), i.e, each ROW is a datapoint, QT/Q are NOT typoes!
    *3/6/2026: note that (d, use_Q, beta) will be treated as static values during jit-compilation upstream! If-statements preferable.
    '''

    # orthogonal transform: x_tm1 = Q z_tm1 (use_Q is static, if-statement preferable).
    x_tm1s = z_tm1s @ Q.T if use_Q else z_tm1s # (n, d)

    #### START OF STANDARD MALA LOGIC ####
        
    # compute our logp's & grads at the current states x_tm1
    logps, grad_logps = logp_and_grad_batched(x_tm1s) # (n,) & (n,d)
    
    # proposal: \xtilde_t = x_tm1 + \eps \grad_x logp(x_tm1) + \sqrt{2\epsilon} \xi_t, \xi_t ~ N(0, I_d)
    xis = jax.vmap(fun=partial(jr.normal, shape=(d,)), in_axes=0, out_axes=0)(keys_norm) # (n, d)
    sqrt2eps_xis = jnp.sqrt(2.0 * epsilon) * xis # (n, d)
    xtilde_ts = x_tm1s + (epsilon * grad_logps) + sqrt2eps_xis # (n, d)

    # compute our logp's & grads at the proposal states xtilde_ts
    logps_tilde, grad_logps_tilde = logp_and_grad_batched(xtilde_ts)

    '''
    x | xtilde ~ N(xtilde + eps * grad log p(xtilde), 2*eps * I)
    xtilde | x ~ N(x + eps * grad log p(x), 2*eps * I)
    '''
    # acceptance probabilities: alpha = min(1, p(\xtilde_t)q(x_tm1 | \xtilde_t) / p(x_tm1)q(\xtilde_t | x_tm1))
    v_num = x_tm1s - (xtilde_ts + (epsilon * grad_logps_tilde))

    # previously: v_den = xtilde_ts - (x_tm1s + (epsilon * grad_logps)), which is just sqrt2eps_xis!
    log_alpha = (logps_tilde - logps) - ( (v_num * v_num).sum(axis=-1) - (sqrt2eps_xis * sqrt2eps_xis).sum(axis=-1) ) / (4.0 * epsilon) # (n,)

    # accept or reject step - generate our uniform drivers (deterministic given keys) + do accept/reject.
    us = jnp.clip(
        jax.vmap(fun=partial(jr.uniform, shape=()), in_axes=0, out_axes=0)(keys_unif), 
        min=1e-30) # uniform drivers, clipped for numerical stability with log.
    gs = sigmoid_accept((log_alpha - jnp.log(us))[:,None], beta=beta) # (n, 1)
    x_ts = gs*xtilde_ts + (1.0 - gs)*x_tm1s

    #### END OF STANDARD MALA LOGIC ####
    
    # orthogonal transform back if necessary z_t = Q^T f_t(Q z_tm1) = Q^T x_ts
    z_ts = x_ts @ Q if use_Q else x_ts

    # return the z_ts
    return z_ts


#### JIT-COMPILABLE PARALLEL SAMPLER FUNCTIONS - NON-USER-FACING! ####

# our jit-compatible sampling function for ONE CHAIN. Not necessarily MALA. Can swap out fhat_t_mcmc for something else.
def _sample_single_chain(
    x0 : Float[Array, "d"],
    x_init : Float[Array, "chain_length d"],
    keys_norm : UInt32[Array, "chain_length 2"],
    keys_unif : UInt32[Array, "chain_length 2"],
    keys_hutchinson : UInt32[Array, "chain_length 2"],
    epsilon : float,
    beta : float,
    damp_factor : float,
    clip_val : float,
    rtol : float,
    atol : float,
    fhat_t_mcmc : Callable, # see fhat_t_mala for an example
    logp_and_grad_batched : Callable[[Float[Array, "n d"]], tuple[Float[Array, "n"], Float[Array, "n d"]]],
    use_Q : bool,
    Q : Float[Array, "d d"],
    chain_length : int, 
    W : int,
    n_probes : int,
    d : int,
    transition_and_jacobian_estimator : Callable = hutchinson, # see hutchinson for an example
    full_trace : bool = False,
    max_iters : int = 10000) -> Union[SampleFinalType, SampleTraceType]:
    '''
    Args:
        x0 - initial condition for our sampler
        x_init - initial guess for all chain_length samples.
        keys_norm - PRNG keys for generating MVN(0, I_d) values for MALA proposals.
        keys_unif - PRNG keys for generating Uniform(0, 1) values for MALA accept/reject.
        key_hutchinson - PRNG keys for generating d-dim Rademachers for Hutchinson estimators.
        epsilon - MALA stepsize for the proposal.
        beta - for the accept/reject sigmoid gradient surrogate, how much to temper? Larger beta: closer to hard accept/reject.
        damp_factor - how much to multiply the (diagonal) Jacobians by.
        clip_val - how much to clip entries in the (diagonal) Jacobians by (after dampening).
        rtol - relative tolerance threshold for Newton iteration convergence.
        atol - absolute tolerance threshold for Newton iteration convergence.
        fhat_t_mcmc - transition kernel for an MCMC method (see fhat_t_mala as an example). Requires a logp_and_grad_batched function.
        logp_and_grad_batched - function that returns (the target log-density and the gradient) at each input.
        use_Q - Boolean of whether we are doing an orthogonal transformation or not.
        Q - dxd orthogonal transformation matrix.
        chain_length - how many samples do we want? (1 chain).
        W - maximum size of our sliding window for local Newton updates.
        n_probes - how many Rademacher vectors are we using to estimate the diag(Jacobian) via Hutchinson's?
        d - how many dimensions is our parameter space for sampling?
        transition_and_jacobian_estimator - function to compute the transition to next sample + estimate the Jacobian (e.g., hutchinson).
        full_trace - do we want intermediate Newton iteration outputs too (True) or just the final samples (False)?
        max_iters - maximum number of (quasi)-Gauss-Newton iterations to run GLOBALLY (summed over all sliding window apps).
    
    Returns:
        (samples_final, iters_final) - our Parallel MALA samples + the no. of iterations needed, if full-trace=False.
        samples_trace - the full-trace of Parallel MALA samples for max_iters iterations, if full-trace=True.

    *Note that (full_trace, use_Q) will be treated as static at jit-compilation, so better to use if-statements.
    '''
    # build our fhat_t - transition function for MALA updates with this logp function, potentially with orthog. transformation.
    fhat_t = partial(
        fhat_t_mcmc, epsilon=epsilon, logp_and_grad_batched=logp_and_grad_batched, 
        d=d, use_Q=use_Q, Q=Q, beta=beta) # only takes (z_tm1s, keys_norm, keys_unif) as input!
    
    # current guess of <all> samples (need to concatenate x0 too) 
    x_cur = jnp.vstack([x0, x_init]) # (chain_length + 1, d)

    # convert to our z_t = Q^T x_t (but note that we're storing rows of x_cur as each sample!)
    z_cur = x_cur @ Q if use_Q else x_cur

    #### FROM HERE ON OUT, EVERYTHING IS IN TERMS OF THE TRANSFORMED COORDINATES Z_CUR ####
    
    # initialize the number of Gauss-Newton iterations performed (total, thru out entire process)
    iiters = jnp.array(0, dtype=jnp.int32)

    # starting sliding window is always gonna be the first W sample positions
    start_window = jnp.array(1, dtype=jnp.int32) # with z0 concat. in z_cur, sample 1 is the first non-z0 sample!

    '''
    val / init_val includes:
        iiters - how many Newton iterations have we performed total already?
        start_window - what is the start of our sliding window of length W always?
        z_cur - what is our current guess of the entire set of samples? (transformed basis)
    '''
    # initialize our "val" that will be updated via the jax.lax.while_loop()
    init_val = (iiters, start_window, z_cur)

    # write a type alias for this val
    ValType : TypeAlias = Tuple[Int[Array, ""], Int[Array, ""], Float[Array, "chain_length_plus_1 d"]]

    '''
    Just need to check if we hit the end of the sliding-window and/or we hit max_iters.
    '''
    # condition for whether we should continue the while loop or not
    def check_window_and_max_iters(val : ValType) -> Bool[Array, ""]:

        # unpack the necessary parts of the val
        iiters, start_window, _ = val
        
        # check whether we've hit maximum iterations or if we've hit end of the chain_length
        return jnp.logical_and(start_window <= chain_length, iiters < max_iters)
    
    '''
    Task: do 1 Newton iteration on x_cur[start_window : end_window+1, :], then shift window.
    - Note that if z_cur is stop-gradiented, then everything derived from it is also stop-gradiented!
    '''
    # doing one newton iteration on a window of size W, then shift the window.
    def one_newton_on_window(val : ValType) -> ValType:

        # unpack the val + make sure not tracking gradients
        iiters, start_window, z_cur = val
        z_cur = jax.lax.stop_gradient(z_cur)

        # what is the working initial-condition for this window?
        z0_window = jax.lax.dynamic_index_in_dim(
            operand=z_cur, index=start_window - 1, axis=0)

        # let's get the relevant keys for norm, unif, and hutchinson
        keys_norm_window = jax.lax.dynamic_slice_in_dim(
            keys_norm, start_index=start_window - 1, slice_size=W, axis=0)
        keys_unif_window = jax.lax.dynamic_slice_in_dim(
            keys_unif, start_index=start_window - 1, slice_size=W, axis=0)
        keys_hutchinson_window = jax.lax.dynamic_slice_in_dim(
            keys_hutchinson, start_index=start_window - 1, slice_size=W, axis=0)
        
        # what is our working guess for this window? Also get the shifted-by-1 timestep current guesses for quasi-DEER.
        z_cur_window = jax.lax.dynamic_slice_in_dim(
            operand=z_cur, start_index=start_window, slice_size=W, axis=0) # (W, d)
        z_cur_window_tm1 = jax.lax.dynamic_slice_in_dim(
            operand=z_cur, start_index=start_window-1, slice_size=W, axis=0) # (W, d)
    
        #### NEWTON ITERATIONS ON THIS WINDOW OF EXACTLY SIZE W ####
        
        '''
        Let us compute all the A's and b's for the current window at once. Store BOTH as (W, d) matrices.
        
        Below, with Hutchinson's estimator, we will use keys tied to absolute position, not relative position in window.
        *Note that we do start_window - 1 because we have 1 more z_cur than keys_{...}.

        *In the future, can swap out transition_and_jacobian_estimator = hutchinson to something else more domain-specific. Plug + play modularity.
        '''
        # compute the A_t^{(i+1)} = diag(J_t), where J_t = dfhat_t / dz (z_{t-1}^{(i)}) via Hutchinson's estimator (transition_and_jacobian_estimator)
        # also gets us the forward-pass fhat_t values for free.
        fhat_t_vals, A_ip1s = transition_and_jacobian_estimator(
            fhat_t=fhat_t, z_tm1s=z_cur_window_tm1, 
            keys_norm=keys_norm_window, keys_unif=keys_unif_window, keys_hutchinson=keys_hutchinson_window, 
            n_probes=n_probes, d=d) # t^th row of A_ip1s is diagonal entries of J_t. Outputs: (W, d), (W, d) 
        
        # multiply these Jacobian approximations by dampening factors and then clip
        A_ip1s = jnp.clip(A_ip1s * damp_factor, min=-clip_val, max=clip_val) # (W, d)

        # compute the b_t^{(i+1)} = fhat_t( z_{t-1}^{(i)} ) - diag(J_t) z_{t-1}^{(i)}
        b_ip1s = fhat_t_vals - (A_ip1s * z_cur_window_tm1) # t^th row of b_ip1s is b_t^{(i+1)}. # (W, d)

        # quasi-DEER state updates: s_t^{(i+1)} = A_t^{(i+1)} s_{t-1}^{(i+1)} + b_t^{(i+1)} via standard JAX parallel scan.
        A_ip1s_cum, b_ip1s_cum = jax.lax.associative_scan(fn=compose_diag, elems=(A_ip1s, b_ip1s), reverse=False, axis=0)
        z_cur_window_ip1s = (A_ip1s_cum * z0_window) + b_ip1s_cum

        # write the updated states in this sliding window z_cur_window_ip1s to z_cur
        z_cur = jax.lax.dynamic_update_slice_in_dim(operand=z_cur, update=z_cur_window_ip1s, start_index=start_window, axis=0)

        #### DETERMINING THE NEXT SLIDING WINDOW ####

        # what is the first non-converged entry in the sliding window according to rtol/atol?
        nonconvgd_mask = (jnp.abs(z_cur_window_ip1s - z_cur_window) - rtol * jnp.abs(z_cur_window)).max(axis=-1) > atol # (W, d) -> (W,)

        # are there ANY nonconverged indices? if so, get the first one. If not, move the entire window (GLOBAL indexing!)
        start_window = jnp.where(jnp.any(nonconvgd_mask), jnp.argmax(nonconvgd_mask)+start_window, W+start_window)
        
        # shift the next start_window, ensure at least length W unless we have exceeded the sequence length already ... expose to kill condition.
        start_window = jnp.where(
            start_window <= chain_length, 
            jnp.minimum(start_window, chain_length - W + 1), 
            start_window
        )

        # increment our number of Newton iterations
        iiters += 1

        # package everything up and return val
        return (iiters, start_window, z_cur)

    #### FINAL EXECUTION + RETURNING ORIGINAL-SCALE SAMPLES ####
    
    # return only the final samples if full_trace=False
    if full_trace:
        
        # a function for the jax.lax.scan for full-trace - need to have an output y for each scan step too.
        def scan_fun(carry, _): # here, carry is the same thing as "val", while x is just None: unused.
            new_carry = one_newton_on_window(carry)
            _, _, y = new_carry
            return new_carry, y

        # apply our jax.lax.scan to get our full-trace samples
        _, z_trace = jax.lax.scan(f=scan_fun, init=init_val, xs=None, length=max_iters)

        # convert z_final back to original scale x_final if called for
        return z_trace @ Q.T if use_Q else z_trace # just the samples_trace.

    else:
        
        # run our JIT-friendly while-loop.
        iters_final, start_window_final, z_final = jax.lax.while_loop(
            cond_fun=check_window_and_max_iters, body_fun=one_newton_on_window, init_val=init_val)

        # convert z_final back to original scale x_final if called for
        samples_final = z_final @ Q.T if use_Q else z_final

        # return our final converged samples & the number of iterations used.
        return samples_final, iters_final


# our jit-compatible sampling function for sampling multiple chains
@partial(jax.jit, static_argnames=(
    "fhat_t_mcmc", "logp_and_grad_batched", "use_Q",
    "chain_length", "W", "n_probes", "d", "transition_and_jacobian_estimator", "full_trace", "max_iters"))
def _sample_multiple_chains_jit(
    x0s : Float[Array, "b d"],
    xs_init : Float[Array, "b chain_length d"],
    keys_norm : UInt32[Array, "b chain_length 2"],
    keys_unif : UInt32[Array, "b chain_length 2"],
    keys_hutchinson : UInt32[Array, "b chain_length 2"],
    epsilon : float,
    beta : float,
    damp_factor : float,
    clip_val : float,
    rtol : float,
    atol : float,
    fhat_t_mcmc : Callable, # see fhat_t_mala for an example
    logp_and_grad_batched : Callable[[Float[Array, "n d"]], tuple[Float[Array, "n"], Float[Array, "n d"]]],
    use_Q : bool,
    Q : Float[Array, "d d"],
    chain_length : int, 
    W : int,
    n_probes : int,
    d : int,
    transition_and_jacobian_estimator : Callable = hutchinson, # see hutchinson for an example
    full_trace : bool = False,
    max_iters : int = 10000) -> Union[BatchedSampleFinalType, BatchedSampleTraceType]:
    '''
    Args:
        x0s - initial conditions for our sampler, per chain.
        xs_init : initial guesses for all chain_length samples for each chain.
        keys_norm - PRNG keys for generating MVN(0, I_d) values for MALA proposals, batched per chain.
        keys_unif - PRNG keys for generating Uniform(0, 1) values for MALA accept/reject, batched per chain.
        key_hutchinson - PRNG keys for generating d-dim Rademachers for Hutchinson estimators, batched per chain.
        epsilon - MALA stepsize for the proposal.
        beta - for the accept/reject sigmoid gradient surrogate, how much to temper? Larger beta: closer to hard accept/reject.
        damp_factor - how much to multiply the (diagonal) Jacobians by.
        clip_val - how much to clip entries in the (diagonal) Jacobians by (after dampening).
        rtol - relative tolerance threshold for Newton iteration convergence.
        atol - absolute tolerance threshold for Newton iteration convergence.
        fhat_t_mcmc - transition kernel for an MCMC method (see fhat_t_mala as an example). Requires a logp_and_grad_batched function.
        logp_and_grad_batched - function that returns (the target log-density and the gradient) at each input.
        use_Q - Boolean of whether we are doing an orthogonal transformation or not.
        Q - dxd orthogonal transformation matrix.
        chain_length - how many samples do we want? (1 chain).
        W - maximum size of our sliding window for local Newton updates.
        n_probes - how many Rademacher vectors are we using to estimate the diag(Jacobian) via Hutchinson's?
        d - how many dimensions is our parameter space for sampling?
        transition_and_jacobian_estimator - function to compute the transition to next sample + estimate the Jacobian (e.g., hutchinson).
        full_trace - do we want intermediate Newton iteration outputs too (True) or just the final samples (False)?
        max_iters - maximum number of (quasi)-Gauss-Newton iterations to run GLOBALLY (summed over all sliding window apps).
    
    Returns:
        (samples_final, iters_final) - our Parallel MALA samples + the no. of iterations needed, if full-trace=False, per chain.
        samples_trace - the full-trace of Parallel MALA samples, if full-trace=True, per chain.
    '''
    # helper wrapper function for one chain under the fixed arguments (chain_length, W, full_trace, max_iters)
    one_chain = partial(
        _sample_single_chain, 
        epsilon=epsilon, beta=beta, damp_factor=damp_factor, clip_val=clip_val, rtol=rtol, atol=atol, 
        fhat_t_mcmc=fhat_t_mcmc, logp_and_grad_batched=logp_and_grad_batched, use_Q=use_Q, Q=Q, 
        chain_length=chain_length, W=W, n_probes=n_probes, d=d, 
        transition_and_jacobian_estimator=transition_and_jacobian_estimator, full_trace=full_trace, max_iters=max_iters)
    
    # vmap over chain axis b (the leftmost axis) + return the batched output.
    return jax.vmap(one_chain, in_axes=(0, 0, 0, 0, 0), out_axes=0)(
        x0s, xs_init, keys_norm, keys_unif, keys_hutchinson
    )


#### PARALLEL MALA CLASS ####


# main modernized parallel MALA class with quasi-DEER, basis transformation, and sliding window.
class ParallelMALA:
    
    # constructor - this is written for B=1 chain, but can support arbitrary B chains in parallel via vmap.
    def __init__(self, 
                 dim : int,
                 epsilon : float,
                 beta : float = 1.0,
                 n_probes : int = 1,
                 basis_transformation : Optional[Float[Array, "d d"]] = None,
                 damp_factor : float = 1.0,
                 clip_val : float = 1e8,
                 atol : float = 1e-4,
                 rtol : float = 1e-3) -> None:
        '''
        Args:
            dim - how many dimensions is our parameter space for sampling?
            epsilon - MALA stepsize for the proposal.
            beta - for the accept/reject sigmoid gradient surrogate, how much to temper? Larger beta: closer to hard accept/reject.
            n_probes - how many Rademacher vectors are we using to estimate the diag(Jacobian) via Hutchinson's?
            basis_transformation - to facilitate more faithful diagonal-Jacobian approximations, 
                either None (no transformation) or an d x d orthogonal matrix Q to do change-of-coordinates.
            damp_factor - how much to multiply the (diagonal) Jacobians by.
            clip_val - how much to clip entries in the (diagonal) Jacobians by (after dampening).
            atol - absolute tolerance threshold for Newton iteration convergence.
            rtol - relative tolerance threshold for Newton iteration convergence.

        *Original DEER code: (atol, rtol) = (1e-7, 1e-4) if jnp.float64 else (1e-4, 1e-3) if jnp.float32 for future benchmarking.
        3/6/2026: Removed self.QT precomputation because according to official documentation,
            because "However, under JIT, the compiler will optimize-away such copies when possible, so this doesn’t have performance impacts in practice."
        3/6/2026: we will NOT store logp, logp_and_grad, or logp_and_grad_batched in self. This is because we want the safest JIT-compilation possible.
            - Instead, users should call logp_and_grad = jax.value_and_grad(self.logp) and 
            logp_and_grad_batched = jax.vmap(fun=logp_and_grad, in_axes=0, out_axes=0) themselves and keep them global.
            - Note that logp - unnormalized log-posterior function that only takes in ONE SINGLE theta as argument and returns a scalar. 
            - Use partial if data is involved. Signature: logp : Callable[[Float[Array, "d"]], Float[Array, ""]], 
        '''
        # internalize these settings (use d instead of dim internally to correspond better with math).
        self.d, self.epsilon, self.beta, self.n_probes = dim, epsilon, beta, n_probes
        self.damp_factor, self.clip_val, self.atol, self.rtol = damp_factor, clip_val, atol, rtol

        '''
        Note that even though we store self.Q as a matrix, we won't use the identity case because it is all governed by self.use_Q.
        '''
        # basis_transformation was for user-readability. Switch for alignment with math.
        if (basis_transformation is not None) and (basis_transformation.shape != (self.d, self.d)):
            raise ValueError(f"basis_transformation must have shape {(self.d, self.d)}, if not None, but got {basis_transformation.shape}.")
        self.use_Q = (basis_transformation is not None) # a boolean flag.
        self.Q = jnp.eye(self.d) if basis_transformation is None else basis_transformation # (d x d) orthogonal matrix s.t. Q^T = Q^{-1}


    # public sample function for MULTIPLE CHAINS - completely determinstic (provided to users for convenience)
    def sample(self, 
               x0s : Float[Array, "b d"],
               xs_init : Float[Array, "b chain_length d"],
               keys_norm : UInt32[Array, "b chain_length 2"],
               keys_unif : UInt32[Array, "b chain_length 2"],
               keys_hutchinson : UInt32[Array, "b chain_length 2"],
               logp_and_grad_batched : Callable[[Float[Array, "n d"]], tuple[Float[Array, "n"], Float[Array, "n d"]]],
               chain_length : int,
               window_size : Optional[int] = None,
               full_trace : bool = False,
               max_iters : int = 10000) -> Union[BatchedSampleFinalType, BatchedSampleTraceType]:
        '''
        Args:
            x0s - initial conditions for our sampler, per chain.
            xs_init : initial guesses for all chain_length samples for each chain.
            keys_norm - PRNG keys for generating MVN(0, I_d) values for MALA proposals, batched per chain.
            keys_unif - PRNG keys for generating Uniform(0, 1) values for MALA accept/reject, batched per chain.
            keys_hutchinson - PRNG keys for generating d-dim Rademachers for Hutchinson estimators, batched per chain.
            chain_length - how many samples do we want per chain?
            window_size - maximum size of our sliding window for local Newton updates.
            full_trace - do we want intermediate Newton iteration outputs too (True) or just the final samples (False)?
            max_iters - maximum number of (quasi)-Gauss-Newton iterations to run GLOBALLY per chain (summed over all sliding window apps).
        
        Returns:
            (samples_final, iters_final) - our Parallel MALA samples + the no. of iterations needed, if full-trace=False, per chain.
            samples_trace - the full-trace of Parallel MALA samples, if full-trace=True, per chain.

        *Remember that JAX PRNG keys are always comprised of 2 unsigned integers!
        '''
        #### PRE-STAGING - converting to more math-friendly notation + sanity-checking dimensions. ####

        # if not using sliding window, do the full chain length.
        W = chain_length if window_size is None else min(window_size, chain_length) 
        if W <= 0:
            raise ValueError("window_size must be positive.")

        # how many chains "b" are we running?
        b, d = x0s.shape

        # check shapes of initial conditions, guesses, and drivers
        if d != self.d:
            raise ValueError(f"x0s must have shape {(b, self.d)}, got {x0s.shape}.")
        if xs_init.shape != (b, chain_length, self.d):
            raise ValueError(f"xs_init must have shape {(b, chain_length, self.d)}, got {xs_init.shape}.")
        if keys_norm.shape != (b, chain_length, 2):
            raise ValueError(f"keys_norm must have shape {(b, chain_length, 2)}, got {keys_norm.shape}.")
        if keys_unif.shape != (b, chain_length, 2):
            raise ValueError(f"keys_unif must have shape {(b, chain_length, 2)}, got {keys_unif.shape}.")
        if keys_hutchinson.shape != (b, chain_length, 2):
            raise ValueError(f"keys_hutchinson must have shape {(b, chain_length, 2)}, got {keys_hutchinson.shape}.")

        ### SOLVING FOR SAMPLES ####

        # call the _sample_multiple_chains_jit function for actually performant-code.
        return _sample_multiple_chains_jit(
            x0s=x0s, xs_init=xs_init, 
            keys_norm=keys_norm, keys_unif=keys_unif, keys_hutchinson=keys_hutchinson, 
            logp_and_grad_batched=logp_and_grad_batched,
            epsilon=self.epsilon, beta=self.beta, damp_factor=self.damp_factor, clip_val=self.clip_val, rtol=self.rtol, atol=self.atol,
            fhat_t_mcmc=fhat_t_mala, use_Q=self.use_Q, Q=self.Q, chain_length=chain_length, W=W, n_probes=self.n_probes, d=self.d, 
            transition_and_jacobian_estimator=hutchinson, full_trace=full_trace, max_iters=max_iters)