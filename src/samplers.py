import jax
jax.config.update("jax_default_matmul_precision", "highest")
import jax.numpy as jnp
import jax.random as jr

from functools import partial

# core DEER + QDEER algorithms
import src
from src import deer, qdeer, windowed_qdeer, elk

from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions

def sigmoid_accept(x):
    zero = jax.nn.sigmoid(x) - jax.lax.stop_gradient(jax.nn.sigmoid(x)) # zero on fwd pass.
    return zero + jax.lax.stop_gradient((x > 0)) 

class ParallelMALA:
    
    # our constructor
    def __init__(self, log_prob, dim, chain_length, max_iter, alg="quasi",
                 clip_val=1.0, damp_factor=1.0, full_trace=False, basis_transformation=False):
        '''
        Args:
            logp - unnormalized log-posterior function that ONLY takes in theta as argument. Use partial.
            dim - how many dimensions is our parameter space for sampling? (added together)
            epsilon - the MALA stepsize.
            quasi - are we taking diagonal or full Jacobian?
            qmem_efficient - are we using the Hutchinson's estimator?
            clip_val - what are we clipping individual gradient entries to in absolute value?
            damp_factor - slightly damping the Jacobian.
        '''
        # 1. internalize + get the target log-prob and grad
        self.log_prob = log_prob
        self.D = dim
        self.chain_length = chain_length
        self.target_log_prob_and_grad = jax.value_and_grad(self.log_prob)
        self.max_iter = max_iter
        self.alg = alg 
        self.clip_val = clip_val
        self.damp_factor = damp_factor
        self.full_trace = full_trace 
        self.basis_transformation = basis_transformation
        
    def mala_fxn_for_seq(self, state, driver, params):
        step_size = params["step_size"]
        key, *skeys = jr.split(driver, 3)

        logprob_state, grad_state = self.target_log_prob_and_grad(state, params["target_params"])
        next_state = state + step_size * grad_state # grad_logp is previously defined score function (todo: have it take in params)
        next_state = next_state + jnp.sqrt(2.0 * step_size) * jr.normal(skeys[0], (state.shape[0],))

        # get new log prob and grad
        logprob_nextstate, grad_nextstate = self.target_log_prob_and_grad(next_state, params["target_params"])

        # accept / reject
        num = logprob_nextstate + tfd.MultivariateNormalDiag(
            loc=next_state+step_size * grad_nextstate,
            scale_diag=jnp.sqrt(2.0 * step_size) * jnp.ones_like(state)).log_prob(
                state)
        den = logprob_state + tfd.MultivariateNormalDiag(
            loc=state+step_size * grad_state,
            scale_diag=jnp.sqrt(2.0 * step_size) * jnp.ones_like(state)).log_prob(
                next_state)
        g = sigmoid_accept(num-den-jnp.log(jr.uniform(skeys[1])))
        next_state = g*next_state + (1.0-g)*state

        return next_state

    def mala_fxn_for_deer(self, state, driver, params):
        # if transform basis, do initial transformation (assume orthogonal)
        if self.basis_transformation:
            state = params["basis"] @ state 

        # then run seq update
        next_state = self.mala_fxn_for_seq(state, driver, params)

        # transform reverse
        if self.basis_transformation:
            next_state = params["basis"].T @ next_state

        return next_state

    def run_parallel_mala(self, key, initial_state, yinit_guess, params):
        drivers = jr.split(key, (self.chain_length,))

        if self.basis_transformation:
            initial_state = params["basis"].T @ initial_state 
            yinit_guess = jnp.einsum('...ij, ...j -> ...i', params["basis"].T, yinit_guess)

        out_states, iters = qdeer.seq1d(
            self.mala_fxn_for_deer, initial_state, drivers, params, 
            yinit_guess=yinit_guess, max_iter=self.max_iter, clip_val=self.clip_val,
            full_trace=self.full_trace, damp_factor=self.damp_factor
        )

        if self.basis_transformation:
            out_states = jnp.einsum('...ij, ...j -> ...i', params["basis"], out_states)

        return out_states, iters 

    def run_sequential_mala(self, key, initial_state, params):

        def _fxn_for_scan(state, driver):
            state = self.mala_fxn_for_seq(state, driver, params)
            return state, state 

        drivers = jr.split(key, (self.chain_length,))
        _, out_states = jax.lax.scan(_fxn_for_scan, initial_state, drivers)

        return out_states


class ParallelHMC:
    # our constructor
    def __init__(self, log_prob, dim, chain_length, max_iter, alg="quasi",
                 clip_val=1.0, damp_factor=1.0, full_trace=False, basis_transformation=False):
        '''
        Args:
            logp - unnormalized log-posterior function that ONLY takes in theta as argument. Use partial.
            dim - how many dimensions is our parameter space for sampling? (added together)
            epsilon - the MALA stepsize.
            quasi - are we taking diagonal or full Jacobian?
            qmem_efficient - are we using the Hutchinson's estimator?
            clip_val - what are we clipping individual gradient entries to in absolute value?
            damp_factor - slightly damping the Jacobian.
        '''
        # 1. internalize + get the target log-prob and grad
        self.log_prob = log_prob
        self.D = dim
        self.chain_length = chain_length
        self.target_log_prob_and_grad = jax.value_and_grad(self.log_prob)
        self.max_iter = max_iter
        self.alg = alg 
        self.clip_val = clip_val
        self.damp_factor = damp_factor
        self.full_trace = full_trace 
        self.basis_transformation = basis_transformation

    def scan_leapfrog(self, state, step_size):
        # Assumes you start and end 
        # with half-step corrections to momentum
        # Add half step before running iteration
        # Subtract half step after running iteration
        z, m = jnp.split(state, 2)
        z += step_size * m
        _, tlp_grad = self.target_log_prob_and_grad(z)
        m += step_size * tlp_grad
        next_state = jnp.concatenate((z, m))
        return next_state

    def hmc_fxn_for_deer(self, state, driver, params):
        seed = driver
        z = state 
        step_size = params['epsilon'] 
        m_seed, mh_seed = jax.random.split(seed)
        tlp, tlp_grad = self.target_log_prob_and_grad(z)
        m = jax.random.normal(m_seed, z.shape)
        energy = 0.5 * jnp.square(m).sum() - tlp
        # start with half-step of momentum
        m += 0.5 * step_size * tlp_grad
        init_state = jnp.concatenate((z, m))
        new_state = jax.lax.fori_loop(0, params['num_leapfrog_steps'],
            lambda i, state : self.scan_leapfrog(state, step_size), 
            init_state)
        new_z, new_m = jnp.split(new_state, 2)
        new_tlp, new_tlp_grad = self.target_log_prob_and_grad(new_z)
        # end with backward half-step of momentum
        new_m -= 0.5 * step_size * new_tlp_grad 
        new_energy = 0.5 * jnp.square(new_m).sum() - new_tlp
        log_accept_ratio = energy - new_energy

        # accept-reject
        u = jax.random.uniform(mh_seed, [])
        g = sigmoid_accept(log_accept_ratio-jnp.log(u))
        z = g*new_z + (1.0-g)*z
        return z

    def run_sequential_hmc(self, key, initial_state, params):

        def _fxn_for_scan(state, driver):
            state = self.hmc_fxn_for_deer(state, driver, params)
            return state, state 

        drivers = jr.split(key, (self.chain_length,))
        _, out_states = jax.lax.scan(_fxn_for_scan, initial_state, drivers)

        return out_states

    def run_parallel_hmc(self, key, initial_state, yinit_guess, params):
        drivers = jr.split(key, (self.chain_length,))

        # if self.basis_transformation:
        #     initial_state = params["basis"].T @ initial_state 
        #     yinit_guess = jnp.einsum('...ij, ...j -> ...i', params["basis"].T, yinit_guess)

        out_states, iters = deer.seq1d(
            self.hmc_fxn_for_deer, initial_state, drivers, params, 
            yinit_guess=yinit_guess, max_iter=self.max_iter, 
            quasi=False, qmem_efficient=False, clip_val=self.clip_val,
            full_trace=self.full_trace, damp_factor=self.damp_factor
        )

        # if self.basis_transformation:
        #     out_states = jnp.einsum('...ij, ...j -> ...i', params["basis"], out_states)

        return out_states, iters 