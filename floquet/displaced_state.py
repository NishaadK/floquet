from __future__ import annotations

from jax import Array, vmap, jit
import jax.numpy as jnp
import jax.tree_util as jtu
import jax.lax.cond as jcond

import dynamiqs as dq
from dynamiqs import QArray

from optimistix import LevenbergMarquardt, least_squares, RESULTS

from jaxtyping import Int, Float, Bool, Complex
from typing import List

from .model import Model
from .options import Options

from warnings import warn

class DisplacedState:
    """Class providing methods for computing displaced states.

    Parameters:
        hilbert_dim: Hilbert space dimension
        model: Model including the Hamiltonian, drive amplitudes, frequencies, state indices
        state_indices: States of interest
        options: Options used
    """
    hilbert_dim: int
    state_indices: List[int]
    exponent_pair: Int[Array, "2 num_fit_terms"]
    fit_cutoff : int
    fit_cutoff_omega_d : int
    fit_cutoff_amp : int
    overlap_cutoff : float

    def __init__(
        self, model: Model, state_indices: List[int], options: Options
    ):
        self.hilbert_dim = model.hilbert_dim
        self.state_indices = state_indices

        # Save drive parameters?
        #self.omega_d_values = model.omega_d_values
        #self.drive_amplitudes = model.drive_amplitudes

        self.fit_cutoff = self.options.fit_cutoff
        self.fit_cutoff_omega_d = min(model.omega_d_values.shape[-1], self.options.fit_cutoff)
        self.fit_cutoff_amp = min(model.drive_amplitudes.shape[-1], self.options.fit_cutoff)
        self.overlap_cutoff =  self.options.overlap_cutoff

        self.exponent_pair = self._create_exponent_pair()

    def displaced_states(
        self, 
        omega_ds: Float[Array, "num_omega_ds"],
        amps: Float[Array, "num_omega_ds num_amps"],
        coefficients: Complex[Array, "num_state_indices hilbert_dim num_omega_ds num_amps num_fit_terms"],
        *,
        state_indices: List[int] = None, 
        bare_same_override: Bool = False,
    ) -> Complex[QArray, "num_state_indices num_omega_ds num_amps hilbert_dim 1"]:
        """Construct approximate displaced states, $\left| \tilde{i}(\omega_d, \xi) \right>$.
        $$
        \left| \tilde{i}(\omega_d, \xi) \right> = \sum_{j} \left| j \right> \left< j \middle| \tilde{i}(\omega_d, \xi) \right>
        $$
        where $\left| j \right>$ are the undriven (bare) states and the coefficients \left< j \middle| \tilde{i}(\omega_d, \xi) \right> are given by a polynomial fit. The polynomial is of the form:
        $$
            \left< j \middle| \tilde{i}(\omega_d, \xi) \right> = \sum_{\substack{k \ge 0, l>0 \\ k+l \leq \text{cutoff}}} C_{i,j,k,l}(\omega_d, \xi) \omega_d^k \xi^l
        $$
        where $C_{i,j,k,l}(\omega_d, \xi)$ are the coefficients to be fitted.

        Note: Careful with the shape of the coefficients vs. that of the output.

        Parameters:
            omega_ds: Drive frequency. Shape: (num_omega_ds,)
            amps: Drive amplitude. Shape: (num_omega_ds, num_amps)
            coefficients: Coefficients to expand the displaced state in terms of the undriven states. Shape: (num_state_indices, hilbert_dim, num_omega_ds, num_amps, num_fit_terms).
            state_indices: States of interest. 
            bare_same_override: Override flag for bare_same. Equivalent to excluding a constant term in the fit or not. Keyword-only argument. 

        Returns:
            The displaced state(s). Shape: (num_state_indices, num_omega_ds, num_amps, hilbert_dim, 1). Careful, that the first index is the array index, not the state index! For example, if state_indices = [0, 2], then result[0, ...] is the displaced state for state index 0, and result[1, ...] is the displaced state for state index 2.
        """
        if state_indices is None:
            state_indices = self.state_indices

        _vmap_displaced_state = vmap(DisplacedState._one_displaced_state, 
                                    in_axes=(None, None, -5, -1, None, None), out_axes=0)
        return _vmap_displaced_state(omega_ds, amps, coefficients, state_indices, 
                                     self.exponent_pair, bare_same_override)

    @staticmethod
    @jit
    def _one_displaced_state( 
        omega_ds: Float[Array, "num_omega_ds"],
        amps: Float[Array, "num_omega_ds num_amps"],
        coefficients: Complex[Array, "hilbert_dim num_omega_ds num_amps num_fit_terms"],
        state_idx: int, 
        exponent_pair: Int[Array, "2 num_fit_terms"],
        bare_same_override: Bool = False,
    ) -> Complex[QArray, "num_omega_ds num_amps hilbert_dim 1"]:

        hilbert_dim = coefficients.shape[-4]

        # Set vec_bare_same based on bare_same_override
        # If bare_same_override is false, return a vector of all False
        # If it is true, then return a boolean vector with True only when idx == state_idx
        vec_bare_same = jcond(
            bare_same_override,
            lambda: jnp.zeros(hilbert_dim, dtype=bool),
            lambda: (dq.fock(hilbert_dim, state_idx).to_jax() == 1)[..., 0]
        )

        # Simply vectorize _compute_polynomial the basis vectors! Isn't that so cool? :)
        _vmap_compute_polynomial = vmap(DisplacedState._compute_polynomial,
                                       in_axes=(None, None, -4, None, -1), out_axes=-1)

        # Vectorized function returns shape: (hilbert_dim, num_omega_ds, num_amps, hilbert_dim). 
        result = _vmap_compute_polynomial(omega_ds, amps, coefficients, exponent_pair, 
                                          vec_bare_same)

        # Reshape to (num_state_indices, hilbert_dim, num_omega_ds, num_amps, hilbert_dim, 1).
        return dq.as_qarray(result[..., None])

    @staticmethod
    @jit
    def _compute_polynomial(
        omega_ds: Float[Array, "num_omega_ds"],
        amps: Float[Array, "num_omega_ds num_amps"],
        coefficients: Complex[Array, "num_omega_ds num_amps num_fit_terms"],
        exponent_pair: Int[Array, "2 num_fit_terms"],
        bare_same: Bool,
    ) -> Complex[Array, "num_omega_ds num_amps"]:
        # Helper function to compute one term of the polynomial
        def _poly_term(omega_d: Float, amp: Float, exponent_pair: Float) -> Float:
            return (omega_d ** exponent_pair[0]) * (amp ** exponent_pair[1])

        # Cartesian vmap (vmap in reverse order)
        _cvmap_poly_term = vmap(vmap(vmap(_poly_term,
               in_axes=(None,None,-1), out_axes=0), # vmap over exponents
               in_axes=(None,-1,None), out_axes=0), # vmap over amp  
               in_axes=(-1,-2,None),   out_axes=0)  # vmap over omega_d

        # Get all the terms i.e. omega_d^k * amp^l, shape: (num_omega_ds, num_amps, num_fit_terms)
        all_ploy_terms = _cvmap_poly_term(omega_ds, amps, exponent_pair)

        # NOTE: coefficients.shape=(num_omega_ds, num_amps, num_fit_terms).
        # This isn't explicitly validated. 
        result = jnp.nansum(coefficients * all_ploy_terms, axis = -1)
        return jcond(bare_same, lambda: 1.0 + result, lambda: result)


    def _create_exponent_pair(self) -> dict:
        """Create the pair of exponents for the polynomial to be fitted. The polynomial is of the form:
        $$
            \sum_{\substack{k \ge 0, l>0 \\ k+l \leq \text{cutoff}}} C_{k,l} \omega_d^k \xi^l
        $$
        where $C_{k,l}$ are the coefficients to be fitted. This function creates the (j,l) pairs

        We truncate the fit if e.g. there is only a single frequency value to scan over but the fit is nominally set to order four. We additionally eliminate the constant term that should always be either zero or one.

        Returns:
            exponent_pair: A 2D array of shape (2,num_fit_terms) where the first column
                contains the exponents for omega_d and the second column contains the exponents
                for amplitude. The number of terms is determined by the cutoff value.
        """

        # Generate all combinations of indices. 
        # Remove amplitude-independent terms (i.e. exponent for the amp, l == 0).
        # This is enforced by the fact the states have to agree at zero drive strength.
        idx_exp_map = jnp.stack(jnp.meshgrid(
                                    jnp.arange(self.fit_cutoff_omega_d), 
                                    jnp.arange(1, self.fit_cutoff_amp), 
                                    indexing='ij'), 
                                axis=-1).reshape(-1, 2)
        
        # Sort. Introduce a fudge factor to ensure that the sorting is stable
        # Only keep terms where the sum of the exponents is less than the cutoff
        weighted_vals = 1.01 * idx_exp_map[:, 0] + idx_exp_map[:, 1]
        sorted_idxs = jnp.argsort(weighted_vals[weighted_vals <= 1.01*self.fit_cutoff])
        return idx_exp_map[sorted_idxs].T # shape = (2,num_fit_terms)


class DisplacedStateFit(DisplacedState):
    """Methods for fitting an ideal displaced state to calculated Floquet modes."""

    def displaced_states_fit(
        self,
        omega_ds: Float[Array, 'num_omega_ds'],
        amps: Float[Array, 'num_omega_ds num_amps'],
        floquet_modes: Complex[QArray, "num_state_indices num_omega_ds num_amps hilbert_dim 1"],
    ) -> Complex[Array, "num_state_indices hilbert_dim num_omega_ds num_amps num_fit_terms"]:
        """Fit the coefficients for the displaced states corresponding to self.state_indices. 
        
        We ignore the floquet modes where we suspect a transition. These are the points where the overlap of the floquet mode with the bare states falls below the threshold (specified in options).

        Parameters:
            omega_ds: Drive frequency. Shape: (num_omega_ds,)
            amps: Drive amplitude. Shape: (num_omega_ds, num_amps)
            floquet_modes: Floquet modes. Shape: (num_state_indices num_omega_ds num_amps hilbert_dim 1)

        Returns:
            Optimized fit coefficients. Shape: (num_state_indices, hilbert_dim, num_omega_ds, num_amps, num_fit_terms).
        """
        num_fit_terms = self.exponent_pair.shape[-1]
        zero_coeffs = jnp.zeros((*floquet_modes.shape[:-4],  # num_state_indices(+ any leading dims)
                                 floquet_modes.shape[-2],    # hilbert_dim
                                 floquet_modes.shape[-4],    # num_omega_ds
                                 floquet_modes.shape[-3],    # num_amps
                                 num_fit_terms,              # num_fit_terms
                               ), dtype=complex)

        # Do we have enough points to fit? 
        # This isn't JAX-friendly. However, ideally this function is run just once. 
        if (omega_ds.shape[-1] < num_fit_terms) or (amps.shape[-2] < num_fit_terms):
            warn("Not enough data points to fit. Returning zeros for the fit",stacklevel=3)
            return zero_coeffs
        
        # Compute overlap with bare states. These are simply displaced states with zero coeffs. 
        # The constant = 1 is taken care of by bare_same
        # overlap_with_bare_states.shape = (num_state_indices, num_omega_ds, num_amps)
        bare_states = self.displaced_states(omega_ds, amps, zero_coeffs, self.state_indices)
        overlap_with_bare_states = dq.overlap(bare_states, floquet_modes).to_jax()

        # Only fit states that we think haven't run into a transition
        # mask.shape =(..., num_state_indices, num_omega_ds, num_amps). 
        mask = (overlap_with_bare_states > self.overlap_cutoff)

        # Tile omega_ds and amps to match the shape of mask
        tiled_omega_ds = jnp.tile(omega_ds[None, :, None], (mask.shape[-3], 1, mask.shape[-1]))
        tiled_amps = jnp.tile(amps[None, :, :], (mask.shape[-3], 1, 1))

        # Apply the mask
        omega_ds_masked = jnp.where(mask, tiled_omega_ds, jnp.nan)
        amps_masked = jnp.where(mask, tiled_amps, jnp.nan)
        floquet_modes_masked = jnp.where(mask[..., None, None], floquet_modes, jnp.nan)

        # Vectorize the over state_indices
        vmap_fit_states_factory = vmap(DisplacedStateFit._fit_states_factory,
                                      in_axes=(-3,-3,-5,-5,-1,None,None), out_axes=0)

        # Fit the real and imaginary parts of the overlap separately
        coeffs_r = DisplacedStateFit._fit_states_factory(
                                            omega_ds=omega_ds_masked,
                                            amps=amps_masked, 
                                            target_states=jnp.real(floquet_modes_masked),
                                            init_coefficients=zero_coeffs, 
                                            state_index=self.state_indices,
                                            exponent_pair=self.exponent_pair, 
                                            bare_same_override=False,
                                            )

        # For the imaginary part, constant term should always be zero 
        coeffs_i = DisplacedStateFit._fit_states_factory(
                                            omega_ds=omega_ds_masked,
                                            amps=amps_masked, 
                                            target_states=jnp.imag(floquet_modes_masked),
                                            init_coefficients=zero_coeffs, 
                                            state_index=self.state_indices,
                                            exponent_pair=self.exponent_pair, 
                                            bare_same_override=True,
                                            )

        return coeffs_r + 1j*coeffs_i

    @staticmethod
    @jit
    def _fit_states_factory(
        omega_ds: Float[Array, "num_omega_ds num_amps"],
        amps: Float[Array, "num_omega_ds num_amps"],
        target_states: Float[QArray, "num_omega_ds num_amps hilbert_dim 1"],
        init_coefficients: Float[Array, "hilbert_dim num_omega_ds num_amps num_fit_terms"],
        state_index: int,
        exponent_pair: Int[Array, "2 num_fit_terms"],
        bare_same_override: Bool = False
    ) -> Float[Array, "hilbert_dim num_omega_ds num_amps num_fit_terms"]:

        partial_displaced_state = jtu.Partial(
            DisplacedState._one_displaced_state,
            omega_ds=omega_ds,
            amps=amps,
            state_idx=state_index,
            exponent_pair=exponent_pair,
            bare_same_override=bare_same_override,
        )

        # Note: nansum takes care of the masked values
        def residuals(coeffs: Array) -> Array:
            predicted = partial_displaced_states(coefficients=coeffs)
            return jnp.nansum(jnp.abs((predicted - target_states).to_jax()) ** 2)

        # Perform optimization using Levenberg-Marquardt
        solver = LevenbergMarquardt(rtol=1e-8, atol=1e-8)
        solution = least_squares(residuals, solver=solver, y0=init_coefficients, throw = False)

        # Return the fitted result if successful. Otherwise, return zeros 
        # Note, warnings are not jittable (see: https://github.com/dynamiqs/dynamiqs/pull/925)
        return jcond(
            solution.result == RESULTS.successful,
            lambda: solution.value,
            lambda: jnp.zeros_like(init_coefficients),
        )