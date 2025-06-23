from __future__ import annotations

import time

import os
num_cores_to_use = 1
os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=true --xla_cpu_multi_thread_eigen_threads=%d"%(num_cores_to_use)

import jax.numpy as jnp
import jax.tree_util as jtu
from jax.lax import jcond, scan as jscan

from jaxtyping import Complex, Float
from typing import List


import numpy as np
from jax import jit, vmap, Array
from dynamiqs import floquet as dq_floquet, QArray, TimeQArray
from dynamiqs.method import Tsit5
from dynamiqs.utils import Options as DqOptions
from scipy.optimize import linear_sum_assignment

from .displaced_state import DisplacedState, DisplacedStateFit, _overlap
from .model import Model
from .options import Options
from .utils.file_io import Serializable
from .utils.parallel import parallel_map
from .utils.helpers import *


class FloquetAnalysis(Serializable):
    """Perform a floquet analysis to identify nonlinear resonances.

    In most workflows, one needs only to call the run() method which performs
    both the displaced state fit and the Blais branch analysis. For an example
    workflow, see the [transmon](../examples/transmon) tutorial.

    Arguments:
        model: Class specifying the model, including the Hamiltonian, drive amplitudes,
            frequencies
        state_indices: State indices of interest. Defaults to [0, 1], indicating the two
            lowest-energy states.
        options: Options for the Floquet analysis.
        init_data_to_save: Initial parameter metadata to save to file. Defaults to None.
    """

    def __init__(
        self,
        model: Model,
        state_indices: list | None = None,
        options: Options = Options(),  # noqa B008
        init_data_to_save: dict | None = None,
    ):
        if state_indices is None:
            state_indices = [0, 1]
        self.model = model
        self.state_indices = state_indices
        self.options = options
        self.init_data_to_save = init_data_to_save
        self.hilbert_dim = model.H0.shape[0]

    def __str__(self) -> str:
        return "Running floquet simulation with parameters: \n" + super().__str__()
    

    @jit
    def run_one_floquet(self, omega_d_amp: tuple[float, float]) -> tuple[QArray, QArray]:
        """Run one instance of the problem for a pair of drive frequency and amp.

        Returns Floquet modes and quasienergies.

        Parameters:
            omega_d_amp: Pair of drive frequency and amp.
        """
        omega_d, amp = omega_d_amp
        T = 2.0 * jnp.pi / omega_d
        tsave = jnp.linspace(0.0, T, 101)  # Example: 101 time points
        H = self.model.hamiltonian(omega_d, amp)
        result = dq_floquet(H, T, tsave, method=Tsit5(), options=DqOptions())
        return result.modes, jnp.real(result.quasienergies)
    
    
    # @jit
    # def identify_floquet_modes(
    #     self,
    #     f_modes_energies: tuple[QArray, QArray],
    #     params_0: tuple[float, float],
    #     displaced_state: DisplacedState,
    #     prev_coeffs: QArray,
    # ) -> QArray:
    #     """Return Floquet modes with largest overlap with ideal displaced state.
    #     Vectorized over all state indices.

    #     Parameters:
    #         f_modes_energies: Output of self.run_one_floquet(params)
    #         params_0: (omega_d_0, amp_0) to use for displaced fit
    #         displaced_state: Instance of DisplacedState
    #         prev_coeffs: Coefficients from the previous amplitude range
    #     """
    #     f_modes_0, _ = f_modes_energies

    #     disp_states = displaced_state.displaced_states(*params_0, prev_coeffs, 
    #                                                         state_indices=self.state_indices,
    #                                                         bare_same_override=True)

    #     # assign_states_batched = vmap(assign_states, in_axes=0)

    #     # could probably just rewrite this with argmax since you only care about 1 entry
    #     # MWPM probably takes way too long to do this. remember to test it
    #     # maxing_idxs = assign_states_batched(overlap_batched(disp_states, f_modes_0))[:, 0]
    #     overlap_mats = vmap(overlap, in_axes=(0, None))(disp_states, f_modes_0)
    #     maxing_idxs = jnp.argmax(overlap_mats, axis=-1)

    #     # this also does not guarantee that the list of indices is single-valued
    #     return f_modes_0[maxing_idxs]
    

    # @jit
    # def _mode_overlap(
    #     self, f_modes_energies: tuple[QArray, QArray], prev_f_modes: QArray
    # ) -> Array:

    #     f_modes_0, _ = f_modes_energies
        
    #     return assign_states(jnp.square(jnp.abs(f_modes_0.T)))

    # @jit
    # def _track_overlap(
    #     self, f_modes_energies: tuple[QArray, QArray], prev_f_modes: QArray
    # ) -> Array:
    #     """Perform Blais branch analysis.

    #     Gorgeous in its simplicity. Simply calculate overlaps of new floquet modes with
    #     those from the previous amplitude step, and order the modes accordingly. So
    #     ordered, compute the mean excitation number, yielding our branches.
    #     """
    #     f_modes_0, _ = f_modes_energies

    #     return assign_states(overlap(prev_f_modes, f_modes_0))
    
    @jit
    def _step_in_amp(
        self, f_modes_energies: tuple[QArray, QArray], prev_f_modes: QArray, 
    ) -> tuple[Array, Array, QArray]:
        
        f_modes_0, f_energies_0 = f_modes_energies
        operands = (f_modes_0, prev_f_modes)
        
        def _track_overlap(curr_modes, prev_modes):
            return assign_states(overlap(prev_modes, curr_modes))
        
        def _mode_overlap(curr_modes, prev_modes):
            return assign_states(jnp.square(jnp.abs(curr_modes.T)))
    
        max_idxs = jcond(self.options.track, _track_overlap, _mode_overlap, *operands)

        f_modes_ordered = f_modes_0[max_idxs]
        nbar = self._calculate_mean_excitation(f_modes_ordered)

        return nbar, f_energies_0[max_idxs], f_modes_ordered
    

    @jit
    def _calculate_mean_excitation(self, f_modes_ordered: Array) -> Array:
        """Mean excitation number of ordered floquet modes.

        Based on Blais arXiv:2402.06615, specifically Eq. (12) but going without the
        integral over floquet modes in one period.
        """
        # sum over bare excitations weighted by excitation number
        bare_overlaps_sq = jnp.square(jnp.abs(f_modes_ordered))
        nbar = jnp.einsum("ik,k->i", bare_overlaps_sq, jnp.arange(self.hilbert_dim))
        return jnp.real(nbar) 
    
    @jit 
    def _floquet_main_for_amp_range(
        self,
        omega_d: float,
        amp_idxs: list,
        displaced_state: DisplacedState,
        prev_coeffs: QArray,
        prev_f_modes: QArray,
    ) -> tuple:
        """Run the Floquet simulation over a specific amplitude range."""

        omega_d_idx = self.model.omega_d_to_idx(omega_d)
        chosen_amps = self.model.drive_amplitudes[omega_d_idx, amp_idxs[0] : amp_idxs[1]]
        

        def _run_floquet_and_calculate(omega_d: Float):
            omega_d_idx = self.model.omega_d_to_idx(omega_d)
            chosen_amps = self.model.drive_amplitudes[omega_d_idx, amp_idxs[0] : amp_idxs[1]]

            for idx, amp in enumerate(chosen_amps):
                params = (omega_d, amp)
                f_modes_energies = self.run_one_floquet(params)

                f_modes_ds = self.identify_floquet_modes(f_modes_energies, params, 
                                                         displaced_state, prev_coeffs)
                nbar, quasienergies, f_modes_ba = self._step_in_amp(f_modes_energies, 
                                                                    prev_f_modes)

                
    @jit
    def _postprocess(self, idx_pair: tuple[int, int], 
                     f_modes_energies_arr: QArray, prev_f_modes: QArray,
                     nbar_for_amp_range: Array, quasienergies_for_amp_range: Array, 
                     f_modes_for_amp_range: Array):
        
        omega_d_idx, amp_idx = idx_pair 
        write_idxs = (omega_d_idx, [amp_idx, amp_idx + 1])

        f_modes_energies = f_modes_energies_arr[omega_d_idx, amp_idx]
        nbar, quasienergies, f_modes = self._step_in_amp(f_modes_energies, prev_f_modes)

        nbar_for_amp_range = self._place_into(*write_idxs, nbar, nbar_for_amp_range)
        quasienergies_for_amp_range = self._place_into(*write_idxs, quasienergies, quasienergies_for_amp_range)
        f_modes_for_amp_range = self._place_into(*write_idxs, f_modes, f_modes_for_amp_range)

        return f_modes, None


        #     return vmap(self.run_one_floquet)((omega_d, amps_for_omega_d))

        # floquet_data = vmap(_run_floquet_and_calculate)(self.model.omega_d_values)
        # return floquet_data  # Process as needed
    

    def run(self, filepath: str | None = None) -> dict:
        """Perform floquet analysis over range of amplitudes and drive frequencies.

        This function largely performs two calculations. The first is the Xiao analysis
        introduced in https://arxiv.org/abs/2304.13656, fitting the extracted Floquet
        modes to the "ideal" displaced state which does not include resonances by design
        (because we fit to a low order polynomial and ignore any floquet modes with
        overlap with the bare state below a given threshold). This analysis produces the
        "scar" plots. The second is the Blais branch analysis, which tracks the Floquet
        modes by stepping in drive amplitude for a given drive frequency. For this
        reason the code is structured to parallelize over drive frequency, but scans in
        a loop over drive amplitude. This way the two calculations can be performed
        simultaneously.

        A nice bonus is that both of the above mentioned calculations determine
        essentially independently whether a resonance occurs. In the first, it is
        deviation of the Floquet mode from the fitted displaced state. In the second,
        it is branch swapping that indicates a resonance, independent of any fit. Thus
        the two simulations can be used for cross validation of one another.

        We perform these simulations iteratively over the drive amplitudes as specified
        by fit_range_fraction. This is to allow for simulations stretching to large
        drive amplitudes, where the overlap with the bare eigenstate would fall below
        the threshold (due to ac Stark shift) even in the absence of any resonances.
        We thus use the fit from the previous range of drive amplitudes as our new bare
        state.
        """
        print(self)
        start_time = time.time()


        run_full_floquet = vmap(vmap(self.run_one_floquet,
                                 in_axes=(None, -1), out_axes=-2), # vmap over amps
                                 in_axes=(-1, -2)  , out_axes=-3) # vmap over freqs
        
        full_floquet_results = run_full_floquet(self.model.omega_d_values, self.model.drive_amplitudes)
        
        freq_amp_shape = self.model.drive_amplitudes.shape
        omega_d_idxs = jnp.arange(freq_amp_shape[0])
        amp_idxs = jnp.arange(freq_amp_shape[1])
        omega_d_amp_idx_grid = jnp.stack(jnp.meshgrid(omega_d_idxs, amp_idxs), axis=-1)
        omega_d_amp_idx_grid = jnp.transpose(omega_d_amp_idx_grid, (1, 0, 2))

        f_modes_tot = jnp.zeros((*freq_amp_shape, self.hilbert_dim, self.hilbert_dim))
        quasienergies_tot =  jnp.zeros(f_modes_tot.shape[:-1])
        nbars_tot = quasienergies_tot.copy() 

        init_f_states = jnp.eye(len(self.hilbert_dim))

        state_processing = jtu.Partial(self._postprocess, 
                                       f_modes_energies_arr=full_floquet_results,
                                       nbar_for_amp_range=nbars_tot, 
                                       quasienergies_for_amp_range=quasienergies_tot,
                                       floquet_modes_for_amp_range=f_modes_tot)
        
        @jit 
        def scan_over_amps(param_idx_array: tuple[int, int]):
            return jscan(state_processing, init_f_states, xs=param_idx_array)
        
        vmap(scan_over_amps)(omega_d_amp_idx_grid)


        

        disp_state = DisplacedStateFit(hilbert_dim=self.hilbert_dim, model=self.model,
                                state_indices=self.state_indices, options=self.options)

        disp_state.displaced_states_fit(self.model.omega_d_values, self.model.drive_amplitudes)

        # bare_overlaps_tot = jnp.zeros(*freq_amp_shape, self.state_indices)
        # intermed_disp_overlaps = bare_overlaps_tot.copy()

        # for all omega_d, the bare states are identical at zero drive. We define
        # two sets of bare modes (prev_f_modes_arr and disp_coeffs_for_prev_amp)
        # because for the fit calculation, the bare modes are specified as fit
        # coefficients, whereas for the Blais calculation, the bare modes are specified
        # as actual kets.


        ####################################################################################################################################################################################
        
        prev_f_modes = jnp.tile(jnp.eye(len(self.state_indices)), (freq_amp_shape[-2], 1, 1))

        disp_state = DisplacedStateFit(
            hilbert_dim=self.hilbert_dim,
            model=self.model,
            state_indices=self.state_indices,
            options=self.options,
        )

        prev_coeffs = disp_state._bare_coeffs(*freq_amp_shape)

        num_fit_ranges = int(jnp.ceil(1 / self.options.fit_range_fraction))
        num_amp_pts_per_range = int(jnp.floor(freq_amp_shape[-1] / num_fit_ranges))
        all_indices = jnp.arange(freq_amp_shape[-1])
        
        amp_idx_set = [all_indices[i*num_amp_pts_per_range: jnp.min((i+1) * num_amp_pts_per_range, 
                                            freq_amp_shape[-1])] for i in range(num_fit_ranges)]
        
        for amp_idxs in amp_idx_set:
            print(f"calculating for amp_range_idx={amp_range_idx}")
            result = self._floquet_main_for_amp_range(amp_idxs, disp_state, prev_coeffs, prev_f_modes)
            bare_overlaps, f_modes, nbars, quasienergies, prev_f_modes = result

            bare_overlaps_tot = self._place_into(amp_idxs, bare_overlaps, bare_overlaps_tot)
            f_modes_tot = self._place_into(amp_idxs, f_modes, f_modes_tot)
            nbars_tot = self._place_into(amp_idxs, nbars, nbars_tot)
            quasienergies_tot = self._place_into(amp_idxs, quasienergies, quasienergies_tot)



        for amp_range_idx in range(num_fit_ranges):
            print(f"calculating for amp_range_idx={amp_range_idx}")
            # edge case if range doesn't fit in neatly
            if amp_range_idx == num_fit_ranges - 1:
                amp_range_idx_final = len(self.model.drive_amplitudes)
            else:
                amp_range_idx_final = (amp_range_idx + 1) * num_amp_pts_per_range
            amp_idxs = [amp_range_idx * num_amp_pts_per_range, amp_range_idx_final]
            # now perform floquet mode calculation for amp_range_idx
            # need to pass forward the floquet modes from the previous amp range
            # which allow us to identify floquet modes that may have been displaced
            # far from the origin
            output = self._floquet_main_for_amp_range(
                amp_idxs, displaced_state, prev_coeffs, prev_f_modes_arr 
            )
            (
                bare_state_overlaps_for_range,
                floquet_modes_for_range,
                avg_excitation_for_range,
                quasienergies_for_range,
                prev_f_modes_arr,
            ) = output
            bare_state_overlaps = self._place_into(
                amp_idxs, bare_state_overlaps_for_range, bare_state_overlaps
            )
            floquet_modes = self._place_into(
                amp_idxs, floquet_modes_for_range, floquet_modes
            )
            avg_excitation = self._place_into(
                amp_idxs, avg_excitation_for_range, avg_excitation
            )
            quasienergies = self._place_into(
                amp_idxs, quasienergies_for_range, quasienergies
            )

            # ovlp_with_bare_states is used as a mask for the fit
            ovlp_with_bare_states = displaced_state.overlap_with_bare_states(
                amp_idxs[0], prev_coeffs, floquet_modes_for_range
            )
            omega_d_amp_slice = list(self.model.omega_d_amp_params(amp_idxs))
            # Compute the fitted 'ideal' displaced state, excluding those
            # floquet modes experiencing resonances.
            new_coefficients = displaced_state.displaced_states_fit(
                omega_d_amp_slice, ovlp_with_bare_states, floquet_modes_for_range
            )
            # Compute overlap of floquet modes with ideal displaced state using this
            # new fit. We use this data as the mask for when we compute the coefficients
            # over the whole range. Note that we pass in floquet_modes as
            # opposed to the more restricted floquet_modes_for_range since we
            # use indexing methods inside of overlap_with_displaced_states, so its
            # easier to pass in the whole array.
            overlaps = displaced_state.overlap_with_displaced_states(
                amp_idxs, new_coefficients, floquet_modes
            )
            intermediate_displaced_state_overlaps = self._place_into(
                amp_idxs, overlaps, intermediate_displaced_state_overlaps
            )
            prev_coeffs = new_coefficients
        # The previously extracted coefficients were valid for the amplitude ranges
        # we asked for the fit over. Now armed with with correctly identified floquet
        # modes, we recompute these coefficients over the whole sea of floquet mode data
        # to get a plot that is free from numerical artifacts associated with
        # the fits being slightly different at the boundary of ranges. We utilize the
        # previously computed overlaps of the floquet modes with the displaced states
        # (stored in intermediate_displaced_state_overlaps) to obtain the mask with
        # which we exclude some data from the fit (because we suspect they've hit
        # resonances).
        amp_idxs = [0, len(self.model.drive_amplitudes)]
        omega_d_amp_slice = list(self.model.omega_d_amp_params(amp_idxs))
        full_displaced_fit = displaced_state.displaced_states_fit(
            omega_d_amp_slice, intermediate_displaced_state_overlaps, floquet_modes
        )
        true_overlaps = displaced_state.overlap_with_displaced_states(
            amp_idxs, full_displaced_fit, floquet_modes
        )
        data_dict = {
            "bare_state_overlaps": bare_state_overlaps,
            "fit_data": full_displaced_fit,
            "displaced_state_overlaps": true_overlaps,
            "intermediate_displaced_state_overlaps": intermediate_displaced_state_overlaps,  # noqa E501
            "quasienergies": quasienergies,
            "avg_excitation": avg_excitation,
        }
        if self.options.save_floquet_modes:
            data_dict["floquet_modes"] = floquet_modes
        print(f"finished in {(time.time() - start_time) / 60} minutes")
        if filepath is not None:
            self.write_to_file(filepath, data_dict)
        return data_dict

    @staticmethod
    @jit 
    def _place_into(
        omega_idx: int, amp_idxs: list, array_for_range: Array, overall_array: Array
    ) -> Array:
        overall_array = overall_array.at[omega_idx, amp_idxs[0] : amp_idxs[1]].set(array_for_range)
        return overall_array

    