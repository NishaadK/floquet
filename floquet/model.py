from __future__ import annotations

from jax import vmap
import jax.numpy as jnp
import dynamiqs as dq

from jaxtyping import ArrayLike, Array, Complex, Float
from dynamiqs import QArray, TimeQArray

from .utils.file_io import Serializable


class Model(Serializable):
    """Specify the model, including the Hamiltonian, drive strengths and frequencies.

    Can be subclassed to e.g. override the hamiltonian() method for a different (but
    still periodic!) Hamiltonian.

    Parameters:
        H0: Drift Hamiltonian, which must be diagonal and in units of the drive amplitude. Shape = (..., hilbert_dim, hilbert_dim)
        H1: Drive operator, which should be unitless (for instance the charge-number operator n of the transmon). It will be multiplied by a drive amplitude that we scan over from drive_parameters.drive_amplitudes. Shape = (hilbert_dim, hilbert_dim)
        omega_d_values: drive frequencies to scan over
        drive_amplitudes: amp values to scan over. Can be one dimensional in which case
            these amplitudes are used for all omega_d, or it can be two dimensional
            in which case the first dimension indexes the drive frequency and the second indexes the amplitude. 

    # Advanced

    ## Batching simulations
    If you wish to vary a Hamiltonian parameter, such as the gate-charge or flux-bias, you can batch the Hamiltonians and simulate the Floquet modes for all Hamiltonians concurrently. Importantly, we currently only support batching over parameters within static Hamiltonian. This is because dynamiqs does not allow modulated TimeQArrays to be batched.
    """
    H0 : Complex[QArray, "... hilbert_dim hilbert_dim"]
    H1 : Complex[QArray, "... hilbert_dim hilbert_dim"]
    omega_d_values : Float[Array, "num_amps num_omega_ds"]
    drive_amplitudes : Float[Array, "num_amps"]
    hilbert_dim : Float

    def __init__(
        self,
        H0: QArray | Array,
        H1: QArray | Array,
        omega_d_values: Array | list,
        drive_amplitudes: Array | list,
    ):
        if not isinstance(H0, QArray):
            H0 = dq.asqarray(jnp.array(H0, dtype=complex))
        if not isinstance(H1, QArray):
            H1 = dq.asqarray(jnp.array(H1, dtype=complex))

        if isinstance(omega_d_values, list):
            omega_d_values = jnp.array(omega_d_values)
        if isinstance(drive_amplitudes, list):
            drive_amplitudes = jnp.array(drive_amplitudes)
        
        # Validate that the omega_d_values is a 1D array
        if jnp.sum(jnp.array(omega_d_values.shape) > 1) != 1:
            raise ValueError(f'omega_d_values has invalid shape {omega_d_values.shape}. Please provide a 1D array.')
        omega_d_values = jnp.squeeze(omega_d_values)

        # Amplitudes are always stored as a 2D array. Shape = (num_omega_ds, num_amps)
        if len(drive_amplitudes.shape) == 1:
            drive_amplitudes = jnp.tile(drive_amplitudes, (len(omega_d_values), 1))
        assert len(drive_amplitudes.shape) == 2
        assert drive_amplitudes.shape[0] == len(omega_d_values)
       
        self.H0 = H0
        self.H1 = H1
        self.hilbert_dim = H0.shape[-1]
        self.omega_d_values = omega_d_values
        self.drive_amplitudes = drive_amplitudes

    def omega_d_to_idx(self, omega_d: float,
        ) -> Float[Array, "num_omega_ds"]:
        """Return index corresponding to omega_d value."""
        return jnp.argmin(jnp.abs(self.omega_d_values - omega_d))

    def amp_to_idx(self, amp: float, omega_d: float,
        ) -> Float[Array, "num_omega_ds num_amps"]:
        """Return index corresponding to amplitude value.

        Because the drive amplitude can depend on the drive frequency, we also must pass
        the drive frequency here.
        """
        omega_d_idx = self.omega_d_to_idx(omega_d)
        return jnp.argmin(jnp.abs(self.drive_amplitudes[:, omega_d_idx] - amp))

    def hamiltonian(
        self, 
        omega_d : float, 
        amp : float,
        ) -> Complex[TimeQArray, "hilbert_dim hilbert_dim"]:
        """Return the Hamiltonian we actually simulate."""
        return dq.constant(self.H0) + dq.modulated(lambda t: jnp.cos(omega_d * t), amp * self.H1)

    def vectorized_hamiltonian(
        self, 
        omega_ds : Float[Array, "num_omega_ds"] = None,
        amps : Float[Array, "num_omega_ds num_amps"] = None,
        ) -> Complex[TimeQArray, "num_omega_ds num_amps hilbert_dim hilbert_dim"]:
        """Return the Hamiltonian, Cartesian vectorization over omega_d and amplitude. The method correctly maps over the 2D drive_amplitudes array, accounting for the freq. dependance. 

        Returns:
            vectorized_hamiltonian: Shape: (..., num_omega_ds, num_amps, hilbert_dim, hilbert_dim)
        """
        if omega_ds is None:
            omega_ds = self.omega_d_values
        if amps is None:
            amps = self.drive_amplitudes

        # vmap over omega_d and corresponding amplitudes
        _vmap_hamiltonian = vmap(vmap(self.hamiltonian,
                                 in_axes=(None, -1), out_axes=-3), # vmap over amps
                                 in_axes=(-1, -2)  , out_axes=-4)  # vmap over omega_ds
        return _vmap_hamiltonian(omega_ds, amps)