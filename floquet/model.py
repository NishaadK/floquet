from __future__ import annotations

import itertools
import json
from abc import ABC, abstractmethod
from itertools import chain, product
from typing import Any

import numpy as np
import qutip as qt

from .utils.file_io import Serializable

PI = np.pi
TWO_PI = 2 * PI

def _as_str(value: Any) -> str:
    """Decode a value that may come back from h5 as bytes/np.bytes_ into str."""
    if isinstance(value, bytes):
        return value.decode()
    return str(value)


def _to_jsonable(obj: Any) -> Any:
    """Recursively convert numpy/containers into JSON-serializable Python types.

    Used so an arbitrary device ``to_dict()`` (which may have non-string keys,
    numpy scalars, ``inf``/``None``) can be stored as a single h5-safe string.
    """
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _to_jsonable(obj.tolist())
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.complexfloating):
        return {"__complex__": [float(obj.real), float(obj.imag)]}
    return obj


def _excitation_numbers_from_system(system: Any) -> np.ndarray | None:
    """Total-quanta weight of each dressed eigenstate of ``system``.

    For a multimode system the "excitation number" of a dressed eigenstate is
    the expectation of the total quanta operator (sum of each subsystem's level
    index), i.e. ``sum_j |<bare_j|dressed_i>|^2 * total_quanta(bare_j)`` evaluated
    in the bare product basis. This reduces to 0, 1, 2, ... for a single mode and
    generalizes it for e.g. a coupled LINC + transmon, where the energy ordering
    of dressed states is no longer the same as their total excitation number.

    Returns ``None`` if ``system`` does not expose the needed structure (then the
    analysis falls back to the index ordering).
    """
    try:
        subsystems = system.subsystem_list
        dims = [int(s.truncated_dim) for s in subsystems]
        evecs = system.hs["evecs"][0]
    except (AttributeError, KeyError, TypeError):
        return None
    dim = int(np.prod(dims))
    if len(evecs) != dim:
        return None
    total_quanta = np.array(
        [sum(np.unravel_index(j, dims)) for j in range(dim)], dtype=float
    )
    # columns are the dressed eigenstates expressed in the bare product basis,
    # ordered consistently with system.H0()
    bare_amplitudes = np.column_stack(
        [np.asarray(ev.full()).flatten() for ev in evecs]
    )
    probs = np.abs(bare_amplitudes) ** 2
    return total_quanta @ probs


class DriveParameters(Serializable):
    """Drive-frequency and drive-amplitude sweep grid plus index helpers.

    This carries the device-independent bookkeeping shared by every model: the
    axes we scan over and the lookups the floquet analysis needs to map a
    (omega_d, amp) value back to its grid index.

    Parameters:
        omega_d_values: drive frequencies to scan over
        drive_amplitudes: amp values to scan over. Can be one dimensional in
            which case these amplitudes are used for all omega_d, or it can be
            two dimensional in which case the first dimension are the amplitudes
            to scan over and the second are the amplitudes for respective drive
            frequencies.
    """

    def __init__(
        self,
        omega_d_values: np.ndarray | list,
        drive_amplitudes: np.ndarray | list,
    ):
        omega_d_values = np.asarray(omega_d_values)
        drive_amplitudes = np.asarray(drive_amplitudes)
        if drive_amplitudes.ndim == 1:
            drive_amplitudes = np.tile(drive_amplitudes, (len(omega_d_values), 1)).T
        assert drive_amplitudes.ndim == 2
        assert drive_amplitudes.shape[1] == len(omega_d_values)

        self.omega_d_values = omega_d_values
        self.drive_amplitudes = drive_amplitudes

    def omega_d_to_idx(self, omega_d: float) -> int:
        """Return index corresponding to omega_d value."""
        return int(np.argmin(np.abs(self.omega_d_values - omega_d)))

    def amp_to_idx(self, amp: float, omega_d: float) -> int:
        """Return index corresponding to amplitude value.

        Because the drive amplitude can depend on the drive frequency, we also
        must pass the drive frequency here.
        """
        omega_d_idx = self.omega_d_to_idx(omega_d)
        return int(np.argmin(np.abs(self.drive_amplitudes[:, omega_d_idx] - amp)))

    def omega_d_amp_params(self, amp_idxs: list) -> itertools.chain:
        """Return ordered chain object of the specified omega_d and amp values."""
        amp_range_vals = self.drive_amplitudes[amp_idxs[0] : amp_idxs[1]]
        _omega_d_amp_params = [
            product([omega_d], amp_vals)
            for omega_d, amp_vals in zip(
                self.omega_d_values, amp_range_vals.T, strict=False
            )
        ]
        return chain(*_omega_d_amp_params)


class HamiltonianModel(DriveParameters, ABC):
    """Abstract base for floquet models.

    Concrete subclasses supply the device-specific pieces the floquet analysis
    relies on:

        - ``self.H0``: the (diagonal) drift Hamiltonian, in units such that it
          can be passed directly to qutip. It MUST be diagonal, i.e. expressed
          in its own eigenbasis, since the displaced-state fit and amplitude
          converters assume the bare states are the computational basis states.
        - ``self.H1`` (optional): a single representative drive operator used
          only by the amplitude converters (ChiacToAmp / XiSqToAmp). It does not
          enter the simulated Hamiltonian. Defaults to ``None``.
        - ``hamiltonian(omega_d_amp)``: the periodic ``qt.QobjEvo`` handed to
          ``qt.FloquetBasis``.

    The sweep grid and index helpers are inherited from ``DriveParameters``.
    Subclasses should call ``super().__init__(omega_d_values, drive_amplitudes)``
    and then assign ``self.H0`` (and optionally ``self.H1``).
    """

    H1: qt.Qobj | None = None

    @abstractmethod
    def hamiltonian(self, omega_d_amp: tuple[float, float]) -> qt.QobjEvo:
        """Return the periodic Hamiltonian we actually simulate."""

    def bare_state_array(self) -> np.ndarray:
        """Return array of bare states (computational basis of the eigenbasis H0)."""
        return np.identity(self.H0.shape[-1], dtype=complex)


class Model(HamiltonianModel):
    """Specify the model, including the Hamiltonian, drive strengths and frequencies.

    Can be subclassed to e.g. override the hamiltonian() method for a different (but
    still periodic!) Hamiltonian.

    Parameters:
        H0: Drift Hamiltonian, which must be diagonal and provided in units such that
            H0 can be passed directly to qutip.
        H1: Drive operator, which should be unitless (for instance the charge-number
            operator n of the transmon). It will be multiplied by a drive amplitude
            that we scan over from drive_parameters.drive_amplitudes.
        omega_d_values: drive frequencies to scan over
        drive_amplitudes: amp values to scan over. Can be one dimensional in which case
            these amplitudes are used for all omega_d, or it can be two dimensional
            in which case the first dimension are the amplitudes to scan over
            and the second are the amplitudes for respective drive frequencies

    """

    def __init__(
        self,
        H0: qt.Qobj | np.ndarray | list,
        H1: qt.Qobj | np.ndarray | list,
        omega_d_values: np.ndarray,
        drive_amplitudes: np.ndarray,
    ):
        super().__init__(omega_d_values, drive_amplitudes)
        if not isinstance(H0, qt.Qobj):
            H0 = qt.Qobj(np.array(H0, dtype=complex))
        if not isinstance(H1, qt.Qobj):
            H1 = qt.Qobj(np.array(H1, dtype=complex))
        self.H0 = H0
        self.H1 = H1

    def hamiltonian(self, omega_d_amp: tuple[float, float]) -> qt.QobjEvo:
        """Return the Hamiltonian we actually simulate."""
        omega_d, amp = omega_d_amp
        return qt.QobjEvo([self.H0, [amp * self.H1, lambda t: np.cos(omega_d * t)]])


class DeviceModel(HamiltonianModel):
    """Build a floquet model from a device `System` and a single drive term.

    This adapts the flexible device classes (e.g. ``FT_Sim.Qutip.device.System``
    and its subsystems) to the interface the floquet analysis expects. The
    ``System`` is responsible for producing the diagonal drift Hamiltonian
    (``system.H0()``) and for promoting a drive operator into the dressed
    eigenbasis (``system.H1_drive_op(op, subsys)``); this class wraps those
    outputs and the time-dependence into a single ``qt.QobjEvo``.

    The drive is a SINGLE term: one operator times one coefficient string. The
    coefficient string may reference ``amp``, ``omega_d``, ``t``, and any name
    supplied via ``static_args``, so forms with an offset inside the cosine such
    as ``"cos(phi_dc + amp * sin(omega_d * t))"`` are expressed without splitting
    into multiple terms.

    Parameters:
        system: Object providing ``H0()`` (diagonalized drift Hamiltonian),
            ``H1_drive_op(op, subsys)`` (operator promoted to the dressed
            eigenbasis), ``subsystem_list`` and ``to_dict()``. Duck-typed so
            floquet need not import the device package.
        drive_subsystem: The subsystem in ``system.subsystem_list`` that is driven.
        drive_op: The drive operator in the *subsystem* basis. It is promoted to
            the dressed eigenbasis via ``system.H1_drive_op``.
        drive_coeff: Time-dependence string for the drive term.
        omega_d_values: drive frequencies to scan over.
        drive_amplitudes: amp values to scan over (see ``DriveParameters``).
        static_args: Constants referenced by ``drive_coeff`` (e.g.
            ``{"phi_dc": 0.5 * np.pi}``). Defaults to an empty dict.
        drive_label: Optional human-readable description of the drive operator,
            e.g. ``"-2*EJ1*cos(phi)  on linc1"``. Stored alongside the data so
            the drive Hamiltonian can be understood from the saved metadata
            (``drive_label`` + ``drive_coeff`` + ``static_args``) without having
            to inspect the operator matrix. Defaults to ``str(drive_subsystem)``.

    Note:
        H0 must be diagonal (an eigenbasis Hamiltonian); ``system.H0()`` already
        satisfies this. ``system.H0()`` also zeroes the ground-state energy
        (``evals - evals[0]``); this is physically irrelevant for quasienergies
        but differs from the plain ``Model``, which does not re-zero H0.

        Serialization: ``write_to_file`` saves the run results plus a record of
        the model (``system.to_dict()`` and the drive metadata). The model can
        be rebuilt with :meth:`from_dict`, supplying a ``System`` reconstructed
        via ``System.from_dict`` (floquet does not import the device package,
        so the caller provides the rebuilt system).
    """

    def __init__(
        self,
        system: Any,
        drive_subsystem: Any,
        drive_op: qt.Qobj,
        drive_coeff: str,
        omega_d_values: np.ndarray,
        drive_amplitudes: np.ndarray,
        static_args: dict | None = None,
        drive_label: str | None = None,
    ):
        super().__init__(omega_d_values, drive_amplitudes)
        self.system = system
        self.drive_subsystem = drive_subsystem
        self.drive_coeff = drive_coeff
        self.static_args = static_args or {}
        self.drive_label = (
            drive_label if drive_label is not None else str(drive_subsystem)
        )

        # Promote the drive operator into the dressed eigenbasis once, and grab
        # the diagonalized drift Hamiltonian from the system.
        self.H1 = system.H1_drive_op(drive_op, drive_subsystem)
        self.H0 = system.H0()
        # Total-quanta weight of each dressed eigenstate, so the branch analysis
        # reports a physically meaningful mean excitation for multimode systems.
        self.excitation_numbers = _excitation_numbers_from_system(system)

    def hamiltonian(self, omega_d_amp: tuple[float, float]) -> qt.QobjEvo:
        """Return the Hamiltonian we actually simulate."""
        omega_d, amp = omega_d_amp
        return qt.QobjEvo(
            [self.H0, [self.H1, self.drive_coeff]],
            args={"amp": amp, "omega_d": omega_d, **self.static_args},
        )

    def drive_summary(self) -> str:
        """Human-readable description of the drive Hamiltonian.

        Returns the operator label, the time-dependence string, and any
        constants, e.g.::

            Drive: (-2*EJ1*cos(phi)  on linc1) * sin(amp * sin(omega_d * t))
              static_args: phi_dc=1.5708

        This is meant to answer "which drive was used?" without inspecting the
        operator matrix.
        """
        lines = [f"Drive: ({self.drive_label})"]
        if self.static_args:
            consts = ", ".join(f"{k}={v}" for k, v in self.static_args.items())
            lines.append(f"  static_args: {consts}")
        return "\n".join(lines)

    def __str__(self) -> str:
        """Concise summary that does not dump the operator/H0 matrices."""
        return (
            "DeviceModel\n"
            f"{self.drive_summary()}\n"
            f"  hilbert_dim: {self.H0.shape[0]}\n"
            f"  omega_d_values: {len(self.omega_d_values)} pts "
            f"in [{self.omega_d_values.min()/TWO_PI:.4g} * TWO_PI, {self.omega_d_values.max()/TWO_PI:.4g} * TWO_PI]\n"
            f"  drive_amplitudes: {self.drive_amplitudes.shape} "
            f"in [{self.drive_amplitudes.min()/PI:.4g} * PI, {self.drive_amplitudes.max()/PI:.4g} * PI]"
        )

    def serialize(self) -> dict:
        """h5-friendly snapshot of the model definition.

        The ``system`` object is not itself Serializable, so we store its
        ``to_dict()`` as a JSON string (``system_json``) -- this keeps the saved
        structure flat and h5-safe even when ``to_dict()`` uses non-string keys
        or numpy scalars. The drive label and coefficient string make the drive
        Hamiltonian human-readable from the saved file; the (dressed-basis)
        operator matrix is stored as ``drive_op``. Rebuild with :meth:`from_dict`.
        """
        try:
            drive_subsystem_idx = self.system.subsystem_list.index(self.drive_subsystem)
        except (AttributeError, ValueError):
            drive_subsystem_idx = -1
        try:
            system_json = json.dumps(_to_jsonable(self.system.to_dict()))
        except (AttributeError, TypeError):
            system_json = "null"
        return {
            "system_json": system_json,
            "drive_subsystem_idx": drive_subsystem_idx,
            "drive_op": self.H1.data.to_array(),
            "drive_coeff": self.drive_coeff,
            "drive_label": self.drive_label,
            "static_args_json": json.dumps(_to_jsonable(self.static_args)),
            "omega_d_values": self.omega_d_values,
            "drive_amplitudes": self.drive_amplitudes,
        }

    @classmethod
    def from_dict(cls, data: dict, system: Any) -> "DeviceModel":
        """Rebuild a ``DeviceModel`` from :meth:`serialize` output.

        Because floquet does not import the device package, the caller supplies a
        reconstructed ``system`` (e.g. ``System.from_dict(data["system"])``). The
        stored ``drive_op`` is the already-dressed operator (in the transformed
        eigenbasis), so it is injected directly as ``H1`` rather than re-promoted
        through ``system.H1_drive_op``.

        Parameters:
            data: Dictionary produced by :meth:`serialize` (or read back from an
                h5 file's ``_init_attrs`` group).
            system: A system object providing ``H0()`` and ``subsystem_list``.

        Returns:
            A ``DeviceModel`` whose ``H0``/``H1`` and drive metadata match the
            saved run.
        """
        obj = cls.__new__(cls)
        DriveParameters.__init__(
            obj, data["omega_d_values"], data["drive_amplitudes"]
        )
        obj.system = system
        idx = int(data.get("drive_subsystem_idx", -1))
        obj.drive_subsystem = (
            system.subsystem_list[idx]
            if (0 <= idx < len(system.subsystem_list))
            else None
        )
        obj.drive_coeff = _as_str(data["drive_coeff"])
        obj.drive_label = _as_str(data.get("drive_label", obj.drive_subsystem))
        if "static_args_json" in data:
            obj.static_args = json.loads(_as_str(data["static_args_json"]))
        else:
            obj.static_args = dict(data.get("static_args", {}))
        # Use the stored dressed operator directly (see docstring).
        obj.H1 = qt.Qobj(np.asarray(data["drive_op"]))
        obj.H0 = system.H0()
        return obj
