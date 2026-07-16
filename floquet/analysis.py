"""Post-processing and plotting for saved Floquet analysis data.

``FloquetResults`` consumes the data produced by
:meth:`floquet.FloquetAnalysis.run` (either the returned ``data_dict`` or the
h5 file it writes) and provides a consistent set of plots and fits:

    - :meth:`infidelity_map`          - the ionization / leakage "scar" heatmap
    - :meth:`branches`                - Blais branch plot (avg excitation or
                                        quasienergy) vs drive strength
    - :meth:`avoided_crossing_fit`    - extract parametric coupling g from an
                                        avoided crossing in the (folded)
                                        quasienergy spectrum vs drive frequency
    - :meth:`g_vs_drive_strength`     - g extracted at each drive amplitude for a
                                        chosen mode pair, with an optional trend

Data array conventions (matching :meth:`FloquetAnalysis.run`):
    - quasienergies, avg_excitation:        shape (n_omega_d, n_amp, hilbert_dim)
    - displaced_state_overlaps, etc.:        shape (n_omega_d, n_amp, n_states)
    - drive_amplitudes:                      shape (n_amp, n_omega_d)
Angular frequencies are stored throughout (omega = 2*pi*f); plotting helpers
convert to ordinary GHz on the axes.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

TWO_PI = 2.0 * np.pi


def _as_str(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode()
    return str(value)


def _load_h5(filepath: str) -> dict:
    """Recursively read an h5 file written by ``write_to_file`` into a dict."""
    import h5py

    def _rec(group: "h5py.Group") -> dict:
        out = {}
        for key in group:
            item = group[key]
            out[key] = _rec(item) if isinstance(item, h5py.Group) else item[()]
        return out

    with h5py.File(filepath, "r") as f:
        return _rec(f)


class FloquetResults:
    """Container + plotting/fitting helpers for one Floquet sweep.

    Parameters:
        omega_d_values: 1-D array of drive frequencies (angular), shape (W,).
        drive_amplitudes: 2-D array of drive amplitudes, shape (A, W).
        data: dict of result arrays (the ``data_dict`` from ``run``); expected
            keys include ``quasienergies``, ``avg_excitation``,
            ``displaced_state_overlaps``, and optionally ``bare_state_overlaps``,
            ``floquet_modes``, ``fit_data``.
        state_indices: the state indices that were tracked (last axis of the
            ``*_overlaps`` arrays). Used to map a physical state label to its
            column. Defaults to ``None`` (treat the label as a column position).
        drive_label: optional human-readable description of the drive (e.g. from
            ``DeviceModel.drive_label``), used in plot titles.
    """

    def __init__(
        self,
        omega_d_values: np.ndarray,
        drive_amplitudes: np.ndarray,
        data: dict,
        state_indices: Sequence[int] | None = None,
        drive_label: str | None = None,
    ):
        self.omega_d_values = np.asarray(omega_d_values)
        self.drive_amplitudes = np.asarray(drive_amplitudes)
        if self.drive_amplitudes.ndim == 1:
            self.drive_amplitudes = np.tile(
                self.drive_amplitudes, (len(self.omega_d_values), 1)
            ).T
        self.data = dict(data)
        self.state_indices = list(state_indices) if state_indices is not None else None
        self.drive_label = drive_label
        self.hilbert_dim = (
            self.data["quasienergies"].shape[-1]
            if "quasienergies" in self.data
            else None
        )

    # ------------------------------------------------------------------ loaders
    @classmethod
    def from_file(cls, filepath: str) -> "FloquetResults":
        """Build from an h5 file written by ``FloquetAnalysis.run(filepath=...)``.

        Reads the result arrays and the drive sweep axes from the saved model
        metadata, without needing to fully reconstruct the model object (so it
        works for both ``Model`` and ``DeviceModel`` outputs).
        """
        raw = _load_h5(filepath)
        init_attrs = raw.get("_init_attrs", {})
        model_meta = init_attrs.get("model", {})
        data = {k: v for k, v in raw.items() if k != "_init_attrs"}

        state_indices = init_attrs.get("state_indices", None)
        if state_indices is not None:
            state_indices = list(np.atleast_1d(state_indices))
        drive_label = model_meta.get("drive_label", None)
        if drive_label is not None:
            drive_label = _as_str(drive_label)
        return cls(
            omega_d_values=model_meta["omega_d_values"],
            drive_amplitudes=model_meta["drive_amplitudes"],
            data=data,
            state_indices=state_indices,
            drive_label=drive_label,
        )

    @classmethod
    def from_run(
        cls, data_dict: dict, model: Any, state_indices: Sequence[int] | None = None
    ) -> "FloquetResults":
        """Build directly from the in-memory ``run`` output and its model."""
        return cls(
            omega_d_values=model.omega_d_values,
            drive_amplitudes=model.drive_amplitudes,
            data=data_dict,
            state_indices=state_indices,
            drive_label=getattr(model, "drive_label", None),
        )

    # ------------------------------------------------------------------ helpers
    def omega_d_to_idx(self, omega_d: float) -> int:
        """Index of the nearest drive frequency (angular units)."""
        return int(np.argmin(np.abs(self.omega_d_values - omega_d)))

    def amp_to_idx(self, amp: float, omega_d_idx: int = 0) -> int:
        """Index of the nearest drive amplitude in the given frequency column."""
        return int(np.argmin(np.abs(self.drive_amplitudes[:, omega_d_idx] - amp)))

    def _state_position(self, state: int) -> int:
        """Map a state label to its column in the overlap arrays."""
        if self.state_indices is not None and state in self.state_indices:
            return self.state_indices.index(state)
        return state

    @property
    def omega_d_GHz(self) -> np.ndarray:
        return self.omega_d_values / TWO_PI


    # -------------------------------------------------------------- infidelity
    def infidelity_map(
        self,
        state: int,
        clip: float | None = 0.5,
        y_values: np.ndarray | None = None,
        y_label: str = "drive amplitude",
        ax: Any = None,
        cmap: str = "Blues",
        plot_states: str = 'displaced',
        **imshow_kw: Any,
    ):
        """Ionization / leakage heatmap: 1 - |<ideal|floquet>|^2 over (freq, amp).

        Parameters:
            state: state label (in ``state_indices``) or column position.
            clip: upper clip for nicer contrast (set ``None`` to disable).
            y_values: optional 1-D array to label the amplitude axis (e.g. the
                induced chi_ac that was held constant across drive frequencies).
                Defaults to the drive amplitudes of the first frequency column.
            y_label: label for the amplitude axis.
        """
        import matplotlib.pyplot as plt

        pos = self._state_position(state)
        try: use_data = self.data[f"{plot_states}_state_overlaps"] 
        except: raise ValueError(f'Invalid state type: {plot_states}. ' \
                                'Options are bare, displaced, intermediate displaced.')
        infidelity = 1.0 - use_data[:, :, pos]**2
        if clip is not None:
            infidelity = np.clip(infidelity, 0.0, clip)

        x = self.omega_d_GHz
        if y_values is None:
            y_values = self.drive_amplitudes[:, 0]
            y_label = 'drive amplitudes'
        y_values = np.asarray(y_values)

        fig = None
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 5))
        extent = [x.min(), x.max(), y_values.min(), y_values.max()]
        im = ax.imshow(
            infidelity.T,
            origin="lower",
            aspect="auto",
            extent=extent,
            cmap=cmap,
            vmin=0.0,
            vmax=clip,
            interpolation="none",
            **imshow_kw,
        )
        ax.set_xlabel(r"$\omega_d/2\pi$ [GHz]")
        ax.set_ylabel(y_label)
        title = f"infidelity to state {state}"
        ax.set_title(title)
        if fig is not None:
            fig.colorbar(im, ax=ax, label=r"$1-|\langle\rm ideal|\psi\rangle|^2$")
        return (fig if fig is not None else ax.figure), ax

    # ----------------------------------------------------------------- branches
    def branches(
        self,
        omega_d_GHz: float,
        quantity: str = "avg_excitation",
        branch_indices: Sequence[int] | None = None,
        idx: bool = False,
        x_values: np.ndarray | None = None,
        x_label: str = "drive amplitude",
        ax: Any = None,
    ):
        """Blais branch plot at a fixed drive frequency.

        Parameters:
            omega_d: drive frequency (angular) or its index if ``idx=True``.
            quantity: ``"avg_excitation"`` or ``"quasienergies"``.
            branch_indices: which branches to plot (default: all).
            x_values: optional 1-D array for the x-axis (e.g. chi_ac). Defaults to
                the drive amplitudes of this frequency column.
        """
        import matplotlib.pyplot as plt

        omega_d = omega_d_GHz * TWO_PI
        omega_idx = omega_d if idx else self.omega_d_to_idx(omega_d)
        y = self.data[quantity][omega_idx]  # (A, hilbert_dim)
        if quantity == "quasienergies":
            y = y / TWO_PI
        if branch_indices is None:
            branch_indices = range(y.shape[-1])
        if x_values is None:
            x_values = self.drive_amplitudes[:, omega_idx]
            x_label = "drive amplitude"
        x_values = np.asarray(x_values)

        fig = None
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 3.5))
        for b in branch_indices:
            ax.plot(x_values, np.real(y[:, b]), label=str(b))
        ax.set_xlabel(x_label)
        ax.set_ylabel(quantity)
        ax.set_title(
            rf"$\omega_d/2\pi$ = {self.omega_d_GHz[omega_idx]:.4f} GHz"
        )
        ax.legend(fontsize=9, ncol=2, loc="upper left", bbox_to_anchor=(1.0, 1.0))
        if fig is not None:
            fig.tight_layout()
        return (fig if fig is not None else ax.figure), ax


    # ---------------------------------------------------- avoided-crossing fit
    def _folded_gap(self, amp_idx: int, mode_pair: tuple[int, int]) -> np.ndarray:
        """Quasienergy gap (GHz) between two branches vs drive freq, folded.

        Quasienergies are defined mod omega_d; a one-photon resonance folds the
        two branches onto each other, so we wrap the difference into one Floquet
        zone (per frequency column) and take its magnitude. The minimum of this
        gap over frequency is ``2 g`` at the avoided crossing.
        """
        m, n = mode_pair
        q = self.data["quasienergies"][:, amp_idx, :] / TWO_PI  # (W, dim), GHz
        omega_GHz = self.omega_d_GHz  # per-column zone width
        delta = q[:, n] - q[:, m]
        delta -= np.round(delta / omega_GHz) * omega_GHz
        return np.abs(delta)

    def avoided_crossing_fit(
        self,
        mode_pair: tuple[int, int],
        near_freq: float,
        amp: float,
        idx: bool = False,
        fit_range: float = 0.05,
        search_range: float = 0.1,
        plot: bool = True,
        ax: Any = None,
        n_drive: int = 1
    ) -> dict:
        """Fit an avoided crossing to extract the parametric coupling g.

        Sweeps drive frequency at a fixed drive amplitude, forms the folded
        quasienergy gap between the two branches in ``mode_pair``, and fits
        ``sqrt(4 g^2 + (omega_d - omega_res)^2)``. Returns ``g`` (the half-gap)
        and the resonant frequency.

        Parameters:
            mode_pair: the two branch indices that cross.
            near_freq: approximate resonance drive frequency [GHz] to search near.
            amp: drive amplitude (matched in the first frequency column) or its
                index if ``idx=True``.
            fit_range: width [GHz] of the window used for the curve fit.
            search_range: width [GHz] of the window used to locate the min gap.

        Returns:
            dict with ``g`` [GHz], ``omega_res`` [GHz], ``pcov``, and the
            ``freq``/``gap``/``fit`` arrays over the fit window.
        """
        from scipy.optimize import curve_fit

        amp_idx = int(amp) if idx else self.amp_to_idx(amp)
        omega_GHz = self.omega_d_GHz
        gap = self._folded_gap(amp_idx, mode_pair)

        # locate the resonance (min gap) within the search window
        lo, hi = near_freq - search_range / 2, near_freq + search_range / 2
        search_mask = (omega_GHz >= lo) & (omega_GHz <= hi)
        if not np.any(search_mask):
            raise ValueError(
                f"no drive frequencies in search window [{lo:.4f}, {hi:.4f}] GHz"
            )
        res_guess = omega_GHz[search_mask][np.argmin(gap[search_mask])]

        # fit window centered on the resonance guess
        flo, fhi = res_guess - fit_range / 2, res_guess + fit_range / 2
        fit_mask = (omega_GHz >= flo) & (omega_GHz <= fhi)
        x, y = omega_GHz[fit_mask], gap[fit_mask]

        def fit_func(w: np.ndarray, g: float, w_res: float) -> np.ndarray:
            return np.sqrt(4.0 * g**2 + (n_drive * (w - w_res)) ** 2)

        popt, pcov = curve_fit(
            fit_func, x, y, p0=[max(np.min(y) / 2, 1e-4), res_guess]
        )
        g = abs(popt[0])
        result = {
            "g": g,
            "omega_res": popt[1],
            "amp_idx": amp_idx,
            "pcov": pcov,
            "freq": x,
            "gap": y,
            "fit": fit_func(x, *popt),
        }

        if plot:
            import matplotlib.pyplot as plt

            fig = None
            if ax is None:
                fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(x, y, ".", color="black", label="folded gap")
            ax.plot(x, result["fit"], "-", color="red", label=f"g = {g * 1e3:.2f} MHz")
            ax.axvline(popt[1], ls="--", color="grey", lw=1)
            ax.set_xlabel(r"drive frequency $\omega_d/2\pi$ [GHz]")
            ax.set_ylabel(r"$|\varepsilon_n-\varepsilon_m|$ [GHz]")
            ax.set_title(f"avoided crossing, modes {mode_pair}, {n_drive} drive photons")
            ax.legend()
            result["fig"] = fig if fig is not None else ax.figure
            result["ax"] = ax
        return result

    # ---------------------------------------------------- g vs drive strength
    def g_vs_drive_strength(
        self,
        mode_pair: tuple[int, int],
        near_freq: float,
        amp_indices: Sequence[int] | None = None,
        fit_range: float = 0.05,
        search_range: float = 0.1,
        n_drive: int = 1,
        trend: int = 1,
        track_resonance: bool = True,
        ax: Any = None,
    ) -> dict:
        """Extract g at each drive amplitude for a mode pair; plot g vs amplitude.

        Repeats :meth:`avoided_crossing_fit` across drive amplitudes for the same
        crossing, optionally following the resonance as it ac-Stark shifts. The
        x-axis is the drive amplitude evaluated at the (shifting) resonant
        frequency for each amplitude.

        Parameters:
            mode_pair: the two crossing branch indices.
            near_freq: initial guess [GHz] for the resonance.
            amp_indices: which amplitude indices to use (default: all but amp=0).
            trend: overlay a trend through the points. ``"linear"`` fits
                ``g = a*amp + b``; an int fits a polynomial of that degree;
                ``None`` disables the overlay.
            track_resonance: update the search guess to the previous amplitude's
                fitted resonance (recommended; the crossing drifts with amp).

        Returns:
            dict with arrays ``amps``, ``g``, ``omega_res`` and (if requested)
            ``trend_coeffs``, plus ``fig``/``ax``.
        """
        import matplotlib.pyplot as plt

        n_amp = self.drive_amplitudes.shape[0]
        if amp_indices is None:
            amp_indices = range(1, n_amp)  # skip zero-amplitude point

        amps, gs, freqs = [], [], []
        guess = near_freq
        for ai in amp_indices:
            try:
                r = self.avoided_crossing_fit(
                    mode_pair,
                    guess,
                    ai,
                    idx=True,
                    fit_range=fit_range,
                    search_range=search_range,
                    plot=False,
                    n_drive=n_drive
                )
                g, w_res = r["g"], r["omega_res"]
                if track_resonance and np.isfinite(w_res):
                    guess = w_res
                res_omega_idx = self.omega_d_to_idx(w_res * TWO_PI)
                amps.append(self.drive_amplitudes[ai, res_omega_idx])
                gs.append(g)
                freqs.append(w_res)
            except (RuntimeError, ValueError):
                amps.append(self.drive_amplitudes[ai, self.omega_d_to_idx(guess * TWO_PI)])
                gs.append(np.nan)
                freqs.append(np.nan)

        amps = np.asarray(amps)
        gs = np.asarray(gs)
        freqs = np.asarray(freqs)

        fig = None
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(amps, gs * 1e3, "o", label="extracted g")
        ax.set_xlabel("drive amplitude")
        ax.set_ylabel("g [MHz]")
        ax.set_title(f"g vs drive strength, modes {mode_pair}")

        trend_coeffs = None
        good = np.isfinite(amps) & np.isfinite(gs)
        if trend is not None and np.count_nonzero(good) >= 2:
            deg = trend
            trend_coeffs = np.polyfit(amps[good], gs[good], deg)
            xs = np.linspace(amps[good].min(), amps[good].max(), 200)
            ax.plot(
                xs,
                np.polyval(trend_coeffs, xs) * 1e3,
                "-",
                color="red",
                label=f"trend (deg {deg})",
            )
        ax.legend()
        if fig is not None:
            fig.tight_layout()

        return {
            "amps": amps,
            "g": gs,
            "omega_res": freqs,
            "trend_coeffs": trend_coeffs,
            "fig": fig if fig is not None else ax.figure,
            "ax": ax,
        }
