"""Moment scaling spectrum ``nu(q)`` -- the fingerprint of strong anomalous diffusion.

``PROJECT.md`` section 5: fit ``<|x|^q> ~ t^(q nu(q))`` for ``q`` on a grid.
A constant ``nu(q)`` means *simple* scaling -- one length scale growing as
``t^nu``, so every moment is fixed once one of them is. A ``nu(q)`` that is
piecewise linear with a kink means *strong* anomalous diffusion: the bulk of the
distribution and its tail grow at different rates, and no single exponent
describes the process.

This is the sharpest observable in the toolbox for Q2, because it is the only
one that identifies strong anomalous diffusion unambiguously. fBm, scaled
Brownian motion and the CTRW all have flat spectra; only the Levy walk kinks.

The Levy walk, for ``1 < g < 2``, has the bilinear spectrum

.. math::

    \\nu(q) = \\begin{cases}
        1/g & q < g \\\\
        1 - (g-1)/q & q > g
    \\end{cases}

joining continuously at ``q = g``, where both branches give ``1/g``. The low-``q``
branch is the stable bulk, which spreads as ``t^(1/g)``; the high-``q`` branch is
the ballistic cone at ``|x| = v t``, whose weight decays as a power of time, and
``nu -> 1`` as ``q -> infinity``. Putting ``q = 2`` recovers
``<x^2> ~ t^(3-g)``, and ``q = 4`` gives ``<x^4> ~ t^(5-g)`` -- which is why the
MSD estimator of a Levy walk gets noisier as time grows.

Reference: Castiglione, Mazzino, Muratore-Ginanneschi & Vulpiani, Physica D 134,
75 (1999).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..config import FitConfig
from .msd import check_layout, fit_powerlaw

__all__ = ["MomentSpectrum", "fit_bilinear_spectrum", "levy_walk_nu", "moment_spectrum"]


def levy_walk_nu(q: NDArray[np.float64], gamma: float) -> NDArray[np.float64]:
    """Analytic bilinear moment spectrum of a Levy walk with ``1 < g < 2``.

    .. math::

        \\nu(q) = \\max\\left(1/g,\\; 1 - (g-1)/q\\right)

    The ``max`` form is equivalent to the piecewise definition and makes the
    continuity at the kink manifest: the two branches cross exactly at ``q = g``.

    Parameters
    ----------
    q : ndarray of float64
        Moment orders, all positive.
    gamma : float
        Flight-duration exponent ``g`` in ``(1, 2)``.

    Returns
    -------
    ndarray of float64
        ``nu(q)``.
    """
    q = np.asarray(q, dtype=np.float64)
    if np.any(q <= 0.0):
        raise ValueError("moment orders must be positive")
    if not 1.0 < gamma < 2.0:
        raise ValueError(f"gamma must lie in (1, 2), got {gamma}")
    return np.maximum(1.0 / gamma, 1.0 - (gamma - 1.0) / q)


@dataclass(frozen=True)
class MomentSpectrum:
    """Measured ``nu(q)`` on a grid of moment orders.

    Attributes
    ----------
    q : ndarray of float64
        The moment orders.
    nu : ndarray of float64
        Fitted ``nu(q) = slope(q) / q`` where ``<|x|^q> ~ t^slope(q)``.
    nu_err : ndarray of float64
        Bootstrap standard error on ``nu(q)``, or ``nan`` where unavailable.
    window : tuple of float
        The fit window in time, shared by every ``q``.
    """

    q: NDArray[np.float64]
    nu: NDArray[np.float64]
    nu_err: NDArray[np.float64]
    window: tuple[float, float]

    def is_flat(self, tolerance: float = 0.02) -> bool:
        """Whether the spectrum is constant to within ``tolerance``.

        A flat spectrum means simple scaling. A spectrum that is not flat is the
        signature of strong anomalous diffusion.
        """
        return bool(np.nanmax(self.nu) - np.nanmin(self.nu) <= tolerance)

    def kink_location(self) -> float:
        """Position of the kink, from a fit of the analytic bilinear family.

        Shorthand for :func:`fit_bilinear_spectrum`, which returns the ``g`` of
        the best-fitting Levy-walk spectrum; the kink sits at ``q = g``. Check
        :meth:`is_flat` first -- a flat spectrum has no kink and the answer is
        meaningless rather than merely imprecise.

        Returns
        -------
        float
            The fitted ``g``.
        """
        return fit_bilinear_spectrum(self)


def fit_bilinear_spectrum(
    spectrum: MomentSpectrum,
    bounds: tuple[float, float] = (1.01, 1.99),
    n_grid: int = 2000,
) -> float:
    """Fit the analytic Levy-walk family ``nu(q; g)`` to a measured spectrum.

    Rather than splitting the ``q`` grid by hand and intersecting two regression
    lines, this fits the one-parameter family
    ``nu(q; g) = max(1/g, 1 - (g-1)/q)`` of :func:`levy_walk_nu` by least
    squares over ``g``. Both branches then constrain the answer jointly, and
    because the kink of that family sits exactly at ``q = g``, the fitted
    parameter *is* the kink location.

    This is much steadier than a two-line intersection when the ``q`` grid is
    coarse or when one branch is short, which is the usual situation: the low-
    ``q`` branch is flat and carries little information about where it ends.

    Parameters
    ----------
    spectrum : MomentSpectrum
        Measured spectrum.
    bounds : tuple of float
        Search range for ``g``, kept inside ``(1, 2)`` where the Levy walk is
        defined.
    n_grid : int
        Points in the brute-force scan. A scan rather than an optimiser because
        the objective has a kink in it, which gradient methods handle badly, and
        because a two-thousand-point scan of a closed form costs nothing.

    Returns
    -------
    float
        Best-fitting ``g``.
    """
    finite = np.isfinite(spectrum.nu)
    if finite.sum() < 3:
        raise ValueError("need at least three finite points to fit a spectrum")
    q = spectrum.q[finite]
    nu = spectrum.nu[finite]
    candidates = np.linspace(bounds[0], bounds[1], n_grid)
    residuals = np.array(
        [float(np.sum((nu - levy_walk_nu(q, g)) ** 2)) for g in candidates]
    )
    return float(candidates[int(np.argmin(residuals))])


def moment_spectrum(
    trajectories: NDArray[Any],
    q_values: NDArray[np.float64],
    t_min: float,
    t_max: float,
    component: int = 0,
    dt: float = 1.0,
    fit: FitConfig | None = None,
) -> MomentSpectrum:
    """Measure ``nu(q)`` by fitting ``<|x|^q> ~ t^(q nu(q))`` at each ``q``.

    Parameters
    ----------
    trajectories : ndarray
        Shape ``(n_particles, n_steps, n_dim)``.
    q_values : ndarray of float64
        Moment orders, e.g. ``numpy.linspace(0.2, 4.0, 20)``.
    t_min, t_max : float
        Fit window in time. Stated by the caller, as everywhere else: for a Levy
        walk it must exclude the first few flight durations, where the walker is
        still purely ballistic and every moment scales with ``nu = 1``.
    component : int
        Spatial component to use; statistics run in 1-D.
    dt : float
        Grid spacing.
    fit : FitConfig, optional
        Bootstrap settings, passed to :func:`~collectivediff.estimators.msd.fit_powerlaw`.

    Returns
    -------
    MomentSpectrum
        With per-``q`` bootstrap errors.

    Notes
    -----
    High moments are dominated by the few largest displacements in the ensemble,
    so ``nu(q)`` at large ``q`` needs many more particles than ``nu(2)`` does for
    the same precision. The bootstrap error reflects that honestly and should be
    consulted before reading a kink off a plot.
    """
    check_layout(trajectories)
    q_values = np.atleast_1d(np.asarray(q_values, dtype=np.float64))
    if np.any(q_values <= 0.0):
        raise ValueError("moment orders must be positive")

    displacement = np.abs(
        np.asarray(
            trajectories[:, :, component] - trajectories[:, :1, component],
            dtype=np.float64,
        )
    )
    t = np.arange(trajectories.shape[1], dtype=np.float64) * dt

    nu = np.empty(q_values.size, dtype=np.float64)
    nu_err = np.empty(q_values.size, dtype=np.float64)
    window = (np.nan, np.nan)
    for index, q in enumerate(q_values):
        result = fit_powerlaw(t, displacement**q, t_min, t_max, fit=fit)
        nu[index] = result.exponent / q
        nu_err[index] = (
            np.nan if result.bootstrap_err is None else result.bootstrap_err / q
        )
        window = result.window
    return MomentSpectrum(q=q_values, nu=nu, nu_err=nu_err, window=window)
