"""Increment and velocity autocorrelation.

The increment ACF is the observable that says *why* a process is anomalous, as
opposed to how much. Reading down the last column of ``PROJECT.md`` section 4:

* Brownian, scaled Brownian, diffusing diffusivity, Levy flight -- a delta, no
  memory at all; their anomaly, where they have one, comes from a
  time-dependent or fluctuating amplitude, not from correlation.
* fBm with ``H < 1/2`` -- negative at lag 1, the increments actively reverse.
* fBm with ``H > 1/2`` -- positive and slowly decaying, a power law
  ``~ k^(2H-2)`` that is not summable for ``H > 1/2``: long-range memory.
* CTRW -- zero correlation, but interrupted by long immobile stretches.
* Levy walk -- ballistic runs, so the correlation persists for the duration of
  a flight.

Two mechanisms sharing an exponent will generally not share this curve, which is
what makes it central to Q2.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from .msd import check_layout

__all__ = ["fbm_increment_acf", "increment_acf", "velocity_acf"]


def fbm_increment_acf(lags: NDArray[np.intp], hurst: float) -> NDArray[np.float64]:
    """Closed-form increment autocorrelation of fBm.

    .. math::

        \\rho(k) = \\tfrac{1}{2}\\left(|k-1|^{2H} - 2|k|^{2H}
                   + |k+1|^{2H}\\right)

    normalised so ``rho(0) = 1``. Asymptotically ``rho(k) ~ H(2H-1) k^{2H-2}``.

    Parameters
    ----------
    lags : ndarray of int
        Lags at which to evaluate.
    hurst : float
        Hurst exponent in ``(0, 1)``.

    Returns
    -------
    ndarray of float64
        The correlation at each lag.
    """
    k = np.asarray(lags, dtype=np.float64)
    two_h = 2.0 * hurst
    return 0.5 * (np.abs(k - 1.0) ** two_h - 2.0 * np.abs(k) ** two_h + (k + 1.0) ** two_h)


def increment_acf(
    trajectories: NDArray[Any],
    max_lag: int,
    component: int = 0,
    normalise: bool = True,
) -> NDArray[np.float64]:
    """Autocorrelation of the one-step increments, averaged over the ensemble.

    .. math::

        \\rho(k) = \\frac{\\langle \\delta x(t)\\,\\delta x(t+k) \\rangle}
                         {\\langle \\delta x(t)^2 \\rangle}

    with the average taken over both particles and starting times. Averaging
    over ``t`` presumes stationary increments; for scaled Brownian motion, whose
    increments age by construction, the result is a time-averaged correlation
    and should be read as such.

    Parameters
    ----------
    trajectories : ndarray
        Shape ``(n_particles, n_steps, n_dim)``.
    max_lag : int
        Largest lag returned; the output covers ``k = 0 .. max_lag``.
    component : int
        Which spatial component to use. Statistics run in 1-D
        (``PROJECT.md`` section 2).
    normalise : bool
        Divide by the variance so ``rho(0) = 1``. If ``False`` the raw
        covariance is returned.

    Returns
    -------
    ndarray of float64
        Shape ``(max_lag + 1,)``.

    Notes
    -----
    Computed directly rather than by FFT. The lag range of interest is a few
    dozen out of thousands of steps, so the direct form is both faster and free
    of the circular-wraparound bookkeeping an FFT would need.
    """
    check_layout(trajectories)
    increments = np.diff(
        np.asarray(trajectories[:, :, component], dtype=np.float64), axis=1
    )
    n_increments = increments.shape[1]
    if not 0 <= max_lag < n_increments:
        raise ValueError(f"max_lag must lie in [0, {n_increments - 1}], got {max_lag}")

    increments = increments - increments.mean()
    out = np.empty(max_lag + 1, dtype=np.float64)
    for lag in range(max_lag + 1):
        stop = n_increments - lag
        out[lag] = np.mean(
            increments[:, :stop] * increments[:, lag:], dtype=np.float64
        )
    if normalise:
        if out[0] <= 0.0:
            raise ValueError("increment variance is zero; cannot normalise")
        out = out / out[0]
    return out


def velocity_acf(
    trajectories: NDArray[Any],
    max_lag: int,
    component: int = 0,
    dt: float = 1.0,
    normalise: bool = True,
) -> NDArray[np.float64]:
    """Velocity autocorrelation on the uniform grid.

    On a grid of spacing ``dt`` the velocity is the increment divided by ``dt``,
    so this differs from :func:`increment_acf` only by ``dt^-2`` and is exactly
    equal to it when normalised. It exists as a separate name because the Levy
    walk is naturally described by a velocity process -- the walker moves at
    constant speed ``v`` along a flight -- and calling the same array a velocity
    correlation there is the physically meaningful reading.

    Parameters
    ----------
    trajectories : ndarray
        Shape ``(n_particles, n_steps, n_dim)``.
    max_lag : int
        Largest lag returned.
    component : int
        Which spatial component to use.
    dt : float
        Grid spacing, 1 by project convention.
    normalise : bool
        Divide by the zero-lag value.

    Returns
    -------
    ndarray of float64
        Shape ``(max_lag + 1,)``.
    """
    if dt <= 0.0:
        raise ValueError(f"dt must be > 0, got {dt}")
    raw = increment_acf(trajectories, max_lag, component, normalise=False)
    return raw / raw[0] if normalise else raw / dt**2
