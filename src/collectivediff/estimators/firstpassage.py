"""First-passage and return statistics.

A first-passage time asks a different question from an MSD: not how far the
walker has spread on average, but when it first reached somewhere. The two are
not interchangeable, and the difference is largest exactly where this project
lives -- for a subdiffusive CTRW the mean first-passage time can diverge while
the MSD is perfectly finite, because a single trapping event can outlast any
measurement.

For one-dimensional Brownian motion started at the origin, the first-passage
time to a level ``L`` has the Levy-Smirnov density

.. math::

    p(t) = \\frac{L}{\\sqrt{4\\pi D t^3}}\\,e^{-L^2/(4Dt)} ,

whose survival probability is ``S(t) = erf(L / sqrt(4 D t))``, decaying as
``t^{-1/2}``: the mean is infinite even in the simplest possible case. Any
summary statistic here must therefore be a median or a quantile, never a mean,
and the functions below return the raw times so the caller cannot accidentally
average them.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.special import erf

from .msd import check_layout

__all__ = [
    "brownian_survival",
    "first_passage_times",
    "return_statistics",
    "survival_probability",
]


def first_passage_times(
    trajectories: NDArray[Any],
    threshold: float,
    component: int = 0,
    absolute: bool = True,
    dt: float = 1.0,
) -> NDArray[np.float64]:
    """First time each trajectory reaches a threshold.

    Parameters
    ----------
    trajectories : ndarray
        Shape ``(n_particles, n_steps, n_dim)``.
    threshold : float
        Level ``L > 0``.
    component : int
        Spatial component to test; statistics run in 1-D.
    absolute : bool
        If ``True``, the two-sided problem ``|x| >= L``; if ``False``, the
        one-sided ``x >= L``.
    dt : float
        Grid spacing.

    Returns
    -------
    ndarray of float64
        Shape ``(n_particles,)``. Particles that never reach the threshold get
        ``inf``, not the trajectory length: censoring them as if they had
        arrived at ``T`` would bias every quantile downwards, and the censoring
        fraction is itself informative.

    Notes
    -----
    On a discrete grid the walker is only observed at integer times, so a
    crossing that happened and reversed between two grid points is missed. The
    resulting times are therefore upper bounds, biased high by less than one
    step; for the thresholds used here, several times the single-step
    displacement, the effect is negligible.
    """
    check_layout(trajectories)
    if threshold <= 0.0:
        raise ValueError(f"threshold must be > 0, got {threshold}")
    signal = np.asarray(trajectories[:, :, component], dtype=np.float64)
    signal = signal - signal[:, :1]
    beyond = np.abs(signal) >= threshold if absolute else signal >= threshold

    reached = beyond.any(axis=1)
    index = np.argmax(beyond, axis=1).astype(np.float64)
    return np.where(reached, index * dt, np.inf)


def survival_probability(
    trajectories: NDArray[Any],
    threshold: float,
    component: int = 0,
    absolute: bool = True,
    dt: float = 1.0,
) -> NDArray[np.float64]:
    """Fraction of trajectories that have not yet reached the threshold.

    .. math::

        S(t) = P(\\tau_{\\mathrm{fp}} > t)

    Parameters
    ----------
    trajectories : ndarray
        Shape ``(n_particles, n_steps, n_dim)``.
    threshold : float
        Level ``L > 0``.
    component : int
        Spatial component to test.
    absolute : bool
        Two-sided if ``True``.
    dt : float
        Grid spacing.

    Returns
    -------
    ndarray of float64
        Shape ``(n_steps,)``, starting at 1 and non-increasing.
    """
    times = first_passage_times(trajectories, threshold, component, absolute, dt)
    grid = np.arange(trajectories.shape[1], dtype=np.float64) * dt
    return np.mean(times[:, None] > grid[None, :], axis=0, dtype=np.float64)


def brownian_survival(
    t: NDArray[np.float64],
    threshold: float,
    diffusivity: float,
) -> NDArray[np.float64]:
    """Analytic one-sided survival probability for 1-D Brownian motion.

    .. math::

        S(t) = \\mathrm{erf}\\!\\left(\\frac{L}{\\sqrt{4Dt}}\\right)

    The two-sided problem has a different, series-valued answer; this reference
    is for ``absolute=False``.

    Parameters
    ----------
    t : ndarray of float64
        Times.
    threshold : float
        Level ``L``.
    diffusivity : float
        ``D``.

    Returns
    -------
    ndarray of float64
        ``S(t)``, equal to 1 at ``t = 0``.
    """
    t = np.asarray(t, dtype=np.float64)
    with np.errstate(divide="ignore"):
        argument = np.where(t > 0.0, threshold / np.sqrt(4.0 * diffusivity * t), np.inf)
    return erf(argument)


def return_statistics(
    trajectories: NDArray[Any],
    tolerance: float | None = None,
    component: int = 0,
    dt: float = 1.0,
) -> dict[str, NDArray[np.float64] | float]:
    """Return-to-origin statistics: how often, and how long between returns.

    A "return" is a sign change of the displacement, optionally requiring the
    walker to come within ``tolerance`` of the origin. Sign changes are used
    rather than exact hits because on a continuous-space grid the walker never
    lands exactly on zero.

    The distinction this measures is between a walker that oscillates about its
    start and one that commits to a direction: fBm with ``H < 1/2`` returns far
    more often than Brownian motion, ``H > 1/2`` far less, and a Levy walk least
    of all because it is ballistic between turns. A CTRW returns at the Brownian
    rate *per jump* but at a much lower rate per unit time, since most of its
    time is spent immobile.

    Parameters
    ----------
    trajectories : ndarray
        Shape ``(n_particles, n_steps, n_dim)``.
    tolerance : float, optional
        If given, a sign change only counts when ``|x| <= tolerance`` at one of
        the two straddling points.
    component : int
        Spatial component to use.
    dt : float
        Grid spacing.

    Returns
    -------
    dict
        ``n_returns`` -- per-particle count, shape ``(n_particles,)``;
        ``rate`` -- mean returns per unit time over the ensemble;
        ``intervals`` -- pooled times between consecutive returns, for a
        distribution; ``never_returned`` -- fraction of particles with none.
    """
    check_layout(trajectories)
    signal = np.asarray(trajectories[:, :, component], dtype=np.float64)
    signal = signal - signal[:, :1]

    crossing = np.sign(signal[:, :-1]) * np.sign(signal[:, 1:]) < 0.0
    if tolerance is not None:
        if tolerance <= 0.0:
            raise ValueError(f"tolerance must be > 0, got {tolerance}")
        near = (np.abs(signal[:, :-1]) <= tolerance) | (np.abs(signal[:, 1:]) <= tolerance)
        crossing &= near

    counts = crossing.sum(axis=1).astype(np.float64)
    duration = (trajectories.shape[1] - 1) * dt

    intervals: list[NDArray[np.float64]] = []
    for row in crossing:
        positions = np.flatnonzero(row)
        if positions.size > 1:
            intervals.append(np.diff(positions).astype(np.float64) * dt)
    pooled = (
        np.concatenate(intervals) if intervals else np.empty(0, dtype=np.float64)
    )

    return {
        "n_returns": counts,
        "rate": float(np.mean(counts, dtype=np.float64) / duration),
        "intervals": pooled,
        "never_returned": float(np.mean(counts == 0.0)),
    }
