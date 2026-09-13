"""Ergodicity breaking: the observable that decides whether one agent is enough.

``PROJECT.md`` section 5 defines, with ``xi = TA-MSD / <TA-MSD>`` at a given lag,

.. math::

    \\mathrm{EB}(\\Delta) = \\mathrm{Var}(\\xi) = \\langle \\xi^2 \\rangle - 1 .

``xi`` is the *dimensionless amplitude scatter*: how far one trajectory's time
average sits from the ensemble's. For an ergodic process the scatter shrinks as
the trajectory lengthens and ``EB -> 0`` as ``T / Delta -> infinity``. For a
subdiffusive CTRW it does not: ``EB`` converges to a nonzero plateau that
depends only on the waiting-time exponent,

.. math::

    \\mathrm{EB}_\\infty(a) = \\frac{2\\,\\Gamma^2(1+a)}{\\Gamma(1+2a)} - 1 ,

and the distribution of ``xi`` tends to a broad, non-degenerate limit law
instead of collapsing onto ``xi = 1``.

That plateau is the quantitative form of the wall in Q1: no amount of memory
along a single trajectory removes it, because it is not a finite-sample effect.

Reference: He, Burov, Metzler & Barkai, PRL 101, 058101 (2008); Metzler, Jeon,
Cherstvy & Barkai, PCCP 16, 24128 (2014).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.special import gamma as gamma_fn

from .msd import ta_msd

__all__ = [
    "amplitude_scatter",
    "brownian_eb",
    "ctrw_eb_plateau",
    "eb_parameter",
    "ergodicity_breaking_curve",
]


def brownian_eb(lag: float | NDArray[np.float64], n_steps: int) -> NDArray[np.float64]:
    """Ergodicity-breaking parameter of one-dimensional Brownian motion.

    .. math::

        \\mathrm{EB}(\\Delta) \\simeq \\frac{4\\Delta}{3T},
        \\qquad \\Delta \\ll T

    the classic self-averaging result: the scatter of the TA-MSD amplitude falls
    to zero in proportion to the fraction of the trajectory one lag occupies.

    This is the **positive control** for the EB implementation, and it matters at
    least as much as the CTRW plateau. A wrong constant factor inside
    :func:`eb_parameter` could still let a CTRW test pass by coincidence -- the
    plateau is a plateau whatever it is multiplied by -- but it cannot survive a
    comparison against a formula with a specific slope *and* a specific
    coefficient.

    Parameters
    ----------
    lag : float or ndarray
        Lag ``Delta`` in steps.
    n_steps : int
        Trajectory length including ``t = 0``, so ``T = n_steps - 1``.

    Returns
    -------
    ndarray of float64
        Predicted ``EB(Delta)``. At ``T = 4096`` this gives 0.00326, 0.01628 and
        0.03255 at lags 10, 50 and 100.
    """
    duration = n_steps - 1
    return 4.0 * np.asarray(lag, dtype=np.float64) / (3.0 * duration)


def ctrw_eb_plateau(alpha_wait: float) -> float:
    """Asymptotic EB parameter of a subdiffusive CTRW.

    .. math::

        \\mathrm{EB}_\\infty(a) = \\frac{2\\,\\Gamma^2(1+a)}{\\Gamma(1+2a)} - 1

    It rises from 0 at ``a = 1`` (where the CTRW becomes ergodic) to 1 as
    ``a -> 0``. At ``a = 1/2`` it equals ``pi/2 - 1 ~ 0.5708``.

    Parameters
    ----------
    alpha_wait : float
        Waiting-time exponent ``a`` in ``(0, 1]``.

    Returns
    -------
    float
        The plateau value.
    """
    if not 0.0 < alpha_wait <= 1.0:
        raise ValueError(f"alpha_wait must lie in (0, 1], got {alpha_wait}")
    return float(
        2.0 * gamma_fn(1.0 + alpha_wait) ** 2 / gamma_fn(1.0 + 2.0 * alpha_wait) - 1.0
    )


def amplitude_scatter(
    trajectories: NDArray[Any],
    lag: int,
    cfg: Any | None = None,
) -> NDArray[np.float64]:
    """Dimensionless TA-MSD amplitude ``xi`` of each trajectory at one lag.

    .. math::

        \\xi_i = \\frac{\\overline{\\delta_i^2(\\Delta)}}
                       {\\big\\langle \\overline{\\delta^2(\\Delta)} \\big\\rangle}

    By construction ``<xi> = 1``; everything interesting is in its spread and
    the shape of its distribution. For an ergodic process the histogram
    concentrates on 1 as ``T`` grows; for a CTRW it converges to a broad limit
    law with a finite density at ``xi = 0`` -- trajectories that spent the whole
    measurement in a single trapping event.

    Parameters
    ----------
    trajectories : ndarray
        Shape ``(n_particles, n_steps, n_dim)``.
    lag : int
        Lag ``Delta`` at which to evaluate.
    cfg : GeneratorConfig, optional
        Guards against a divergent second moment.

    Returns
    -------
    ndarray of float64
        Shape ``(n_particles,)``, mean exactly 1.
    """
    values = ta_msd(trajectories, np.array([lag], dtype=np.intp), cfg)[:, 0]
    mean = float(np.mean(values, dtype=np.float64))
    if mean <= 0.0:
        raise ValueError(
            f"ensemble-averaged TA-MSD at lag {lag} is not positive ({mean}); "
            "cannot normalise the amplitude scatter"
        )
    return values / mean


def eb_parameter(
    trajectories: NDArray[Any],
    lag: int,
    cfg: Any | None = None,
) -> float:
    """Ergodicity-breaking parameter at one lag.

    .. math::

        \\mathrm{EB}(\\Delta) = \\langle \\xi^2 \\rangle - 1

    Parameters
    ----------
    trajectories : ndarray
        Shape ``(n_particles, n_steps, n_dim)``.
    lag : int
        Lag ``Delta``.
    cfg : GeneratorConfig, optional
        Guards against a divergent second moment.

    Returns
    -------
    float
        ``Var(xi)``. Zero for a perfectly self-averaging process.

    Notes
    -----
    The population variance is used, not the sample variance: ``EB`` is defined
    as ``<xi^2> - 1`` with ``<xi> = 1`` exactly by construction, so there is no
    mean to have consumed a degree of freedom. The difference is ``O(1/N)`` and
    matters only for the very small ensembles used in unit tests.
    """
    xi = amplitude_scatter(trajectories, lag, cfg)
    return float(np.mean(xi**2, dtype=np.float64) - 1.0)


def ergodicity_breaking_curve(
    trajectories: NDArray[Any],
    lags: NDArray[np.intp],
    cfg: Any | None = None,
) -> NDArray[np.float64]:
    """EB parameter across a range of lags.

    Plotted against ``Delta / T`` this is the diagnostic that separates the
    ergodic mechanisms from the non-ergodic ones at a glance: fBm and Brownian
    motion decay towards zero as the lag shrinks relative to ``T``, while a CTRW
    flattens onto :func:`ctrw_eb_plateau`.

    Parameters
    ----------
    trajectories : ndarray
        Shape ``(n_particles, n_steps, n_dim)``.
    lags : ndarray of int
        Lags at which to evaluate.
    cfg : GeneratorConfig, optional
        Guards against a divergent second moment.

    Returns
    -------
    ndarray of float64
        Shape ``(len(lags),)``.
    """
    lags = np.asarray(lags, dtype=np.intp)
    values = ta_msd(trajectories, lags, cfg)
    means = np.mean(values, axis=0, dtype=np.float64)
    if np.any(means <= 0.0):
        raise ValueError("ensemble-averaged TA-MSD is not positive at every lag")
    xi = values / means
    return np.mean(xi**2, axis=0, dtype=np.float64) - 1.0
