"""Scaled Brownian motion: Gaussian and Markovian, but non-stationary.

SBM is the cheapest way to make an anomalous exponent, and for that reason it is
a useful adversary: it matches the exponent of fBm or of a CTRW without sharing
either mechanism's correlation structure. Separating it from those is part of
what Q2 asks.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..config import DT, SBMConfig
from ._common import as_trajectories, require_generator

__all__ = ["sbm"]


def sbm(cfg: SBMConfig, rng: np.random.Generator) -> NDArray[np.float64]:
    """Generate scaled-Brownian-motion trajectories.

    The diffusivity is a deterministic function of the *absolute* time,
    ``D(t) = alpha D0 t^(alpha - 1)``, which gives

    .. math::

        \\langle r^2(t) \\rangle = 2 d D_0 t^{\\alpha} .

    Rather than integrating ``D(t)`` with an Euler step, the increments are
    drawn exactly from the law of the process: the variance of the increment
    over ``[t_i, t_{i+1}]`` is ``2 D0 (t_{i+1}^alpha - t_i^alpha)`` per
    component. The result is exact at every grid point for any ``alpha``,
    including the strongly time-dependent small-``alpha`` case where an Euler
    step would be worst.

    The process is Gaussian with a delta increment autocorrelation but is *not*
    stationary, hence not ergodic: its time-averaged MSD ages, decaying as a
    power of the measurement time.

    Parameters
    ----------
    cfg : SBMConfig
        Ensemble size, trajectory length, dimension, exponent ``alpha`` and
        prefactor ``D0``.
    rng : numpy.random.Generator
        Explicit generator.

    Returns
    -------
    ndarray of float64
        Shape ``(n_particles, n_steps, n_dim)``, starting at the origin.
    """
    require_generator(rng)
    n_particles, n_steps, n_dim = cfg.shape

    t = np.arange(n_steps, dtype=np.float64) * DT
    variance = 2.0 * cfg.diffusivity * np.diff(t**cfg.alpha)
    sigma = np.sqrt(variance)[None, :, None]

    increments = sigma * rng.standard_normal((n_particles, n_steps - 1, n_dim))
    positions = np.zeros((n_particles, n_steps, n_dim), dtype=np.float64)
    np.cumsum(increments, axis=1, out=positions[:, 1:, :])
    return as_trajectories(positions, n_particles, n_steps, n_dim)
