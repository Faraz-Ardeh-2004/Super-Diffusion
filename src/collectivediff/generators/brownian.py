"""Ordinary Brownian motion: the reference case everything else is measured against."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..config import DT, BrownianConfig
from ._common import as_trajectories, require_generator

__all__ = ["brownian"]


def brownian(cfg: BrownianConfig, rng: np.random.Generator) -> NDArray[np.float64]:
    """Generate Brownian trajectories.

    Increments are independent Gaussians with per-component variance
    ``2 D dt``, so in ``d`` dimensions

    .. math::

        \\langle r^2(t) \\rangle = 2 d D t ,

    exactly, at every grid point and not merely asymptotically. The process is
    Gaussian, Markovian, ergodic, and has a delta increment autocorrelation --
    the null hypothesis against which every anomaly in this project is judged.

    Parameters
    ----------
    cfg : BrownianConfig
        Ensemble size, trajectory length, dimension and diffusivity ``D``.
    rng : numpy.random.Generator
        Explicit generator; no global RNG is touched.

    Returns
    -------
    ndarray of float64
        Shape ``(n_particles, n_steps, n_dim)``, starting at the origin.
    """
    require_generator(rng)
    n_particles, n_steps, n_dim = cfg.shape
    sigma = np.sqrt(2.0 * cfg.diffusivity * DT)
    increments = sigma * rng.standard_normal((n_particles, n_steps - 1, n_dim))
    positions = np.zeros((n_particles, n_steps, n_dim), dtype=np.float64)
    np.cumsum(increments, axis=1, out=positions[:, 1:, :])
    return as_trajectories(positions, n_particles, n_steps, n_dim)
