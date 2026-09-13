"""Diffusing diffusivity: Brownian yet non-Gaussian.

The odd one out in this project. Its MSD is exactly linear -- ``alpha = 1``,
indistinguishable from Brownian motion by any MSD-based estimator -- yet its
propagator has exponential rather than Gaussian tails at short times. It is the
standing counterexample to reading a mechanism off an exponent, and therefore
one of the sharpest tests of Q2.

Reference: Chechkin, Seno, Metzler & Sokolov, PRX 7, 021002 (2017).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..config import DT, DDMConfig
from ._common import as_trajectories, require_generator

__all__ = ["ddm", "ou_process"]


def ou_process(
    cfg: DDMConfig,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """Simulate the auxiliary Ornstein-Uhlenbeck process exactly.

    .. math::

        dY = -\\frac{Y}{\\tau}\\,dt + \\sigma\\,dW

    is a linear SDE, so it is integrated by its exact transition law rather than
    by an Euler step:

    .. math::

        Y_{t+\\Delta} = Y_t e^{-\\Delta/\\tau}
        + \\sqrt{\\tfrac{\\sigma^2\\tau}{2}\\left(1 - e^{-2\\Delta/\\tau}\\right)}\\,\\xi ,
        \\qquad \\xi \\sim \\mathcal{N}(0,1) .

    Using the exact law matters here because ``tau`` is a free parameter that
    the phase-2 sweeps will push towards ``dt``, where an Euler step would
    silently mis-set the stationary variance and hence ``<D>``.

    The initial condition is drawn from the stationary distribution
    ``N(0, sigma^2 tau / 2)``, so the ensemble is stationary from ``t = 0`` and
    the MSD is linear from the first step rather than after a burn-in.

    Parameters
    ----------
    cfg : DDMConfig
        Diffusing-diffusivity config.
    rng : numpy.random.Generator
        Explicit generator.

    Returns
    -------
    ndarray of float64
        Shape ``(n_particles, n_steps, n_aux)``.
    """
    require_generator(rng)
    n_particles, n_steps, _ = cfg.shape

    decay = float(np.exp(-DT / cfg.tau))
    stationary_std = float(np.sqrt(cfg.sigma**2 * cfg.tau / 2.0))
    kick_std = stationary_std * float(np.sqrt(1.0 - decay**2))

    y = np.empty((n_particles, n_steps, cfg.n_aux), dtype=np.float64)
    y[:, 0, :] = stationary_std * rng.standard_normal((n_particles, cfg.n_aux))
    kicks = kick_std * rng.standard_normal((n_particles, n_steps - 1, cfg.n_aux))
    for step in range(1, n_steps):
        y[:, step, :] = decay * y[:, step - 1, :] + kicks[:, step - 1, :]
    return y


def ddm(cfg: DDMConfig, rng: np.random.Generator) -> NDArray[np.float64]:
    """Generate diffusing-diffusivity trajectories.

    The diffusivity is itself a stochastic process, ``D(t) = |Y(t)|^2`` with
    ``Y`` an ``n_aux``-component OU process, and the walker takes a Gaussian
    step of per-component variance ``2 D(t) dt`` conditioned on it.

    Since the step and ``D`` are independent given the past,

    .. math::

        \\langle r^2(t) \\rangle = 2 d \\langle D \\rangle t ,
        \\qquad \\langle D \\rangle = n_{\\mathrm{aux}}\\,\\sigma^2\\tau/2 ,

    exactly linear. The *propagator*, on the other hand, is a superposition of
    Gaussians weighted by the distribution of ``D``, which for ``n_aux = 1``
    gives exponential (Laplace) tails. Those tails survive for ``t << tau`` and
    cross over to Gaussian for ``t >> tau``, once the walker has sampled many
    independent diffusivities.

    Parameters
    ----------
    cfg : DDMConfig
        Diffusing-diffusivity config.
    rng : numpy.random.Generator
        Explicit generator.

    Returns
    -------
    ndarray of float64
        Shape ``(n_particles, n_steps, n_dim)``, starting at the origin.
    """
    require_generator(rng)
    n_particles, n_steps, n_dim = cfg.shape

    y = ou_process(cfg, rng)
    diffusivity = np.sum(y**2, axis=2)
    # The step over [t, t+dt] uses the diffusivity at the start of the interval.
    sigma = np.sqrt(2.0 * diffusivity[:, :-1] * DT)[:, :, None]
    increments = sigma * rng.standard_normal((n_particles, n_steps - 1, n_dim))

    positions = np.zeros((n_particles, n_steps, n_dim), dtype=np.float64)
    np.cumsum(increments, axis=1, out=positions[:, 1:, :])
    return as_trajectories(positions, n_particles, n_steps, n_dim)
