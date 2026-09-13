"""Continuous-time random walk with heavy-tailed waiting times.

This is the crux of Q1. The CTRW is weakly non-ergodic: its *ensemble* MSD
scales as ``t^a``, but the time-averaged MSD of any single trajectory grows
*linearly* in the lag whatever ``a`` is, with an amplitude that stays a random
variable no matter how long the trajectory runs. A single observed agent
therefore cannot reveal the exponent -- an information-theoretic wall, not a
finite-sample nuisance.

Reference: He, Burov, Metzler & Barkai, PRL 101, 058101 (2008).

Structure follows ``PROJECT.md`` section 2: the raw event sequence in continuous
time is built here, and the projection onto the uniform grid is delegated to
:func:`~collectivediff.generators.resampling.resample_step`, which is tested on
its own.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..config import DT, CTRWConfig
from ._common import as_trajectories, pareto_durations_covering, require_generator
from .resampling import resample_step

__all__ = ["ctrw", "ctrw_events"]


def ctrw_events(
    cfg: CTRWConfig,
    rng: np.random.Generator,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Build the raw continuous-time event sequence, before any gridding.

    Waiting times are drawn from the Pareto tail
    ``P(tau > t) = (t / tau0)^(-a)`` with ``a`` in ``(0, 1)``, so the mean
    waiting time diverges. Jumps are Gaussian with per-component standard
    deviation ``jump_scale``; keeping them light-tailed is deliberate, so that
    CTRW is a pure waiting-time mechanism and mixes in no Levy statistics.

    Parameters
    ----------
    cfg : CTRWConfig
        CTRW config.
    rng : numpy.random.Generator
        Explicit generator.

    Returns
    -------
    event_times : ndarray of float64
        Shape ``(n_particles, n_events + 1)``; column 0 is the initial
        condition at ``t = 0``, column ``k`` is the time of the ``k``-th jump.
    event_positions : ndarray of float64
        Shape ``(n_particles, n_events + 1, n_dim)``; the position *after* each
        jump, starting at the origin.
    """
    require_generator(rng)
    n_particles, _, n_dim = cfg.shape

    waits = pareto_durations_covering(
        rng,
        n_particles=n_particles,
        duration=cfg.total_duration,
        tail_exponent=cfg.alpha_wait,
        scale=cfg.tau0,
        oversample=cfg.oversample,
    )
    n_events = waits.shape[1]
    if cfg.jump_distribution != "gaussian":  # pragma: no cover - blocked by the config
        raise ValueError(f"unsupported jump_distribution {cfg.jump_distribution!r}")
    jumps = cfg.jump_scale * rng.standard_normal((n_particles, n_events, n_dim))

    event_times = np.zeros((n_particles, n_events + 1), dtype=np.float64)
    np.cumsum(waits, axis=1, out=event_times[:, 1:])
    event_positions = np.zeros((n_particles, n_events + 1, n_dim), dtype=np.float64)
    np.cumsum(jumps, axis=1, out=event_positions[:, 1:, :])
    return event_times, event_positions


def ctrw(cfg: CTRWConfig, rng: np.random.Generator) -> NDArray[np.float64]:
    """Generate subdiffusive CTRW trajectories on the uniform grid.

    The walker waits, then jumps, then waits again. Between jumps it does not
    move, so the grid representation is a zero-order hold on the event sequence
    and a long waiting time appears as a flat stretch.

    Asymptotically, for ``t >> tau0``,

    .. math::

        \\langle r^2(t) \\rangle \\simeq
        \\frac{d\\,\\sigma_{\\mathrm{jump}}^2}{\\Gamma(1-a)\\Gamma(1+a)}
        \\left(\\frac{t}{\\tau_0}\\right)^{a} ,

    subdiffusive since ``a < 1``. The leading correction is of relative order
    ``(t/tau0)^(-a)`` and decays slowly, so the exponent should be fitted well
    beyond ``tau0``.

    Parameters
    ----------
    cfg : CTRWConfig
        CTRW config.
    rng : numpy.random.Generator
        Explicit generator.

    Returns
    -------
    ndarray of float64
        Shape ``(n_particles, n_steps, n_dim)``, starting at the origin.

    Notes
    -----
    With ``aging_time = t_a > 0`` the walker starts at process time ``-t_a`` and
    the observation window is ``[t_a, t_a + T]``, with positions measured
    relative to where the walker was when observation began. This is not a
    cosmetic shift: at ``t_a > 0`` the walker is typically already inside a long
    waiting period drawn before the observer arrived, so the first observed
    displacement is delayed, the ensemble MSD acquires an ``a``-dependent
    prefactor, and the EB plateau changes. ``t_a = 0`` reproduces the ordinary
    CTRW bit for bit, and is what phase 1 validates.
    """
    require_generator(rng)
    n_particles, n_steps, n_dim = cfg.shape
    event_times, event_positions = ctrw_events(cfg, rng)
    grid = cfg.aging_time + np.arange(n_steps, dtype=np.float64) * DT
    positions = resample_step(event_times, event_positions, grid)
    positions = positions - positions[:, :1, :]
    return as_trajectories(positions, n_particles, n_steps, n_dim)
