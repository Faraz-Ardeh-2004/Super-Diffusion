"""Levy flight and Levy walk -- superficially similar, statistically opposite.

Both draw on a heavy tail, but they couple it to space differently and that
difference decides whether the second moment exists at all:

* a **Levy flight** takes an instantaneous jump of stable-distributed length
  once per time step. Arbitrarily long jumps cost no time, the walker's
  effective velocity is unbounded, and ``<x^2>`` diverges. Only fractional
  moments ``q < s`` are meaningful.
* a **Levy walk** travels at a *finite speed* for a heavy-tailed duration. A
  long excursion now costs proportionally long, which cuts off the tail of the
  displacement at ``v t`` and leaves a finite MSD, ``<x^2(t)> ~ t^(3-g)``.

The Levy walk is also the project's example of *strong* anomalous diffusion: its
moment spectrum ``nu(q)`` is piecewise linear with a kink, which is what
separates it from fBm at matched ``alpha`` (Q2).

References: Zaburdaev, Denisov & Klafter, RMP 87, 483 (2015); Chambers, Mallows
& Stuck, JASA 71, 340 (1976) for the stable sampler.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..config import DT, LevyFlightConfig, LevyWalkConfig
from ._common import (
    as_trajectories,
    pareto_durations_covering,
    random_directions,
    require_generator,
)
from .resampling import resample_linear

__all__ = [
    "isotropic_stable",
    "levy_flight",
    "levy_walk",
    "levy_walk_events",
    "positive_stable",
    "symmetric_stable",
]


def symmetric_stable(
    rng: np.random.Generator,
    size: tuple[int, ...],
    stability: float,
    scale: float = 1.0,
) -> NDArray[np.float64]:
    """Draw symmetric ``s``-stable random variables (Chambers-Mallows-Stuck).

    With ``U`` uniform on ``(-pi/2, pi/2)`` and ``W`` standard exponential,

    .. math::

        X = \\frac{\\sin(sU)}{(\\cos U)^{1/s}}
            \\left(\\frac{\\cos(U - sU)}{W}\\right)^{(1-s)/s},
        \\qquad s \\ne 1,

    and ``X = tan(U)`` for ``s = 1`` (Cauchy). The tail is
    ``P(|X| > x) ~ x^(-s)``, so ``E|X|^q`` is finite exactly for ``q < s`` and
    the variance diverges for every ``s < 2``.

    Parameters
    ----------
    rng : numpy.random.Generator
        Explicit generator.
    size : tuple of int
        Output shape.
    stability : float
        Stability index ``s`` in ``(0, 2)``.
    scale : float
        Multiplicative scale parameter.

    Returns
    -------
    ndarray of float64
        Shape ``size``.
    """
    require_generator(rng)
    if not 0.0 < stability < 2.0:
        raise ValueError(f"stability must lie in (0, 2), got {stability}")

    u = rng.uniform(-0.5 * np.pi, 0.5 * np.pi, size=size)
    w = rng.exponential(1.0, size=size)
    if np.isclose(stability, 1.0):
        return scale * np.tan(u)
    s = stability
    numerator = np.sin(s * u)
    denominator = np.cos(u) ** (1.0 / s)
    tail = (np.cos(u - s * u) / w) ** ((1.0 - s) / s)
    return scale * (numerator / denominator) * tail


def positive_stable(
    rng: np.random.Generator,
    size: tuple[int, ...],
    alpha: float,
) -> NDArray[np.float64]:
    """Draw a positive ``alpha``-stable variable, ``0 < alpha < 1`` (Kanter 1975).

    With ``U`` uniform on ``(0, pi)`` and ``W`` standard exponential,

    .. math::

        A = \\frac{\\sin(\\alpha U)}{(\\sin U)^{1/\\alpha}}
            \\left(\\frac{\\sin((1-\\alpha)U)}{W}\\right)^{(1-\\alpha)/\\alpha} ,

    which is strictly positive and satisfies
    ``E[exp(-lambda A)] = exp(-lambda^alpha)``. Used as the subordinator of the
    sub-Gaussian construction in :func:`isotropic_stable`.

    Parameters
    ----------
    rng : numpy.random.Generator
        Explicit generator.
    size : tuple of int
        Output shape.
    alpha : float
        Stability index in ``(0, 1)``.

    Returns
    -------
    ndarray of float64
        Strictly positive samples of shape ``size``.

    References
    ----------
    Kanter, Ann. Probab. 3, 697 (1975).
    """
    require_generator(rng)
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must lie in (0, 1), got {alpha}")
    u = rng.uniform(0.0, np.pi, size=size)
    w = rng.exponential(1.0, size=size)
    return (np.sin(alpha * u) / np.sin(u) ** (1.0 / alpha)) * (
        np.sin((1.0 - alpha) * u) / w
    ) ** ((1.0 - alpha) / alpha)


def isotropic_stable(
    rng: np.random.Generator,
    size: tuple[int, ...],
    stability: float,
    scale: float,
    n_dim: int,
) -> NDArray[np.float64]:
    """Draw an isotropic symmetric ``s``-stable vector.

    ``PROJECT.md`` section 2 requires an explicit isotropic 2-D implementation
    for the Levy flight, because -- unlike Brownian motion, fBm and scaled
    Brownian motion -- the stable law is *not separable*: a vector with
    independent stable components is not rotationally symmetric. Its heavy tail
    lies along the coordinate axes, so long jumps happen preferentially
    north-south and east-west. Measured on the 99th-percentile jumps, the
    relative spread of the angular histogram is 0.70 for the component-wise
    construction against 0.05 for the isotropic one.

    The isotropic law is obtained by *sub-ordination*: if ``A`` is positive
    ``s/2``-stable with ``E[exp(-lambda A)] = exp(-lambda^(s/2))`` and ``G`` is
    an isotropic Gaussian with per-component variance ``2 c^2``, then
    ``X = sqrt(A) G`` has characteristic function

    .. math::

        E[e^{i \\mathbf{k}\\cdot\\mathbf{X}}]
        = E\\!\\left[e^{-A |\\mathbf{k}|^2 c^2}\\right]
        = e^{-c^{s} |\\mathbf{k}|^{s}} ,

    isotropic by construction and with exactly the 1-D marginal of
    :func:`symmetric_stable` at the same ``scale``.

    Parameters
    ----------
    rng : numpy.random.Generator
        Explicit generator.
    size : tuple of int
        Leading shape; the spatial axis is appended.
    stability : float
        Stability index ``s`` in ``(0, 2)``.
    scale : float
        Scale parameter ``c``.
    n_dim : int
        1 or 2. In one dimension isotropy is vacuous and this delegates to
        :func:`symmetric_stable`, which is cheaper and avoids the subordinator.

    Returns
    -------
    ndarray of float64
        Shape ``size + (n_dim,)``.
    """
    require_generator(rng)
    if n_dim == 1:
        return symmetric_stable(rng, size, stability, scale)[..., None]
    if n_dim != 2:
        raise ValueError(f"n_dim must be 1 or 2, got {n_dim}")
    subordinator = positive_stable(rng, size, 0.5 * stability)
    gaussian = rng.standard_normal(tuple(size) + (n_dim,)) * np.sqrt(2.0) * scale
    return np.sqrt(subordinator)[..., None] * gaussian


def levy_flight(cfg: LevyFlightConfig, rng: np.random.Generator) -> NDArray[np.float64]:
    """Generate Levy-flight trajectories: one stable jump per time step.

    Each spatial component takes an independent symmetric ``s``-stable
    increment per step. By stability, the displacement after ``t`` steps is
    itself ``s``-stable with scale ``t^(1/s)``, so every fractional moment of
    order ``q < s`` grows as

    .. math::

        \\langle |r(t)|^q \\rangle \\sim t^{q/s} ,

    i.e. ``nu(q) = 1/s`` independently of ``q``. The second moment diverges and
    :func:`~collectivediff.estimators.msd.ea_msd` refuses to compute it for this
    config.

    Parameters
    ----------
    cfg : LevyFlightConfig
        Levy-flight config.
    rng : numpy.random.Generator
        Explicit generator.

    Returns
    -------
    ndarray of float64
        Shape ``(n_particles, n_steps, n_dim)``, starting at the origin.

    Notes
    -----
    In two dimensions the increment is drawn from the *isotropic* stable law by
    the sub-Gaussian construction of :func:`isotropic_stable`, not from
    independent components. The stable law is not separable, so the cheap
    component-wise version would send its long jumps preferentially along the
    coordinate axes (``PROJECT.md`` section 2).
    """
    require_generator(rng)
    n_particles, n_steps, n_dim = cfg.shape
    increments = isotropic_stable(
        rng, (n_particles, n_steps - 1), cfg.stability, cfg.scale, n_dim
    )
    positions = np.zeros((n_particles, n_steps, n_dim), dtype=np.float64)
    np.cumsum(increments, axis=1, out=positions[:, 1:, :])
    return as_trajectories(positions, n_particles, n_steps, n_dim)


def levy_walk_events(
    cfg: LevyWalkConfig,
    rng: np.random.Generator,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Build the raw continuous-time flight sequence, before any gridding.

    Flight durations follow ``P(tau > t) = (t / tau0)^(-g)`` with ``g`` in
    ``(1, 2)``: the mean duration is finite but the variance is not. Each flight
    is travelled at constant speed ``v`` in a direction drawn uniformly.

    Parameters
    ----------
    cfg : LevyWalkConfig
        Levy-walk config.
    rng : numpy.random.Generator
        Explicit generator.

    Returns
    -------
    event_times : ndarray of float64
        Shape ``(n_particles, n_flights + 1)``; turning points, starting at 0.
    event_positions : ndarray of float64
        Shape ``(n_particles, n_flights + 1, n_dim)``; position at each turning
        point, starting at the origin.
    """
    require_generator(rng)
    n_particles, _, n_dim = cfg.shape

    durations = pareto_durations_covering(
        rng,
        n_particles=n_particles,
        duration=cfg.duration,
        tail_exponent=cfg.gamma,
        scale=cfg.tau0,
        oversample=cfg.oversample,
    )
    n_flights = durations.shape[1]
    directions = random_directions(rng, (n_particles, n_flights), n_dim)
    displacements = cfg.speed * durations[:, :, None] * directions

    event_times = np.zeros((n_particles, n_flights + 1), dtype=np.float64)
    np.cumsum(durations, axis=1, out=event_times[:, 1:])
    event_positions = np.zeros((n_particles, n_flights + 1, n_dim), dtype=np.float64)
    np.cumsum(displacements, axis=1, out=event_positions[:, 1:, :])
    return event_times, event_positions


def levy_walk(cfg: LevyWalkConfig, rng: np.random.Generator) -> NDArray[np.float64]:
    """Generate superdiffusive Levy-walk trajectories on the uniform grid.

    Because the walker moves *during* a flight, the grid representation is a
    linear interpolation of the turning points, not a zero-order hold. The
    ensemble MSD is finite and grows as

    .. math::

        \\langle r^2(t) \\rangle \\sim t^{3-g}, \\qquad 1 < g < 2 ,

    which is superdiffusive but subballistic. The prefactor depends on
    conventions (whether the flight in progress at ``t = 0`` is aged, whether
    the walker rests between flights) and is not asserted anywhere in this
    project; the exponent is.

    Parameters
    ----------
    cfg : LevyWalkConfig
        Levy-walk config.
    rng : numpy.random.Generator
        Explicit generator.

    Returns
    -------
    ndarray of float64
        Shape ``(n_particles, n_steps, n_dim)``, starting at the origin.

    Notes
    -----
    Flights start fresh at ``t = 0`` -- this is the non-equilibrated, "ordinary"
    Levy walk. An equilibrated ensemble would age the first flight, which shifts
    the amplitude and the approach to the asymptotic exponent but not the
    exponent itself.
    """
    require_generator(rng)
    n_particles, n_steps, n_dim = cfg.shape
    event_times, event_positions = levy_walk_events(cfg, rng)
    grid = np.arange(n_steps, dtype=np.float64) * DT
    positions = resample_linear(event_times, event_positions, grid)
    return as_trajectories(positions, n_particles, n_steps, n_dim)
