"""Helpers shared by the generators: layout enforcement and random directions.

Nothing here is a mechanism. It exists so that every generator returns the same
thing in the same shape and draws its randomness the same way.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "as_trajectories",
    "pareto_durations_covering",
    "random_directions",
    "require_generator",
]


def require_generator(rng: np.random.Generator) -> np.random.Generator:
    """Reject anything that is not an explicit ``numpy`` Generator.

    ``PROJECT.md`` section 2 forbids global ``np.random.*`` calls. Legacy
    ``RandomState`` objects are rejected too: they draw from a different stream,
    so a figure made with one would not be reproducible from ``(config, seed)``.

    Parameters
    ----------
    rng : numpy.random.Generator
        Candidate generator.

    Returns
    -------
    numpy.random.Generator
        The same object.

    Raises
    ------
    TypeError
        If ``rng`` is not a :class:`numpy.random.Generator`.
    """
    if not isinstance(rng, np.random.Generator):
        raise TypeError(
            "every stochastic function takes an explicit np.random.Generator "
            f"(PROJECT.md section 2), got {type(rng).__name__}"
        )
    return rng


def as_trajectories(
    positions: NDArray[np.float64],
    n_particles: int,
    n_steps: int,
    n_dim: int,
) -> NDArray[np.float64]:
    """Enforce the project's array layout on a generator's output.

    Parameters
    ----------
    positions : ndarray
        Candidate output.
    n_particles, n_steps, n_dim : int
        Expected shape, normally taken straight from ``cfg.shape``.

    Returns
    -------
    ndarray of float64
        C-contiguous array of shape ``(n_particles, n_steps, n_dim)`` starting
        at the origin.

    Raises
    ------
    ValueError
        If the shape is wrong or the trajectories do not start at the origin.
        Both are cheap to check and expensive to debug later.
    """
    expected = (n_particles, n_steps, n_dim)
    if positions.shape != expected:
        raise ValueError(f"generator produced shape {positions.shape}, expected {expected}")
    out = np.ascontiguousarray(positions, dtype=np.float64)
    if not np.allclose(out[:, 0, :], 0.0):
        raise ValueError("trajectories must start at the origin")
    if not np.all(np.isfinite(out)):
        raise ValueError("generator produced non-finite positions")
    return out


def random_directions(
    rng: np.random.Generator,
    size: tuple[int, ...],
    n_dim: int,
) -> NDArray[np.float64]:
    """Draw unit vectors uniformly on the unit sphere in ``n_dim`` dimensions.

    In one dimension this is a fair coin on ``{-1, +1}``; in two it is a uniform
    angle. Used by the Levy walk, where each flight has a random direction and a
    heavy-tailed duration.

    Parameters
    ----------
    rng : numpy.random.Generator
        Explicit generator.
    size : tuple of int
        Leading shape of the output, e.g. ``(n_particles, n_flights)``.
    n_dim : int
        1 or 2.

    Returns
    -------
    ndarray of float64
        Shape ``size + (n_dim,)``, every row of unit Euclidean norm.
    """
    require_generator(rng)
    if n_dim == 1:
        signs = rng.integers(0, 2, size=size).astype(np.float64) * 2.0 - 1.0
        return signs[..., None]
    if n_dim == 2:
        angle = rng.uniform(0.0, 2.0 * np.pi, size=size)
        return np.stack([np.cos(angle), np.sin(angle)], axis=-1)
    raise ValueError(f"n_dim must be 1 or 2, got {n_dim}")


def _expected_event_count(duration: float, tail_exponent: float, scale: float) -> float:
    """Mean number of Pareto events needed to reach ``duration``.

    For the tail ``P(tau > t) = (t / scale)^(-a)``:

    * ``a > 1`` -- the mean waiting time ``a scale / (a - 1)`` is finite, so the
      count grows linearly, ``<n(t)> ~ t (a - 1) / (a scale)``.
    * ``a < 1`` -- the mean diverges and renewal theory gives the sublinear
      ``<n(t)> ~ sin(pi a) / (pi a) * (t / scale)^a``.

    This is only used to size the first allocation; the caller keeps drawing
    until every particle is genuinely covered, so an underestimate costs a
    round rather than correctness.
    """
    if tail_exponent > 1.0:
        mean_wait = tail_exponent * scale / (tail_exponent - 1.0)
        return duration / mean_wait
    ratio = np.sin(np.pi * tail_exponent) / (np.pi * tail_exponent)
    return float(ratio * (duration / scale) ** tail_exponent)


def pareto_durations_covering(
    rng: np.random.Generator,
    n_particles: int,
    duration: float,
    tail_exponent: float,
    scale: float,
    oversample: float = 2.0,
    max_rounds: int = 64,
) -> NDArray[np.float64]:
    """Draw Pareto-tailed durations until every particle's clock passes ``duration``.

    Durations follow

    .. math::

        P(\\tau > t) = (t / \\tau_0)^{-a}, \\qquad t \\ge \\tau_0 ,

    sampled by inversion, ``tau = tau0 * U^(-1/a)`` with ``U`` uniform on
    ``(0, 1]``. Shared by the CTRW (``a < 1``, waiting times) and the Levy walk
    (``a > 1``, flight durations).

    The number of events needed is itself a heavy-tailed random variable, so a
    fixed allocation would truncate some particles' event sequences. Drawing
    proceeds in rounds, doubling the block size, until *every* particle is
    covered. Truncation is never accepted: for a CTRW it would cap the longest
    waiting times, which are exactly what produces the subdiffusion.

    Parameters
    ----------
    rng : numpy.random.Generator
        Explicit generator.
    n_particles : int
        Number of independent event sequences.
    duration : float
        Time that must be covered, i.e. the last grid point.
    tail_exponent : float
        ``a``, strictly positive.
    scale : float
        ``tau0``, the lower cut-off of the tail.
    oversample : float
        Safety factor on the initial allocation.
    max_rounds : int
        Guard against a runaway loop; exceeding it raises.

    Returns
    -------
    ndarray of float64
        Shape ``(n_particles, n_events)``, with
        ``durations.sum(axis=1) >= duration`` for every row.

    Raises
    ------
    RuntimeError
        If ``max_rounds`` rounds still leave a particle uncovered.
    """
    require_generator(rng)
    estimate = _expected_event_count(duration, tail_exponent, scale)
    block = max(int(np.ceil(oversample * estimate)) + 8, 16)

    durations = np.empty((n_particles, 0), dtype=np.float64)
    total = np.zeros(n_particles, dtype=np.float64)
    for _ in range(max_rounds):
        new = scale * (1.0 - rng.random((n_particles, block))) ** (-1.0 / tail_exponent)
        durations = np.concatenate([durations, new], axis=1)
        total += new.sum(axis=1)
        if np.all(total >= duration):
            return durations
        block *= 2
    raise RuntimeError(
        f"could not cover duration {duration} for all {n_particles} particles in "
        f"{max_rounds} rounds with tail exponent {tail_exponent}"
    )
