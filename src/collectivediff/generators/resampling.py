"""Resampling a continuous-time event sequence onto the uniform grid.

``PROJECT.md`` section 2 requires that generators with intrinsically continuous
time (CTRW, Levy walk) build their raw event sequence first and only then be
sampled onto the uniform grid, by an explicit and separately tested function.
This module is that function, and the two representations never mix elsewhere.

The two mechanisms need genuinely different interpolation, and that difference
*is* the physics:

* **CTRW** -- the walker sits still between jumps and moves instantaneously at
  an event. Between grid points the position is the last event's position: a
  zero-order hold, :func:`resample_step`.
* **Levy walk** -- the walker travels at constant speed *during* a flight, so
  displacement stays coupled to elapsed time and the MSD is finite. Between
  event points the position interpolates linearly:
  :func:`resample_linear`.

Using the step rule for a Levy walk would discard exactly the space-time
coupling that makes it a walk rather than a flight.

Both functions take the same input convention: ``event_times[:, 0] == 0`` and
``event_positions[:, 0, :] == 0``, i.e. the initial condition is the zeroth
event.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

__all__ = ["resample_step", "resample_linear"]


def _validate_events(
    event_times: NDArray[np.float64],
    event_positions: NDArray[np.float64],
    grid: NDArray[np.float64],
) -> None:
    """Check the shared preconditions of the two resamplers.

    Raises
    ------
    ValueError
        If shapes disagree, times are not non-decreasing, the zeroth event is
        not the origin at ``t = 0``, or the event sequence does not cover the
        whole grid for every particle. The last one matters: silently holding
        the last known position past the end of the event sequence would fake a
        long trapping event and bias the MSD downwards.
    """
    if event_times.ndim != 2:
        raise ValueError(f"event_times must be 2-D (n_particles, n_events), got {event_times.shape}")
    if event_positions.ndim != 3 or event_positions.shape[:2] != event_times.shape:
        raise ValueError(
            "event_positions must have shape (n_particles, n_events, n_dim) matching "
            f"event_times {event_times.shape}, got {event_positions.shape}"
        )
    if grid.ndim != 1:
        raise ValueError(f"grid must be 1-D, got shape {grid.shape}")
    if np.any(np.diff(event_times, axis=1) < 0.0):
        raise ValueError("event_times must be non-decreasing along the event axis")
    if not np.allclose(event_times[:, 0], 0.0) or not np.allclose(event_positions[:, 0, :], 0.0):
        raise ValueError("the zeroth event must be the origin at t = 0")
    if np.any(event_times[:, -1] < grid[-1]):
        shortfall = int(np.sum(event_times[:, -1] < grid[-1]))
        raise ValueError(
            f"{shortfall} particle(s) have an event sequence ending before the last "
            f"grid point {grid[-1]:g}; generate more events rather than extrapolating"
        )


def _last_event_index(
    event_times: NDArray[np.float64],
    grid: NDArray[np.float64],
) -> NDArray[np.intp]:
    """Index of the latest event at or before each grid point, per particle.

    Parameters
    ----------
    event_times : ndarray of float64
        Shape ``(n_particles, n_events)``, non-decreasing along axis 1.
    grid : ndarray of float64
        Shape ``(n_steps,)``, increasing.

    Returns
    -------
    ndarray of intp
        Shape ``(n_particles, n_steps)``, values in ``[0, n_events - 1]``.

    Notes
    -----
    ``np.searchsorted`` is 1-D only, so this loops over particles. The
    alternative -- broadcasting a ``(n_particles, n_events, n_steps)`` boolean
    -- is quadratic in memory and was rejected for that reason, not for speed.
    The loop is over a few thousand rows of a cheap C call and has never shown
    up in a profile.
    """
    n_particles = event_times.shape[0]
    index = np.empty((n_particles, grid.size), dtype=np.intp)
    for i in range(n_particles):
        index[i] = np.searchsorted(event_times[i], grid, side="right") - 1
    return np.clip(index, 0, event_times.shape[1] - 1)


def resample_step(
    event_times: NDArray[np.float64],
    event_positions: NDArray[np.float64],
    grid: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Zero-order hold: the position of the last event at or before each time.

    .. math::

        x(t) = x_{n(t)}, \\qquad n(t) = \\max\\{n : T_n \\le t\\}

    This is the CTRW rule: the walker is immobile between jumps, so a long
    waiting time shows up as a flat stretch of trajectory.

    Parameters
    ----------
    event_times : ndarray of float64
        Shape ``(n_particles, n_events)``, non-decreasing, first column zero.
    event_positions : ndarray of float64
        Shape ``(n_particles, n_events, n_dim)``, position *after* each event,
        first row the origin.
    grid : ndarray of float64
        Uniform grid, shape ``(n_steps,)``, starting at 0.

    Returns
    -------
    ndarray of float64
        Shape ``(n_particles, n_steps, n_dim)``.
    """
    _validate_events(event_times, event_positions, grid)
    index = _last_event_index(event_times, grid)
    return np.take_along_axis(event_positions, index[:, :, None], axis=1)


def resample_linear(
    event_times: NDArray[np.float64],
    event_positions: NDArray[np.float64],
    grid: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Linear interpolation between consecutive event points.

    .. math::

        x(t) = x_n + (x_{n+1} - x_n)\\,\\frac{t - T_n}{T_{n+1} - T_n},
        \\qquad T_n \\le t < T_{n+1}

    This is the Levy-walk rule: the walker moves at constant speed along the
    current flight, so a grid point falling inside a long flight lands part-way
    along it rather than at its start.

    Parameters
    ----------
    event_times : ndarray of float64
        Shape ``(n_particles, n_events)``, non-decreasing, first column zero.
    event_positions : ndarray of float64
        Shape ``(n_particles, n_events, n_dim)``, position at each event time,
        first row the origin.
    grid : ndarray of float64
        Uniform grid, shape ``(n_steps,)``, starting at 0.

    Returns
    -------
    ndarray of float64
        Shape ``(n_particles, n_steps, n_dim)``.
    """
    _validate_events(event_times, event_positions, grid)
    n_events = event_times.shape[1]
    lower = np.clip(_last_event_index(event_times, grid), 0, n_events - 2)
    upper = lower + 1

    t_lo = np.take_along_axis(event_times, lower, axis=1)
    t_hi = np.take_along_axis(event_times, upper, axis=1)
    x_lo = np.take_along_axis(event_positions, lower[:, :, None], axis=1)
    x_hi = np.take_along_axis(event_positions, upper[:, :, None], axis=1)

    span = t_hi - t_lo
    # A zero-length interval carries no information about where inside it the
    # grid point lies; fall back to the left endpoint instead of dividing by 0.
    fraction = np.where(span > 0.0, (grid[None, :] - t_lo) / np.where(span > 0.0, span, 1.0), 0.0)
    return x_lo + (x_hi - x_lo) * fraction[:, :, None]
