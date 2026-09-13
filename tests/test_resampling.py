"""Unit tests for the continuous-time to uniform-grid resamplers.

``PROJECT.md`` section 2 requires the event sequence and the gridded trajectory
to be distinct representations joined by an explicit, separately tested
function. These are those tests: they use hand-built event sequences with known
answers, so a failure points at the resampler and not at a generator.
"""

from __future__ import annotations

import numpy as np
import pytest

from collectivediff.generators.resampling import resample_linear, resample_step


def _one_particle(times: list[float], positions: list[float]):
    """Wrap a 1-D event sequence in the ``(n_particles, n_events, n_dim)`` layout."""
    return (
        np.array([times], dtype=np.float64),
        np.array([positions], dtype=np.float64)[:, :, None],
    )


class TestResampleStep:
    """Zero-order hold: the CTRW rule."""

    def test_holds_position_between_events(self) -> None:
        """Between jumps the walker does not move, so the trace is a staircase.

        Events at t = 0, 2.5, 6 with positions 0, 1, -3. On the integer grid
        0..6 the expected trace is 0 0 0 1 1 1 -3: the jump at 2.5 is only
        visible from t = 3 onwards.
        """
        times, positions = _one_particle([0.0, 2.5, 6.0], [0.0, 1.0, -3.0])
        grid = np.arange(7, dtype=np.float64)
        out = resample_step(times, positions, grid)
        expected = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0, -3.0])
        np.testing.assert_allclose(out[0, :, 0], expected)

    def test_event_exactly_on_grid_point_is_included(self) -> None:
        """An event at exactly ``t`` counts at ``t``: the hold is right-continuous."""
        times, positions = _one_particle([0.0, 2.0, 5.0], [0.0, 7.0, 9.0])
        grid = np.arange(6, dtype=np.float64)
        out = resample_step(times, positions, grid)
        np.testing.assert_allclose(out[0, :, 0], [0.0, 0.0, 7.0, 7.0, 7.0, 9.0])

    def test_two_dimensional_and_multi_particle(self) -> None:
        """Both particles and both components are resampled independently."""
        times = np.array([[0.0, 1.0, 4.0], [0.0, 3.0, 4.0]])
        positions = np.array(
            [[[0.0, 0.0], [1.0, 2.0], [5.0, 6.0]], [[0.0, 0.0], [-1.0, 1.0], [0.0, 0.0]]]
        )
        grid = np.arange(5, dtype=np.float64)
        out = resample_step(times, positions, grid)
        np.testing.assert_allclose(out[0, :, 0], [0.0, 1.0, 1.0, 1.0, 5.0])
        np.testing.assert_allclose(out[0, :, 1], [0.0, 2.0, 2.0, 2.0, 6.0])
        np.testing.assert_allclose(out[1, :, 0], [0.0, 0.0, 0.0, -1.0, 0.0])


class TestResampleLinear:
    """Linear interpolation: the Levy-walk rule."""

    def test_interpolates_within_a_flight(self) -> None:
        """A grid point inside a flight lands part-way along it, not at its start.

        One flight from t = 0 to t = 4 covering 0 -> 8 at constant speed gives
        exactly 0, 2, 4, 6, 8 on the integer grid. Under a zero-order hold the
        same events would give 0, 0, 0, 0, 8 -- which is the space-time coupling
        being destroyed, and the reason the two rules are separate functions.
        """
        times, positions = _one_particle([0.0, 4.0], [0.0, 8.0])
        grid = np.arange(5, dtype=np.float64)
        out = resample_linear(times, positions, grid)
        np.testing.assert_allclose(out[0, :, 0], [0.0, 2.0, 4.0, 6.0, 8.0])

    def test_constant_speed_is_preserved_across_turns(self) -> None:
        """Speed stays constant through a direction reversal.

        Flights: 0 -> 3 at +1, then 3 -> 6 at -1. Displacement is
        0 1 2 3 2 1 0, and every step has unit length.
        """
        times, positions = _one_particle([0.0, 3.0, 6.0], [0.0, 3.0, 0.0])
        grid = np.arange(7, dtype=np.float64)
        out = resample_linear(times, positions, grid)
        np.testing.assert_allclose(out[0, :, 0], [0.0, 1.0, 2.0, 3.0, 2.0, 1.0, 0.0])
        np.testing.assert_allclose(np.abs(np.diff(out[0, :, 0])), 1.0)

    def test_matches_step_at_the_turning_points(self) -> None:
        """The two rules agree exactly where a grid point coincides with an event."""
        times, positions = _one_particle([0.0, 2.0, 5.0], [0.0, 4.0, -1.0])
        grid = np.array([0.0, 2.0, 5.0])
        np.testing.assert_allclose(
            resample_linear(times, positions, grid), resample_step(times, positions, grid)
        )

    def test_two_dimensional_speed(self) -> None:
        """In 2-D the interpolated speed equals the flight speed."""
        times = np.array([[0.0, 5.0]])
        positions = np.array([[[0.0, 0.0], [3.0, 4.0]]])  # length 5 in time 5 -> speed 1
        grid = np.arange(6, dtype=np.float64)
        out = resample_linear(times, positions, grid)
        speeds = np.linalg.norm(np.diff(out[0], axis=0), axis=1)
        np.testing.assert_allclose(speeds, 1.0)


class TestValidation:
    """The resamplers refuse malformed or short event sequences."""

    def test_rejects_sequence_that_does_not_reach_the_last_grid_point(self) -> None:
        """Silently holding past the end would fake a long trap and bias the MSD."""
        times, positions = _one_particle([0.0, 2.0], [0.0, 1.0])
        grid = np.arange(6, dtype=np.float64)
        with pytest.raises(ValueError, match="ending before the last"):
            resample_step(times, positions, grid)

    def test_rejects_non_monotonic_event_times(self) -> None:
        times, positions = _one_particle([0.0, 3.0, 1.0], [0.0, 1.0, 2.0])
        with pytest.raises(ValueError, match="non-decreasing"):
            resample_step(times, positions, np.arange(2, dtype=np.float64))

    def test_rejects_non_origin_start(self) -> None:
        times, positions = _one_particle([0.0, 4.0], [1.0, 2.0])
        with pytest.raises(ValueError, match="origin"):
            resample_step(times, positions, np.arange(3, dtype=np.float64))

    def test_rejects_mismatched_shapes(self) -> None:
        times = np.zeros((2, 3))
        times[:, 1:] = [[1.0, 4.0], [1.0, 4.0]]
        with pytest.raises(ValueError, match="event_positions"):
            resample_step(times, np.zeros((2, 5, 1)), np.arange(3, dtype=np.float64))

    def test_zero_length_interval_does_not_divide_by_zero(self) -> None:
        """Duplicate event times are degenerate but must not produce NaN."""
        times, positions = _one_particle([0.0, 2.0, 2.0, 5.0], [0.0, 1.0, 3.0, 4.0])
        out = resample_linear(times, positions, np.arange(6, dtype=np.float64))
        assert np.all(np.isfinite(out))
