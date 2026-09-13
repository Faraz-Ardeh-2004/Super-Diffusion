"""Unit tests for the MSD estimators and the power-law fit.

These test the *machinery*, not the physics: exact synthetic power laws, exact
straight-line trajectories, and analytically known MSDs. If a generator
validation fails, these tests say whether the fitter is to blame.
"""

from __future__ import annotations

import numpy as np
import pytest

from collectivediff.config import BrownianConfig, LevyFlightConfig
from collectivediff.estimators import (
    DivergentMomentError,
    check_layout,
    ea_msd,
    fit_powerlaw,
    fractional_moment,
)


class TestFitPowerLaw:
    """Recovery of a known exponent and prefactor."""

    @pytest.mark.parametrize("exponent", [0.25, 0.5, 1.0, 1.5, 2.0])
    @pytest.mark.parametrize("prefactor", [0.3, 7.0])
    def test_exact_power_law_is_recovered_to_machine_precision(
        self, exponent: float, prefactor: float
    ) -> None:
        """A noiseless power law is a straight line in log-log; the fit is exact."""
        x = np.logspace(0, 3, 64)
        y = prefactor * x**exponent
        fit = fit_powerlaw(x, y, 1.0, 1000.0)
        assert fit.exponent == pytest.approx(exponent, abs=1e-12)
        assert fit.prefactor == pytest.approx(prefactor, rel=1e-12)
        assert fit.exponent_err == pytest.approx(0.0, abs=1e-9)
        assert fit.n_points == 64

    def test_window_is_respected_and_reported(self) -> None:
        """Only points inside ``[x_min, x_max]`` enter the fit."""
        x = np.arange(1.0, 101.0)
        y = x**1.5
        fit = fit_powerlaw(x, y, 10.0, 50.0)
        assert fit.window == (10.0, 50.0)
        assert fit.n_points == 41
        assert fit.exponent == pytest.approx(1.5, abs=1e-12)

    def test_window_selects_the_right_branch_of_a_broken_power_law(self) -> None:
        """The window is the whole point: a crossover is resolved, not averaged.

        ``y = x`` below 100 and ``y ~ x^2`` above it. Fitting the full range
        would return something in between and be meaningless.
        """
        x = np.logspace(0, 4, 200)
        y = np.where(x < 100.0, x, x**2 / 100.0)
        assert fit_powerlaw(x, y, 1.0, 90.0).exponent == pytest.approx(1.0, abs=1e-9)
        assert fit_powerlaw(x, y, 110.0, 1e4).exponent == pytest.approx(2.0, abs=1e-9)

    def test_non_positive_points_are_dropped(self) -> None:
        """``MSD(0) = 0`` exactly and has no logarithm; it must not poison the fit."""
        x = np.arange(0.0, 51.0)
        y = x**0.8
        fit = fit_powerlaw(x, y, None, None)
        assert fit.n_points == 50
        assert fit.exponent == pytest.approx(0.8, abs=1e-12)

    def test_weights_shift_the_fit_towards_the_weighted_points(self) -> None:
        """Weighting is applied, and uniform weights reproduce the unweighted fit."""
        x = np.logspace(0, 2, 40)
        rng = np.random.default_rng(0)
        y = x**1.0 * np.exp(0.1 * rng.standard_normal(40))
        plain = fit_powerlaw(x, y, 1.0, 100.0)
        uniform = fit_powerlaw(x, y, 1.0, 100.0, weights=np.full(40, 3.0))
        assert uniform.exponent == pytest.approx(plain.exponent, abs=1e-12)

        # Weight the first half heavily: the fit must move towards its local slope.
        heavy = np.where(x < 10.0, 1e3, 1.0)
        local = fit_powerlaw(x, y, 1.0, 9.9)
        weighted = fit_powerlaw(x, y, 1.0, 100.0, weights=heavy)
        assert abs(weighted.exponent - local.exponent) < abs(plain.exponent - local.exponent)

    def test_standard_error_grows_with_scatter(self) -> None:
        """More scatter about the same law means a larger reported error."""
        x = np.logspace(0, 2, 50)
        rng = np.random.default_rng(1)
        noise = rng.standard_normal(50)
        quiet = fit_powerlaw(x, x * np.exp(0.02 * noise), 1.0, 100.0)
        loud = fit_powerlaw(x, x * np.exp(0.20 * noise), 1.0, 100.0)
        assert loud.exponent_err > 5.0 * quiet.exponent_err

    def test_too_few_points_raises(self) -> None:
        x = np.array([1.0, 2.0, 3.0, 4.0])
        with pytest.raises(ValueError, match="at least 3"):
            fit_powerlaw(x, x, 3.5, 4.5)

    def test_negative_weights_rejected(self) -> None:
        x = np.logspace(0, 1, 10)
        with pytest.raises(ValueError, match="strictly positive"):
            fit_powerlaw(x, x, None, None, weights=-np.ones(10))


class TestEnsembleMSD:
    """The ensemble MSD on trajectories whose answer is known by hand."""

    def test_ballistic_trajectory(self) -> None:
        """``x(t) = v t`` gives ``MSD(t) = v^2 t^2`` exactly, so the exponent is 2."""
        t = np.arange(64, dtype=np.float64)
        x = (3.0 * t)[None, :, None]
        msd = ea_msd(x)
        np.testing.assert_allclose(msd, 9.0 * t**2)
        assert fit_powerlaw(t, msd, 1.0, 63.0).exponent == pytest.approx(2.0, abs=1e-12)

    def test_msd_is_measured_from_the_initial_position(self) -> None:
        """Shifting a whole trajectory leaves the MSD unchanged."""
        rng = np.random.default_rng(0)
        x = rng.standard_normal((5, 20, 2))
        np.testing.assert_allclose(ea_msd(x), ea_msd(x + 17.0))
        assert ea_msd(x)[0] == 0.0

    def test_two_dimensional_msd_sums_components(self) -> None:
        t = np.arange(10, dtype=np.float64)
        x = np.stack([3.0 * t, 4.0 * t], axis=-1)[None]
        np.testing.assert_allclose(ea_msd(x), 25.0 * t**2)

    def test_refuses_to_compute_a_divergent_msd(self) -> None:
        """A Levy flight has no second moment; the MSD path must not return a number."""
        x = np.zeros((3, 10, 1))
        cfg = LevyFlightConfig(n_particles=3, n_steps=10, stability=1.5)
        with pytest.raises(DivergentMomentError, match="diverges"):
            ea_msd(x, cfg)
        # A mechanism with a finite second moment passes through untouched.
        assert ea_msd(x, BrownianConfig(n_particles=3, n_steps=10)).shape == (10,)

    def test_layout_is_enforced(self) -> None:
        """Shape and dimension are checked; both float widths are allowed.

        ``float32`` is what comes back from the cache and ``float64`` is what a
        generator produces, so both pass (``PROJECT.md`` section 2). Integer
        arrays do not: a squared displacement would overflow silently.
        """
        with pytest.raises(ValueError, match="n_particles, n_steps, n_dim"):
            check_layout(np.zeros((10, 3)))
        with pytest.raises(ValueError, match="n_dim"):
            check_layout(np.zeros((2, 3, 5)))
        with pytest.raises(ValueError, match="float32 or float64"):
            check_layout(np.zeros((2, 3, 1), dtype=np.int64))

        for dtype in (np.float32, np.float64):
            assert check_layout(np.zeros((2, 3, 1), dtype=dtype)).shape == (2, 3, 1)

    def test_accumulation_is_float64_whatever_the_input(self) -> None:
        """Cached float32 input still accumulates in float64.

        The convention exists because single precision is ample for holding a
        coordinate but not for summing millions of squares of one.
        """
        t = np.arange(64, dtype=np.float64)
        exact = (3.0 * t)[None, :, None]
        for dtype in (np.float32, np.float64):
            msd = ea_msd(exact.astype(dtype))
            assert msd.dtype == np.float64
            np.testing.assert_allclose(msd, 9.0 * t**2, rtol=1e-6)


class TestFractionalMoment:
    """The observable that replaces the MSD when the second moment does not exist."""

    def test_q_equals_two_reproduces_the_msd(self) -> None:
        rng = np.random.default_rng(2)
        x = rng.standard_normal((50, 30, 2))
        np.testing.assert_allclose(fractional_moment(x, 2.0), ea_msd(x))

    def test_ballistic_scaling(self) -> None:
        """``|x| = v t`` gives ``<|x|^q> = v^q t^q``, so ``nu(q) = 1`` for all q."""
        t = np.arange(1, 64, dtype=np.float64)
        x = np.concatenate([[0.0], 2.0 * t])[None, :, None]
        tt = np.arange(64, dtype=np.float64)
        for q in (0.5, 1.0, 3.0):
            fit = fit_powerlaw(tt, fractional_moment(x, q), 1.0, 63.0)
            assert fit.exponent / q == pytest.approx(1.0, abs=1e-12)

    def test_rejects_non_positive_order(self) -> None:
        with pytest.raises(ValueError, match="q must be"):
            fractional_moment(np.zeros((2, 3, 1)), 0.0)
