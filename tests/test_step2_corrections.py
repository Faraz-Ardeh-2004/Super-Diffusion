"""Tests for the corrections requested in ``STEP2_REVIEW.md``.

One class per numbered correction, plus the two changes that follow from the
updated ``PROJECT.md`` rather than from the review list: ``float32`` cached
arrays and the isotropic two-dimensional Levy flight.
"""

from __future__ import annotations

import numpy as np
import pytest

from collectivediff.config import (
    BrownianConfig,
    CTRWConfig,
    FBMConfig,
    FitConfig,
    LevyFlightConfig,
    LevyWalkConfig,
)
from collectivediff.estimators import ea_msd, fit_powerlaw, squared_displacement
from collectivediff.generators import generate
from collectivediff.generators.levy import (
    isotropic_stable,
    positive_stable,
    symmetric_stable,
)
from collectivediff.validation import (
    ctrw_finite_time_msd,
    ctrw_predicted_bias,
    ctrw_renewal_count,
)

# See pyproject.toml: excluded from the default `pytest` invocation.
# Run `pytest -m slow` for this file's validation tests.
pytestmark = pytest.mark.slow


def run(cfg) -> np.ndarray:
    return generate(cfg, np.random.default_rng(cfg.seed))


class TestBootstrapUncertainty:
    """Correction 2: one resampling mechanism supplies both the error and the weights."""

    def test_bootstrap_error_far_exceeds_the_ols_error(self) -> None:
        """The whole point: OLS residuals understate the exponent uncertainty.

        EA-MSD points at different times are built from the same particles, so
        they are strongly autocorrelated and the residual scatter about a fitted
        line is much smaller than the run-to-run spread of the line itself.
        Measured ratio at this configuration is about 60; asserted above 5,
        which no correctly-behaving pair of estimates could satisfy by accident.
        """
        cfg = CTRWConfig(n_particles=2000, n_steps=2048, alpha_wait=0.7, seed=0)
        t = np.arange(cfg.n_steps, dtype=np.float64)
        fit = fit_powerlaw(
            t, squared_displacement(run(cfg), cfg), cfg.duration / 10.0, cfg.duration
        )
        assert fit.bootstrap_err is not None
        assert fit.bootstrap_err > 5.0 * fit.exponent_err

    def test_bootstrap_error_matches_the_seed_to_seed_spread(self) -> None:
        """The bootstrap must reproduce what repeating the experiment would show.

        This is the assertion that makes the error bar trustworthy rather than
        merely large. Ten independent seeds give a spread of the fitted exponent;
        the bootstrap standard error from a *single* seed must agree with it to
        within a factor of two -- the sampling error of a standard deviation
        from ten samples is already 24 per cent.
        """
        exponents, bootstrap_errors = [], []
        for seed in range(10):
            cfg = BrownianConfig(n_particles=1000, n_steps=1024, seed=seed)
            t = np.arange(cfg.n_steps, dtype=np.float64)
            fit = fit_powerlaw(t, squared_displacement(run(cfg), cfg), 1.0, cfg.duration)
            exponents.append(fit.exponent)
            bootstrap_errors.append(fit.bootstrap_err)

        seed_spread = float(np.std(exponents, ddof=1))
        typical = float(np.median(bootstrap_errors))
        assert 0.5 < typical / seed_spread < 2.0

    def test_bootstrap_scales_as_one_over_sqrt_N(self) -> None:
        """Correction 3, on the bootstrap error rather than the seed spread.

        Quadrupling the ensemble must halve the error bar. Verified separately
        on the seed-to-seed spread itself: ``SD * sqrt(N)`` is flat to within
        10 per cent over N from 500 to 8000 for Brownian motion, fBm and CTRW,
        which is the scatter expected of a standard deviation estimated from
        24 seeds. A departure here would mean the uncertainty is structural
        rather than sampling.
        """
        errors = {}
        for n_particles in (500, 8000):
            cfg = BrownianConfig(n_particles=n_particles, n_steps=1024, seed=0)
            t = np.arange(cfg.n_steps, dtype=np.float64)
            fit = fit_powerlaw(t, squared_displacement(run(cfg), cfg), 1.0, cfg.duration)
            errors[n_particles] = fit.bootstrap_err

        ratio = errors[500] / errors[8000]
        assert ratio == pytest.approx(4.0, rel=0.35)

    def test_gls_weighting_is_applied_and_reported(self) -> None:
        """Weights come from the bootstrap variance, and the choice is recorded."""
        cfg = LevyWalkConfig(n_particles=1500, n_steps=2048, gamma=1.5, seed=0)
        t = np.arange(cfg.n_steps, dtype=np.float64)
        per_particle = squared_displacement(run(cfg), cfg)
        gls = fit_powerlaw(t, per_particle, 40.0, cfg.duration)
        plain = fit_powerlaw(
            t, per_particle, 40.0, cfg.duration, fit=FitConfig(use_gls=False)
        )
        assert gls.weighting == "gls"
        assert plain.weighting == "uniform"
        assert gls.bootstrap_err is not None and plain.bootstrap_err is not None

    def test_one_dimensional_input_gets_no_bootstrap(self) -> None:
        """A synthetic curve has no ensemble behind it, and says so."""
        x = np.logspace(0, 2, 40)
        fit = fit_powerlaw(x, x**1.5, 1.0, 100.0)
        assert fit.bootstrap_err is None
        assert fit.bootstrap_ci is None
        assert fit.n_bootstrap == 0
        assert fit.uncertainty == fit.exponent_err

    def test_uncertainty_property_prefers_the_bootstrap(self) -> None:
        cfg = BrownianConfig(n_particles=400, n_steps=512, seed=0)
        t = np.arange(cfg.n_steps, dtype=np.float64)
        fit = fit_powerlaw(t, squared_displacement(run(cfg), cfg), 1.0, cfg.duration)
        assert fit.uncertainty == fit.bootstrap_err

    def test_bootstrap_is_reproducible_from_the_config(self) -> None:
        """An error bar is reproducible from ``(config, seed)`` like everything else."""
        cfg = BrownianConfig(n_particles=400, n_steps=512, seed=0)
        t = np.arange(cfg.n_steps, dtype=np.float64)
        per_particle = squared_displacement(run(cfg), cfg)
        first = fit_powerlaw(t, per_particle, 1.0, cfg.duration)
        again = fit_powerlaw(t, per_particle, 1.0, cfg.duration)
        different = fit_powerlaw(
            t, per_particle, 1.0, cfg.duration, fit=FitConfig(bootstrap_seed=1)
        )
        assert first.bootstrap_err == again.bootstrap_err
        assert first.bootstrap_err != different.bootstrap_err

    def test_confidence_interval_brackets_the_estimate(self) -> None:
        cfg = BrownianConfig(n_particles=1000, n_steps=1024, seed=0)
        t = np.arange(cfg.n_steps, dtype=np.float64)
        fit = fit_powerlaw(t, squared_displacement(run(cfg), cfg), 1.0, cfg.duration)
        low, high = fit.bootstrap_ci
        assert low < fit.exponent < high
        assert low < 1.0 < high


class TestCTRWFiniteTimeCorrection:
    """Correction 4: the bias is corrected in closed form, not merely avoided."""

    def test_predicted_bias_matches_the_measured_one(self) -> None:
        """Closed form against measurement at N = 20000, T = 4096, window [T/10, T].

        Predicted and measured, respectively: ``+0.046`` and ``+0.040`` at
        ``a = 0.3``, ``+0.015`` and ``+0.015`` at ``a = 0.5``, ``-0.013`` and
        ``-0.019`` at ``a = 0.7``. Agreement to about 0.006, which is the
        bootstrap error on the measurement, so the correction is limited by the
        measurement rather than by the expansion.
        """
        for alpha_wait in (0.3, 0.5, 0.7):
            cfg = CTRWConfig(
                n_particles=20000, n_steps=4096, alpha_wait=alpha_wait, seed=0
            )
            t = np.arange(cfg.n_steps, dtype=np.float64)
            fit = fit_powerlaw(
                t,
                squared_displacement(run(cfg), cfg),
                cfg.duration / 10.0,
                cfg.duration,
                fit=FitConfig(n_bootstrap=50),
            )
            predicted = ctrw_predicted_bias(cfg, cfg.duration / 10.0, cfg.duration)
            measured = fit.exponent - alpha_wait
            assert predicted == pytest.approx(measured, abs=0.015)
            # Correcting by the prediction lands on the true exponent.
            assert fit.exponent - predicted == pytest.approx(alpha_wait, abs=0.02)

    def test_bias_reverses_sign_with_alpha(self) -> None:
        """Positive at small ``a``, negative as ``a -> 1``, crossing near 0.60.

        The additive ``-1`` in the renewal count dominates when ``<n(t)>`` is
        only of order ten events and biases the exponent up; the slowly decaying
        ``(tau0/t)^(1-a)`` term dominates as ``a -> 1`` and biases it down.
        """
        window = (409.6, 4095.0)
        biases = [
            ctrw_predicted_bias(
                CTRWConfig(n_steps=4096, alpha_wait=a), window[0], window[1]
            )
            for a in (0.3, 0.5, 0.7, 0.9)
        ]
        assert biases[0] > 0.0 and biases[1] > 0.0
        assert biases[2] < 0.0 and biases[3] < 0.0
        assert all(np.diff(biases[:3]) < 0.0)

    def test_bias_does_not_simply_shrink_with_T(self) -> None:
        """A naive "the bias vanishes as T grows" check would be wrong.

        Because the two corrections pull in opposite directions and decay at
        different rates, the bias is not monotone in ``T`` at fixed ``a``. At
        ``a = 0.5`` it falls sharply between T = 4096 and T = 65536; at
        ``a = 0.8`` it barely moves. Anything asserting uniform decay would fail
        for the wrong reason.
        """
        def bias(alpha_wait: float, n_steps: int) -> float:
            cfg = CTRWConfig(n_steps=n_steps, alpha_wait=alpha_wait)
            return ctrw_predicted_bias(cfg, cfg.duration / 10.0, cfg.duration)

        fast = abs(bias(0.5, 4096)) - abs(bias(0.5, 65536))
        slow = abs(bias(0.8, 4096)) - abs(bias(0.8, 65536))
        assert fast > 0.008
        assert slow < 0.008

    def test_renewal_count_reproduces_the_measured_count(self) -> None:
        """The correction is validated against the renewal process itself.

        Measuring ``<n(t)>`` directly separates the waiting-time machinery from
        the jumps, so a failure here points at the expansion rather than at the
        MSD.
        """
        from collectivediff.generators.ctrw import ctrw_events
        from collectivediff.generators.resampling import _last_event_index

        for alpha_wait in (0.3, 0.7):
            cfg = CTRWConfig(
                n_particles=8000, n_steps=4096, alpha_wait=alpha_wait, seed=0
            )
            event_times, _ = ctrw_events(cfg, np.random.default_rng(0))
            grid = np.arange(cfg.n_steps, dtype=np.float64)
            measured = _last_event_index(event_times, grid).astype(np.float64).mean(axis=0)
            predicted = ctrw_renewal_count(cfg, grid)
            for t_index in (500, 1500, 4000):
                assert measured[t_index] == pytest.approx(
                    predicted[t_index], rel=0.05
                )

    def test_finite_time_msd_is_the_renewal_count_times_the_jump_variance(self) -> None:
        cfg = CTRWConfig(n_dim=1, jump_scale=2.0, alpha_wait=0.6)
        t = np.array([100.0, 1000.0])
        np.testing.assert_allclose(
            ctrw_finite_time_msd(cfg, t), 4.0 * ctrw_renewal_count(cfg, t)
        )

    def test_expansion_is_documented_as_unreliable_near_alpha_one(self) -> None:
        """At ``a = 0.9`` the closed form and the measurement diverge.

        Recorded as a test so the limitation cannot be forgotten: higher-order
        terms matter there, and the correction must not be applied above about
        ``a = 0.8``.
        """
        cfg = CTRWConfig(n_particles=4000, n_steps=4096, alpha_wait=0.9, seed=0)
        t = np.arange(cfg.n_steps, dtype=np.float64)
        measured = ea_msd(run(cfg), cfg)
        predicted = ctrw_finite_time_msd(cfg, t)
        ratio = measured[-1] / predicted[-1]
        assert ratio > 1.15


class TestCTRWNewParameters:
    """Correction 5: jump distribution and aging time exist, defaults unchanged."""

    def test_gaussian_jumps_are_the_default_and_the_only_option(self) -> None:
        assert CTRWConfig().jump_distribution == "gaussian"
        with pytest.raises(ValueError, match="jump_distribution"):
            CTRWConfig(jump_distribution="levy")

    def test_aging_defaults_to_zero_and_changes_nothing(self) -> None:
        """The default must reproduce the non-aged CTRW bit for bit.

        Otherwise adding the parameter would silently invalidate every result
        measured before it existed.
        """
        base = CTRWConfig(n_particles=200, n_steps=512, alpha_wait=0.7, seed=0)
        explicit = CTRWConfig(
            n_particles=200, n_steps=512, alpha_wait=0.7, aging_time=0.0, seed=0
        )
        assert base.aging_time == 0.0
        assert not base.is_aged
        np.testing.assert_array_equal(run(base), run(explicit))

    def test_aging_time_enters_the_cache_key(self) -> None:
        """Two ages must not share a cache entry."""
        from collectivediff.cache import config_hash

        assert config_hash(CTRWConfig(aging_time=0.0)) != config_hash(
            CTRWConfig(aging_time=10.0)
        )

    def test_aged_walker_moves_less(self) -> None:
        """An observer arriving late finds a walker deeper in a trapping event.

        The aged ensemble is dominated by long waiting times drawn before
        observation began, so both the MSD over the observed window and the
        fraction of moving steps fall as ``t_a`` grows. Measured at
        ``a = 0.7``, N = 3000, T = 1024: MSD at the end of the window falls from
        49.4 through 44.5 to 30.7 as ``t_a`` goes 0, 100, 1000, while the
        immobile fraction rises from 0.950 to 0.970.

        Phase 1 validates only the non-aged case; this test exists so that the
        parameter is known to be wired through rather than merely accepted.
        """
        msds, immobile = [], []
        for aging_time in (0.0, 100.0, 1000.0):
            cfg = CTRWConfig(
                n_particles=3000,
                n_steps=1024,
                alpha_wait=0.7,
                aging_time=aging_time,
                seed=1,
            )
            x = run(cfg)
            msds.append(ea_msd(x, cfg)[-1])
            immobile.append(float(np.mean(np.diff(x[:, :, 0], axis=1) == 0.0)))

        assert msds[0] > msds[1] > msds[2]
        assert immobile[0] < immobile[1] < immobile[2]
        assert msds[2] < 0.75 * msds[0]

    def test_aged_trajectories_still_start_at_the_origin(self) -> None:
        """Positions are measured from where the walker was when watching began."""
        cfg = CTRWConfig(n_particles=64, n_steps=128, aging_time=50.0, seed=0)
        np.testing.assert_array_equal(run(cfg)[:, 0, :], 0.0)

    def test_negative_aging_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="aging_time"):
            CTRWConfig(aging_time=-1.0)


class TestIsotropicLevyFlight:
    """PROJECT.md section 2: the stable law is not separable, so 2-D must be explicit."""

    @pytest.mark.parametrize("alpha", [0.25, 0.5, 0.75, 0.9])
    def test_positive_stable_has_the_right_laplace_transform(self, alpha: float) -> None:
        """``E[exp(-lambda A)] = exp(-lambda^alpha)``, checked at several lambda.

        Tolerance 0.005 from two million samples; measured worst deviation over
        this grid is 0.0005.
        """
        sample = positive_stable(np.random.default_rng(0), (2_000_000,), alpha)
        assert np.all(sample > 0.0)
        for lam in (0.1, 0.5, 1.0, 2.0, 5.0):
            assert float(np.mean(np.exp(-lam * sample))) == pytest.approx(
                float(np.exp(-(lam**alpha))), abs=0.005
            )

    @pytest.mark.parametrize("stability", [0.8, 1.2, 1.5, 1.8])
    def test_marginal_matches_the_one_dimensional_stable_law(
        self, stability: float
    ) -> None:
        """The sub-Gaussian vector has exactly the 1-D marginal of ``symmetric_stable``.

        Compared by absolute-value quantiles, which are finite for a stable law
        where moments are not. Tolerance 5 per cent on each quantile from
        200000 samples; measured worst is about 2 per cent.
        """
        rng = np.random.default_rng(0)
        vector = isotropic_stable(rng, (200_000,), stability, 1.0, n_dim=2)
        reference = symmetric_stable(rng, (200_000,), stability, 1.0)
        probs = [10, 25, 50, 75, 90]
        np.testing.assert_allclose(
            np.percentile(np.abs(vector[:, 0]), probs),
            np.percentile(np.abs(reference), probs),
            rtol=0.05,
        )

    def test_two_dimensional_flight_is_isotropic(self) -> None:
        """Long jumps go in every direction equally.

        The test that the component-wise construction fails. Taking the
        99th-percentile jumps and histogramming their angles into twelve bins,
        the relative spread of the bin counts is about 0.05 for the isotropic
        law against about 0.70 for independent components -- the latter sends
        its long jumps along the coordinate axes.
        """
        cfg = LevyFlightConfig(
            n_particles=3000, n_steps=256, stability=1.5, n_dim=2, seed=0
        )
        steps = np.diff(run(cfg), axis=1).reshape(-1, 2)
        radius = np.linalg.norm(steps, axis=1)
        tail = steps[radius > np.percentile(radius, 99.0)]
        counts, _ = np.histogram(
            np.arctan2(tail[:, 1], tail[:, 0]), bins=12, range=(-np.pi, np.pi)
        )
        assert counts.std() / counts.mean() < 0.20

        rng = np.random.default_rng(0)
        component_wise = np.stack(
            [symmetric_stable(rng, (200_000,), 1.5, 1.0) for _ in range(2)], axis=1
        )
        radius = np.linalg.norm(component_wise, axis=1)
        tail = component_wise[radius > np.percentile(radius, 99.0)]
        naive, _ = np.histogram(
            np.arctan2(tail[:, 1], tail[:, 0]), bins=12, range=(-np.pi, np.pi)
        )
        assert naive.std() / naive.mean() > 0.4

    def test_one_dimensional_case_is_unchanged(self) -> None:
        """In 1-D isotropy is vacuous and the cheaper sampler is used."""
        rng_a = np.random.default_rng(3)
        rng_b = np.random.default_rng(3)
        np.testing.assert_array_equal(
            isotropic_stable(rng_a, (1000,), 1.5, 2.0, n_dim=1)[:, 0],
            symmetric_stable(rng_b, (1000,), 1.5, 2.0),
        )

    def test_separable_mechanisms_are_untouched(self) -> None:
        """Brownian motion and fBm *are* separable, so 2-D stays componentwise.

        PROJECT.md draws the distinction explicitly; this pins it down so that
        a future change to the Levy path does not leak into the Gaussian ones.
        """
        for cfg in (
            BrownianConfig(n_particles=4000, n_steps=256, n_dim=2, seed=0),
            FBMConfig(n_particles=4000, n_steps=256, n_dim=2, hurst=0.7, seed=0),
        ):
            steps = np.diff(run(cfg), axis=1).reshape(-1, 2)
            correlation = np.corrcoef(steps[:, 0], steps[:, 1])[0, 1]
            assert abs(correlation) < 0.02


class TestStorageDtype:
    """PROJECT.md section 2: float32 on disk, float64 in the accumulators."""

    def test_estimators_accept_the_cache_dtype(self, tmp_path) -> None:
        """A cached-and-reloaded ensemble gives the same answer to six digits."""
        from collectivediff import cache

        cfg = BrownianConfig(n_particles=500, n_steps=512, seed=0)
        trajectories = run(cfg)
        cache.save_arrays(cfg, {"x": trajectories}, root=tmp_path)
        reloaded = cache.load_arrays(cfg, root=tmp_path)["x"]

        assert reloaded.dtype == np.float32
        direct = ea_msd(trajectories, cfg)
        from_cache = ea_msd(reloaded, cfg)
        assert from_cache.dtype == np.float64
        np.testing.assert_allclose(from_cache[1:], direct[1:], rtol=1e-5)

    def test_fitted_exponent_survives_the_round_trip(self, tmp_path) -> None:
        """Single precision must not move a fitted exponent at the reported precision."""
        from collectivediff import cache

        cfg = FBMConfig(n_particles=1000, n_steps=1024, hurst=0.7, seed=0)
        trajectories = run(cfg)
        cache.save_arrays(cfg, {"x": trajectories}, root=tmp_path)
        reloaded = cache.load_arrays(cfg, root=tmp_path)["x"]

        t = np.arange(cfg.n_steps, dtype=np.float64)
        direct = fit_powerlaw(t, squared_displacement(trajectories, cfg), 1.0, cfg.duration)
        cached = fit_powerlaw(t, squared_displacement(reloaded, cfg), 1.0, cfg.duration)
        assert cached.exponent == pytest.approx(direct.exponent, abs=1e-4)
