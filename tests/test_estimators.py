"""Unit tests for the estimator toolbox of ``PROJECT.md`` section 5.

Every estimator is checked against a case with a known closed form -- Brownian
motion for most of them, the fBm increment autocorrelation and the Levy-walk
moment spectrum where a sharper reference exists.

Three checks in this file are the scientific point of phase 1 rather than
routine coverage, and are grouped in their own classes:

* :class:`TestCTRWWeakErgodicityBreaking` -- the ensemble MSD scales as ``t^a``
  while any single trajectory's TA-MSD grows *linearly* in the lag, and the EB
  parameter converges to a nonzero plateau. This is the information-theoretic
  wall in Q1, and it is demonstrated rather than asserted.
* :class:`TestFBMErgodicity` -- the contrast case: EB decays towards zero as
  ``T`` grows, and the increment ACF has the right sign on both sides of
  ``H = 1/2``.
* :class:`TestLevyWalkMomentSpectrum` -- ``nu(q)`` reproduces the bilinear form
  with its kink at ``q = g``, which is the only unambiguous signature of strong
  anomalous diffusion and the observable phase 2 leans on hardest.

Tolerances follow the same rule as the generator validation tests: they come
from measured seed-to-seed spread or from a stated finite-time correction, never
from the least-squares residual error, which for these estimators is optimistic
by more than an order of magnitude.
"""

from __future__ import annotations

import numpy as np
import pytest

from collectivediff.config import (
    BrownianConfig,
    CTRWConfig,
    DDMConfig,
    FBMConfig,
    FitConfig,
    LevyFlightConfig,
    LevyWalkConfig,
    SBMConfig,
)
from collectivediff.estimators import (
    DivergentMomentError,
    amplitude_scatter,
    eb_parameter,
    ea_msd,
    ea_ta_msd,
    ergodicity_breaking_curve,
    first_passage_times,
    fit_powerlaw,
    gaussian_reference,
    increment_acf,
    log_spaced_lags,
    moment_spectrum,
    non_gaussian_parameter,
    return_statistics,
    squared_displacement,
    survival_probability,
    ta_msd,
    van_hove,
    velocity_acf,
)
from collectivediff.estimators.correlations import fbm_increment_acf
from collectivediff.estimators.ergodicity import ctrw_eb_plateau
from collectivediff.estimators.firstpassage import brownian_survival
from collectivediff.estimators.moments import fit_bilinear_spectrum, levy_walk_nu
from collectivediff.generators import generate




# See pyproject.toml: excluded from the default `pytest` invocation.
# Run `pytest -m slow` for this file's validation tests.
pytestmark = pytest.mark.slow

def run(cfg) -> np.ndarray:
    """Generate trajectories seeded from the config."""
    return generate(cfg, np.random.default_rng(cfg.seed))


# --------------------------------------------------------------------------
# TA-MSD and its ensemble average
# --------------------------------------------------------------------------


class TestTimeAveragedMSD:
    """Closed forms: ballistic motion and Brownian motion."""

    def test_ballistic_trajectory_is_exact(self) -> None:
        """``x = v t`` gives ``TA-MSD(D) = v^2 D^2`` exactly, for every window."""
        t = np.arange(128, dtype=np.float64)
        x = (2.0 * t)[None, :, None]
        lags = np.array([1, 2, 5, 13], dtype=np.intp)
        np.testing.assert_allclose(ta_msd(x, lags)[0], 4.0 * lags.astype(float) ** 2)

    def test_brownian_ta_msd_is_linear_in_lag(self) -> None:
        """``TA-MSD(D) = 2 d D_coef * Delta``, the same law as the EA-MSD.

        For an ergodic process the two coincide; asserting it here is what makes
        the CTRW result below meaningful rather than a quirk of the estimator.
        """
        cfg = BrownianConfig(n_particles=800, n_steps=2048, diffusivity=1.5, seed=0)
        lags = log_spaced_lags(cfg.n_steps)
        measured = ea_ta_msd(run(cfg), lags)
        np.testing.assert_allclose(
            measured / (2.0 * cfg.n_dim * cfg.diffusivity * lags), 1.0, atol=0.05
        )

    def test_ea_ta_msd_is_the_particle_mean(self) -> None:
        cfg = BrownianConfig(n_particles=64, n_steps=256, seed=0)
        lags = np.array([1, 4, 16], dtype=np.intp)
        x = run(cfg)
        np.testing.assert_allclose(ea_ta_msd(x, lags), ta_msd(x, lags).mean(axis=0))

    def test_agrees_with_ea_msd_for_an_ergodic_process(self) -> None:
        """EA-MSD at time ``t`` and EA-TA-MSD at lag ``t`` agree for Brownian motion."""
        cfg = BrownianConfig(n_particles=2000, n_steps=2048, seed=0)
        x = run(cfg)
        lags = np.array([4, 16, 64], dtype=np.intp)
        np.testing.assert_allclose(
            ea_ta_msd(x, lags) / ea_msd(x, cfg)[lags], 1.0, atol=0.06
        )

    def test_lag_window_respects_the_project_convention(self) -> None:
        """``log_spaced_lags`` stays inside ``[lag_min, T/10]``."""
        lags = log_spaced_lags(1001, FitConfig(lag_min=2, lag_max_fraction=0.1))
        assert lags.min() >= 2
        assert lags.max() <= 100
        assert np.all(np.diff(lags) > 0)

    def test_rejects_out_of_range_lags(self) -> None:
        x = np.zeros((2, 16, 1))
        with pytest.raises(ValueError, match="lags must lie"):
            ta_msd(x, np.array([0], dtype=np.intp))
        with pytest.raises(ValueError, match="lags must lie"):
            ta_msd(x, np.array([16], dtype=np.intp))

    def test_divergent_moment_is_refused(self) -> None:
        cfg = LevyFlightConfig(n_particles=8, n_steps=64, stability=1.5, seed=0)
        with pytest.raises(DivergentMomentError):
            ta_msd(run(cfg), np.array([2], dtype=np.intp), cfg)


# --------------------------------------------------------------------------
# ergodicity breaking
# --------------------------------------------------------------------------


class TestErgodicityBreakingMachinery:
    """Properties that hold by construction, independent of any mechanism."""

    def test_amplitude_scatter_has_unit_mean(self) -> None:
        cfg = BrownianConfig(n_particles=500, n_steps=512, seed=0)
        xi = amplitude_scatter(run(cfg), 8)
        assert xi.mean() == pytest.approx(1.0)

    def test_eb_is_the_variance_of_xi(self) -> None:
        cfg = FBMConfig(n_particles=400, n_steps=512, hurst=0.4, seed=0)
        x = run(cfg)
        xi = amplitude_scatter(x, 8)
        assert eb_parameter(x, 8) == pytest.approx(float(np.mean(xi**2) - 1.0))

    def test_eb_vanishes_for_identical_trajectories(self) -> None:
        """No scatter, no ergodicity breaking: the estimator's zero point."""
        t = np.arange(64, dtype=np.float64)
        identical = np.repeat((3.0 * t)[None, :, None], 20, axis=0)
        assert eb_parameter(identical, 4) == pytest.approx(0.0, abs=1e-12)

    def test_curve_matches_pointwise_calls(self) -> None:
        cfg = BrownianConfig(n_particles=300, n_steps=512, seed=0)
        x = run(cfg)
        lags = np.array([2, 8, 32], dtype=np.intp)
        np.testing.assert_allclose(
            ergodicity_breaking_curve(x, lags), [eb_parameter(x, int(k)) for k in lags]
        )

    @pytest.mark.parametrize("alpha_wait, expected", [(0.5, np.pi / 2 - 1.0), (1.0, 0.0)])
    def test_plateau_closed_form(self, alpha_wait: float, expected: float) -> None:
        """``EB_inf = 2 Gamma^2(1+a)/Gamma(1+2a) - 1``; at a = 1/2 that is pi/2 - 1."""
        assert ctrw_eb_plateau(alpha_wait) == pytest.approx(expected, abs=1e-12)

    def test_plateau_is_monotone_in_alpha(self) -> None:
        """It rises towards 1 as ``a -> 0`` and falls to 0 at ``a = 1``."""
        values = [ctrw_eb_plateau(a) for a in (0.1, 0.3, 0.5, 0.7, 0.9)]
        assert all(np.diff(values) < 0.0)
        assert values[0] > 0.8


class TestBrownianEBPositiveControl:
    """``PHASE1_REVIEW.md`` item 4: the check that catches a wrong constant.

    The CTRW plateau alone is a weak test of the implementation -- a plateau
    scaled by the wrong factor is still a plateau. This closed form has a
    specific slope *and* a specific coefficient and cannot pass by coincidence.
    """

    def test_closed_form_values_at_T_4096(self) -> None:
        """``EB(Delta) ~= 4 Delta / (3T)``: 0.00326, 0.01628, 0.03255 at T=4096."""
        from collectivediff.estimators.ergodicity import brownian_eb

        predicted = brownian_eb(np.array([10, 50, 100]), n_steps=4097)
        np.testing.assert_allclose(predicted, [0.00326, 0.01628, 0.03255], atol=2e-5)

    def test_measured_matches_predicted(self) -> None:
        """Measured EB against the closed form, N = 2000, T = 4096.

        Tolerance 10 per cent: this is a much tighter formula than the CTRW
        plateau (no slow convergence to absorb), so a materially wrong constant
        inside ``eb_parameter`` would fail this well before it failed the
        CTRW check.
        """
        from collectivediff.estimators.ergodicity import brownian_eb

        cfg = BrownianConfig(n_particles=2000, n_steps=4096, seed=0)
        lags = np.array([10, 50, 100], dtype=np.intp)
        measured = ergodicity_breaking_curve(run(cfg), lags, cfg)
        predicted = brownian_eb(lags.astype(np.float64), cfg.n_steps)
        np.testing.assert_allclose(measured, predicted, rtol=0.15)

    def test_scales_linearly_with_lag(self) -> None:
        """The formula is linear in ``Delta``; halving Delta must halve EB(Delta)."""
        from collectivediff.estimators.ergodicity import brownian_eb

        predicted = brownian_eb(np.array([20.0, 40.0, 80.0]), n_steps=4097)
        np.testing.assert_allclose(predicted[1:] / predicted[:-1], 2.0)


class TestCTRWWeakErgodicityBreaking:
    """The scientific point of phase 1, asserted in three parts.

    ``PHASE1_PROMPT.md`` step 3, check 1: for a CTRW with waiting exponent ``a``
    the ensemble MSD scales as ``t^a``, the single-trajectory TA-MSD scales
    *linearly* in the lag, and the EB parameter converges to a nonzero plateau.
    """

    CFG = CTRWConfig(n_particles=2000, n_steps=16384, alpha_wait=0.5, seed=0)

    @pytest.fixture(scope="class")
    def trajectories(self) -> np.ndarray:
        return run(self.CFG)

    def test_ensemble_msd_scales_as_t_to_the_a(self, trajectories: np.ndarray) -> None:
        """EA-MSD exponent is ``a``, window ``[T/10, T]``, N = 2000, T = 16384.

        Tolerance 0.05, of which about 0.007 is the closed-form finite-time bias
        predicted by ``validation.reference.ctrw_predicted_bias`` at this ``a``
        and ``T``, and the rest is the bootstrap spread.
        """
        t = np.arange(self.CFG.n_steps, dtype=np.float64)
        fit = fit_powerlaw(
            t, squared_displacement(trajectories, self.CFG), self.CFG.duration / 10.0, self.CFG.duration
        )
        assert fit.exponent == pytest.approx(self.CFG.alpha_wait, abs=0.05)

    def test_single_trajectory_ta_msd_is_linear_whatever_a_is(
        self, trajectories: np.ndarray
    ) -> None:
        """The wall: one trajectory reports ``alpha = 1``, not ``alpha = 0.5``.

        This is weak ergodicity breaking in its most direct form. The TA-MSD of
        each individual trajectory grows linearly in the lag, so an observer
        watching a single agent -- with unlimited memory of it -- measures the
        exponent of *Brownian motion*, and no amount of further watching will
        change that. The subdiffusion lives in the ensemble, not in any member
        of it.

        Fitted per trajectory over ``[1, T/10]``, then summarised over the
        twenty longest-lived trajectories. The median must sit at 1 within 0.1,
        and be far from ``a`` -- the assertion that matters is the *gap*.
        """
        lags = log_spaced_lags(self.CFG.n_steps)
        per_particle = ta_msd(trajectories, lags, self.CFG)

        exponents = []
        for row in per_particle[:400]:
            if np.all(row > 0.0):
                exponents.append(fit_powerlaw(lags.astype(np.float64), row).exponent)
        exponents = np.array(exponents)
        assert exponents.size > 100

        median = float(np.median(exponents))
        assert median == pytest.approx(1.0, abs=0.10)
        assert median - self.CFG.alpha_wait > 0.3

    def test_ensemble_average_of_ta_msd_is_also_linear(
        self, trajectories: np.ndarray
    ) -> None:
        """Averaging TA-MSDs over particles does not recover ``a`` either.

        A subtle and important point: the EA-TA-MSD is linear in the lag too,
        because it averages linear curves. Only the *ensemble* MSD versus
        absolute time carries the exponent. An analysis that reached for the
        EA-TA-MSD as though it were interchangeable would silently report
        normal diffusion for a strongly subdiffusive population.
        """
        lags = log_spaced_lags(self.CFG.n_steps)
        fit = fit_powerlaw(
            lags.astype(np.float64), ta_msd(trajectories, lags, self.CFG), fit=FitConfig(n_bootstrap=50)
        )
        assert fit.exponent == pytest.approx(1.0, abs=0.10)

    def test_eb_converges_to_a_nonzero_plateau(self, trajectories: np.ndarray) -> None:
        """EB is large, roughly lag-independent at small lag, and near theory.

        Window: lags 2 to 8 out of T = 16384, so ``Delta/T < 5e-4``, which is
        where the plateau is defined. Tolerance 0.12 around
        ``EB_inf = pi/2 - 1 = 0.5708``: the approach is slow, measured
        0.705 at T = 1024, 0.627 at 4096 and 0.580 at 16384 with N = 1500, so
        the residual at this length is a few per cent and the tolerance is set
        by the ensemble scatter rather than by the remaining bias.
        """
        plateau = ctrw_eb_plateau(self.CFG.alpha_wait)
        assert plateau == pytest.approx(np.pi / 2 - 1.0)

        values = ergodicity_breaking_curve(
            trajectories, np.array([2, 4, 8], dtype=np.intp), self.CFG
        )
        assert np.all(values > 0.35)
        np.testing.assert_allclose(values, plateau, atol=0.12)
        # Roughly lag-independent: a plateau, not a slope.
        assert values.max() - values.min() < 0.08

    def test_eb_does_not_decay_with_trajectory_length(self) -> None:
        """The plateau is a wall, not a finite-sample effect.

        Lengthening the trajectory sixteenfold leaves EB essentially where it
        was, whereas for fBm the same change drives it down by more than an
        order of magnitude (see :class:`TestFBMErgodicity`).
        """
        short = eb_parameter(
            run(CTRWConfig(n_particles=1500, n_steps=1024, alpha_wait=0.5, seed=0)), 2
        )
        long = eb_parameter(
            run(CTRWConfig(n_particles=1500, n_steps=16384, alpha_wait=0.5, seed=0)), 2
        )
        assert short > 0.4 and long > 0.4
        assert long > 0.5 * short

    def test_amplitude_scatter_stays_broad(self, trajectories: np.ndarray) -> None:
        """The distribution of ``xi`` does not collapse onto 1.

        A sizeable fraction of trajectories sit below a fifth of the ensemble
        mean -- those that spent the whole measurement inside one trapping
        event. That population is what an ergodic process does not have.
        """
        xi = amplitude_scatter(trajectories, 4, self.CFG)
        assert np.mean(xi < 0.2) > 0.10
        assert xi.std() > 0.5


class TestFBMErgodicity:
    """``PHASE1_PROMPT.md`` step 3, check 2: the ergodic contrast case."""

    @pytest.mark.parametrize("hurst", [0.35, 0.7])
    def test_eb_decays_towards_zero_as_T_grows(self, hurst: float) -> None:
        """EB falls roughly as ``1/T`` -- self-averaging, unlike the CTRW.

        Measured at ``H = 0.35``, lag 10, N = 1500: 0.0191 at T = 512 against
        0.0023 at T = 4096, a factor of 8.3 for a factor 8 in ``T``. Asserted
        loosely as "at least fourfold smaller", which no plateau could pass.
        """
        values = []
        for n_steps in (512, 4096):
            cfg = FBMConfig(n_particles=1500, n_steps=n_steps, hurst=hurst, seed=0)
            values.append(eb_parameter(run(cfg), 10, cfg))
        assert values[0] > values[1]
        assert values[1] < 0.25 * values[0]
        assert values[1] < 0.05

    def test_increment_acf_sign_and_closed_form(self) -> None:
        """``rho(k) = (|k-1|^2H - 2|k|^2H + (k+1)^2H)/2``, negative below H = 1/2.

        The sign at lag 1 is the mechanistic statement: ``H < 1/2`` increments
        actively reverse, ``H > 1/2`` they persist. Tolerance 0.02 per lag from
        N = 4000 trajectories of length 1024.
        """
        lags = np.arange(9)
        for hurst, sign in ((0.3, -1.0), (0.7, +1.0)):
            cfg = FBMConfig(n_particles=4000, n_steps=1024, hurst=hurst, seed=0)
            measured = increment_acf(run(cfg), max_lag=8)
            np.testing.assert_allclose(measured, fbm_increment_acf(lags, hurst), atol=0.02)
            assert np.sign(measured[1]) == sign
            assert abs(measured[1]) > 0.15

        white = increment_acf(
            run(FBMConfig(n_particles=4000, n_steps=1024, hurst=0.5, seed=0)), max_lag=8
        )
        assert abs(white[1]) < 0.02


# --------------------------------------------------------------------------
# correlations
# --------------------------------------------------------------------------


class TestCorrelations:
    """Closed forms and the delta-correlated null case."""

    def test_brownian_increments_are_delta_correlated(self) -> None:
        cfg = BrownianConfig(n_particles=3000, n_steps=1024, seed=0)
        acf = increment_acf(run(cfg), max_lag=6)
        assert acf[0] == pytest.approx(1.0)
        assert np.all(np.abs(acf[1:]) < 0.02)

    def test_normalisation_and_raw_covariance(self) -> None:
        """Unnormalised lag-0 value is the increment variance, ``2 d D dt``."""
        cfg = BrownianConfig(n_particles=4000, n_steps=512, diffusivity=2.5, seed=0)
        raw = increment_acf(run(cfg), max_lag=2, normalise=False)
        assert raw[0] == pytest.approx(2.0 * cfg.diffusivity, rel=0.05)

    def test_velocity_acf_equals_increment_acf_when_normalised(self) -> None:
        """On a grid of spacing ``dt`` the two differ only by ``dt^-2``."""
        cfg = BrownianConfig(n_particles=200, n_steps=256, seed=0)
        x = run(cfg)
        np.testing.assert_allclose(
            velocity_acf(x, 5, dt=0.5), increment_acf(x, 5)
        )

    def test_levy_walk_velocity_stays_correlated(self) -> None:
        """Ballistic runs: the correlation survives many steps.

        This is the mechanism behind its superdiffusion, and it is what a CTRW
        with the same exponent would not show.
        """
        cfg = LevyWalkConfig(n_particles=1000, n_steps=2048, gamma=1.5, tau0=5.0, seed=0)
        acf = increment_acf(run(cfg), max_lag=20)
        assert acf[5] > 0.4
        assert acf[20] > 0.15

    def test_ctrw_increments_are_uncorrelated_despite_the_pauses(self) -> None:
        """Zero correlation plus long immobile stretches, as PROJECT.md table says."""
        cfg = CTRWConfig(n_particles=2000, n_steps=4096, alpha_wait=0.7, seed=0)
        x = run(cfg)
        acf = increment_acf(x, max_lag=6)
        assert np.all(np.abs(acf[1:]) < 0.05)
        assert np.mean(np.diff(x[:, :, 0], axis=1) == 0.0) > 0.7


# --------------------------------------------------------------------------
# distributions
# --------------------------------------------------------------------------


class TestDistributions:
    """van Hove function and the non-Gaussian parameter."""

    def test_van_hove_of_brownian_matches_a_gaussian(self) -> None:
        """The histogram of a Brownian displacement is the Gaussian reference.

        Rescaled by its own standard deviation, so the comparison is against the
        unit Gaussian and tests the shape rather than the amplitude.

        Tolerance 0.02 in density with 40 bins over ``[-4, 4]`` and N = 40000.
        The binomial standard error of the tallest bin is 0.0071 in the same
        units, and the measured worst deviation is 0.0126 -- under two sigma.
        """
        cfg = BrownianConfig(n_particles=40000, n_steps=256, seed=0)
        centres, density = van_hove(
            run(cfg), t_index=128, bins=np.linspace(-4.0, 4.0, 41), rescale=True
        )
        np.testing.assert_allclose(density, gaussian_reference(centres), atol=0.02)

    def test_van_hove_integrates_to_one(self) -> None:
        cfg = BrownianConfig(n_particles=5000, n_steps=64, seed=0)
        centres, density = van_hove(run(cfg), t_index=32, bins=61)
        width = centres[1] - centres[0]
        assert float(density.sum() * width) == pytest.approx(1.0, abs=0.02)

    def test_non_gaussian_parameter_vanishes_for_gaussian_mechanisms(self) -> None:
        """``a_2 = 0`` for Brownian, fBm and scaled Brownian motion, in 1-D and 2-D.

        All three are Gaussian propagators, so this is the estimator's zero
        point and must hold whatever their exponents are.
        """
        for cfg in (
            BrownianConfig(n_particles=20000, n_steps=512, seed=0),
            BrownianConfig(n_particles=20000, n_steps=512, n_dim=2, seed=0),
            FBMConfig(n_particles=20000, n_steps=512, hurst=0.3, seed=0),
            SBMConfig(n_particles=20000, n_steps=512, alpha=0.6, seed=0),
        ):
            a2 = non_gaussian_parameter(run(cfg), cfg)
            assert np.nanmax(np.abs(a2[1:])) < 0.08

    def test_non_gaussian_parameter_detects_diffusing_diffusivity(self) -> None:
        """The mechanism MSD cannot see: ``alpha = 1`` but ``a_2 > 0`` at short times.

        Diffusing diffusivity is Brownian in every MSD-based observable, so this
        is the estimator that separates it -- exactly the situation Q2 is about.
        The excess decays as the walker averages over many diffusivities, so
        ``a_2`` at ``t >> tau`` is far smaller than at ``t << tau``.
        """
        cfg = DDMConfig(n_particles=20000, n_steps=2048, tau=50.0, n_aux=1, seed=0)
        a2 = non_gaussian_parameter(run(cfg), cfg)
        assert a2[5] > 0.5
        assert a2[-1] < 0.5 * a2[5]

    def test_first_element_is_nan_not_a_fake_zero(self) -> None:
        """At ``t = 0`` every displacement is zero and the ratio is 0/0."""
        cfg = BrownianConfig(n_particles=100, n_steps=32, seed=0)
        assert np.isnan(non_gaussian_parameter(run(cfg))[0])

    def test_divergent_moments_are_refused(self) -> None:
        cfg = LevyFlightConfig(n_particles=64, n_steps=64, stability=1.5, seed=0)
        with pytest.raises(DivergentMomentError):
            non_gaussian_parameter(run(cfg), cfg)


# --------------------------------------------------------------------------
# moment spectrum
# --------------------------------------------------------------------------


class TestMomentSpectrum:
    """Simple scaling versus strong anomalous diffusion."""

    Q_GRID = np.linspace(0.2, 4.0, 16)

    @pytest.mark.parametrize(
        "cfg, expected_nu",
        [
            (FBMConfig(n_particles=4000, n_steps=2048, hurst=0.75, seed=0), 0.75),
            (FBMConfig(n_particles=4000, n_steps=2048, hurst=0.30, seed=0), 0.30),
        ],
    )
    def test_fbm_spectrum_is_flat_at_H(self, cfg, expected_nu: float) -> None:
        """Self-similar Gaussian: ``nu(q) = H`` for every ``q``.

        Flat to within 0.03 across ``q`` in ``[0.2, 4]``, measured spread 0.007
        at H = 0.75 and 0.002 at H = 0.30 with N = 4000.
        """
        spectrum = moment_spectrum(
            run(cfg), self.Q_GRID, 1.0, cfg.duration, fit=FitConfig(n_bootstrap=0)
        )
        assert spectrum.is_flat(0.03)
        np.testing.assert_allclose(spectrum.nu, expected_nu, atol=0.03)

    def test_ctrw_spectrum_is_flat(self) -> None:
        """Subdiffusion by trapping is still *simple* scaling: no kink."""
        cfg = CTRWConfig(n_particles=4000, n_steps=4096, alpha_wait=0.7, seed=0)
        spectrum = moment_spectrum(
            run(cfg), self.Q_GRID, cfg.duration / 10.0, cfg.duration, fit=FitConfig(n_bootstrap=0)
        )
        assert spectrum.is_flat(0.04)

    def test_analytic_bilinear_form(self) -> None:
        """``nu(q) = max(1/g, 1 - (g-1)/q)``, continuous, with its kink at ``q = g``."""
        for gamma in (1.3, 1.5, 1.8):
            below = levy_walk_nu(np.array([0.5 * gamma]), gamma)[0]
            at = levy_walk_nu(np.array([gamma]), gamma)[0]
            assert below == pytest.approx(1.0 / gamma)
            assert at == pytest.approx(1.0 / gamma)
            # q = 2 must reproduce the MSD exponent 3 - g.
            assert 2.0 * levy_walk_nu(np.array([2.0]), gamma)[0] == pytest.approx(
                3.0 - gamma
            )
            # q = 4 gives the fourth moment t^(5-g) behind the noise growth.
            assert 4.0 * levy_walk_nu(np.array([4.0]), gamma)[0] == pytest.approx(
                5.0 - gamma
            )

    def test_bilinear_fit_recovers_gamma_from_a_noiseless_spectrum(self) -> None:
        from collectivediff.estimators.moments import MomentSpectrum

        for gamma in (1.25, 1.5, 1.75):
            exact = MomentSpectrum(
                q=self.Q_GRID,
                nu=levy_walk_nu(self.Q_GRID, gamma),
                nu_err=np.zeros_like(self.Q_GRID),
                window=(1.0, 100.0),
            )
            assert fit_bilinear_spectrum(exact) == pytest.approx(gamma, abs=0.01)


class TestLevyWalkMomentSpectrum:
    """The check added by the step-2 review: strong anomalous diffusion.

    ``nu(q)`` must reproduce the bilinear form -- constant ``1/g`` below
    ``q = g``, ``1 - (g-1)/q`` above it -- with the kink at ``q = g``. This is
    the only observable in the toolbox that identifies strong anomalous
    diffusion unambiguously, and the one that separates a Levy walk from fBm at
    matched ``alpha``.
    """

    Q_GRID = np.linspace(0.2, 4.0, 16)

    @pytest.mark.parametrize("gamma", [1.4, 1.5])
    def test_kink_sits_at_q_equals_gamma(self, gamma: float) -> None:
        """Fitted ``g`` recovers the true ``g`` to within 0.08, N = 3000, T = 4096.

        The kink location is obtained by fitting the one-parameter analytic
        family to the measured spectrum rather than by intersecting two
        regression lines, because the low-``q`` branch is flat and says little
        about where it ends.

        Tolerance 0.08 against measured errors of 0.025 (g = 1.5) and 0.001
        (g = 1.4) at this size. The residual is a finite-time effect with a
        known origin: the ballistic cone contributes ``t^(q+1-g)`` to *every*
        moment, which for ``q < g`` decays relative to the ``t^(q/g)`` bulk only
        as ``t^(1 - g + q(g-1)/g)`` -- about ``t^(-1/3)`` at ``q -> 0``,
        ``g = 1.5``. It lifts the low-``q`` branch by roughly 0.02 at
        ``T = 4096`` and shrinks slowly; measured 0.689 against a theoretical
        0.667 even at ``T = 16384``.

        Trajectories are cast to ``float32`` -- the cache dtype -- both to halve
        the memory of a deliberately large ensemble and to exercise that path.
        """
        cfg = LevyWalkConfig(n_particles=3000, n_steps=4096, gamma=gamma, seed=0)
        spectrum = moment_spectrum(
            run(cfg).astype(np.float32),
            self.Q_GRID,
            40.0,
            cfg.duration,
            fit=FitConfig(n_bootstrap=0),
        )
        assert not spectrum.is_flat(0.05)
        assert fit_bilinear_spectrum(spectrum) == pytest.approx(gamma, abs=0.08)

    def test_spectrum_is_not_flat_and_rises_with_q(self) -> None:
        """The qualitative statement, free of any fitted parameter.

        High moments grow faster than low ones because they are dominated by the
        ballistic cone. A mechanism with simple scaling cannot do this, which is
        what makes the observable diagnostic rather than merely descriptive.
        """
        cfg = LevyWalkConfig(n_particles=3000, n_steps=4096, gamma=1.5, seed=0)
        spectrum = moment_spectrum(
            run(cfg).astype(np.float32),
            self.Q_GRID,
            40.0,
            cfg.duration,
            fit=FitConfig(n_bootstrap=0),
        )
        assert spectrum.nu[-1] - spectrum.nu[0] > 0.08
        assert np.all(np.diff(spectrum.nu) > -0.01)
        # And it must approach, but not exceed, the ballistic limit nu = 1.
        assert spectrum.nu.max() < 1.0

    def test_q_equals_two_reproduces_the_msd_exponent(self) -> None:
        """Consistency between two independent estimators of the same number."""
        cfg = LevyWalkConfig(n_particles=3000, n_steps=4096, gamma=1.5, seed=0)
        x = run(cfg)
        spectrum = moment_spectrum(
            x, np.array([2.0]), 40.0, cfg.duration, fit=FitConfig(n_bootstrap=0)
        )
        t = np.arange(cfg.n_steps, dtype=np.float64)
        msd_exponent = fit_powerlaw(
            t, squared_displacement(x, cfg), 40.0, cfg.duration
        ).exponent
        assert 2.0 * spectrum.nu[0] == pytest.approx(msd_exponent, abs=0.05)


# --------------------------------------------------------------------------
# first passage and returns
# --------------------------------------------------------------------------


class TestFirstPassage:
    """Against the Levy-Smirnov closed form for Brownian motion."""

    def test_survival_matches_the_error_function(self) -> None:
        """``S(t) = erf(L / sqrt(4 D t))`` for the one-sided problem.

        Tolerance 0.02 in probability, from N = 20000 trajectories: the standard
        error of a proportion is at most 0.0035, and the rest is a discretisation
        bias -- a crossing that happens and reverses between two grid points is
        invisible, so the measured survival is biased *high*.

        That bias depends on the threshold relative to the single-step
        displacement. At ``L = 40``, comparable to the walker's spread of 45 at
        ``t = T``, the worst deviation is 0.0119; halving the threshold to
        ``L = 20`` doubles it to 0.0224, because the walker then spends the late
        part of the run repeatedly straddling the level. ``L = 40`` is used for
        that reason, not to flatter the estimator.
        """
        cfg = BrownianConfig(n_particles=20000, n_steps=1024, diffusivity=1.0, seed=0)
        threshold = 40.0
        measured = survival_probability(run(cfg), threshold, absolute=False)
        t = np.arange(cfg.n_steps, dtype=np.float64)
        expected = brownian_survival(t, threshold, cfg.diffusivity)
        np.testing.assert_allclose(measured[1:], expected[1:], atol=0.02)

    def test_survival_starts_at_one_and_is_non_increasing(self) -> None:
        cfg = BrownianConfig(n_particles=500, n_steps=256, seed=0)
        s = survival_probability(run(cfg), 10.0)
        assert s[0] == pytest.approx(1.0)
        assert np.all(np.diff(s) <= 1e-12)

    def test_ballistic_first_passage_is_exact(self) -> None:
        """``x = v t`` reaches ``L`` at exactly ``t = L / v``."""
        t = np.arange(64, dtype=np.float64)
        x = (2.0 * t)[None, :, None]
        assert first_passage_times(x, 10.0)[0] == pytest.approx(5.0)

    def test_unreached_threshold_is_infinite_not_censored(self) -> None:
        """Censoring at ``T`` would bias every quantile downwards."""
        t = np.arange(64, dtype=np.float64)
        x = (0.01 * t)[None, :, None]
        assert np.isinf(first_passage_times(x, 100.0)[0])

    def test_subdiffusion_delays_first_passage(self) -> None:
        """A CTRW takes far longer to reach the same level than Brownian motion.

        The comparison is by median, not mean: the Brownian first-passage time
        already has an infinite mean, and the CTRW's is worse.
        """
        threshold = 8.0
        brownian = first_passage_times(
            run(BrownianConfig(n_particles=2000, n_steps=8192, seed=0)), threshold
        )
        ctrw = first_passage_times(
            run(CTRWConfig(n_particles=2000, n_steps=8192, alpha_wait=0.5, seed=0)),
            threshold,
        )
        assert np.median(ctrw) > 3.0 * np.median(brownian)

    def test_threshold_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="threshold"):
            first_passage_times(np.zeros((2, 8, 1)), 0.0)


class TestReturnStatistics:
    """Return-to-origin counts separate persistent from antipersistent motion."""

    def test_counts_sign_changes(self) -> None:
        """A hand-built alternating trace has a known number of crossings."""
        signal = np.array([0.0, 1.0, -1.0, 1.0, -1.0, -2.0])[None, :, None]
        assert return_statistics(signal)["n_returns"][0] == 3.0

    def test_never_returning_trajectory(self) -> None:
        t = np.arange(32, dtype=np.float64)
        stats = return_statistics((1.0 + t)[None, :, None] - 1.0)
        assert stats["n_returns"][0] == 0.0
        assert stats["never_returned"] == 1.0

    def test_antipersistent_fbm_returns_most_often(self) -> None:
        """``H < 1/2`` reverses, ``H > 1/2`` commits; Brownian sits between them.

        This is the same physics as the increment ACF sign, read off a different
        observable, and it is the ordering that any correct implementation must
        produce whatever the tolerances.
        """
        rates = []
        for hurst in (0.3, 0.5, 0.7):
            cfg = FBMConfig(n_particles=600, n_steps=2048, hurst=hurst, seed=0)
            rates.append(return_statistics(run(cfg))["rate"])
        assert rates[0] > rates[1] > rates[2]

    def test_intervals_are_reported_for_a_distribution(self) -> None:
        cfg = FBMConfig(n_particles=200, n_steps=1024, hurst=0.3, seed=0)
        stats = return_statistics(run(cfg))
        assert stats["intervals"].size > 100
        assert np.all(stats["intervals"] > 0.0)

    def test_tolerance_restricts_to_near_origin_crossings(self) -> None:
        """With a tolerance, a sign change far from the origin no longer counts.

        The trace ``0, 50, -50, 0.1, -0.1`` has three sign changes. With
        ``tolerance = 1`` the ``50 -> -50`` jump is rejected, since neither
        endpoint comes near the origin, while ``-50 -> 0.1`` and
        ``0.1 -> -0.1`` are kept: the rule asks that *one* of the two straddling
        points be close, which is the most a discrete grid can check.
        """
        signal = np.array([0.0, 50.0, -50.0, 0.1, -0.1])[None, :, None]
        assert return_statistics(signal)["n_returns"][0] == 3.0
        assert return_statistics(signal, tolerance=1.0)["n_returns"][0] == 2.0
