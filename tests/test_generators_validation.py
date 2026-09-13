"""Validation tests: every generator reproduces its analytic signature.

``PROJECT.md`` section 2 gives this rule the highest priority in the project --
a generator is not usable until a test asserts that it reproduces its analytic
ensemble MSD within a stated tolerance over a stated window. Each test below
states its window, tolerance and ensemble size in its docstring, together with
the measurement those numbers came from.

How the tolerances were set
---------------------------
They are **not** the standard error reported by
:class:`~collectivediff.estimators.msd.PowerLawFit`. That error assumes
independent points, whereas the ensemble MSD is built from the same particles at
every ``t`` and is therefore strongly autocorrelated; measured over twelve seeds
it understates the real spread of the fitted exponent by a factor of roughly
twenty-five (reported ~0.0005, actual seed-to-seed standard deviation ~0.012 at
``N = 2000``, ``T = 1024``).

Every tolerance here was instead set from the *measured* seed-to-seed standard
deviation of the fitted exponent, at roughly three to four sigma, plus any known
finite-time bias. The measured numbers are quoted in the docstrings so a future
reader can tell a real regression from a re-tuned tolerance.

Seeds are fixed, so each test is deterministic; the seed spread is quoted only
to justify the tolerance.
"""

from __future__ import annotations

import numpy as np
import pytest

from collectivediff.config import (
    BrownianConfig,
    CTRWConfig,
    DDMConfig,
    FBMConfig,
    GeneratorConfig,
    LevyFlightConfig,
    LevyWalkConfig,
    SBMConfig,
)
from collectivediff.estimators import (
    DivergentMomentError,
    ea_msd,
    fit_powerlaw,
    fractional_moment,
)
from collectivediff.generators import GENERATORS, generate
from collectivediff.generators.fbm import (
    circulant_eigenvalues,
    fgn_autocovariance,
    fractional_gaussian_noise,
)
from collectivediff.generators.fbm import _validated_eigenvalues
from collectivediff.validation import analytic_msd, mean_diffusivity




# See pyproject.toml: excluded from the default `pytest` invocation.
# Run `pytest -m slow` for this file's validation tests.
pytestmark = pytest.mark.slow

# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def run(cfg: GeneratorConfig) -> np.ndarray:
    """Generate trajectories with the generator seeded from the config."""
    return generate(cfg, np.random.default_rng(cfg.seed))


def times(cfg: GeneratorConfig) -> np.ndarray:
    """The uniform grid belonging to a config, in units of ``dt = 1``."""
    return np.arange(cfg.n_steps, dtype=np.float64)


def msd_exponent(cfg: GeneratorConfig, t_min: float, t_max: float | None = None) -> float:
    """Fitted exponent of the ensemble MSD over ``[t_min, t_max]``."""
    t = times(cfg)
    msd = ea_msd(run(cfg), cfg)
    return fit_powerlaw(t, msd, t_min, cfg.duration if t_max is None else t_max).exponent


def relative_msd_error(cfg: GeneratorConfig) -> np.ndarray:
    """Pointwise ``|MSD_measured / MSD_analytic - 1|``, excluding ``t = 0``."""
    t = times(cfg)[1:]
    return np.abs(ea_msd(run(cfg), cfg)[1:] / analytic_msd(cfg, t) - 1.0)


# --------------------------------------------------------------------------
# properties every generator must have
# --------------------------------------------------------------------------

ALL_CONFIGS = [
    BrownianConfig(n_particles=32, n_steps=128, seed=0),
    FBMConfig(n_particles=32, n_steps=128, hurst=0.3, seed=0),
    FBMConfig(n_particles=32, n_steps=128, hurst=0.7, n_dim=2, seed=0),
    SBMConfig(n_particles=32, n_steps=128, alpha=0.6, seed=0),
    CTRWConfig(n_particles=32, n_steps=128, alpha_wait=0.7, seed=0),
    LevyFlightConfig(n_particles=32, n_steps=128, stability=1.5, seed=0),
    LevyWalkConfig(n_particles=32, n_steps=128, gamma=1.5, n_dim=2, seed=0),
    DDMConfig(n_particles=32, n_steps=128, seed=0),
]


class TestRegistry:
    """Contract shared by every registered generator."""

    def test_every_mechanism_is_registered(self) -> None:
        """All seven mechanisms of PROJECT.md section 4 are reachable by name."""
        assert set(GENERATORS) == {
            "brownian",
            "fbm",
            "sbm",
            "ctrw",
            "levy_flight",
            "levy_walk",
            "ddm",
        }

    @pytest.mark.parametrize("cfg", ALL_CONFIGS, ids=lambda c: f"{type(c).name}-{c.n_dim}d")
    def test_layout_and_initial_condition(self, cfg: GeneratorConfig) -> None:
        """Output obeys the ``(n_particles, n_steps, n_dim)`` float64 rule."""
        x = run(cfg)
        assert x.shape == cfg.shape
        assert x.dtype == np.float64
        assert np.all(np.isfinite(x))
        np.testing.assert_array_equal(x[:, 0, :], 0.0)

    @pytest.mark.parametrize("cfg", ALL_CONFIGS, ids=lambda c: f"{type(c).name}-{c.n_dim}d")
    def test_determinism(self, cfg: GeneratorConfig) -> None:
        """Same config and seed give bit-identical output; a different seed does not.

        This is the ``(config, seed) -> figure`` reproducibility rule of
        PROJECT.md section 2 reduced to its smallest testable form.
        """
        np.testing.assert_array_equal(run(cfg), run(cfg))
        other = generate(cfg, np.random.default_rng(cfg.seed + 1))
        assert not np.allclose(run(cfg), other)

    @pytest.mark.parametrize("cfg", ALL_CONFIGS, ids=lambda c: f"{type(c).name}-{c.n_dim}d")
    def test_rejects_implicit_randomness(self, cfg: GeneratorConfig) -> None:
        """A legacy RandomState draws from a different stream and is refused."""
        with pytest.raises(TypeError, match="explicit np.random.Generator"):
            generate(cfg, np.random.RandomState(0))  # type: ignore[arg-type]

    def test_unknown_config_is_reported(self) -> None:
        class Bogus(BrownianConfig):
            name = "bogus"

        with pytest.raises(KeyError, match="no generator registered"):
            generate(Bogus(), np.random.default_rng(0))


# --------------------------------------------------------------------------
# Brownian
# --------------------------------------------------------------------------


class TestBrownian:
    """``<r^2(t)> = 2 d D t`` exactly, at every grid point."""

    def test_exponent(self) -> None:
        """alpha = 1 to within 0.03, window [1, T], N = 4000, T = 1024.

        Window: the full grid, because the law is exact at every ``t`` -- there
        is no crossover to avoid and every particle contributes at every point.
        Tolerance: the seed-to-seed standard deviation of the fitted exponent is
        0.012 at N = 2000 (twelve seeds), so ~0.008 at N = 4000; 0.03 is
        3.5 sigma. The measured deviation at this seed is 0.0002.
        """
        cfg = BrownianConfig(n_particles=4000, n_steps=1024, seed=0)
        assert msd_exponent(cfg, 1.0) == pytest.approx(1.0, abs=0.03)

    @pytest.mark.parametrize("n_dim", [1, 2])
    @pytest.mark.parametrize("diffusivity", [0.5, 2.0])
    def test_amplitude_matches_2dDt(self, n_dim: int, diffusivity: float) -> None:
        """The prefactor, not just the slope: median relative error below 5 per cent.

        The relative standard deviation of an ensemble MSD of N Gaussian walkers
        is sqrt(2/N) = 2.2 per cent at N = 4000, and the points are strongly
        correlated in ``t``, so the worst point over a 1024-point curve runs
        about 2.5x that. Asserting the median at 5 per cent and the maximum at
        12 per cent tests the amplitude without testing the luck of one seed.
        """
        cfg = BrownianConfig(
            n_particles=4000, n_steps=1024, n_dim=n_dim, diffusivity=diffusivity, seed=0
        )
        error = relative_msd_error(cfg)
        assert np.median(error) < 0.05
        assert error.max() < 0.12

    def test_increments_are_uncorrelated_and_gaussian(self) -> None:
        """The null hypothesis: delta increment ACF and zero excess kurtosis.

        Everything anomalous in this project is defined against this, so a
        Brownian generator that quietly correlated its increments would
        invalidate the comparisons rather than just itself.
        """
        cfg = BrownianConfig(n_particles=4000, n_steps=512, seed=0)
        dx = np.diff(run(cfg)[:, :, 0], axis=1)
        variance = dx.var()
        for lag in (1, 2, 5):
            acf = np.mean(dx[:, :-lag] * dx[:, lag:]) / variance
            assert abs(acf) < 0.01
        kurtosis = np.mean(dx**4) / variance**2 - 3.0
        assert abs(kurtosis) < 0.05


# --------------------------------------------------------------------------
# fractional Brownian motion
# --------------------------------------------------------------------------


class TestFBM:
    """``<r^2(t)> = 2 d D t^(2H)`` exactly, from an exact Davies-Harte sampler."""

    @pytest.mark.parametrize("hurst", [0.3, 0.5, 0.7, 0.9])
    def test_exponent(self, hurst: float) -> None:
        """alpha = 2H to within 0.03, window [1, T], N = 4000, T = 1024.

        Same window and tolerance rationale as Brownian: the law is exact at
        every ``t``, and the measured seed-to-seed spread is 0.012 at N = 2000.
        Measured deviations at this seed: 0.002 (H = 0.3), 0.004 (H = 0.5),
        0.006 (H = 0.7).
        """
        cfg = FBMConfig(n_particles=4000, n_steps=1024, hurst=hurst, seed=0)
        assert msd_exponent(cfg, 1.0) == pytest.approx(2.0 * hurst, abs=0.03)

    @pytest.mark.parametrize("hurst", [0.3, 0.7])
    @pytest.mark.parametrize("n_dim", [1, 2])
    def test_amplitude(self, hurst: float, n_dim: int) -> None:
        """Prefactor ``2 d D``: median relative error below 5 per cent, N = 4000."""
        cfg = FBMConfig(
            n_particles=4000, n_steps=1024, n_dim=n_dim, hurst=hurst, seed=0
        )
        error = relative_msd_error(cfg)
        assert np.median(error) < 0.05
        assert error.max() < 0.12

    def test_hurst_one_half_reproduces_brownian_law(self) -> None:
        """H = 1/2 must fall back onto ``2 d D t``, with the same ``D``.

        A generator that got the normalisation right only for anomalous H would
        make fBm incomparable with the Brownian control.
        """
        cfg = FBMConfig(n_particles=6000, n_steps=512, hurst=0.5, diffusivity=1.7, seed=0)
        brownian_like = BrownianConfig(
            n_particles=6000, n_steps=512, diffusivity=1.7, seed=0
        )
        t = times(cfg)[1:]
        measured = ea_msd(run(cfg), cfg)[1:]
        np.testing.assert_allclose(
            measured / analytic_msd(brownian_like, t), 1.0, atol=0.10
        )

    @pytest.mark.parametrize("hurst", [0.2, 0.3, 0.5, 0.7, 0.8])
    def test_increment_covariance_matches_closed_form(self, hurst: float) -> None:
        """The sampler is exact, so the *whole* covariance must match, not just t^2H.

        This is the strongest available check on Davies-Harte and the reason the
        method was chosen: an ad-hoc filter can reproduce the MSD exponent while
        getting the correlation structure wrong, and the correlation structure is
        what phase 2 uses to tell fBm from its look-alikes.

        Tolerance 0.01 on each of the first six lags, from 20000 series of
        length 64: the standard error of a covariance estimated from ``n``
        series is about ``1/sqrt(n) = 0.007``.
        """
        noise = fractional_gaussian_noise(20000, 64, hurst, np.random.default_rng(0))
        measured = np.array(
            [np.mean(noise[:, : 64 - k] * noise[:, k:]) for k in range(6)]
        )
        np.testing.assert_allclose(measured, fgn_autocovariance(6, hurst), atol=0.01)

    def test_increment_acf_sign(self) -> None:
        """H < 1/2 anticorrelates increments, H > 1/2 correlates them.

        The sign of ``gamma(1)`` is what separates sub- from superdiffusive fBm
        mechanistically rather than by exponent alone.
        """
        rng = np.random.default_rng(1)
        for hurst, expected_sign in ((0.3, -1.0), (0.7, +1.0)):
            noise = fractional_gaussian_noise(4000, 128, hurst, rng)
            acf1 = np.mean(noise[:, :-1] * noise[:, 1:]) / np.mean(noise**2)
            assert np.sign(acf1) == expected_sign
            assert abs(acf1) > 0.15
        white = fractional_gaussian_noise(4000, 128, 0.5, rng)
        assert abs(np.mean(white[:, :-1] * white[:, 1:]) / np.mean(white**2)) < 0.02

    def test_paired_samples_are_independent(self) -> None:
        """Each FFT yields two series; they must not be correlated with each other.

        The real and imaginary parts of the transform are independent only
        because the circulant's covariance is real. If that ever stopped holding,
        half the ensemble would be a copy of the other half.
        """
        noise = fractional_gaussian_noise(8000, 64, 0.7, np.random.default_rng(0))
        even, odd = noise[0::2], noise[1::2]
        cross = np.mean(even * odd) / np.sqrt(np.mean(even**2) * np.mean(odd**2))
        assert abs(cross) < 0.02

    def test_autocovariance_closed_form(self) -> None:
        """``gamma(0) = 1`` and ``gamma(k) = 0`` for k >= 1 at H = 1/2."""
        np.testing.assert_allclose(fgn_autocovariance(8, 0.5), [1.0] + [0.0] * 7, atol=1e-12)
        assert fgn_autocovariance(4, 0.3)[1] < 0.0
        assert fgn_autocovariance(4, 0.7)[1] > 0.0

    @pytest.mark.parametrize("hurst", [0.05, 0.25, 0.5, 0.75, 0.95])
    @pytest.mark.parametrize("n_samples", [16, 129, 512])
    def test_embedding_is_valid(self, hurst: float, n_samples: int) -> None:
        """The circulant embedding is non-negative definite over the usable range."""
        eigenvalues = circulant_eigenvalues(n_samples, hurst)
        assert eigenvalues.min() >= -1e-10 * eigenvalues.max()

    def test_negative_eigenvalues_raise_rather_than_truncate(self) -> None:
        """The guard is real: a materially negative eigenvalue is fatal, not clipped.

        No ``(H, n)`` in the usable range triggers this -- the embedding is valid
        for fGn -- so the guard is exercised directly. Silently truncating would
        sample a different covariance than requested, which is exactly the
        failure mode that would be invisible downstream.
        """
        poisoned = np.array([4.0, 1.0, -0.5, 1.0])
        with pytest.raises(ValueError, match="non-negativity"):
            _validated_eigenvalues(poisoned, hurst=0.3, n_samples=2)
        # Rounding-level negatives are clipped, not raised.
        clipped = _validated_eigenvalues(np.array([4.0, -1e-16, 1.0]), 0.3, 2)
        assert clipped.min() == 0.0


# --------------------------------------------------------------------------
# scaled Brownian motion
# --------------------------------------------------------------------------


class TestSBM:
    """``<r^2(t)> = 2 d D0 t^alpha`` exactly, both sub- and superdiffusive."""

    @pytest.mark.parametrize("alpha", [0.3, 0.6, 1.0, 1.4])
    def test_exponent(self, alpha: float) -> None:
        """alpha recovered to within 0.03, window [1, T], N = 4000, T = 1024.

        Exact law, so the full window is used. Measured deviations at this seed:
        0.002 (alpha = 0.6), 0.0002 (alpha = 1.4). Seed spread 0.011 at
        N = 2000 over twelve seeds.
        """
        cfg = SBMConfig(n_particles=4000, n_steps=1024, alpha=alpha, seed=0)
        assert msd_exponent(cfg, 1.0) == pytest.approx(alpha, abs=0.03)

    @pytest.mark.parametrize("alpha", [0.6, 1.4])
    def test_amplitude(self, alpha: float) -> None:
        """Prefactor ``2 d D0``: median relative error below 5 per cent, N = 4000."""
        cfg = SBMConfig(n_particles=4000, n_steps=1024, alpha=alpha, seed=0)
        error = relative_msd_error(cfg)
        assert np.median(error) < 0.05
        assert error.max() < 0.12

    def test_alpha_one_is_brownian(self) -> None:
        """``alpha = 1`` reduces exactly to Brownian motion with ``D = D0``."""
        cfg = SBMConfig(n_particles=4000, n_steps=256, alpha=1.0, diffusivity=0.8, seed=3)
        reference = BrownianConfig(
            n_particles=4000, n_steps=256, diffusivity=0.8, seed=3
        )
        np.testing.assert_array_equal(run(cfg), run(reference))

    def test_increments_are_non_stationary(self) -> None:
        """The mechanism, not just the exponent: increment variance ages as t^(alpha-1).

        SBM shares its exponent with fBm and CTRW but reaches it by a
        deterministically shrinking step size. A generator that produced
        stationary increments would still pass the MSD test.
        """
        cfg = SBMConfig(n_particles=4000, n_steps=1024, alpha=0.6, seed=0)
        dx = np.diff(run(cfg)[:, :, 0], axis=1)
        early = dx[:, 1:11].var()
        late = dx[:, -10:].var()
        assert late < 0.2 * early
        # Predicted ratio of the variances, 2 D0 (t^a - (t-1)^a) at the two windows.
        t = times(cfg)
        predicted = np.diff(t**cfg.alpha)
        assert late / early == pytest.approx(
            predicted[-10:].mean() / predicted[1:11].mean(), rel=0.15
        )


# --------------------------------------------------------------------------
# CTRW
# --------------------------------------------------------------------------


class TestCTRW:
    """``<r^2(t)> ~ t^a`` asymptotically; the crux mechanism of Q1."""

    @pytest.mark.parametrize("alpha_wait", [0.5, 0.7])
    def test_exponent(self, alpha_wait: float) -> None:
        """alpha = a to within 0.05, window [T/10, T], N = 2000, T = 4096.

        Window: the CTRW reaches its asymptotic scaling only for ``t >> tau0``,
        so the first decade is excluded. Fitting from ``T/10`` also keeps the
        window inside the regime where the renewal count is large.

        Tolerance 0.05 covers two effects measured separately. The seed-to-seed
        standard deviation is 0.012 (a = 0.7) to 0.034 (a = 0.5) over ten seeds.
        On top of that sits a genuine finite-time bias of about -0.02, from the
        subleading corrections derived in
        ``validation.reference.ctrw_msd_amplitude``: the relative correction
        decays only as ``(t/tau0)^-(1-a)``, and an additive ``-1`` from the
        renewal count pulls the other way. Measured deviations at this seed:
        0.019 (a = 0.5), 0.024 (a = 0.7).

        ``a`` near 0 or 1 is deliberately excluded -- there the bias reaches
        0.07 at this ``T`` and a fair test would need a far longer trajectory.
        """
        cfg = CTRWConfig(n_particles=2000, n_steps=4096, alpha_wait=alpha_wait, seed=0)
        assert msd_exponent(cfg, cfg.duration / 10.0) == pytest.approx(
            alpha_wait, abs=0.05
        )

    def test_ensemble_msd_is_clearly_subdiffusive(self) -> None:
        """The exponent is far enough below 1 that no tolerance could hide it.

        This is the assertion that actually matters scientifically: a CTRW whose
        ensemble MSD came out linear would be a broken generator, not a noisy one.
        """
        cfg = CTRWConfig(n_particles=2000, n_steps=4096, alpha_wait=0.5, seed=0)
        assert msd_exponent(cfg, cfg.duration / 10.0) < 0.7

    def test_asymptotic_amplitude(self) -> None:
        """MSD at ``t = T`` is within 50 per cent of the leading asymptotic form.

        A loose bound on purpose. The leading amplitude
        ``d sigma^2 (t/tau0)^a / (Gamma(1-a) Gamma(1+a))`` carries a relative
        correction of order ``(t/tau0)^-(1-a)``, which at a = 0.7 and t = 4095
        is still about 8 per cent, plus the additive ``-1``. The measured ratio
        at this seed is 1.13. Tightening this would test the asymptotic
        expansion, not the generator.
        """
        cfg = CTRWConfig(n_particles=2000, n_steps=4096, alpha_wait=0.7, seed=0)
        ratio = ea_msd(run(cfg), cfg)[-1] / analytic_msd(cfg, times(cfg)[-1:])[0]
        assert 0.5 < ratio < 1.5

    def test_trajectories_contain_long_immobile_stretches(self) -> None:
        """The mechanism: a heavy-tailed waiting time must show up as a flat trace.

        Roughly half of all grid steps should show no displacement at all for
        a = 0.5, because the walker is waiting. A generator that produced the
        right exponent by shrinking its jumps instead would fail here.
        """
        cfg = CTRWConfig(n_particles=500, n_steps=2048, alpha_wait=0.5, seed=0)
        dx = np.diff(run(cfg)[:, :, 0], axis=1)
        immobile = np.mean(dx == 0.0)
        assert immobile > 0.5
        # The longest single pause should be a sizeable fraction of the trajectory.
        longest = max(
            max((len(s) for s in "".join("01"[int(v)] for v in row).split("1")), default=0)
            for row in (dx[:20] != 0.0)
        )
        assert longest > cfg.n_steps / 20

    def test_waiting_times_follow_the_pareto_tail(self) -> None:
        """Raw event sequence: ``P(tau > t) = (t/tau0)^-a`` before any gridding.

        Checked on the continuous-time representation, which is where the tail
        actually lives -- the gridded trajectory cannot resolve it.
        """
        from collectivediff.generators.ctrw import ctrw_events

        cfg = CTRWConfig(n_particles=4000, n_steps=1024, alpha_wait=0.6, tau0=2.0, seed=0)
        event_times, _ = ctrw_events(cfg, np.random.default_rng(0))
        waits = np.diff(event_times, axis=1).ravel()
        assert waits.min() >= cfg.tau0 - 1e-12
        for threshold in (4.0, 16.0, 64.0):
            survival = np.mean(waits > threshold)
            expected = (threshold / cfg.tau0) ** (-cfg.alpha_wait)
            assert survival == pytest.approx(expected, rel=0.05)


# --------------------------------------------------------------------------
# Levy flight
# --------------------------------------------------------------------------


class TestLevyFlight:
    """No MSD at all: validate a fractional moment and guard the MSD path."""

    @pytest.mark.parametrize("stability", [0.8, 1.2, 1.5, 1.8])
    def test_fractional_moment_exponent(self, stability: float) -> None:
        """``<|r|^q> ~ t^(q/s)`` with ``q = s/2``, to within 0.03, N = 4000, T = 1024.

        The second moment diverges, so PROJECT.md section 4 requires a
        fractional moment instead. ``q = s/2`` sits safely below the stability
        index, where the moment exists and its estimator has finite variance.

        Window [1, T]: the process is exactly self-similar at every scale, so
        there is no crossover to avoid. Tolerance 0.03 against a measured
        seed-to-seed standard deviation of 0.007 over eight seeds; measured
        deviations at this seed are 0.005-0.007 across all four ``s``.
        """
        cfg = LevyFlightConfig(
            n_particles=4000, n_steps=1024, stability=stability, seed=0
        )
        q = 0.5 * stability
        fit = fit_powerlaw(times(cfg), fractional_moment(run(cfg), q), 1.0, cfg.duration)
        assert fit.exponent / q == pytest.approx(1.0 / stability, abs=0.03)

    def test_msd_path_refuses_to_return_a_number(self) -> None:
        """Required by PROJECT.md section 4: no finite number from a diverging moment."""
        cfg = LevyFlightConfig(n_particles=64, n_steps=128, stability=1.5, seed=0)
        with pytest.raises(DivergentMomentError):
            ea_msd(run(cfg), cfg)
        with pytest.raises(DivergentMomentError):
            analytic_msd(cfg, times(cfg))
        with pytest.raises(ValueError, match="diverges"):
            _ = cfg.alpha_analytic

    def test_empirical_second_moment_does_not_converge(self) -> None:
        """The reason for the guard, demonstrated rather than asserted.

        Doubling the ensemble should halve the standard error of a moment that
        exists. For the second moment of a Levy flight it does no such thing:
        the sample value is set by the single largest jump drawn, so it jumps
        around by factors of order unity between ensembles of any size.
        """
        cfg = LevyFlightConfig(n_particles=20000, n_steps=256, stability=1.2, seed=0)
        x = run(cfg)[:, -1, 0]
        chunks = np.array([np.mean(c**2) for c in np.array_split(x, 10)])
        assert chunks.std() / chunks.mean() > 0.5

        # A fractional moment with q < s, by contrast, is stable across chunks.
        fractional = np.array([np.mean(np.abs(c) ** 0.6) for c in np.array_split(x, 10)])
        assert fractional.std() / fractional.mean() < 0.1

    def test_jumps_have_the_stable_tail(self) -> None:
        """``P(|X| > x) ~ x^-s``: the tail index is what makes the moment diverge."""
        from collectivediff.generators.levy import symmetric_stable

        rng = np.random.default_rng(0)
        for stability in (1.0, 1.5):
            sample = np.abs(symmetric_stable(rng, (400000,), stability))
            high, low = np.mean(sample > 100.0), np.mean(sample > 10.0)
            assert np.log(low / high) / np.log(10.0) == pytest.approx(stability, abs=0.1)


# --------------------------------------------------------------------------
# Levy walk
# --------------------------------------------------------------------------


class TestLevyWalk:
    """``<r^2(t)> ~ t^(3-g)``: finite MSD, because displacement costs time."""

    @pytest.mark.parametrize("gamma", [1.4, 1.5, 1.6])
    def test_exponent(self, gamma: float) -> None:
        """alpha = 3 - g to within 0.10, window [40, T], N = 3000, T = 4096.

        Window: the lower edge excludes the ballistic transient of the first few
        flights (``tau0 = 1``), and the fit runs to ``T`` because the upper
        decade still carries signal even though it is the noisiest part.

        Tolerance 0.10 is looser than for the Gaussian mechanisms, for a
        structural reason worth recording: the Levy-walk MSD estimator gets
        *noisier with time*. Because ``<r^4> ~ t^(5-g)`` grows faster than
        ``<r^2>^2 ~ t^(6-2g)``, the relative standard error of the ensemble MSD
        grows as roughly ``t^((g-1)/2)`` -- measured at g = 1.4, N = 3000, it
        runs from 1.9 per cent at t = 10 to 5.7 per cent at t = 4095. The
        resulting seed-to-seed standard deviation of the fitted exponent is
        0.021-0.031 over eight seeds, with a residual bias below 0.02
        (measured means 1.580 / 1.500 / 1.411 against targets 1.60 / 1.50 /
        1.40). 0.10 is therefore about 3.5 sigma.
        """
        cfg = LevyWalkConfig(n_particles=3000, n_steps=4096, gamma=gamma, seed=0)
        assert msd_exponent(cfg, 40.0) == pytest.approx(3.0 - gamma, abs=0.10)

    def test_exponent_tracks_gamma(self) -> None:
        """The exponent must *move* with ``g``, not merely sit near one value.

        A generator with a hard-wired superdiffusive exponent would pass each
        individual tolerance above; it cannot pass this.
        """
        exponents = [
            msd_exponent(
                LevyWalkConfig(n_particles=3000, n_steps=4096, gamma=g, seed=0), 40.0
            )
            for g in (1.4, 1.6)
        ]
        assert exponents[0] - exponents[1] > 0.10

    def test_is_superdiffusive_but_subballistic(self) -> None:
        """``1 < alpha < 2``: the space-time coupling bounds it away from both ends."""
        alpha = msd_exponent(
            LevyWalkConfig(n_particles=3000, n_steps=4096, gamma=1.5, seed=0), 40.0
        )
        assert 1.1 < alpha < 1.9

    def test_speed_is_constant_along_the_trajectory(self) -> None:
        """The defining property: the walker never teleports.

        Every gridded step must have length at most ``v dt`` -- exactly ``v dt``
        on a step lying inside one flight, and less only on a step that straddles
        a turning point. This bound is what makes the second moment finite, and
        it is exactly what a Levy *flight* lacks.

        ``tau0 = 5`` gives a mean flight duration of ``g tau0 / (g - 1) = 15``
        steps, so about one step in fifteen straddles a turn and the rest run at
        full speed; measured 95 per cent, asserted above 85. With ``tau0 = dt``
        the same test would only reach 76 per cent and would be measuring the
        turn rate rather than the speed bound.
        """
        cfg = LevyWalkConfig(
            n_particles=200, n_steps=1024, gamma=1.5, tau0=5.0, speed=2.0, n_dim=2, seed=0
        )
        step = np.linalg.norm(np.diff(run(cfg), axis=1), axis=2)
        assert step.max() <= cfg.speed + 1e-9
        assert np.mean(step > 0.95 * cfg.speed) > 0.85

    def test_msd_amplitude_has_no_closed_form_reference(self) -> None:
        """The reference module says so explicitly instead of inventing a prefactor."""
        cfg = LevyWalkConfig(n_particles=8, n_steps=64, gamma=1.5, seed=0)
        with pytest.raises(NotImplementedError, match="prefactor"):
            analytic_msd(cfg, times(cfg))


# --------------------------------------------------------------------------
# diffusing diffusivity
# --------------------------------------------------------------------------


class TestDDM:
    """``alpha = 1`` exactly, but the propagator is not Gaussian."""

    def test_exponent(self) -> None:
        """alpha = 1 to within 0.03, window [1, T], N = 4000, T = 1024.

        The MSD is exactly linear in the mean, so the full window applies.
        Measured deviation at this seed 0.0025; seed spread 0.009 over twelve
        seeds at N = 2000.
        """
        cfg = DDMConfig(n_particles=4000, n_steps=1024, tau=30.0, seed=0)
        assert msd_exponent(cfg, 1.0) == pytest.approx(1.0, abs=0.03)

    @pytest.mark.parametrize("n_aux", [1, 3])
    def test_amplitude_uses_the_stationary_mean_diffusivity(self, n_aux: int) -> None:
        """``<r^2> = 2 d <D> t`` with ``<D> = n_aux sigma^2 tau / 2``.

        This tests the OU normalisation, which is where an Euler step would go
        wrong: the stationary variance of the auxiliary process sets the
        amplitude, and getting it wrong would rescale every DDM result.
        """
        cfg = DDMConfig(n_particles=4000, n_steps=1024, tau=20.0, n_aux=n_aux, seed=0)
        error = relative_msd_error(cfg)
        assert np.median(error) < 0.06
        assert error.max() < 0.15

    def test_auxiliary_process_is_stationary_with_the_right_variance(self) -> None:
        """The OU process starts in its stationary state and stays there.

        Starting from a delta would give a burn-in during which the MSD is not
        linear -- precisely the artefact that would be mistaken for anomalous
        diffusion.
        """
        from collectivediff.generators.ddm import ou_process

        cfg = DDMConfig(n_particles=8000, n_steps=512, tau=25.0, sigma=1.3, seed=0)
        y = ou_process(cfg, np.random.default_rng(0))
        expected = cfg.sigma**2 * cfg.tau / 2.0
        variance = y[:, :, 0].var(axis=0)
        np.testing.assert_allclose(variance, expected, rtol=0.08)
        assert mean_diffusivity(cfg) == pytest.approx(cfg.n_aux * expected)

        # And its autocorrelation decays as exp(-lag / tau).
        centred = y[:, :, 0] - y[:, :, 0].mean()
        for lag in (10, 25, 50):
            acf = np.mean(centred[:, :-lag] * centred[:, lag:]) / expected
            assert acf == pytest.approx(np.exp(-lag / cfg.tau), abs=0.05)

    def test_propagator_is_non_gaussian_at_short_times(self) -> None:
        """The whole point of the mechanism: excess kurtosis at ``t << tau``.

        A generator that produced a linear MSD from ordinary Gaussian steps
        would pass every test above. For ``n_aux = 1`` the displacement is a
        Gaussian mixed over ``D = Y^2``, which has heavy (exponential) tails;
        by ``t >> tau`` the walker has averaged over many independent
        diffusivities and the tails relax towards Gaussian.
        """
        cfg = DDMConfig(n_particles=20000, n_steps=2048, tau=50.0, n_aux=1, seed=0)
        x = run(cfg)[:, :, 0]

        def excess_kurtosis(sample: np.ndarray) -> float:
            return float(np.mean(sample**4) / np.mean(sample**2) ** 2 - 3.0)

        short = excess_kurtosis(x[:, 5])
        long = excess_kurtosis(x[:, -1])
        assert short > 1.0
        assert long < 0.5 * short
