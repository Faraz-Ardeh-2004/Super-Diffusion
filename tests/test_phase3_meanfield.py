"""Tests for the Phase 3 mean-field model (PROJECT.md section 7, PHASE3_PROMPT.md step 1).

V1's reference target changed during development: an earlier draft of
``PROJECT.md`` section 7 claimed the ``J = 0`` reduction was exactly
fractional Gaussian noise. It is exactly ARFIMA(0,d,0) instead -- the two
share the asymptotic exponent ``1 + 2d`` but not the short-lag correlation
(``rho(1) = d/(1-d)`` for ARFIMA against ``2^{2d} - 1`` for fGn). These tests
validate against the corrected target. See ``NOTES.md`` for the diagnosis.
"""

from __future__ import annotations

import numpy as np
import pytest

from collectivediff.config import CONFIG_REGISTRY, MeanFieldConfig, config_from_dict, config_to_dict
from collectivediff.dynamics import (
    arfima_acf,
    arfima_effective_exponent,
    coupling_for_target_crossover,
    deviation_spectrum,
    fractional_diff_kappa,
    simulate_meanfield,
    whittle_d_only_estimate,
    whittle_dj_estimate,
)
from collectivediff.estimators.correlations import increment_acf
from collectivediff.estimators.ergodicity import eb_parameter
from collectivediff.estimators.msd import ea_msd, fit_powerlaw


class TestMeanFieldConfig:
    def test_rejects_out_of_range_parameters(self) -> None:
        with pytest.raises(ValueError, match="d must lie"):
            MeanFieldConfig(d=0.5)
        with pytest.raises(ValueError, match="J must be"):
            MeanFieldConfig(J=-0.1)
        with pytest.raises(ValueError, match="K must be"):
            MeanFieldConfig(K=0)
        with pytest.raises(ValueError, match="sigma must be"):
            MeanFieldConfig(sigma=0.0)
        with pytest.raises(ValueError, match="burn_in must be"):
            MeanFieldConfig(K=100, burn_in=50)
        with pytest.raises(ValueError, match="1-D only"):
            MeanFieldConfig(n_dim=2)

    def test_alpha_analytic_and_hurst(self) -> None:
        cfg = MeanFieldConfig(d=0.3)
        assert cfg.alpha_analytic == pytest.approx(1.6)
        assert cfg.hurst == pytest.approx(0.8)

    def test_round_trips_through_config_registry(self) -> None:
        cfg = MeanFieldConfig(n_particles=50, n_steps=100, d=0.2, J=0.3, K=64, burn_in=256, seed=3)
        assert "meanfield" in CONFIG_REGISTRY
        restored = config_from_dict(config_to_dict(cfg))
        assert restored == cfg


class TestFractionalDiffKappa:
    def test_matches_direct_gamma_ratio_for_small_k(self) -> None:
        from scipy.special import gamma as gamma_fn

        d = 0.3
        k = 6
        kappa = fractional_diff_kappa(d, k)
        s = np.arange(1, k + 1)
        c_direct = gamma_fn(s - d) / (gamma_fn(-d) * gamma_fn(s + 1.0))
        assert np.allclose(kappa, -c_direct, atol=1e-10)

    def test_kappa_one_equals_d(self) -> None:
        # c_1 = -d (from Gamma(1+d) = d Gamma(d)), so kappa(1) = d.
        for d in (-0.3, 0.0, 0.4):
            assert fractional_diff_kappa(d, 1)[0] == pytest.approx(d)


class TestArfimaClosedForm:
    def test_rho_zero_is_one(self) -> None:
        for d in (-0.4, 0.0, 0.45):
            assert arfima_acf(d, [0])[0] == pytest.approx(1.0)

    def test_rho_one_equals_d_over_one_minus_d(self) -> None:
        for d in (-0.3, 0.15, 0.25, 0.4):
            assert arfima_acf(d, [1])[0] == pytest.approx(d / (1.0 - d))

    def test_d_zero_is_white_noise(self) -> None:
        acf = arfima_acf(0.0, [0, 1, 2, 3])
        assert acf[0] == 1.0
        assert np.all(acf[1:] == 0.0)

    def test_effective_exponent_below_asymptote_for_positive_d_converging_up(self) -> None:
        # For d > 0, ARFIMA's MSD exponent approaches 1 + 2d from below inside
        # any finite window (confirmed by direct summation of the exact ACF).
        d = 0.25
        exponent = arfima_effective_exponent(d, 2, 100)
        assert exponent < 1.0 + 2.0 * d
        assert exponent == pytest.approx(1.0 + 2.0 * d, abs=0.02)


@pytest.mark.slow
class TestSimulateMeanField:
    def test_output_shapes_and_internal_consistency(self) -> None:
        cfg = MeanFieldConfig(n_particles=20, n_steps=50, K=32, d=0.2, J=0.1, burn_in=128, seed=1)
        sim = simulate_meanfield(cfg, np.random.default_rng(1))
        assert sim.positions.shape == (20, 50, 1)
        assert sim.velocities.shape == (20, 49, 1)
        assert sim.mean_field.shape == (49,)
        assert np.allclose(sim.positions[:, 0, :], 0.0)
        assert np.allclose(sim.velocities[:, :, 0], np.diff(sim.positions[:, :, 0], axis=1))
        assert np.allclose(sim.mean_field, sim.velocities[:, :, 0].mean(axis=0))
        com = sim.center_of_mass()
        assert com.shape == (50,)
        assert com[0] == 0.0
        assert np.allclose(np.diff(com), sim.mean_field)

    def test_d_zero_reduces_to_brownian_motion(self) -> None:
        # kappa is identically zero at d = 0, so this is exactly Brownian
        # motion in velocity increments -- no truncation or fidelity gap to
        # account for, unlike d != 0.
        cfg = MeanFieldConfig(n_particles=500, n_steps=300, K=64, d=0.0, J=0.0, burn_in=256, seed=2)
        sim = simulate_meanfield(cfg, np.random.default_rng(2))
        acf = increment_acf(sim.positions, 5)
        assert acf[0] == pytest.approx(1.0)
        assert np.max(np.abs(acf[1:])) < 0.05

    def test_v1_acf_matches_exact_arfima_at_j_zero(self) -> None:
        n_particles, n_steps, k = 300, 300, 128
        max_lag = k // 10
        for d in (-0.25, 0.25):
            cfg = MeanFieldConfig(
                n_particles=n_particles, n_steps=n_steps, K=k, d=d, J=0.0,
                burn_in=4 * k, seed=2,
            )
            sim = simulate_meanfield(cfg, np.random.default_rng(2))
            acf_sim = increment_acf(sim.positions, max_lag)
            acf_exact = arfima_acf(d, np.arange(max_lag + 1))
            assert np.max(np.abs(acf_sim - acf_exact)) < 0.03

    def test_v1_ea_msd_exponent_matches_exact_arfima_window_target(self) -> None:
        # The target is the exact process's own in-window exponent, not the
        # bare asymptote 1 + 2d, which the exact process does not reach
        # inside a short window either (see arfima_effective_exponent).
        n_particles, n_steps, k = 300, 300, 128
        max_lag = k // 10
        for d in (-0.25, 0.25):
            cfg = MeanFieldConfig(
                n_particles=n_particles, n_steps=n_steps, K=k, d=d, J=0.0,
                burn_in=4 * k, seed=2,
            )
            sim = simulate_meanfield(cfg, np.random.default_rng(2))
            ea = ea_msd(sim.positions)
            t = np.arange(n_steps, dtype=np.float64)
            fit = fit_powerlaw(t, ea, 2.0, float(max_lag), fit=None)
            target = arfima_effective_exponent(d, 2, max_lag)
            assert abs(fit.exponent - target) < 0.08

    def test_v1_eb_shrinks_as_t_grows(self) -> None:
        d = 0.25
        k = 64
        eb_lag = max(2, k // 20)
        eb_values = []
        for n_steps in (150, 600):
            cfg = MeanFieldConfig(
                n_particles=300, n_steps=n_steps, K=k, d=d, J=0.0,
                burn_in=4 * k, seed=4,
            )
            sim = simulate_meanfield(cfg, np.random.default_rng(4))
            eb_values.append(eb_parameter(sim.positions, eb_lag))
        assert eb_values[1] < eb_values[0]


class TestCouplingForTargetCrossover:
    def test_matches_closed_form(self) -> None:
        for d in (0.15, 0.25, 0.35, 0.45):
            for tau_c in (8.0, 16.0, 32.0, 64.0):
                j = coupling_for_target_crossover(d, tau_c)
                assert j == pytest.approx(tau_c ** (-d))

    def test_rejects_d_zero(self) -> None:
        with pytest.raises(ValueError, match="no crossover"):
            coupling_for_target_crossover(0.0, 16.0)


class TestDeviationSpectrum:
    def test_zero_coupling_reduces_to_arfima_spectrum(self) -> None:
        # At J = 0, |z^d + 0|^2 = |z|^{2d}, i.e. the plain ARFIMA(0,d,0)
        # spectral density -- no separate closed form needed to check this,
        # it falls out of the formula directly.
        omega = np.array([0.5, 1.0, 2.0])
        d = 0.25
        S = deviation_spectrum(omega, d, 0.0)
        z = 1.0 - np.exp(-1j * omega)
        expected = 1.0 / (2.0 * np.pi * np.abs(z**d) ** 2)
        assert np.allclose(S, expected)

    def test_low_frequency_limit_for_positive_d(self) -> None:
        # d > 0: S_delta(0) -> sigma^2 / (2 pi J^2) as omega -> 0, but the
        # fractional term z^d ~ omega^d vanishes only algebraically, so
        # convergence is slow (~6% off even at omega = 1e-8 for d = 0.25) --
        # loose tolerance is the correct expectation here, not a bug.
        j = 0.3
        omega = np.array([1e-8])
        S = deviation_spectrum(omega, 0.25, j)
        assert S[0] == pytest.approx(1.0 / (2.0 * np.pi * j**2), rel=0.1)

    def test_rejects_omega_out_of_range(self) -> None:
        with pytest.raises(ValueError, match="omega must lie"):
            deviation_spectrum(np.array([0.0]), 0.25, 0.3)
        with pytest.raises(ValueError, match="omega must lie"):
            deviation_spectrum(np.array([4.0]), 0.25, 0.3)


@pytest.mark.slow
class TestWhittleDjEstimate:
    def test_recovers_known_d_and_j(self) -> None:
        n_particles, n_steps, k = 300, 512, 256
        d, j = 0.3, 0.05
        cfg = MeanFieldConfig(
            n_particles=n_particles, n_steps=n_steps, K=k, d=d, J=j,
            burn_in=4 * k, seed=31,
        )
        sim = simulate_meanfield(cfg, np.random.default_rng(31))
        delta = sim.velocities[:, :, 0] - sim.mean_field[None, :]
        n = delta.shape[1]
        fft_vals = np.fft.rfft(delta - delta.mean(axis=1, keepdims=True), axis=1)[:, 1:]
        freqs = 2.0 * np.pi * np.arange(1, fft_vals.shape[1] + 1) / n
        periodogram = (np.abs(fft_vals) ** 2 / (2.0 * np.pi * n)).mean(axis=0)

        d_hat, j_hat = whittle_dj_estimate(freqs, periodogram)
        assert d_hat == pytest.approx(d, abs=0.03)
        assert j_hat == pytest.approx(j, abs=0.03)

    def test_multistart_beats_single_start_on_a_known_bad_case(self) -> None:
        # d = 0.35, J = 0 previously converged to d_hat = 0.38 from a
        # single L-BFGS-B run started at the bounds' midpoint; multi-start
        # must not regress back to that local optimum.
        n_particles, n_steps, k = 400, 1024, 1024
        d, j = 0.35, 0.0
        cfg = MeanFieldConfig(
            n_particles=n_particles, n_steps=n_steps, K=k, d=d, J=j,
            burn_in=4 * k, seed=31,
        )
        sim = simulate_meanfield(cfg, np.random.default_rng(31))
        delta = sim.velocities[:, :, 0] - sim.mean_field[None, :]
        n = delta.shape[1]
        fft_vals = np.fft.rfft(delta - delta.mean(axis=1, keepdims=True), axis=1)[:, 1:]
        freqs = 2.0 * np.pi * np.arange(1, fft_vals.shape[1] + 1) / n
        periodogram = (np.abs(fft_vals) ** 2 / (2.0 * np.pi * n)).mean(axis=0)

        d_hat, _ = whittle_dj_estimate(freqs, periodogram)
        assert abs(d_hat - d) < 0.02


@pytest.mark.slow
class TestWhittleDOnlyEstimate:
    def test_matches_dj_estimate_at_j_zero(self) -> None:
        # At J = 0 there is no coupling to be blind to, so the naive and
        # model-aware fits on the same data should agree closely.
        n_particles, n_steps, k = 400, 1024, 1024
        d = 0.25
        cfg = MeanFieldConfig(
            n_particles=n_particles, n_steps=n_steps, K=k, d=d, J=0.0,
            burn_in=4 * k, seed=41,
        )
        sim = simulate_meanfield(cfg, np.random.default_rng(41))
        v = sim.velocities[:, :, 0]
        n = v.shape[1]
        fft_vals = np.fft.rfft(v - v.mean(axis=1, keepdims=True), axis=1)[:, 1:]
        freqs = 2.0 * np.pi * np.arange(1, fft_vals.shape[1] + 1) / n
        periodogram = (np.abs(fft_vals) ** 2 / (2.0 * np.pi * n)).mean(axis=0)

        d_naive = whittle_d_only_estimate(freqs, periodogram)
        d_aware, _ = whittle_dj_estimate(freqs, periodogram)
        assert d_naive == pytest.approx(d_aware, abs=0.01)
        assert d_naive == pytest.approx(d, abs=0.02)

    @pytest.mark.parametrize("d", [0.25, -0.25])
    def test_biased_by_coupling_regardless_of_true_sign(self, d: float) -> None:
        # The naive fit's bias runs toward antipersistence regardless of the
        # sign of the true d (found empirically, not assumed -- see
        # whittle_d_only_estimate's docstring and NOTES.md step 2). The
        # model-aware fit on the same data stays near the true d either way;
        # that contrast is step 3's whole point, not just prose.
        n_particles, n_steps, k = 400, 1024, 1024
        cfg = MeanFieldConfig(
            n_particles=n_particles, n_steps=n_steps, K=k, d=d, J=0.35,
            burn_in=4 * k, seed=41,
        )
        sim = simulate_meanfield(cfg, np.random.default_rng(41))
        v = sim.velocities[:, :, 0]
        n = v.shape[1]
        fft_vals = np.fft.rfft(v - v.mean(axis=1, keepdims=True), axis=1)[:, 1:]
        freqs = 2.0 * np.pi * np.arange(1, fft_vals.shape[1] + 1) / n
        periodogram = (np.abs(fft_vals) ** 2 / (2.0 * np.pi * n)).mean(axis=0)

        d_naive = whittle_d_only_estimate(freqs, periodogram)
        d_aware, _ = whittle_dj_estimate(freqs, periodogram)
        assert d_naive < d_aware - 0.1
        assert d_aware == pytest.approx(d, abs=0.02)


@pytest.mark.slow
class TestWindowedEaMsdRegimeAsymmetry:
    def test_typical_agent_regime_shows_predicted_sign_asymmetry(self) -> None:
        # PROJECT.md section 7: coupling suppresses superdiffusive memory
        # (d > 0, exponent should fall toward 1 as J grows) but not
        # subdiffusive memory (d < 0, exponent should stay near 1 + 2d). A
        # windowed (not local -- step 1 V4's trap) ensemble EA-MSD fit is
        # model-free, so it can't inherit whittle_d_only_estimate's
        # misspecification bias, which declines for both signs.
        n_particles, n_steps, k = 400, 1024, 1024
        burn_in = 4 * k
        max_lag = k // 10
        t = np.arange(n_steps, dtype=np.float64)

        def windowed_exponent(d: float, j: float) -> float:
            cfg = MeanFieldConfig(
                n_particles=n_particles, n_steps=n_steps, K=k, d=d, J=j,
                burn_in=burn_in, seed=41,
            )
            sim = simulate_meanfield(cfg, np.random.default_rng(41))
            ea = ea_msd(sim.positions)
            return fit_powerlaw(t, ea, 2.0, float(max_lag), fit=None).exponent

        e_pos_weak = windowed_exponent(0.25, 0.0)
        e_pos_strong = windowed_exponent(0.25, 0.5)
        e_neg_weak = windowed_exponent(-0.25, 0.0)
        e_neg_strong = windowed_exponent(-0.25, 0.5)

        assert e_pos_strong < e_pos_weak - 0.15  # d > 0: falls with J
        assert abs(e_neg_strong - e_neg_weak) < 0.1  # d < 0: stays flat
