"""Tests for Phase 2 mechanism identifiability, dataset, classifiers, ablation, and conditional estimation."""

from __future__ import annotations

import numpy as np
import pytest

from collectivediff.config import CTRWConfig, FitConfig, LevyWalkConfig, SBMConfig
from collectivediff.estimators.msd import fit_powerlaw, log_spaced_lags, ta_msd
from collectivediff.generators import generate
from collectivediff.inference import (
    MECHANISM_NAMES,
    DatasetConfig,
    compute_pairwise_error_matrix,
    count_levy_walk_flights,
    draw_random_config,
    estimate_exponent_conditional,
    estimate_exponent_ctrw,
    estimate_exponent_levy_walk,
    estimate_exponent_sbm,
    fit_and_evaluate_classifiers,
    generate_dataset,
    run_ablation_study,
    run_pairwise_feature_ranking,
)


class TestDatasetGeneration:
    def test_continuous_parameter_generation(self) -> None:
        rng = np.random.default_rng(100)
        params_drawn = []
        for _ in range(20):
            _, p = draw_random_config("fbm", n_steps=64, rng=rng, extrapolate=False)
            params_drawn.append(p["hurst"])

        # Check continuous: values should all be distinct
        assert len(set(params_drawn)) == len(params_drawn)
        assert all(0.15 <= h <= 0.85 for h in params_drawn)

    def test_extrapolation_parameters_outside_range(self) -> None:
        rng = np.random.default_rng(200)
        for _ in range(20):
            _, p = draw_random_config("ctrw", n_steps=64, rng=rng, extrapolate=True)
            a = p["alpha_wait"]
            assert (0.08 <= a <= 0.18) or (0.87 <= a <= 0.95)

    def test_dataset_generation_dimensions_and_labels(self) -> None:
        cfg = DatasetConfig(n_samples_per_class=5, n_steps=64, split="train", seed=10)
        ds = generate_dataset(cfg)
        assert ds.features.shape == (30, len(ds.feature_names))
        assert len(ds.labels) == 30
        assert set(ds.labels) == set(range(6))
        assert np.allclose(ds.class_prior, 1.0 / 6.0)


class TestClassificationAndPairwise:
    def test_pairwise_error_matrix_properties(self) -> None:
        # Construct synthetic confusion matrix
        # 3 classes, 100 samples each
        cm = np.array(
            [
                [90, 8, 2],
                [10, 85, 5],
                [0, 4, 96],
            ]
        )
        p_err = compute_pairwise_error_matrix(cm)
        assert p_err.shape == (3, 3)
        assert np.allclose(p_err, p_err.T)  # symmetric
        assert np.all(np.diag(p_err) == 0.0)  # zero diagonal
        # Pair (0, 1): (8 + 10) / (100 + 100) = 18 / 200 = 0.09
        assert np.isclose(p_err[0, 1], 0.09)

    def test_classifiers_train_and_evaluate(self) -> None:
        cfg_tr = DatasetConfig(n_samples_per_class=10, n_steps=64, split="train", seed=10)
        cfg_te = DatasetConfig(n_samples_per_class=10, n_steps=64, split="test_interp", seed=20)
        cfg_ex = DatasetConfig(n_samples_per_class=10, n_steps=64, split="test_extrap", seed=30)

        ds_tr = generate_dataset(cfg_tr)
        ds_te = generate_dataset(cfg_te)
        ds_ex = generate_dataset(cfg_ex)

        results = fit_and_evaluate_classifiers(ds_tr, ds_te, ds_ex, seed=42)
        assert set(results.keys()) == {"logistic_regression", "lda", "gradient_boosting"}

        for name, res in results.items():
            assert 0.0 <= res.interp_accuracy <= 1.0
            assert 0.0 <= res.extrap_accuracy <= 1.0
            assert res.confusion_matrix_interp.shape == (6, 6)


class TestAblationAndRanking:
    def test_ablation_runs_and_returns_valid_metrics(self) -> None:
        cfg_tr = DatasetConfig(n_samples_per_class=8, n_steps=64, split="train", seed=1)
        cfg_te = DatasetConfig(n_samples_per_class=8, n_steps=64, split="test_interp", seed=2)
        ds_tr = generate_dataset(cfg_tr)
        ds_te = generate_dataset(cfg_te)

        res = run_ablation_study(ds_tr, ds_te, clf_name="logistic_regression")
        assert 0.0 <= res.baseline_accuracy <= 1.0
        assert len(res.leave_one_out_individual) == len(ds_tr.feature_names)
        assert len(res.single_feature_groups) == 7

    def test_pairwise_ranking_separates_fbm_and_ctrw(self) -> None:
        cfg_tr = DatasetConfig(n_samples_per_class=15, n_steps=128, split="train", seed=10)
        cfg_te = DatasetConfig(n_samples_per_class=15, n_steps=128, split="test_interp", seed=20)
        ds_tr = generate_dataset(cfg_tr)
        ds_te = generate_dataset(cfg_te)

        rank_res = run_pairwise_feature_ranking(ds_tr, ds_te, "fbm", "ctrw")
        # Immobility or ACF should be among top features separating CTRW and fBm
        top_3 = [name for name, _ in rank_res.ranked_features[:3]]
        assert any("immobile" in name or "acf" in name or "gap" in name for name in top_3)


class TestConditionalEstimation:
    def test_sbm_exponent_recovered_from_nonstationarity(self) -> None:
        rng = np.random.default_rng(42)
        # SBM with alpha = 0.5
        traj = generate(SBMConfig(n_particles=1, n_steps=2048, alpha=0.5), rng)
        alpha_hat = estimate_exponent_sbm(traj)
        assert np.isclose(alpha_hat, 0.5, atol=0.20)

    def test_ctrw_exponent_recovered_from_pauses(self) -> None:
        rng = np.random.default_rng(42)
        # CTRW with a = 0.6
        traj = generate(CTRWConfig(n_particles=1, n_steps=4096, alpha_wait=0.6), rng)
        a_hat = estimate_exponent_ctrw(traj)
        assert 0.3 <= a_hat <= 0.9

    def test_conditional_dispatcher(self) -> None:
        rng = np.random.default_rng(42)
        traj = generate(SBMConfig(n_particles=1, n_steps=512, alpha=0.7), rng)
        # Dispatching as SBM gives non-stationary estimate, as Brownian gives 1.0
        assert estimate_exponent_conditional(traj, "brownian") == 1.0
        assert estimate_exponent_conditional(traj, "ddm") == 1.0
        assert 0.4 <= estimate_exponent_conditional(traj, "sbm") <= 1.2

    def test_levy_walk_correction_improves_on_naive_mae(self) -> None:
        """GLS-on-TA-MSD and a moment-spectrum kink fit were both tried for
        this mechanism and both came out *worse* than the naive Phase 1 fit on
        a held-out set (documented in ``PROJECT.md``); this asserts the
        flight-duration correction that replaced them actually clears that
        bar, rather than asserting it reaches any kind of information floor --
        the Levy walk has no closed-form Fisher information to be close to.
        """
        n_steps = 2048
        rng = np.random.default_rng(2024)
        n_samples = 40
        naive_errors = []
        corrected_errors = []
        for _ in range(n_samples):
            g = float(rng.uniform(1.2, 1.8))
            traj_seed = int(rng.integers(1, 2**31 - 1))
            cfg = LevyWalkConfig(n_particles=1, n_steps=n_steps, gamma=g, seed=traj_seed)
            traj = generate(cfg, np.random.default_rng(traj_seed))
            alpha_true = 3.0 - g

            lags = log_spaced_lags(n_steps, FitConfig())
            naive_fit = fit_powerlaw(
                lags.astype(np.float64),
                ta_msd(traj, lags),
                float(lags[0]),
                float(lags[-1]),
                fit=FitConfig(n_bootstrap=0),
            )
            naive_alpha = float(np.clip(naive_fit.exponent, 1.05, 1.95))
            corrected_alpha = estimate_exponent_levy_walk(traj)

            naive_errors.append(abs(naive_alpha - alpha_true))
            corrected_errors.append(abs(corrected_alpha - alpha_true))

        assert np.mean(corrected_errors) < np.mean(naive_errors)

    def test_levy_walk_flight_count_stays_usable_across_gamma_range(self) -> None:
        """The Hill estimate blended into the correction is only as good as
        the flight count behind it, which shrinks as ``g -> 1`` (long
        flights, few of them). Check it does not collapse to an unusable
        handful anywhere in the range Step 6 actually draws from.
        """
        n_steps = 2048
        rng = np.random.default_rng(7)
        for g in (1.2, 1.5, 1.8):
            counts = [
                count_levy_walk_flights(
                    generate(
                        LevyWalkConfig(n_particles=1, n_steps=n_steps, gamma=g, seed=s),
                        np.random.default_rng(s),
                    )
                )
                for s in rng.integers(1, 2**31 - 1, size=10)
            ]
            assert min(counts) >= 10
