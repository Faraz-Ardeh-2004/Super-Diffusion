"""Tests for the Phase 3 step 2-5 study orchestration (``studies_phase3.py``)
and their figures (``viz/phase3.py``).

Small-scale structural checks only -- the physics itself (sign asymmetry,
Whittle recovery, V5) is validated at full scale in ``tests/test_phase3_meanfield.py``
and documented with real numbers in ``NOTES.md``. These tests exist to catch
plumbing breakage (wrong keys, wrong shapes, non-finite output), not to
re-derive the science on every run.
"""

from __future__ import annotations

import numpy as np
import pytest

from collectivediff.studies_phase3 import (
    combined_j_sweep,
    generate_step5_animation_data,
    run_step2_two_regimes,
    run_step3_observers,
    run_step3_residual_surface,
    run_step3_v5_check,
    run_step4_field_variance_vs_n,
    run_step4_k_scaling,
    run_step4_q3_correlation,
)
from collectivediff.viz.phase3 import (
    animate_meanfield_comparison,
    plot_crossover_illustration,
    plot_observer_bias_variance,
    plot_q3_correlation,
)


class TestCombinedJSweep:
    def test_starts_at_zero_and_is_sorted(self) -> None:
        sweep = combined_j_sweep(0.25)
        assert sweep[0] == 0.0
        assert sweep == sorted(sweep)
        assert all(j >= 0.0 for j in sweep)

    def test_symmetric_in_sign_of_d(self) -> None:
        assert combined_j_sweep(0.3) == combined_j_sweep(-0.3)


@pytest.mark.slow
class TestStep2ToStep4Plumbing:
    def test_step2_output_structure_and_asymmetry_direction(self) -> None:
        payload = run_step2_two_regimes(
            d_values=(0.25, -0.25), n_particles=100, n_steps=128, k=64,
        )
        assert set(payload) >= {"results_positive_d", "results_negative_d"}
        pos = payload["results_positive_d"][0]["points"]
        neg = payload["results_negative_d"][0]["points"]
        assert all(np.isfinite(p["alpha_typical_agent"]) for p in pos)
        assert all(np.isfinite(p["alpha_coherent"]) for p in pos + neg)
        # Qualitative direction only at this tiny scale (noisy) -- exact
        # magnitudes are checked at full scale elsewhere.
        assert pos[-1]["alpha_typical_agent"] <= pos[0]["alpha_typical_agent"] + 0.3

    def test_step3_output_structure(self) -> None:
        payload = run_step3_observers(
            d_values=(0.25,), n_particles=80, n_steps=128, k=64,
            n_realizations=2, n_agents_sampled=4,
        )
        point = payload["results"][0]["points"][0]
        for key in (
            "isolated_model_agnostic", "isolated_model_aware",
            "field_aware", "combination_agent_field",
        ):
            assert np.isfinite(point[key]["mean"])
            assert point[key]["variance"] >= 0.0

    def test_step3_v5_check_structure(self) -> None:
        payload = run_step3_v5_check(
            d_values=(0.25,), n_particles=80, n_steps=128, k=64,
            n_realizations=3, n_agents_sampled=4,
        )
        entry = payload["results"][0]
        assert entry["isolated_single_param"]["variance"] > 0.0
        assert entry["field"]["variance"] > 0.0
        assert entry["variance_ratio_isolated_over_field"] > 0.0

    def test_step4_metrics_well_formed(self) -> None:
        payload = run_step4_q3_correlation(
            j_values=(0.0, 0.2), n_particles=80, n_steps=128, k=64,
            n_draws=6, n_agents_sampled=2,
        )
        for point in payload["results"]:
            for key in (
                "isolated_model_agnostic", "isolated_model_aware",
                "field_aware", "combination_agent_field",
            ):
                m = point[key]
                assert -1.0 <= m["correlation"] <= 1.0
                assert m["nrmse"] >= 0.0
                assert m["r_squared"] <= 1.0

    def test_step3_residual_surface_structure(self) -> None:
        payload = run_step3_residual_surface(
            points=((0.3, 0.05), (-0.3, 0.05)),
            n_particles=60, n_steps=128, k=64,
            n_agents_sampled=8, grid_n=9,
        )
        assert len(payload["results"]) == 2
        for r in payload["results"]:
            assert 0 <= r["n_pinned"] <= r["n_agents_sampled"]
            assert np.isfinite(r["stats_including_pinned"]["mean"])
            assert len(r["objective_surface"]) == 9
            assert len(r["objective_surface"][0]) == 9
            assert isinstance(r["grid_matches_optimizer_result"], bool)

    def test_step4_field_variance_vs_n_structure(self) -> None:
        payload = run_step4_field_variance_vs_n(
            n_values=(20, 40), n_steps=128, k=64, n_realizations=4,
        )
        assert [r["N"] for r in payload["results"]] == [20, 40]
        for r in payload["results"]:
            assert r["variance"] > 0.0
            assert r["variance_ci95_low"] <= r["variance"] <= r["variance_ci95_high"]

    def test_step4_k_scaling_structure(self) -> None:
        payload = run_step4_k_scaling(
            k_values=(16, 32, 64), n_particles=10, n_steps=16, burn_in_multiple=2,
        )
        assert [r["K"] for r in payload["results"]] == [16, 32, 64]
        assert all(r["elapsed_seconds"] > 0.0 for r in payload["results"])
        assert np.isfinite(payload["loglog_slope_time_vs_k"])


@pytest.mark.slow
class TestPhase3Figures:
    def test_figures_and_animation_build_from_real_studies(self) -> None:
        import matplotlib
        matplotlib.use("Agg")

        step2 = run_step2_two_regimes(
            d_values=(0.25, -0.25), n_particles=80, n_steps=128, k=64,
        )
        step3 = run_step3_observers(
            d_values=(0.25, -0.25), n_particles=80, n_steps=128, k=64,
            n_realizations=2, n_agents_sampled=4,
        )
        step4 = run_step4_q3_correlation(
            j_values=(0.0, 0.2), n_particles=80, n_steps=128, k=64,
            n_draws=6, n_agents_sampled=2,
        )
        # plot_crossover_illustration reads the msd_illustration payload
        # shape directly; build a tiny stand-in rather than running the
        # expensive K=16384 study here.
        illustration_payload = {
            "d": 0.25,
            "asymptote_1_plus_2d": 1.5,
            "curves": [
                {
                    "J": j,
                    "lags": list(range(1, 20)),
                    "local_exponent": list(np.linspace(1.5, 1.0, 19)),
                    "local_exponent_coherent": list(np.full(19, 1.5)),
                }
                for j in (0.0, 0.3)
            ],
        }

        fig1 = plot_crossover_illustration(illustration_payload)
        fig2 = plot_observer_bias_variance(step3)
        fig3 = plot_q3_correlation(step4)
        assert fig1 is not None and fig2 is not None and fig3 is not None

        anim_data = generate_step5_animation_data(
            d=0.25, n_particles=10, n_steps=30, k=32,
        )
        anim = animate_meanfield_comparison(
            anim_data["positions_weak"], anim_data["positions_strong"],
            anim_data["com_weak"], anim_data["com_strong"], max_frames=5,
        )
        assert anim is not None
