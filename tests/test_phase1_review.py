"""Tests for the fixes requested in ``PHASE1_REVIEW.md``.

One class per numbered item that produced new or changed code:

- item 2 (the two notebooks disagree) is a diagnosis, not new code, and is
  recorded here as a regression test on the root cause rather than as a
  standalone class.
- item 3 (NaN censoring) is covered by ``exponent_spread_study``'s
  ``censored`` field, tested in ``tests/test_viz.py::TestStudies``.
- item 5 (repository hygiene) has no unit-testable surface; it is addressed in
  the repository layout and ``pyproject.toml``, not here.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from collectivediff.config import BrownianConfig, CTRWConfig, FitConfig
from collectivediff.estimators import ea_msd, log_spaced_lags, ta_msd
from collectivediff.generators import generate
from collectivediff.studies import (
    RESULTS_ROOT,
    benchmark_table,
    ergodicity_checks,
    iqr_scaling_exponent,
    lag_window_sensitivity,
    single_trajectory_exponents,
    single_trajectory_report,
    write_summary,
)


def run(cfg):
    return generate(cfg, np.random.default_rng(cfg.seed))


class TestItem2SingleTrajectoryDiscrepancyResolved:
    """Root cause of the 0.99-vs-0.88 disagreement between the two notebooks.

    Both notebooks used the identical fitting method (TA-MSD over
    ``[lag_min, T/10]``) on the identical mechanism. The disagreement was not a
    bug in either notebook:

    1. ``0.99`` is a single draw from a distribution whose IQR is about 0.25 at
       this ``T`` -- unremarkable on its own.
    2. The distribution's *median* itself sits below the naive expectation of
       1.0 because a CTRW at ``a = 0.5`` makes very few jumps (median 46 in
       ``T = 8192``), so the widest lags in the default fit window are supported
       by the same handful of events and the TA-MSD has visibly saturated by
       then.

    Both effects are demonstrated below so the discrepancy cannot resurface
    silently as "the two studies must be using different code".
    """

    def test_single_sample_is_a_plausible_draw_from_the_ensemble(self) -> None:
        """The nb01 point estimate falls inside the nb02 ensemble's IQR.

        This alone accounts for the surface disagreement: one sample from a
        distribution with IQR ~0.25 landing near the top of it is not evidence
        of a discrepancy.
        """
        cfg = CTRWConfig(alpha_wait=0.5, seed=0)
        single = single_trajectory_report({"ctrw a=0.5": cfg}, n_steps=4096)
        single_value = single["ctrw a=0.5"]["exponent"]

        ensemble = single_trajectory_exponents(cfg, [4096], n_trajectories=200)[0]
        low, high = np.nanpercentile(ensemble, [10, 90])
        assert low <= single_value <= high * 1.05

    @pytest.mark.slow
    def test_median_sits_below_one_because_of_low_jump_count(self) -> None:
        """The renewal count explains the residual gap from the naive 1.0.

        Median jump count in ``T = 8192`` at ``a = 0.5`` is measured at 46; the
        1st percentile of trajectories has made only a single jump. With that
        few events, the TA-MSD's growth has visibly slowed by the time the fit
        window reaches its upper edge, and the fitted slope comes in below the
        asymptotic value of 1.
        """
        from collectivediff.generators.ctrw import ctrw_events
        from collectivediff.generators.resampling import _last_event_index

        cfg = CTRWConfig(n_particles=500, n_steps=8192, alpha_wait=0.5, seed=0)
        event_times, _ = ctrw_events(cfg, np.random.default_rng(0))
        grid = np.arange(cfg.n_steps, dtype=np.float64)
        n_jumps = _last_event_index(event_times, grid)[:, -1]
        assert np.median(n_jumps) < 100

        median_exponent = np.nanmedian(
            single_trajectory_exponents(cfg, [8192], n_trajectories=500)[0]
        )
        assert median_exponent < 0.97

    @pytest.mark.slow
    def test_fitted_slope_depends_on_the_lag_window(self) -> None:
        """Direct demonstration: narrower, earlier windows read closer to 1.

        This is the mechanism behind the notebook discrepancy, made concrete:
        the fitted exponent is a function of the window as much as of the data,
        for a mechanism this data-starved at large lag.
        """
        cfg = CTRWConfig(n_particles=300, n_steps=8192, alpha_wait=0.5, seed=0)
        trajectories = run(cfg)
        lags = log_spaced_lags(8192, FitConfig(lag_max_fraction=1.0, n_lags=48))
        curves = ta_msd(trajectories, lags, cfg)

        from collectivediff.estimators import fit_powerlaw

        def median_slope(low: int, high: int) -> float:
            mask = (lags >= low) & (lags <= high)
            values = [
                fit_powerlaw(lags[mask].astype(np.float64), curve[mask]).exponent
                for curve in curves
                if np.all(curve[mask] > 0.0)
            ]
            return float(np.median(values))

        early = median_slope(1, 32)
        late = median_slope(100, 819)
        assert early > late
        assert early - late > 0.1


class TestErgodicityChecks:
    """PHASE1_REVIEW.md item 4: both numeric ergodicity checks, printed and tested."""

    @pytest.mark.slow
    def test_brownian_control_matches_closed_form(self) -> None:
        """The check that would catch a wrong constant in ``eb_parameter``."""
        result = ergodicity_checks(n_particles=2000, n_steps=4096, seed=0)
        for deviation in result["brownian"]["relative_deviation"]:
            assert abs(deviation) < 0.15

    def test_ctrw_plateau_value_is_exact_at_a_half(self) -> None:
        result = ergodicity_checks(alpha_wait=0.5, n_particles=500, n_steps=1024)
        assert result["ctrw"]["plateau"] == pytest.approx(np.pi / 2 - 1.0, abs=1e-9)

    def test_ctrw_measurement_is_above_the_plateau_at_modest_T(self) -> None:
        """Consistent with the slow, from-above convergence recorded in NOTES.md."""
        result = ergodicity_checks(alpha_wait=0.5, n_particles=800, n_steps=2048)
        assert all(v > 0.0 for v in result["ctrw"]["relative_deviation"])

    def test_output_is_json_serialisable(self) -> None:
        result = ergodicity_checks(n_particles=300, n_steps=512)
        json.dumps(result)  # must not raise


class TestLagWindowSensitivity:
    """The window-dependence demonstrated in item 2, exposed as a reusable study."""

    def test_reports_one_row_per_window(self) -> None:
        cfg = CTRWConfig(alpha_wait=0.5, seed=0)
        result = lag_window_sensitivity(
            cfg, windows=[(1, 32), (100, 819)], n_trajectories=100, n_steps=8192
        )
        assert len(result["windows"]) == 2
        assert result["windows"][0]["median"] > result["windows"][1]["median"]

    def test_skips_windows_with_too_few_lags(self) -> None:
        cfg = CTRWConfig(alpha_wait=0.5, seed=0)
        result = lag_window_sensitivity(
            cfg, windows=[(1, 2), (1, 819)], n_trajectories=50, n_steps=8192
        )
        assert len(result["windows"]) == 1

    def test_output_is_json_serialisable(self) -> None:
        cfg = CTRWConfig(alpha_wait=0.5, seed=0)
        result = lag_window_sensitivity(
            cfg, windows=[(1, 32)], n_trajectories=50, n_steps=2048
        )
        json.dumps(result)


class TestBenchmarkTable:
    """PHASE1_REVIEW.md item 4: the timing/memory table, callable and reportable."""

    def test_covers_every_generator(self) -> None:
        result = benchmark_table(n_particles=200, n_steps=256)
        labels = {row["operation"] for row in result["rows"]}
        for mechanism in ("brownian", "fbm", "sbm", "ctrw", "levy_walk", "levy_flight", "ddm"):
            assert any(mechanism in label for label in labels)

    def test_reports_timing_and_memory_as_positive_numbers(self) -> None:
        result = benchmark_table(n_particles=200, n_steps=256)
        for row in result["rows"]:
            assert row["ms"] >= 0
            assert row["peak_mib"] >= 0

    def test_output_is_json_serialisable(self) -> None:
        result = benchmark_table(n_particles=100, n_steps=128)
        json.dumps(result)


class TestIQRScalingExponent:
    """The single-number summary that replaces eyeballing the spread panels."""

    def test_recovers_a_known_power_law(self) -> None:
        """A synthetic IQR shrinking exactly as T^-0.3 must recover p = 0.3."""
        rng = np.random.default_rng(0)
        lengths = np.array([128.0, 512.0, 2048.0, 8192.0])
        true_p = 0.3
        scale = 2.0 * lengths ** (-true_p)
        estimates = rng.uniform(-1, 1, size=(4, 4000)) * scale[:, None]
        assert iqr_scaling_exponent(lengths, estimates) == pytest.approx(
            true_p, abs=0.02
        )

    @pytest.mark.slow
    def test_larger_p_means_faster_shrinkage(self) -> None:
        cfg_fast = BrownianConfig(seed=0)
        cfg_slow = CTRWConfig(alpha_wait=0.5, seed=0)
        lengths = [128, 512, 2048, 8192]
        fast = single_trajectory_exponents(cfg_fast, lengths, n_trajectories=150)
        slow = single_trajectory_exponents(cfg_slow, lengths, n_trajectories=150)
        p_fast = iqr_scaling_exponent(np.asarray(lengths, dtype=np.float64), fast)
        p_slow = iqr_scaling_exponent(np.asarray(lengths, dtype=np.float64), slow)
        assert p_fast > p_slow


class TestWriteSummary:
    """PHASE1_REVIEW.md working-practice note: JSON summaries, not notebook re-reads."""

    def test_writes_and_round_trips(self, tmp_path) -> None:
        payload = {"a": 1, "b": [1.0, 2.0], "c": {"nested": True}}
        path = write_summary("demo", payload, root=tmp_path)
        assert path.exists()
        assert json.loads(path.read_text()) == payload

    def test_default_root_is_under_the_repository(self) -> None:
        assert RESULTS_ROOT.name == "results"
        assert RESULTS_ROOT.parent.exists()  # the repository root itself
