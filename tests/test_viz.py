"""Tests for the visualisation layer and the composed studies.

A figure cannot be asserted correct the way an exponent can, so these check the
properties that *are* checkable and that would silently rot otherwise: the
palette invariants, the no-``show()`` rule, that every figure function returns a
figure, and that the animation modes produce the frame counts and artists they
claim. The scientific content of the figures is tested through the estimators
they call, not here.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest
from matplotlib.figure import Figure

from collectivediff.config import (
    BrownianConfig,
    CTRWConfig,
    DDMConfig,
    FBMConfig,
    FitConfig,
    LevyWalkConfig,
    SBMConfig,
)
from collectivediff.estimators import increment_acf, moment_spectrum
from collectivediff.estimators.ergodicity import ctrw_eb_plateau
from collectivediff.generators import GENERATORS, generate
from collectivediff.studies import (
    ensemble_msd_curve,
    exponent_spread_study,
    mechanism_diagnostics,
    single_trajectory_report,
)
from collectivediff.viz import (
    STYLES,
    animate_trajectories,
    palette,
    plot_amplitude_scatter,
    plot_exponent_spread,
    plot_increment_acf,
    plot_moment_spectrum,
    plot_msd,
    plot_single_trajectory_panel,
    plot_trajectories,
    plot_van_hove,
)
from collectivediff.viz.animate import _frame_indices, save_animation


def run(cfg) -> np.ndarray:
    return generate(cfg, np.random.default_rng(cfg.seed))


@pytest.fixture(autouse=True)
def _close_figures():
    """Close every figure after each test.

    Figures created through ``pyplot`` stay in its global registry until closed,
    so a suite that makes a few dozen of them warns about a leak and holds their
    memory. The same applies to any loop over mechanisms in a notebook, which is
    why the viz package documents it.
    """
    yield
    import matplotlib.pyplot as plt

    plt.close("all")


class TestPalette:
    """The one-colour-per-entity rule, enforced rather than trusted."""

    def test_every_registered_mechanism_has_a_colour(self) -> None:
        """A new generator must not be able to appear without a hue.

        Otherwise the first figure to plot it either crashes or silently borrows
        another mechanism's colour, which is worse.
        """
        assert set(palette.MECHANISM_COLOUR) == set(GENERATORS)
        assert set(palette.MECHANISM_LABEL) >= set(GENERATORS)

    def test_colours_are_distinct(self) -> None:
        assert len(set(palette.MECHANISM_COLOUR.values())) == len(
            palette.MECHANISM_COLOUR
        )

    def test_unknown_mechanism_raises_rather_than_defaulting(self) -> None:
        with pytest.raises(KeyError, match="no colour assigned"):
            palette.mechanism_colour("not_a_mechanism")

    def test_accepts_a_config_or_a_name(self) -> None:
        cfg = FBMConfig(hurst=0.7)
        assert palette.mechanism_colour(cfg) == palette.mechanism_colour("fbm")
        assert palette.mechanism_label(cfg) == "fBm"

    def test_categorical_order_is_the_validated_one(self) -> None:
        """The separation guarantees were measured on adjacent pairs in this order.

        Reordering silently invalidates them, so the sequence is pinned.
        """
        assert palette.CATEGORICAL[:4] == ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
        assert palette.MECHANISM_COLOUR["brownian"] == palette.CATEGORICAL[0]

    def test_rc_params_paint_the_surface_explicitly(self) -> None:
        """A transparent figure would inherit an unreadable dark notebook theme."""
        rc = palette.rc_params()
        assert rc["figure.facecolor"] == palette.SURFACE
        assert rc["axes.facecolor"] == palette.SURFACE
        assert rc["savefig.facecolor"] == palette.SURFACE


class TestStaticFigures:
    """Every function returns a figure and none of them calls ``show()``."""

    @pytest.fixture(scope="class")
    def small(self) -> dict:
        cfg = BrownianConfig(n_particles=200, n_steps=512, seed=0)
        return {"cfg": cfg, "x": run(cfg)}

    def test_no_figure_function_calls_show(self, monkeypatch, small) -> None:
        """``show()`` in library code would block a script and duplicate output."""
        import matplotlib.pyplot as plt

        def explode(*args, **kwargs):
            raise AssertionError("a figure function called show()")

        monkeypatch.setattr(plt, "show", explode)
        cfg, x = small["cfg"], small["x"]
        t, msd, fit = ensemble_msd_curve(cfg, fit=FitConfig(n_bootstrap=0))
        assert isinstance(plot_msd({"brownian": (t, msd)}, {"brownian": fit}), Figure)
        assert isinstance(plot_trajectories(x, "brownian"), Figure)
        assert isinstance(plot_van_hove(x, [16, 128], "brownian"), Figure)

    def test_msd_shades_the_fit_window(self, small) -> None:
        """The shaded band is part of the claim, not decoration."""
        cfg = small["cfg"]
        t, msd, fit = ensemble_msd_curve(cfg, fit=FitConfig(n_bootstrap=0))
        fig = plot_msd({"brownian": (t, msd)}, {"brownian": fit})
        ax = fig.axes[0]
        assert len(ax.patches) >= 1
        assert ax.get_xscale() == "log" and ax.get_yscale() == "log"

    def test_overlapping_fit_windows_are_merged(self, small) -> None:
        """Two identical windows must shade once, not twice.

        Stacking the alpha would invent a darker band that means nothing.
        """
        cfg = small["cfg"]
        t, msd, fit = ensemble_msd_curve(cfg, fit=FitConfig(n_bootstrap=0))
        fig = plot_msd(
            {"brownian": (t, msd), "fbm": (t, msd)},
            {"brownian": fit, "fbm": fit},
        )
        assert len(fig.axes[0].patches) == 1

    def test_legend_present_for_multiple_series(self, small) -> None:
        """Identity is never carried by colour alone."""
        cfg = small["cfg"]
        t, msd, _ = ensemble_msd_curve(cfg, fit=FitConfig(n_bootstrap=0))
        one = plot_msd({"brownian": (t, msd)})
        two = plot_msd({"brownian": (t, msd), "fbm": (t, msd)})
        assert one.axes[0].get_legend() is None
        assert two.axes[0].get_legend() is not None

    def test_same_mechanism_twice_shares_colour_and_differs_in_style(self, small) -> None:
        """fBm at two Hurst values is one entity, drawn in one hue."""
        cfg = small["cfg"]
        t, msd, _ = ensemble_msd_curve(cfg, fit=FitConfig(n_bootstrap=0))
        fig = plot_msd({"fbm H=0.3": (t, msd), "fbm H=0.7": (t, msd * 2.0)})
        drawn = [line for line in fig.axes[0].lines]
        assert drawn[0].get_color() == drawn[1].get_color()
        assert drawn[0].get_linestyle() != drawn[1].get_linestyle()

    def test_van_hove_masks_empty_bins(self) -> None:
        """A zero count is no measurement; on a log axis it must not spike."""
        cfg = BrownianConfig(n_particles=300, n_steps=256, seed=0)
        fig = plot_van_hove(run(cfg), [128], "brownian", bins=201)
        ydata = fig.axes[0].lines[0].get_ydata()
        assert np.isnan(ydata).any()
        assert np.nanmin(ydata) > 0.0

    def test_amplitude_scatter_reports_eb(self) -> None:
        cfg = CTRWConfig(n_particles=400, n_steps=1024, alpha_wait=0.5, seed=0)
        fig = plot_amplitude_scatter(
            run(cfg), 8, "ctrw", cfg, plateau=ctrw_eb_plateau(0.5)
        )
        texts = " ".join(t.get_text() for t in fig.axes[0].texts)
        assert "EB" in texts and "theory" in texts

    def test_increment_acf_overlays_theory(self) -> None:
        cfg = FBMConfig(n_particles=400, n_steps=512, hurst=0.3, seed=0)
        fig = plot_increment_acf({"fbm H=0.3": increment_acf(run(cfg), 6)}, 0.3)
        labels = [line.get_label() for line in fig.axes[0].lines]
        assert any("theory" in str(label) for label in labels)

    def test_moment_spectrum_draws_the_straight_line_reference(self) -> None:
        """Without a flat guide, a gently rising curve and a bilinear one look alike."""
        cfg = LevyWalkConfig(n_particles=400, n_steps=1024, gamma=1.5, seed=0)
        spectrum = moment_spectrum(
            run(cfg), np.linspace(0.5, 3.0, 6), 40.0, cfg.duration,
            fit=FitConfig(n_bootstrap=0),
        )
        fig = plot_moment_spectrum({"levy_walk g=1.5": spectrum}, gamma_reference=1.5)
        assert isinstance(fig, Figure)
        assert len(fig.axes[0].lines) >= 2

    def test_exponent_spread_marks_the_truth(self) -> None:
        study, truths, _summary = exponent_spread_study(
            {"brownian": BrownianConfig(seed=0)},
            lengths=(128, 512),
            n_trajectories=20,
        )
        fig = plot_exponent_spread(study, truths)
        labels = [line.get_label() for line in fig.axes[0].lines]
        assert "true exponent" in labels

    def test_single_trajectory_panel_has_two_rows_per_mechanism(self) -> None:
        report = single_trajectory_report(
            {"brownian": BrownianConfig(seed=0), "ctrw a=0.5": CTRWConfig(seed=0)},
            n_steps=512,
        )
        fig = plot_single_trajectory_panel(report)
        assert len(fig.axes) == 4

    def test_two_dimensional_trajectories_use_equal_aspect(self) -> None:
        cfg = BrownianConfig(n_particles=5, n_steps=128, n_dim=2, seed=0)
        fig = plot_trajectories(run(cfg), "brownian")
        assert fig.axes[0].get_aspect() == 1.0


class TestAnimation:
    """Three modes, one implementation."""

    @pytest.mark.parametrize("mode", ["single", "few", "cloud"])
    def test_every_mode_builds_and_draws(self, mode: str) -> None:
        style = STYLES[mode]
        cfg = BrownianConfig(
            n_particles=max(style.n_particles, 1), n_steps=200, n_dim=2, seed=0
        )
        anim = animate_trajectories(run(cfg), mode, mechanism="brownian")
        artists = anim._func(50)
        assert artists
        assert isinstance(anim._fig, Figure)

    def test_frames_are_subsampled_for_long_walks(self) -> None:
        """A thousand-step walk must not become a thousand-frame clip."""
        assert len(_frame_indices(50, 240)) == 50
        frames = _frame_indices(10_000, 150)
        assert len(frames) <= 151
        assert frames[0] == 0 and frames[-1] == 9_999

    def test_cloud_mode_uses_a_scatter_trail_not_lines(self) -> None:
        """Thousands of per-particle line artists would cost far more than they show."""
        assert STYLES["cloud"].line_width == 0.0
        assert STYLES["cloud"].trail_stride > 1
        assert STYLES["single"].line_width > 0.0

    def test_handful_mode_gives_each_walker_its_own_hue(self) -> None:
        assert STYLES["few"].distinct_colours
        assert not STYLES["cloud"].distinct_colours

    def test_two_dimensional_axes_share_one_limit(self) -> None:
        """Independent x and y limits under equal aspect squash the box."""
        cfg = BrownianConfig(n_particles=1, n_steps=200, n_dim=2, seed=0)
        anim = animate_trajectories(run(cfg), "single", mechanism="brownian")
        ax = anim._fig.axes[0]
        assert ax.get_xlim() == ax.get_ylim()

    def test_one_dimensional_mode_plots_against_time(self) -> None:
        cfg = BrownianConfig(n_particles=1, n_steps=200, seed=0)
        anim = animate_trajectories(run(cfg), "single", mechanism="brownian")
        assert anim._fig.axes[0].get_xlim() == (0, 199)

    def test_rejects_wrong_layout(self) -> None:
        with pytest.raises(ValueError, match="n_particles, n_steps, n_dim"):
            animate_trajectories(np.zeros((4, 8)), "single")

    @pytest.mark.slow
    def test_gif_fallback_writes_a_file(self, tmp_path) -> None:
        """When ffmpeg is missing, writing nothing would be the worst outcome."""
        cfg = BrownianConfig(n_particles=1, n_steps=40, n_dim=2, seed=0)
        anim = animate_trajectories(run(cfg), "single", mechanism="brownian")
        written = save_animation(anim, tmp_path / "walk.gif", fps=8, dpi=50)
        assert written.exists() and written.suffix == ".gif"


class TestStudies:
    """The composed analyses notebooks call instead of defining logic themselves."""

    def test_single_trajectory_report_covers_every_mechanism(self) -> None:
        configs = {
            "brownian": BrownianConfig(seed=0),
            "fbm H=0.3": FBMConfig(hurst=0.3, seed=0),
            "sbm a=0.6": SBMConfig(alpha=0.6, seed=0),
            "ctrw a=0.5": CTRWConfig(alpha_wait=0.5, seed=0),
            "levy_walk g=1.5": LevyWalkConfig(gamma=1.5, seed=0),
        }
        report = single_trajectory_report(configs, n_steps=2048)
        assert set(report) == set(configs)
        for entry in report.values():
            assert entry["trajectory"].shape[0] == 1
            assert entry["ta_msd"].shape == entry["lags"].shape

    def test_non_ergodic_mechanisms_report_alpha_near_one(self) -> None:
        """The narrative of notebook 01, pinned as a test.

        A single CTRW or scaled-Brownian trajectory reports normal diffusion
        whatever its ensemble exponent is, because in both cases the
        time-averaged MSD is linear in the lag. If this ever stopped being true
        the notebook's argument would be wrong, so it is asserted rather than
        merely written down.
        """
        report = single_trajectory_report(
            {
                "ctrw a=0.5": CTRWConfig(alpha_wait=0.5, seed=0),
                "sbm a=0.6": SBMConfig(alpha=0.6, seed=0),
                "fbm H=0.3": FBMConfig(hurst=0.3, seed=0),
            },
            n_steps=4096,
        )
        assert report["ctrw a=0.5"]["exponent"] == pytest.approx(1.0, abs=0.15)
        assert report["sbm a=0.6"]["exponent"] == pytest.approx(1.0, abs=0.15)
        # The ergodic control does find its own exponent.
        assert report["fbm H=0.3"]["exponent"] == pytest.approx(0.6, abs=0.15)

    @pytest.mark.slow
    def test_spread_narrows_for_both_but_the_ctrw_converges_to_the_wrong_answer(
        self,
    ) -> None:
        """The corrected closing result of phase 1 (``PHASE1_REVIEW.md`` item 1).

        The CTRW spread is not stuck: fitting ``IQR ~ T^-p`` gives ``p ~ 0.11``
        for CTRW against ``p ~ 0.26`` for Brownian motion -- about three times
        slower, not zero. What makes the CTRW result the important one is bias,
        not variance: its median *converges*, tightly, onto ``alpha ~ 1`` while
        the truth is 0.5. An observer with a longer trajectory becomes more
        confident and no more correct. That is a materially different, and
        stronger, claim than "the CTRW estimate stays noisy".
        """
        study, truths, summary = exponent_spread_study(
            {
                "brownian": BrownianConfig(seed=0),
                "ctrw a=0.5": CTRWConfig(alpha_wait=0.5, seed=0),
            },
            lengths=(128, 512, 2048, 8192),
            n_trajectories=200,
        )

        # Both narrow -- neither p is anywhere near zero.
        assert summary["brownian"]["p"] > 0.20
        assert summary["ctrw a=0.5"]["p"] > 0.05
        # But CTRW narrows distinctly slower: this is the comparison that matters,
        # not either absolute value.
        assert summary["brownian"]["p"] > 2.0 * summary["ctrw a=0.5"]["p"]

        # The CTRW median is not merely biased, it is converging on the wrong
        # value with growing confidence: bias stays large while IQR keeps falling.
        ctrw_bias = np.abs(summary["ctrw a=0.5"]["bias"])
        ctrw_iqr = summary["ctrw a=0.5"]["iqr"]
        assert ctrw_bias[-1] > 0.25
        assert ctrw_iqr[-1] < ctrw_iqr[0]

        # Brownian, by contrast, is unbiased throughout.
        assert all(abs(b) < 0.08 for b in summary["brownian"]["bias"])

        # No censored trajectory is silently dropped from the report.
        assert summary["ctrw a=0.5"]["censored"][0] > 0.0
        assert summary["brownian"]["censored"] == [0.0, 0.0, 0.0, 0.0]

        fig = plot_exponent_spread(study, truths)
        assert isinstance(fig, Figure)

    def test_mechanism_diagnostics_returns_one_consistent_ensemble(self) -> None:
        """Every panel must describe the same ensemble, not several seeds."""
        cfg = FBMConfig(n_particles=200, n_steps=512, hurst=0.7, seed=0)
        result = mechanism_diagnostics(
            cfg, q_values=np.array([1.0, 2.0]), fit=FitConfig(n_bootstrap=0)
        )
        assert result["trajectories"].shape == cfg.shape
        assert result["msd"].shape == (cfg.n_steps,)
        assert result["ea_ta_msd"].shape == result["lags"].shape
        assert result["spectrum"].nu.shape == (2,)

    def test_ensemble_msd_curve_picks_a_late_window_for_slow_mechanisms(self) -> None:
        """CTRW and the Levy walk need the early decade excluded; others do not."""
        ctrw = CTRWConfig(n_particles=200, n_steps=1024, alpha_wait=0.5, seed=0)
        brownian = BrownianConfig(n_particles=200, n_steps=1024, seed=0)
        _, _, slow = ensemble_msd_curve(ctrw, fit=FitConfig(n_bootstrap=0))
        _, _, fast = ensemble_msd_curve(brownian, fit=FitConfig(n_bootstrap=0))
        assert slow.window[0] > 100.0
        assert fast.window[0] == 1.0
