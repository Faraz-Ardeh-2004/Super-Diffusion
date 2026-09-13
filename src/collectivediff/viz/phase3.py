"""Phase 3 static figures: the crossover picture, observer comparison, Q3 correlation.

Following ``PHASE3_PROMPT.md`` step 5. Each function takes the JSON payload a
``studies_phase3`` study already wrote (not raw simulation output -- the
figures read the same numbers the text report quotes) and returns a
:class:`~matplotlib.figure.Figure`. Colour and style follow
:mod:`collectivediff.viz.palette`, same as every other figure in the project.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray

from . import palette

__all__ = [
    "animate_meanfield_comparison",
    "plot_crossover_illustration",
    "plot_observer_bias_variance",
    "plot_q3_correlation",
]


def plot_crossover_illustration(payload: Mapping[str, Any]) -> Figure:
    """Local exponent vs lag, one panel per ``J``, agent curve with coherent overlay.

    Reads :func:`~collectivediff.studies_phase3.run_step1_v4_msd_illustration`'s
    payload. **Qualitative**, as that study's payload itself documents: a
    single-realization local-slope curve is noisy (``NOTES.md`` step 1 V4),
    which is exactly why it is not used as this phase's quantitative
    estimator -- but a noisy curve is still the right thing to *look at*
    for the crossover picture step 5 asks for.

    Parameters
    ----------
    payload : mapping
        ``step1_v4_msd_illustration.json`` contents.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    curves = payload["curves"]
    d = payload["d"]
    asymptote = payload["asymptote_1_plus_2d"]
    n = len(curves)

    with plt.rc_context(palette.rc_params()):
        fig, axes = plt.subplots(1, n, figsize=(4.4 * n, 4.0), sharey=True)
        if n == 1:
            axes = [axes]
        for ax, curve, colour in zip(axes, curves, palette.CATEGORICAL):
            lags = np.asarray(curve["lags"], dtype=np.float64)
            agent = np.asarray(curve["local_exponent"], dtype=np.float64)
            coherent = np.asarray(curve.get("local_exponent_coherent", []), dtype=np.float64)
            ax.semilogx(lags, agent, color=colour, lw=1.6, label="single agent")
            if coherent.size:
                ax.semilogx(
                    lags, coherent, color=palette.REFERENCE, lw=1.2, ls="--",
                    label="center of mass",
                )
            ax.axhline(asymptote, color=palette.GRID, lw=1.0, ls=":")
            ax.axhline(1.0, color=palette.GRID, lw=1.0, ls=":")
            ax.set_title(f"J = {curve['J']:.3g}")
            ax.set_xlabel("lag")
            ax.legend(frameon=False, fontsize=8, loc="upper right")
        axes[0].set_ylabel("local exponent")
        fig.suptitle(f"Crossover picture, d = {d:g} (asymptote 1+2d = {asymptote:g})")
        fig.tight_layout()
    return fig


def plot_observer_bias_variance(payload: Mapping[str, Any]) -> Figure:
    """Bias and variance vs ``J`` for the three observers, split by sign of ``d``.

    Reads :func:`~collectivediff.studies_phase3.run_step3_observers`'s
    payload. Two rows (bias, variance) x two columns (``d > 0``, ``d < 0``),
    one line per observer.

    Parameters
    ----------
    payload : mapping
        ``step3_observers.json`` contents.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    observers = [
        ("isolated_model_agnostic", "isolated, model-agnostic", palette.CATEGORICAL[0]),
        ("isolated_model_aware", "isolated, model-aware", palette.CATEGORICAL[1]),
        ("field_aware", "field-aware", palette.CATEGORICAL[2]),
        ("combination_agent_field", "agent + field combination", palette.CATEGORICAL[3]),
    ]
    d_entries = payload["results"]

    with plt.rc_context(palette.rc_params()):
        fig, axes = plt.subplots(2, len(d_entries), figsize=(5.2 * len(d_entries), 7.2), sharex=False)
        if len(d_entries) == 1:
            axes = axes.reshape(2, 1)
        for col, entry in enumerate(d_entries):
            d = entry["d"]
            j_vals = [p["J"] for p in entry["points"]]
            for key, label, colour in observers:
                bias = [p[key]["bias"] for p in entry["points"]]
                var = [p[key]["variance"] for p in entry["points"]]
                axes[0, col].plot(j_vals, bias, "o-", color=colour, lw=1.4, ms=4, label=label)
                axes[1, col].plot(j_vals, var, "o-", color=colour, lw=1.4, ms=4, label=label)
            axes[0, col].axhline(0.0, color=palette.GRID, lw=1.0)
            axes[0, col].set_title(f"d = {d:g}")
            axes[1, col].set_xlabel("J")
            axes[1, col].set_yscale("log")
        axes[0, 0].set_ylabel("bias(d_hat)")
        axes[1, 0].set_ylabel("variance(d_hat)")
        axes[0, -1].legend(frameon=False, fontsize=8, loc="best")
        fig.suptitle("Observer bias and variance vs coupling strength")
        fig.tight_layout()
    return fig


def plot_q3_correlation(payload: Mapping[str, Any]) -> Figure:
    """Q3: NRMSE (left) and correlation (right) vs ``J``, each observer.

    Reads :func:`~collectivediff.studies_phase3.run_step4_q3_correlation`'s
    payload. **NRMSE is the panel that actually distinguishes the
    observers.** A first version of this figure plotted only correlation,
    which came out 0.93-0.99 for every observer at every `J` and looked flat
    -- a property of `d` being drawn from a wide range (`U(-0.4, 0.4)`,
    which any estimator that gets the rough sign and scale right correlates
    well with), not a property of estimator quality. NRMSE isolates the
    actual unexplained error and is kept as the left, primary panel;
    correlation is kept on the right only for continuity with the earlier
    number.

    Parameters
    ----------
    payload : mapping
        ``step4_q3_correlation.json`` contents.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    series = [
        ("isolated_model_agnostic", "isolated, model-agnostic", palette.CATEGORICAL[0]),
        ("isolated_model_aware", "isolated, model-aware", palette.CATEGORICAL[1]),
        ("field_aware", "field-aware", palette.CATEGORICAL[2]),
        ("combination_agent_field", "agent + field combination", palette.CATEGORICAL[3]),
    ]
    points = payload["results"]
    j_vals = [p["J"] for p in points]

    with plt.rc_context(palette.rc_params()):
        fig, (ax_nrmse, ax_corr) = plt.subplots(1, 2, figsize=(11.0, 4.6))
        for key, label, colour in series:
            nrmse_vals = [p[key]["nrmse"] for p in points]
            corr_vals = [p[key]["correlation"] for p in points]
            ax_nrmse.plot(j_vals, nrmse_vals, "o-", color=colour, lw=1.6, ms=5, label=label)
            ax_corr.plot(j_vals, corr_vals, "o-", color=colour, lw=1.6, ms=5, label=label)
        ax_nrmse.set_xlabel("J")
        ax_nrmse.set_ylabel("NRMSE(d_hat, true coherent d)")
        ax_nrmse.set_title("primary: normalised RMSE (lower is better)")
        ax_nrmse.legend(frameon=False, fontsize=8, loc="best")

        corr_vals_all = [p[key]["correlation"] for p in points for key, _, _ in series]
        y_lo = max(0.0, min(corr_vals_all) - 0.05)
        ax_corr.axhline(1.0, color=palette.GRID, lw=1.0, ls=":")
        ax_corr.set_ylim(y_lo, 1.02)
        ax_corr.set_xlabel("J")
        ax_corr.set_ylabel("correlation(d_hat, true coherent d)")
        ax_corr.set_title("for continuity only -- see NRMSE")

        fig.suptitle("Q3: how well does one agent's inferred d track the collective's")
        fig.tight_layout()
    return fig


def animate_meanfield_comparison(
    positions_weak: NDArray[np.float64],
    positions_strong: NDArray[np.float64],
    com_weak: NDArray[np.float64],
    com_strong: NDArray[np.float64],
    j_weak: float = 0.0,
    j_strong: float = 0.3,
    highlight_index: int = 0,
    max_frames: int = 150,
    interval_ms: int = 40,
    figsize: tuple[float, float] = (9.6, 4.8),
    dpi: float = 90.0,
) -> Any:
    """Cloud of agents, one highlighted, center of mass marked, ``J`` weak vs strong side by side.

    ``PHASE3_PROMPT.md`` step 5's point made visible: at ``J = 0`` the
    highlighted agent wanders anomalously while the center of mass barely
    moves (amplitude ``~ 1/sqrt(N)``); at strong coupling the agent jitters
    closer to normally-diffusive while the center of mass still carries the
    memory (the conservation law, V2 -- its own regime is ``J``-independent).

    Parameters
    ----------
    positions_weak, positions_strong : ndarray of float64
        Shape ``(N, T, 2)`` -- two independent 1-D mean-field runs combined
        as x/y components (``PROJECT.md`` section 7: 2-D is for the
        animation only). Weak/strong refer to ``J``, not ``d``.
    com_weak, com_strong : ndarray of float64
        Shape ``(T, 2)``, the corresponding center-of-mass trajectories
        (:meth:`~collectivediff.dynamics.meanfield.MeanFieldTrajectories.center_of_mass`
        per component).
    j_weak, j_strong : float
        Coupling strengths, for the panel titles.
    highlight_index : int
        Which agent gets its own trail and marker.
    max_frames : int
        Frame count cap (reuses :mod:`collectivediff.viz.animate`'s striding).
    interval_ms : int
        Delay between frames.
    figsize : tuple of float
        Figure size in inches.
    dpi : float
        Animation resolution.

    Returns
    -------
    matplotlib.animation.FuncAnimation
    """
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    from .animate import _frame_indices, _limits

    n_steps = positions_weak.shape[1]
    frames = _frame_indices(n_steps, max_frames)
    xlim = _limits(np.concatenate([positions_weak[..., 0], positions_strong[..., 0]]))
    ylim = _limits(np.concatenate([positions_weak[..., 1], positions_strong[..., 1]]))

    with plt.rc_context(palette.rc_params()):
        fig, (ax_weak, ax_strong) = plt.subplots(1, 2, figsize=figsize, dpi=dpi)
        artists = {}
        for ax, title in ((ax_weak, f"J = {j_weak:g}"), (ax_strong, f"J = {j_strong:g}")):
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            ax.set_aspect("equal")
            ax.set_title(title)
            cloud = ax.scatter([], [], s=8, color=palette.CATEGORICAL[0], alpha=0.35, zorder=2)
            (trail,) = ax.plot([], [], color=palette.CATEGORICAL[1], lw=1.2, zorder=3)
            (marker,) = ax.plot([], [], "o", color=palette.CATEGORICAL[1], ms=7, zorder=4)
            (com_trail,) = ax.plot([], [], color=palette.REFERENCE, lw=1.6, ls="--", zorder=3)
            (com_marker,) = ax.plot([], [], "*", color=palette.REFERENCE, ms=12, zorder=4)
            artists[ax] = (cloud, trail, marker, com_trail, com_marker)
        fig.suptitle("Isolated agent vs. center of mass")
        fig.tight_layout()

        def update(frame_idx: int):
            i = frames[frame_idx]
            updated = []
            for ax, pos, com in (
                (ax_weak, positions_weak, com_weak),
                (ax_strong, positions_strong, com_strong),
            ):
                cloud, trail, marker, com_trail, com_marker = artists[ax]
                cloud.set_offsets(pos[:, i, :])
                trail.set_data(pos[highlight_index, : i + 1, 0], pos[highlight_index, : i + 1, 1])
                marker.set_data([pos[highlight_index, i, 0]], [pos[highlight_index, i, 1]])
                com_trail.set_data(com[: i + 1, 0], com[: i + 1, 1])
                com_marker.set_data([com[i, 0]], [com[i, 1]])
                updated.extend([cloud, trail, marker, com_trail, com_marker])
            return updated

        return FuncAnimation(
            fig, update, frames=len(frames), interval=interval_ms, blit=False
        )
