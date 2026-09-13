"""Diagnostic figures for Phase 2: mechanism identifiability, confusion, and exponent recovery.

Following project conventions:
- Every function returns its Figure and never calls plt.show().
- Uses palette styling, colors, and fonts.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray

from . import palette
from .static import _figure

__all__ = [
    "plot_conditional_exponent_comparison",
    "plot_confusion_matrix_family",
    "plot_feature_importance_ranking",
    "plot_information_floor_efficiency",
    "plot_pairwise_error_vs_length",
]


def plot_confusion_matrix_family(
    results_by_length: Mapping[int | str, Mapping[str, Any]],
    classifier_name: str = "gradient_boosting",
    lengths: Sequence[int] = (128, 512, 2048, 8192),
    mechanism_names: Sequence[str] | None = None,
) -> Figure:
    """Plot a grid of confusion matrices indexed by trajectory length T."""
    if mechanism_names is None:
        mechanism_names = ["brownian", "fbm", "ctrw", "sbm", "ddm", "levy_walk"]
    short_labels = [palette.MECHANISM_LABEL.get(m, m) for m in mechanism_names]

    n_panels = len(lengths)
    with plt.rc_context(palette.rc_params()):
        fig, axes = plt.subplots(1, n_panels, figsize=(3.6 * n_panels, 3.4), sharey=True)
    if n_panels == 1:
        axes = [axes]
    fig.patch.set_facecolor(palette.SURFACE)

    for ax, length in zip(axes, lengths):
        step_dict = results_by_length[str(length)]
        clf_dict = step_dict[classifier_name]
        cm = np.array(clf_dict["confusion_matrix_interp"], dtype=np.float64)
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_norm = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums > 0)

        im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1.0)
        ax.set_title(f"$T = {length}$ (Acc: {clf_dict['interp_accuracy']:.2f})", fontsize=10)
        ax.set_xticks(range(len(short_labels)))
        ax.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(short_labels)))
        ax.set_yticklabels(short_labels, fontsize=8)
        ax.set_xlabel("Predicted")

        # Annotate cell values
        for i in range(len(short_labels)):
            for j in range(len(short_labels)):
                val = cm_norm[i, j]
                text_col = "white" if val > 0.55 else palette.TEXT_PRIMARY
                ax.text(
                    j,
                    i,
                    f"{val:.2f}" if val >= 0.01 else "",
                    ha="center",
                    va="center",
                    color=text_col,
                    fontsize=7,
                )

    axes[0].set_ylabel("True mechanism")
    fig.suptitle(f"Mechanism Confusion Matrix Family ({classifier_name})", y=1.02, fontsize=12)
    fig.tight_layout()
    return fig


def plot_pairwise_error_vs_length(
    lengths: Sequence[int],
    diagnostic_curves: Mapping[str, Sequence[float]],
) -> Figure:
    """Plot pairwise error rate against trajectory length T, highlighting the information wall."""
    fig, ax = _figure(width=6.5, height=4.2)
    x = np.asarray(lengths, dtype=np.float64)

    style_map = {
        "fbm_vs_ctrw": (palette.CATEGORICAL[1], "-", "fBm vs CTRW (Separates)"),
        "sbm_vs_brownian": (palette.CATEGORICAL[2], "--", "SBM vs Brownian (Decaying Error)"),
        "ddm_vs_brownian": (palette.CATEGORICAL[6], ":", "DDM vs Brownian (Short-time NGP)"),
    }

    for key, (color, ls, label) in style_map.items():
        if key in diagnostic_curves:
            y = np.asarray(diagnostic_curves[key], dtype=np.float64)
            ax.plot(x, y, color=color, linestyle=ls, marker="o", label=label, linewidth=2)

    ax.set_xscale("log", base=2)
    ax.set_xlabel("Trajectory length $T$")
    ax.set_ylabel("Pairwise misclassification rate")
    ax.set_ylim(-0.02, 0.52)
    ax.axhline(0.5, color=palette.REFERENCE, linestyle=":", alpha=0.6, label="Chance level (0.50)")
    ax.set_title("Information Walls: Pairwise Error Rate vs Observation Window $T$")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()
    return fig


def plot_feature_importance_ranking(
    ablation_summary: Mapping[str, Any],
    top_k: int = 10,
) -> Figure:
    """Plot ranking of individual features by single-feature accuracy."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 4.2))
    fig.patch.set_facecolor(palette.SURFACE)

    lr_res = ablation_summary["logistic_regression_ablation"]
    single_feats = lr_res["single_feature_individual"]
    sorted_feats = sorted(single_feats.items(), key=lambda item: item[1], reverse=True)[:top_k]

    names = [k[0] for k in sorted_feats][::-1]
    accs = [k[1] for k in sorted_feats][::-1]

    y_pos = np.arange(len(names))
    ax1.barh(y_pos, accs, color=palette.CATEGORICAL[0], alpha=0.85, height=0.6)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(names, fontsize=8)
    ax1.set_xlabel("Single-Feature Accuracy")
    ax1.set_title("Top Discriminating Features (Overall)", fontsize=10)
    ax1.axvline(1.0 / 6.0, color=palette.REFERENCE, linestyle=":", label="Chance (1/6)")
    ax1.set_xlim(0, 1.0)
    ax1.legend(frameon=False, fontsize=7)

    # Group accuracies
    group_accs = lr_res["single_feature_groups"]
    sorted_grps = sorted(group_accs.items(), key=lambda item: item[1], reverse=True)
    g_names = [g[0] for g in sorted_grps][::-1]
    g_vals = [g[1] for g in sorted_grps][::-1]
    g_pos = np.arange(len(g_names))

    ax2.barh(g_pos, g_vals, color=palette.CATEGORICAL[1], alpha=0.85, height=0.6)
    ax2.set_yticks(g_pos)
    ax2.set_yticklabels(g_names, fontsize=8)
    ax2.set_xlabel("Group Accuracy")
    ax2.set_title("Feature Group Importance", fontsize=10)
    ax2.set_xlim(0, 1.0)

    fig.suptitle(f"Feature Ablation Analysis ($T = {ablation_summary['n_steps']}$)", fontsize=12)
    fig.tight_layout()
    return fig


def plot_information_floor_efficiency(
    info_results: Mapping[str, Any],
) -> Figure:
    """Plot estimator variance and asymptotic efficiency against Cramér-Rao lower bound."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 4.0))
    fig.patch.set_facecolor(palette.SURFACE)

    lengths = np.array(info_results["lengths"], dtype=np.float64)
    h_dict = info_results["results"]

    # Panel 1: Variances for H = 0.3
    h03 = h_dict["H_0.3"]
    ax1.plot(lengths, h03["crlb"], "k--", label=r"$\mathrm{CRLB}(H)$ Bound ($I(H)^{-1}$)", linewidth=2)
    ax1.plot(lengths, h03["var_whittle"], color=palette.CATEGORICAL[0], marker="s", label=r"Whittle MLE $\mathrm{Var}(\hat{H})$")
    ax1.plot(lengths, h03["var_tamsd"], color=palette.CATEGORICAL[1], marker="o", label=r"TA-MSD Fit $\mathrm{Var}(\hat{\alpha}_{\mathrm{TA}}/2)$")
    ax1.set_xscale("log", base=2)
    ax1.set_yscale("log")
    ax1.set_xlabel("Trajectory length $T$")
    ax1.set_ylabel(r"Variance $\mathrm{Var}(\hat{H})$")
    ax1.set_title(r"Estimator Variance vs Bound ($H = 0.3$)", fontsize=10)
    ax1.legend(frameon=False, fontsize=8)

    # Panel 2: Efficiency ratio CRLB / Var
    ax2.plot(lengths, h03["eff_whittle"], color=palette.CATEGORICAL[0], marker="s", label="Whittle MLE ($H=0.3$)", linewidth=2)
    ax2.plot(lengths, h03["eff_tamsd"], color=palette.CATEGORICAL[1], marker="o", label="TA-MSD ($H=0.3$)", linewidth=2)
    if "H_0.7" in h_dict:
        h07 = h_dict["H_0.7"]
        ax2.plot(lengths, h07["eff_whittle"], color=palette.CATEGORICAL[0], linestyle="--", marker="^", label="Whittle MLE ($H=0.7$)")
        ax2.plot(lengths, h07["eff_tamsd"], color=palette.CATEGORICAL[1], linestyle="--", marker="v", label="TA-MSD ($H=0.7$)")

    ax2.axhline(1.0, color=palette.REFERENCE, linestyle=":", alpha=0.7, label="Asymptotically Efficient (1.0)")
    ax2.set_xscale("log", base=2)
    ax2.set_xlabel("Trajectory length $T$")
    ax2.set_ylabel(r"Efficiency $\mathrm{CRLB}/\mathrm{Var}(\hat{H})$")
    ax2.set_ylim(-0.05, 1.15)
    ax2.set_title("Information Floor Efficiency", fontsize=10)
    ax2.legend(frameon=False, fontsize=8)

    fig.suptitle("Information Floor & Asymptotic Efficiency", fontsize=12)
    fig.tight_layout()
    return fig


def plot_conditional_exponent_comparison(
    cond_results: Mapping[str, Any],
) -> Figure:
    """Plot per-mechanism MAE across Naive, Predicted, and Oracle conditions."""
    fig, ax = _figure(width=7.5, height=4.2)

    mechs = sorted(cond_results["per_mechanism_mae_naive"].keys())
    labels = [palette.MECHANISM_LABEL.get(m, m) for m in mechs]
    x = np.arange(len(mechs))
    width = 0.26

    mae_naive = [cond_results["per_mechanism_mae_naive"][m] for m in mechs]
    mae_pred = [cond_results["per_mechanism_mae_predicted"][m] for m in mechs]
    mae_oracle = [cond_results["per_mechanism_mae_oracle"][m] for m in mechs]

    ax.bar(x - width, mae_naive, width, label="1. Naive (Phase 1 TA-MSD)", color=palette.CATEGORICAL[7], alpha=0.85)
    ax.bar(x, mae_pred, width, label="2. Predicted Mechanism + Correction", color=palette.CATEGORICAL[0], alpha=0.85)
    ax.bar(x + width, mae_oracle, width, label="3. Oracle (Known Mechanism)", color=palette.CATEGORICAL[2], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel(r"Exponent Mean Absolute Error $|\hat{\alpha} - \alpha_{\mathrm{true}}|$")
    ax.set_title(
        f"Closing the Loop: Conditional Exponent Recovery ($T = {cond_results['n_steps']}$)\n"
        f"Overall MAE: Naive {cond_results['mae_naive']:.3f} $\\to$ Predicted {cond_results['mae_predicted']:.3f} (Oracle {cond_results['mae_oracle']:.3f})"
    )
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return fig
