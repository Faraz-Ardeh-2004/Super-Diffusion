"""Full Phase 2 executable studies and report generators.

Following ``PHASE2_PROMPT.md``:
- Every study writes a small JSON summary to ``results/phase2/``.
- Profiles feature extraction, caches datasets, reports timings.
- Delivers Steps 1-6 end-to-end with clear physical interpretations.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray

from .config import FBMConfig
from .features import (
    FEATURE_GROUPS,
    FEATURE_NAMES,
    cramer_rao_bound_fbm,
    extract_features,
    whittle_mle_hurst,
)
from .generators import generate
from .inference.ablation import run_ablation_study, run_pairwise_feature_ranking
from .inference.classification import (
    ClassificationResult,
    compute_pairwise_error_matrix,
    fit_and_evaluate_classifiers,
    make_classifier,
    sweep_classification_over_lengths,
)
from .inference.conditional import (
    ConditionalEvaluationResult,
    count_levy_walk_flights,
    estimate_exponent_conditional,
    evaluate_conditional_exponent_estimation,
)
from .inference.dataset import (
    MECHANISM_NAMES,
    DatasetConfig,
    TrajectoryDataset,
    draw_random_config,
    load_or_generate_dataset,
)
from .studies import RESULTS_ROOT, write_summary

__all__ = [
    "PHASE2_RESULTS_ROOT",
    "run_step2_dataset_study",
    "run_step3_classification_study",
    "run_step4_ablation_study",
    "run_step5_information_floor_study",
    "run_step6_conditional_exponent_study",
    "write_phase2_summary",
]

#: Phase 2 results directory
PHASE2_RESULTS_ROOT: Path = RESULTS_ROOT / "phase2"


def write_phase2_summary(name: str, payload: dict[str, Any]) -> Path:
    """Write summary JSON to results/phase2/<name>.json."""
    path = PHASE2_RESULTS_ROOT / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def run_step2_dataset_study(
    lengths: Sequence[int] = (128, 512, 2048, 8192),
    n_samples_per_class: int = 100,
    seed: int = 42,
) -> dict[str, Any]:
    """Profile feature extraction and dataset generation across trajectory lengths."""
    timings: dict[str, float] = {}
    shapes: dict[str, list[int]] = {}

    for n_steps in lengths:
        t0 = time.perf_counter()
        cfg = DatasetConfig(
            n_samples_per_class=n_samples_per_class,
            n_steps=n_steps,
            split="train",
            seed=seed + n_steps,
        )
        ds = load_or_generate_dataset(cfg)
        elapsed = time.perf_counter() - t0
        timings[str(n_steps)] = elapsed
        shapes[str(n_steps)] = list(ds.features.shape)

    summary = {
        "trajectory_lengths": list(lengths),
        "n_samples_per_class": n_samples_per_class,
        "n_classes": len(MECHANISM_NAMES),
        "mechanisms": MECHANISM_NAMES,
        "feature_names": FEATURE_NAMES,
        "timings_seconds": timings,
        "dataset_shapes": shapes,
        "class_prior": [1.0 / len(MECHANISM_NAMES)] * len(MECHANISM_NAMES),
    }
    write_phase2_summary("step2_dataset_summary", summary)
    return summary


def run_step3_classification_study(
    lengths: Sequence[int] = (128, 512, 2048, 8192),
    n_samples_per_class: int = 150,
) -> dict[str, Any]:
    """Run classifier sweep and pairwise error analysis across trajectory lengths."""
    all_results = sweep_classification_over_lengths(
        lengths=lengths,
        n_samples_per_class=n_samples_per_class,
    )

    summary_by_t: dict[str, Any] = {}
    pairwise_errors_by_t: dict[str, Any] = {}

    for n_steps, step_dict in all_results.items():
        summary_by_t[str(n_steps)] = {
            clf_name: res.to_dict() for clf_name, res in step_dict.items()
        }
        # Record gradient boosting pairwise errors
        gb_res = step_dict["gradient_boosting"]
        pairwise_errors_by_t[str(n_steps)] = gb_res.pairwise_error_interp.tolist()

    # Track specific diagnostic pairs across T:
    # 1. fbm vs ctrw
    # 2. sbm vs brownian
    # 3. ddm vs brownian
    idx_fbm = MECHANISM_NAMES.index("fbm")
    idx_ctrw = MECHANISM_NAMES.index("ctrw")
    idx_sbm = MECHANISM_NAMES.index("sbm")
    idx_brownian = MECHANISM_NAMES.index("brownian")
    idx_ddm = MECHANISM_NAMES.index("ddm")

    diagnostic_curves: dict[str, list[float]] = {
        "fbm_vs_ctrw": [],
        "sbm_vs_brownian": [],
        "ddm_vs_brownian": [],
    }

    for n_steps in lengths:
        p_mat = all_results[n_steps]["gradient_boosting"].pairwise_error_interp
        diagnostic_curves["fbm_vs_ctrw"].append(float(p_mat[idx_fbm, idx_ctrw]))
        diagnostic_curves["sbm_vs_brownian"].append(float(p_mat[idx_sbm, idx_brownian]))
        diagnostic_curves["ddm_vs_brownian"].append(float(p_mat[idx_ddm, idx_brownian]))

    payload = {
        "lengths": list(lengths),
        "results_by_length": summary_by_t,
        "diagnostic_pairwise_curves": diagnostic_curves,
        "mechanism_names": MECHANISM_NAMES,
    }
    write_phase2_summary("step3_classification_summary", payload)
    return payload


def run_step4_ablation_study(
    n_steps: int = 2048,
    n_samples_per_class: int = 150,
) -> dict[str, Any]:
    """Run full feature ablation and pairwise ranking study."""
    train_cfg = DatasetConfig(
        n_samples_per_class=n_samples_per_class,
        n_steps=n_steps,
        split="train",
        seed=1000 + n_steps,
    )
    test_cfg = DatasetConfig(
        n_samples_per_class=n_samples_per_class,
        n_steps=n_steps,
        split="test_interp",
        seed=2000 + n_steps,
    )

    train_ds = load_or_generate_dataset(train_cfg)
    test_ds = load_or_generate_dataset(test_cfg)

    # 1. Global feature ablation with logistic regression and gradient boosting
    lr_ablation = run_ablation_study(train_ds, test_ds, clf_name="logistic_regression")
    gb_ablation = run_ablation_study(train_ds, test_ds, clf_name="gradient_boosting")

    # 2. Pairwise feature rankings for diagnostic pairs
    pairs = [
        ("fbm", "ctrw"),
        ("sbm", "brownian"),
        ("ddm", "brownian"),
    ]
    pairwise_rankings: dict[str, list[dict[str, Any]]] = {}
    for a, b in pairs:
        res = run_pairwise_feature_ranking(train_ds, test_ds, a, b)
        pairwise_rankings[f"{a}_vs_{b}"] = [
            {"feature": name, "accuracy": float(acc)} for name, acc in res.ranked_features
        ]

    payload = {
        "n_steps": n_steps,
        "logistic_regression_ablation": lr_ablation.to_dict(),
        "gradient_boosting_ablation": gb_ablation.to_dict(),
        "pairwise_rankings": pairwise_rankings,
    }
    write_phase2_summary("step4_ablation_summary", payload)
    return payload


def run_step5_information_floor_study(
    lengths: Sequence[int] = (128, 512, 2048, 8192),
    hurst_values: Sequence[float] = (0.3, 0.7),
    n_trajectories: int = 100,
    seed: int = 500,
) -> dict[str, Any]:
    """Compare empirical estimator variance against the Cramér-Rao lower bound."""
    results_by_h: dict[str, Any] = {}

    for h in hurst_values:
        crlb_list: list[float] = []
        var_whittle_list: list[float] = []
        var_tamsd_list: list[float] = []
        eff_whittle_list: list[float] = []
        eff_tamsd_list: list[float] = []

        for n_steps in lengths:
            crlb = cramer_rao_bound_fbm(hurst=h, n_steps=n_steps)
            crlb_list.append(crlb)

            cfg = FBMConfig(
                n_particles=n_trajectories,
                n_steps=n_steps,
                hurst=h,
                diffusivity=1.0,
                seed=seed,
            )
            rng = np.random.default_rng(seed)
            trajs = generate(cfg, rng)

            h_whittle: list[float] = []
            h_tamsd: list[float] = []

            for i in range(n_trajectories):
                traj = trajs[i : i + 1]
                # Whittle MLE
                h_w = whittle_mle_hurst(traj)
                h_whittle.append(h_w)

                # TA-MSD estimator: alpha / 2
                f_dict = extract_features(traj)
                # Feature 0 is tamds_exponent
                alpha_ta = f_dict[0]
                h_tamsd.append(alpha_ta / 2.0)

            var_w = float(np.var(h_whittle))
            var_ta = float(np.var(h_tamsd))

            var_whittle_list.append(var_w)
            var_tamsd_list.append(var_ta)

            eff_w = float(crlb / max(var_w, 1e-12))
            eff_ta = float(crlb / max(var_ta, 1e-12))

            eff_whittle_list.append(eff_w)
            eff_tamsd_list.append(eff_ta)

        results_by_h[f"H_{h}"] = {
            "crlb": crlb_list,
            "var_whittle": var_whittle_list,
            "var_tamsd": var_tamsd_list,
            "eff_whittle": eff_whittle_list,
            "eff_tamsd": eff_tamsd_list,
        }

    payload = {
        "lengths": list(lengths),
        "hurst_values": list(hurst_values),
        "results": results_by_h,
    }
    write_phase2_summary("step5_information_floor", payload)
    return payload


def run_step6_conditional_exponent_study(
    n_steps: int = 2048,
    n_samples_per_class: int = 150,
) -> dict[str, Any]:
    """Run conditional exponent estimation comparing Naive, Predicted, and Oracle."""
    train_cfg = DatasetConfig(
        n_samples_per_class=n_samples_per_class,
        n_steps=n_steps,
        split="train",
        seed=1000 + n_steps,
    )
    test_cfg = DatasetConfig(
        n_samples_per_class=n_samples_per_class,
        n_steps=n_steps,
        split="test_interp",
        seed=2000 + n_steps,
    )

    train_ds = load_or_generate_dataset(train_cfg)
    test_ds = load_or_generate_dataset(test_cfg)

    # Train classifier (Gradient Boosting)
    clf = make_classifier("gradient_boosting", seed=42)
    clf.fit(train_ds.features, train_ds.labels)
    predicted_label_indices = clf.predict(test_ds.features)
    predicted_mechs = [test_ds.mechanism_names[idx] for idx in predicted_label_indices]

    # Reconstruct test trajectories from test parameters
    trajectories: list[NDArray[np.float64]] = []
    true_mechs: list[str] = []
    true_alphas: list[float] = []

    param_rng = np.random.default_rng(test_cfg.seed)
    for class_idx, mech in enumerate(test_ds.mechanism_names):
        for i in range(test_cfg.n_samples_per_class):
            traj_seed = int(param_rng.integers(1, 2**31 - 1))
            cfg, p_record = draw_random_config(
                mechanism=mech,
                n_steps=n_steps,
                rng=param_rng,
                extrapolate=False,
                seed=traj_seed,
            )
            traj = generate(cfg, np.random.default_rng(traj_seed))
            trajectories.append(traj)
            true_mechs.append(mech)
            true_alphas.append(float(p_record["alpha_true"]))

    res = evaluate_conditional_exponent_estimation(
        trajectories=trajectories,
        true_mechanisms=true_mechs,
        true_alphas=true_alphas,
        predicted_mechanisms=predicted_mechs,
        n_steps=n_steps,
    )

    payload = res.to_dict()
    # Diagnostic for the levy_walk correction: the flight-duration Hill
    # estimate it blends in is only as trustworthy as the flight count behind
    # it (PHASE2_GLS_PATCH.md), which is not otherwise visible from the MAE
    # alone.
    levy_walk_flight_counts = [
        count_levy_walk_flights(traj)
        for traj, mech in zip(trajectories, true_mechs)
        if mech == "levy_walk"
    ]
    if levy_walk_flight_counts:
        payload["levy_walk_flight_count"] = {
            "min": int(np.min(levy_walk_flight_counts)),
            "median": float(np.median(levy_walk_flight_counts)),
            "max": int(np.max(levy_walk_flight_counts)),
        }
    write_phase2_summary("step6_conditional_exponent", payload)
    return payload
