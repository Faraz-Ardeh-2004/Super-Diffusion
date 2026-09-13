"""Feature and group ablation studies (Phase 2, Step 4).

Following ``PHASE2_PROMPT.md``:
- Report classification performance with each feature/group removed (leave-one-out).
- Report classification performance with each feature/group alone (single-feature).
- Ranking of observables by discriminating power, overall and per mechanism pair.
- Interpret results physically.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from ..features import FEATURE_GROUPS, FEATURE_NAMES
from .classification import compute_pairwise_error_matrix, make_classifier
from .dataset import TrajectoryDataset

__all__ = [
    "AblationStudyResult",
    "PairwiseAblationResult",
    "run_ablation_study",
    "run_pairwise_feature_ranking",
]


@dataclass
class AblationStudyResult:
    """Summary of leave-one-out and single-feature ablation."""

    baseline_accuracy: float
    leave_one_out_individual: dict[str, float]  # feature_name -> acc
    leave_one_out_groups: dict[str, float]  # group_name -> acc
    single_feature_individual: dict[str, float]  # feature_name -> acc
    single_feature_groups: dict[str, float]  # group_name -> acc
    n_steps: int
    classifier_name: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline_accuracy": float(self.baseline_accuracy),
            "leave_one_out_individual": self.leave_one_out_individual,
            "leave_one_out_groups": self.leave_one_out_groups,
            "single_feature_individual": self.single_feature_individual,
            "single_feature_groups": self.single_feature_groups,
            "n_steps": int(self.n_steps),
            "classifier_name": self.classifier_name,
        }


@dataclass
class PairwiseAblationResult:
    """Single-feature discriminatory power for a specific mechanism pair."""

    pair: tuple[str, str]
    feature_accuracies: dict[str, float]  # feature_name -> accuracy on pair
    ranked_features: list[tuple[str, float]]  # sorted (name, accuracy)


def _evaluate_subset(
    train_features: NDArray[np.float64],
    train_labels: NDArray[np.intp],
    test_features: NDArray[np.float64],
    test_labels: NDArray[np.intp],
    feature_indices: Sequence[int],
    clf_name: str = "logistic_regression",
    seed: int = 0,
) -> float:
    """Train on feature subset and return test accuracy."""
    idx = list(feature_indices)
    if len(idx) == 0:
        return 0.0
    x_tr = train_features[:, idx]
    x_te = test_features[:, idx]

    model = make_classifier(clf_name, seed=seed)
    model.fit(x_tr, train_labels)
    return float(model.score(x_te, test_labels))


def run_ablation_study(
    train_ds: TrajectoryDataset,
    test_ds: TrajectoryDataset,
    clf_name: str = "logistic_regression",
    seed: int = 0,
) -> AblationStudyResult:
    """Run full leave-one-out and single-feature ablation on dataset."""
    all_names = train_ds.feature_names
    n_feats = len(all_names)
    all_indices = list(range(n_feats))
    name_to_idx = {name: i for i, name in enumerate(all_names)}

    # Baseline with all features
    baseline_acc = _evaluate_subset(
        train_ds.features,
        train_ds.labels,
        test_ds.features,
        test_ds.labels,
        all_indices,
        clf_name=clf_name,
        seed=seed,
    )

    # 1. Individual features: Leave-one-out and Single-feature
    loo_ind: dict[str, float] = {}
    single_ind: dict[str, float] = {}

    for name in all_names:
        idx = name_to_idx[name]
        # Single feature
        single_ind[name] = _evaluate_subset(
            train_ds.features,
            train_ds.labels,
            test_ds.features,
            test_ds.labels,
            [idx],
            clf_name=clf_name,
            seed=seed,
        )
        # Leave-one-out
        sub_indices = [i for i in all_indices if i != idx]
        loo_ind[name] = _evaluate_subset(
            train_ds.features,
            train_ds.labels,
            test_ds.features,
            test_ds.labels,
            sub_indices,
            clf_name=clf_name,
            seed=seed,
        )

    # 2. Group ablation
    loo_groups: dict[str, float] = {}
    single_groups: dict[str, float] = {}

    for grp_name, grp_features in FEATURE_GROUPS.items():
        grp_indices = [name_to_idx[f] for f in grp_features if f in name_to_idx]
        if not grp_indices:
            continue
        # Single group
        single_groups[grp_name] = _evaluate_subset(
            train_ds.features,
            train_ds.labels,
            test_ds.features,
            test_ds.labels,
            grp_indices,
            clf_name=clf_name,
            seed=seed,
        )
        # Leave group out
        grp_sub = [i for i in all_indices if i not in grp_indices]
        loo_groups[grp_name] = _evaluate_subset(
            train_ds.features,
            train_ds.labels,
            test_ds.features,
            test_ds.labels,
            grp_sub,
            clf_name=clf_name,
            seed=seed,
        )

    return AblationStudyResult(
        baseline_accuracy=baseline_acc,
        leave_one_out_individual=loo_ind,
        leave_one_out_groups=loo_groups,
        single_feature_individual=single_ind,
        single_feature_groups=single_groups,
        n_steps=train_ds.config.n_steps,
        classifier_name=clf_name,
    )


def run_pairwise_feature_ranking(
    train_ds: TrajectoryDataset,
    test_ds: TrajectoryDataset,
    mech_a: str,
    mech_b: str,
    clf_name: str = "logistic_regression",
    seed: int = 0,
) -> PairwiseAblationResult:
    """Rank single features by accuracy in separating mechanism pair (A, B)."""
    idx_a = train_ds.mechanism_names.index(mech_a)
    idx_b = train_ds.mechanism_names.index(mech_b)

    mask_tr = (train_ds.labels == idx_a) | (train_ds.labels == idx_b)
    mask_te = (test_ds.labels == idx_a) | (test_ds.labels == idx_b)

    x_tr = train_ds.features[mask_tr]
    y_tr = train_ds.labels[mask_tr]
    x_te = test_ds.features[mask_te]
    y_te = test_ds.labels[mask_te]

    feature_accs: dict[str, float] = {}
    for i, name in enumerate(train_ds.feature_names):
        acc = _evaluate_subset(
            x_tr,
            y_tr,
            x_te,
            y_te,
            [i],
            clf_name=clf_name,
            seed=seed,
        )
        feature_accs[name] = acc

    ranked = sorted(feature_accs.items(), key=lambda item: item[1], reverse=True)
    return PairwiseAblationResult(
        pair=(mech_a, mech_b),
        feature_accuracies=feature_accs,
        ranked_features=ranked,
    )
