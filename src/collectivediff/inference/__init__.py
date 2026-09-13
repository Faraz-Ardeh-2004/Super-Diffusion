"""Single-agent mechanism identification and regime inference module (Phase 2).

Provides dataset generators, interpretable and non-linear classifiers,
feature ablation studies, and conditional exponent recovery.
"""

from __future__ import annotations

from .ablation import (
    AblationStudyResult,
    PairwiseAblationResult,
    run_ablation_study,
    run_pairwise_feature_ranking,
)
from .classification import (
    CLASSIFIER_NAMES,
    ClassificationResult,
    PairwiseErrorResult,
    compute_pairwise_error_matrix,
    fit_and_evaluate_classifiers,
    make_classifier,
    sweep_classification_over_lengths,
)
from .conditional import (
    ConditionalEvaluationResult,
    count_levy_walk_flights,
    estimate_exponent_conditional,
    estimate_exponent_ctrw,
    estimate_exponent_fbm,
    estimate_exponent_levy_walk,
    estimate_exponent_sbm,
    evaluate_conditional_exponent_estimation,
)
from .dataset import (
    MECHANISM_NAMES,
    DatasetConfig,
    TrajectoryDataset,
    draw_random_config,
    generate_dataset,
    load_or_generate_dataset,
)

__all__ = [
    # dataset
    "MECHANISM_NAMES",
    "DatasetConfig",
    "TrajectoryDataset",
    "draw_random_config",
    "generate_dataset",
    "load_or_generate_dataset",
    # classification
    "CLASSIFIER_NAMES",
    "ClassificationResult",
    "PairwiseErrorResult",
    "compute_pairwise_error_matrix",
    "make_classifier",
    "fit_and_evaluate_classifiers",
    "sweep_classification_over_lengths",
    # ablation
    "AblationStudyResult",
    "PairwiseAblationResult",
    "run_ablation_study",
    "run_pairwise_feature_ranking",
    # conditional
    "ConditionalEvaluationResult",
    "estimate_exponent_sbm",
    "estimate_exponent_ctrw",
    "estimate_exponent_fbm",
    "estimate_exponent_levy_walk",
    "count_levy_walk_flights",
    "estimate_exponent_conditional",
    "evaluate_conditional_exponent_estimation",
]
