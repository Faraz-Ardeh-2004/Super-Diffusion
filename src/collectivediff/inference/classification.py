"""Mechanism classification models, evaluation, and pairwise error analysis (Phase 2, Step 3).

Following ``PHASE2_PROMPT.md``:
- Interpretable models first: Multinomial Logistic Regression and Linear Discriminant Analysis.
- Upper bound: Gradient Boosting (HistGradientBoostingClassifier).
- Family of confusion matrices indexed by T.
- Pairwise error rate against T for every mechanism pair.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .dataset import DatasetConfig, TrajectoryDataset, load_or_generate_dataset

__all__ = [
    "CLASSIFIER_NAMES",
    "ClassificationResult",
    "PairwiseErrorResult",
    "compute_pairwise_error_matrix",
    "fit_and_evaluate_classifiers",
    "make_classifier",
    "sweep_classification_over_lengths",
]

CLASSIFIER_NAMES: list[str] = [
    "logistic_regression",
    "lda",
    "gradient_boosting",
]


def make_classifier(name: str, seed: int = 0) -> Any:
    """Construct a pipeline for the requested classifier."""
    if name == "logistic_regression":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=2000,
                        random_state=seed,
                        solver="lbfgs",
                    ),
                ),
            ]
        )
    elif name == "lda":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LinearDiscriminantAnalysis()),
            ]
        )
    elif name == "gradient_boosting":
        return HistGradientBoostingClassifier(random_state=seed)
    else:
        raise ValueError(f"Unknown classifier name {name!r}; known: {CLASSIFIER_NAMES}")


def compute_pairwise_error_matrix(
    conf_mat: NDArray[np.int64],
) -> NDArray[np.float64]:
    """Compute pairwise error rate between each pair of classes (A, B).

    .. math::

        \\mathrm{Error}(A, B) = \\frac{N_{A \\to B} + N_{B \\to A}}{N_A + N_B}

    Parameters
    ----------
    conf_mat : ndarray of int
        Confusion matrix where entry (i, j) is ground truth i predicted as j.

    Returns
    -------
    ndarray of float64
        Symmetric (K, K) matrix of pairwise error rates. Diagonal is 0.0.
    """
    k = conf_mat.shape[0]
    pairwise = np.zeros((k, k), dtype=np.float64)
    for i in range(k):
        for j in range(i + 1, k):
            n_i = conf_mat[i, :].sum()
            n_j = conf_mat[j, :].sum()
            err_ij = conf_mat[i, j] + conf_mat[j, i]
            denom = n_i + n_j
            rate = float(err_ij / denom) if denom > 0 else 0.0
            pairwise[i, j] = rate
            pairwise[j, i] = rate
    return pairwise


@dataclass
class ClassificationResult:
    """Results for one classifier at one trajectory length T."""

    classifier_name: str
    n_steps: int
    train_accuracy: float
    interp_accuracy: float
    extrap_accuracy: float
    confusion_matrix_interp: NDArray[np.int64]
    confusion_matrix_extrap: NDArray[np.int64]
    pairwise_error_interp: NDArray[np.float64]
    mechanism_names: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "classifier_name": self.classifier_name,
            "n_steps": int(self.n_steps),
            "train_accuracy": float(self.train_accuracy),
            "interp_accuracy": float(self.interp_accuracy),
            "extrap_accuracy": float(self.extrap_accuracy),
            "confusion_matrix_interp": self.confusion_matrix_interp.tolist(),
            "confusion_matrix_extrap": self.confusion_matrix_extrap.tolist(),
            "pairwise_error_interp": self.pairwise_error_interp.tolist(),
            "mechanism_names": self.mechanism_names,
        }


@dataclass
class PairwiseErrorResult:
    """Pairwise error rates tracked across trajectory lengths T."""

    pair: tuple[str, str]
    lengths: list[int]
    error_rates: list[float]


def fit_and_evaluate_classifiers(
    train_ds: TrajectoryDataset,
    test_interp_ds: TrajectoryDataset,
    test_extrap_ds: TrajectoryDataset,
    classifiers: Sequence[str] = ("logistic_regression", "lda", "gradient_boosting"),
    seed: int = 0,
) -> dict[str, ClassificationResult]:
    """Train and evaluate all classifiers on the given datasets.

    Parameters
    ----------
    train_ds : TrajectoryDataset
    test_interp_ds : TrajectoryDataset
    test_extrap_ds : TrajectoryDataset
    classifiers : sequence of str
    seed : int

    Returns
    -------
    dict of str to ClassificationResult
    """
    results: dict[str, ClassificationResult] = {}
    k = len(train_ds.mechanism_names)

    for clf_name in classifiers:
        model = make_classifier(clf_name, seed=seed)
        model.fit(train_ds.features, train_ds.labels)

        train_acc = float(model.score(train_ds.features, train_ds.labels))
        interp_acc = float(model.score(test_interp_ds.features, test_interp_ds.labels))
        extrap_acc = float(model.score(test_extrap_ds.features, test_extrap_ds.labels))

        interp_preds = model.predict(test_interp_ds.features)
        extrap_preds = model.predict(test_extrap_ds.features)

        cm_interp = confusion_matrix(test_interp_ds.labels, interp_preds, labels=np.arange(k))
        cm_extrap = confusion_matrix(test_extrap_ds.labels, extrap_preds, labels=np.arange(k))

        pairwise_err = compute_pairwise_error_matrix(cm_interp)

        results[clf_name] = ClassificationResult(
            classifier_name=clf_name,
            n_steps=train_ds.config.n_steps,
            train_accuracy=train_acc,
            interp_accuracy=interp_acc,
            extrap_accuracy=extrap_acc,
            confusion_matrix_interp=cm_interp,
            confusion_matrix_extrap=cm_extrap,
            pairwise_error_interp=pairwise_err,
            mechanism_names=train_ds.mechanism_names,
        )

    return results


def sweep_classification_over_lengths(
    lengths: Sequence[int] = (128, 512, 2048, 8192),
    n_samples_per_class: int = 150,
    classifiers: Sequence[str] = ("logistic_regression", "lda", "gradient_boosting"),
    train_seed: int = 1000,
    test_interp_seed: int = 2000,
    test_extrap_seed: int = 3000,
) -> dict[int, dict[str, ClassificationResult]]:
    """Run full classification study across trajectory lengths T.

    Parameters
    ----------
    lengths : sequence of int
        List of trajectory lengths (e.g. 128, 512, 2048, 8192, 32768).
    n_samples_per_class : int
        Trajectories per mechanism.
    classifiers : sequence of str
    train_seed, test_interp_seed, test_extrap_seed : int
        Disjoint seeds for reproducibility.

    Returns
    -------
    dict mapping n_steps to {clf_name: ClassificationResult}
    """
    all_results: dict[int, dict[str, ClassificationResult]] = {}

    for n_steps in lengths:
        train_cfg = DatasetConfig(
            n_samples_per_class=n_samples_per_class,
            n_steps=n_steps,
            split="train",
            seed=train_seed + n_steps,
        )
        interp_cfg = DatasetConfig(
            n_samples_per_class=n_samples_per_class,
            n_steps=n_steps,
            split="test_interp",
            seed=test_interp_seed + n_steps,
        )
        extrap_cfg = DatasetConfig(
            n_samples_per_class=n_samples_per_class,
            n_steps=n_steps,
            split="test_extrap",
            seed=test_extrap_seed + n_steps,
        )

        train_ds = load_or_generate_dataset(train_cfg)
        interp_ds = load_or_generate_dataset(interp_cfg)
        extrap_ds = load_or_generate_dataset(extrap_cfg)

        step_results = fit_and_evaluate_classifiers(
            train_ds=train_ds,
            test_interp_ds=interp_ds,
            test_extrap_ds=extrap_ds,
            classifiers=classifiers,
            seed=train_seed,
        )
        all_results[n_steps] = step_results

    return all_results
