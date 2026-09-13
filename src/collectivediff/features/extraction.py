"""Feature extraction for isolated-observer single trajectories (Phase 2, Step 1).

Following ``PHASE2_PROMPT.md``:
- Observer is isolated: takes a single trajectory of shape ``(n_steps, n_dim)``
  and computes named, scale-invariant features.
- Every feature is invariant under spatial rescaling ``x -> c * x`` for ``c > 0``.
- Reuses Phase 1 estimators where applicable.
- Handles edge cases (e.g. zero jumps in CTRW) gracefully.
"""

from __future__ import annotations

from typing import Any, Final, Sequence

import numpy as np
from numpy.typing import NDArray

from ..config import FitConfig
from ..estimators.correlations import increment_acf
from ..estimators.msd import check_layout, fit_powerlaw, log_spaced_lags, ta_msd

__all__ = [
    "FEATURE_NAMES",
    "FEATURE_GROUPS",
    "extract_features",
    "extract_feature_dict",
    "extract_feature_matrix",
]

#: Standard list of all extracted feature names
FEATURE_NAMES: Final[list[str]] = [
    "tamds_exponent",
    "tamds_ratio",
    "acf_lag1",
    "acf_lag2",
    "acf_lag3",
    "acf_lag5",
    "acf_lag10",
    "ngp_lag1",
    "ngp_lag5",
    "ngp_lag10",
    "immobile_fraction",
    "mean_gap_length",
    "max_gap_fraction",
    "p_var_ratio_p1",
    "p_var_ratio_p2",
    "p_var_ratio_p3",
    "maximal_excursion_ratio",
    "range_to_std_ratio",
    "increment_var_ratio",
]

#: Logical grouping of features for ablation studies
FEATURE_GROUPS: Final[dict[str, list[str]]] = {
    "tamds": ["tamds_exponent", "tamds_ratio"],
    "acf": ["acf_lag1", "acf_lag2", "acf_lag3", "acf_lag5", "acf_lag10"],
    "ngp": ["ngp_lag1", "ngp_lag5", "ngp_lag10"],
    "immobility": ["immobile_fraction", "mean_gap_length", "max_gap_fraction"],
    "p_variation": ["p_var_ratio_p1", "p_var_ratio_p2", "p_var_ratio_p3"],
    "excursion": ["maximal_excursion_ratio", "range_to_std_ratio"],
    "nonstationarity": ["increment_var_ratio"],
}


def _ensure_single_trajectory_3d(trajectory: NDArray[Any]) -> NDArray[np.float64]:
    """Ensure shape is (1, n_steps, 1) float64 for 1D analysis."""
    traj = np.asarray(trajectory, dtype=np.float64)
    if traj.ndim == 1:
        traj = traj[np.newaxis, :, np.newaxis]
    elif traj.ndim == 2:
        traj = traj[np.newaxis, :, :]
    elif traj.ndim == 3:
        if traj.shape[0] != 1:
            raise ValueError(
                f"Isolated observer accepts exactly 1 trajectory, got shape {traj.shape}"
            )
    else:
        raise ValueError(f"Expected 1D, 2D or 3D array, got ndim={traj.ndim}")
    check_layout(traj)
    return traj


def extract_feature_dict(
    trajectory: NDArray[Any],
    component: int = 0,
) -> dict[str, float]:
    """Extract named scale-invariant features from a single trajectory.

    Parameters
    ----------
    trajectory : ndarray
        Shape ``(n_steps,)``, ``(n_steps, n_dim)``, or ``(1, n_steps, n_dim)``.
    component : int
        Spatial component to extract features from (1D convention).

    Returns
    -------
    dict of str to float
        Dictionary mapping feature name to scalar value.
    """
    traj_3d = _ensure_single_trajectory_3d(trajectory)
    n_steps = traj_3d.shape[1]
    series = traj_3d[0, :, component]  # (n_steps,)
    increments = np.diff(series)
    inc_std = float(np.std(increments))

    features: dict[str, float] = {}

    # 1. TA-MSD exponent and ratio
    fit_cfg = FitConfig(lag_min=1, lag_max_fraction=0.1, n_lags=min(16, max(2, n_steps // 10)))
    lags = log_spaced_lags(n_steps, fit_cfg)
    try:
        t_msd = ta_msd(traj_3d, lags)[:, :]  # (1, len(lags))
        t_msd_vals = t_msd[0]
        if len(lags) >= 2 and np.all(t_msd_vals > 1e-15):
            fit_res = fit_powerlaw(
                lags.astype(np.float64),
                t_msd,
                float(lags[0]),
                float(lags[-1]),
                fit=FitConfig(n_bootstrap=0),
            )
            features["tamds_exponent"] = float(fit_res.exponent)
            features["tamds_ratio"] = float(t_msd_vals[-1] / t_msd_vals[0])
        else:
            features["tamds_exponent"] = 0.0
            features["tamds_ratio"] = 1.0
    except Exception:
        features["tamds_exponent"] = 0.0
        features["tamds_ratio"] = 1.0

    # 2. Increment ACF at lags 1, 2, 3, 5, 10
    max_acf_lag = min(10, n_steps - 2)
    if inc_std > 1e-15:
        try:
            inc_acf = increment_acf(traj_3d, max_lag=max_acf_lag, component=component, normalise=True)
            for lag in [1, 2, 3, 5, 10]:
                if lag < len(inc_acf):
                    features[f"acf_lag{lag}"] = float(inc_acf[lag])
                else:
                    features[f"acf_lag{lag}"] = 0.0
        except Exception:
            for lag in [1, 2, 3, 5, 10]:
                features[f"acf_lag{lag}"] = 0.0
    else:
        for lag in [1, 2, 3, 5, 10]:
            features[f"acf_lag{lag}"] = 0.0

    # 3. Non-Gaussian parameter on increments at lags 1, 5, 10
    for lag in [1, 5, 10]:
        if n_steps > lag:
            diffs = series[lag:] - series[:-lag]
            m2 = float(np.mean(diffs**2))
            m4 = float(np.mean(diffs**4))
            if m2 > 1e-15:
                a2 = m4 / (3.0 * (m2**2)) - 1.0
            else:
                a2 = 0.0
            features[f"ngp_lag{lag}"] = float(a2)
        else:
            features[f"ngp_lag{lag}"] = 0.0

    # 4. Immobility and gap distribution
    immobile = (increments == 0.0) | (np.abs(increments) <= 1e-6 * inc_std) if inc_std > 0 else np.ones_like(increments, dtype=bool)
    features["immobile_fraction"] = float(np.mean(immobile))

    # Gap lengths (runs of True in immobile mask)
    gap_lengths: list[int] = []
    current_gap = 0
    for is_imm in immobile:
        if is_imm:
            current_gap += 1
        else:
            if current_gap > 0:
                gap_lengths.append(current_gap)
                current_gap = 0
    if current_gap > 0:
        gap_lengths.append(current_gap)

    if len(gap_lengths) > 0:
        features["mean_gap_length"] = float(np.mean(gap_lengths) / n_steps)
        features["max_gap_fraction"] = float(np.max(gap_lengths) / n_steps)
    else:
        features["mean_gap_length"] = 0.0
        features["max_gap_fraction"] = 0.0

    # 5. p-variation ratios (lag-2 vs lag-1)
    step1_abs = np.abs(increments)
    step2_abs = np.abs(series[2::2] - series[:-2:2])
    for p in [1, 2, 3]:
        sum1 = float(np.sum(step1_abs**p))
        sum2 = float(np.sum(step2_abs**p))
        if sum1 > 1e-15:
            features[f"p_var_ratio_p{p}"] = float(sum2 / sum1)
        else:
            features[f"p_var_ratio_p{p}"] = 1.0

    # 6. Excursions and range
    pos_rel = series - series[0]
    total_std = float(np.std(series))
    inc_norm = float(np.sqrt(np.sum(increments**2)))
    if inc_norm > 1e-15:
        features["maximal_excursion_ratio"] = float(np.max(np.abs(pos_rel)) / inc_norm)
    else:
        features["maximal_excursion_ratio"] = 0.0

    if total_std > 1e-15:
        features["range_to_std_ratio"] = float((np.max(series) - np.min(series)) / total_std)
    else:
        features["range_to_std_ratio"] = 0.0

    # 7. Non-stationarity: variance ratio of first half vs second half
    half_idx = len(increments) // 2
    if half_idx > 0:
        var_first = float(np.var(increments[:half_idx]))
        var_second = float(np.var(increments[half_idx:]))
        if var_second > 1e-15 and var_first > 1e-15:
            features["increment_var_ratio"] = float(var_first / var_second)
        elif var_second <= 1e-15 and var_first <= 1e-15:
            features["increment_var_ratio"] = 1.0
        elif var_second <= 1e-15:
            features["increment_var_ratio"] = 100.0
        else:
            features["increment_var_ratio"] = 0.01
    else:
        features["increment_var_ratio"] = 1.0

    return features


def extract_features(
    trajectory: NDArray[Any],
    feature_names: Sequence[str] | None = None,
    component: int = 0,
) -> NDArray[np.float64]:
    """Extract a feature vector from a single trajectory.

    Parameters
    ----------
    trajectory : ndarray
        Shape ``(n_steps,)``, ``(n_steps, n_dim)``, or ``(1, n_steps, n_dim)``.
    feature_names : sequence of str, optional
        Subset of features to extract. Defaults to :data:`FEATURE_NAMES`.
    component : int
        Spatial component.

    Returns
    -------
    ndarray of float64
        Shape ``(n_features,)``.
    """
    f_dict = extract_feature_dict(trajectory, component=component)
    names = FEATURE_NAMES if feature_names is None else list(feature_names)
    return np.array([f_dict[name] for name in names], dtype=np.float64)


def extract_feature_matrix(
    trajectories: NDArray[Any],
    feature_names: Sequence[str] | None = None,
    component: int = 0,
) -> NDArray[np.float64]:
    """Extract features for an ensemble of trajectories independently.

    Guarantees no ensemble information leakage: each particle's feature vector
    is computed strictly in isolation.

    Parameters
    ----------
    trajectories : ndarray
        Shape ``(n_particles, n_steps, n_dim)``.
    feature_names : sequence of str, optional
        Features to extract.
    component : int
        Spatial component.

    Returns
    -------
    ndarray of float64
        Shape ``(n_particles, n_features)``.
    """
    check_layout(trajectories)
    n_particles = trajectories.shape[0]
    names = FEATURE_NAMES if feature_names is None else list(feature_names)
    n_features = len(names)

    matrix = np.empty((n_particles, n_features), dtype=np.float64)
    for i in range(n_particles):
        matrix[i] = extract_features(
            trajectories[i : i + 1],
            feature_names=names,
            component=component,
        )
    return matrix
