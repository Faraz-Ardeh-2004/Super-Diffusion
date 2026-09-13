"""Conditional exponent estimation given predicted mechanism (Phase 2, Step 6).

Following ``PHASE2_PROMPT.md``:
- Apply the appropriate correction to each predicted mechanism.
- Evaluates 3 conditions:
  1. Mechanism unknown (Phase 1 naive TA-MSD estimator).
  2. Mechanism predicted by classifier.
  3. Mechanism known exactly (oracle).
- Measures the gain from Phase 2 and the cost of misclassification.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.special import gamma

from ..features.extraction import _ensure_single_trajectory_3d, extract_feature_dict
from ..features.spectral import whittle_mle_hurst

__all__ = [
    "ConditionalEvaluationResult",
    "count_levy_walk_flights",
    "estimate_exponent_conditional",
    "estimate_exponent_ctrw",
    "estimate_exponent_fbm",
    "estimate_exponent_levy_walk",
    "estimate_exponent_sbm",
    "evaluate_conditional_exponent_estimation",
]


def estimate_exponent_sbm(
    trajectory: NDArray[Any],
    n_blocks: int = 8,
) -> float:
    """Estimate SBM anomalous exponent alpha from increment non-stationarity.

    For SBM, local increment variance scales as :math:`\\langle \\delta x^2(t) \\rangle \\sim t^{\\alpha-1}`.
    Fitting :math:`\\log \\sigma^2(t)` against :math:`\\log t` gives slope :math:`s \\approx \\alpha - 1`.
    """
    traj_3d = _ensure_single_trajectory_3d(trajectory)
    series = traj_3d[0, :, 0]
    increments = np.diff(series)
    n = len(increments)
    block_size = max(4, n // n_blocks)
    actual_blocks = n // block_size

    if actual_blocks < 3:
        return 1.0

    t_mids: list[float] = []
    variances: list[float] = []
    for b in range(actual_blocks):
        chunk = increments[b * block_size : (b + 1) * block_size]
        var = float(np.var(chunk))
        if var > 1e-15:
            t_mids.append((b + 0.5) * block_size)
            variances.append(var)

    if len(t_mids) < 3:
        return 1.0

    log_t = np.log(t_mids)
    log_var = np.log(variances)
    slope, _ = np.polyfit(log_t, log_var, deg=1)
    alpha_est = 1.0 + float(slope)
    return float(np.clip(alpha_est, 0.05, 1.95))


def estimate_exponent_ctrw(
    trajectory: NDArray[Any],
    tau0: float = 1.0,
) -> float:
    """Estimate CTRW waiting-time exponent 'a' from pause durations and event count.

    Uses maximum likelihood on observed immobile stretch durations with finite-time
    corrections from ``PROJECT.md`` section 4.
    """
    traj_3d = _ensure_single_trajectory_3d(trajectory)
    series = traj_3d[0, :, 0]
    increments = np.diff(series)
    inc_std = float(np.std(increments))
    immobile = (increments == 0.0) | (np.abs(increments) <= 1e-6 * inc_std) if inc_std > 0 else np.ones_like(increments, dtype=bool)

    # Extract pause lengths
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

    n_steps = len(series)
    n_events = max(1, len(gap_lengths))

    if len(gap_lengths) >= 3:
        # Pareto tail MLE on durations
        durations = np.array(gap_lengths, dtype=np.float64)
        log_ratios = np.log(np.maximum(durations, 1.0))
        denom = float(np.sum(log_ratios))
        if denom > 1e-6:
            a_mle = float(len(durations) / denom)
        else:
            a_mle = 0.5
    else:
        a_mle = 0.5

    # Renewal count scaling: <n(t)> ~ t^a / (Gamma(1-a) Gamma(1+a))
    # Approximate inversion: a ~ log(n_events) / log(n_steps)
    a_count = float(np.log(max(2, n_events)) / np.log(n_steps))

    # Blend MLE and event count for finite-time stability
    a_est = 0.6 * a_mle + 0.4 * a_count
    return float(np.clip(a_est, 0.05, 0.95))


def estimate_exponent_fbm(
    trajectory: NDArray[Any],
) -> float:
    """Estimate fBm exponent alpha = 2H via Whittle MLE."""
    h_hat = whittle_mle_hurst(trajectory)
    return float(np.clip(2.0 * h_hat, 0.05, 1.95))


def _naive_tamds_exponent(trajectory: NDArray[Any]) -> float:
    """The uncorrected Phase 1 TA-MSD fit, used as the flight-MLE fallback."""
    traj_3d = _ensure_single_trajectory_3d(trajectory)
    f_dict = extract_feature_dict(traj_3d)
    tamds_exp = f_dict.get("tamds_exponent", 1.5)
    return float(np.clip(tamds_exp, 1.05, 1.95))


def _subgrid_flight_durations(series: NDArray[np.float64]) -> NDArray[np.float64]:
    """Reconstruct Levy walk flight durations from a gridded 1D trajectory.

    A flight travels at constant speed in a fixed direction (a fair coin on
    ``{-1, +1}`` in 1D -- ``PROJECT.md`` section 2, ``LevyWalkConfig``), so on
    the ``dt = 1`` grid every step interior to one flight has the same
    displacement magnitude, the flight speed, estimated here as
    ``max(|diff|)`` since no mixture of two directions can exceed it. A step
    that straddles a turning point instead shows the time-weighted average of
    the two flights either side of it, and is solved for the fractional
    turning time exactly, since both flight speeds and both directions are
    already known from the pure steps neighbouring it.

    This sub-step resolution is what makes the reconstruction usable at all:
    `tau0` is order 1, the same order as the grid spacing, so a large fraction
    of flights are shorter than one step. Rounding turning points to the
    nearest grid point instead (the naive detector) collapses exactly that
    short-duration part of the tail -- which is also the part a Hill estimator
    needs least, since it only uses the upper tail, but which corrupts the
    duration *count* and hence which steps get identified as boundaries at
    all. A straddling step whose neighbouring pure steps agree in direction
    (an even number of reversals cancelled out within one step) is not
    identifiable from a single displacement value and is skipped, at the cost
    of undercounting the very shortest flights.

    The first and last recovered flights are censored -- their true start or
    end lies outside the trajectory -- and are dropped.

    Parameters
    ----------
    series : ndarray of float64
        Single 1D position trace, shape ``(n_steps,)``.

    Returns
    -------
    ndarray of float64
        Interior flight durations in grid-step units. Empty if fewer than
        three turning points were found.
    """
    d = np.diff(series)
    n = d.size
    speed_hat = float(np.max(np.abs(d))) if n else 0.0
    if speed_hat <= 0.0:
        return np.empty(0, dtype=np.float64)

    tol = 0.02 * speed_hat
    is_pure = np.abs(np.abs(d) - speed_hat) < tol
    sign = np.where(d >= 0.0, 1.0, -1.0)

    # sign of the next pure step at or after each index, for resolving a
    # straddling step's outgoing direction in one forward pass
    next_pure_sign = np.full(n, np.nan)
    running = np.nan
    for i in range(n - 1, -1, -1):
        if is_pure[i]:
            running = sign[i]
        next_pure_sign[i] = running

    boundaries: list[float] = [0.0]
    current_dir = np.nan
    for i in range(n):
        if is_pure[i]:
            if np.isnan(current_dir):
                current_dir = sign[i]
            elif sign[i] != current_dir:
                boundaries.append(float(i))
                current_dir = sign[i]
        else:
            v_out = next_pure_sign[i]
            if np.isnan(current_dir) or np.isnan(v_out) or v_out == current_dir:
                continue
            v_in_val, v_out_val = current_dir * speed_hat, v_out * speed_hat
            tau = np.clip((d[i] - v_out_val) / (v_in_val - v_out_val), 0.0, 1.0)
            boundaries.append(float(i) + float(tau))
            current_dir = v_out
    boundaries.append(float(n))

    durations = np.diff(np.asarray(boundaries, dtype=np.float64))
    return durations[1:-1] if durations.size > 2 else np.empty(0, dtype=np.float64)


def _hill_tail_exponent(durations: NDArray[np.float64], top_frac: float = 0.35) -> float | None:
    """Hill estimator of the Pareto tail index ``g`` from a duration sample.

    Uses only the largest ``top_frac`` of the sample. The Hill estimator is a
    tail statistic by construction, and for reconstructed Levy walk flight
    durations the bulk near ``tau0`` is exactly the part the grid's sub-step
    quantisation corrupts most, so restricting to the tail also discards most
    of that corruption rather than fighting it.

    Returns ``None`` if the sample is too small or degenerate to fit,
    signalling the caller to fall back to the naive estimate.
    """
    if durations.size < 5:
        return None
    ordered = np.sort(durations)[::-1]
    k = int(np.clip(round(top_frac * ordered.size), 5, ordered.size - 1))
    threshold = ordered[k]
    if threshold <= 0.0:
        return None
    mean_log_ratio = float(np.mean(np.log(ordered[:k] / threshold)))
    if mean_log_ratio <= 1e-9:
        return None
    return 1.0 / mean_log_ratio


def count_levy_walk_flights(trajectory: NDArray[Any]) -> int:
    """Number of interior flights :func:`estimate_exponent_levy_walk` recovers.

    Diagnostic only. The Hill estimate it blends in is only as trustworthy as
    this count, which falls as `g` approaches 1 and flights run long relative
    to the trajectory length -- report it alongside the MAE rather than
    quoting one aggregate number that hides a possibly flight-starved tail of
    the parameter range.
    """
    traj_3d = _ensure_single_trajectory_3d(trajectory)
    if traj_3d.shape[2] != 1:
        return 0
    return int(_subgrid_flight_durations(traj_3d[0, :, 0]).size)


def estimate_exponent_levy_walk(
    trajectory: NDArray[Any],
    naive_weight: float = 0.6,
    top_frac: float = 0.35,
) -> float:
    """Blend the naive TA-MSD fit with a flight-duration MLE -- the Levy walk correction.

    Two corrections were tried and rejected before this one on empirical
    grounds, both documented in ``PROJECT.md``:

    - A GLS-weighted TA-MSD fit (``Sigma`` from bootstrapping contiguous time
      blocks, per the empirical-weights convention). It is *worse* than the
      naive OLS fit, including with the true ensemble covariance in place of
      the bootstrap estimate -- so this is not an estimation problem. The
      short lags are precise but biased toward the within-flight ballistic
      regime; the long lags are asymptotically correct but, for one
      trajectory, individually very noisy. Inverse-variance weighting
      concentrates on the precise-but-biased end, which makes the fit worse,
      not better.
    - A single-trajectory moment-spectrum kink fit
      (:func:`~collectivediff.estimators.moments.fit_bilinear_spectrum`).
      Needs an ensemble to resolve high moments that the isolated observer
      does not have; a pseudo-ensemble built by splitting the one trajectory
      into segments does not recover enough of it to beat naive either.

    What works is going after the quantity the Levy walk is actually built
    from -- flight duration -- the direct analogue of the CTRW waiting-time
    MLE already used by :func:`estimate_exponent_ctrw`. The raw continuous-time
    event stream (:func:`~collectivediff.generators.levy.levy_walk_events`) is
    not available here; the isolated observer only ever sees the gridded
    trajectory, so flights are reconstructed from it
    (:func:`_subgrid_flight_durations`) instead of read off directly. That
    reconstruction is noisier than the true events -- a Pareto MLE on the true
    event durations reaches an MAE around 0.05 on the same held-out set where
    naive TA-MSD gets about 0.13 -- but it is still informative once reduced
    to a Hill tail estimate (:func:`_hill_tail_exponent`).

    The Hill estimate is blended with the naive exponent rather than used
    alone because its reliability depends on the flight count
    (:func:`count_levy_walk_flights`), which is not observed until estimation
    time and which shrinks as `g` approaches 1; blending degrades gracefully
    on a flight-starved trajectory instead of overcommitting to a noisy tail
    fit the way using it alone would.

    There is no fBm-style CRLB for any of this: the Levy walk has no
    closed-form Fisher information, so the gain reported here is an
    improvement over the naive fit's MAE on a held-out set, never proximity to
    an information floor.

    Parameters
    ----------
    trajectory : ndarray
        Single Levy walk trajectory.
    naive_weight : float
        Weight on the naive TA-MSD exponent in the blend; ``1 - naive_weight``
        goes to the flight-duration estimate. Tuned on a held-out set at
        ``T = 2048`` drawn the way Step 6 draws it
        (``gamma ~ U(1.2, 1.8)``, ``tau0, speed ~ U(0.5, 2.0)``); not derived
        from theory. The first tuning pass used fixed ``tau0 = speed = 1``,
        which looked like a much larger win (MAE roughly halved) than it is:
        with ``tau0`` also varying, short flights are more often sub-grid and
        the harder-to-reconstruct end of the distribution carries more
        weight, so the realised gain on Step 6's actual draw is smaller
        (~10% relative) and only visible when averaged over enough
        trajectories -- any single batch of ~150 can come out a wash by
        chance, which is what the first re-run of Step 6 with this same
        estimator showed before the weight and ``top_frac`` were retuned
        against the realistic draw.
    top_frac : float
        Fraction of the largest reconstructed durations used by the Hill
        estimator.

    Returns
    -------
    float
        Blended exponent estimate, clipped to ``(1.05, 1.95)``.
    """
    naive_alpha = _naive_tamds_exponent(trajectory)
    traj_3d = _ensure_single_trajectory_3d(trajectory)
    if traj_3d.shape[2] != 1:
        # Sub-grid reconstruction assumes the 1-D +/-1 direction law; a 2-D
        # isotropic Levy walk would need an angle-based detector instead.
        return naive_alpha

    durations = _subgrid_flight_durations(traj_3d[0, :, 0])
    gamma_hat = _hill_tail_exponent(durations, top_frac)
    if gamma_hat is None:
        return naive_alpha

    flight_alpha = float(np.clip(3.0 - np.clip(gamma_hat, 1.01, 1.99), 1.05, 1.95))
    blended = naive_weight * naive_alpha + (1.0 - naive_weight) * flight_alpha
    return float(np.clip(blended, 1.05, 1.95))


def estimate_exponent_conditional(
    trajectory: NDArray[Any],
    mechanism: str,
) -> float:
    """Estimate the diffusion exponent alpha conditionally given the mechanism.

    Parameters
    ----------
    trajectory : ndarray
        Single trajectory.
    mechanism : str
        Predicted or known mechanism name.

    Returns
    -------
    float
        Recovered anomalous exponent alpha.
    """
    if mechanism == "brownian":
        return 1.0
    elif mechanism == "ddm":
        return 1.0
    elif mechanism == "fbm":
        return estimate_exponent_fbm(trajectory)
    elif mechanism == "sbm":
        return estimate_exponent_sbm(trajectory)
    elif mechanism == "ctrw":
        return estimate_exponent_ctrw(trajectory)
    elif mechanism == "levy_walk":
        return estimate_exponent_levy_walk(trajectory)
    else:
        # Fallback to TA-MSD exponent
        traj_3d = _ensure_single_trajectory_3d(trajectory)
        f_dict = extract_feature_dict(traj_3d)
        return float(f_dict.get("tamds_exponent", 1.0))


@dataclass
class ConditionalEvaluationResult:
    """Evaluation of exponent estimation across the 3 conditions."""

    n_steps: int
    mae_naive: float
    mae_predicted: float
    mae_oracle: float
    rmse_naive: float
    rmse_predicted: float
    rmse_oracle: float
    per_mechanism_mae_naive: dict[str, float]
    per_mechanism_mae_predicted: dict[str, float]
    per_mechanism_mae_oracle: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_steps": int(self.n_steps),
            "mae_naive": float(self.mae_naive),
            "mae_predicted": float(self.mae_predicted),
            "mae_oracle": float(self.mae_oracle),
            "rmse_naive": float(self.rmse_naive),
            "rmse_predicted": float(self.rmse_predicted),
            "rmse_oracle": float(self.rmse_oracle),
            "per_mechanism_mae_naive": self.per_mechanism_mae_naive,
            "per_mechanism_mae_predicted": self.per_mechanism_mae_predicted,
            "per_mechanism_mae_oracle": self.per_mechanism_mae_oracle,
        }


def evaluate_conditional_exponent_estimation(
    trajectories: list[NDArray[np.float64]],
    true_mechanisms: list[str],
    true_alphas: list[float],
    predicted_mechanisms: list[str],
    n_steps: int,
) -> ConditionalEvaluationResult:
    """Evaluate exponent estimation across naive, predicted, and oracle conditions."""
    n_samples = len(trajectories)
    naive_alphas = np.empty(n_samples, dtype=np.float64)
    pred_alphas = np.empty(n_samples, dtype=np.float64)
    oracle_alphas = np.empty(n_samples, dtype=np.float64)
    true_alpha_arr = np.array(true_alphas, dtype=np.float64)

    for i in range(n_samples):
        traj = trajectories[i]
        f_dict = extract_feature_dict(traj)
        naive_alphas[i] = f_dict.get("tamds_exponent", 1.0)
        pred_alphas[i] = estimate_exponent_conditional(traj, predicted_mechanisms[i])
        oracle_alphas[i] = estimate_exponent_conditional(traj, true_mechanisms[i])

    err_naive = np.abs(naive_alphas - true_alpha_arr)
    err_pred = np.abs(pred_alphas - true_alpha_arr)
    err_oracle = np.abs(oracle_alphas - true_alpha_arr)

    mae_naive = float(np.mean(err_naive))
    mae_pred = float(np.mean(err_pred))
    mae_oracle = float(np.mean(err_oracle))

    rmse_naive = float(np.sqrt(np.mean((naive_alphas - true_alpha_arr) ** 2)))
    rmse_pred = float(np.sqrt(np.mean((pred_alphas - true_alpha_arr) ** 2)))
    rmse_oracle = float(np.sqrt(np.mean((oracle_alphas - true_alpha_arr) ** 2)))

    # Per-mechanism breakdowns
    unique_mechs = sorted(set(true_mechanisms))
    mech_mae_naive: dict[str, float] = {}
    mech_mae_pred: dict[str, float] = {}
    mech_mae_oracle: dict[str, float] = {}

    for m in unique_mechs:
        mask = [tm == m for tm in true_mechanisms]
        mech_mae_naive[m] = float(np.mean(err_naive[mask]))
        mech_mae_pred[m] = float(np.mean(err_pred[mask]))
        mech_mae_oracle[m] = float(np.mean(err_oracle[mask]))

    return ConditionalEvaluationResult(
        n_steps=n_steps,
        mae_naive=mae_naive,
        mae_predicted=mae_pred,
        mae_oracle=mae_oracle,
        rmse_naive=rmse_naive,
        rmse_predicted=rmse_pred,
        rmse_oracle=rmse_oracle,
        per_mechanism_mae_naive=mech_mae_naive,
        per_mechanism_mae_predicted=mech_mae_pred,
        per_mechanism_mae_oracle=mech_mae_oracle,
    )
