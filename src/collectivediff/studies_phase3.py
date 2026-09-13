"""Full Phase 3 executable studies and report generators.

Following ``PHASE3_PROMPT.md``:
- Step 1 validates the coupled model before anything else runs on top of it.
- Every study writes a small JSON summary to ``results/phase3/``.

V1's reference target changed during development: ``PROJECT.md`` section 7
originally claimed the ``J = 0`` reduction was exactly fractional Gaussian
noise. It is exactly ARFIMA(0,d,0) instead -- the two share the asymptotic
exponent but not the short-lag correlation -- and V1 as first written, which
compared against Davies-Harte fGn, could not have passed for that reason. See
``NOTES.md`` for the full diagnosis; this module validates against the
corrected target, :func:`~collectivediff.dynamics.meanfield.arfima_acf`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .config import MeanFieldConfig
from .dynamics import (
    arfima_acf,
    arfima_effective_exponent,
    coupling_for_target_crossover,
    deviation_spectrum,
    simulate_meanfield,
    whittle_d_only_estimate,
    whittle_dj_estimate,
    whittle_objective,
)
from .estimators.correlations import increment_acf
from .estimators.ergodicity import eb_parameter
from .estimators.msd import ea_msd, fit_powerlaw
from .studies import RESULTS_ROOT, write_summary

__all__ = [
    "PHASE3_RESULTS_ROOT",
    "RECALIBRATED_TAU_C",
    "combined_j_sweep",
    "run_step1_v1_reduction_check",
    "run_step1_v2_conservation_law",
    "run_step1_v3_deviation_spectrum",
    "run_step1_v4_msd_illustration",
    "run_step1_v4_whittle_dj",
    "run_step2_two_regimes",
    "run_step3_observers",
    "run_step3_residual_surface",
    "run_step3_v5_check",
    "run_step4_field_variance_vs_n",
    "run_step4_k_scaling",
    "run_step4_n_scaling",
    "run_step4_q3_correlation",
    "generate_step5_animation_data",
    "sweep_j_values",
    "whittle_d_estimate",
    "write_phase3_summary",
]

#: Phase 3 results directory
PHASE3_RESULTS_ROOT: Path = RESULTS_ROOT / "phase3"

#: Target crossovers for the recalibrated J sweep (PHASE3_V2_PREP.md), all
#: comfortably inside the trustworthy K/10 window at the phase's default K=1024.
RECALIBRATED_TAU_C: tuple[float, ...] = (8.0, 16.0, 32.0, 64.0)


def sweep_j_values(d: float) -> list[float]:
    """The recalibrated ``J`` sweep for one ``d``: the uncoupled control plus
    one ``J`` per target crossover in :data:`RECALIBRATED_TAU_C`.

    Replaces the fixed grid of ``PHASE3_PROMPT.md`` (whose predicted ``tau_c``
    at ``d = 0.25`` ranges from 16 to 160000, mostly outside the ``K/10``
    window) with one where every point lands inside the window by
    construction. Uses ``abs(d)``, so it is well-defined -- and mirrors the
    positive-``d`` grid -- for ``d < 0`` too, even though ``tau_c`` has no
    crossover interpretation on that side (``PROJECT.md`` section 7): the
    point there is a *comparable coupling strength* for the invariance test,
    not a crossover target.
    """
    if d == 0.0:
        return [0.0]
    return [0.0] + [coupling_for_target_crossover(d, tau_c) for tau_c in RECALIBRATED_TAU_C]


def whittle_d_estimate(series: NDArray[np.float64]) -> float:
    """Whittle MLE of ``d`` from a velocity-like series (e.g. the mean field).

    **Fits the ARFIMA(0,d,0) model** (:func:`~collectivediff.dynamics.meanfield.whittle_d_only_estimate`
    at ``J = 0``, correctly specified for the mean field, which is exactly
    ARFIMA -- not :func:`~collectivediff.features.spectral.whittle_mle_hurst`,
    which fits fGn.

    This was not the original implementation. The first version called
    `whittle_mle_hurst` directly, reusing Phase 2's Whittle fit "per
    PHASE3_PROMPT.md's what this phase inherits" -- but that fits the *wrong*
    model to `<v>` for exactly the reason V1 already established for a
    single agent (`NOTES.md`, step 1): ARFIMA and fGn share the asymptotic
    exponent but not the same spectral shape, so fitting fGn to genuinely
    ARFIMA data is misspecified. It went uncaught through V2, step 2's
    coherent regime, and step 3/4's field-aware observer because none of
    those checks compared the field's fitted variance to its analytic Whittle
    efficiency -- the V5 check did, and that is what surfaced it: fGn-fit
    field variance came out at roughly a quarter of the ARFIMA CRLB
    (`6 / (T pi^2)`, `pi^2/6` being the Whittle Fisher information per
    observation for ARFIMA(0,d,0), independent of `d`) rather than close to
    1, which is a mathematical impossibility for a correctly-specified
    unbiased estimator and was the tell that the *model*, not the data or the
    coupling, was wrong. Re-fit with the correct model: efficiency 1.01,
    isolated/field variance ratio 1.14 at `J=0` (`NOTES.md` has the full
    before/after).

    Parameters
    ----------
    series : ndarray of float64
        1-D velocity-like series, shape ``(n,)``, e.g.
        :attr:`~collectivediff.dynamics.meanfield.MeanFieldTrajectories.mean_field`.

    Returns
    -------
    float
        ``d_hat``.
    """
    freqs, periodogram = _periodogram(series[None, :])
    return whittle_d_only_estimate(freqs, periodogram)


def write_phase3_summary(name: str, payload: dict[str, Any]) -> Path:
    """Write summary JSON to ``results/phase3/<name>.json``."""
    return write_summary(name, payload, root=PHASE3_RESULTS_ROOT)


def _one_d_v1_check(
    d: float,
    n_particles: int,
    n_steps: int,
    k: int,
    burn_in: int,
    seed: int,
) -> dict[str, Any]:
    """V1 for a single ``d`` at ``J = 0``: ACF, in-window EA-MSD exponent, EB trend."""
    max_lag = max(5, k // 10)

    cfg = MeanFieldConfig(
        n_particles=n_particles, n_steps=n_steps, K=k, d=d, J=0.0,
        sigma=1.0, burn_in=burn_in, seed=seed,
    )
    sim = simulate_meanfield(cfg, np.random.default_rng(seed))

    # -- ACF against the exact ARFIMA(0,d,0) closed form --
    acf_sim = increment_acf(sim.positions, max_lag)
    acf_exact = arfima_acf(d, np.arange(max_lag + 1))
    acf_max_abs_dev = float(np.max(np.abs(acf_sim - acf_exact)))

    # -- EA-MSD exponent over [lag_min, K/10] against the exact ARFIMA target --
    lag_min = 2
    t = np.arange(n_steps, dtype=np.float64)
    ea = ea_msd(sim.positions)
    fit = fit_powerlaw(t, ea, float(lag_min), float(max_lag), fit=None)
    exponent_target = arfima_effective_exponent(d, lag_min, max_lag) if d != 0.0 else 1.0
    exponent_abs_dev = float(abs(fit.exponent - exponent_target))

    # -- EB trend: must fall as T grows at fixed absolute lag --
    eb_lag = max(2, k // 20)
    cfg_long = MeanFieldConfig(
        n_particles=n_particles, n_steps=4 * n_steps, K=k, d=d, J=0.0,
        sigma=1.0, burn_in=burn_in, seed=seed,
    )
    sim_long = simulate_meanfield(cfg_long, np.random.default_rng(seed))
    eb_short_t = eb_parameter(sim.positions, eb_lag)
    eb_long_t = eb_parameter(sim_long.positions, eb_lag)

    return {
        "d": d,
        "max_lag": max_lag,
        "acf_max_abs_deviation_from_exact_arfima": acf_max_abs_dev,
        "ea_msd_exponent_fitted": fit.exponent,
        "ea_msd_exponent_exact_arfima_target": exponent_target,
        "ea_msd_exponent_asymptote_1_plus_2d": 1.0 + 2.0 * d,
        "ea_msd_exponent_abs_deviation_from_exact_target": exponent_abs_dev,
        "eb_at_lag": eb_lag,
        "eb_n_steps": n_steps,
        "eb_value_at_n_steps": eb_short_t,
        "eb_n_steps_x4": 4 * n_steps,
        "eb_value_at_n_steps_x4": eb_long_t,
        "eb_shrank_with_t": bool(eb_long_t < eb_short_t),
    }


def run_step1_v1_reduction_check(
    d_values: tuple[float, ...] = (-0.25, 0.0, 0.25),
    n_particles: int = 800,
    n_steps: int = 512,
    k: int = 1024,
    seed: int = 7,
) -> dict[str, Any]:
    """V1 -- reduction to ARFIMA(0,d,0) at ``J = 0`` (``PHASE3_PROMPT.md`` step 1).

    Three checks per ``d``, all against the model's own exact closed form
    (:mod:`collectivediff.dynamics.meanfield`), not against Davies-Harte fGn:

    1. Increment ACF over ``[0, K/10]`` against :func:`arfima_acf`.
    2. EA-MSD power-law fit over ``[2, K/10]`` against
       :func:`arfima_effective_exponent` evaluated on the same window --
       *not* against the bare asymptote ``1 + 2d``, which the exact process
       does not reach inside a short window either.
    3. Ergodicity-breaking parameter at a fixed lag, checked to fall as ``T``
       grows fourfold.

    Parameters
    ----------
    d_values : tuple of float
        Fractional differencing parameters to check.
    n_particles, n_steps, k : int
        Ensemble size, trajectory length, and memory truncation.
    seed : int
        Shared seed across ``d`` values (each config's own tag still makes
        the draws distinct).

    Returns
    -------
    dict
        Per-``d`` results, written to ``results/phase3/step1_v1_reduction.json``.
    """
    burn_in = 4 * k
    per_d = [
        _one_d_v1_check(d, n_particles, n_steps, k, burn_in, seed) for d in d_values
    ]
    payload: dict[str, Any] = {
        "n_particles": n_particles,
        "n_steps": n_steps,
        "K": k,
        "burn_in": burn_in,
        "results": per_d,
    }
    write_phase3_summary("step1_v1_reduction", payload)
    return payload


def _one_dj_v2_point(
    d: float, j: float, n_particles: int, n_steps: int, k: int, burn_in: int, seed: int
) -> dict[str, Any]:
    """One ``(d, J)`` point of V2: Whittle ``d`` on the mean field."""
    cfg = MeanFieldConfig(
        n_particles=n_particles, n_steps=n_steps, K=k, d=d, J=j,
        sigma=1.0, burn_in=burn_in, seed=seed,
    )
    sim = simulate_meanfield(cfg, np.random.default_rng(seed))
    d_hat_field = whittle_d_estimate(sim.mean_field)
    return {"J": j, "d_hat_field": d_hat_field}


def _n_scaling_check(
    d: float, j: float, k: int, burn_in: int, seed: int, n_values: tuple[int, ...]
) -> dict[str, Any]:
    """Var(mean_field) at fixed (d, J), across N -- should scale as sigma^2 / N."""
    n_steps = 512
    variances: dict[int, float] = {}
    for n_particles in n_values:
        cfg = MeanFieldConfig(
            n_particles=n_particles, n_steps=n_steps, K=k, d=d, J=j,
            sigma=1.0, burn_in=burn_in, seed=seed,
        )
        sim = simulate_meanfield(cfg, np.random.default_rng(seed))
        variances[n_particles] = float(np.var(sim.mean_field, ddof=1))
    n_lo, n_hi = min(n_values), max(n_values)
    predicted_ratio = n_hi / n_lo
    measured_ratio = variances[n_lo] / variances[n_hi]
    return {
        "d": d,
        "J": j,
        "variance_by_n": {str(n): v for n, v in variances.items()},
        "predicted_ratio_var_lo_over_hi": predicted_ratio,
        "measured_ratio_var_lo_over_hi": measured_ratio,
    }


def run_step1_v2_conservation_law(
    d_values: tuple[float, ...] = (0.15, 0.25, 0.35, 0.45, -0.15, -0.25, -0.35, -0.45),
    n_particles: int = 400,
    n_steps: int = 512,
    k: int = 1024,
    seed: int = 13,
    n_scaling_d: float = 0.25,
    n_scaling_j: float = 0.3,
    n_scaling_values: tuple[int, ...] = (200, 1600),
) -> dict[str, Any]:
    """V2 -- conservation law (``PHASE3_PROMPT.md`` step 1; ``PHASE3_V2_PREP.md``).

    For every ``d`` in ``d_values``, Whittle-fits ``d`` from the mean field
    across the recalibrated ``J`` sweep (:func:`sweep_j_values`) and reports
    the spread across ``J`` -- the conservation law predicts this spread is
    consistent with Whittle's own fit noise, not a drift with ``J``,
    *regardless of the sign of `d`*: `<v>(t)` is `ARFIMA(0,d,0)` at every `J`
    on both sides (``PROJECT.md`` section 7), even though a single *agent's*
    regime only survives coupling for `d < 0`.

    Separately checks ``Var(mean_field) ~ sigma^2 / N`` at one representative
    ``(d, J)`` across two ``N`` (V5's companion check, run once here since it
    does not depend on the sign of ``d``).

    Parameters
    ----------
    d_values : tuple of float
        Fractional differencing parameters, both signs.
    n_particles, n_steps, k : int
        Ensemble size, trajectory length, memory truncation.
    seed : int
        Shared seed.
    n_scaling_d, n_scaling_j : float
        The ``(d, J)`` point used for the ``Var ~ 1/N`` check.
    n_scaling_values : tuple of int
        The two ``N`` values compared.

    Returns
    -------
    dict
        Written to ``results/phase3/step1_v2_conservation_law.json``.
    """
    burn_in = 4 * k
    per_d: list[dict[str, Any]] = []
    for d in d_values:
        points = [
            _one_dj_v2_point(d, j, n_particles, n_steps, k, burn_in, seed)
            for j in sweep_j_values(d)
        ]
        d_hats = np.array([p["d_hat_field"] for p in points])
        per_d.append(
            {
                "d": d,
                "points": points,
                "d_hat_field_mean": float(d_hats.mean()),
                "d_hat_field_std_across_J": float(d_hats.std(ddof=1)),
                "d_hat_field_max_abs_deviation_from_mean": float(
                    np.max(np.abs(d_hats - d_hats.mean()))
                ),
            }
        )

    n_scaling = _n_scaling_check(
        n_scaling_d, n_scaling_j, k, burn_in, seed, n_scaling_values
    )

    payload: dict[str, Any] = {
        "n_particles": n_particles,
        "n_steps": n_steps,
        "K": k,
        "burn_in": burn_in,
        "tau_c_targets": list(RECALIBRATED_TAU_C),
        "results": per_d,
        "n_scaling_check": n_scaling,
    }
    write_phase3_summary("step1_v2_conservation_law", payload)
    return payload


def _periodogram(series: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Ensemble-averaged periodogram of ``delta_i = v_i - <v>``, one row per agent.

    Same convention as :func:`~collectivediff.features.spectral.whittle_log_likelihood`
    (``|FFT|^2 / (2 pi n)``), matched by :func:`~collectivediff.dynamics.meanfield.deviation_spectrum`.
    """
    n = series.shape[1]
    fft_vals = np.fft.rfft(series - series.mean(axis=1, keepdims=True), axis=1)[:, 1:]
    freqs = 2.0 * np.pi * np.arange(1, fft_vals.shape[1] + 1, dtype=np.float64) / n
    periodogram = (np.abs(fft_vals) ** 2) / (2.0 * np.pi * n)
    return freqs, periodogram.mean(axis=0)


def run_step1_v3_deviation_spectrum(
    d_values: tuple[float, ...] = (0.25, -0.25),
    j: float = 0.3,
    n_particles: int = 500,
    n_steps: int = 1024,
    k: int = 1024,
    seed: int = 17,
) -> dict[str, Any]:
    """V3 -- deviation spectrum against its closed form (``PHASE3_PROMPT.md`` step 1).

    Periodogram of ``delta_i = v_i - <v>``, averaged over agents, against
    :func:`~collectivediff.dynamics.meanfield.deviation_spectrum`. Compared
    over two frequency bands, both reported:

    - **trustworthy**, ``omega >= 2 pi / (K/10)``: lags ``<= K/10``, the
      window every claim about the coupled model is restricted to.
    - **extended**, the full range down to ``omega = 2 pi / (T - 1)``: lags
      up to the trajectory length itself, i.e. deliberately beyond the
      trustworthy window, to see whether -- and how far past ``K/10`` -- the
      truncated model's own deviation spectrum still tracks the closed form
      before the ``K``-truncation artifact of ``NOTES.md`` (finite ``S(0)``
      where the exact model diverges or vanishes) takes over.

    Parameters
    ----------
    d_values : tuple of float
        One positive and one negative ``d`` by default, to see the sign
        asymmetry of ``PROJECT.md`` section 7 in the raw spectrum shape, not
        just in a fitted exponent.
    j : float
        Coupling strength, shared across ``d_values`` (``J = 0.3`` sits
        inside the recalibrated window's crossover range for every ``d`` in
        :data:`RECALIBRATED_TAU_C`).
    n_particles, n_steps, k : int
        Ensemble size, trajectory length, memory truncation.
    seed : int
        Shared seed.

    Returns
    -------
    dict
        Written to ``results/phase3/step1_v3_deviation_spectrum.json``.
    """
    burn_in = 4 * k
    trustworthy_omega_min = 2.0 * np.pi / (k / 10.0)

    per_d: list[dict[str, Any]] = []
    for d in d_values:
        cfg = MeanFieldConfig(
            n_particles=n_particles, n_steps=n_steps, K=k, d=d, J=j,
            sigma=1.0, burn_in=burn_in, seed=seed,
        )
        sim = simulate_meanfield(cfg, np.random.default_rng(seed))
        delta = sim.velocities[:, :, 0] - sim.mean_field[None, :]
        freqs, periodogram = _periodogram(delta)
        theory = deviation_spectrum(freqs, d, j)
        log_ratio = np.log(periodogram / theory)

        trustworthy = freqs >= trustworthy_omega_min
        per_d.append(
            {
                "d": d,
                "J": j,
                "n_freq_trustworthy_band": int(trustworthy.sum()),
                "n_freq_extended_band": int((~trustworthy).sum()),
                "trustworthy_band_mean_abs_log_ratio": float(
                    np.mean(np.abs(log_ratio[trustworthy]))
                ),
                "extended_band_mean_abs_log_ratio": float(
                    np.mean(np.abs(log_ratio[~trustworthy]))
                )
                if (~trustworthy).any()
                else None,
                "omega_min_trustworthy": trustworthy_omega_min,
            }
        )

    payload: dict[str, Any] = {
        "n_particles": n_particles,
        "n_steps": n_steps,
        "K": k,
        "burn_in": burn_in,
        "J": j,
        "results": per_d,
    }
    write_phase3_summary("step1_v3_deviation_spectrum", payload)
    return payload


def run_step1_v4_whittle_dj(
    d_values: tuple[float, ...] = (0.15, 0.25, 0.35, 0.45),
    j_fractions_of_half_d: tuple[float, ...] = (0.1, 0.3, 0.6),
    n_particles: int = 400,
    n_steps: int = 1024,
    k: int = 1024,
    seed: int = 31,
) -> dict[str, Any]:
    """V4 -- joint ``(d, J)`` Whittle fit to the deviation periodogram.

    Replaces an earlier V4 design that tried to locate the memory crossover
    ``tau_c`` from the local slope of a single agent's EA-MSD. That measurement
    was unusably noisy even after averaging several independent realizations
    -- the local exponent never approached ``1 + 2d`` at short lag for any
    tested ``J``, including couplings well inside the region the crossover
    picture was supposed to hold -- and the root cause was the sweep itself:
    a `J` comparable to or larger than `kappa(1) = d` competes with the memory
    term already at the shortest lags, not just "beyond tau_c" as the
    low-frequency asymptotic derivation of `tau_c` assumes. See ``NOTES.md``
    for the full diagnosis.

    This instead fits :func:`~collectivediff.dynamics.meanfield.deviation_spectrum`
    to the ensemble-averaged periodogram of ``delta_i = v_i - <v>`` by joint
    Whittle MLE over ``(d, J)`` -- the whole spectral shape at once, with no
    window or lag choice, and a pass/fail criterion with no free parameter:
    does ``J_hat`` recover the ``J`` used to generate the data, and does
    ``d_hat`` stay independent of ``J``. `J` is chosen from
    ``J < |d| / 2`` so memory dominates at short lags, deliberately outside
    the regime where the abandoned MSD-crossover measurement was attempted.

    Parameters
    ----------
    d_values : tuple of float
        Fractional differencing parameters, positive (matching
        ``PROJECT.md`` section 7's V4 scope).
    j_fractions_of_half_d : tuple of float
        `J` grid as fractions of `|d| / 2`, shared across `d_values` (all
        `< 1` so the constraint holds for every `d`); `J = 0` is always
        included as the uncoupled control.
    n_particles, n_steps, k : int
        Ensemble size, trajectory length, memory truncation.
    seed : int
        Shared seed.

    Returns
    -------
    dict
        Written to ``results/phase3/step1_v4_whittle_dj.json``.
    """
    burn_in = 4 * k
    per_d: list[dict[str, Any]] = []
    for d in d_values:
        j_max = abs(d) / 2.0
        j_values = [0.0] + [frac * j_max for frac in j_fractions_of_half_d]

        points: list[dict[str, Any]] = []
        for j in j_values:
            cfg = MeanFieldConfig(
                n_particles=n_particles, n_steps=n_steps, K=k, d=d, J=j,
                sigma=1.0, burn_in=burn_in, seed=seed,
            )
            sim = simulate_meanfield(cfg, np.random.default_rng(seed))
            delta = sim.velocities[:, :, 0] - sim.mean_field[None, :]
            freqs, periodogram = _periodogram(delta)
            d_hat, j_hat = whittle_dj_estimate(freqs, periodogram)
            points.append(
                {
                    "J": j,
                    "d_hat": d_hat,
                    "J_hat": j_hat,
                    "J_abs_error": abs(j_hat - j),
                }
            )

        d_hats = np.array([p["d_hat"] for p in points])
        per_d.append(
            {
                "d": d,
                "j_max_constraint": j_max,
                "points": points,
                "d_hat_mean": float(d_hats.mean()),
                "d_hat_std_across_J": float(d_hats.std(ddof=1)),
                "d_hat_max_abs_deviation_from_true": float(np.max(np.abs(d_hats - d))),
            }
        )

    payload: dict[str, Any] = {
        "n_particles": n_particles,
        "n_steps": n_steps,
        "K": k,
        "burn_in": burn_in,
        "results": per_d,
    }
    write_phase3_summary("step1_v4_whittle_dj", payload)
    return payload


def run_step1_v4_msd_illustration(
    d: float = 0.45,
    j_values: tuple[float, ...] = (0.05, 0.15, 0.3),
    n_particles: int = 150,
    n_steps: int = 2048,
    k: int = 16384,
    burn_in_multiple: int = 2,
    seed: int = 23,
) -> dict[str, Any]:
    """Single-agent EA-MSD local-exponent curve, for step 5's crossover figure only.

    **Qualitative illustration, not a quantitative test** -- see
    :func:`run_step1_v4_whittle_dj` for the actual V4 validation. This is the
    same local-slope measurement the original V4 design tried and found too
    noisy to use as a pass/fail criterion; kept at one point, at a much
    larger ``K`` (so the trustworthy ``K / 10`` window has enough room to
    show the shape even if not the precise crossover lag), because step 5
    wants a real picture of the crossover, not a schematic. ``burn_in`` is
    relaxed to ``2 K`` (config allows any value ``>= K``) rather than the
    usual ``4 K``, since ``K = 16384`` makes the default prohibitively slow
    and this is not a precision measurement.

    Parameters
    ----------
    d : float
        Fractional differencing parameter.
    j_values : tuple of float
        Coupling strengths to compare.
    n_particles, n_steps, k : int
        Ensemble size, trajectory length, memory truncation.
    burn_in_multiple : int
        Burn-in as a multiple of ``k``.
    seed : int
        Shared seed.

    Returns
    -------
    dict
        Written to ``results/phase3/step1_v4_msd_illustration.json``.
    """
    burn_in = burn_in_multiple * k
    max_lag = max(5, k // 10)
    lags = np.unique(np.round(np.logspace(0, np.log10(max_lag), 30)).astype(np.intp))
    lags = lags[lags >= 1]

    curves: list[dict[str, Any]] = []
    for j in j_values:
        cfg = MeanFieldConfig(
            n_particles=n_particles, n_steps=n_steps, K=k, d=d, J=j,
            sigma=1.0, burn_in=burn_in, seed=seed,
        )
        sim = simulate_meanfield(cfg, np.random.default_rng(seed))
        msd = ea_msd(sim.positions)
        log_lag = np.log(lags.astype(np.float64))
        log_msd = np.log(msd[lags])
        local_exponent = np.gradient(log_msd, log_lag)

        # Coherent mode's own squared displacement, same (J, seed) realization,
        # for step 5's overlay -- the crossover picture is only meaningful
        # shown against the coherent curve that stays at 1 + 2d regardless of J.
        com = sim.center_of_mass()
        com_sq_disp = com[lags] ** 2
        log_com = np.log(np.maximum(com_sq_disp, 1e-300))
        local_exponent_coherent = np.gradient(log_com, log_lag)

        curves.append(
            {
                "J": j,
                "lags": lags.tolist(),
                "local_exponent": local_exponent.tolist(),
                "local_exponent_coherent": local_exponent_coherent.tolist(),
            }
        )

    payload: dict[str, Any] = {
        "d": d,
        "asymptote_1_plus_2d": 1.0 + 2.0 * d,
        "n_particles": n_particles,
        "n_steps": n_steps,
        "K": k,
        "burn_in": burn_in,
        "max_lag": max_lag,
        "curves": curves,
        "note": "qualitative illustration for step 5's figure, not a quantitative test",
    }
    write_phase3_summary("step1_v4_msd_illustration", payload)
    return payload


def combined_j_sweep(d: float) -> list[float]:
    """Wide ``J`` sweep spanning weak (memory-dominant) through strong
    (crossover-inducing) coupling, for step 2's regime table.

    V4's ``J < |d| / 2`` grid was deliberately weak, chosen so memory
    dominates at short lags -- the opposite of what step 2 needs to see, which
    is the typical-agent regime actually crossing over toward 1 as coupling
    strengthens. Combines that weak grid with the ``tau_c``-targeted strong
    grid already validated for V2 (:func:`sweep_j_values`), giving points from
    near-zero coupling up through couplings strong enough to push a `d > 0`
    typical agent noticeably toward normal diffusion. Uses ``abs(d)`` so the
    same magnitudes apply on both signs, for a direct comparison.
    """
    ad = abs(d)
    weak = [frac * ad / 2.0 for frac in (0.1, 0.3, 0.6)]
    strong = [coupling_for_target_crossover(ad, tau_c) for tau_c in RECALIBRATED_TAU_C]
    return sorted(set([0.0] + weak + strong))


def _one_dj_step2_point(
    d: float, j: float, n_particles: int, n_steps: int, k: int, burn_in: int, seed: int
) -> dict[str, Any]:
    """One ``(d, J)`` point: typical-agent and coherent exponents.

    **Typical-agent regime is a windowed EA-MSD power-law fit**
    (``fit_powerlaw`` over ``[2, K/10]`` on the ensemble-averaged EA-MSD,
    ``PROJECT.md``'s standard TA/EA-MSD fitting convention) -- not the naive
    single-parameter Whittle fit tried first. The naive Whittle
    (:func:`~collectivediff.dynamics.meanfield.whittle_d_only_estimate`,
    forcing ``J = 0`` in the model) turned out to give a *misleading* number
    for ``d < 0``: it declines toward 0 as ``J`` grows for *both* signs of
    ``d`` (e.g. ``d = -0.25``: implied alpha runs 0.50 -> 0.02), contradicting
    ``PROJECT.md`` section 7's prediction that subdiffusive memory survives
    coupling -- confirmed spurious against the windowed EA-MSD fit on the
    *same* simulated data, which stays flat near the asymptote for
    ``d = -0.25`` across the whole sweep (0.55 -> 0.52) while still declining
    correctly for ``d = 0.25`` (1.54 -> 1.18). The naive Whittle fit's decline
    for `d < 0` is a model-misspecification artefact -- forcing a pure
    ARFIMA shape onto genuinely `J`-reshaped data biases the fit regardless
    of which sign of `d` the *true*, generative exponent has -- not a
    reflection of the true regime, which the windowed MSD fit reads directly
    with no model assumption to get wrong. See ``NOTES.md`` for the full
    three-way comparison. Both Whittle variants are still reported alongside
    (`*_naive_whittle`, `*_model_aware`) because the contrast itself is
    informative: model-aware recovers the true, constant `d` on both signs
    (that's what "model-aware" buys you, and is step 3's subject); naive
    Whittle shows *why* an uninformed observer can be misled on either sign,
    which the windowed MSD fit -- being model-free -- is not susceptible to.

    The coherent estimate is the single-parameter Whittle fit already used
    for V2 (:func:`whittle_d_estimate`), since ``<v>`` is plain
    ARFIMA(0,d,0) with no ``J`` dependence to fit and no misspecification risk.
    """
    cfg = MeanFieldConfig(
        n_particles=n_particles, n_steps=n_steps, K=k, d=d, J=j,
        sigma=1.0, burn_in=burn_in, seed=seed,
    )
    sim = simulate_meanfield(cfg, np.random.default_rng(seed))

    max_lag = max(5, k // 10)
    ea = ea_msd(sim.positions)
    t = np.arange(n_steps, dtype=np.float64)
    msd_fit = fit_powerlaw(t, ea, 2.0, float(max_lag), fit=None)

    freqs, periodogram_v = _periodogram(sim.velocities[:, :, 0])
    d_hat_naive_whittle = whittle_d_only_estimate(freqs, periodogram_v)
    d_hat_model_aware, j_hat_model_aware = whittle_dj_estimate(freqs, periodogram_v)
    d_hat_coherent = whittle_d_estimate(sim.mean_field)
    return {
        "J": j,
        "alpha_typical_agent": msd_fit.exponent,
        "alpha_typical_agent_naive_whittle": 1.0 + 2.0 * d_hat_naive_whittle,
        "d_hat_typical_agent_model_aware": d_hat_model_aware,
        "J_hat_typical_agent_model_aware": j_hat_model_aware,
        "alpha_typical_agent_model_aware": 1.0 + 2.0 * d_hat_model_aware,
        "d_hat_coherent": d_hat_coherent,
        "alpha_coherent": 1.0 + 2.0 * d_hat_coherent,
    }


def run_step2_two_regimes(
    d_values: tuple[float, ...] = (0.15, 0.25, 0.35, 0.45, -0.15, -0.25, -0.35, -0.45),
    n_particles: int = 400,
    n_steps: int = 1024,
    k: int = 1024,
    seed: int = 41,
) -> dict[str, Any]:
    """Step 2 -- the two collective regimes, split by sign of ``d`` (``PHASE3_PROMPT.md``).

    ``PROJECT.md`` section 7's prediction is sign-dependent (confirmed
    analytically and in the deviation spectrum, V3/V4):

    - **d > 0**: coupling kills superdiffusive memory. Typical-agent regime
      (ensemble EA-MSD exponent, estimated here via Whittle rather than an
      MSD slope) should fall from ``1 + 2d`` toward 1 as ``J`` strengthens;
      coherent regime (center-of-mass / mean-field exponent) stays at
      ``1 + 2d`` regardless of ``J`` (the conservation law, V2).
    - **d < 0**: coupling does *not* kill subdiffusive memory. Both regimes
      should stay near ``1 + 2d`` across the whole sweep.

    Seeing this asymmetry appear here, from an entirely different observable
    (ensemble EA-MSD growth exponent) than the deviation-spectrum fit V3/V4
    used, is an independent confirmation of the same physical claim -- the
    point of running it, not a redundant check.

    Parameters
    ----------
    d_values : tuple of float
        Both signs, matching V1/V2/V4's convention.
    n_particles, n_steps, k : int
        Ensemble size, trajectory length, memory truncation.
    seed : int
        Shared seed.

    Returns
    -------
    dict
        ``results_positive_d`` and ``results_negative_d`` reported
        separately, per direction. Written to
        ``results/phase3/step2_two_regimes.json``.
    """
    burn_in = 4 * k
    per_d: list[dict[str, Any]] = []
    for d in d_values:
        points = [
            _one_dj_step2_point(d, j, n_particles, n_steps, k, burn_in, seed)
            for j in combined_j_sweep(d)
        ]
        per_d.append(
            {
                "d": d,
                "asymptote_1_plus_2d": 1.0 + 2.0 * d,
                "points": points,
            }
        )

    payload: dict[str, Any] = {
        "n_particles": n_particles,
        "n_steps": n_steps,
        "K": k,
        "burn_in": burn_in,
        "results_positive_d": [e for e in per_d if e["d"] > 0],
        "results_negative_d": [e for e in per_d if e["d"] < 0],
    }
    write_phase3_summary("step2_two_regimes", payload)
    return payload


def _train_step3_classifier(
    n_steps: int, n_samples_per_class: int = 100, seed: int = 42
) -> tuple[Any, list[str], list[str]]:
    """Train the Phase 2 classifier once per trajectory length, for reuse
    across every ``(d, J)`` point of step 3's isolated model-agnostic observer.
    """
    from .inference.classification import make_classifier
    from .inference.dataset import DatasetConfig, load_or_generate_dataset

    train_cfg = DatasetConfig(
        n_samples_per_class=n_samples_per_class, n_steps=n_steps, split="train", seed=5000 + n_steps
    )
    train_ds = load_or_generate_dataset(train_cfg)
    clf = make_classifier("gradient_boosting", seed=seed)
    clf.fit(train_ds.features, train_ds.labels)
    return clf, train_ds.mechanism_names, train_ds.feature_names


def _isolated_model_agnostic_d(
    trajectory: NDArray[np.float64], clf: Any, mechanism_names: list[str], feature_names: list[str]
) -> float:
    """Step 3's isolated, model-agnostic observer: Phase 2's corrected pipeline."""
    from .features.extraction import extract_features
    from .inference.conditional import estimate_exponent_conditional

    feat = extract_features(trajectory, feature_names).reshape(1, -1)
    mechanism = mechanism_names[int(clf.predict(feat)[0])]
    alpha_hat = estimate_exponent_conditional(trajectory, mechanism)
    return (alpha_hat - 1.0) / 2.0


def _boundary_proximity(
    d_hat: float,
    j_hat: float,
    d_bounds: tuple[float, float] = (-0.49, 0.49),
    j_bounds: tuple[float, float] = (1e-4, 2.0),
) -> float:
    """Distance from ``(d_hat, j_hat)`` to the nearest search-box edge."""
    d_lo, d_hi = d_bounds
    j_lo, j_hi = j_bounds
    return min(d_hat - d_lo, d_hi - d_hat, j_hat - j_lo, j_hi - j_hat)


def run_step3_residual_surface(
    points: tuple[tuple[float, float], ...] = (
        (0.25, coupling_for_target_crossover(0.25, 8.0)),
        (-0.25, coupling_for_target_crossover(0.25, 64.0)),
    ),
    n_particles: int = 400,
    n_steps: int = 1024,
    k: int = 1024,
    n_agents_sampled: int = 60,
    grid_n: int = 41,
    pin_tol: float = 0.01,
    base_seed: int = 1300,
) -> dict[str, Any]:
    """Item 3's residual-surface diagnostic (pre-phase-4 close-out, NOTES.md).

    ``run_step3_observers`` shows isolated model-aware
    (:func:`~collectivediff.dynamics.meanfield.whittle_dj_estimate` on one
    agent's own periodogram) with far higher variance than isolated
    model-agnostic at every ``J`` -- e.g. ``d=0.25, J=0``: agnostic variance
    0.00051 vs aware 0.0076. An ad hoc 60-agent sample at one point found 8
    (13%) of individual fits pinned at the ``d_hat`` upper bound ``0.49``,
    and a scan of the profiled objective along ``d`` at one pinned case's
    fitted ``J`` was strictly decreasing all the way to the bound -- a
    genuine boundary-seeking optimum for that noise realization, not an
    interior optimum the multi-start missed (``NOTES.md``). That
    investigation was never saved as a reproducible artifact; this is that
    artifact, generalised to a couple of ``(d, J)`` points, with an explicit
    check that a fine grid search over the whole box doesn't find a better
    point than the multi-start optimizer already reports (the thing that
    *would* indicate a bug rather than a genuine finite-sample effect).

    Default points are ``(d=0.25, tau_c=8)`` and ``(d=-0.25, tau_c=64)`` --
    members of :func:`combined_j_sweep`'s strong grid, picked from
    ``results/phase3/step3_observers.json`` as the point with the largest
    isolated model-aware variance (``d=0.25``) and the point with the
    largest isolated model-aware bias (``d=-0.25``) in that table.

    For each point: one simulation, ``n_agents_sampled`` individual agents
    each fit by :func:`~collectivediff.dynamics.meanfield.whittle_dj_estimate`,
    and -- for the single sampled agent whose fit lands closest to either
    search-box edge -- a ``grid_n x grid_n`` evaluation of
    :func:`~collectivediff.dynamics.meanfield.whittle_objective` over the
    whole ``(d, J)`` box. The grid reuses that agent's already-computed
    periodogram, so it costs nothing beyond the per-agent fits already run.

    Parameters
    ----------
    points : tuple of (float, float)
        ``(d, J)`` points to examine.
    n_particles, n_steps, k : int
        Ensemble size, trajectory length, memory truncation.
    n_agents_sampled : int
        Individual agents fit per point.
    grid_n : int
        Grid resolution per axis for the residual-surface scan.
    pin_tol : float
        An estimate counts as boundary-pinned if either parameter lands
        within this distance of its search bound.
    base_seed : int
        Simulation seed; the agent-sampling seed is derived from it the same
        way as elsewhere in this module (``base_seed * 7919 + 1``).

    Returns
    -------
    dict
        Written to ``results/phase3/step3_residual_surface.json``.
    """
    burn_in = 4 * k
    d_bounds = (-0.49, 0.49)
    j_bounds = (1e-4, 2.0)
    d_grid = np.linspace(d_bounds[0], d_bounds[1], grid_n)
    j_grid = np.linspace(j_bounds[0], j_bounds[1], grid_n)

    per_point: list[dict[str, Any]] = []
    for d, j in points:
        cfg = MeanFieldConfig(
            n_particles=n_particles, n_steps=n_steps, K=k, d=d, J=j,
            sigma=1.0, burn_in=burn_in, seed=base_seed,
        )
        sim = simulate_meanfield(cfg, np.random.default_rng(base_seed))

        sample_rng = np.random.default_rng(base_seed * 7919 + 1)
        agent_idx = sample_rng.choice(n_particles, size=n_agents_sampled, replace=False)

        d_hats: list[float] = []
        pinned: list[bool] = []
        worst_proximity = np.inf
        worst: tuple[int, NDArray[np.float64], NDArray[np.float64], float, float] | None = None
        for ai in agent_idx:
            freqs, per_v = _periodogram(sim.velocities[int(ai) : int(ai) + 1, :, 0])
            d_hat, j_hat = whittle_dj_estimate(freqs, per_v)
            proximity = _boundary_proximity(d_hat, j_hat, d_bounds, j_bounds)
            d_hats.append(d_hat)
            pinned.append(proximity <= pin_tol)
            if proximity < worst_proximity:
                worst_proximity = proximity
                worst = (int(ai), freqs, per_v, d_hat, j_hat)

        assert worst is not None
        worst_agent, worst_freqs, worst_per, worst_d_hat, worst_j_hat = worst

        surface = np.array(
            [
                [whittle_objective(worst_freqs, worst_per, dd, jj) for jj in j_grid]
                for dd in d_grid
            ]
        )
        gi, gj = np.unravel_index(np.argmin(surface), surface.shape)
        grid_d_argmin, grid_j_argmin = float(d_grid[gi]), float(j_grid[gj])
        d_step, j_step = float(d_grid[1] - d_grid[0]), float(j_grid[1] - j_grid[0])
        grid_matches_optimizer = bool(
            abs(grid_d_argmin - worst_d_hat) <= 2.0 * d_step
            and abs(grid_j_argmin - worst_j_hat) <= 2.0 * j_step
        )

        d_scan = np.array(
            [whittle_objective(worst_freqs, worst_per, dd, worst_j_hat) for dd in d_grid]
        )
        trend = np.diff(d_scan) if worst_d_hat >= 0 else np.diff(d_scan[::-1])
        monotonic_toward_bound = bool(np.all(trend <= 1e-6))

        d_arr = np.array(d_hats)
        pinned_arr = np.array(pinned)
        n_pinned = int(pinned_arr.sum())
        stats_excl = (
            _stats(list(d_arr[~pinned_arr]), d) if n_pinned < len(d_hats) else None
        )

        per_point.append(
            {
                "d": d,
                "J": j,
                "n_agents_sampled": n_agents_sampled,
                "n_pinned": n_pinned,
                "pinned_fraction": n_pinned / n_agents_sampled,
                "stats_including_pinned": _stats(d_hats, d),
                "stats_excluding_pinned": stats_excl,
                "worst_agent_index": worst_agent,
                "worst_agent_d_hat": worst_d_hat,
                "worst_agent_j_hat": worst_j_hat,
                "worst_agent_boundary_proximity": float(worst_proximity),
                "worst_agent_is_pinned": bool(worst_proximity <= pin_tol),
                "grid_d_argmin": grid_d_argmin,
                "grid_j_argmin": grid_j_argmin,
                "grid_matches_optimizer_result": grid_matches_optimizer,
                "d_scan_monotonic_toward_pinned_bound": monotonic_toward_bound,
                "d_grid": d_grid.tolist(),
                "j_grid": j_grid.tolist(),
                "objective_surface": surface.tolist(),
            }
        )

    payload: dict[str, Any] = {
        "n_particles": n_particles,
        "n_steps": n_steps,
        "K": k,
        "burn_in": burn_in,
        "n_agents_sampled": n_agents_sampled,
        "grid_n": grid_n,
        "pin_tol": pin_tol,
        "results": per_point,
    }
    write_phase3_summary("step3_residual_surface", payload)
    return payload


def run_step4_field_variance_vs_n(
    d: float = 0.25,
    n_values: tuple[int, ...] = (50, 100, 400, 1600),
    n_steps: int = 1024,
    k: int = 1024,
    n_realizations: int = 20,
    base_seed: int = 1400,
) -> dict[str, Any]:
    """Tighter-CI check of ``run_step4_n_scaling``'s field-variance-vs-``N`` trend.

    ``run_step4_n_scaling`` at its default ``n_realizations=10`` showed
    ``field_variance`` rising monotonically with ``N`` (0.000248 -> 0.000340
    -> 0.000417 -> 0.000555 across ``N = 50, 100, 400, 1600``), which
    shouldn't happen: ``n_steps`` is fixed across the sweep and Whittle
    precision for a spectral *shape* parameter is scale-invariant (V5) --
    ``Var(d_hat_field)`` should not depend on ``N`` at all. With only 10
    realizations a variance estimate has ~47% relative sampling noise
    (chi-squared, 9 d.o.f.), so the rise could be noise, but a monotonic
    trend across 4 points has only ~4% probability by chance.

    Uses ``J = 0`` only rather than re-running both ``J`` values: the
    conservation law (V2) makes the field's spectrum exactly
    ``J``-independent, and ``run_step4_n_scaling``'s own output confirms
    this empirically -- ``field_variance`` was bit-identical between its
    ``J=0.0`` and ``J=0.3`` rows at every ``N``. Re-checking at a second
    ``J`` would test nothing new; the simulations saved by dropping it go
    into more realizations per ``N`` instead, for a materially tighter
    variance-of-variance estimate at the one point that matters.

    Reports a chi-squared 95% confidence interval on each per-``N`` variance
    estimate (``(n-1) Var / chi2_{quantile}``) so the trend can be judged
    against sampling noise rather than eyeballed, plus the fraction of
    individual ``d_hat`` draws landing within 0.01 of the ``d`` search bound
    ``+-0.49`` at each ``N`` -- if the rise is driven by more boundary-pinned
    fits at large ``N`` (analogous to item 3's finding for the isolated
    model-aware observer), that fraction should itself rise with ``N``.

    Parameters
    ----------
    d : float
        Fractional differencing parameter, one representative ``d > 0``,
        matching ``run_step4_n_scaling``'s default.
    n_values : tuple of int
        Ensemble sizes to sweep.
    n_steps, k : int
        Trajectory length, memory truncation.
    n_realizations : int
        Independent simulations per ``N``.
    base_seed : int
        First realization's seed.

    Returns
    -------
    dict
        Written to ``results/phase3/step4_field_variance_vs_n.json``.
    """
    from scipy.stats import chi2

    burn_in = 4 * k
    dof = n_realizations - 1
    per_n: list[dict[str, Any]] = []
    for n in n_values:
        estimates: list[float] = []
        for r in range(n_realizations):
            seed = base_seed + r
            cfg = MeanFieldConfig(
                n_particles=n, n_steps=n_steps, K=k, d=d, J=0.0,
                sigma=1.0, burn_in=burn_in, seed=seed,
            )
            sim = simulate_meanfield(cfg, np.random.default_rng(seed))
            estimates.append(whittle_d_estimate(sim.mean_field))

        arr = np.array(estimates)
        var = float(arr.var(ddof=1))
        ci_low = dof * var / float(chi2.ppf(0.975, dof))
        ci_high = dof * var / float(chi2.ppf(0.025, dof))
        near_bound = float(np.mean(np.abs(np.abs(arr) - 0.49) < 0.01))
        per_n.append(
            {
                "N": n,
                "variance": var,
                "variance_ci95_low": ci_low,
                "variance_ci95_high": ci_high,
                "mean": float(arr.mean()),
                "fraction_near_d_bound": near_bound,
            }
        )

    payload: dict[str, Any] = {
        "d": d,
        "J": 0.0,
        "n_steps": n_steps,
        "K": k,
        "n_realizations": n_realizations,
        "n_values": list(n_values),
        "note": (
            "J = 0 only: field variance is exactly J-independent by the "
            "conservation law (V2), confirmed bit-identical between the "
            "J=0.0 and J=0.3 rows of run_step4_n_scaling; simulations saved "
            "by dropping the second J go into more realizations per N here."
        ),
        "results": per_n,
    }
    write_phase3_summary("step4_field_variance_vs_n", payload)
    return payload


def run_step4_k_scaling(
    k_values: tuple[int, ...] = (256, 1024, 4096, 16384),
    n_particles: int = 50,
    n_steps: int = 64,
    d: float = 0.25,
    burn_in_multiple: int = 2,
    seed: int = 1500,
) -> dict[str, Any]:
    """Wall-clock cost of :func:`~collectivediff.dynamics.meanfield.simulate_meanfield`
    versus ``K`` -- caps Phase 4's feasible ``K`` (item 4b, pre-phase-4 close-out).

    ``NOTES.md``'s only prior data point ("16x K gives ~110x time") compared
    runs at different ``N`` and different burn-in *multiples* of ``K``, not a
    controlled sweep. This holds ``N``, ``T`` and the burn-in multiple fixed
    (``burn_in = 2K``, matching the only precedent for ``K = 16384`` in this
    codebase, :func:`run_step1_v4_msd_illustration`) and measures wall time
    with ``time.perf_counter()`` directly -- there is no benchmarking
    fixture in this project, and one measurement per ``K`` is enough to see
    the shape, which is the point (a single-run comparison, same caveat as
    every other timing note in ``NOTES.md``).

    Fits a log-log OLS slope, ``time ~ K^p``. The *algorithmic* baseline is
    not ``p = 1``: :func:`simulate_meanfield`'s per-step cost is ``O(N K)``
    (the memory-kernel matmul), but the number of steps run scales with the
    burn-in, which is itself ``O(K)`` here -- so the naive prediction from
    complexity alone is already ``p = 2``. Anything materially above 2 is
    not explained by algorithmic complexity and points to memory bandwidth
    (the ``(N, K)`` ring buffer stops fitting in cache) instead.

    Parameters
    ----------
    k_values : tuple of int
        Memory truncations to sweep, increasing.
    n_particles, n_steps : int
        Held fixed and small so the largest ``K`` finishes in reasonable
        wall time; not representative of phase 3's analysis scale -- this is
        a timing study only, not a physics result.
    d : float
        Fractional differencing parameter; timing does not depend on ``d``,
        held fixed rather than swept.
    burn_in_multiple : int
        Burn-in as a multiple of ``K``.
    seed : int
        Shared seed (timing should not depend on the noise realization, but
        one is needed to run the simulation).

    Returns
    -------
    dict
        Written to ``results/phase3/step4_k_scaling.json``.
    """
    import time

    per_k: list[dict[str, Any]] = []
    for k in k_values:
        burn_in = burn_in_multiple * k
        cfg = MeanFieldConfig(
            n_particles=n_particles, n_steps=n_steps, K=k, d=d, J=0.0,
            sigma=1.0, burn_in=burn_in, seed=seed,
        )
        t0 = time.perf_counter()
        simulate_meanfield(cfg, np.random.default_rng(seed))
        elapsed = time.perf_counter() - t0
        per_k.append(
            {
                "K": k,
                "burn_in": burn_in,
                "total_steps": burn_in + n_steps - 1,
                "elapsed_seconds": elapsed,
            }
        )

    log_k = np.log(np.array([p["K"] for p in per_k], dtype=np.float64))
    log_t = np.log(np.array([p["elapsed_seconds"] for p in per_k], dtype=np.float64))
    slope, intercept = np.polyfit(log_k, log_t, 1)

    payload: dict[str, Any] = {
        "n_particles": n_particles,
        "n_steps": n_steps,
        "d": d,
        "burn_in_multiple": burn_in_multiple,
        "results": per_k,
        "loglog_slope_time_vs_k": float(slope),
        "loglog_intercept": float(intercept),
    }
    write_phase3_summary("step4_k_scaling", payload)
    return payload


def _stats(values: list[float], reference: float) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "bias": float(arr.mean() - reference),
        "variance": float(arr.var(ddof=1)),
        "std": float(arr.std(ddof=1)),
        "n": int(arr.size),
    }


def run_step3_observers(
    d_values: tuple[float, ...] = (0.25, -0.25),
    n_particles: int = 400,
    n_steps: int = 1024,
    k: int = 1024,
    n_realizations: int = 10,
    n_agents_sampled: int = 20,
    base_seed: int = 100,
) -> dict[str, Any]:
    """Step 3 -- three observers, bias and variance against the coherent ``d``.

    - **Isolated, model-agnostic**: Phase 2's corrected pipeline (classifier
      + mechanism-specific correction) on a single agent's own position
      ``x_i``, unaware of the mean field or the coupled model. Reuses Phase 2
      exactly, per ``PHASE3_PROMPT.md``'s "what this phase inherits" -- not
      reimplemented.
    - **Isolated, model-aware**: joint ``(d, J)`` Whittle fit on a single
      agent's own ``v_i`` (:func:`~collectivediff.dynamics.meanfield.whittle_dj_estimate`,
      the same tool validated in step 1 V4). "Model-aware" means it knows
      the coupled functional form; step 2 already showed this recovers the
      true, ``J``-independent ``d`` when the periodogram is clean
      (ensemble-averaged). Here each estimate comes from *one* agent's own
      periodogram, which is far noisier -- that noise is exactly the
      quantity this step measures, not something to average away.
    - **Field-aware**: single-parameter Whittle fit on ``<v>``
      (:func:`whittle_d_estimate`, reused from V2), one estimate per
      realization since there is only one mean field per simulation.

    Bias and variance are pooled across ``n_agents_sampled`` agents (sampled
    without replacement per realization, since fitting all `n_particles`
    agents individually is not needed for this comparison and is far more
    expensive) and ``n_realizations`` independent simulations, both against
    the true `d` (which the conservation law makes identical to the
    coherent-mode `d` at every `J` -- V2). The field estimate is necessarily
    one value per realization; comparing its variance to the isolated
    observers' variance (computed across (agent, realization) pairs) is
    exactly V5's comparison.

    **V5, at J = 0**: the mean field is one more fGn/ARFIMA series, not an
    ``N``-fold average of information -- ``Var(d_field)`` should be
    comparable to ``Var(d_isolated)``, not smaller by a factor of ``N``. This
    has an analytic reason, not just an empirical one: Whittle/MLE precision
    for a spectral *shape* parameter like ``d`` depends on the series length
    and the shape of the spectral density, not its overall scale -- and
    ``<v>``'s spectral density is the same ARFIMA shape as an individual
    agent's, just scaled by ``sigma^2 / N``. A uniform rescaling of the
    spectral density leaves the Fisher information for ``d`` unchanged
    (``d(log(c*S))/dd = d(log S)/dd`` for any constant ``c``), so reducing
    the field's *amplitude* by averaging should not, on its own, improve the
    precision of a *shape* estimate at all.

    Parameters
    ----------
    d_values : tuple of float
        One positive, one negative, matching V1-V4's convention.
    n_particles, n_steps, k : int
        Ensemble size, trajectory length, memory truncation.
    n_realizations : int
        Independent simulations per ``(d, J)`` point.
    n_agents_sampled : int
        Agents sampled (without replacement) per realization for the
        isolated observers.
    base_seed : int
        First realization's seed; realization ``r`` uses ``base_seed + r``.

    Returns
    -------
    dict
        Written to ``results/phase3/step3_observers.json``.
    """
    burn_in = 4 * k
    clf, mechanism_names, feature_names = _train_step3_classifier(n_steps)

    per_d: list[dict[str, Any]] = []
    for d in d_values:
        per_j: list[dict[str, Any]] = []
        for j in combined_j_sweep(d):
            agnostic: list[float] = []
            aware: list[float] = []
            field: list[float] = []
            combo: list[float] = []
            for r in range(n_realizations):
                seed = base_seed + r
                cfg = MeanFieldConfig(
                    n_particles=n_particles, n_steps=n_steps, K=k, d=d, J=j,
                    sigma=1.0, burn_in=burn_in, seed=seed,
                )
                sim = simulate_meanfield(cfg, np.random.default_rng(seed))
                d_hat_field = whittle_d_estimate(sim.mean_field)
                field.append(d_hat_field)

                sample_rng = np.random.default_rng(seed * 7919 + 1)
                agent_idx = sample_rng.choice(n_particles, size=n_agents_sampled, replace=False)
                for ai in agent_idx:
                    d_hat_agn = _isolated_model_agnostic_d(
                        sim.positions[ai], clf, mechanism_names, feature_names
                    )
                    agnostic.append(d_hat_agn)

                    freqs, per_v = _periodogram(sim.velocities[ai : ai + 1, :, 0])
                    d_hat_aw, _ = whittle_dj_estimate(freqs, per_v)
                    aware.append(d_hat_aw)
                    combo.append(0.5 * (d_hat_aw + d_hat_field))

            per_j.append(
                {
                    "J": j,
                    "isolated_model_agnostic": _stats(agnostic, d),
                    "isolated_model_aware": _stats(aware, d),
                    "field_aware": _stats(field, d),
                    "combination_agent_field": _stats(combo, d),
                }
            )
        per_d.append({"d": d, "points": per_j})

    payload: dict[str, Any] = {
        "n_particles": n_particles,
        "n_steps": n_steps,
        "K": k,
        "burn_in": burn_in,
        "n_realizations": n_realizations,
        "n_agents_sampled": n_agents_sampled,
        "results": per_d,
    }
    write_phase3_summary("step3_observers", payload)
    return payload


def run_step4_q3_correlation(
    j_values: tuple[float, ...] = (0.0, 0.1, 0.3, 0.5),
    n_particles: int = 400,
    n_steps: int = 1024,
    k: int = 1024,
    n_draws: int = 30,
    n_agents_sampled: int = 5,
    base_seed: int = 500,
) -> dict[str, Any]:
    """Step 4 -- Q3 as an error metric (``PHASE3_PROMPT.md``).

    For each ``J``, draws ``n_draws`` independent realizations with
    ``d ~ U(-0.4, 0.4)`` and scores each observer's estimate against the
    true (= coherent, by the conservation law) ``d``. The direct, numeric
    answer to "how correlated is one agent's inferred behaviour with the
    collective's," in the three versions ``PHASE3_PROMPT.md`` asks for: by
    memory alone (model-agnostic), by memory with the model known
    (model-aware), and with the field.

    **Primary metric is NRMSE (RMSE normalised by the true `d`'s standard
    deviation) and `R^2`, not the bare correlation.** A first version
    reported only correlation and found 0.93-0.99 for every observer at
    every `J` -- a diagnosis of the metric, not a result: `d` is drawn from
    a wide range (`U(-0.4, 0.4)`), and correlation is dominated by how much
    of that spread an estimator gets the *sign and rough scale* of, not by
    its precision. `NRMSE`/`R^2` isolate the actual unexplained error instead,
    and are what distinguishes the observers (`correlation` is still
    reported alongside for continuity, not as the number to read).

    Reuses step 3's observers and classifier exactly (:func:`_train_step3_classifier`,
    :func:`_isolated_model_agnostic_d`, :func:`whittle_dj_estimate`,
    :func:`whittle_d_estimate`) -- a new scoring study over the same
    machinery, not a new estimator.

    Parameters
    ----------
    j_values : tuple of float
        Coupling strengths to compare, sign-agnostic since ``d`` is drawn
        from both signs.
    n_particles, n_steps, k : int
        Ensemble size, trajectory length, memory truncation.
    n_draws : int
        Independent ``d`` realizations per ``J``.
    n_agents_sampled : int
        Agents sampled per realization for the isolated observers.
    base_seed : int
        First realization's seed.

    Returns
    -------
    dict
        Written to ``results/phase3/step4_q3_correlation.json``.
    """
    burn_in = 4 * k
    clf, mechanism_names, feature_names = _train_step3_classifier(n_steps)
    d_draws = np.random.default_rng(999).uniform(-0.4, 0.4, size=n_draws)

    def metrics(true_vals: list[float], est_vals: list[float]) -> dict[str, float]:
        true_arr = np.asarray(true_vals, dtype=np.float64)
        est_arr = np.asarray(est_vals, dtype=np.float64)
        residual = est_arr - true_arr
        ss_res = float(np.sum(residual**2))
        ss_tot = float(np.sum((true_arr - true_arr.mean()) ** 2))
        rmse = float(np.sqrt(np.mean(residual**2)))
        std_true = float(true_arr.std(ddof=1))
        return {
            "correlation": float(np.corrcoef(true_arr, est_arr)[0, 1]),
            "rmse": rmse,
            "nrmse": rmse / std_true,
            "r_squared": 1.0 - ss_res / ss_tot,
        }

    per_j: list[dict[str, Any]] = []
    for j in j_values:
        agn_true: list[float] = []
        agn_est: list[float] = []
        aw_true: list[float] = []
        aw_est: list[float] = []
        field_true: list[float] = []
        field_est: list[float] = []
        combo_true: list[float] = []
        combo_est: list[float] = []

        for i, d in enumerate(d_draws):
            seed = base_seed + i
            cfg = MeanFieldConfig(
                n_particles=n_particles, n_steps=n_steps, K=k, d=float(d), J=j,
                sigma=1.0, burn_in=burn_in, seed=seed,
            )
            sim = simulate_meanfield(cfg, np.random.default_rng(seed))
            d_hat_field = whittle_d_estimate(sim.mean_field)
            field_true.append(float(d))
            field_est.append(d_hat_field)

            sample_rng = np.random.default_rng(seed * 7919 + 1)
            agent_idx = sample_rng.choice(n_particles, size=n_agents_sampled, replace=False)
            for ai in agent_idx:
                d_agn = _isolated_model_agnostic_d(
                    sim.positions[ai], clf, mechanism_names, feature_names
                )
                agn_true.append(float(d))
                agn_est.append(d_agn)

                freqs, per_v = _periodogram(sim.velocities[ai : ai + 1, :, 0])
                d_aw, _ = whittle_dj_estimate(freqs, per_v)
                aw_true.append(float(d))
                aw_est.append(d_aw)
                combo_true.append(float(d))
                combo_est.append(0.5 * (d_aw + d_hat_field))

        per_j.append(
            {
                "J": j,
                "isolated_model_agnostic": metrics(agn_true, agn_est),
                "isolated_model_aware": metrics(aw_true, aw_est),
                "field_aware": metrics(field_true, field_est),
                "combination_agent_field": metrics(combo_true, combo_est),
                "n_draws": n_draws,
            }
        )

    payload: dict[str, Any] = {
        "n_particles": n_particles,
        "n_steps": n_steps,
        "K": k,
        "burn_in": burn_in,
        "n_draws": n_draws,
        "n_agents_sampled": n_agents_sampled,
        "d_range": [-0.4, 0.4],
        "results": per_j,
    }
    write_phase3_summary("step4_q3_correlation", payload)
    return payload


def generate_step5_animation_data(
    d: float = 0.25,
    j_weak: float = 0.0,
    j_strong: float = 0.3,
    n_particles: int = 60,
    n_steps: int = 400,
    k: int = 512,
    seed: int = 700,
) -> dict[str, NDArray[np.float64]]:
    """2-D positions and center-of-mass trajectories for step 5's animation.

    ``PROJECT.md`` section 7: 2-D is for the animation only, built from two
    *independent* 1-D mean-field runs, one per component -- not a native 2-D
    generator, since the coupled model is defined and validated in 1-D. Not
    written to JSON (the arrays are the deliverable, fed straight to
    :func:`~collectivediff.viz.phase3.animate_meanfield_comparison`).

    Parameters
    ----------
    d : float
        Fractional differencing parameter.
    j_weak, j_strong : float
        Coupling strengths for the two side-by-side panels.
    n_particles, n_steps, k : int
        Kept small -- this is for a legible animation, not a measurement.
    seed : int
        Base seed; each of the four component simulations gets a distinct
        offset from it.

    Returns
    -------
    dict
        ``positions_weak``, ``positions_strong`` (each ``(N, T, 2)``) and
        ``com_weak``, ``com_strong`` (each ``(T, 2)``).
    """
    burn_in = 4 * k

    def sim_2d(j: float, seed_offset: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        positions_components = []
        com_components = []
        for component in range(2):
            s = seed + seed_offset + component
            cfg = MeanFieldConfig(
                n_particles=n_particles, n_steps=n_steps, K=k, d=d, J=j,
                sigma=1.0, burn_in=burn_in, seed=s,
            )
            sim = simulate_meanfield(cfg, np.random.default_rng(s))
            positions_components.append(sim.positions[:, :, 0])
            com_components.append(sim.center_of_mass())
        positions = np.stack(positions_components, axis=-1)
        com = np.stack(com_components, axis=-1)
        return positions, com

    positions_weak, com_weak = sim_2d(j_weak, 0)
    positions_strong, com_strong = sim_2d(j_strong, 100)
    return {
        "positions_weak": positions_weak,
        "positions_strong": positions_strong,
        "com_weak": com_weak,
        "com_strong": com_strong,
    }


def run_step3_v5_check(
    d_values: tuple[float, ...] = (0.25, -0.25),
    n_particles: int = 400,
    n_steps: int = 1024,
    k: int = 1024,
    n_realizations: int = 15,
    n_agents_sampled: int = 20,
    base_seed: int = 900,
) -> dict[str, Any]:
    """V5 -- at ``J = 0``, is the field's precision really ``N``-fold better?

    A targeted, matched-model version of the comparison step 3's main sweep
    cannot cleanly make: that sweep's "isolated, model-aware" observer fits
    *two* parameters ``(d, J)`` per agent (needed once ``J > 0`` is on the
    table), while the field observer fits *one* (``<v>`` has no ``J`` to
    fit). Comparing their variances conflates "does averaging over `N`
    agents help" with "is a two-parameter fit noisier than a one-parameter
    fit on the same amount of data" -- the second is true regardless of `N`
    and would swamp the first.

    At ``J = 0`` there is no ``J`` to fit for *anyone*, so both sides can use
    the same one-parameter model (:func:`~collectivediff.dynamics.meanfield.whittle_d_only_estimate`,
    correctly specified here since ``J`` really is 0): one isolated agent's
    own ``v_i`` versus the ``N``-agent mean field ``<v>``. ``PROJECT.md``
    section 7 predicts these should have *comparable* variance, not the
    field being ``N`` times more precise, because Whittle/MLE precision for
    a spectral shape parameter depends on series length and shape, not
    amplitude, and ``<v>`` has the same ARFIMA shape as an individual agent
    -- only scaled by ``sigma^2 / N`` -- so the ``N``-fold amplitude
    reduction does not, on its own, buy `N`-fold better precision on `d`
    (see :func:`run_step3_observers`'s docstring for the one-line algebra).

    Returns
    -------
    dict
        Written to ``results/phase3/step3_v5_check.json``.
    """
    burn_in = 4 * k
    per_d: list[dict[str, Any]] = []
    for d in d_values:
        isolated: list[float] = []
        field: list[float] = []
        for r in range(n_realizations):
            seed = base_seed + r
            cfg = MeanFieldConfig(
                n_particles=n_particles, n_steps=n_steps, K=k, d=d, J=0.0,
                sigma=1.0, burn_in=burn_in, seed=seed,
            )
            sim = simulate_meanfield(cfg, np.random.default_rng(seed))
            field.append(whittle_d_estimate(sim.mean_field))

            sample_rng = np.random.default_rng(seed * 7919 + 1)
            agent_idx = sample_rng.choice(n_particles, size=n_agents_sampled, replace=False)
            for ai in agent_idx:
                freqs, per_v = _periodogram(sim.velocities[ai : ai + 1, :, 0])
                isolated.append(whittle_d_only_estimate(freqs, per_v))

        isolated_stats = _stats(isolated, d)
        field_stats = _stats(field, d)
        # Whittle Fisher information for ARFIMA(0,d,0) is pi^2/6 per
        # observation, independent of d (verified numerically against the
        # closed-form integral in NOTES.md); n_vel = n_steps - 1 observations.
        crlb = 6.0 / ((n_steps - 1) * np.pi**2)
        per_d.append(
            {
                "d": d,
                "crlb": crlb,
                "isolated_single_param": isolated_stats,
                "isolated_efficiency": crlb / isolated_stats["variance"],
                "field": field_stats,
                "field_efficiency": crlb / field_stats["variance"],
                "variance_ratio_isolated_over_field": isolated_stats["variance"] / field_stats["variance"],
            }
        )

    payload: dict[str, Any] = {
        "n_particles": n_particles,
        "n_steps": n_steps,
        "K": k,
        "burn_in": burn_in,
        "n_realizations": n_realizations,
        "n_agents_sampled": n_agents_sampled,
        "results": per_d,
    }
    write_phase3_summary("step3_v5_check", payload)
    return payload


def run_step4_n_scaling(
    d: float = 0.25,
    j_values: tuple[float, ...] = (0.0, 0.3),
    n_values: tuple[int, ...] = (50, 100, 400, 1600),
    n_steps: int = 1024,
    k: int = 1024,
    n_realizations: int = 10,
    base_seed: int = 1100,
) -> dict[str, Any]:
    """How estimator variance scales with ``N`` -- required before phase 4.

    Phase 4 is entirely a question of how error scales with the number of
    observed agents; nothing in phase 3's main sweep (run at a single
    ``N = 400``) measured that scaling directly. Three quantities, each
    across ``n_realizations`` independent simulations at every ``(J, N)``:

    - **field-aware** (:func:`whittle_d_estimate` on ``<v>``): predicted
      flat in ``N``. ``<v>``'s spectral *shape* doesn't depend on how many
      agents formed it, only its amplitude (``sigma^2 / N``) does, and
      Whittle precision for a shape parameter is scale-invariant (V5).
    - **isolated single-agent** (:func:`~collectivediff.dynamics.meanfield.whittle_dj_estimate`
      on one agent's own ``v_i``): predicted flat in ``N`` too, as a control
      -- one agent's own precision has no reason to depend on how many other
      agents exist, only on its own series length.
    - **typical-agent windowed EA-MSD exponent** (``fit_powerlaw`` over
      ``[2, K/10]`` on the ``N``-agent ensemble-averaged EA-MSD, step 2's
      quantity): predicted variance ``~ 1/N`` (error ``~ N^{-1/2}``), the
      ordinary ensemble-averaging result -- this is the one quantity here
      that actually is an average over ``N`` independent-ish agents within
      a single realization, unlike the other two.

    If field-aware variance instead falls with ``N``, phase 4's premise (that
    an additional observed agent's value comes from something other than
    shrinking the field's own uncertainty) is wrong and needs revisiting
    before phase 4 starts.

    Parameters
    ----------
    d : float
        Fractional differencing parameter, one representative ``d > 0``.
    j_values : tuple of float
        Coupling strengths, including the ``J = 0`` control.
    n_values : tuple of int
        Ensemble sizes to sweep.
    n_steps, k : int
        Trajectory length, memory truncation.
    n_realizations : int
        Independent simulations per ``(J, N)`` point.
    base_seed : int
        First realization's seed.

    Returns
    -------
    dict
        Written to ``results/phase3/step4_n_scaling.json``.
    """
    burn_in = 4 * k
    max_lag = max(5, k // 10)
    t = np.arange(n_steps, dtype=np.float64)

    per_j: list[dict[str, Any]] = []
    for j in j_values:
        per_n: list[dict[str, Any]] = []
        for n in n_values:
            field_estimates: list[float] = []
            isolated_estimates: list[float] = []
            msd_exponents: list[float] = []
            for r in range(n_realizations):
                seed = base_seed + r
                cfg = MeanFieldConfig(
                    n_particles=n, n_steps=n_steps, K=k, d=d, J=j,
                    sigma=1.0, burn_in=burn_in, seed=seed,
                )
                sim = simulate_meanfield(cfg, np.random.default_rng(seed))
                field_estimates.append(whittle_d_estimate(sim.mean_field))

                freqs, per_v = _periodogram(sim.velocities[0:1, :, 0])
                d_hat_iso, _ = whittle_dj_estimate(freqs, per_v)
                isolated_estimates.append(d_hat_iso)

                ea = ea_msd(sim.positions)
                fit = fit_powerlaw(t, ea, 2.0, float(max_lag), fit=None)
                msd_exponents.append(fit.exponent)

            field_arr = np.array(field_estimates)
            iso_arr = np.array(isolated_estimates)
            msd_arr = np.array(msd_exponents)
            per_n.append(
                {
                    "N": n,
                    "field_variance": float(field_arr.var(ddof=1)),
                    "isolated_variance": float(iso_arr.var(ddof=1)),
                    "msd_exponent_variance": float(msd_arr.var(ddof=1)),
                    "msd_exponent_mean": float(msd_arr.mean()),
                }
            )
        per_j.append({"J": j, "points": per_n})

    payload: dict[str, Any] = {
        "d": d,
        "n_steps": n_steps,
        "K": k,
        "burn_in": burn_in,
        "n_realizations": n_realizations,
        "n_values": list(n_values),
        "results": per_j,
    }
    write_phase3_summary("step4_n_scaling", payload)
    return payload
