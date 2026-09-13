"""Composed analyses that notebooks call but must not contain.

``PROJECT.md`` section 2 keeps all logic in the package: notebooks import, call
and plot. The estimator-variance study of ``PHASE1_PROMPT.md`` step 5 is several
dozen lines of generation and fitting, so it lives here rather than in a cell.

Nothing in this module is a new estimator. It composes
:mod:`collectivediff.generators` and :mod:`collectivediff.estimators` into the
particular sweeps the phase-1 notebooks show.
"""

from __future__ import annotations

import json
import time
import tracemalloc
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from .config import BrownianConfig, CTRWConfig, FitConfig, GeneratorConfig
from .estimators import (
    ea_msd,
    ea_ta_msd,
    ergodicity_breaking_curve,
    fit_powerlaw,
    increment_acf,
    log_spaced_lags,
    moment_spectrum,
    squared_displacement,
    ta_msd,
)
from .estimators.ergodicity import brownian_eb, ctrw_eb_plateau
from .estimators.moments import MomentSpectrum
from .generators import generate

__all__ = [
    "RESULTS_ROOT",
    "SingleTrajectoryStudy",
    "benchmark_table",
    "ensemble_msd_curve",
    "ergodicity_checks",
    "exponent_spread_study",
    "iqr_scaling_exponent",
    "lag_window_sensitivity",
    "mechanism_diagnostics",
    "single_trajectory_exponents",
    "single_trajectory_report",
    "write_summary",
]

#: Where studies write their JSON summaries and media.
RESULTS_ROOT = Path(__file__).resolve().parents[2] / "results"


def write_summary(name: str, payload: Mapping[str, Any], root: Path | None = None) -> Path:
    """Write a study's numbers to ``results/<name>.json``.

    Figures are for reading; these files are for *quoting*. A progress report
    should be assembled from them rather than from notebook outputs, which are
    dominated by base64 image payloads and cost far more to read than the dozen
    numbers they contain.

    Parameters
    ----------
    name : str
        File stem, e.g. ``"exponent_spread"``.
    payload : mapping
        JSON-serialisable summary.
    root : Path, optional
        Defaults to :data:`RESULTS_ROOT`.

    Returns
    -------
    pathlib.Path
        The file written.
    """
    root = RESULTS_ROOT if root is None else Path(root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _with_length(cfg: GeneratorConfig, n_steps: int) -> GeneratorConfig:
    """Copy a config with a different trajectory length."""
    from dataclasses import replace

    return replace(cfg, n_steps=n_steps)


def ensemble_msd_curve(
    cfg: GeneratorConfig,
    t_min: float | None = None,
    fit: FitConfig | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], Any]:
    """Generate an ensemble and return ``(t, EA-MSD, PowerLawFit)``.

    Parameters
    ----------
    cfg : GeneratorConfig
        Any mechanism with a finite second moment.
    t_min : float, optional
        Lower edge of the fit window. Defaults to ``T/10`` for CTRW and the Levy
        walk, whose asymptotics need the early decade excluded, and to 1 for the
        mechanisms whose law is exact at every ``t``.
    fit : FitConfig, optional
        Bootstrap settings.

    Returns
    -------
    t : ndarray of float64
    msd : ndarray of float64
    result : PowerLawFit
        Carrying the bootstrap error, not the least-squares one.
    """
    trajectories = generate(cfg, np.random.default_rng(cfg.seed))
    t = np.arange(cfg.n_steps, dtype=np.float64)
    per_particle = squared_displacement(trajectories, cfg)
    if t_min is None:
        t_min = cfg.duration / 10.0 if type(cfg).name in ("ctrw", "levy_walk") else 1.0
    result = fit_powerlaw(t, per_particle, t_min, cfg.duration, fit=fit)
    return t, np.mean(per_particle, axis=0), result


def single_trajectory_exponents(
    cfg: GeneratorConfig,
    lengths: Sequence[int],
    n_trajectories: int = 200,
    fit: FitConfig | None = None,
) -> NDArray[np.float64]:
    """Fit a TA-MSD exponent to each trajectory separately, at several lengths.

    This is the observable an experimenter has when only one agent is visible:
    the time-averaged MSD of that one trajectory, fitted over
    ``[lag_min, T/10]``. Repeating it over many trajectories gives the
    *distribution* of what a single-agent observer would conclude, which is the
    quantity Q1 asks about.

    One ensemble is generated at the longest length and truncated for the
    shorter ones. That is exact rather than a shortcut: the first ``T`` steps of
    a trajectory of length ``T_max`` *are* a trajectory of length ``T``, for
    every mechanism here, and it keeps the comparison across lengths free of
    ensemble-to-ensemble noise.

    Parameters
    ----------
    cfg : GeneratorConfig
        Mechanism. Its ``n_steps`` is overridden by ``max(lengths)`` and its
        ``n_particles`` by ``n_trajectories``.
    lengths : sequence of int
        Trajectory lengths to study.
    n_trajectories : int
        How many independent trajectories per length.
    fit : FitConfig, optional
        Supplies the lag window.

    Returns
    -------
    ndarray of float64
        Shape ``(len(lengths), n_trajectories)``. Entries are ``nan`` where a
        trajectory could not be fitted.

    Notes
    -----
    **The ``nan`` entries are not noise and must be reported, not absorbed.**
    A CTRW trajectory fails to fit precisely when it spent the whole measurement
    inside a single trapping event: zero displacement at every lag, so no
    log-log fit exists. Those are exactly the trajectories carrying the
    ergodicity-breaking signal, and quietly dropping them biases the study
    towards the mobile subpopulation. Measured censoring for ``a = 0.5``:
    7.5 per cent at ``T = 128``, 3.5 at 512, and 1.5 at 2048 and 8192; zero for
    every ergodic mechanism at every length. :func:`exponent_spread_study`
    therefore returns the fraction alongside the numbers so a reader can judge
    whether a median over the survivors means anything.

    **On the fitted value itself.** The lag window comes from ``FitConfig``,
    which defaults to the project's ``[1, T/10]``. For a CTRW that window
    reaches into a regime where a single trajectory's TA-MSD has stopped growing
    linearly, because with a median of only 46 jumps in ``T = 8192`` the widest
    windows contain the same handful of events. The fitted slope is
    correspondingly sensitive to the window -- median 0.965 over lags 1 to 32,
    0.890 over the default 1 to 819, 0.787 over 100 to 819 -- against a
    theoretical 1.0. The project convention is kept rather than tuned, but the
    sensitivity is real and phase 2 must not read a single-trajectory CTRW
    exponent without stating its window.
    """
    from dataclasses import replace

    fit = FitConfig() if fit is None else fit
    lengths = sorted(int(length) for length in lengths)
    base = replace(cfg, n_steps=lengths[-1], n_particles=n_trajectories)
    trajectories = generate(base, np.random.default_rng(base.seed))

    out = np.full((len(lengths), n_trajectories), np.nan, dtype=np.float64)
    for row, length in enumerate(lengths):
        lags = log_spaced_lags(length, fit)
        per_particle = ta_msd(trajectories[:, :length, :], lags, cfg)
        for column, curve in enumerate(per_particle):
            if np.all(curve > 0.0):
                out[row, column] = fit_powerlaw(
                    lags.astype(np.float64), curve
                ).exponent
    return out


class SingleTrajectoryStudy(dict):
    """Result of :func:`exponent_spread_study`: label -> (lengths, estimates)."""


def iqr_scaling_exponent(
    lengths: NDArray[np.float64],
    estimates: NDArray[np.float64],
) -> float:
    """Fit ``IQR ~ T^(-p)`` to the spread of an estimator against trajectory length.

    Reports as one number what a reader would otherwise have to eyeball off a
    panel. A self-averaging estimator built on ``T`` independent-ish windows has
    ``p = 1/2``; in practice the log-log fit over four lengths returns somewhat
    less, because the TA-MSD windows overlap.

    What the number is *for* is the comparison between mechanisms, not its
    absolute value: Brownian motion, fBm and scaled Brownian motion all land
    near ``p = 0.27``, and a CTRW at ``p = 0.11`` is visibly a different regime.

    Parameters
    ----------
    lengths : ndarray of float64
        Trajectory lengths.
    estimates : ndarray of float64
        Shape ``(len(lengths), n_trajectories)``; ``nan`` entries are ignored.

    Returns
    -------
    float
        ``p``, positive when the spread shrinks.
    """
    lengths = np.asarray(lengths, dtype=np.float64)
    iqr = np.nanpercentile(estimates, 75, axis=1) - np.nanpercentile(
        estimates, 25, axis=1
    )
    usable = np.isfinite(iqr) & (iqr > 0.0) & (lengths > 0.0)
    if usable.sum() < 2:
        return float("nan")
    slope = np.polyfit(np.log(lengths[usable]), np.log(iqr[usable]), 1)[0]
    return float(-slope)


def exponent_spread_study(
    configs: Mapping[str, GeneratorConfig],
    lengths: Sequence[int] = (128, 512, 2048, 8192),
    n_trajectories: int = 200,
    fit: FitConfig | None = None,
) -> tuple[SingleTrajectoryStudy, dict[str, float], dict[str, dict[str, Any]]]:
    """Spread *and bias* of the single-trajectory exponent against ``T``.

    The closing study of phase 1. Its result is a **bias** result, not a
    variance result, and the summary returned here is built to say so:

    * every mechanism's spread shrinks, including the CTRW's -- just three times
      more slowly (``p`` around 0.11 against 0.27);
    * for CTRW and scaled Brownian motion the estimator converges on
      ``alpha ~ 1`` while the truth is 0.5 and 0.6, so the observer watching one
      agent is not facing noise that patience would cure but a confident wrong
      answer.

    Note that this measures scatter in the fitted **slope** of a single
    trajectory's TA-MSD. That is a different quantity from the
    ergodicity-breaking parameter, which measures scatter in the TA-MSD
    **amplitude** at one fixed lag. The two are related in origin and must not
    be quoted for one another.

    Parameters
    ----------
    configs : mapping
        ``label -> config``.
    lengths : sequence of int
        Trajectory lengths.
    n_trajectories : int
        Trajectories per mechanism.
    fit : FitConfig, optional

    Returns
    -------
    study : SingleTrajectoryStudy
        ``label -> (lengths, estimates)``.
    truths : dict
        ``label -> true ensemble exponent``.
    summary : dict
        ``label -> {"median", "iqr", "bias", "censored", "p"}``, all as plain
        lists or floats so the whole thing serialises straight to JSON.
        ``censored`` is the fraction of trajectories whose fit failed at each
        length; see :func:`single_trajectory_exponents` for why it is reported
        rather than absorbed into a ``nanmedian``.
    """
    lengths = sorted(int(length) for length in lengths)
    grid = np.asarray(lengths, dtype=np.float64)
    study = SingleTrajectoryStudy()
    truths: dict[str, float] = {}
    summary: dict[str, dict[str, Any]] = {}

    for label, cfg in configs.items():
        estimates = single_trajectory_exponents(cfg, lengths, n_trajectories, fit)
        study[label] = (grid, estimates)
        truths[label] = cfg.alpha_analytic

        median = np.nanmedian(estimates, axis=1)
        iqr = np.nanpercentile(estimates, 75, axis=1) - np.nanpercentile(
            estimates, 25, axis=1
        )
        summary[label] = {
            "lengths": lengths,
            "truth": float(cfg.alpha_analytic),
            "median": [float(v) for v in median],
            "iqr": [float(v) for v in iqr],
            "bias": [float(v - cfg.alpha_analytic) for v in median],
            "censored": [float(np.mean(~np.isfinite(row))) for row in estimates],
            "p": iqr_scaling_exponent(grid, estimates),
        }
    return study, truths, summary


def single_trajectory_report(
    configs: Mapping[str, GeneratorConfig],
    n_steps: int = 4096,
    index: int = 0,
    fit: FitConfig | None = None,
) -> dict[str, dict[str, Any]]:
    """What one observed agent, per mechanism, would report about itself.

    The narrative of ``01_single_particle.ipynb``: take a single trajectory from
    each mechanism, fit its own time-averaged MSD, and put the answer beside the
    truth. The mechanisms are chosen in the notebook to span sub-, normal and
    superdiffusion, and the point is how little the single-trajectory answers
    separate them -- and, for the CTRW, how confidently wrong one of them is.

    Parameters
    ----------
    configs : mapping
        ``label -> config``. ``n_particles`` is overridden to ``index + 1``.
    n_steps : int
        Trajectory length.
    index : int
        Which trajectory of the generated batch to report on.
    fit : FitConfig, optional
        Supplies the lag window.

    Returns
    -------
    dict
        ``label -> {"trajectory", "lags", "ta_msd", "exponent", "truth"}``.
        ``exponent`` is ``nan`` when the trajectory never moved, which is a
        possible outcome for a CTRW and is reported rather than hidden.
    """
    from dataclasses import replace

    fit = FitConfig() if fit is None else fit
    report: dict[str, dict[str, Any]] = {}
    for label, cfg in configs.items():
        base = replace(cfg, n_steps=n_steps, n_particles=index + 1)
        trajectory = generate(base, np.random.default_rng(base.seed))[index : index + 1]
        lags = log_spaced_lags(n_steps, fit)
        curve = ta_msd(trajectory, lags, cfg)[0]
        exponent = (
            fit_powerlaw(lags.astype(np.float64), curve).exponent
            if np.all(curve > 0.0)
            else np.nan
        )
        report[label] = {
            "trajectory": trajectory,
            "lags": lags,
            "ta_msd": curve,
            "exponent": exponent,
            "truth": cfg.alpha_analytic,
        }
    return report


def mechanism_diagnostics(
    cfg: GeneratorConfig,
    q_values: NDArray[np.float64] | None = None,
    max_acf_lag: int = 12,
    fit: FitConfig | None = None,
) -> dict[str, Any]:
    """One pass over a generated ensemble producing every phase-1 diagnostic.

    Generating once and deriving everything from that array keeps a notebook
    cell to a single call and, more importantly, guarantees the panels of a
    figure describe the *same* ensemble rather than several independently seeded
    ones.

    Parameters
    ----------
    cfg : GeneratorConfig
        Mechanism with a finite second moment.
    q_values : ndarray, optional
        Moment orders; defaults to 16 points on ``[0.2, 4]`` as in
        ``PROJECT.md`` section 5.
    max_acf_lag : int
        Largest increment-ACF lag.
    fit : FitConfig, optional

    Returns
    -------
    dict
        Keys ``trajectories``, ``t``, ``msd``, ``msd_fit``, ``lags``,
        ``ea_ta_msd``, ``acf``, ``spectrum``.
    """
    if q_values is None:
        q_values = np.linspace(0.2, 4.0, 16)
    trajectories = generate(cfg, np.random.default_rng(cfg.seed))
    t = np.arange(cfg.n_steps, dtype=np.float64)
    per_particle = squared_displacement(trajectories, cfg)

    is_slow = type(cfg).name in ("ctrw", "levy_walk")
    t_min = cfg.duration / 10.0 if is_slow else 1.0
    lags = log_spaced_lags(cfg.n_steps, fit)

    return {
        "trajectories": trajectories,
        "t": t,
        "msd": np.mean(per_particle, axis=0),
        "msd_fit": fit_powerlaw(t, per_particle, t_min, cfg.duration, fit=fit),
        "lags": lags,
        "ea_ta_msd": ea_ta_msd(trajectories, lags, cfg),
        "acf": increment_acf(trajectories, max_acf_lag),
        "spectrum": moment_spectrum(
            trajectories, q_values, t_min, cfg.duration, fit=fit
        ),
    }


def ergodicity_checks(
    alpha_wait: float = 0.5,
    n_particles: int = 2000,
    n_steps: int = 4096,
    lags: Sequence[int] = (10, 50, 100),
    seed: int = 0,
) -> dict[str, Any]:
    """The two numeric ergodicity checks, measured against their closed forms.

    Both are needed, and the Brownian one is not the easy warm-up it looks like.

    * **CTRW plateau.** ``EB(Delta)`` converges to
      ``EB_inf = 2 Gamma(1+a)^2 / Gamma(1+2a) - 1``, which at ``a = 1/2`` is
      exactly ``pi/2 - 1 = 0.5708``. This is the headline result, but it is a
      weak test of the *implementation*: a plateau multiplied by a wrong
      constant is still a plateau, and would still look like one.
    * **Brownian positive control.** ``EB(Delta) ~= 4 Delta / (3T)``, a specific
      slope with a specific coefficient. A wrong factor inside
      :func:`~collectivediff.estimators.ergodicity.eb_parameter` cannot survive
      this one, which is why it is run alongside rather than instead.

    Parameters
    ----------
    alpha_wait : float
        CTRW waiting exponent.
    n_particles, n_steps : int
        Ensemble size and trajectory length.
    lags : sequence of int
        Lags at which to compare. The CTRW plateau wants small ``Delta/T``; the
        Brownian formula holds for ``Delta << T``.
    seed : int
        Seed for both ensembles.

    Returns
    -------
    dict
        JSON-serialisable, with ``ctrw`` and ``brownian`` entries each carrying
        measured values, predictions and relative deviations.
    """
    lags_array = np.asarray(list(lags), dtype=np.intp)

    ctrw_cfg = CTRWConfig(
        n_particles=n_particles, n_steps=n_steps, alpha_wait=alpha_wait, seed=seed
    )
    ctrw_measured = ergodicity_breaking_curve(
        generate(ctrw_cfg, np.random.default_rng(seed)), lags_array, ctrw_cfg
    )
    plateau = ctrw_eb_plateau(alpha_wait)

    brownian_cfg = BrownianConfig(n_particles=n_particles, n_steps=n_steps, seed=seed)
    brownian_measured = ergodicity_breaking_curve(
        generate(brownian_cfg, np.random.default_rng(seed)), lags_array, brownian_cfg
    )
    brownian_predicted = brownian_eb(lags_array.astype(np.float64), n_steps)

    return {
        "n_particles": n_particles,
        "n_steps": n_steps,
        "lags": [int(lag) for lag in lags_array],
        "ctrw": {
            "alpha_wait": float(alpha_wait),
            "measured": [float(v) for v in ctrw_measured],
            "plateau": float(plateau),
            "relative_deviation": [float(v / plateau - 1.0) for v in ctrw_measured],
        },
        "brownian": {
            "measured": [float(v) for v in brownian_measured],
            "predicted": [float(v) for v in brownian_predicted],
            "relative_deviation": [
                float(m / p - 1.0) for m, p in zip(brownian_measured, brownian_predicted)
            ],
        },
    }


def lag_window_sensitivity(
    cfg: GeneratorConfig,
    windows: Sequence[tuple[int, int]],
    n_trajectories: int = 200,
    n_steps: int = 8192,
) -> dict[str, Any]:
    """How much the single-trajectory exponent depends on the lag window chosen.

    Written for the CTRW, where the answer is "a lot". Theory says a single
    trajectory's TA-MSD is linear in the lag, but with a median of 46 jumps in
    ``T = 8192`` the widest windows of the project's default ``[1, T/10]``
    contain the same few events and the curve has stopped growing. The fitted
    slope slides from about 0.96 near the short-lag end to 0.79 at the long-lag
    end, against a theoretical 1.0.

    Phase 2's confusion matrix is built on these estimates, so the window is a
    parameter of the result and has to be reported with it.

    Parameters
    ----------
    cfg : GeneratorConfig
        Mechanism.
    windows : sequence of (int, int)
        Inclusive lag windows to compare.
    n_trajectories : int
        Trajectories per window.
    n_steps : int
        Trajectory length.

    Returns
    -------
    dict
        JSON-serialisable, one entry per window.
    """
    from dataclasses import replace

    base = replace(cfg, n_steps=n_steps, n_particles=n_trajectories)
    trajectories = generate(base, np.random.default_rng(base.seed))
    lags = log_spaced_lags(n_steps, FitConfig(lag_max_fraction=1.0, n_lags=48))
    curves = ta_msd(trajectories, lags, cfg)

    rows = []
    for low, high in windows:
        mask = (lags >= low) & (lags <= high)
        if mask.sum() < 3:
            continue
        fitted = [
            fit_powerlaw(lags[mask].astype(np.float64), curve[mask]).exponent
            for curve in curves
            if np.all(curve[mask] > 0.0)
        ]
        rows.append(
            {
                "window": [int(low), int(high)],
                "n_fitted": len(fitted),
                "median": float(np.median(fitted)) if fitted else float("nan"),
                "mean": float(np.mean(fitted)) if fitted else float("nan"),
            }
        )
    return {
        "mechanism": type(cfg).name,
        "n_steps": n_steps,
        "n_trajectories": n_trajectories,
        "windows": rows,
    }


def benchmark_table(n_particles: int = 4000, n_steps: int = 4096) -> dict[str, Any]:
    """Wall time and peak allocation for each generator and estimator.

    Peak memory is dominated by the *generators*, not the estimators: the Levy
    walk and fBm allocate several times their output, in event arrays and
    complex FFT buffers respectively. That is the number a phase-2 sweep needs
    in order to choose between raising ``n_particles`` and chunking.

    Parameters
    ----------
    n_particles, n_steps : int
        Size of the benchmark ensemble.

    Returns
    -------
    dict
        JSON-serialisable ``{"config": ..., "rows": [{"operation", "ms",
        "peak_mib"}, ...]}``.
    """
    from .config import DDMConfig, FBMConfig, LevyFlightConfig, LevyWalkConfig, SBMConfig

    rows: list[dict[str, Any]] = []

    def bench(label: str, call: Callable[[], Any]) -> Any:
        tracemalloc.start()
        start = time.perf_counter()
        value = call()
        elapsed = time.perf_counter() - start
        peak = tracemalloc.get_traced_memory()[1] / 2**20
        tracemalloc.stop()
        rows.append(
            {"operation": label, "ms": round(elapsed * 1e3), "peak_mib": round(peak)}
        )
        return value

    common = dict(n_particles=n_particles, n_steps=n_steps, seed=0)
    generators = {
        "generate brownian": BrownianConfig(**common),
        "generate fbm": FBMConfig(hurst=0.7, **common),
        "generate sbm": SBMConfig(alpha=0.6, **common),
        "generate ctrw": CTRWConfig(alpha_wait=0.5, **common),
        "generate levy_walk": LevyWalkConfig(gamma=1.5, **common),
        "generate levy_flight": LevyFlightConfig(stability=1.5, **common),
        "generate ddm": DDMConfig(tau=30.0, **common),
    }
    for label, cfg in generators.items():
        bench(label, lambda c=cfg: generate(c, np.random.default_rng(0)))

    cfg = BrownianConfig(**common)
    x = generate(cfg, np.random.default_rng(0))
    lags = log_spaced_lags(n_steps)
    t = np.arange(n_steps, dtype=np.float64)
    per_particle = bench("squared_displacement", lambda: squared_displacement(x, cfg))
    bench("ea_msd", lambda: ea_msd(x, cfg))
    bench("ta_msd (32 lags)", lambda: ta_msd(x, lags, cfg))
    bench("eb curve (32 lags)", lambda: ergodicity_breaking_curve(x, lags, cfg))
    bench("increment_acf (32)", lambda: increment_acf(x, 32))
    bench(
        "fit_powerlaw (no bootstrap)",
        lambda: fit_powerlaw(t, np.mean(per_particle, axis=0), 1.0, n_steps - 1.0),
    )
    bench(
        "fit_powerlaw (200 bootstrap)",
        lambda: fit_powerlaw(t, per_particle, 1.0, n_steps - 1.0),
    )
    bench(
        "moment_spectrum (16 q)",
        lambda: moment_spectrum(
            x, np.linspace(0.2, 4.0, 16), 1.0, n_steps - 1.0, fit=FitConfig(n_bootstrap=0)
        ),
    )
    return {
        "n_particles": n_particles,
        "n_steps": n_steps,
        "float64_array_mib": round(n_particles * n_steps * 8 / 2**20),
        "rows": rows,
    }
