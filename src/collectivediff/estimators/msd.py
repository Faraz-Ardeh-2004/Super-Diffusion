"""Mean squared displacement in its three forms, and power-law fitting.

Definitions are those of ``PROJECT.md`` section 5:

* **EA-MSD** -- ensemble average of the squared displacement from the initial
  position, a function of absolute time ``t``.
* **TA-MSD** -- time average along one trajectory, a function of lag ``D``.
* **EA-TA-MSD** -- the ensemble average of the above.

For an ergodic process the last two coincide as ``T / D -> infinity``. The
distance between them is the whole subject of Q1, so they are separate
functions and never silently substituted for one another.

Two conventions carried by this module
--------------------------------------

**Dtype.** ``PROJECT.md`` section 2 stores cached arrays as ``float32`` and
performs estimator accumulations in ``float64``. Every reduction here therefore
passes ``dtype=np.float64`` explicitly rather than inheriting the input dtype:
single precision is ample for holding a coordinate, but summing millions of
squares in it is not.

**Fit windows.** ``fit_powerlaw`` has no default window, because the two cases
are different claims. For *TA-MSD versus lag* the window ``[lag_min, T/10]``
exists because the number of contributing windows collapses at large lag. For
*EA-MSD versus time* every particle contributes at every ``t``, so the window
instead asserts where the asymptotics hold -- for CTRW and the Levy walk it must
exclude the early-time regime where finite-time corrections dominate.

**Uncertainty.** Bootstrap over particles, never fit residuals. See
:func:`fit_powerlaw`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..config import FitConfig

__all__ = [
    "DivergentMomentError",
    "PowerLawFit",
    "check_layout",
    "ea_msd",
    "ea_ta_msd",
    "fit_powerlaw",
    "fractional_moment",
    "log_spaced_lags",
    "squared_displacement",
    "ta_msd",
]


class DivergentMomentError(ValueError):
    """Raised when a moment that does not exist is requested.

    The motivating case is the Levy flight, whose second moment diverges: a
    finite number computed from a finite sample of a diverging quantity is not
    an estimate of anything, and returning one silently is the failure mode
    ``PROJECT.md`` section 4 explicitly forbids.
    """


def check_layout(trajectories: NDArray[Any]) -> NDArray[Any]:
    """Assert the project's array layout and return the array unchanged.

    Both ``float32`` and ``float64`` are accepted: the former is what comes back
    from the cache, the latter what a generator produces in memory. What is
    *not* accepted is anything else, including integer arrays, which would make
    a squared displacement overflow silently.

    Parameters
    ----------
    trajectories : ndarray
        Candidate trajectory array.

    Returns
    -------
    ndarray
        The same array, shape ``(n_particles, n_steps, n_dim)``.

    Raises
    ------
    ValueError
        If the array is not 3-D floating point with ``n_dim`` in ``{1, 2}``.
    """
    if trajectories.ndim != 3:
        raise ValueError(
            "trajectories must have shape (n_particles, n_steps, n_dim), got "
            f"ndim={trajectories.ndim} with shape {trajectories.shape}"
        )
    if trajectories.dtype not in (np.float32, np.float64):
        raise ValueError(
            f"trajectories must be float32 or float64, got {trajectories.dtype}"
        )
    if trajectories.shape[2] not in (1, 2):
        raise ValueError(f"n_dim must be 1 or 2, got {trajectories.shape[2]}")
    return trajectories


def _guard_divergent(cfg: Any | None, what: str) -> None:
    """Refuse a second-moment computation for a mechanism that has none."""
    if cfg is not None and not cfg.msd_is_finite:
        raise DivergentMomentError(
            f"the second moment of {type(cfg).name!r} diverges, so its {what} is "
            "not defined; use fractional_moment with q < stability, the median "
            "absolute displacement, or the interquantile range instead "
            "(PROJECT.md section 4)"
        )


# --------------------------------------------------------------------------
# the three MSDs
# --------------------------------------------------------------------------


def squared_displacement(
    trajectories: NDArray[Any],
    cfg: Any | None = None,
) -> NDArray[np.float64]:
    """Per-particle squared displacement from the initial position.

    .. math::

        r_i^2(t) = |\\mathbf{r}_i(t) - \\mathbf{r}_i(0)|^2

    This is the *per-particle* quantity whose particle-average is the EA-MSD.
    It is returned separately because the bootstrap in :func:`fit_powerlaw`
    resamples particles, and therefore needs the ensemble before it is averaged
    away.

    Parameters
    ----------
    trajectories : ndarray
        Shape ``(n_particles, n_steps, n_dim)``, ``float32`` or ``float64``.
    cfg : GeneratorConfig, optional
        If it reports ``msd_is_finite = False`` this function refuses.

    Returns
    -------
    ndarray of float64
        Shape ``(n_particles, n_steps)``.
    """
    check_layout(trajectories)
    _guard_divergent(cfg, "MSD")
    displacement = trajectories - trajectories[:, :1, :]
    return np.sum(displacement * displacement, axis=2, dtype=np.float64)


def ea_msd(
    trajectories: NDArray[Any],
    cfg: Any | None = None,
) -> NDArray[np.float64]:
    """Ensemble-averaged mean squared displacement from the initial position.

    .. math::

        \\mathrm{MSD}(t) = \\big\\langle |\\mathbf{r}(t) - \\mathbf{r}(0)|^2
        \\big\\rangle

    Parameters
    ----------
    trajectories : ndarray
        Shape ``(n_particles, n_steps, n_dim)``.
    cfg : GeneratorConfig, optional
        Guards against a divergent second moment.

    Returns
    -------
    ndarray of float64
        Shape ``(n_steps,)``; ``MSD(0) = 0`` by construction.
    """
    return np.mean(squared_displacement(trajectories, cfg), axis=0, dtype=np.float64)


def log_spaced_lags(
    n_steps: int,
    fit: FitConfig | None = None,
) -> NDArray[np.intp]:
    """Lags for a TA-MSD fit, log-uniform inside ``[lag_min, fraction * T]``.

    Log spacing matters: a linear grid would put almost every point in the
    last decade, which is the noisiest part of a time average and the part the
    window exists to limit.

    Parameters
    ----------
    n_steps : int
        Trajectory length including ``t = 0``, so ``T = n_steps - 1``.
    fit : FitConfig, optional
        Supplies ``lag_min``, ``lag_max_fraction`` and ``n_lags``.

    Returns
    -------
    ndarray of intp
        Strictly increasing unique lags, at least one, at most ``n_lags``.
    """
    fit = FitConfig() if fit is None else fit
    duration = n_steps - 1
    lag_max = max(int(np.floor(fit.lag_max_fraction * duration)), fit.lag_min)
    if lag_max > duration:
        raise ValueError(f"lag window exceeds the trajectory: {lag_max} > {duration}")
    raw = np.logspace(np.log10(fit.lag_min), np.log10(lag_max), fit.n_lags)
    return np.unique(np.round(raw).astype(np.intp))


def ta_msd(
    trajectories: NDArray[Any],
    lags: NDArray[np.intp],
    cfg: Any | None = None,
) -> NDArray[np.float64]:
    """Time-averaged MSD of each trajectory, as a function of lag.

    .. math::

        \\overline{\\delta^2_i(\\Delta)} =
        \\frac{1}{T - \\Delta} \\sum_{t=0}^{T-\\Delta}
        |\\mathbf{r}_i(t+\\Delta) - \\mathbf{r}_i(t)|^2

    This is the observable an experimenter actually has when only one agent is
    visible, which is why Q1 is posed in terms of it. For an ergodic process it
    converges to the EA-MSD; for a CTRW it does not, and grows *linearly* in
    ``Delta`` whatever the waiting-time exponent is.

    Parameters
    ----------
    trajectories : ndarray
        Shape ``(n_particles, n_steps, n_dim)``.
    lags : ndarray of int
        Lags at which to evaluate, each in ``[1, n_steps - 1]``.
    cfg : GeneratorConfig, optional
        Guards against a divergent second moment.

    Returns
    -------
    ndarray of float64
        Shape ``(n_particles, len(lags))``.

    Notes
    -----
    The loop is over lags, not over particles or time: each iteration is one
    vectorised difference over the whole ensemble, and ``len(lags)`` is a few
    dozen. The number of contributing windows falls as ``T - Delta``, which is
    the reason for the ``T/10`` cap in ``PROJECT.md`` section 2.
    """
    check_layout(trajectories)
    _guard_divergent(cfg, "TA-MSD")
    lags = np.asarray(lags, dtype=np.intp)
    duration = trajectories.shape[1] - 1
    if lags.size == 0:
        raise ValueError("need at least one lag")
    if lags.min() < 1 or lags.max() > duration:
        raise ValueError(
            f"lags must lie in [1, {duration}], got [{lags.min()}, {lags.max()}]"
        )

    out = np.empty((trajectories.shape[0], lags.size), dtype=np.float64)
    for column, lag in enumerate(lags):
        step = trajectories[:, lag:, :] - trajectories[:, :-lag, :]
        out[:, column] = np.mean(
            np.sum(step * step, axis=2, dtype=np.float64), axis=1, dtype=np.float64
        )
    return out


def ea_ta_msd(
    trajectories: NDArray[Any],
    lags: NDArray[np.intp],
    cfg: Any | None = None,
) -> NDArray[np.float64]:
    """Ensemble average of the time-averaged MSD.

    .. math::

        \\big\\langle \\overline{\\delta^2(\\Delta)} \\big\\rangle

    Averaging restores the ensemble exponent for a CTRW even though no single
    trajectory shows it -- which is exactly why a single observed agent is not
    enough, and why the project asks how many agents are.

    Parameters
    ----------
    trajectories : ndarray
        Shape ``(n_particles, n_steps, n_dim)``.
    lags : ndarray of int
        Lags at which to evaluate.
    cfg : GeneratorConfig, optional
        Guards against a divergent second moment.

    Returns
    -------
    ndarray of float64
        Shape ``(len(lags),)``.
    """
    return np.mean(ta_msd(trajectories, lags, cfg), axis=0, dtype=np.float64)


def fractional_moment(
    trajectories: NDArray[Any],
    q: float,
    per_particle: bool = False,
) -> NDArray[np.float64]:
    """Fractional moment of the displacement from the initial position.

    .. math::

        m_q(t) = \\big\\langle |\\mathbf{r}(t) - \\mathbf{r}(0)|^q
        \\big\\rangle

    For a self-similar process with ``|r| ~ t^nu`` this scales as
    ``m_q(t) ~ t^(q nu)``. It is the observable to use when the second moment
    does not exist: a symmetric ``s``-stable Levy flight has finite ``m_q`` for
    every ``q < s``, with ``nu = 1/s``.

    Parameters
    ----------
    trajectories : ndarray
        Shape ``(n_particles, n_steps, n_dim)``.
    q : float
        Moment order, ``q > 0``. The caller is responsible for keeping ``q``
        below the stability index when the process is stable.
    per_particle : bool
        If ``True``, return the ``(n_particles, n_steps)`` contributions rather
        than their mean, for bootstrapping.

    Returns
    -------
    ndarray of float64
        Shape ``(n_steps,)``, or ``(n_particles, n_steps)`` if ``per_particle``.
    """
    check_layout(trajectories)
    if q <= 0.0:
        raise ValueError(f"q must be > 0, got {q}")
    displacement = trajectories - trajectories[:, :1, :]
    radius = np.sqrt(np.sum(displacement * displacement, axis=2, dtype=np.float64))
    contributions = radius**q
    return contributions if per_particle else np.mean(contributions, axis=0, dtype=np.float64)


# --------------------------------------------------------------------------
# power-law fitting
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PowerLawFit:
    """Result of a least-squares fit of ``y = A x^b`` in log-log space.

    Attributes
    ----------
    exponent : float
        Fitted ``b``.
    exponent_err : float
        Residual-based least-squares error on ``b``.

        **This is a goodness-of-fit diagnostic only and must not be quoted as
        the uncertainty on the exponent** (``PROJECT.md`` section 2). It assumes
        independent points, whereas an EA-MSD is built from the same particles
        at every ``t`` and is strongly autocorrelated; measured against the
        seed-to-seed spread it is optimistic by a factor of about twenty-five.
        Use :attr:`bootstrap_err` instead.
    bootstrap_err : float or None
        Standard deviation of the exponent over particle resamples -- the
        authoritative uncertainty. ``None`` when the fit was not bootstrapped,
        which happens only for a 1-D input curve or a single particle.
    bootstrap_ci : tuple of float or None
        Percentile confidence interval (2.5 to 97.5) over the resamples.
    n_bootstrap : int
        Number of resamples actually performed.
    prefactor : float
        Fitted ``A``, i.e. ``exp(intercept)``.
    window : tuple of float
        ``(x_min, x_max)`` actually used, after the window restriction and the
        removal of non-positive points.
    n_points : int
        Number of points entering the fit.
    weighting : str
        ``"uniform"`` or ``"gls"``, the latter meaning inverse-variance weights
        taken from the bootstrap.
    """

    exponent: float
    exponent_err: float
    prefactor: float
    window: tuple[float, float]
    n_points: int
    bootstrap_err: float | None = None
    bootstrap_ci: tuple[float, float] | None = None
    n_bootstrap: int = 0
    weighting: str = "uniform"

    @property
    def uncertainty(self) -> float:
        """The number to quote: bootstrap error if available, else the OLS one.

        Accessing this rather than a named field makes the fallback explicit at
        the call site instead of hiding an unbootstrapped error bar behind the
        same name as a bootstrapped one.
        """
        return self.exponent_err if self.bootstrap_err is None else self.bootstrap_err

    def __str__(self) -> str:
        error = (
            f"+/- {self.bootstrap_err:.4f} (bootstrap, n={self.n_bootstrap})"
            if self.bootstrap_err is not None
            else f"+/- {self.exponent_err:.4f} (OLS diagnostic only)"
        )
        return (
            f"{self.exponent:.4f} {error}, A={self.prefactor:.4g}, "
            f"window=[{self.window[0]:g}, {self.window[1]:g}], "
            f"n={self.n_points}, {self.weighting}"
        )


def _weighted_loglog_fit(
    log_x: NDArray[np.float64],
    log_y: NDArray[np.float64],
    weights: NDArray[np.float64],
) -> tuple[float, float]:
    """Solve the two-parameter weighted least-squares problem. Returns ``(b, log A)``."""
    design = np.column_stack([log_x, np.ones_like(log_x)])
    sqrt_w = np.sqrt(weights)
    coeffs, *_ = np.linalg.lstsq(design * sqrt_w[:, None], log_y * sqrt_w, rcond=None)
    return float(coeffs[0]), float(coeffs[1])


def _ols_error(
    log_x: NDArray[np.float64],
    log_y: NDArray[np.float64],
    weights: NDArray[np.float64],
    slope: float,
    intercept: float,
) -> float:
    """Residual-scaled standard error on the slope (the ``chi2/dof`` convention)."""
    design = np.column_stack([log_x, np.ones_like(log_x)])
    residuals = log_y - (slope * log_x + intercept)
    dof = log_x.size - 2
    if dof <= 0:
        return float("nan")
    chi2_per_dof = float(np.sum(weights * residuals**2) / dof)
    covariance = np.linalg.inv((design * weights[:, None]).T @ design)
    return float(np.sqrt(chi2_per_dof * covariance[0, 0]))


def _bootstrap_means(
    per_particle: NDArray[np.float64],
    n_bootstrap: int,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """Bootstrap replicates of the particle mean.

    Resampling ``n`` particles with replacement is exactly drawing multinomial
    counts over the ``n`` particles, so the replicates are obtained as one
    matrix product rather than ``n_bootstrap`` fancy-indexed copies. For
    4000 particles and 200 resamples this is a fraction of a second instead of
    several, and its peak memory is the counts matrix rather than a copy of the
    ensemble.

    Parameters
    ----------
    per_particle : ndarray of float64
        Shape ``(n_particles, n_points)``.
    n_bootstrap : int
        Number of resamples.
    rng : numpy.random.Generator
        Explicit generator.

    Returns
    -------
    ndarray of float64
        Shape ``(n_bootstrap, n_points)``.
    """
    n_particles = per_particle.shape[0]
    counts = rng.multinomial(
        n_particles, np.full(n_particles, 1.0 / n_particles), size=n_bootstrap
    ).astype(np.float64)
    return (counts @ per_particle) / n_particles


def fit_powerlaw(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    x_min: float | None = None,
    x_max: float | None = None,
    weights: NDArray[np.float64] | None = None,
    fit: FitConfig | None = None,
) -> PowerLawFit:
    """Fit ``y = A x^b`` in log-log space, with bootstrap uncertainty by default.

    Parameters
    ----------
    x : ndarray of float64
        Abscissa, shape ``(n,)``.
    y : ndarray of float64
        Either the curve to fit, shape ``(n,)``, or the **per-particle
        contributions**, shape ``(n_particles, n)``, whose particle-mean is the
        curve. Passing the 2-D form is the default path and is what enables the
        bootstrap; the 1-D form is for synthetic curves that have no ensemble
        behind them.
    x_min, x_max : float, optional
        Inclusive window on ``x``. There is deliberately no default: the right
        window is a statement about the process, not a universal constant (see
        the module docstring).
    weights : ndarray of float64, optional
        Explicit per-point weights, overriding the bootstrap-derived ones.
    fit : FitConfig, optional
        Supplies ``n_bootstrap``, ``bootstrap_seed`` and ``use_gls``.

    Returns
    -------
    PowerLawFit
        With :attr:`~PowerLawFit.bootstrap_err` populated whenever an ensemble
        was supplied and ``n_bootstrap > 0``.

    Raises
    ------
    ValueError
        If fewer than three points survive the window.

    Notes
    -----
    The procedure is two-pass, because the weights and the uncertainty come from
    the same resampling:

    1. resample particles, rebuild the curve, fit each replicate with uniform
       weights, and take the per-point variance of ``log y`` across replicates;
    2. refit the point estimate *and* every replicate with inverse-variance
       weights, and report the spread of the reweighted replicates.

    Only the diagonal of the empirical lag-lag covariance is used as weights.
    The full covariance is what the correlation structure really is, but
    estimating an ``n x n`` covariance from ``n_bootstrap`` replicates is
    rank-deficient whenever ``n_bootstrap < n``, which is the normal case here
    (a few hundred replicates against a thousand time points). Inverting a
    rank-deficient covariance would manufacture confidence rather than measure
    it. The diagonal gives an efficient point estimate; the correlation it
    ignores is precisely what the bootstrap spread -- not the residuals --
    accounts for.

    Empirical weights are used rather than an analytic variance model such as
    the Levy walk's ``t^(g-1)``, because the analytic form depends on the very
    exponent being estimated and would require iterating on it.
    """
    fit = FitConfig() if fit is None else fit
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    if y.ndim == 2:
        per_particle: NDArray[np.float64] | None = y
        curve = np.mean(y, axis=0, dtype=np.float64)
    elif y.ndim == 1:
        per_particle = None
        curve = y
    else:
        raise ValueError(f"y must be 1-D or 2-D, got shape {y.shape}")
    if x.shape != curve.shape:
        raise ValueError(f"x and y must agree in length, got {x.shape}, {curve.shape}")

    keep = np.isfinite(x) & np.isfinite(curve) & (x > 0.0) & (curve > 0.0)
    if x_min is not None:
        keep &= x >= x_min
    if x_max is not None:
        keep &= x <= x_max
    if keep.sum() < 3:
        raise ValueError(
            f"need at least 3 positive points in the window [{x_min}, {x_max}] "
            f"to fit a power law, got {int(keep.sum())}"
        )

    log_x = np.log(x[keep])
    log_y = np.log(curve[keep])

    # ---- resampling: replicate curves, used for both weights and spread ----
    replicates: NDArray[np.float64] | None = None
    if per_particle is not None and fit.n_bootstrap > 0 and per_particle.shape[0] > 1:
        rng = np.random.default_rng(fit.bootstrap_seed)
        drawn = _bootstrap_means(per_particle[:, keep], fit.n_bootstrap, rng)
        # A replicate can contain a non-positive point where the original did
        # not; those replicates cannot be log-fitted and are dropped.
        usable = np.all(drawn > 0.0, axis=1)
        replicates = np.log(drawn[usable]) if usable.any() else None

    # ---- weights ----
    weighting = "uniform"
    if weights is not None:
        w = np.asarray(weights, dtype=np.float64)[keep]
        weighting = "explicit"
    elif replicates is not None and fit.use_gls and replicates.shape[0] > 2:
        variance = replicates.var(axis=0, ddof=1)
        floor = max(float(np.median(variance)) * 1e-6, np.finfo(np.float64).tiny)
        w = 1.0 / np.maximum(variance, floor)
        w = w / w.mean()
        weighting = "gls"
    else:
        w = np.ones_like(log_x)
    if np.any(w <= 0.0):
        raise ValueError("weights must be strictly positive")

    slope, intercept = _weighted_loglog_fit(log_x, log_y, w)
    ols_err = _ols_error(log_x, log_y, w, slope, intercept)

    # ---- uncertainty from the reweighted replicates ----
    bootstrap_err: float | None = None
    bootstrap_ci: tuple[float, float] | None = None
    n_done = 0
    if replicates is not None:
        slopes = np.array(
            [_weighted_loglog_fit(log_x, row, w)[0] for row in replicates]
        )
        n_done = int(slopes.size)
        if n_done > 1:
            bootstrap_err = float(slopes.std(ddof=1))
            low, high = np.percentile(slopes, [2.5, 97.5])
            bootstrap_ci = (float(low), float(high))

    return PowerLawFit(
        exponent=slope,
        exponent_err=ols_err,
        prefactor=float(np.exp(intercept)),
        window=(float(x[keep].min()), float(x[keep].max())),
        n_points=int(keep.sum()),
        bootstrap_err=bootstrap_err,
        bootstrap_ci=bootstrap_ci,
        n_bootstrap=n_done,
        weighting=weighting,
    )
