"""Analytic reference MSDs, one per mechanism, for the validation-first rule.

``PROJECT.md`` section 4 tabulates the exponent of every mechanism. Where the
full closed form of the ensemble MSD is known exactly at every ``t`` -- not just
asymptotically -- it is written out here, so a validation test can check the
*amplitude* as well as the slope. Where only the asymptotic scaling is known in
a convention-independent way, that is said explicitly and only the exponent is
offered.

Which is which:

===============  =========================================================
Mechanism        Reference available
===============  =========================================================
Brownian         exact at all ``t``: ``2 d D t``
fBm              exact at all ``t``: ``2 d D t^(2H)``
scaled Brownian  exact at all ``t``: ``2 d D0 t^alpha``
diffusing diff.  exact at all ``t`` in the mean: ``2 d <D> t``
CTRW             asymptotic amplitude known, ``t >> tau0`` only
Levy walk        exponent only (prefactor is convention-dependent)
Levy flight      none -- the second moment diverges
===============  =========================================================
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.special import gamma as gamma_fn

from ..config import (
    BrownianConfig,
    CTRWConfig,
    DDMConfig,
    FBMConfig,
    LevyFlightConfig,
    LevyWalkConfig,
    SBMConfig,
)
from ..estimators.msd import DivergentMomentError

__all__ = [
    "analytic_msd",
    "ctrw_finite_time_msd",
    "ctrw_msd_amplitude",
    "ctrw_predicted_bias",
    "ctrw_renewal_count",
    "has_exact_msd",
    "mean_diffusivity",
]


def mean_diffusivity(cfg: DDMConfig) -> float:
    """Stationary mean diffusivity of the diffusing-diffusivity model.

    With ``D = |Y|^2`` and ``Y`` an ``n_aux``-component Ornstein-Uhlenbeck
    process ``dY = -Y dt / tau + sigma dW``, each component is stationary
    Gaussian with variance ``sigma^2 tau / 2``, so

    .. math::

        \\langle D \\rangle = n_{\\mathrm{aux}} \\, \\sigma^2 \\tau / 2 .

    Parameters
    ----------
    cfg : DDMConfig
        Diffusing-diffusivity config.

    Returns
    -------
    float
        ``<D>`` in the stationary state.
    """
    return cfg.n_aux * cfg.sigma**2 * cfg.tau / 2.0


def ctrw_msd_amplitude(cfg: CTRWConfig) -> float:
    """Asymptotic amplitude ``K`` in ``MSD(t) -> K t^a`` for the CTRW.

    For waiting times with the Pareto tail
    ``P(tau > t) = (t / tau0)^(-a)``, ``0 < a < 1``, the mean number of
    renewals up to ``t`` grows as

    .. math::

        \\langle n(t) \\rangle \\simeq
        \\frac{(t/\\tau_0)^a}{\\Gamma(1-a)\\,\\Gamma(1+a)}
        = \\frac{\\sin(\\pi a)}{\\pi a} \\, (t/\\tau_0)^a ,

    and each renewal contributes an independent jump with per-component
    variance ``jump_scale^2``, so in ``d`` dimensions

    .. math::

        \\langle r^2(t) \\rangle \\simeq
        d \\, \\sigma_{\\mathrm{jump}}^2 \\, \\langle n(t) \\rangle .

    Parameters
    ----------
    cfg : CTRWConfig
        CTRW config.

    Returns
    -------
    float
        ``K`` such that ``MSD(t) -> K t^a`` as ``t / tau0 -> infinity``.

    Notes
    -----
    This is the *leading* term only, and the corrections to it decay slowly
    enough to matter at every trajectory length this project will ever run.
    Expanding the Laplace transform of the pure Pareto density,

    .. math::

        \\hat\\psi(s) = 1 - \\Gamma(1-a)(\\tau_0 s)^a
                       + \\frac{a\\,\\tau_0 s}{1-a} + O(s^2) ,

    and inverting the renewal equation gives

    .. math::

        \\langle n(t)\\rangle \\simeq
        \\frac{(t/\\tau_0)^a}{\\Gamma(1-a)\\Gamma(1+a)}
        \\left[1 + \\frac{a}{(1-a)\\Gamma(1-a)}
        \\left(\\frac{\\tau_0}{t}\\right)^{1-a}\\right] - 1 .

    Two consequences drive the choice of validation parameters:

    * the relative correction decays as ``(t/tau0)^-(1-a)``, so it vanishes
      slowly as ``a -> 1``: at ``a = 0.9`` and ``t = 4000 tau0`` it is still
      of order 40 per cent, and the fitted exponent is biased low by ~0.07;
    * the additive ``-1`` dominates instead when ``a`` is small, because
      ``<n(t)>`` is then only of order ten events; at ``a = 0.3`` it biases the
      fitted exponent *high* by ~0.07.

    The bias therefore changes sign with ``a``, passing through zero near
    ``a ~ 0.55``. Both branches were checked against the measured renewal count
    (``a = 0.3``: predicted ratio to the leading term 0.906, measured 0.910;
    ``a = 0.7``: predicted 1.067, measured 1.074). None of this is a defect of
    the generator -- it is what a finite trajectory of a genuinely scale-free
    process looks like -- but it means the exponent must be validated at
    intermediate ``a`` with a stated window, and its residual bias demonstrated
    to shrink as ``T`` grows rather than assumed away.
    """
    a = cfg.alpha_wait
    renewal_amplitude = 1.0 / (gamma_fn(1.0 - a) * gamma_fn(1.0 + a))
    return cfg.n_dim * cfg.jump_scale**2 * renewal_amplitude * cfg.tau0 ** (-a)


def ctrw_renewal_count(cfg: CTRWConfig, t: NDArray[np.float64]) -> NDArray[np.float64]:
    """Finite-time mean renewal count ``<n(t)>`` for the CTRW.

    The leading term of :func:`ctrw_msd_amplitude` plus the two corrections that
    are large at any trajectory length this project runs:

    .. math::

        \\langle n(t)\\rangle \\simeq
        \\frac{(t/\\tau_0)^a}{\\Gamma(1-a)\\Gamma(1+a)}
        \\left[1 + \\frac{a}{(1-a)\\Gamma(1-a)}
        \\left(\\frac{\\tau_0}{t}\\right)^{1-a}\\right] - 1 .

    Parameters
    ----------
    cfg : CTRWConfig
        CTRW config.
    t : ndarray of float64
        Times, in units of ``dt``.

    Returns
    -------
    ndarray of float64
        ``<n(t)>``, clipped at zero (the expansion goes negative for
        ``t < tau0``, where it does not apply).

    Notes
    -----
    Verified against a direct measurement of the renewal count at
    ``N = 20000``: predicted ratio to the pure leading term 0.906 against a
    measured 0.910 at ``a = 0.3``, and 1.067 against 1.074 at ``a = 0.7``.
    The expansion degrades as ``a -> 1``, where higher-order terms matter: at
    ``a = 0.9`` and ``t = 4000 tau0`` it predicts a ratio of 1.41 where 1.74 is
    measured. Do not use it as a correction above ``a ~ 0.8``.
    """
    t = np.asarray(t, dtype=np.float64)
    a = cfg.alpha_wait
    scaled = np.where(t > 0.0, t / cfg.tau0, np.nan)
    leading = scaled**a / (gamma_fn(1.0 - a) * gamma_fn(1.0 + a))
    correction = 1.0 + a / ((1.0 - a) * gamma_fn(1.0 - a)) * scaled ** (-(1.0 - a))
    return np.clip(np.nan_to_num(leading * correction - 1.0, nan=0.0), 0.0, None)


def ctrw_finite_time_msd(cfg: CTRWConfig, t: NDArray[np.float64]) -> NDArray[np.float64]:
    """CTRW ensemble MSD including the finite-time renewal corrections.

    .. math::

        \\langle r^2(t)\\rangle = d\\,\\sigma_{\\mathrm{jump}}^2\\,
        \\langle n(t)\\rangle ,

    exact given ``<n(t)>``, because the jumps are independent of the clock and
    have finite variance.

    This is the form to fit, or to predict the bias of, rather than a pure power
    law -- see :func:`ctrw_predicted_bias`.

    Parameters
    ----------
    cfg : CTRWConfig
        CTRW config.
    t : ndarray of float64
        Times, in units of ``dt``.

    Returns
    -------
    ndarray of float64
        ``<r^2(t)>`` with corrections included.
    """
    return cfg.n_dim * cfg.jump_scale**2 * ctrw_renewal_count(cfg, t)


def ctrw_predicted_bias(
    cfg: CTRWConfig,
    t_min: float,
    t_max: float,
    n_points: int = 512,
) -> float:
    """Bias a pure power-law fit will show over ``[t_min, t_max]``, in closed form.

    Fits a straight line to ``log`` of :func:`ctrw_finite_time_msd` over the
    window and returns ``slope - a``. Because the correction is known
    analytically, an exponent measured on a CTRW should be *reported alongside*
    this number, or corrected by it, rather than merely measured on a window
    chosen to make it small (``PROJECT.md`` section 4).

    Parameters
    ----------
    cfg : CTRWConfig
        CTRW config.
    t_min, t_max : float
        Fit window, in units of ``dt``.
    n_points : int
        Number of log-spaced points used to evaluate the closed form. This is a
        quadrature parameter, not a statistical one -- the curve is noiseless.

    Returns
    -------
    float
        Predicted ``fitted_exponent - alpha_wait``. Positive means the fit will
        overestimate.

    Notes
    -----
    The sign reverses with ``a``. The additive ``-1`` dominates at small ``a``,
    where ``<n(t)>`` is only of order ten events, and biases the exponent *up*;
    the slowly decaying ``(tau0/t)^(1-a)`` term dominates as ``a -> 1`` and
    biases it *down*. For a window of ``[0.1T, T]`` the zero crossing sits at
    ``a = 0.604`` for ``T = 4096``, drifting to ``a = 0.566`` by ``T = 65536``.
    A "the bias shrinks as T grows" check would therefore be wrong: at
    ``a = 0.5`` the bias falls from 0.014 to 0.003 between those lengths, while
    at ``a = 0.8`` it barely moves, from -0.025 to -0.020.
    """
    t = np.logspace(np.log10(t_min), np.log10(t_max), n_points)
    msd = ctrw_finite_time_msd(cfg, t)
    good = msd > 0.0
    if good.sum() < 3:
        raise ValueError(
            f"finite-time form is non-positive over [{t_min}, {t_max}]; the "
            "expansion does not apply below a few tau0"
        )
    slope = np.polyfit(np.log(t[good]), np.log(msd[good]), 1)[0]
    return float(slope - cfg.alpha_wait)


def has_exact_msd(cfg: Any) -> bool:
    """Whether :func:`analytic_msd` is exact at every ``t`` for this config.

    Parameters
    ----------
    cfg : GeneratorConfig
        Any generator config.

    Returns
    -------
    bool
        ``True`` for Brownian, fBm, scaled Brownian and diffusing diffusivity;
        ``False`` for CTRW (asymptotic only), Levy walk (exponent only) and
        Levy flight (no MSD at all).
    """
    return isinstance(cfg, (BrownianConfig, FBMConfig, SBMConfig, DDMConfig))


def analytic_msd(cfg: Any, t: NDArray[np.float64]) -> NDArray[np.float64]:
    """Analytic ensemble MSD for a generator config.

    Parameters
    ----------
    cfg : GeneratorConfig
        Any generator config except :class:`~collectivediff.config.LevyWalkConfig`
        and :class:`~collectivediff.config.LevyFlightConfig`.
    t : ndarray of float64
        Times at which to evaluate, in units of ``dt = 1``.

    Returns
    -------
    ndarray of float64
        ``<r^2(t)>``. Exact for the mechanisms listed by :func:`has_exact_msd`,
        asymptotic (``t >> tau0``) for the CTRW.

    Raises
    ------
    DivergentMomentError
        For a Levy flight, whose second moment does not exist.
    NotImplementedError
        For a Levy walk: the exponent ``3 - g`` is known and is what validation
        checks, but the prefactor depends on conventions (wait-first versus
        jump-first, treatment of the flight in progress) that are not worth
        pinning down for a test that is about the scaling.
    """
    t = np.asarray(t, dtype=np.float64)
    d = cfg.n_dim

    if isinstance(cfg, BrownianConfig):
        return 2.0 * d * cfg.diffusivity * t
    if isinstance(cfg, FBMConfig):
        return 2.0 * d * cfg.diffusivity * t ** (2.0 * cfg.hurst)
    if isinstance(cfg, SBMConfig):
        return 2.0 * d * cfg.diffusivity * t**cfg.alpha
    if isinstance(cfg, DDMConfig):
        return 2.0 * d * mean_diffusivity(cfg) * t
    if isinstance(cfg, CTRWConfig):
        return ctrw_msd_amplitude(cfg) * t**cfg.alpha_wait
    if isinstance(cfg, LevyWalkConfig):
        raise NotImplementedError(
            "no convention-independent closed form for the Levy-walk MSD "
            f"prefactor; validate the exponent {cfg.alpha_analytic} instead"
        )
    if isinstance(cfg, LevyFlightConfig):
        raise DivergentMomentError(
            "the second moment of a Levy flight diverges; there is no analytic "
            "MSD to compare against (PROJECT.md section 4)"
        )
    raise TypeError(f"no analytic MSD registered for {type(cfg).__name__}")
