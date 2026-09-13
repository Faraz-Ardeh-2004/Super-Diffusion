"""Truncated ARFIMA(0,d,0) agents under consensus mean-field coupling (Phase 3).

``PROJECT.md`` section 7 derives the model this simulates. Two structural facts
from that derivation are load-bearing here, not just narrative:

* **The recursion is velocity, not position.** ``v_i(t+1)`` is built from its
  own truncated memory plus a consensus pull, and position is the cumulative
  sum of velocity -- the same relationship every other generator in this
  package uses between an increment process and its trajectory.
* **The mean field cancels the coupling term exactly.** Averaging the update
  over agents makes the ``J`` term vanish (``<v> - v_i`` averages to zero), so
  ``<v>(t)`` obeys the *uncoupled* recursion at every ``J``. This is why
  ``<v>`` is stored as a first-class output rather than recomputed from
  ``velocities`` at each call site: getting the average right once here is the
  entire content of the conservation law in ``PROJECT.md`` section 7 (V2).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.special import gamma as gamma_fn

from ..config import MeanFieldConfig
from ..generators._common import as_trajectories, require_generator

__all__ = [
    "MeanFieldTrajectories",
    "arfima_acf",
    "arfima_effective_exponent",
    "arfima_exact_msd",
    "coupling_for_target_crossover",
    "deviation_spectrum",
    "fractional_diff_kappa",
    "simulate_meanfield",
    "truncated_arfima_acf",
    "whittle_d_only_estimate",
    "whittle_dj_estimate",
    "whittle_objective",
]


def coupling_for_target_crossover(d: float, tau_c: float) -> float:
    """Coupling ``J`` that puts the memory crossover at a target ``tau_c``.

    Inverts ``tau_c ~ J^(-1/|d|)`` (``PROJECT.md`` section 7, ``d > 0``):

    .. math::

        J = \\tau_c^{-|d|}

    Sweeping a fixed ``J`` grid across ``d`` scatters ``tau_c`` over many
    decades (at ``d = 0.25`` the default ``PHASE3_PROMPT.md`` grid gives
    ``tau_c`` from 16 to 160000, and only the ``J = 0.5`` point lands inside
    the trustworthy ``K / 10`` window). Choosing ``J`` from a target ``tau_c``
    instead keeps every sweep point inside a window chosen in advance.

    Only meaningful for ``d > 0``: for ``d < 0`` the fractional term dominates
    ``J`` at every reachable frequency and there is no crossover to target
    (``PROJECT.md`` section 7). Used with ``abs(d)`` there is still a
    well-defined number, useful for mirroring a comparable *coupling
    strength* across the sign of ``d`` even though it has no crossover
    interpretation on that side.

    Parameters
    ----------
    d : float
        Fractional differencing parameter (only its magnitude is used).
    tau_c : float
        Target crossover lag, ``> 0``.

    Returns
    -------
    float
        Coupling strength ``J``.
    """
    if tau_c <= 0.0:
        raise ValueError(f"tau_c must be > 0, got {tau_c}")
    if d == 0.0:
        raise ValueError("d = 0 has no memory and hence no crossover to target")
    return float(tau_c ** (-abs(d)))


def deviation_spectrum(
    omega: NDArray[np.float64], d: float, j: float, sigma: float = 1.0
) -> NDArray[np.float64]:
    """Closed-form spectral density of ``delta_i = v_i - <v>`` (``PROJECT.md`` sec. 7).

    .. math::

        S_\\delta(\\omega) = \\frac{\\sigma^2}
        {\\left| (1 - e^{-i\\omega})^d + J e^{-i\\omega} \\right|^2}

    Low-frequency behaviour depends on the *sign* of ``d``: as
    ``omega -> 0``, ``(1 - e^{-i omega})^d -> 0`` for ``d > 0``, so the
    denominator tends to ``J`` and ``S_delta(0) = sigma^2 / (2 pi J^2)`` is
    finite -- the deviation is normally diffusive at long times, with a
    memory crossover at ``tau_c ~ J^(-1/|d|)``. For ``d < 0`` the same term
    *diverges*, dominates ``J`` at every reachable frequency, and the
    anomalous exponent survives coupling untouched -- verified numerically in
    ``NOTES.md`` (deviation exponent near 0.5 for `d = -0.25` across the
    whole `J` sweep, against a fall from 1.48 to 1.19 at `d = +0.25`).

    The explicit ``1 / (2 pi)`` matches the periodogram convention used
    throughout this package (:func:`~collectivediff.features.spectral.fgn_spectral_density`,
    :func:`~collectivediff.features.spectral.whittle_log_likelihood`):
    ``periodogram = |FFT|^2 / (2 pi n)`` estimates *this* normalisation of
    ``S``, not the bare ``sigma^2 / |denominator|^2`` a textbook AR spectral
    density is usually written with. Confirmed empirically against a
    simulated periodogram before shipping: the two agreed in *shape* but
    differed by a flat factor of `2 pi` (ratio `0.157-0.165` across a decade
    of frequencies, against `1/(2 pi) = 0.159`) until this factor was added.

    Parameters
    ----------
    omega : ndarray of float64
        Angular frequencies, ``(0, pi]``. Excludes exactly 0, where the
        expression is singular for ``d < 0`` and a removable ``0/0`` for
        ``J = 0``.
    d : float
        Fractional differencing parameter.
    j : float
        Consensus coupling strength.
    sigma : float
        Innovation noise standard deviation.

    Returns
    -------
    ndarray of float64
        ``S_delta(omega)``, in the periodogram-matched normalisation.
    """
    omega = np.asarray(omega, dtype=np.float64)
    if np.any(omega <= 0.0) or np.any(omega > np.pi + 1e-9):
        raise ValueError("omega must lie in (0, pi]")
    z = 1.0 - np.exp(-1j * omega)
    denominator = z**d + j * np.exp(-1j * omega)
    return sigma**2 / (2.0 * np.pi * np.abs(denominator) ** 2)


def _profiled_whittle_objective(
    omega: NDArray[np.float64], periodogram: NDArray[np.float64], d: float, j: float
) -> float:
    """Whittle objective at ``(d, J)`` with the noise scale ``sigma^2`` profiled out.

    ``sigma`` enters :func:`deviation_spectrum` as a pure multiplicative
    prefactor of the whole spectrum, for *any* fixed ``(d, J)`` -- so it can
    be eliminated analytically rather than fit as a third free parameter,
    exactly as :func:`~collectivediff.features.spectral.whittle_mle_hurst`
    does for fGn. Not doing this was a real bug, found when
    :func:`whittle_d_only_estimate` was applied to the mean field: its true
    scale is ``sigma^2 / N``, not the ``sigma = 1`` every fit implicitly
    assumed by calling :func:`deviation_spectrum` at its default. For an
    individual agent's own series (scale exactly 1, matching the config)
    this never showed up; for the mean field the unprofiled fit pinned
    ``d_hat`` to the search bound instead of converging (``NOTES.md``, V5).

    Parameters
    ----------
    omega : ndarray of float64
        Frequencies, ``(0, pi]``.
    periodogram : ndarray of float64
        Periodogram values at ``omega``.
    d, j : float
        Candidate parameters.

    Returns
    -------
    float
        ``m * log(sigma2_hat) + sum(log(shape))``, the profiled Whittle
        objective (to minimise), where ``shape = deviation_spectrum(omega, d, j, sigma=1)``.
    """
    shape = deviation_spectrum(omega, d, j, sigma=1.0)
    m = omega.size
    sigma2_hat = float(np.sum(periodogram / shape)) / m
    if sigma2_hat <= 0.0:
        return 1e12
    return float(np.sum(np.log(shape)) + m * np.log(sigma2_hat))


def whittle_objective(
    omega: NDArray[np.float64], periodogram: NDArray[np.float64], d: float, j: float
) -> float:
    """Public entry point to :func:`_profiled_whittle_objective`.

    Exists so a caller can scan the fitted surface directly (e.g. to check
    whether a boundary-pinned :func:`whittle_dj_estimate` result is a genuine
    optimum or a missed interior one -- ``NOTES.md``, step 3's residual-surface
    diagnostic) without reaching into a private function.
    """
    return _profiled_whittle_objective(omega, periodogram, d, j)


def whittle_d_only_estimate(
    omega: NDArray[np.float64],
    periodogram: NDArray[np.float64],
    d_bounds: tuple[float, float] = (-0.49, 0.49),
) -> float:
    """Naive (coupling-blind) Whittle MLE of ``d``, forcing ``J = 0`` in the model.

    Fits :func:`deviation_spectrum` with ``J`` fixed at 0 -- i.e. the plain
    ARFIMA(0,d,0) shape -- to data that may actually have been generated with
    ``J > 0``. This is deliberately the *wrong* model whenever the data is
    coupled, and that is the point: unlike :func:`whittle_dj_estimate` (which
    knows about ``J`` and correctly recovers the true, ``J``-independent
    ``d`` -- that's what "model-aware" means), a model that cannot represent
    coupling at all is biased by it, since the ``+ J e^{-i omega}`` term in
    :func:`deviation_spectrum` reshapes the *whole* spectrum, not only the
    low-frequency limit the ``d > 0`` crossover argument is about.

    **Important, found empirically, not assumed:** this bias runs toward
    antipersistence (``d_hat`` pulled toward the lower bound) *regardless of
    the sign of the true ``d``* -- confirmed at ``d = -0.25``, where fitted
    ``alpha`` runs from 0.50 down to 0.02 as ``J`` grows from 0 to 0.6, even
    though ``PROJECT.md`` section 7 predicts subdiffusive memory survives
    coupling untouched (confirmed separately, both by the model-aware fit on
    the same data staying flat, and by a windowed ensemble EA-MSD power-law
    fit -- a genuinely model-free measurement -- also staying flat). So this
    function is *not* a stand-in for "the true regime a naive single-agent
    observer would report" on both signs of ``d``; it is a stand-in for
    "what a coupling-blind spectral fit is biased toward," which happens to
    agree with the true regime for ``d > 0`` and does not for ``d < 0``. Use
    a windowed EA-MSD fit (``PROJECT.md``'s standard convention) where the
    model-free answer matters and this function where the specific bias of a
    misspecified spectral fit is the thing being characterized. See
    ``NOTES.md`` (step 2) for the three-way comparison this was found in.

    Parameters
    ----------
    omega : ndarray of float64
        Frequencies, ``(0, pi]``.
    periodogram : ndarray of float64
        Periodogram values at ``omega``.
    d_bounds : tuple of float
        Search bounds.

    Returns
    -------
    float
        ``d_hat``.
    """
    from scipy.optimize import minimize_scalar

    def objective(d: float) -> float:
        return _profiled_whittle_objective(omega, periodogram, d, 0.0)

    result = minimize_scalar(objective, bounds=d_bounds, method="bounded")
    return float(result.x)


def whittle_dj_estimate(
    omega: NDArray[np.float64],
    periodogram: NDArray[np.float64],
    d_bounds: tuple[float, float] = (-0.49, 0.49),
    j_bounds: tuple[float, float] = (1e-4, 2.0),
) -> tuple[float, float]:
    """Joint Whittle MLE of ``(d, J)`` from a deviation periodogram.

    Minimises ``sum[log S_delta(w; d, J) + I(w) / S_delta(w; d, J)]`` over
    ``(d, J)`` jointly, the standard Whittle objective
    (:func:`~collectivediff.features.spectral.whittle_log_likelihood`) with
    :func:`deviation_spectrum` as the model. This replaced an earlier V4
    design that tried to locate the memory crossover ``tau_c`` from the local
    slope of a single agent's MSD: that measurement turned out to be
    unusably noisy at reachable ``N`` even after averaging several
    independent realizations, because the crossover picture derived from
    ``S_delta``'s low-frequency limit does not describe the *short-lag*
    behaviour a local MSD slope is sensitive to (``NOTES.md`` has the
    diagnosis). Whittle fits the *whole* spectral shape at once rather than
    trying to read off one feature of it, and is a harder test besides: it
    has no free window or lag choice, and success means recovering the exact
    ``J`` used to generate the data, not just a consistent trend.

    Multi-start: the objective surface is not convex in ``(d, J)`` (two
    parameters entering through a modulus of a sum of complex functions), and
    a single L-BFGS-B run from the midpoint of the bounds was found to
    converge to a poor local optimum whenever the true ``J`` was far from
    that midpoint -- in practice, whenever ``J`` was small, which is most of
    this phase's sweep (``NOTES.md`` has the numbers: one bad start gave
    ``d_hat = 0.38`` against a true ``0.35``, while a start near the data's
    own scale gave ``0.354``). Restarting from a small grid and keeping the
    lowest objective removes this without needing a smarter (and much more
    fragile) single initial guess.

    **Second multi-start fix.** The first grid (3x3: ``d0 in (-0.2, 0.0,
    0.2)``, ``j0 in (1e-3, 0.05, 0.3)``) still missed a materially better
    basin for ``d < 0`` at strong ``J`` -- found by the pre-phase-4
    residual-surface diagnostic (``NOTES.md``,
    :func:`~collectivediff.studies_phase3.run_step3_residual_surface`): at
    ``d=-0.25, J=0.354``, a full grid search over the whole ``(d, J)`` box
    located a basin at ``d~-0.31, J~0.275-0.3`` scoring ~1.3 nats better than
    the 9-start fit's answer, which the old grid never seeded near (its most
    negative ``d0`` was ``-0.2`` and its largest ``j0`` was ``0.3``, right at
    the missed basin's edge with nothing beyond it to descend from). Widened
    to a 7x7=49-start grid spanning ``d0 in [-0.4, 0.4]`` and
    ``j0`` up to ``1.0`` so both signs of ``d`` and stronger ``J`` have a
    start on the correct side of the missed basin. Refitting the same 60
    diagnostic agents with this grid changed 29/60 (48%) of the ``d<0`` fits
    by more than 0.02 and roughly halved the mean bias (-0.148 -> -0.070);
    the ``d>0`` side was already correct at 9 starts and is unchanged by the
    wider grid (0/60 fits moved).

    Parameters
    ----------
    omega : ndarray of float64
        Frequencies the periodogram was evaluated at, ``(0, pi]``.
    periodogram : ndarray of float64
        Periodogram values at ``omega`` (e.g. from an ensemble-averaged
        deviation periodogram).
    d_bounds, j_bounds : tuple of float
        Search bounds.

    Returns
    -------
    tuple of float
        ``(d_hat, J_hat)``.
    """
    from scipy.optimize import minimize

    def objective(params: NDArray[np.float64]) -> float:
        d, j = params
        return _profiled_whittle_objective(omega, periodogram, d, j)

    best: tuple[float, float, float] | None = None
    for d0 in (-0.4, -0.267, -0.133, 0.0, 0.133, 0.267, 0.4):
        for j0 in (1e-3, 0.02, 0.05, 0.1, 0.3, 0.6, 1.0):
            result = minimize(
                objective, [d0, j0], method="L-BFGS-B", bounds=[d_bounds, j_bounds]
            )
            if best is None or result.fun < best[0]:
                best = (float(result.fun), float(result.x[0]), float(result.x[1]))
    assert best is not None
    return best[1], best[2]


def arfima_acf(d: float, lags: NDArray[np.float64] | list[int]) -> NDArray[np.float64]:
    """Exact autocorrelation of untruncated ARFIMA(0,d,0), the ``K -> inf`` limit.

    .. math::

        \\rho(k) = \\frac{\\Gamma(1-d)\\,\\Gamma(k+d)}{\\Gamma(d)\\,\\Gamma(k+1-d)}

    This, not fractional Gaussian noise, is what the ``J = 0`` reduction of
    ``PROJECT.md`` section 7 actually converges to as ``K -> inf``: ARFIMA and
    fGn share the asymptotic exponent ``alpha = 1 + 2d`` (both are ``k^{2d-1}``
    at large lag) but differ at short lag, e.g. ``rho(1) = d / (1 - d)``
    against fGn's ``2^{2d} - 1``, which are not the same number for any
    ``d != 0``. This is the closed form :func:`truncated_arfima_acf` converges
    to as its truncation length grows, and the correct reference for
    validating :func:`simulate_meanfield` at short lag -- Davies-Harte fGn is
    not.

    Parameters
    ----------
    d : float
        Fractional differencing parameter in ``(-0.5, 0.5)``.
    lags : array_like of int
        Lags at which to evaluate, ``>= 0``.

    Returns
    -------
    ndarray of float64
        ``rho(k)``, with ``rho(0) = 1``.
    """
    k = np.asarray(lags, dtype=np.float64)
    if d == 0.0:
        # Gamma(d) has a pole at d = 0; the d = 0 process is white noise.
        return np.where(k == 0.0, 1.0, 0.0)
    return gamma_fn(1.0 - d) * gamma_fn(k + d) / (gamma_fn(d) * gamma_fn(k + 1.0 - d))


def arfima_exact_msd(d: float, t_max: int) -> NDArray[np.float64]:
    """Exact EA-MSD of untruncated ARFIMA(0,d,0) velocity, unit innovation variance.

    .. math::

        \\mathrm{MSD}(t) = t + 2 \\sum_{k=1}^{t-1} (t - k)\\, \\rho(k)

    -- the variance of a partial sum of a stationary process, computed
    directly from :func:`arfima_acf` with no simulation or truncation
    involved. Convergence of the implied local exponent to the asymptotic
    ``1 + 2d`` is fast but not instantaneous (a few percent by lag 10 for
    ``|d| <= 0.45``), so this is the reference curve for validating a coupled
    simulation's short-window EA-MSD fit, not the bare asymptote.

    Parameters
    ----------
    d : float
        Fractional differencing parameter.
    t_max : int
        Number of time points, ``t = 0 .. t_max - 1``.

    Returns
    -------
    ndarray of float64
        Shape ``(t_max,)``, ``msd[0] = 0``.
    """
    if t_max < 1:
        raise ValueError(f"t_max must be >= 1, got {t_max}")
    rho = arfima_acf(d, np.arange(t_max))
    msd = np.zeros(t_max, dtype=np.float64)
    for t in range(1, t_max):
        k = np.arange(1, t)
        msd[t] = t + 2.0 * np.sum((t - k) * rho[1:t])
    return msd


def arfima_effective_exponent(d: float, lag_min: int, lag_max: int) -> float:
    """OLS log-log slope of the exact ARFIMA MSD over ``[lag_min, lag_max]``.

    The honest reference target for an EA-MSD power-law fit over that window
    on simulated data. A test asserting the fitted exponent equals
    ``1 + 2d`` inside a short window is asserting something the exact process
    itself does not satisfy; this computes what it actually predicts there.

    Parameters
    ----------
    d : float
        Fractional differencing parameter.
    lag_min, lag_max : int
        Fit window, ``1 <= lag_min < lag_max``.

    Returns
    -------
    float
        OLS slope of ``log(MSD)`` versus ``log(t)`` over ``[lag_min, lag_max]``.
    """
    if not 1 <= lag_min < lag_max:
        raise ValueError(f"need 1 <= lag_min < lag_max, got {lag_min}, {lag_max}")
    msd = arfima_exact_msd(d, lag_max + 1)
    t = np.arange(lag_min, lag_max + 1, dtype=np.float64)
    slope, _ = np.polyfit(np.log(t), np.log(msd[lag_min : lag_max + 1]), 1)
    return float(slope)


def fractional_diff_kappa(d: float, k: int) -> NDArray[np.float64]:
    """Truncated fractional-differencing weights ``kappa(s) = -c_s``, ``s = 1..K``.

    ``(1 - B)^d = sum_{s=0}^inf c_s B^s`` with ``c_0 = 1`` and the standard
    recursion ``c_s = c_{s-1} * (s - 1 - d) / s``, computed by a cumulative
    product rather than the ``Gamma(s - d) / (Gamma(-d) Gamma(s + 1))`` closed
    form: for ``K`` of order ``10^3`` the Gamma arguments overflow float64 long
    before the ratio does.

    Parameters
    ----------
    d : float
        Fractional differencing parameter in ``(-0.5, 0.5)``.
    k : int
        Truncation length, ``>= 1``.

    Returns
    -------
    ndarray of float64
        Shape ``(k,)``, ``kappa[s - 1] = kappa(s)`` for ``s = 1 .. k``. Decays
        as ``s^(-1-d)`` for large ``s``, so it is summable for ``d > 0`` and
        only conditionally informative near ``d = -0.5``.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    s = np.arange(1, k + 1, dtype=np.float64)
    ratio = (s - 1.0 - d) / s
    c = np.cumprod(ratio)
    return -c


def truncated_arfima_acf(d: float, k: int, n_lags: int, n_freq: int = 1 << 16) -> NDArray[np.float64]:
    """Exact autocorrelation of the *truncated* AR(K) recursion, not of fGn.

    ``v(t) = sum_{s=1}^K kappa(s) v(t-s) + eps(t)`` has spectral density
    ``S(w) = sigma^2 / |P_K(e^{-iw})|^2`` with ``P_K(z) = 1 + sum_{s=1}^K c_s z^s``
    (``c_s = -kappa(s)``). As ``K -> inf``, ``P_K(z) -> (1-z)^d`` by
    definition and this recovers the fGn spectral density exactly. At finite
    ``K`` it does not, and the gap is not small: the exact fGn spectral
    density has a pole at ``w = 0`` (``P_infinity(1) = 0`` for ``d > 0``), and
    the finite partial sum ``P_K(1) = sum_{s=0}^K c_s`` only approaches 0 as a
    slow power of ``K``, because a truncated sum is worst at reproducing a
    limit exactly at the point it is singular. Concretely, at ``K = 1024``,
    ``d = 0.25`` this is a ~20% relative error in ``rho(1)`` (0.33 truncated
    vs. 0.41 exact fGn) that barely improves between ``K = 64`` and
    ``K = 1024`` -- see ``NOTES.md`` for the numbers and the diagnosis.
    Restricting a comparison to lags ``<= K / 10`` (as ``PHASE3_PROMPT.md``
    specifies) does not fix this, because the bias lives at the *shortest*
    lags, which are always inside that window.

    This function computes the truncated model's *own* exact autocorrelation
    (via the spectral density above and an inverse FFT) so that a simulated
    trajectory can be checked against the model it actually implements,
    separately from how well that model approximates ideal fGn.

    Parameters
    ----------
    d : float
        Fractional differencing parameter.
    k : int
        Truncation length (must match the simulation's ``K``).
    n_lags : int
        Number of lags to return, ``0 .. n_lags - 1``.
    n_freq : int
        FFT grid size; must comfortably exceed ``k`` for the polynomial
        evaluation to resolve ``P_K`` accurately near ``w = 0``.

    Returns
    -------
    ndarray of float64
        Shape ``(n_lags,)``, normalised so ``acf[0] = 1``.
    """
    if n_freq <= k:
        raise ValueError(f"n_freq ({n_freq}) must exceed k ({k})")
    kappa = fractional_diff_kappa(d, k)
    c = np.concatenate(([1.0], -kappa))
    poly = np.zeros(n_freq, dtype=np.float64)
    poly[: k + 1] = c
    p_k = np.fft.fft(poly)
    spectral_density = 1.0 / np.abs(p_k) ** 2
    spectral_density[0] = 0.0
    autocovariance = np.fft.ifft(spectral_density).real
    return autocovariance[:n_lags] / autocovariance[0]


@dataclass(frozen=True)
class MeanFieldTrajectories:
    """Output of :func:`simulate_meanfield`.

    Attributes
    ----------
    positions : ndarray of float64
        Shape ``(N, T, 1)``, cumulative sum of ``velocities`` from the origin.
    velocities : ndarray of float64
        Shape ``(N, T - 1, 1)``, ``velocities[i] = diff(positions[i])``
        exactly -- stored separately only because several Phase 3 observers
        (the model-aware Whittle fit, in particular) work on velocity
        directly and re-differencing positions every call would be pure
        overhead.
    mean_field : ndarray of float64
        Shape ``(T - 1,)``, the population-average velocity at each recorded
        step. First-class output, not recomputed from ``velocities`` at the
        call site (see the module docstring).
    config : MeanFieldConfig
        The config that produced this simulation.
    """

    positions: NDArray[np.float64]
    velocities: NDArray[np.float64]
    mean_field: NDArray[np.float64]
    config: MeanFieldConfig

    def center_of_mass(self) -> NDArray[np.float64]:
        """Center-of-mass position: cumulative sum of :attr:`mean_field`.

        Shape ``(T,)``, starting at 0. The coherent-mode trajectory of
        ``PHASE3_PROMPT.md`` step 2.
        """
        com = np.empty(self.mean_field.size + 1, dtype=np.float64)
        com[0] = 0.0
        np.cumsum(self.mean_field, out=com[1:])
        return com


def simulate_meanfield(
    cfg: MeanFieldConfig, rng: np.random.Generator
) -> MeanFieldTrajectories:
    """Simulate the consensus-coupled ARFIMA(0,d,0) agents of ``PROJECT.md`` section 7.

    The ring buffer holds ``v_i(t), v_i(t-1), ..., v_i(t-K+1)`` for every
    agent, column 0 the most recent. Each step is one ``(N, K) @ (K,)``
    product for the memory term plus an ``O(N)`` consensus term -- the
    ``O(N K)`` per-step cost ``PHASE3_PROMPT.md`` budgets for.

    Burn-in starts the ring buffer at all zeros and runs ``cfg.burn_in``
    update steps before recording anything, so that by the first recorded
    step every agent's memory window is populated by the recursion itself
    rather than by an arbitrary initial condition.

    Parameters
    ----------
    cfg : MeanFieldConfig
        Ensemble size ``N = cfg.n_particles``, length ``T = cfg.n_steps``,
        memory ``K``, coupling ``J``, fractional parameter ``d``, noise
        ``sigma``, and ``burn_in``.
    rng : numpy.random.Generator
        Explicit generator.

    Returns
    -------
    MeanFieldTrajectories
    """
    require_generator(rng)
    n_particles, n_steps, n_dim = cfg.shape
    n_vel = n_steps - 1
    kappa = fractional_diff_kappa(cfg.d, cfg.K)

    ring = np.zeros((n_particles, cfg.K), dtype=np.float64)
    total_steps = cfg.burn_in + n_vel
    noise = cfg.sigma * rng.standard_normal((n_particles, total_steps))

    velocities = np.empty((n_particles, n_vel), dtype=np.float64)
    mean_field = np.empty(n_vel, dtype=np.float64)

    for step in range(total_steps):
        current = ring[:, 0]
        field_now = float(current.mean())
        new_v = ring @ kappa + cfg.J * (field_now - current) + noise[:, step]

        ring[:, 1:] = ring[:, :-1]
        ring[:, 0] = new_v

        record = step - cfg.burn_in
        if record >= 0:
            velocities[:, record] = new_v
            mean_field[record] = float(new_v.mean())

    positions = np.zeros((n_particles, n_steps, n_dim), dtype=np.float64)
    np.cumsum(velocities, axis=1, out=positions[:, 1:, 0])
    positions = as_trajectories(positions, n_particles, n_steps, n_dim)

    return MeanFieldTrajectories(
        positions=positions,
        velocities=velocities[:, :, None],
        mean_field=mean_field,
        config=cfg,
    )
