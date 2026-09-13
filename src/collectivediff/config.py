"""Frozen configuration dataclasses for every simulation in the project.

Following the "config as data" convention of ``PROJECT.md``: simulation
parameters live here, never at call sites and never in notebooks.  Every
config is immutable, JSON-serialisable, and round-trips through
:func:`config_from_dict`, so it can be stored alongside any cached output.

The base class :class:`SimConfig` carries what every simulation needs
(ensemble size, trajectory length, dimension, seed).  One subclass per
generator carries that generator's own parameters together with the analytic
exponent it is supposed to reproduce, exposed as
:attr:`GeneratorConfig.alpha_analytic` so that validation tests never have to
restate it.

Notes
-----
Time is in integer steps of size ``dt = 1`` (``PROJECT.md`` section 2).
Generators with intrinsically continuous time (CTRW, Levy walk) generate a raw
event sequence first and are then resampled onto this uniform grid; the extra
parameters controlling that raw sequence live in their own configs.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from typing import Any, ClassVar, Final

__all__ = [
    "SimConfig",
    "GeneratorConfig",
    "BrownianConfig",
    "FBMConfig",
    "SBMConfig",
    "CTRWConfig",
    "LevyFlightConfig",
    "LevyWalkConfig",
    "DDMConfig",
    "MeanFieldConfig",
    "FitConfig",
    "CONFIG_REGISTRY",
    "config_to_dict",
    "config_from_dict",
    "config_to_json",
    "config_from_json",
]

#: Time step of the uniform grid.  Fixed by convention, not a free parameter.
DT: Final[float] = 1.0


# --------------------------------------------------------------------------
# base
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SimConfig:
    """Parameters shared by every simulation.

    Parameters
    ----------
    n_particles : int
        Number of independent trajectories in the ensemble.
    n_steps : int
        Number of time points per trajectory, including the initial condition
        at ``t = 0``.  A trajectory therefore spans ``t in [0, n_steps - 1]``
        in units of ``dt = 1``.
    n_dim : int
        Spatial dimension, 1 or 2 (``PROJECT.md`` section 2).
    seed : int
        Seed for the ``numpy.random.Generator`` handed to the generator.  No
        stochastic function in this package touches the global RNG.
    """

    n_particles: int = 1000
    n_steps: int = 1024
    n_dim: int = 1
    seed: int = 0

    #: Stable tag used for JSON round-trip and for the generator registry.
    name: ClassVar[str] = "sim"

    def __post_init__(self) -> None:
        if self.n_particles < 1:
            raise ValueError(f"n_particles must be >= 1, got {self.n_particles}")
        if self.n_steps < 2:
            raise ValueError(f"n_steps must be >= 2, got {self.n_steps}")
        if self.n_dim not in (1, 2):
            raise ValueError(f"n_dim must be 1 or 2, got {self.n_dim}")

    @property
    def shape(self) -> tuple[int, int, int]:
        """Shape of the trajectory array this config produces.

        Returns
        -------
        tuple of int
            ``(n_particles, n_steps, n_dim)``, the only array layout allowed
            to escape a function.
        """
        return (self.n_particles, self.n_steps, self.n_dim)

    @property
    def duration(self) -> float:
        """Physical duration ``(n_steps - 1) * dt`` of one trajectory."""
        return (self.n_steps - 1) * DT


@dataclass(frozen=True)
class GeneratorConfig(SimConfig):
    """Base class for the per-generator configs.

    Subclasses must override :attr:`alpha_analytic` with the ensemble-MSD
    exponent their mechanism is expected to reproduce, so validation tests can
    read the target off the config rather than hard-coding it.
    """

    @property
    def alpha_analytic(self) -> float:
        """Analytic exponent ``alpha`` in ``<x^2(t)> ~ t^alpha``."""
        raise NotImplementedError

    @property
    def msd_is_finite(self) -> bool:
        """Whether the second moment of the propagator exists at all.

        ``False`` for Levy flights, whose MSD diverges; code paths computing an
        MSD must raise or warn instead of returning a finite number from a
        diverging quantity (``PROJECT.md`` section 4).
        """
        return True


# --------------------------------------------------------------------------
# one config per generator
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class BrownianConfig(GeneratorConfig):
    """Ordinary Brownian motion.

    Ensemble MSD ``<x^2(t)> = 2 d D t``, so ``alpha = 1``.

    Parameters
    ----------
    diffusivity : float
        ``D``, the diffusion coefficient, in squared length per unit time.
    """

    diffusivity: float = 1.0

    name: ClassVar[str] = "brownian"

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.diffusivity <= 0.0:
            raise ValueError(f"diffusivity must be > 0, got {self.diffusivity}")

    @property
    def alpha_analytic(self) -> float:
        return 1.0


@dataclass(frozen=True)
class FBMConfig(GeneratorConfig):
    """Fractional Brownian motion, generated by Davies-Harte embedding.

    Ensemble MSD ``<x^2(t)> = 2 d D t^(2H)``, so ``alpha = 2H``.  ``H < 1/2``
    is subdiffusive with anticorrelated increments, ``H > 1/2`` superdiffusive
    with positively correlated increments, ``H = 1/2`` is Brownian.

    Parameters
    ----------
    hurst : float
        Hurst exponent ``H`` in ``(0, 1)``.
    diffusivity : float
        Generalised diffusivity ``D``, in squared length per ``time^(2H)``.
    """

    hurst: float = 0.3
    diffusivity: float = 1.0

    name: ClassVar[str] = "fbm"

    def __post_init__(self) -> None:
        super().__post_init__()
        if not 0.0 < self.hurst < 1.0:
            raise ValueError(f"hurst must lie in (0, 1), got {self.hurst}")
        if self.diffusivity <= 0.0:
            raise ValueError(f"diffusivity must be > 0, got {self.diffusivity}")

    @property
    def alpha_analytic(self) -> float:
        return 2.0 * self.hurst


@dataclass(frozen=True)
class SBMConfig(GeneratorConfig):
    """Scaled Brownian motion: Brownian motion with a time-dependent diffusivity.

    With ``D(t) = alpha * D0 * t^(alpha - 1)`` the ensemble MSD is
    ``<x^2(t)> = 2 d D0 t^alpha``.  The process is Gaussian and Markovian but
    non-stationary, hence non-ergodic.

    Parameters
    ----------
    alpha : float
        Anomalous exponent ``alpha`` in ``(0, 2)``.  ``alpha < 1`` subdiffusive,
        ``alpha > 1`` superdiffusive.
    diffusivity : float
        Prefactor ``D0``, in squared length per ``time^alpha``.
    """

    alpha: float = 0.5
    diffusivity: float = 1.0

    name: ClassVar[str] = "sbm"

    def __post_init__(self) -> None:
        super().__post_init__()
        if not 0.0 < self.alpha < 2.0:
            raise ValueError(f"alpha must lie in (0, 2), got {self.alpha}")
        if self.diffusivity <= 0.0:
            raise ValueError(f"diffusivity must be > 0, got {self.diffusivity}")

    @property
    def alpha_analytic(self) -> float:
        return self.alpha


@dataclass(frozen=True)
class CTRWConfig(GeneratorConfig):
    """Continuous-time random walk with heavy-tailed waiting times.

    Waiting times are drawn from a Pareto tail
    ``psi(tau) ~ (a / tau0) (tau / tau0)^(-1-a)`` for ``tau >= tau0`` with
    ``a in (0, 1)``, so the mean waiting time diverges and the ensemble MSD
    scales subdiffusively as ``<x^2(t)> ~ t^a``.  Jumps are Gaussian with
    finite variance.

    The process is weakly non-ergodic: the single-trajectory TA-MSD grows
    *linearly* in the lag whatever ``a`` is, with a random amplitude that never
    settles (He et al., PRL 101, 058101).

    Parameters
    ----------
    alpha_wait : float
        Waiting-time tail exponent ``a`` in ``(0, 1)``.
    tau0 : float
        Waiting-time scale, the lower cut-off of the Pareto tail.
    jump_scale : float
        Standard deviation of one jump, per spatial component.
    jump_distribution : str
        Jump-length law.  Only ``"gaussian"`` is implemented, and that is
        deliberate rather than provisional (``PROJECT.md`` section 4): finite
        jump variance keeps the CTRW a *pure waiting-time* mechanism, which is
        what makes it a clean contrast against fBm at ``H < 1/2`` in Q2.
        Heavy-tailed jumps would blur that contrast into the Levy walk.  The
        parameter exists so the choice is recorded in every cached config and
        so a future variant does not silently change the meaning of old caches.
    aging_time : float
        Age ``t_a`` of the process when observation begins, in units of ``dt``.
        Zero is the ordinary, non-aged CTRW.  A non-zero value means the walker
        started at ``-t_a`` and is first observed at ``t = 0`` of the grid,
        which is the situation of any observer who begins watching an agent at
        an arbitrary moment -- so it is load-bearing for the research question,
        not a refinement.  Aging changes the ensemble MSD, the time-averaged
        MSD and the EB plateau.  Phase 1 validates ``t_a = 0`` only.
    oversample : float
        Safety factor on the number of raw events generated before resampling
        onto the uniform grid.  The event sequence must cover the full grid
        duration for every particle; a heavy tail makes the required event
        count a random variable, so generation is done in blocks until every
        particle's clock passes the end of the grid, starting from
        ``oversample`` times the naive estimate.
    """

    alpha_wait: float = 0.7
    tau0: float = 1.0
    jump_scale: float = 1.0
    jump_distribution: str = "gaussian"
    aging_time: float = 0.0
    oversample: float = 2.0

    name: ClassVar[str] = "ctrw"

    #: Jump laws this generator knows how to draw.
    JUMP_DISTRIBUTIONS: ClassVar[tuple[str, ...]] = ("gaussian",)

    def __post_init__(self) -> None:
        super().__post_init__()
        if not 0.0 < self.alpha_wait < 1.0:
            raise ValueError(f"alpha_wait must lie in (0, 1), got {self.alpha_wait}")
        if self.tau0 <= 0.0:
            raise ValueError(f"tau0 must be > 0, got {self.tau0}")
        if self.jump_scale <= 0.0:
            raise ValueError(f"jump_scale must be > 0, got {self.jump_scale}")
        if self.jump_distribution not in self.JUMP_DISTRIBUTIONS:
            raise ValueError(
                f"jump_distribution must be one of {self.JUMP_DISTRIBUTIONS}, "
                f"got {self.jump_distribution!r}"
            )
        if self.aging_time < 0.0:
            raise ValueError(f"aging_time must be >= 0, got {self.aging_time}")
        if self.oversample < 1.0:
            raise ValueError(f"oversample must be >= 1, got {self.oversample}")

    @property
    def alpha_analytic(self) -> float:
        return self.alpha_wait

    @property
    def is_aged(self) -> bool:
        """Whether observation starts after a non-zero age ``t_a``."""
        return self.aging_time > 0.0

    @property
    def total_duration(self) -> float:
        """Process time that must be covered: the age plus the observed window."""
        return self.aging_time + self.duration


@dataclass(frozen=True)
class LevyFlightConfig(GeneratorConfig):
    """Levy flight: instantaneous jumps drawn from a symmetric stable law.

    Increments follow a symmetric ``s``-stable distribution with ``s in (0, 2)``.
    The second moment *diverges*, so the MSD is meaningless and
    :attr:`msd_is_finite` is ``False``.  Validation uses a fractional moment
    ``<|x|^q> ~ t^(q/s)`` with ``q < s`` instead.

    Parameters
    ----------
    stability : float
        Stability index ``s`` in ``(0, 2)``; ``s = 2`` would be Gaussian and is
        excluded.
    scale : float
        Scale parameter of the stable law, per spatial component.
    """

    stability: float = 1.5
    scale: float = 1.0

    name: ClassVar[str] = "levy_flight"

    def __post_init__(self) -> None:
        super().__post_init__()
        if not 0.0 < self.stability < 2.0:
            raise ValueError(f"stability must lie in (0, 2), got {self.stability}")
        if self.scale <= 0.0:
            raise ValueError(f"scale must be > 0, got {self.scale}")

    @property
    def alpha_analytic(self) -> float:
        """Undefined: the second moment of a Levy flight diverges."""
        raise ValueError(
            "the MSD of a Levy flight diverges; there is no alpha to compare "
            "against. Validate a fractional moment of order q < stability "
            "instead (PROJECT.md section 4)."
        )

    @property
    def msd_is_finite(self) -> bool:
        return False

    @property
    def nu_analytic(self) -> float:
        """Fractional-moment exponent: ``<|x|^q> ~ t^(q * nu)`` with ``nu = 1/s``."""
        return 1.0 / self.stability


@dataclass(frozen=True)
class LevyWalkConfig(GeneratorConfig):
    """Levy walk: ballistic flights of heavy-tailed duration at finite speed.

    Flight durations follow a Pareto tail with exponent ``g in (1, 2)``, each
    flight is travelled at constant speed ``v`` in a random direction, so
    displacement stays coupled to elapsed time and the MSD is finite:
    ``<x^2(t)> ~ t^(3 - g)``, superdiffusive.  The moment spectrum ``nu(q)`` is
    piecewise linear with a kink -- the strong-anomalous fingerprint that
    separates a Levy walk from fBm at equal ``alpha``.

    Parameters
    ----------
    gamma : float
        Flight-duration tail exponent ``g`` in ``(1, 2)``.
    tau0 : float
        Lower cut-off of the flight-duration tail.
    speed : float
        Constant speed ``v`` during a flight.
    oversample : float
        Safety factor on the raw event count before resampling, as in
        :class:`CTRWConfig`.
    """

    gamma: float = 1.5
    tau0: float = 1.0
    speed: float = 1.0
    oversample: float = 2.0

    name: ClassVar[str] = "levy_walk"

    def __post_init__(self) -> None:
        super().__post_init__()
        if not 1.0 < self.gamma < 2.0:
            raise ValueError(f"gamma must lie in (1, 2), got {self.gamma}")
        if self.tau0 <= 0.0:
            raise ValueError(f"tau0 must be > 0, got {self.tau0}")
        if self.speed <= 0.0:
            raise ValueError(f"speed must be > 0, got {self.speed}")
        if self.oversample < 1.0:
            raise ValueError(f"oversample must be >= 1, got {self.oversample}")

    @property
    def alpha_analytic(self) -> float:
        return 3.0 - self.gamma


@dataclass(frozen=True)
class DDMConfig(GeneratorConfig):
    """Diffusing diffusivity: Brownian yet non-Gaussian.

    The diffusivity itself is a stochastic process,
    ``D(t) = Y(t) . Y(t)`` with ``Y`` an ``n_aux``-component Ornstein-Uhlenbeck
    process ``dY = -Y dt / tau + sigma dW``.  The MSD stays linear
    (``alpha = 1``) but the propagator has exponential tails at short times and
    crosses over to Gaussian beyond the correlation time ``tau``
    (Chechkin et al., PRX 7, 021002).

    Parameters
    ----------
    tau : float
        Correlation time of the diffusivity, in units of ``dt``.
    sigma : float
        Noise amplitude of the auxiliary OU process.
    n_aux : int
        Number of OU components entering ``D = |Y|^2``.  ``n_aux = 1`` gives the
        strongest non-Gaussianity; large ``n_aux`` averages it away.
    """

    tau: float = 30.0
    sigma: float = 1.0
    n_aux: int = 1

    name: ClassVar[str] = "ddm"

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.tau <= 0.0:
            raise ValueError(f"tau must be > 0, got {self.tau}")
        if self.sigma <= 0.0:
            raise ValueError(f"sigma must be > 0, got {self.sigma}")
        if self.n_aux < 1:
            raise ValueError(f"n_aux must be >= 1, got {self.n_aux}")

    @property
    def alpha_analytic(self) -> float:
        return 1.0


@dataclass(frozen=True)
class MeanFieldConfig(GeneratorConfig):
    """Truncated ARFIMA(0,d,0) velocity agents under consensus mean-field coupling.

    ``PROJECT.md`` section 7: each agent's velocity follows the truncated
    AR(inf) representation of ARFIMA(0,d,0), plus a consensus pull toward the
    instantaneous mean field,

    .. math::

        v_i(t+1) = \\sum_{s=1}^{K} \\kappa(s)\\, v_i(t+1-s)
                 + J \\left(\\langle v \\rangle(t) - v_i(t)\\right)
                 + \\sigma\\, \\eta_i(t+1) ,

    with :math:`\\kappa(s) = -c_s` the fractional-differencing weights of
    :math:`(1-B)^d` and :math:`\\eta_i` unit Gaussian white noise. At ``J = 0``
    the consensus term vanishes and this reduces, up to truncation at lag
    ``K``, exactly to ARFIMA(0,d,0) velocity -- **not** fractional Gaussian
    noise, a distinction that matters at short lag even though the two share
    the asymptotic exponent ``alpha = 1 + 2d``. ARFIMA's closed-form
    autocorrelation is ``rho(k) = Gamma(1-d)Gamma(k+d) / (Gamma(d)Gamma(k+1-d))``
    (:func:`~collectivediff.dynamics.meanfield.arfima_acf`), giving
    ``rho(1) = d / (1-d)``, against fGn's ``2^{2d} - 1`` -- different numbers
    for any ``d != 0``. All-to-all consensus coupling conserves the population
    mean exactly (the ``J`` term cancels in the average), so ``<v>(t)`` is
    ARFIMA(0,d,0) with variance ``sigma^2 / N`` for *every* ``J`` -- coupling
    changes each agent's own regime, never the coherent one.

    Parameters
    ----------
    d : float
        Fractional differencing parameter in ``(-0.5, 0.5)``;
        ``alpha = 1 + 2d`` is the coherent-mode exponent, at every ``J``.
    J : float
        Consensus coupling strength, ``>= 0``. ``J = 0`` reduces to
        independent ARFIMA(0,d,0) agents.
    K : int
        Memory truncation length. Every claim about the coupled model is
        restricted to lags ``<= K / 10`` (``PHASE3_PROMPT.md``).
    sigma : float
        Innovation noise standard deviation.
    burn_in : int
        Update steps discarded before ``t = 0`` so the ring buffer's memory is
        fully populated by the time recording starts. Must be at least ``K``.
    """

    d: float = 0.0
    J: float = 0.0
    K: int = 1024
    sigma: float = 1.0
    burn_in: int = 4096

    name: ClassVar[str] = "meanfield"

    def __post_init__(self) -> None:
        super().__post_init__()
        if not -0.5 < self.d < 0.5:
            raise ValueError(f"d must lie in (-0.5, 0.5), got {self.d}")
        if self.J < 0.0:
            raise ValueError(f"J must be >= 0, got {self.J}")
        if self.K < 1:
            raise ValueError(f"K must be >= 1, got {self.K}")
        if self.sigma <= 0.0:
            raise ValueError(f"sigma must be > 0, got {self.sigma}")
        if self.burn_in < self.K:
            raise ValueError(
                f"burn_in must be >= K ({self.K}) so the memory is fully "
                f"populated at t = 0, got {self.burn_in}"
            )
        if self.n_dim != 1:
            raise ValueError(
                "MeanFieldConfig is 1-D only; a 2-D field is two independent "
                "1-D simulations, one per component (PROJECT.md section 7)"
            )

    @property
    def hurst(self) -> float:
        """Asymptotic Hurst-equivalent exponent, ``H = d + 1/2``.

        ARFIMA(0,d,0) is not fBm, but shares its asymptotic self-similarity
        exponent; this is that shared value, useful for reading off the
        regime (sub/normal/superdiffusive), not a claim of exact equivalence.
        """
        return self.d + 0.5

    @property
    def alpha_analytic(self) -> float:
        """Coherent-mode exponent ``1 + 2d`` -- holds at every ``J``."""
        return 1.0 + 2.0 * self.d


# --------------------------------------------------------------------------
# estimator-side config
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class FitConfig:
    """Lag window, weighting and bootstrap settings for a power-law fit.

    Two distinct windows are described by ``PROJECT.md`` section 2 and must not
    be conflated.  ``[lag_min, lag_max_fraction * T]`` is the *TA-MSD versus
    lag* window: beyond it the number of contributing windows in a time average
    collapses and the estimate is noise.  The *EA-MSD versus time* window is a
    different claim -- every particle contributes at every ``t``, so it asserts
    where the asymptotics hold -- and is stated per call site, not here.

    Uncertainty is estimated by bootstrap over particles, never from the fit
    residuals.  EA-MSD points at different times are built from the same
    particles and are strongly correlated, so ordinary least-squares standard
    errors understate the exponent uncertainty by more than an order of
    magnitude.  The same resampling supplies the empirical lag-lag covariance
    used for generalised least squares, so one mechanism solves both the
    confidence-interval problem and the weighting problem.

    Parameters
    ----------
    lag_min : int
        Smallest lag included in a TA-MSD fit, in steps.
    lag_max_fraction : float
        Largest lag as a fraction of the trajectory length ``T``; the project
        convention is ``T / 10``.
    n_lags : int
        Number of lags sampled log-uniformly inside the window.
    n_bootstrap : int
        Number of particle resamples used for the confidence interval and the
        empirical covariance.  200 is enough for a standard error; a percentile
        interval on the tails wants more.  Zero disables bootstrapping, which
        is only appropriate for a noiseless synthetic curve.
    bootstrap_seed : int
        Seed for the resampling RNG, so an error bar is reproducible from the
        config exactly like a trajectory is.
    use_gls : bool
        Whether to reweight the fit by the inverse of the bootstrap variance
        per point.  Empirical weights are preferred over an analytic form
        because the analytic one (``t^-(g-1)`` for the Levy walk) would require
        iterating on the very exponent being estimated.
    """

    lag_min: int = 1
    lag_max_fraction: float = 0.1
    n_lags: int = 32
    n_bootstrap: int = 200
    bootstrap_seed: int = 0
    use_gls: bool = True

    name: ClassVar[str] = "fit"

    def __post_init__(self) -> None:
        if self.lag_min < 1:
            raise ValueError(f"lag_min must be >= 1, got {self.lag_min}")
        if not 0.0 < self.lag_max_fraction <= 1.0:
            raise ValueError(
                f"lag_max_fraction must lie in (0, 1], got {self.lag_max_fraction}"
            )
        if self.n_lags < 3:
            raise ValueError(f"n_lags must be >= 3 for a fit, got {self.n_lags}")
        if self.n_bootstrap < 0:
            raise ValueError(f"n_bootstrap must be >= 0, got {self.n_bootstrap}")


# --------------------------------------------------------------------------
# serialisation
# --------------------------------------------------------------------------

#: Every config class that can be recovered from JSON, keyed by ``name``.
CONFIG_REGISTRY: Final[dict[str, type]] = {
    cls.name: cls
    for cls in (
        SimConfig,
        BrownianConfig,
        FBMConfig,
        SBMConfig,
        CTRWConfig,
        LevyFlightConfig,
        LevyWalkConfig,
        DDMConfig,
        MeanFieldConfig,
        FitConfig,
    )
}


def config_to_dict(cfg: Any) -> dict[str, Any]:
    """Convert a config to a plain JSON-ready dict.

    The class tag is stored under the ``"name"`` key so the dict round-trips
    through :func:`config_from_dict`.

    Parameters
    ----------
    cfg : dataclass instance
        Any frozen config defined in this module.

    Returns
    -------
    dict
        Field values plus ``{"name": cfg.name}``.
    """
    data = asdict(cfg)
    data["name"] = type(cfg).name
    return data


def config_from_dict(data: dict[str, Any]) -> Any:
    """Rebuild a config from the dict produced by :func:`config_to_dict`.

    Parameters
    ----------
    data : dict
        Must contain a ``"name"`` key present in :data:`CONFIG_REGISTRY`.

    Returns
    -------
    dataclass instance
        The reconstructed config.

    Raises
    ------
    KeyError
        If ``name`` is missing or unknown.
    """
    payload = dict(data)
    try:
        tag = payload.pop("name")
    except KeyError as exc:
        raise KeyError("config dict has no 'name' tag") from exc
    if tag not in CONFIG_REGISTRY:
        raise KeyError(f"unknown config tag {tag!r}; known: {sorted(CONFIG_REGISTRY)}")
    cls = CONFIG_REGISTRY[tag]
    known = {f.name for f in fields(cls)}
    unknown = set(payload) - known
    if unknown:
        raise KeyError(f"unknown fields for {tag!r}: {sorted(unknown)}")
    return cls(**payload)


def config_to_json(cfg: Any) -> str:
    """Serialise a config to canonical JSON.

    Keys are sorted and separators fixed so that the same config always yields
    byte-identical JSON -- this string is what the cache hashes.

    Parameters
    ----------
    cfg : dataclass instance
        Any frozen config defined in this module.

    Returns
    -------
    str
        Canonical JSON representation.
    """
    return json.dumps(config_to_dict(cfg), sort_keys=True, separators=(",", ":"))


def config_from_json(text: str) -> Any:
    """Inverse of :func:`config_to_json`.

    Parameters
    ----------
    text : str
        JSON produced by :func:`config_to_json`.

    Returns
    -------
    dataclass instance
        The reconstructed config.
    """
    return config_from_dict(json.loads(text))
