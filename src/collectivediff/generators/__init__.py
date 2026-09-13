"""Trajectory generators and the ``name -> callable`` registry.

Every mechanism of ``PROJECT.md`` section 4 is reachable by name, so a sweep can
iterate over mechanisms without importing each module by hand:

>>> import numpy as np
>>> from collectivediff.config import FBMConfig
>>> from collectivediff.generators import generate
>>> x = generate(FBMConfig(n_particles=4, n_steps=64, hurst=0.7),
...              np.random.default_rng(0))
>>> x.shape
(4, 64, 1)

Dispatch is on the *config type*, not on a free-floating string: a config
carries its own parameters, so ``generate(cfg, rng)`` cannot be called with a
mechanism and a mismatched parameter set.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import NDArray

from ..config import (
    BrownianConfig,
    CTRWConfig,
    DDMConfig,
    FBMConfig,
    GeneratorConfig,
    LevyFlightConfig,
    LevyWalkConfig,
    SBMConfig,
)
from .brownian import brownian
from .ctrw import ctrw, ctrw_events
from .ddm import ddm, ou_process
from .fbm import fbm, fgn_autocovariance, fractional_gaussian_noise
from .levy import levy_flight, levy_walk, levy_walk_events, symmetric_stable
from .resampling import resample_linear, resample_step
from .sbm import sbm

__all__ = [
    "GENERATORS",
    "CONFIG_FOR",
    "generate",
    "brownian",
    "fbm",
    "sbm",
    "ctrw",
    "levy_flight",
    "levy_walk",
    "ddm",
    "ctrw_events",
    "levy_walk_events",
    "ou_process",
    "fgn_autocovariance",
    "fractional_gaussian_noise",
    "symmetric_stable",
    "resample_step",
    "resample_linear",
]

#: Registry: generator name -> callable ``(cfg, rng) -> trajectories``.
GENERATORS: dict[str, Callable[..., NDArray[np.float64]]] = {
    BrownianConfig.name: brownian,
    FBMConfig.name: fbm,
    SBMConfig.name: sbm,
    CTRWConfig.name: ctrw,
    LevyFlightConfig.name: levy_flight,
    LevyWalkConfig.name: levy_walk,
    DDMConfig.name: ddm,
}

#: Inverse lookup: generator name -> the config class it consumes.
CONFIG_FOR: dict[str, type[GeneratorConfig]] = {
    BrownianConfig.name: BrownianConfig,
    FBMConfig.name: FBMConfig,
    SBMConfig.name: SBMConfig,
    CTRWConfig.name: CTRWConfig,
    LevyFlightConfig.name: LevyFlightConfig,
    LevyWalkConfig.name: LevyWalkConfig,
    DDMConfig.name: DDMConfig,
}


def generate(cfg: GeneratorConfig, rng: np.random.Generator) -> NDArray[np.float64]:
    """Run the generator that belongs to a config.

    Parameters
    ----------
    cfg : GeneratorConfig
        Any generator config; the mechanism is its type.
    rng : numpy.random.Generator
        Explicit generator, as required by ``PROJECT.md`` section 2.

    Returns
    -------
    ndarray of float64
        Shape ``(n_particles, n_steps, n_dim)``.

    Raises
    ------
    KeyError
        If the config's ``name`` has no registered generator.
    """
    tag = type(cfg).name
    try:
        generator = GENERATORS[tag]
    except KeyError as exc:
        raise KeyError(
            f"no generator registered for {tag!r}; known: {sorted(GENERATORS)}"
        ) from exc
    return generator(cfg, rng)
