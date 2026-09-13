"""Inferring collective diffusion regimes from partial agent observation.

See ``PROJECT.md`` at the repository root for the research question, the
conventions every module must follow, and the phase roadmap.

Phase 1 exposes the configuration layer and the cache; generators, estimators
and visualisation are added in the steps that follow.
"""

from __future__ import annotations

from . import cache, config
from .config import (
    DT,
    BrownianConfig,
    CTRWConfig,
    DDMConfig,
    FBMConfig,
    FitConfig,
    GeneratorConfig,
    LevyFlightConfig,
    LevyWalkConfig,
    SBMConfig,
    SimConfig,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "DT",
    "cache",
    "config",
    "SimConfig",
    "GeneratorConfig",
    "BrownianConfig",
    "FBMConfig",
    "SBMConfig",
    "CTRWConfig",
    "LevyFlightConfig",
    "LevyWalkConfig",
    "DDMConfig",
    "FitConfig",
]
