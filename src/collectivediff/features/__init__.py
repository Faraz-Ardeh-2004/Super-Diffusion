"""Feature extraction module for single-trajectory isolated observers.

Provides scale-invariant feature extractors and spectral estimation tools.
"""

from __future__ import annotations

from .extraction import (
    FEATURE_GROUPS,
    FEATURE_NAMES,
    extract_feature_dict,
    extract_feature_matrix,
    extract_features,
)
from .spectral import (
    cramer_rao_bound_fbm,
    fgn_spectral_density,
    fisher_information_fbm,
    whittle_log_likelihood,
    whittle_mle_hurst,
)

__all__ = [
    "FEATURE_NAMES",
    "FEATURE_GROUPS",
    "extract_features",
    "extract_feature_dict",
    "extract_feature_matrix",
    "fgn_spectral_density",
    "whittle_log_likelihood",
    "whittle_mle_hurst",
    "fisher_information_fbm",
    "cramer_rao_bound_fbm",
]
