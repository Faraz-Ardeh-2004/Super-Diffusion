"""Coupled agent dynamics -- phase 3.

Modular dynamics (phase 4) are still deliberately absent: ``PROJECT.md``
section 6 gates phase 4 on phase 3's tests passing first.
"""

from __future__ import annotations

from .meanfield import (
    MeanFieldTrajectories,
    arfima_acf,
    arfima_effective_exponent,
    arfima_exact_msd,
    coupling_for_target_crossover,
    deviation_spectrum,
    fractional_diff_kappa,
    simulate_meanfield,
    truncated_arfima_acf,
    whittle_d_only_estimate,
    whittle_dj_estimate,
    whittle_objective,
)

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
