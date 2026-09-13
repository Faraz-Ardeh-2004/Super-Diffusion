"""Analytic reference MSDs and tolerance checks.

The validation-first rule of ``PROJECT.md`` has the highest priority in the
project: a generator is not usable until a test asserts that it reproduces its
analytic ensemble MSD within a stated tolerance over a stated window.
"""

from __future__ import annotations

from .reference import (
    analytic_msd,
    ctrw_finite_time_msd,
    ctrw_msd_amplitude,
    ctrw_predicted_bias,
    ctrw_renewal_count,
    has_exact_msd,
    mean_diffusivity,
)

__all__ = [
    "analytic_msd",
    "ctrw_finite_time_msd",
    "ctrw_msd_amplitude",
    "ctrw_predicted_bias",
    "ctrw_renewal_count",
    "has_exact_msd",
    "mean_diffusivity",
]
