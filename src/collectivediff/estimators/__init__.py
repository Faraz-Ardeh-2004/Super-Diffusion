"""Estimator toolbox for ``PROJECT.md`` section 5.

Modules
-------
msd
    EA-MSD, TA-MSD, EA-TA-MSD, fractional moments, power-law fitting.
ergodicity
    Ergodicity-breaking parameter and the amplitude scatter of
    ``xi = TA-MSD / <TA-MSD>``.
distributions
    van Hove displacement distribution and the non-Gaussian parameter.
correlations
    Increment and velocity autocorrelation.
moments
    Moment scaling spectrum ``nu(q)``.
firstpassage
    First-passage times to a threshold and return-to-origin statistics.
"""

from __future__ import annotations

from .correlations import increment_acf, velocity_acf
from .distributions import (
    gaussian_reference,
    non_gaussian_parameter,
    van_hove,
)
from .ergodicity import (
    amplitude_scatter,
    eb_parameter,
    ergodicity_breaking_curve,
)
from .firstpassage import (
    first_passage_times,
    return_statistics,
    survival_probability,
)
from .moments import MomentSpectrum, moment_spectrum
from .msd import (
    DivergentMomentError,
    PowerLawFit,
    check_layout,
    ea_msd,
    ea_ta_msd,
    fit_powerlaw,
    fractional_moment,
    log_spaced_lags,
    squared_displacement,
    ta_msd,
)

__all__ = [
    # msd
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
    # ergodicity
    "amplitude_scatter",
    "eb_parameter",
    "ergodicity_breaking_curve",
    # distributions
    "gaussian_reference",
    "non_gaussian_parameter",
    "van_hove",
    # correlations
    "increment_acf",
    "velocity_acf",
    # moments
    "MomentSpectrum",
    "moment_spectrum",
    # first passage
    "first_passage_times",
    "return_statistics",
    "survival_probability",
]
