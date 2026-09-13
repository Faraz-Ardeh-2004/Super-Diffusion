"""Displacement distributions: the van Hove function and its non-Gaussianity.

An exponent says how fast a walker spreads; the *shape* of the propagator says
how. This module supplies the observable that separates mechanisms sharing an
exponent -- most sharply diffusing diffusivity, which is exactly Brownian in its
MSD (``alpha = 1``) and yet has exponential tails at short times.

``PROJECT.md`` section 5 defines the non-Gaussian parameter in ``d`` dimensions:

.. math::

    a_2(t) = \\frac{\\langle r^4 \\rangle}{(1 + 2/d)\\,\\langle r^2 \\rangle^2} - 1 ,

which vanishes for a Gaussian propagator in any dimension and is positive for a
heavy-tailed one.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from .msd import DivergentMomentError, check_layout

__all__ = [
    "gaussian_reference",
    "non_gaussian_parameter",
    "van_hove",
]


def _displacement(trajectories: NDArray[Any], t_index: int) -> NDArray[np.float64]:
    """Displacement of every particle from its start, at one time index."""
    check_layout(trajectories)
    n_steps = trajectories.shape[1]
    if not 0 <= t_index < n_steps:
        raise ValueError(f"t_index must lie in [0, {n_steps - 1}], got {t_index}")
    return np.asarray(
        trajectories[:, t_index, :] - trajectories[:, 0, :], dtype=np.float64
    )


def van_hove(
    trajectories: NDArray[Any],
    t_index: int,
    bins: int | NDArray[np.float64] = 101,
    component: int = 0,
    normalise: bool = True,
    rescale: bool = False,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Self part of the van Hove function: the displacement distribution.

    .. math::

        G_s(x, t) = \\big\\langle \\delta\\big(x - [x_i(t) - x_i(0)]\\big)
        \\big\\rangle

    estimated as a histogram over particles of a single Cartesian component.
    ``PROJECT.md`` section 2 runs all statistics in one dimension, so a single
    component is the default rather than the radial distance -- the Gaussian
    reference is then a plain Gaussian, with no Jacobian factor to confuse the
    comparison.

    Parameters
    ----------
    trajectories : ndarray
        Shape ``(n_particles, n_steps, n_dim)``.
    t_index : int
        Time index at which to take the distribution.
    bins : int or ndarray
        Bin count or explicit bin edges, as for ``numpy.histogram``.
    component : int
        Which spatial component to histogram.
    normalise : bool
        Return a density rather than raw counts.
    rescale : bool
        Divide displacements by their own standard deviation before
        histogramming. This is how distributions at different times are put on
        one axis: a self-similar Gaussian process collapses onto a single curve,
        and any failure to collapse is itself the signal.

    Returns
    -------
    centres : ndarray of float64
        Bin centres, shape ``(n_bins,)``.
    density : ndarray of float64
        Histogram values, shape ``(n_bins,)``.
    """
    displacement = _displacement(trajectories, t_index)[:, component]
    if rescale:
        spread = float(np.std(displacement))
        if spread <= 0.0:
            raise ValueError(
                f"displacements at t_index={t_index} have zero spread; cannot rescale"
            )
        displacement = displacement / spread
    density, edges = np.histogram(displacement, bins=bins, density=normalise)
    centres = 0.5 * (edges[:-1] + edges[1:])
    return centres, density.astype(np.float64)


def gaussian_reference(
    centres: NDArray[np.float64],
    variance: float = 1.0,
) -> NDArray[np.float64]:
    """Gaussian density with the given variance, for overlaying on a van Hove plot.

    .. math::

        G(x) = \\frac{1}{\\sqrt{2\\pi\\sigma^2}}\\,e^{-x^2/(2\\sigma^2)}

    Parameters
    ----------
    centres : ndarray of float64
        Abscissa, normally the bin centres returned by :func:`van_hove`.
    variance : float
        ``sigma^2``. Use 1.0 together with ``rescale=True``.

    Returns
    -------
    ndarray of float64
        The density at ``centres``.
    """
    if variance <= 0.0:
        raise ValueError(f"variance must be > 0, got {variance}")
    return np.exp(-(centres**2) / (2.0 * variance)) / np.sqrt(2.0 * np.pi * variance)


def non_gaussian_parameter(
    trajectories: NDArray[Any],
    cfg: Any | None = None,
) -> NDArray[np.float64]:
    """Non-Gaussian parameter ``a_2(t)`` of the displacement distribution.

    .. math::

        a_2(t) = \\frac{\\langle r^4 \\rangle}
                       {(1 + 2/d)\\,\\langle r^2 \\rangle^2} - 1

    Zero for a Gaussian propagator in any dimension; positive for heavy tails.
    In one dimension the factor is ``1 + 2/1 = 3``, recovering the familiar
    excess kurtosis normalised by three.

    Parameters
    ----------
    trajectories : ndarray
        Shape ``(n_particles, n_steps, n_dim)``.
    cfg : GeneratorConfig, optional
        If it reports ``msd_is_finite = False`` this raises: ``a_2`` is built
        from the second and fourth moments, and for a Levy flight neither
        exists.

    Returns
    -------
    ndarray of float64
        Shape ``(n_steps,)``. Element 0 is ``nan``: at ``t = 0`` every
        displacement is exactly zero and the ratio is 0/0.
    """
    check_layout(trajectories)
    if cfg is not None and not cfg.msd_is_finite:
        raise DivergentMomentError(
            f"the second and fourth moments of {type(cfg).name!r} both diverge, "
            "so the non-Gaussian parameter is not defined; compare the van Hove "
            "distribution against the stable law instead"
        )
    n_dim = trajectories.shape[2]
    displacement = trajectories - trajectories[:, :1, :]
    squared = np.sum(displacement * displacement, axis=2, dtype=np.float64)

    second = np.mean(squared, axis=0, dtype=np.float64)
    fourth = np.mean(squared**2, axis=0, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = fourth / ((1.0 + 2.0 / n_dim) * second**2) - 1.0
    return np.where(second > 0.0, ratio, np.nan)
