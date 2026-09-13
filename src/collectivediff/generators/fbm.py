"""Fractional Brownian motion by exact Davies-Harte circulant embedding.

The method is exact: it draws from the true multivariate Gaussian law of
fractional Gaussian noise, not from an approximation of it. That matters here
because fBm is the project's only mechanism that is simultaneously anomalous,
Gaussian and ergodic, so it is the control against which the *non*-ergodic
mechanisms are compared. An ad-hoc filter or a truncated Cholesky would leave
correlation error that is indistinguishable from the effects being measured.

Reference: Davies & Harte, Biometrika 74, 95 (1987); Wood & Chan, J. Comput.
Graph. Stat. 3, 409 (1994); Dieker, "Simulation of fractional Brownian motion"
(2004), chapter 2.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..config import FBMConfig
from ._common import as_trajectories, require_generator

__all__ = [
    "fbm",
    "circulant_eigenvalues",
    "fgn_autocovariance",
    "fractional_gaussian_noise",
]

#: Relative tolerance on negative circulant eigenvalues. Anything more negative
#: than ``-EIGENVALUE_TOL * max(lambda)`` is a genuine failure of the embedding,
#: not floating-point noise, and is raised rather than clipped.
EIGENVALUE_TOL: float = 1e-10


def fgn_autocovariance(n_lags: int, hurst: float) -> NDArray[np.float64]:
    """Autocovariance of unit-variance fractional Gaussian noise.

    For increments of unit-variance fBm with Hurst exponent ``H``,

    .. math::

        \\gamma(k) = \\tfrac{1}{2}\\left(|k-1|^{2H} - 2|k|^{2H}
                     + |k+1|^{2H}\\right) .

    ``gamma(0) = 1``; ``gamma(k) < 0`` for ``H < 1/2`` (antipersistent) and
    ``gamma(k) > 0`` for ``H > 1/2`` (persistent); ``gamma(k) = 0`` for
    ``k >= 1`` at ``H = 1/2``, recovering white noise.

    Parameters
    ----------
    n_lags : int
        Number of lags to return, i.e. ``k = 0 .. n_lags - 1``.
    hurst : float
        Hurst exponent in ``(0, 1)``.

    Returns
    -------
    ndarray of float64
        Shape ``(n_lags,)``.
    """
    k = np.arange(n_lags, dtype=np.float64)
    two_h = 2.0 * hurst
    return 0.5 * (np.abs(k - 1.0) ** two_h - 2.0 * k**two_h + (k + 1.0) ** two_h)


def circulant_eigenvalues(n_samples: int, hurst: float) -> NDArray[np.float64]:
    """Eigenvalues of the circulant embedding of the fGn covariance.

    The covariance of fGn is symmetric Toeplitz, so it embeds in a circulant
    matrix of size ``m = 2 n`` whose first row is

    ``[gamma(0), ..., gamma(n), gamma(n-1), ..., gamma(1)]``.

    A circulant is diagonalised by the DFT, so its eigenvalues are the DFT of
    that row, and the embedding is a valid covariance exactly when they are all
    non-negative.

    Parameters
    ----------
    n_samples : int
        Number of fGn samples the embedding must support.
    hurst : float
        Hurst exponent in ``(0, 1)``.

    Returns
    -------
    ndarray of float64
        Shape ``(2 * n_samples,)``, real by symmetry of the first row.
    """
    gamma = fgn_autocovariance(n_samples + 1, hurst)
    first_row = np.concatenate([gamma, gamma[-2:0:-1]])
    return np.fft.fft(first_row).real


def _validated_eigenvalues(
    eigenvalues: NDArray[np.float64],
    hurst: float,
    n_samples: int,
) -> NDArray[np.float64]:
    """Assert the non-negativity condition, then clip float noise to zero.

    Parameters
    ----------
    eigenvalues : ndarray of float64
        Output of :func:`circulant_eigenvalues`.
    hurst, n_samples
        Only used to build a useful error message.

    Returns
    -------
    ndarray of float64
        The eigenvalues with any ``-1e-16``-scale negatives clipped to zero.

    Raises
    ------
    ValueError
        If any eigenvalue is more negative than ``-EIGENVALUE_TOL`` times the
        largest one. Truncating a materially negative eigenvalue would sample
        from a different covariance than the one requested and quietly
        contaminate every correlation measurement downstream, so this raises
        instead (``PHASE1_PROMPT.md`` step 2).
    """
    worst = float(eigenvalues.min())
    largest = float(eigenvalues.max())
    if worst < -EIGENVALUE_TOL * largest:
        raise ValueError(
            "Davies-Harte circulant embedding failed the non-negativity "
            f"condition for H={hurst} and n_samples={n_samples}: smallest "
            f"eigenvalue {worst:.3e} against largest {largest:.3e}. The "
            "embedding is invalid; truncating would sample a different "
            "covariance, so this raises rather than silently continuing."
        )
    return np.clip(eigenvalues, 0.0, None)


def fractional_gaussian_noise(
    n_series: int,
    n_samples: int,
    hurst: float,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """Draw exact unit-variance fractional Gaussian noise.

    The covariance matrix of fGn is symmetric Toeplitz, so it embeds in a
    circulant matrix of size ``m = 2 n`` whose first row is

    ``[gamma(0), ..., gamma(n), gamma(n-1), ..., gamma(1)]``.

    A circulant is diagonalised by the DFT, so its eigenvalues are
    ``lambda = FFT(c)``, and if they are all non-negative a sample is obtained
    from one FFT of complex Gaussian noise scaled by ``sqrt(lambda / 2m)``.

    Both the real and the imaginary part of that transform are valid,
    *independent* fGn samples -- the transform is a proper complex Gaussian with
    a real covariance -- so one FFT yields two series and the work is halved.

    Parameters
    ----------
    n_series : int
        Number of independent series to draw.
    n_samples : int
        Length of each series.
    hurst : float
        Hurst exponent in ``(0, 1)``.
    rng : numpy.random.Generator
        Explicit generator.

    Returns
    -------
    ndarray of float64
        Shape ``(n_series, n_samples)``, each row unit-variance fGn.

    Raises
    ------
    ValueError
        If the circulant embedding has a materially negative eigenvalue. The
        embedding is then invalid and the method cannot be repaired by
        truncation: clipping to zero would silently return samples from a
        different covariance than the one requested.
    """
    require_generator(rng)
    if n_samples < 1:
        raise ValueError(f"n_samples must be >= 1, got {n_samples}")

    m = 2 * n_samples
    eigenvalues = _validated_eigenvalues(
        circulant_eigenvalues(n_samples, hurst), hurst, n_samples
    )

    n_pairs = (n_series + 1) // 2
    noise = rng.standard_normal((n_pairs, m)) + 1j * rng.standard_normal((n_pairs, m))
    spectrum = noise * np.sqrt(eigenvalues / (2.0 * m))
    transform = np.fft.fft(spectrum, axis=1)[:, :n_samples]

    out = np.empty((2 * n_pairs, n_samples), dtype=np.float64)
    out[0::2] = np.sqrt(2.0) * transform.real
    out[1::2] = np.sqrt(2.0) * transform.imag
    return out[:n_series]


def fbm(cfg: FBMConfig, rng: np.random.Generator) -> NDArray[np.float64]:
    """Generate fractional Brownian motion trajectories.

    fBm is the cumulative sum of fractional Gaussian noise, scaled so that

    .. math::

        \\langle r^2(t) \\rangle = 2 d D t^{2H} ,

    exactly at every grid point. ``H < 1/2`` gives subdiffusion with
    anticorrelated increments, ``H > 1/2`` superdiffusion with positively
    correlated increments, and ``H = 1/2`` reduces to Brownian motion with the
    same ``D``.

    Parameters
    ----------
    cfg : FBMConfig
        Ensemble size, trajectory length, dimension, Hurst exponent ``H`` and
        generalised diffusivity ``D``.
    rng : numpy.random.Generator
        Explicit generator.

    Returns
    -------
    ndarray of float64
        Shape ``(n_particles, n_steps, n_dim)``, starting at the origin.
    """
    require_generator(rng)
    n_particles, n_steps, n_dim = cfg.shape
    n_increments = n_steps - 1

    noise = fractional_gaussian_noise(n_particles * n_dim, n_increments, cfg.hurst, rng)
    increments = np.sqrt(2.0 * cfg.diffusivity) * noise.reshape(n_particles, n_dim, n_increments)
    increments = np.moveaxis(increments, 1, 2)

    positions = np.zeros((n_particles, n_steps, n_dim), dtype=np.float64)
    np.cumsum(increments, axis=1, out=positions[:, 1:, :])
    return as_trajectories(positions, n_particles, n_steps, n_dim)
