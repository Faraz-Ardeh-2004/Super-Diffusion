"""Whittle spectral likelihood and Cramér-Rao lower bound for fBm/fGn (Phase 2, Step 5).

Following ``PHASE2_PROMPT.md``:
- Fractional Gaussian Noise (fGn) has an exact spectral density.
- Whittle likelihood gives an asymptotically efficient estimator of Hurst exponent H.
- Fisher information provides the Cramér-Rao lower bound (CRLB) on Var(H).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize_scalar
from scipy.special import gamma

__all__ = [
    "cramer_rao_bound_fbm",
    "fgn_spectral_density",
    "fisher_information_fbm",
    "whittle_log_likelihood",
    "whittle_mle_hurst",
]


def fgn_spectral_density(
    freqs: NDArray[np.float64],
    hurst: float,
    diffusivity: float = 1.0,
    n_terms: int = 100,
) -> NDArray[np.float64]:
    """Spectral density of fractional Gaussian noise (fGn) on [-pi, pi].

    .. math::

        f(\\lambda; H) = 2 c_H (1 - \\cos \\lambda)
        \\sum_{j=-\\infty}^{\\infty} |\\lambda + 2\\pi j|^{-2H-1}

    with :math:`c_H = \\frac{\\sigma^2}{2\\pi} \\sin(\\pi H) \\Gamma(2H+1)`.

    Parameters
    ----------
    freqs : ndarray of float64
        Frequencies in ``(0, pi]``.
    hurst : float
        Hurst exponent ``H in (0, 1)``.
    diffusivity : float
        Scale factor ``sigma^2 = 2 * D``.
    n_terms : int
        Number of alias terms in the summation.

    Returns
    -------
    ndarray of float64
        Spectral density values.
    """
    lam = np.asarray(freqs, dtype=np.float64)
    if np.any(lam <= 0.0) or np.any(lam > np.pi + 1e-6):
        raise ValueError("Frequencies must lie in (0, pi]")
    if not 0.0 < hurst < 1.0:
        raise ValueError(f"Hurst exponent must lie in (0, 1), got {hurst}")

    sigma2 = 2.0 * diffusivity
    c_h = (sigma2 / (2.0 * np.pi)) * np.sin(np.pi * hurst) * gamma(2.0 * hurst + 1.0)
    exponent = -2.0 * hurst - 1.0

    # Aliasing sum j in [-n_terms, n_terms]
    total_sum = np.abs(lam) ** exponent
    for j in range(1, n_terms + 1):
        total_sum += np.abs(lam + 2.0 * np.pi * j) ** exponent
        total_sum += np.abs(lam - 2.0 * np.pi * j) ** exponent

    # Tail approximation for j > n_terms via integration
    # integral_{n_terms}^\infty (2 pi x)^exponent dx
    tail = 2.0 * ((2.0 * np.pi * n_terms) ** (exponent + 1.0)) / (2.0 * np.pi * (2.0 * hurst))
    total_sum += tail

    factor = 2.0 * c_h * (1.0 - np.cos(lam))
    return factor * total_sum


def whittle_log_likelihood(
    increments: NDArray[np.float64],
    hurst: float,
    diffusivity: float = 1.0,
) -> float:
    """Whittle approximation to Gaussian log-likelihood for fGn.

    Parameters
    ----------
    increments : ndarray of float64
        1D increment series of length N.
    hurst : float
        Candidate Hurst exponent in ``(0, 1)``.
    diffusivity : float
        Scale parameter.

    Returns
    -------
    float
        Whittle negative log-likelihood (to minimize).
    """
    n = len(increments)
    # Periodogram at Fourier frequencies
    fft_vals = np.fft.rfft(increments - np.mean(increments))[1:]  # skip zero freq
    freqs = 2.0 * np.pi * np.arange(1, len(fft_vals) + 1, dtype=np.float64) / n
    periodogram = (np.abs(fft_vals) ** 2) / (2.0 * np.pi * n)

    spec = fgn_spectral_density(freqs, hurst, diffusivity=diffusivity)
    # Profile out scale: min sum(log spec) + sum(I / spec)
    val = float(np.sum(np.log(spec) + periodogram / spec))
    return val


def whittle_mle_hurst(
    trajectory: NDArray[Any],
    component: int = 0,
    bounds: tuple[float, float] = (0.05, 0.95),
) -> float:
    """Estimate Hurst exponent H of a single trajectory via Whittle MLE.

    Parameters
    ----------
    trajectory : ndarray
        Shape ``(n_steps,)``, ``(n_steps, 1)``, or ``(1, n_steps, 1)``.
    component : int
        Spatial component.
    bounds : tuple of float
        Optimization bounds for H.

    Returns
    -------
    float
        Maximum likelihood estimate of Hurst exponent.
    """
    traj = np.asarray(trajectory, dtype=np.float64)
    if traj.ndim == 3:
        series = traj[0, :, component]
    elif traj.ndim == 2:
        series = traj[:, component]
    else:
        series = traj

    increments = np.diff(series)
    n = len(increments)
    fft_vals = np.fft.rfft(increments - np.mean(increments))[1:]
    freqs = 2.0 * np.pi * np.arange(1, len(fft_vals) + 1, dtype=np.float64) / n
    periodogram = (np.abs(fft_vals) ** 2) / (2.0 * np.pi * n)
    m = len(freqs)

    def objective(h: float) -> float:
        # Scale-profiled Whittle objective
        spec_shape = fgn_spectral_density(freqs, h, diffusivity=1.0)
        # Profiled sigma2
        sigma2_hat = np.sum(periodogram / spec_shape) / m
        if sigma2_hat <= 0:
            return 1e12
        return float(np.sum(np.log(spec_shape)) + m * np.log(sigma2_hat))

    res = minimize_scalar(objective, bounds=bounds, method="bounded")
    return float(res.x)


def fisher_information_fbm(
    hurst: float,
    n_steps: int,
    n_quad: int = 500,
) -> float:
    """Fisher information I(H) for Whittle estimator of fGn.

    .. math::

        I(H) = \\frac{N}{4\\pi} \\int_{-\\pi}^{\\pi}
        \\left( \\frac{\\partial \\log f(\\lambda; H)}{\\partial H} \\right)^2 d\\lambda
             = \\frac{N}{2\\pi} \\int_0^{\\pi}
        \\left( \\frac{\\partial \\log f(\\lambda; H)}{\\partial H} \\right)^2 d\\lambda

    Parameters
    ----------
    hurst : float
        Hurst exponent.
    n_steps : int
        Trajectory length (N = n_steps - 1 increments).
    n_quad : int
        Integration grid size.

    Returns
    -------
    float
        Fisher information I(H).
    """
    n_increments = n_steps - 1
    freqs = np.linspace(1e-4, np.pi, n_quad)
    dh = 1e-4

    f_plus = fgn_spectral_density(freqs, hurst + dh)
    f_minus = fgn_spectral_density(freqs, hurst - dh)
    f_mid = fgn_spectral_density(freqs, hurst)

    dlogf_dh = (np.log(f_plus) - np.log(f_minus)) / (2.0 * dh)
    integrand = dlogf_dh**2

    # Numerical integration over [0, pi]
    integral = float(np.trapezoid(integrand, freqs))
    i_h = (n_increments / (2.0 * np.pi)) * integral
    return i_h


def cramer_rao_bound_fbm(hurst: float, n_steps: int) -> float:
    """Cramér-Rao lower bound on the variance of any unbiased estimator of H.

    .. math::

        \\mathrm{CRLB}(H) = \\frac{1}{I(H)}

    Parameters
    ----------
    hurst : float
        Hurst exponent.
    n_steps : int
        Trajectory length.

    Returns
    -------
    float
        Minimum attainable variance Var(H).
    """
    i_h = fisher_information_fbm(hurst, n_steps)
    return 1.0 / i_h
