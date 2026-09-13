"""Tests for Whittle spectral likelihood, Fisher information, and CRLB (Step 5)."""

from __future__ import annotations

import numpy as np
import pytest

from collectivediff.config import FBMConfig
from collectivediff.features import (
    cramer_rao_bound_fbm,
    fgn_spectral_density,
    fisher_information_fbm,
    whittle_mle_hurst,
)
from collectivediff.generators import generate


class TestWhittleSpectralDensity:
    def test_spectral_density_monotonicity_and_positivity(self) -> None:
        freqs = np.linspace(0.01, np.pi, 50)
        # For subdiffusive fGn (H < 0.5), high frequencies have higher power (negative autocorrelation)
        spec_sub = fgn_spectral_density(freqs, hurst=0.3)
        assert np.all(spec_sub > 0.0)
        assert spec_sub[-1] > spec_sub[0]

        # For superdiffusive fGn (H > 0.5), low frequencies have higher power (positive persistence)
        spec_super = fgn_spectral_density(freqs, hurst=0.7)
        assert np.all(spec_super > 0.0)
        assert spec_super[0] > spec_super[-1]

    def test_cramer_rao_bound_scales_as_one_over_n(self) -> None:
        crlb_128 = cramer_rao_bound_fbm(hurst=0.5, n_steps=128)
        crlb_512 = cramer_rao_bound_fbm(hurst=0.5, n_steps=512)
        crlb_2048 = cramer_rao_bound_fbm(hurst=0.5, n_steps=2048)

        ratio_1 = crlb_128 / crlb_512
        ratio_2 = crlb_512 / crlb_2048

        # N quadruples, CRLB should scale as ~4x
        assert np.isclose(ratio_1, 4.0, rtol=0.05)
        assert np.isclose(ratio_2, 4.0, rtol=0.05)

    def test_whittle_mle_recovers_hurst_on_clean_trajectory(self) -> None:
        rng = np.random.default_rng(123)
        true_h = 0.7
        # Long trajectory for accurate single-trajectory recovery
        traj = generate(FBMConfig(n_particles=1, n_steps=4096, hurst=true_h), rng)
        h_est = whittle_mle_hurst(traj)
        assert np.isclose(h_est, true_h, atol=0.05)
