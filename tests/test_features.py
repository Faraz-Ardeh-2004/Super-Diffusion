"""Tests for single-trajectory feature extraction and scale invariance (Step 1)."""

from __future__ import annotations

import numpy as np
import pytest

from collectivediff.config import (
    BrownianConfig,
    CTRWConfig,
    DDMConfig,
    FBMConfig,
    LevyWalkConfig,
    SBMConfig,
)
from collectivediff.features import (
    FEATURE_GROUPS,
    FEATURE_NAMES,
    extract_feature_dict,
    extract_feature_matrix,
    extract_features,
)
from collectivediff.generators import generate


class TestFeatureDefinitions:
    def test_feature_groups_cover_all_features(self) -> None:
        grouped_features = set()
        for group, features in FEATURE_GROUPS.items():
            for f in features:
                grouped_features.add(f)
        assert grouped_features == set(FEATURE_NAMES)

    def test_single_trajectory_extraction_dict(self) -> None:
        rng = np.random.default_rng(42)
        traj = generate(BrownianConfig(n_particles=1, n_steps=256), rng)
        f_dict = extract_feature_dict(traj[0])
        assert set(f_dict.keys()) == set(FEATURE_NAMES)
        for k, v in f_dict.items():
            assert np.isfinite(v), f"Feature {k} returned non-finite value {v}"

    def test_feature_vector_dimensions_and_shapes(self) -> None:
        rng = np.random.default_rng(42)
        traj_3d = generate(BrownianConfig(n_particles=1, n_steps=256), rng)  # (1, 256, 1)
        traj_2d = traj_3d[0]  # (256, 1)
        traj_1d = traj_3d[0, :, 0]  # (256,)

        vec_3d = extract_features(traj_3d)
        vec_2d = extract_features(traj_2d)
        vec_1d = extract_features(traj_1d)

        assert vec_3d.shape == (len(FEATURE_NAMES),)
        assert np.allclose(vec_3d, vec_2d, rtol=1e-10)
        assert np.allclose(vec_3d, vec_1d, rtol=1e-10)

    def test_feature_matrix_ensemble(self) -> None:
        rng = np.random.default_rng(42)
        trajs = generate(FBMConfig(n_particles=5, n_steps=128), rng)
        mat = extract_feature_matrix(trajs)
        assert mat.shape == (5, len(FEATURE_NAMES))
        # Verify independence: computing particle 2 directly equals row 2
        vec2 = extract_features(trajs[2:3])
        assert np.allclose(mat[2], vec2, rtol=1e-12)


class TestScaleInvariance:
    """Mandatory scale invariance test: every feature must be invariant under x -> c * x."""

    @pytest.mark.parametrize(
        "cfg",
        [
            BrownianConfig(n_particles=1, n_steps=512, diffusivity=1.5, seed=10),
            FBMConfig(n_particles=1, n_steps=512, hurst=0.3, diffusivity=1.0, seed=11),
            FBMConfig(n_particles=1, n_steps=512, hurst=0.7, diffusivity=2.0, seed=12),
            CTRWConfig(n_particles=1, n_steps=512, alpha_wait=0.5, jump_scale=1.0, seed=13),
            SBMConfig(n_particles=1, n_steps=512, alpha=0.6, diffusivity=1.0, seed=14),
            DDMConfig(n_particles=1, n_steps=512, tau=30.0, sigma=1.0, seed=15),
            LevyWalkConfig(n_particles=1, n_steps=512, gamma=1.5, speed=2.0, seed=16),
        ],
    )
    @pytest.mark.parametrize("scale_factor", [0.05, 3.7, 100.0])
    def test_all_features_scale_invariant(self, cfg, scale_factor: float) -> None:
        rng = np.random.default_rng(cfg.seed)
        traj = generate(cfg, rng)  # (1, 512, 1)
        scaled_traj = scale_factor * traj

        f_raw = extract_features(traj)
        f_scaled = extract_features(scaled_traj)

        for name, v_raw, v_scaled in zip(FEATURE_NAMES, f_raw, f_scaled):
            # Tolerate small numerical differences in power law fits / sample moments
            assert np.allclose(
                v_scaled, v_raw, rtol=1e-4, atol=1e-6
            ), f"Feature {name} failed scale invariance on {type(cfg).name}: {v_raw} vs {v_scaled} (factor {scale_factor})"
