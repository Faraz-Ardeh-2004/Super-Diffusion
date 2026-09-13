"""Dataset generation for single-trajectory mechanism classification (Phase 2, Step 2).

Following ``PHASE2_PROMPT.md``:
- Continuously drawn parameters per mechanism so models cannot memorise discrete sets.
- Disjoint seeds for train, interpolation test, and extrapolation test splits.
- Explicit uniform class prior across mechanisms.
- Trajectory lengths T in {128, 512, 2048, 8192, 32768}.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Sequence

import numpy as np
from numpy.typing import NDArray

from ..cache import CACHE_ROOT
from ..config import (
    BrownianConfig,
    CTRWConfig,
    DDMConfig,
    FBMConfig,
    GeneratorConfig,
    LevyWalkConfig,
    SBMConfig,
)
from ..features import FEATURE_NAMES, extract_features
from ..generators import generate

__all__ = [
    "MECHANISM_NAMES",
    "DatasetConfig",
    "TrajectoryDataset",
    "draw_random_config",
    "generate_dataset",
    "load_or_generate_dataset",
]

#: Registered mechanisms for classification
MECHANISM_NAMES: Final[list[str]] = [
    "brownian",
    "fbm",
    "ctrw",
    "sbm",
    "ddm",
    "levy_walk",
]


@dataclass(frozen=True)
class DatasetConfig:
    """Configuration for a multi-mechanism labelled dataset.

    Parameters
    ----------
    n_samples_per_class : int
        Number of trajectories generated per mechanism.
    n_steps : int
        Trajectory length (T = n_steps - 1).
    split : str
        One of ``"train"``, ``"test_interp"``, ``"test_extrap"``.
    seed : int
        RNG seed for parameter drawing and trajectory generation.
    mechanisms : tuple of str
        Mechanisms to include. Defaults to all 6 mechanisms.
    """

    n_samples_per_class: int = 100
    n_steps: int = 512
    split: str = "train"
    seed: int = 1000
    mechanisms: tuple[str, ...] = tuple(MECHANISM_NAMES)

    def __post_init__(self) -> None:
        if self.n_samples_per_class < 1:
            raise ValueError(f"n_samples_per_class must be >= 1, got {self.n_samples_per_class}")
        if self.n_steps < 16:
            raise ValueError(f"n_steps must be >= 16, got {self.n_steps}")
        if self.split not in ("train", "test_interp", "test_extrap"):
            raise ValueError(f"Unknown split {self.split!r}")


@dataclass
class TrajectoryDataset:
    """Labelled feature dataset extracted from single trajectories."""

    features: NDArray[np.float64]  # (n_samples, n_features)
    labels: NDArray[np.intp]  # (n_samples,)
    mechanism_names: list[str]
    parameters: list[dict[str, Any]]
    feature_names: list[str]
    config: DatasetConfig

    @property
    def n_samples(self) -> int:
        return len(self.labels)

    @property
    def class_prior(self) -> NDArray[np.float64]:
        """Empirical class prior distribution."""
        counts = np.bincount(self.labels, minlength=len(self.mechanism_names))
        return counts / counts.sum()


def draw_random_config(
    mechanism: str,
    n_steps: int,
    rng: np.random.Generator,
    extrapolate: bool = False,
    seed: int = 0,
) -> tuple[GeneratorConfig, dict[str, Any]]:
    """Draw a configuration with continuously sampled parameters.

    Parameters
    ----------
    mechanism : str
        One of ``"brownian"``, ``"fbm"``, ``"ctrw"``, ``"sbm"``, ``"ddm"``, ``"levy_walk"``.
    n_steps : int
        Trajectory length.
    rng : np.random.Generator
        Generator for sampling continuous parameter values.
    extrapolate : bool
        If True, sample from outside the interpolation parameter range.
    seed : int
        Seed for the physical trajectory generation.

    Returns
    -------
    cfg : GeneratorConfig
    params : dict
        Logged parameter record including true alpha.
    """
    params: dict[str, Any] = {"mechanism": mechanism, "extrapolate": extrapolate}

    if mechanism == "brownian":
        if extrapolate:
            # Diffusivity outside standard range
            diff = float(rng.uniform(0.02, 0.15) if rng.random() < 0.5 else rng.uniform(6.0, 20.0))
        else:
            diff = float(rng.uniform(0.2, 5.0))
        cfg = BrownianConfig(n_particles=1, n_steps=n_steps, diffusivity=diff, seed=seed)
        params.update({"diffusivity": diff, "alpha_true": 1.0})

    elif mechanism == "fbm":
        if extrapolate:
            hurst = float(rng.uniform(0.05, 0.14) if rng.random() < 0.5 else rng.uniform(0.86, 0.95))
        else:
            hurst = float(rng.uniform(0.15, 0.85))
        diff = float(rng.uniform(0.5, 2.0))
        cfg = FBMConfig(n_particles=1, n_steps=n_steps, hurst=hurst, diffusivity=diff, seed=seed)
        params.update({"hurst": hurst, "diffusivity": diff, "alpha_true": 2.0 * hurst})

    elif mechanism == "ctrw":
        if extrapolate:
            a = float(rng.uniform(0.08, 0.18) if rng.random() < 0.5 else rng.uniform(0.87, 0.95))
        else:
            a = float(rng.uniform(0.2, 0.85))
        tau0 = float(rng.uniform(0.5, 2.0))
        jump_scale = float(rng.uniform(0.5, 2.0))
        cfg = CTRWConfig(
            n_particles=1,
            n_steps=n_steps,
            alpha_wait=a,
            tau0=tau0,
            jump_scale=jump_scale,
            seed=seed,
        )
        params.update({"alpha_wait": a, "tau0": tau0, "jump_scale": jump_scale, "alpha_true": a})

    elif mechanism == "sbm":
        if extrapolate:
            alpha = float(rng.uniform(0.1, 0.28) if rng.random() < 0.5 else rng.uniform(1.72, 1.95))
        else:
            alpha = float(rng.uniform(0.3, 1.7))
        diff = float(rng.uniform(0.5, 2.0))
        cfg = SBMConfig(n_particles=1, n_steps=n_steps, alpha=alpha, diffusivity=diff, seed=seed)
        params.update({"alpha": alpha, "diffusivity": diff, "alpha_true": alpha})

    elif mechanism == "ddm":
        if extrapolate:
            tau = float(rng.uniform(3.0, 9.0) if rng.random() < 0.5 else rng.uniform(70.0, 150.0))
        else:
            tau = float(rng.uniform(10.0, 60.0))
        sigma = float(rng.uniform(0.5, 2.0))
        cfg = DDMConfig(n_particles=1, n_steps=n_steps, tau=tau, sigma=sigma, n_aux=1, seed=seed)
        params.update({"tau": tau, "sigma": sigma, "n_aux": 1, "alpha_true": 1.0})

    elif mechanism == "levy_walk":
        if extrapolate:
            g = float(rng.uniform(1.05, 1.18) if rng.random() < 0.5 else rng.uniform(1.82, 1.95))
        else:
            g = float(rng.uniform(1.2, 1.8))
        speed = float(rng.uniform(0.5, 2.0))
        tau0 = float(rng.uniform(0.5, 2.0))
        cfg = LevyWalkConfig(
            n_particles=1,
            n_steps=n_steps,
            gamma=g,
            tau0=tau0,
            speed=speed,
            seed=seed,
        )
        params.update({"gamma": g, "tau0": tau0, "speed": speed, "alpha_true": 3.0 - g})

    else:
        raise ValueError(f"Unknown mechanism {mechanism!r}")

    return cfg, params


def generate_dataset(
    config: DatasetConfig,
    feature_names: Sequence[str] | None = None,
) -> TrajectoryDataset:
    """Generate labelled feature matrix across mechanisms for a given config.

    Parameters
    ----------
    config : DatasetConfig
        Simulation and split configuration.
    feature_names : sequence of str, optional
        Features to compute. Defaults to :data:`FEATURE_NAMES`.

    Returns
    -------
    TrajectoryDataset
    """
    f_names = FEATURE_NAMES if feature_names is None else list(feature_names)
    mechanisms = list(config.mechanisms)
    extrapolate = config.split == "test_extrap"

    # RNG for parameter choices
    param_rng = np.random.default_rng(config.seed)

    features_list: list[NDArray[np.float64]] = []
    labels_list: list[int] = []
    params_list: list[dict[str, Any]] = []

    # Iterate over classes and generate trajectories
    for class_idx, mech in enumerate(mechanisms):
        for i in range(config.n_samples_per_class):
            traj_seed = int(param_rng.integers(1, 2**31 - 1))
            cfg, p_record = draw_random_config(
                mechanism=mech,
                n_steps=config.n_steps,
                rng=param_rng,
                extrapolate=extrapolate,
                seed=traj_seed,
            )
            # Generate single trajectory with isolated observer
            traj_rng = np.random.default_rng(traj_seed)
            trajectory = generate(cfg, traj_rng)  # shape (1, n_steps, 1)

            feat_vec = extract_features(trajectory, feature_names=f_names)
            features_list.append(feat_vec)
            labels_list.append(class_idx)
            params_list.append(p_record)

    features_arr = np.array(features_list, dtype=np.float64)
    labels_arr = np.array(labels_list, dtype=np.intp)

    return TrajectoryDataset(
        features=features_arr,
        labels=labels_arr,
        mechanism_names=mechanisms,
        parameters=params_list,
        feature_names=f_names,
        config=config,
    )


def load_or_generate_dataset(
    config: DatasetConfig,
    cache_root: Path | None = None,
) -> TrajectoryDataset:
    """Load cached dataset or generate and save if absent."""
    root = CACHE_ROOT if cache_root is None else Path(cache_root)
    root.mkdir(parents=True, exist_ok=True)

    cfg_dict = {
        "n_samples_per_class": config.n_samples_per_class,
        "n_steps": config.n_steps,
        "split": config.split,
        "seed": config.seed,
        "mechanisms": list(config.mechanisms),
    }
    digest = hashlib.sha256(json.dumps(cfg_dict, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    cache_file = root / f"dataset-{digest}.npz"
    sidecar = root / f"dataset-{digest}.json"

    if cache_file.exists() and sidecar.exists():
        with np.load(cache_file) as npz:
            features = npz["features"]
            labels = npz["labels"]
        with open(sidecar, "r", encoding="utf-8") as f:
            meta = json.load(f)
        return TrajectoryDataset(
            features=features,
            labels=labels,
            mechanism_names=meta["mechanism_names"],
            parameters=meta["parameters"],
            feature_names=meta["feature_names"],
            config=config,
        )

    ds = generate_dataset(config)

    # Save to cache
    np.savez(cache_file, features=ds.features, labels=ds.labels)
    sidecar_content = {
        "config": cfg_dict,
        "mechanism_names": ds.mechanism_names,
        "feature_names": ds.feature_names,
        "parameters": ds.parameters,
    }
    with open(sidecar, "w", encoding="utf-8") as f:
        json.dump(sidecar_content, f, indent=2)

    return ds
