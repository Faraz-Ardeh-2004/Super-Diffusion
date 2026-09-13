"""Trivial smoke tests for the phase-1 skeleton.

No simulation logic exists yet, so these only check that the package imports,
that configs are immutable, validated and JSON round-trippable, and that the
cache stores and returns arrays under a config-derived key.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest

import collectivediff as cd
from collectivediff import cache, config


def test_package_imports() -> None:
    """The package and every declared subpackage import cleanly."""
    import collectivediff.dynamics  # noqa: F401
    import collectivediff.estimators  # noqa: F401
    import collectivediff.generators  # noqa: F401
    import collectivediff.inference  # noqa: F401
    import collectivediff.sampling  # noqa: F401
    import collectivediff.validation  # noqa: F401
    import collectivediff.viz  # noqa: F401

    assert cd.__version__ == "0.1.0"
    assert cd.DT == 1.0


ALL_GENERATOR_CONFIGS = [
    config.BrownianConfig(),
    config.FBMConfig(hurst=0.3),
    config.FBMConfig(hurst=0.7),
    config.SBMConfig(alpha=0.6),
    config.CTRWConfig(alpha_wait=0.7),
    config.LevyFlightConfig(stability=1.5),
    config.LevyWalkConfig(gamma=1.4),
    config.DDMConfig(),
]


@pytest.mark.parametrize("cfg", ALL_GENERATOR_CONFIGS, ids=lambda c: type(c).name)
def test_configs_are_frozen(cfg: config.GeneratorConfig) -> None:
    """Configs are immutable, so a cached array can never drift from its key."""
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.n_steps = 4  # type: ignore[misc]


@pytest.mark.parametrize("cfg", ALL_GENERATOR_CONFIGS, ids=lambda c: type(c).name)
def test_config_json_round_trip(cfg: config.GeneratorConfig) -> None:
    """Canonical JSON round-trips back to an equal config."""
    assert config.config_from_json(config.config_to_json(cfg)) == cfg


@pytest.mark.parametrize("cfg", ALL_GENERATOR_CONFIGS, ids=lambda c: type(c).name)
def test_shape_matches_layout_convention(cfg: config.GeneratorConfig) -> None:
    """``shape`` is the ``(n_particles, n_steps, n_dim)`` layout of PROJECT.md."""
    assert cfg.shape == (cfg.n_particles, cfg.n_steps, cfg.n_dim)
    assert cfg.duration == cfg.n_steps - 1


def test_analytic_exponents() -> None:
    """Each config reports the exponent of PROJECT.md section 4."""
    assert config.BrownianConfig().alpha_analytic == 1.0
    assert config.FBMConfig(hurst=0.3).alpha_analytic == pytest.approx(0.6)
    assert config.SBMConfig(alpha=0.6).alpha_analytic == pytest.approx(0.6)
    assert config.CTRWConfig(alpha_wait=0.7).alpha_analytic == pytest.approx(0.7)
    assert config.LevyWalkConfig(gamma=1.4).alpha_analytic == pytest.approx(1.6)
    assert config.DDMConfig().alpha_analytic == 1.0


def test_levy_flight_has_no_finite_msd() -> None:
    """Asking a Levy flight for an MSD exponent raises rather than lying."""
    cfg = config.LevyFlightConfig(stability=1.5)
    assert cfg.msd_is_finite is False
    assert cfg.nu_analytic == pytest.approx(1.0 / 1.5)
    with pytest.raises(ValueError, match="diverges"):
        _ = cfg.alpha_analytic


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"n_dim": 3}, "n_dim"),
        ({"n_particles": 0}, "n_particles"),
        ({"n_steps": 1}, "n_steps"),
        ({"diffusivity": 0.0}, "diffusivity"),
    ],
)
def test_invalid_parameters_are_rejected(kwargs: dict, message: str) -> None:
    """Out-of-range parameters fail at construction, not deep inside a run."""
    with pytest.raises(ValueError, match=message):
        config.BrownianConfig(**kwargs)


@pytest.mark.parametrize(
    "cls, kwargs",
    [
        (config.FBMConfig, {"hurst": 1.0}),
        (config.SBMConfig, {"alpha": 2.0}),
        (config.CTRWConfig, {"alpha_wait": 1.0}),
        (config.LevyFlightConfig, {"stability": 2.0}),
        (config.LevyWalkConfig, {"gamma": 1.0}),
        (config.DDMConfig, {"tau": 0.0}),
    ],
)
def test_generator_parameter_ranges(cls: type, kwargs: dict) -> None:
    """Each generator rejects parameters outside its admissible range."""
    with pytest.raises(ValueError):
        cls(**kwargs)


def test_hash_depends_on_every_field() -> None:
    """Changing any parameter changes the cache key; equal configs share it."""
    base = config.FBMConfig(hurst=0.3, seed=0)
    assert cache.config_hash(base) == cache.config_hash(config.FBMConfig(hurst=0.3))
    assert cache.config_hash(base) != cache.config_hash(config.FBMConfig(hurst=0.31))
    assert cache.config_hash(base) != cache.config_hash(
        config.FBMConfig(hurst=0.3, seed=1)
    )
    # Same numbers, different mechanism: must not collide.
    assert cache.config_hash(config.BrownianConfig()) != cache.config_hash(
        config.SBMConfig(alpha=1.0)
    )


def test_cache_round_trip(tmp_path: Path) -> None:
    """Arrays come back as float32 within single precision; a miss returns None.

    ``PROJECT.md`` section 2 stores cached arrays as ``float32`` and keeps
    ``float64`` for estimator accumulations, so the round trip is lossy by
    design. Seven significant digits is some five orders of magnitude finer
    than the statistical error on anything this project measures.
    """
    cfg = config.BrownianConfig(n_particles=4, n_steps=8, n_dim=2)
    assert cache.load_arrays(cfg, root=tmp_path) is None

    rng = np.random.default_rng(0)
    trajectories = rng.standard_normal(cfg.shape)
    path = cache.save_arrays(cfg, {"trajectories": trajectories}, root=tmp_path)

    assert path.exists()
    assert path.with_suffix(".json").exists()
    loaded = cache.load_arrays(cfg, root=tmp_path)
    assert loaded is not None
    assert set(loaded) == {"trajectories"}
    assert loaded["trajectories"].dtype == np.float32
    np.testing.assert_allclose(loaded["trajectories"], trajectories, rtol=1e-6)
    assert not list(tmp_path.glob("*.tmp*"))


def test_cache_preserves_integer_arrays(tmp_path: Path) -> None:
    """Only floats are demoted: an event count has no rounding budget to spend."""
    cfg = config.CTRWConfig(n_particles=4, n_steps=8)
    counts = np.arange(32, dtype=np.int64).reshape(4, 8)
    cache.save_arrays(cfg, {"counts": counts}, root=tmp_path)
    loaded = cache.load_arrays(cfg, root=tmp_path)
    assert loaded is not None
    assert loaded["counts"].dtype == np.int64
    np.testing.assert_array_equal(loaded["counts"], counts)


def test_cached_computes_once(tmp_path: Path) -> None:
    """``cached`` calls the compute function only on a cache miss."""
    cfg = config.BrownianConfig(n_particles=2, n_steps=4)
    calls = 0

    def compute() -> dict[str, np.ndarray]:
        nonlocal calls
        calls += 1
        return {"x": np.zeros(cfg.shape)}

    first = cache.cached(cfg, compute, root=tmp_path)
    second = cache.cached(cfg, compute, root=tmp_path)
    assert calls == 1
    np.testing.assert_array_equal(first["x"], second["x"])

    cache.cached(cfg, compute, root=tmp_path, use_cache=False)
    assert calls == 2


def test_cache_rejects_reserved_key(tmp_path: Path) -> None:
    """The stored-config key cannot be shadowed by a data array."""
    cfg = config.BrownianConfig(n_particles=1, n_steps=2)
    with pytest.raises(KeyError):
        cache.save_arrays(cfg, {cache.CONFIG_KEY: np.zeros(1)}, root=tmp_path)


def test_clear_cache(tmp_path: Path) -> None:
    """Clearing removes both the archive and its config sidecar."""
    cfg = config.BrownianConfig(n_particles=1, n_steps=2)
    cache.save_arrays(cfg, {"x": np.zeros(cfg.shape)}, root=tmp_path)
    assert cache.clear_cache(root=tmp_path) == 2
    assert cache.load_arrays(cfg, root=tmp_path) is None
