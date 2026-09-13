"""Disposable on-disk cache for trajectory arrays, keyed by config hash.

Following ``PROJECT.md`` section 2: trajectory arrays are cached to
``data/cache/<hash>.npz`` where the hash is derived from the canonical JSON of
the config that produced them, and the config is stored alongside the arrays.
The cache is disposable and gitignored; nothing in the project may depend on it
being warm.

The config JSON is written twice on purpose: once inside the ``.npz`` under the
key :data:`CONFIG_KEY`, so an archive is self-describing if it is moved, and
once as a sibling ``<hash>.json``, so a human can read what a cache entry is
without opening it with numpy.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Callable, Final

import numpy as np

from .config import config_from_json, config_to_json

__all__ = [
    "CACHE_ROOT",
    "CONFIG_KEY",
    "config_hash",
    "cache_path",
    "save_arrays",
    "load_arrays",
    "cached",
    "clear_cache",
]

#: Default cache directory: ``<repo>/data/cache``, resolved from this file so
#: it does not depend on the working directory a notebook happens to run in.
CACHE_ROOT: Final[Path] = Path(__file__).resolve().parents[2] / "data" / "cache"

#: Key under which the canonical config JSON is stored inside the ``.npz``.
CONFIG_KEY: Final[str] = "__config__"

#: Number of hex characters kept from the SHA-256 digest.
HASH_LENGTH: Final[int] = 16

#: On-disk dtype for floating-point arrays (``PROJECT.md`` section 2).
STORAGE_DTYPE: Final[np.dtype] = np.dtype(np.float32)


def _to_storage_dtype(array: np.ndarray) -> np.ndarray:
    """Demote floating-point arrays to the on-disk dtype.

    ``PROJECT.md`` section 2 stores cached arrays as ``float32`` and keeps
    ``float64`` only for estimator accumulations. The trade is worth making
    because single precision carries about seven significant digits, some five
    orders of magnitude finer than the statistical error on any quantity this
    project measures, while halving the cache footprint of trajectory arrays
    that routinely run to hundreds of megabytes.

    Integer and boolean arrays pass through unchanged: an event count or a
    particle index has no rounding budget to spend.

    Parameters
    ----------
    array : ndarray
        Array on its way to disk.

    Returns
    -------
    ndarray
        ``float32`` if the input was floating point, otherwise the input.
    """
    array = np.asarray(array)
    if array.dtype.kind == "f":
        return array.astype(STORAGE_DTYPE, copy=False)
    return array


def config_hash(cfg: Any, length: int = HASH_LENGTH) -> str:
    """Hash a config to a short hex string.

    The digest is taken over the canonical JSON of the config, so two configs
    with the same field values hash identically regardless of construction
    order, and any parameter change produces a different cache entry.

    Parameters
    ----------
    cfg : dataclass instance
        Any config from :mod:`collectivediff.config`.
    length : int, optional
        Number of hex characters to keep from the SHA-256 digest.

    Returns
    -------
    str
        Lower-case hex digest prefix, e.g. ``"3f2a91c0d4e5b678"``.
    """
    digest = hashlib.sha256(config_to_json(cfg).encode("utf-8")).hexdigest()
    return digest[:length]


def cache_path(cfg: Any, root: Path | str | None = None, suffix: str = ".npz") -> Path:
    """Path of the cache entry belonging to a config.

    Parameters
    ----------
    cfg : dataclass instance
        Any config from :mod:`collectivediff.config`.
    root : path-like, optional
        Cache directory; defaults to :data:`CACHE_ROOT`.
    suffix : str, optional
        File suffix, ``".npz"`` for the arrays and ``".json"`` for the
        human-readable config sidecar.

    Returns
    -------
    pathlib.Path
        ``<root>/<name>-<hash><suffix>``.  The generator name is kept in the
        filename purely so that a cache directory can be skimmed by eye.
    """
    root = CACHE_ROOT if root is None else Path(root)
    return root / f"{type(cfg).name}-{config_hash(cfg)}{suffix}"


def save_arrays(
    cfg: Any,
    arrays: dict[str, np.ndarray],
    root: Path | str | None = None,
) -> Path:
    """Store arrays under the config's cache key.

    Writing is atomic: the archive is written to a temporary file in the same
    directory and then renamed, so an interrupted run cannot leave a truncated
    ``.npz`` that a later run would happily load.

    Parameters
    ----------
    cfg : dataclass instance
        Config that produced the arrays.
    arrays : dict of str to ndarray
        Arrays to store.  The key :data:`CONFIG_KEY` is reserved.
    root : path-like, optional
        Cache directory; defaults to :data:`CACHE_ROOT`.

    Returns
    -------
    pathlib.Path
        Path of the written ``.npz``.

    Raises
    ------
    KeyError
        If ``arrays`` uses the reserved config key.
    """
    if CONFIG_KEY in arrays:
        raise KeyError(f"{CONFIG_KEY!r} is reserved for the stored config")
    path = cache_path(cfg, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = config_to_json(cfg)
    stored = {key: _to_storage_dtype(value) for key, value in arrays.items()}
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    np.savez_compressed(tmp, **stored, **{CONFIG_KEY: np.asarray(payload)})
    # np.savez appends ".npz" when the name does not already end in it.
    written = tmp if tmp.exists() else tmp.with_suffix(tmp.suffix + ".npz")
    os.replace(written, path)
    cache_path(cfg, root, suffix=".json").write_text(payload, encoding="utf-8")
    return path


def load_arrays(
    cfg: Any,
    root: Path | str | None = None,
) -> dict[str, np.ndarray] | None:
    """Load the arrays cached under a config, if any.

    The config stored inside the archive is compared against ``cfg`` and a
    mismatch raises: that would mean a hash collision or a hand-edited cache
    file, and silently returning the wrong trajectories is exactly the failure
    this project cannot afford.

    Parameters
    ----------
    cfg : dataclass instance
        Config whose cache entry is wanted.
    root : path-like, optional
        Cache directory; defaults to :data:`CACHE_ROOT`.

    Returns
    -------
    dict of str to ndarray or None
        The stored arrays without the config entry, or ``None`` on a miss.

    Raises
    ------
    ValueError
        If the stored config does not match ``cfg``.
    """
    path = cache_path(cfg, root)
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as handle:
        stored = str(handle[CONFIG_KEY].item()) if CONFIG_KEY in handle else None
        arrays = {key: handle[key] for key in handle.files if key != CONFIG_KEY}
    if stored is not None and config_from_json(stored) != cfg:
        raise ValueError(
            f"cache entry {path.name} stores a different config than requested; "
            "delete it rather than trusting it"
        )
    return arrays


def cached(
    cfg: Any,
    compute: Callable[[], dict[str, np.ndarray]],
    root: Path | str | None = None,
    use_cache: bool = True,
) -> dict[str, np.ndarray]:
    """Return cached arrays for a config, computing and storing them on a miss.

    Parameters
    ----------
    cfg : dataclass instance
        Config identifying the result.
    compute : callable
        Zero-argument callable returning the arrays to cache.  It must be a
        deterministic function of ``cfg`` alone, seed included, or the cache
        breaks reproducibility rather than supporting it.
    root : path-like, optional
        Cache directory; defaults to :data:`CACHE_ROOT`.
    use_cache : bool, optional
        If ``False``, always recompute and do not read or write the cache.

    Returns
    -------
    dict of str to ndarray
        The arrays, from disk or freshly computed.
    """
    if not use_cache:
        return compute()
    hit = load_arrays(cfg, root)
    if hit is not None:
        return hit
    arrays = compute()
    save_arrays(cfg, arrays, root)
    return arrays


def clear_cache(root: Path | str | None = None) -> int:
    """Delete every cache entry under ``root``.

    Parameters
    ----------
    root : path-like, optional
        Cache directory; defaults to :data:`CACHE_ROOT`.

    Returns
    -------
    int
        Number of files removed.  Used by the "reproducible from a clean cache"
        gate between phases (``PROJECT.md`` section 6).
    """
    root = CACHE_ROOT if root is None else Path(root)
    if not root.is_dir():
        return 0
    removed = 0
    for path in list(root.glob("*.npz")) + list(root.glob("*.json")):
        path.unlink()
        removed += 1
    return removed
