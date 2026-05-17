"""HDF5 results cache. Owner: Smit Savani."""

from pathlib import Path

import h5py
import numpy as np


class CacheMiss(KeyError):
    """Raised when a requested result is not in the HDF5 cache."""


def _group_path(algorithm: str, level: int, sample: str) -> str:
    return f"results/{algorithm}/{level}/{sample}"


def save_result(
    algorithm: str,
    level: int,
    sample: str,
    metrics: dict,
    reconstruction: np.ndarray,
    cache_path: str | Path = "data/cache/results.h5",
) -> None:
    """Write metrics and reconstruction to the HDF5 cache.

    Args:
        algorithm: Algorithm name, e.g. "abc1".
        level: Difficulty level 1–7.
        sample: Sample identifier "a", "b", or "c".
        metrics: Dict of metric name → scalar float value.
        reconstruction: 256×256 uint8 segmentation array.
        cache_path: Path to the HDF5 file.
    """
    Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
    group = _group_path(algorithm, level, sample)

    with h5py.File(str(cache_path), "a") as f:
        if group in f:
            del f[group]
        grp = f.require_group(group)
        grp.create_dataset("reconstruction", data=reconstruction, compression="gzip")
        for key, value in metrics.items():
            grp.create_dataset(key, data=float(value))


def load_result(
    algorithm: str,
    level: int,
    sample: str,
    cache_path: str | Path = "data/cache/results.h5",
) -> tuple[dict, np.ndarray]:
    """Read metrics and reconstruction from the HDF5 cache.

    Args:
        algorithm: Algorithm name.
        level: Difficulty level 1–7.
        sample: Sample identifier.
        cache_path: Path to the HDF5 file.

    Returns:
        Tuple of (metrics_dict, reconstruction_array).

    Raises:
        CacheMiss: If the result is not cached.
    """
    group = _group_path(algorithm, level, sample)
    cache_path = Path(cache_path)

    if not cache_path.exists():
        raise CacheMiss(f"Cache file not found: {cache_path}")

    with h5py.File(str(cache_path), "r") as f:
        if group not in f:
            raise CacheMiss(f"No cached result for {algorithm}/level{level}/{sample}")
        grp = f[group]
        reconstruction = grp["reconstruction"][()]
        metrics = {
            k: float(grp[k][()])
            for k in grp.keys()
            if k != "reconstruction"
        }

    return metrics, reconstruction


def is_cached(
    algorithm: str,
    level: int,
    sample: str,
    cache_path: str | Path = "data/cache/results.h5",
) -> bool:
    """Return True if a result exists in the cache for this combination."""
    cache_path = Path(cache_path)
    if not cache_path.exists():
        return False
    group = _group_path(algorithm, level, sample)
    with h5py.File(str(cache_path), "r") as f:
        return group in f
