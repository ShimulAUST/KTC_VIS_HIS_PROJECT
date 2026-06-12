"""Measure CUQI8 reconstruction runtime per level and patch it into the HDF5 cache.

Runs ONE sample per container to minimise peak RAM usage (CUQI8's FEM solver
is memory-heavy and OOM-kills when all samples run together on Mac).

Usage:
    python scripts/benchmark_runtime_cuqi8.py                   # levels 1-7, all samples
    python scripts/benchmark_runtime_cuqi8.py --levels 1 2      # only L1, L2
    python scripts/benchmark_runtime_cuqi8.py --samples a b     # only samples A, B
    python scripts/benchmark_runtime_cuqi8.py --memory 10g      # override RAM cap (default 8g)
    python scripts/benchmark_runtime_cuqi8.py --dry-run         # print commands, don't run
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import h5py
import numpy as np
import scipy.io

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CACHE_PATH = _PROJECT_ROOT / "data" / "cache" / "results.h5"
_MEASUREMENTS_DIR = _PROJECT_ROOT / "data" / "raw" / "ktc2023" / "measurements"

_IMAGE = "muzammal5566/ktc2023-cuqi8:latest"
_DEFAULT_MEMORY = "8g"

_INDEX_TO_SAMPLE = {1: "a", 2: "b", 3: "c", 4: "d"}
_SAMPLE_TO_INDEX = {v: k for k, v in _INDEX_TO_SAMPLE.items()}
_ALL_SAMPLES = ["a", "b", "c", "d"]


def _write_training_data(training_dir: Path, level: int, sample: str) -> None:
    """Write ref.mat + one data<idx>.mat for a single sample."""
    sys.path.insert(0, str(_PROJECT_ROOT))
    from ktc_vis.data.loader import KTCDataLoader
    from ktc_vis.data.subsampler import subsample_measurement

    loader = KTCDataLoader()
    shutil.copy2(_MEASUREMENTS_DIR / "ref.mat", training_dir / "ref.mat")

    idx = _SAMPLE_TO_INDEX[sample]
    m = loader.load(1, sample)
    if level != 1:
        m = subsample_measurement(m, level)
    scipy.io.savemat(
        str(training_dir / f"data{idx}.mat"),
        {
            "Inj": m.current_matrix.T,
            "Mpat": getattr(m, "mpat", np.zeros((32, 31))),
            "Uel": m.voltage_matrix.flatten(),
        },
    )


def _docker_cmd(
    training_dir: Path, output_dir: Path, level: int, memory: str | None, swap: str | None
) -> list[str]:
    cmd = [
        "docker", "run", "--rm",
        "--platform", "linux/amd64",
        "--shm-size", "512m",
        "-v", f"{training_dir}:/app/TrainingData",
        "-v", f"{output_dir}:/app/Output",
    ]
    if memory:
        cmd += ["--memory", memory]
        if swap:
            cmd += ["--memory-swap", swap]
    cmd += [
        _IMAGE,
        "conda", "run", "--no-capture-output", "-n", "env",
        "python", "main.py", "TrainingData", "Output", str(level),
    ]
    return cmd


def _patch_runtime(level: int, sample: str, runtime_s: float) -> bool:
    if not _CACHE_PATH.exists():
        print(f"  ! cache not found: {_CACHE_PATH} — skipping write")
        return False
    with h5py.File(str(_CACHE_PATH), "a") as f:
        group = f"results/cuqi8/{level}/{sample}"
        if group not in f:
            print(f"  ! no cache entry for {group} — skipping")
            return False
        grp = f[group]
        if "runtime" in grp:
            del grp["runtime"]
        grp.create_dataset("runtime", data=float(runtime_s))
    return True


def _fmt(s: float) -> str:
    return f"{s:.2f}s" if s < 60 else f"{int(s // 60)}m {int(s) % 60:02d}s"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--levels", nargs="+", type=int, choices=range(1, 8),
                        default=list(range(1, 8)))
    parser.add_argument("--samples", nargs="+", choices=_ALL_SAMPLES,
                        default=_ALL_SAMPLES)
    parser.add_argument("--memory", default=None,
                        help="Docker memory limit (e.g. 11g). "
                             "Omit to let Docker use all available VM memory (recommended on Mac).")
    parser.add_argument("--swap", default=None,
                        help="Docker memory+swap total (e.g. 20g). "
                             "Allows spilling to disk when RAM is exhausted. "
                             "Must be >= --memory.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the docker command without executing.")
    parser.add_argument("--no-cache-update", action="store_true",
                        help="Measure timing but do not write to the HDF5 cache.")
    args = parser.parse_args()

    print(f"CUQI8 runtime benchmark — image: {_IMAGE}")
    print(f"Cache:  {_CACHE_PATH}")
    mem_info = args.memory or "unlimited (Docker VM ceiling)"
    swap_info = f"  |  swap ceiling: {args.swap}" if args.swap else ""
    print(f"Memory: {mem_info} per container  |  shm: 512m{swap_info}")
    print(f"Mode:   per-sample (one container per sample to minimise peak RAM)")
    print("─" * 60)

    bench_start = time.perf_counter()
    total_patched = 0

    for level in args.levels:
        print(f"\n  Level {level}")
        for sample in args.samples:
            with tempfile.TemporaryDirectory(prefix=f"cuqi8_L{level}{sample}_") as tmp:
                tmp_path = Path(tmp)
                training_dir = tmp_path / "TrainingData"
                output_dir = tmp_path / "Output"
                training_dir.mkdir()
                output_dir.mkdir()

                _write_training_data(training_dir, level, sample)

                cmd = _docker_cmd(training_dir, output_dir, level, args.memory, args.swap)

                if args.dry_run:
                    print(f"    {sample.upper()}  " + " ".join(cmd))
                    continue

                print(f"    {sample.upper()}  ⏳ docker run … ", end="", flush=True)
                t0 = time.perf_counter()
                try:
                    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
                except FileNotFoundError:
                    print("✗  docker not found on PATH")
                    sys.exit(1)
                elapsed = time.perf_counter() - t0

                if result.returncode != 0:
                    hint = " (OOM — try --memory 12g)" if result.returncode == 137 else ""
                    print(f"✗  exit={result.returncode} after {_fmt(elapsed)}{hint}")
                    stderr = (result.stderr or "").strip()
                    if stderr:
                        for line in stderr.splitlines()[-10:]:
                            print(f"      {line}")
                    continue

                print(f"✓  {_fmt(elapsed)}")

                if not args.no_cache_update:
                    if _patch_runtime(level, sample, elapsed):
                        total_patched += 1
                        print(f"      wrote runtime={elapsed:.3f}s to cache")

    total = _fmt(time.perf_counter() - bench_start)
    print("\n" + "─" * 60)
    print(f"Done in {total}. Updated {total_patched} runtime values in cache.")


if __name__ == "__main__":
    main()
