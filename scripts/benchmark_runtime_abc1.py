"""Measure ABC1 reconstruction runtime per level and patch it into the HDF5 cache.

Mirrors the manual docker invocation:

    docker run --rm -v "$PWD:/workspace" -w /workspace/ktc_vis/external/KTC2023_ABC \
        ktc2023-abc-python \
        python main_python.py TrainingData \
        /workspace/data/raw/ktc2023/reference_outputs/abc1/level<N> <N>

For each level the script:
  1. Times the docker run with time.perf_counter().
  2. Detects which samples the container wrote (data1.mat … data4.mat).
  3. Writes runtime = elapsed / num_samples into
     results/abc1/<level>/<sample>/runtime in the HDF5 cache.

Usage:
    python scripts/benchmark_runtime_abc1.py                  # levels 1-7
    python scripts/benchmark_runtime_abc1.py --levels 1 2     # only L1, L2
    python scripts/benchmark_runtime_abc1.py --dry-run        # print commands, don't run
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

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CACHE_PATH = _PROJECT_ROOT / "data" / "cache" / "results.h5"
_OUTPUT_ROOT_HOST = _PROJECT_ROOT / "data" / "raw" / "ktc2023" / "reference_outputs" / "abc1"
_TRAINING_DATA_HOST = _PROJECT_ROOT / "ktc_vis" / "external" / "KTC2023_ABC" / "TrainingData"

_IMAGE = "ktc2023-abc-python"
_ABC_PATH_IN_CONTAINER = "/workspace/ktc_vis/external/KTC2023_ABC"
_OUTPUT_ROOT_IN_CONTAINER = "/workspace/data/raw/ktc2023/reference_outputs/abc1"

# data<N>.mat → sample letter
_INDEX_TO_SAMPLE = {1: "a", 2: "b", 3: "c", 4: "d"}
_SAMPLE_TO_INDEX = {v: k for k, v in _INDEX_TO_SAMPLE.items()}


def _docker_cmd(level: int) -> list[str]:
    out_dir = f"{_OUTPUT_ROOT_IN_CONTAINER}/level{level}"
    return [
        "docker", "run", "--rm",
        "-v", f"{_PROJECT_ROOT}:/workspace",
        "-w", _ABC_PATH_IN_CONTAINER,
        _IMAGE,
        "python", "main_python.py", "TrainingData", out_dir, str(level),
    ]


def _docker_cmd_single_sample(level: int, sample_idx: int, host_training_dir: Path) -> list[str]:
    """Docker command that processes only ONE data<idx>.mat by mounting a temp
    TrainingData directory containing just that sample + ref.mat."""
    out_dir = f"{_OUTPUT_ROOT_IN_CONTAINER}/level{level}"
    return [
        "docker", "run", "--rm",
        "-v", f"{_PROJECT_ROOT}:/workspace",
        "-v", f"{host_training_dir}:/tmp/td_one",
        "-w", _ABC_PATH_IN_CONTAINER,
        _IMAGE,
        "python", "main_python.py", "/tmp/td_one", out_dir, str(level),
    ]


def _detect_samples(level: int) -> list[str]:
    level_dir = _OUTPUT_ROOT_HOST / f"level{level}"
    samples: list[str] = []
    for idx, sample in _INDEX_TO_SAMPLE.items():
        if (level_dir / f"data{idx}.mat").exists():
            samples.append(sample)
    return samples


def _patch_runtime(level: int, samples: list[str], per_sample_runtime: float) -> int:
    if not _CACHE_PATH.exists():
        print(f"  ! cache not found: {_CACHE_PATH} — skipping write")
        return 0
    patched = 0
    with h5py.File(str(_CACHE_PATH), "a") as f:
        for sample in samples:
            group = f"results/abc1/{level}/{sample}"
            if group not in f:
                print(f"  ! no cache entry for {group} — skipping")
                continue
            grp = f[group]
            if "runtime" in grp:
                del grp["runtime"]
            grp.create_dataset("runtime", data=float(per_sample_runtime))
            patched += 1
    return patched


def _patch_one(level: int, sample: str, runtime_s: float) -> bool:
    """Write a single (level, sample) runtime to the cache. Returns True on success."""
    if not _CACHE_PATH.exists():
        print(f"  ! cache not found: {_CACHE_PATH} — skipping write")
        return False
    with h5py.File(str(_CACHE_PATH), "a") as f:
        group = f"results/abc1/{level}/{sample}"
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
    parser.add_argument("--samples", nargs="+", choices=["a", "b", "c", "d"],
                        default=["a", "b", "c", "d"],
                        help="In per-sample mode: which samples to time (default: all).")
    parser.add_argument("--per-sample", action="store_true",
                        help="Run ONE data<idx>.mat per container, isolating per-sample runtime. "
                             "~4x slower than the default per-level batched mode.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the docker command for each level without executing.")
    parser.add_argument("--no-cache-update", action="store_true",
                        help="Run docker and measure timing but do not write to the HDF5 cache.")
    args = parser.parse_args()

    if args.per_sample:
        _run_per_sample(args)
        return

    print(f"ABC1 runtime benchmark — image: {_IMAGE}")
    print(f"Cache: {_CACHE_PATH}")
    print(f"Output dir per level: {_OUTPUT_ROOT_HOST / 'level<N>'}")
    print("─" * 60)

    bench_start = time.perf_counter()
    total_patched = 0

    for level in args.levels:
        cmd = _docker_cmd(level)
        print(f"\n  Level {level}")
        if args.dry_run:
            print("    " + " ".join(cmd))
            continue

        print(f"    ⏳ docker run … ", end="", flush=True)
        t0 = time.perf_counter()
        try:
            result = subprocess.run(cmd, check=False, capture_output=True, text=True)
        except FileNotFoundError:
            print("✗  docker executable not found on PATH")
            sys.exit(1)
        elapsed = time.perf_counter() - t0

        if result.returncode != 0:
            print(f"✗  exit={result.returncode} after {_fmt(elapsed)}")
            stderr = (result.stderr or "").strip()
            if stderr:
                print("    --- stderr (last 20 lines) ---")
                for line in stderr.splitlines()[-20:]:
                    print(f"    {line}")
                print("    ---")
            continue

        samples = _detect_samples(level)
        n = len(samples)
        per_sample = elapsed / n if n else float("nan")
        print(f"✓  {_fmt(elapsed)}  ({n} samples → {per_sample:.2f}s each)")

        if not args.no_cache_update:
            patched = _patch_runtime(level, samples, per_sample)
            total_patched += patched
            print(f"    wrote runtime={per_sample:.3f}s to {patched} cache entries")

    total = _fmt(time.perf_counter() - bench_start)
    print("\n" + "─" * 60)
    print(f"Done in {total}. Updated {total_patched} runtime values in cache.")


def _run_per_sample(args) -> None:
    """Per-sample mode: one container per (level, sample), recording real per-sample timing."""
    print(f"ABC1 runtime benchmark — image: {_IMAGE}  [PER-SAMPLE mode]")
    print(f"Cache: {_CACHE_PATH}")
    print(f"Source data: {_TRAINING_DATA_HOST}")
    print("─" * 60)

    ref_src = _TRAINING_DATA_HOST / "ref.mat"
    if not ref_src.exists():
        print(f"✗ missing required ref.mat at {ref_src}")
        sys.exit(1)

    bench_start = time.perf_counter()
    total_patched = 0

    for level in args.levels:
        print(f"\n  Level {level}")
        for sample in args.samples:
            idx = _SAMPLE_TO_INDEX[sample]
            data_src = _TRAINING_DATA_HOST / f"data{idx}.mat"
            if not data_src.exists():
                print(f"    {sample.upper()}  ✗  missing source: {data_src}")
                continue

            with tempfile.TemporaryDirectory(prefix="abc1_td_") as td:
                td_path = Path(td)
                shutil.copy2(ref_src, td_path / "ref.mat")
                shutil.copy2(data_src, td_path / f"data{idx}.mat")
                cmd = _docker_cmd_single_sample(level, idx, td_path)

                if args.dry_run:
                    print(f"    {sample.upper()}  dry-run:")
                    print("      " + " ".join(cmd))
                    continue

                print(f"    {sample.upper()}  ⏳ docker run … ", end="", flush=True)
                t0 = time.perf_counter()
                try:
                    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
                except FileNotFoundError:
                    print("✗ docker executable not found on PATH")
                    sys.exit(1)
                elapsed = time.perf_counter() - t0

                if result.returncode != 0:
                    print(f"✗  exit={result.returncode} after {_fmt(elapsed)}")
                    stderr = (result.stderr or "").strip()
                    if stderr:
                        for line in stderr.splitlines()[-10:]:
                            print(f"      {line}")
                    continue

                print(f"✓  {_fmt(elapsed)}")
                if not args.no_cache_update and _patch_one(level, sample, elapsed):
                    total_patched += 1

    total = _fmt(time.perf_counter() - bench_start)
    print("\n" + "─" * 60)
    print(f"Done in {total}. Updated {total_patched} per-sample runtime values in cache.")


if __name__ == "__main__":
    main()
