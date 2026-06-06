"""Measure PNPE2E reconstruction runtime per level and patch it into the HDF5 cache.

Mirrors the manual docker invocation (CPU mode via map_location monkey-patch):

    docker run --rm -v "$PWD:/workspace" -w /workspace/ktc_vis/external/KTC2023_PNPE2E \
        ktc2023-pnpe2e \
        bash -lc "mkdir -p <out> && python -c \"<load-on-cpu shim>; from main import main; main('TrainingData', '<out>', <N>)\""

For each level the script:
  1. Times the docker run with time.perf_counter().
  2. Detects which samples the container wrote.
  3. Writes runtime = elapsed / num_samples into
     results/pnpe2e/<level>/<sample>/runtime in the HDF5 cache.

Usage:
    python scripts/benchmark_runtime_pnpe2e.py                  # levels 1-7
    python scripts/benchmark_runtime_pnpe2e.py --levels 1 2     # only L1, L2
    python scripts/benchmark_runtime_pnpe2e.py --dry-run        # print commands, don't run
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import h5py

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CACHE_PATH = _PROJECT_ROOT / "data" / "cache" / "results.h5"
_OUTPUT_ROOT_HOST = _PROJECT_ROOT / "data" / "raw" / "ktc2023" / "reference_outputs" / "pnpe2e"

_IMAGE = "ktc2023-pnpe2e"
_PNPE_PATH_IN_CONTAINER = "/workspace/ktc_vis/external/KTC2023_PNPE2E"
_OUTPUT_ROOT_IN_CONTAINER = "/workspace/data/raw/ktc2023/reference_outputs/pnpe2e"

# Filename convention in pnpe2e outputs (matches reference_adapter._FILENAME_FORMAT["pnpe2e"]).
# <idx>.mat → sample letter
_INDEX_TO_SAMPLE = {1: "a", 2: "b", 3: "c", 4: "d"}

# CPU shim: force every torch.load() to map to CPU when caller did not specify map_location.
_CPU_LOAD_SHIM = (
    "import torch; "
    "_old=torch.load; "
    "torch.load=lambda *a,**k: _old(*a, map_location=torch.device('cpu'), **k) "
    "if 'map_location' not in k else _old(*a, **k); "
)


def _python_payload(level: int) -> str:
    out_dir = f"{_OUTPUT_ROOT_IN_CONTAINER}/level{level}"
    return (
        f"{_CPU_LOAD_SHIM}"
        f"from main import main; "
        f"main('TrainingData', '{out_dir}', {level})"
    )


def _bash_payload(level: int) -> str:
    out_dir = f"{_OUTPUT_ROOT_IN_CONTAINER}/level{level}"
    py = _python_payload(level).replace('"', r'\"')
    return f'mkdir -p {out_dir} && python -c "{py}"'


def _docker_cmd(level: int) -> list[str]:
    return [
        "docker", "run", "--rm",
        "-v", f"{_PROJECT_ROOT}:/workspace",
        "-w", _PNPE_PATH_IN_CONTAINER,
        _IMAGE,
        "bash", "-lc", _bash_payload(level),
    ]


def _detect_samples(level: int) -> list[str]:
    level_dir = _OUTPUT_ROOT_HOST / f"level{level}"
    samples: list[str] = []
    for idx, sample in _INDEX_TO_SAMPLE.items():
        if (level_dir / f"{idx}.mat").exists():
            samples.append(sample)
    return samples


def _patch_runtime(level: int, samples: list[str], per_sample_runtime: float) -> int:
    if not _CACHE_PATH.exists():
        print(f"  ! cache not found: {_CACHE_PATH} — skipping write")
        return 0
    patched = 0
    with h5py.File(str(_CACHE_PATH), "a") as f:
        for sample in samples:
            group = f"results/pnpe2e/{level}/{sample}"
            if group not in f:
                print(f"  ! no cache entry for {group} — skipping")
                continue
            grp = f[group]
            if "runtime" in grp:
                del grp["runtime"]
            grp.create_dataset("runtime", data=float(per_sample_runtime))
            patched += 1
    return patched


def _fmt(s: float) -> str:
    return f"{s:.2f}s" if s < 60 else f"{int(s // 60)}m {int(s) % 60:02d}s"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--levels", nargs="+", type=int, choices=range(1, 8),
                        default=list(range(1, 8)))
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the docker command for each level without executing.")
    parser.add_argument("--no-cache-update", action="store_true",
                        help="Run docker and measure timing but do not write to the HDF5 cache.")
    args = parser.parse_args()

    print(f"PNPE2E runtime benchmark — image: {_IMAGE} (CPU mode)")
    print(f"Cache: {_CACHE_PATH}")
    print(f"Output dir per level: {_OUTPUT_ROOT_HOST / 'level<N>'}")
    print("─" * 60)

    bench_start = time.perf_counter()
    total_patched = 0

    for level in args.levels:
        cmd = _docker_cmd(level)
        print(f"\n  Level {level}")
        if args.dry_run:
            # Print a readable form (the bash payload as one shell-quoted string).
            shown = cmd[:-1] + [f'"{cmd[-1]}"']
            print("    " + " ".join(shown))
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
                print("    --- stderr (last 25 lines) ---")
                for line in stderr.splitlines()[-25:]:
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


if __name__ == "__main__":
    main()
