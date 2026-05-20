"""Populate the HDF5 cache from pre-computed reference outputs.

Instead of running Docker containers, this script reads the published
reconstructions already in data/raw/ktc2023/reference_outputs/ and computes
all metrics against the ground truth. Much faster than run_benchmark.py.

Usage:
    python scripts/populate_cache_from_reference.py           # all algorithms
    python scripts/populate_cache_from_reference.py --algorithms abc1 cuqi8
    python scripts/populate_cache_from_reference.py --overwrite
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
from ktc_vis.adapters.reference_adapter import ReferenceOutputAdapter
from ktc_vis.cache.hdf5_store import is_cached, save_result
from ktc_vis.data.loader import KTCDataLoader
from ktc_vis.metrics.engine import MetricsEngine

ALGORITHMS = ["abc1", "cuqi8", "pnpe2e"]
LEVELS = list(range(1, 8))
SAMPLES = ["a", "b", "c", "d"]


def _fmt(s: float) -> str:
    s = int(s)
    return f"{s//60}m {s%60:02d}s" if s >= 60 else f"{s}s"


def main() -> None:
    parser = argparse.ArgumentParser(description="Populate HDF5 cache from reference outputs")
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--algorithms", nargs="+", choices=ALGORITHMS, default=ALGORITHMS)
    parser.add_argument("--levels", nargs="+", type=int, choices=range(1, 8), default=LEVELS)
    parser.add_argument("--samples", nargs="+", choices=["a", "b", "c", "d"], default=SAMPLES)
    parser.add_argument("--overwrite", action="store_true",
                        help="Recompute metrics even if already cached")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    cache_path = Path(config["data"]["cache_path"])

    loader = KTCDataLoader()
    total = len(args.algorithms) * len(args.levels) * len(args.samples)
    done = skipped = failed = 0
    start = time.perf_counter()

    print(f"\nPopulating cache from reference outputs")
    print(f"{len(args.algorithms)} algorithms × {len(args.levels)} levels × "
          f"{len(args.samples)} samples = {total} entries")
    print(f"Cache: {cache_path}\n{'─' * 56}")

    for alg in args.algorithms:
        adapter = ReferenceOutputAdapter(alg)
        engine = MetricsEngine(adapter, cache_path)
        print(f"\n  {alg.upper()}")

        for level in args.levels:
            for sample in args.samples:
                done += 1
                tag = f"    L{level}/{sample.upper()}"

                if not args.overwrite and is_cached(alg, level, sample, cache_path):
                    print(f"{tag}  ↩  already cached")
                    skipped += 1
                    continue

                try:
                    measurement = loader.load(level, sample)
                    t0 = time.perf_counter()
                    recon = adapter.reconstruct(measurement)
                    metrics = engine._compute_metrics_from_reconstruction(
                        measurement, recon, runtime=0.0
                    )
                    save_result(alg, level, sample, metrics, recon, cache_path)
                    elapsed = time.perf_counter() - t0
                    ssim = metrics.get("ssim", float("nan"))
                    print(f"{tag}  ✓  SSIM={ssim:.3f}  ({_fmt(elapsed)})")
                except FileNotFoundError as e:
                    print(f"{tag}  ✗  missing: {e}")
                    failed += 1
                except Exception as e:
                    print(f"{tag}  ✗  {e}")
                    failed += 1

    total_elapsed = _fmt(time.perf_counter() - start)
    success = done - skipped - failed
    print(f"\n{'─' * 56}")
    print(f"Done in {total_elapsed}.")
    print(f"  Cached : {success}")
    print(f"  Skipped: {skipped} (already in cache)")
    if failed:
        print(f"  Failed : {failed}")
    print(f"Cache: {cache_path}\n")


if __name__ == "__main__":
    main()
