"""Run the full benchmark and populate the HDF5 cache.

Usage:
    python scripts/run_benchmark.py                        # all algorithms
    python scripts/run_benchmark.py --algorithms abc1      # one algorithm
    python scripts/run_benchmark.py --levels 1 2 3         # specific levels
    python scripts/run_benchmark.py --overwrite            # recompute cached
"""

import argparse
import sys
import traceback
from pathlib import Path

import yaml

# Make sure project root is on the path when run as a script
sys.path.insert(0, str(Path(__file__).parent.parent))

from ktc_vis.adapters.abc1_adapter import ABC1Adapter
from ktc_vis.adapters.cuqi8_adapter import CUQI8Adapter
from ktc_vis.adapters.pnpe2e_adapter import PNPE2EAdapter
from ktc_vis.cache.hdf5_store import is_cached
from ktc_vis.data.loader import KTCDataLoader
from ktc_vis.metrics.engine import MetricsEngine

ADAPTER_MAP = {
    "abc1": ABC1Adapter,
    "cuqi8": CUQI8Adapter,
    "pnpe2e": PNPE2EAdapter,
}


def run_benchmark(
    algorithms: list[str],
    levels: list[int],
    samples: list[str],
    cache_path: Path,
    overwrite: bool,
) -> None:
    loader = KTCDataLoader()
    total = len(algorithms) * len(levels) * len(samples)
    done = 0
    failed = []

    print(f"\nBenchmark: {len(algorithms)} algorithms × {len(levels)} levels × "
          f"{len(samples)} samples = {total} runs\n")

    for alg_name in algorithms:
        adapter = ADAPTER_MAP[alg_name]()
        engine = MetricsEngine(adapter, cache_path)

        for level in levels:
            for sample in samples:
                done += 1
                tag = f"[{done:>2}/{total}] {alg_name.upper()} L{level}/{sample.upper()}"

                if not overwrite and is_cached(alg_name, level, sample, cache_path):
                    print(f"{tag}  ↩  cached — skipped")
                    continue

                print(f"{tag}  ⏳ running ...", end="", flush=True)
                try:
                    measurement = loader.load(level, sample)
                    metrics = engine.compute_all(measurement, overwrite=overwrite)
                    ssim = metrics.get("ssim", float("nan"))
                    rt = metrics.get("runtime", float("nan"))
                    print(f"  ✓  SSIM={ssim:.3f}  t={rt:.1f}s")
                except Exception as e:
                    print(f"  ✗  FAILED: {e}")
                    failed.append((alg_name, level, sample, str(e)))
                    if "--verbose" in sys.argv:
                        traceback.print_exc()

    print(f"\n{'─'*54}")
    print(f"Done. {total - len(failed)}/{total} succeeded.")
    if failed:
        print(f"\nFailed runs ({len(failed)}):")
        for alg, lvl, smp, err in failed:
            print(f"  {alg.upper()} L{lvl}/{smp.upper()}: {err}")
    print(f"Cache: {cache_path}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="KTC-Vis benchmark runner")
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--algorithms", nargs="+", choices=list(ADAPTER_MAP))
    parser.add_argument("--levels", nargs="+", type=int, choices=range(1, 8))
    parser.add_argument("--samples", nargs="+", choices=["a", "b", "c"])
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    algorithms = args.algorithms or config["benchmark"]["algorithms"]
    levels = args.levels or config["benchmark"]["levels"]
    samples = args.samples or config["benchmark"]["samples"]
    cache_path = Path(config["data"]["cache_path"])

    run_benchmark(algorithms, levels, samples, cache_path, args.overwrite)


if __name__ == "__main__":
    main()
