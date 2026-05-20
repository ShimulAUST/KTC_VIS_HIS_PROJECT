"""Run the full benchmark and populate the HDF5 cache.

Usage:
    python scripts/run_benchmark.py                        # all algorithms
    python scripts/run_benchmark.py --algorithms abc1      # one algorithm
    python scripts/run_benchmark.py --levels 1 2 3         # specific levels
    python scripts/run_benchmark.py --overwrite            # recompute cached
    python scripts/run_benchmark.py --stream               # show docker output
    python scripts/run_benchmark.py --timeout 900          # 15-min container limit
    python scripts/run_benchmark.py --no-batch             # disable level batching
"""

import argparse
import sys
import time
import traceback
from collections import deque
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from ktc_vis.adapters.abc1_adapter import ABC1Adapter
from ktc_vis.adapters.cuqi8_adapter import CUQI8Adapter
from ktc_vis.adapters.pnpe2e_adapter import PNPE2EAdapter
from ktc_vis.cache.hdf5_store import is_cached, save_result, load_result
from ktc_vis.data.loader import KTCDataLoader
from ktc_vis.metrics.engine import MetricsEngine

ADAPTER_MAP = {
    "abc1": ABC1Adapter,
    "cuqi8": CUQI8Adapter,
    "pnpe2e": PNPE2EAdapter,
}


def _fmt_seconds(s: float) -> str:
    s = int(s)
    if s < 60:
        return f"{s}s"
    m, sec = divmod(s, 60)
    if m < 60:
        return f"{m}m {sec:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h {m:02d}m"


def _eta_str(run_times: deque, remaining: int) -> str:
    if not run_times or remaining == 0:
        return "?"
    avg = sum(run_times) / len(run_times)
    return _fmt_seconds(avg * remaining)


def run_benchmark(
    algorithms: list[str],
    levels: list[int],
    samples: list[str],
    cache_path: Path,
    overwrite: bool,
    timeout: int,
    stream: bool,
    batch: bool,
) -> None:
    loader = KTCDataLoader()
    bench_start = time.perf_counter()
    run_times: deque = deque(maxlen=10)  # rolling window for ETA

    # ── Platform warning on Apple Silicon ────────────────────────────────────
    import platform
    if platform.machine() == "arm64":
        print(
            "\n  NOTE: Apple Silicon detected. CUQI8 and PNPE2E images are x86_64-only\n"
            "  and run under Rosetta 2 emulation — expect 2–5× slower than native.\n"
            "  ABC1 runs natively (arm64 image available).\n"
        )

    # ── Count total docker runs (batching collapses samples per level) ───────
    total_docker_runs = 0
    for alg_name in algorithms:
        adapter_cls = ADAPTER_MAP[alg_name]
        use_batch = batch and adapter_cls.supports_level_batching
        if use_batch:
            for level in levels:
                uncached = [s for s in samples if overwrite or not is_cached(alg_name, level, s, cache_path)]
                if uncached:
                    total_docker_runs += 1  # one docker run per level
        else:
            for level in levels:
                for sample in samples:
                    if overwrite or not is_cached(alg_name, level, sample, cache_path):
                        total_docker_runs += 1

    total_combos = len(algorithms) * len(levels) * len(samples)
    print(
        f"\nBenchmark: {len(algorithms)} algorithm(s) × {len(levels)} levels × "
        f"{len(samples)} samples = {total_combos} results\n"
        f"Docker runs needed: {total_docker_runs}  "
        f"(level-batching {'ON' if batch else 'OFF'}  |  "
        f"timeout {timeout}s  |  stream {'ON' if stream else 'OFF'})\n"
        f"{'─' * 60}"
    )

    docker_run_idx = 0
    failed = []

    for alg_name in algorithms:
        adapter = ADAPTER_MAP[alg_name](timeout=timeout, stream=stream)
        engine = MetricsEngine(adapter, cache_path)
        use_batch = batch and adapter.supports_level_batching

        print(f"\n  Algorithm: {alg_name.upper()}")

        for level in levels:
            uncached_samples = [
                s for s in samples
                if overwrite or not is_cached(alg_name, level, s, cache_path)
            ]
            cached_samples = [s for s in samples if s not in uncached_samples]

            # ── Show cached skips ─────────────────────────────────────────
            for sample in cached_samples:
                print(f"    L{level}/{sample.upper()}  ↩  cached — skipped")

            if not uncached_samples:
                continue

            if use_batch:
                # ── Level batch: one docker run → all samples ─────────────
                docker_run_idx += 1
                elapsed_total = _fmt_seconds(time.perf_counter() - bench_start)
                eta = _eta_str(run_times, total_docker_runs - docker_run_idx + 1)
                tag = (
                    f"    L{level}/[{','.join(s.upper() for s in uncached_samples)}]  "
                    f"[docker {docker_run_idx}/{total_docker_runs}]  "
                    f"elapsed {elapsed_total} | ETA ~{eta}"
                )
                print(f"{tag}")
                print(f"    ⏳ running {alg_name.upper()} level {level} (batch)...", flush=True)

                t0 = time.perf_counter()
                try:
                    recons = adapter.reconstruct_level(level, uncached_samples)
                    docker_time = time.perf_counter() - t0
                    run_times.append(docker_time)

                    # Compute metrics and cache each sample
                    for sample in uncached_samples:
                        measurement = loader.load(level, sample)
                        metrics = engine._compute_metrics_from_reconstruction(
                            measurement, recons[sample], docker_time / len(uncached_samples)
                        )
                        save_result(alg_name, level, sample, metrics, recons[sample], cache_path)
                        ssim = metrics.get("ssim", float("nan"))
                        print(
                            f"    ✓  L{level}/{sample.upper()}  SSIM={ssim:.3f}  "
                            f"(batch t={_fmt_seconds(docker_time)})"
                        )
                except Exception as e:
                    docker_time = time.perf_counter() - t0
                    stderr = getattr(e, "stderr", "") or ""
                    print(f"    ✗  BATCH FAILED ({_fmt_seconds(docker_time)}): {e}")
                    if stderr:
                        print(f"    --- container stderr ---\n{stderr}\n    ---")
                    if "--verbose" in sys.argv:
                        traceback.print_exc()
                    for sample in uncached_samples:
                        failed.append((alg_name, level, sample, str(e)))

            else:
                # ── Per-sample mode ───────────────────────────────────────
                for sample in uncached_samples:
                    docker_run_idx += 1
                    elapsed_total = _fmt_seconds(time.perf_counter() - bench_start)
                    eta = _eta_str(run_times, total_docker_runs - docker_run_idx + 1)
                    tag = (
                        f"    L{level}/{sample.upper()}  "
                        f"[{docker_run_idx}/{total_docker_runs}]  "
                        f"elapsed {elapsed_total} | ETA ~{eta}"
                    )
                    print(f"{tag}  ⏳ running...", end="  ", flush=True)

                    t0 = time.perf_counter()
                    try:
                        measurement = loader.load(level, sample)
                        metrics = engine.compute_all(measurement, overwrite=overwrite)
                        elapsed = time.perf_counter() - t0
                        run_times.append(elapsed)
                        ssim = metrics.get("ssim", float("nan"))
                        print(f"✓  SSIM={ssim:.3f}  t={_fmt_seconds(elapsed)}")
                    except Exception as e:
                        elapsed = time.perf_counter() - t0
                        stderr = getattr(e, "stderr", "") or ""
                        print(f"✗  FAILED ({_fmt_seconds(elapsed)}): {e}")
                        if stderr:
                            print(f"    --- container stderr ---\n{stderr}\n    ---")
                        if "--verbose" in sys.argv:
                            traceback.print_exc()
                        failed.append((alg_name, level, sample, str(e)))

    total_elapsed = _fmt_seconds(time.perf_counter() - bench_start)
    success = total_combos - len(failed)
    print(f"\n{'─' * 60}")
    print(f"Done in {total_elapsed}. {success}/{total_combos} results cached.")
    if failed:
        print(f"\nFailed ({len(failed)}):")
        for alg, lvl, smp, err in failed:
            print(f"  {alg.upper()} L{lvl}/{smp.upper()}: {err}")
    print(f"Cache: {cache_path}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="KTC-Vis benchmark runner")
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--algorithms", nargs="+", choices=list(ADAPTER_MAP))
    parser.add_argument("--levels", nargs="+", type=int, choices=range(1, 8))
    parser.add_argument("--samples", nargs="+", choices=["a", "b", "c"])
    parser.add_argument("--overwrite", action="store_true",
                        help="Recompute results even if already cached")
    parser.add_argument("--stream", action="store_true",
                        help="Stream docker container stdout/stderr to terminal")
    parser.add_argument("--timeout", type=int, default=600,
                        help="Seconds before a docker container is killed (default: 600)")
    parser.add_argument("--no-batch", dest="batch", action="store_false",
                        help="Disable level batching (run one docker per sample)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print full tracebacks on failure")
    parser.set_defaults(batch=True)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    algorithms = args.algorithms or config["benchmark"]["algorithms"]
    levels = args.levels or config["benchmark"]["levels"]
    samples = args.samples or config["benchmark"]["samples"]
    cache_path = Path(config["data"]["cache_path"])

    run_benchmark(
        algorithms=algorithms,
        levels=levels,
        samples=samples,
        cache_path=cache_path,
        overwrite=args.overwrite,
        timeout=args.timeout,
        stream=args.stream,
        batch=args.batch,
    )


if __name__ == "__main__":
    main()
