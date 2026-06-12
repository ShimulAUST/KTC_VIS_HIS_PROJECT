"""Backfill HDF5 cache with every M3 metric and emit a per-algorithm metrics log.

No docker / no algorithm re-runs: every reconstruction is read from the
existing ``data/cache/results.h5`` and the full metric battery (image quality,
shape, class, measurement-domain, runtime) is recomputed via
``MetricsEngine._compute_metrics_from_reconstruction``. Recomputed values are
written back to the same HDF5 file so the dashboard reads them instantly.

The script also emits ``data/cache/metrics_log.md`` — one section per
algorithm, structured like the existing ``runtime_log.md`` so the two logs sit
side-by-side. Every metric row reports per-(level, sample) values plus level
means and an overall mean.

Usage:
    python scripts/backfill_metrics_log.py            # rewrite all 84 entries
    python scripts/backfill_metrics_log.py --dry-run  # report only, don't touch
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ktc_vis.cache.hdf5_store import load_result, save_result
from ktc_vis.data.loader import KTCDataLoader
from ktc_vis.metrics.engine import MetricsEngine

ALGS: tuple[str, ...] = ("abc1", "cuqi8", "pnpe2e")
SAMPLES: tuple[str, ...] = ("a", "b", "c", "d")
LEVELS: tuple[int, ...] = tuple(range(1, 8))

CACHE_PATH = ROOT / "data" / "cache" / "results.h5"
LOG_PATH = ROOT / "data" / "cache" / "metrics_log.md"

# (key, display label, format spec, higher-is-better?)
METRIC_SPEC: list[tuple[str, str, str, bool]] = [
    ("ssim",                   "SSIM",                ".3f",  True),
    ("ssim_min",               "Spatial SSIM (min)",  ".3f",  True),
    ("hausdorff",              "Hausdorff (px)",      ".1f",  False),
    ("position_error",         "Position Err (px)",   ".1f",  False),
    ("resolution",             "Resolution (px)",     ".1f",  False),
    ("confusion_accuracy",     "Confusion Acc.",      ".3f",  True),
    ("iou_mean",               "Mean IoU",            ".3f",  True),
    ("iou_water",              "IoU Water",           ".3f",  True),
    ("iou_resistive",          "IoU Resistive",       ".3f",  True),
    ("iou_conductive",         "IoU Conductive",      ".3f",  True),
    ("dice_mean",              "Mean Dice",           ".3f",  True),
    ("dice_water",             "Dice Water",          ".3f",  True),
    ("dice_resistive",         "Dice Resistive",      ".3f",  True),
    ("dice_conductive",        "Dice Conductive",     ".3f",  True),
    ("voltage_residual",       "Voltage Residual",    ".4f",  False),
    ("resistance_consistency", "Resistance Consist.", ".3f",  True),
    ("current_sensitivity",    "Current Sensitivity", ".3f",  True),
]


class _NoopAdapter:
    """Placeholder adapter — ``MetricsEngine`` never invokes it because we
    feed the cached reconstruction directly to ``_compute_metrics_from_reconstruction``.
    """

    name = "noop"

    def reconstruct(self, measurement):
        raise RuntimeError("Adapter must not be called during backfill")


def _recompute_one(
    engine: MetricsEngine,
    loader: KTCDataLoader,
    alg: str,
    level: int,
    sample: str,
) -> tuple[dict | None, np.ndarray | None]:
    try:
        cached_metrics, reconstruction = load_result(
            alg, level, sample, cache_path=CACHE_PATH
        )
    except Exception as e:
        print(f"  [{alg} L{level}/{sample}] cache miss: {e}")
        return None, None

    if reconstruction is None:
        print(f"  [{alg} L{level}/{sample}] no reconstruction in cache")
        return None, None

    measurement = loader.load(level=level, sample=sample)
    fresh = engine._compute_metrics_from_reconstruction(
        measurement,
        reconstruction.astype(np.uint8),
        runtime=float(cached_metrics.get("runtime", 0.0)),
    )
    return fresh, reconstruction


def backfill(dry_run: bool = False) -> dict[str, dict[tuple[int, str], dict]]:
    """Recompute metrics for every (alg, level, sample) and (optionally) persist.

    Returns:
        Nested mapping ``results[alg][(level, sample)] = metrics_dict``.
    """
    loader = KTCDataLoader(ROOT / "data" / "raw" / "ktc2023")
    engine = MetricsEngine(_NoopAdapter(), cache_path=CACHE_PATH)
    results: dict[str, dict[tuple[int, str], dict]] = {a: {} for a in ALGS}

    t0 = time.perf_counter()
    n_ok = n_skip = 0
    for alg in ALGS:
        print(f"\n[{alg}]")
        for level in LEVELS:
            for sample in SAMPLES:
                metrics, recon = _recompute_one(engine, loader, alg, level, sample)
                if metrics is None or recon is None:
                    n_skip += 1
                    continue
                results[alg][(level, sample)] = metrics
                n_ok += 1
                if not dry_run:
                    save_result(
                        alg, level, sample, metrics, recon, cache_path=CACHE_PATH,
                    )
        print(f"  -> {sum(1 for _ in results[alg])} entries refreshed")

    dt = time.perf_counter() - t0
    print(f"\nDone: {n_ok} refreshed, {n_skip} skipped in {dt:.1f}s "
          f"({'DRY RUN — cache untouched' if dry_run else 'cache updated'})")
    return results


# ── Log emission ────────────────────────────────────────────────────────────────

def _fmt(value: float | None, spec: str) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return " _n/a_"
    return format(float(value), spec)


def _metric_table(
    results_for_alg: dict[tuple[int, str], dict],
    key: str,
    label: str,
    spec: str,
    higher_is_better: bool,
) -> str:
    """Render one metric's L1–L7 × sample-A–D table."""
    arrow = "↑" if higher_is_better else "↓"
    width = max(len(label), 8) + 1
    out: list[str] = []
    out.append(f"### {label} ({arrow})")
    out.append("")
    out.append("| Level | Sample A | Sample B | Sample C | Sample D | Level mean |")
    out.append("|------:|---------:|---------:|---------:|---------:|-----------:|")
    level_means: list[float] = []
    all_vals: list[float] = []
    for level in LEVELS:
        row_vals: list[float | None] = []
        for sample in SAMPLES:
            entry = results_for_alg.get((level, sample))
            v = entry.get(key) if entry else None
            row_vals.append(v)
            if v is not None:
                all_vals.append(float(v))
        present = [v for v in row_vals if v is not None]
        lm = float(np.mean(present)) if present else None
        if lm is not None:
            level_means.append(lm)
        cells = " | ".join(f"{_fmt(v, spec):>8}" for v in row_vals)
        out.append(
            f"|   L{level} | {cells} | "
            f"{_fmt(lm, spec):>10} |"
        )
    if all_vals:
        out.append(
            f"\n**Overall mean across L1–L7:** {format(float(np.mean(all_vals)), spec)} "
            f"(min {format(float(np.min(all_vals)), spec)}, "
            f"max {format(float(np.max(all_vals)), spec)})"
        )
    out.append("")
    return "\n".join(out)


def render_log(results: dict[str, dict[tuple[int, str], dict]]) -> str:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines: list[str] = []
    lines.append("# Reconstruction Metrics Log")
    lines.append("")
    lines.append(f"_Generated: {timestamp}_")
    lines.append("")
    lines.append(
        "All 17 dashboard metrics per (algorithm, level, sample), recomputed from\n"
        "the cached reconstructions in `data/cache/results.h5`. Companion to\n"
        "`runtime_log.md` (which records wall-clock times only)."
    )
    lines.append("")
    lines.append("- `↑` higher is better, `↓` lower is better")
    lines.append("- Measurement-domain metrics use the Born / linearised forward surrogate "
                 "(`ktc_vis.metrics._voltage_surrogate`).")
    lines.append("")

    for alg in ALGS:
        lines.append(f"## {alg.upper()}")
        lines.append("")
        per = results.get(alg, {})
        if not per:
            lines.append("_No cached reconstructions available._\n")
            continue
        for key, label, spec, hib in METRIC_SPEC:
            lines.append(_metric_table(per, key, label, spec, hib))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="Compute metrics but do not modify the HDF5 cache.")
    parser.add_argument("--no-log", action="store_true",
                        help="Skip writing metrics_log.md.")
    args = parser.parse_args()

    results = backfill(dry_run=args.dry_run)

    if not args.no_log:
        LOG_PATH.write_text(render_log(results), encoding="utf-8")
        print(f"\nWrote {LOG_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
