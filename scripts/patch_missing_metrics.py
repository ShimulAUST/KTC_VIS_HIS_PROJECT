"""Patch missing metrics into the HDF5 cache without re-running reconstructions.

Reads reconstruction arrays already stored in results.h5, recomputes only the
metrics that are absent, and writes them back. Existing metric values are left
unchanged (runtime, ssim, iou_*, hausdorff, etc.).

Missing metrics (9):
  ssim_min, confusion_accuracy,
  dice_water, dice_resistive, dice_conductive, dice_mean,
  voltage_residual, resistance_consistency, current_sensitivity

Usage:
    python scripts/patch_missing_metrics.py
    python scripts/patch_missing_metrics.py --dry-run
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import scipy.io

sys.path.insert(0, str(Path(__file__).parent.parent))

from ktc_vis.data.loader import KTCDataLoader
from ktc_vis.metrics.class_metrics import compute_confusion_matrix, compute_per_class_dice
from ktc_vis.metrics.image_quality import compute_spatial_ssim_map
from ktc_vis.metrics.measurement import (
    compute_current_sensitivity_balance,
    compute_resistance_consistency,
    compute_voltage_residual,
)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CACHE_PATH   = _PROJECT_ROOT / "data" / "cache" / "results.h5"
_RAW_DIR      = _PROJECT_ROOT / "data" / "raw" / "ktc2023"
_SAMPLE_MAP   = {"a": 1, "b": 2, "c": 3, "d": 4}

ALGOS   = ["abc1", "cuqi8", "pnpe2e"]
LEVELS  = list(range(1, 8))
SAMPLES = ["a", "b", "c", "d"]

PATCH_KEYS = [
    "ssim_min",
    "confusion_accuracy",
    "dice_water",
    "dice_resistive",
    "dice_conductive",
    "dice_mean",
    "voltage_residual",
    "resistance_consistency",
    "current_sensitivity",
]


def _load_gt(level: int, sample: str) -> np.ndarray:
    idx = _SAMPLE_MAP[sample]
    path = _RAW_DIR / "ground_truth" / f"true{idx}.mat"
    return scipy.io.loadmat(str(path))["truth"].astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be patched without writing.")
    args = parser.parse_args()

    loader = KTCDataLoader()
    patched = skipped = failed = 0
    t_start = time.perf_counter()

    print(f"Patching {len(PATCH_KEYS)} missing metrics into {_CACHE_PATH.name}")
    print(f"Dry-run: {args.dry_run}")
    print("-" * 60)

    with h5py.File(str(_CACHE_PATH), "a") as f:
        for alg in ALGOS:
            print(f"\n  {alg.upper()}")
            for level in LEVELS:
                for sample in SAMPLES:
                    key = f"results/{alg}/{level}/{sample}"
                    if key not in f:
                        print(f"    L{level}/{sample.upper()}  SKIP (not in cache)")
                        skipped += 1
                        continue

                    grp = f[key]
                    needed = [k for k in PATCH_KEYS if k not in grp]
                    if not needed:
                        skipped += 1
                        continue

                    try:
                        # Load reconstruction from cache
                        recon = grp["reconstruction"][()].astype(np.uint8)
                        gt    = _load_gt(level, sample)

                        # Image quality
                        new_vals: dict[str, float] = {}
                        if "ssim_min" in needed:
                            ssim_map = compute_spatial_ssim_map(recon, gt)
                            new_vals["ssim_min"] = float(ssim_map.min())

                        # Class metrics
                        dice_needed = [k for k in needed
                                       if k.startswith("dice") or k == "confusion_accuracy"]
                        if dice_needed:
                            dice = compute_per_class_dice(recon, gt)
                            valid = [v for v in dice.values() if not np.isnan(v)]
                            if "confusion_accuracy" in needed:
                                cm = compute_confusion_matrix(recon, gt)
                                new_vals["confusion_accuracy"] = float(np.mean(np.diag(cm)))
                            if "dice_water" in needed:
                                new_vals["dice_water"] = float(dice["water"]) if not np.isnan(dice["water"]) else 0.0
                            if "dice_resistive" in needed:
                                new_vals["dice_resistive"] = float(dice["resistive"]) if not np.isnan(dice["resistive"]) else 0.0
                            if "dice_conductive" in needed:
                                new_vals["dice_conductive"] = float(dice["conductive"]) if not np.isnan(dice["conductive"]) else 0.0
                            if "dice_mean" in needed:
                                new_vals["dice_mean"] = float(np.mean(valid)) if valid else 0.0

                        # Measurement domain (needs voltage/current/resistance matrices)
                        meas_needed = [k for k in needed if k in
                                       ("voltage_residual", "resistance_consistency", "current_sensitivity")]
                        if meas_needed:
                            measurement = loader.load(level=level, sample=sample)
                            if "voltage_residual" in needed:
                                new_vals["voltage_residual"] = compute_voltage_residual(measurement)
                            if "resistance_consistency" in needed:
                                new_vals["resistance_consistency"] = compute_resistance_consistency(measurement)
                            if "current_sensitivity" in needed:
                                new_vals["current_sensitivity"] = compute_current_sensitivity_balance(measurement)

                        tag = f"    L{level}/{sample.upper()}"
                        if args.dry_run:
                            summary = ", ".join(f"{k}={v:.4f}" for k, v in new_vals.items())
                            print(f"{tag}  would patch: {summary}")
                        else:
                            for k, v in new_vals.items():
                                grp.create_dataset(k, data=v)
                            print(f"{tag}  OK  ({len(new_vals)} keys added)")
                        patched += 1

                    except Exception as exc:
                        print(f"    L{level}/{sample.upper()}  ERROR: {exc}")
                        failed += 1

    elapsed = time.perf_counter() - t_start
    action = "Would patch" if args.dry_run else "Patched"
    print(f"\n{'-' * 60}")
    print(f"{action} {patched} entries  |  skipped {skipped}  |  failed {failed}")
    print(f"Time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
