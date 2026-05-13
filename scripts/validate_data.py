"""CLI: verify KTC2023 .mat files are correctly downloaded and structured.

Usage:
    python scripts/validate_data.py
    python scripts/validate_data.py --raw-dir /custom/path/ktc2023

Note: All levels contain the full 2356 measurements in the raw files.
Difficulty-level subsampling is handled internally by each algorithm container.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import scipy.io

LEVELS = list(range(1, 8))
SAMPLES = [1, 2, 3]  # data1.mat → "a", data2.mat → "b", data3.mat → "c"


def check_level(level_dir: Path, level: int) -> list[str]:
    """Validate all files for one difficulty level. Returns list of error strings."""
    errors = []

    for sample_idx in SAMPLES:
        data_path = level_dir / f"data{sample_idx}.mat"
        gt_path = level_dir / f"{sample_idx}_true.mat"

        # --- measurement file ---
        if not data_path.exists():
            errors.append(f"MISSING: {data_path}")
        else:
            try:
                d = scipy.io.loadmat(str(data_path))
                for key in ("Inj", "Mpat", "Uel"):
                    if key not in d:
                        errors.append(f"MISSING KEY '{key}': {data_path}")
                if "Uel" in d and np.array(d["Uel"]).size == 0:
                    errors.append(f"EMPTY 'Uel': {data_path}")
            except Exception as e:
                errors.append(f"UNREADABLE: {data_path} — {e}")

        # --- ground truth file ---
        if not gt_path.exists():
            errors.append(f"MISSING: {gt_path}")
        else:
            try:
                g = scipy.io.loadmat(str(gt_path))
                if "truth" not in g:
                    errors.append(f"MISSING KEY 'truth': {gt_path}")
                else:
                    truth = np.array(g["truth"])
                    if truth.shape != (256, 256):
                        errors.append(
                            f"WRONG SHAPE: {gt_path} — {truth.shape}, expected (256, 256)"
                        )
                    unexpected = set(np.unique(truth)) - {0, 1, 2}
                    if unexpected:
                        errors.append(
                            f"BAD VALUES: {gt_path} — unexpected pixel values {unexpected}"
                        )
            except Exception as e:
                errors.append(f"UNREADABLE: {gt_path} — {e}")

    # --- reference file ---
    ref_path = level_dir / "ref.mat"
    if not ref_path.exists():
        errors.append(f"MISSING: {ref_path}")
    else:
        try:
            r = scipy.io.loadmat(str(ref_path))
            for key in ("Injref", "Uelref"):
                if key not in r:
                    errors.append(f"MISSING KEY '{key}': {ref_path}")
        except Exception as e:
            errors.append(f"UNREADABLE: {ref_path} — {e}")

    return errors


def main(raw_dir: Path) -> int:
    print(f"Validating KTC2023 dataset at: {raw_dir}\n")
    all_ok = True

    for level in LEVELS:
        level_dir = raw_dir / f"level{level}"
        if not level_dir.is_dir():
            print(f"[FAIL] level{level}/ directory not found")
            all_ok = False
            continue

        errors = check_level(level_dir, level)
        if errors:
            all_ok = False
            print(f"[FAIL] level{level}/")
            for e in errors:
                print(f"       {e}")
        else:
            d = scipy.io.loadmat(str(level_dir / "data1.mat"))
            n_meas = int(np.array(d["Uel"]).size)
            print(f"[OK]   level{level}/  ({n_meas} measurements, 3 samples, ground truths OK)")

    print()
    if all_ok:
        print("All checks passed. Dataset is ready.")
        return 0
    else:
        print("Some checks failed. Re-download from: https://zenodo.org/records/10986692")
        return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate KTC2023 .mat files")
    parser.add_argument(
        "--raw-dir",
        default="data/raw/ktc2023",
        help="Path to the ktc2023 root directory (default: data/raw/ktc2023)",
    )
    args = parser.parse_args()
    sys.exit(main(Path(args.raw_dir)))
