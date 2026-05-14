"""Stage the shared KTC2023 dataset into ``data/raw/ktc2023/``.

The external repositories under ``ktc_vis/external/`` each ship the same
KTC2023 measurements and ground truths, plus their own published
reconstruction outputs. This script consolidates them into a single
canonical layout used by ``ktc_vis/data/loader.py``:

    data/raw/ktc2023/
    ├── measurements/         shared inputs  (data1-4.mat, ref.mat)
    ├── ground_truth/         shared targets (true1-4.mat)
    └── reference_outputs/    per-algorithm published reconstructions
        ├── abc1/             (data1-4.mat)
        ├── cuqi8/            (1-4.mat)
        └── pnpe2e/           (1-4.mat)

Run once after cloning the external repos::

    python scripts/stage_dataset.py
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXTERNAL = ROOT / "ktc_vis" / "external"
DEST = ROOT / "data" / "raw" / "ktc2023"

# (source_glob_relative_to_external, dest_subdir)
COPY_PLAN: list[tuple[str, str]] = [
    # Shared inputs — canonical source is KTC2023_ABC (byte-identical in all repos)
    ("KTC2023_ABC/TrainingData/*.mat", "measurements"),
    ("KTC2023_ABC/GroundTruths/*.mat", "ground_truth"),
    # Per-algorithm reference outputs
    ("KTC2023_ABC/Outputs/*.mat", "reference_outputs/abc1"),
    ("KTC2023_CUQI8/Output/*.mat", "reference_outputs/cuqi8"),
    ("KTC2023_PNPE2E/Output/*.mat", "reference_outputs/pnpe2e"),
]


def main() -> int:
    if not EXTERNAL.exists():
        print(f"[ERROR] {EXTERNAL} not found. Clone the external repos first.", file=sys.stderr)
        return 1

    DEST.mkdir(parents=True, exist_ok=True)
    total = 0
    for pattern, subdir in COPY_PLAN:
        src_dir = EXTERNAL / Path(pattern).parent
        glob = Path(pattern).name
        dst_dir = DEST / subdir
        dst_dir.mkdir(parents=True, exist_ok=True)

        files = sorted(src_dir.glob(glob))
        if not files:
            print(f"[WARN] no files matched {pattern}")
            continue

        for f in files:
            shutil.copy2(f, dst_dir / f.name)
            total += 1
        print(f"[OK]  {len(files):2d} files  {pattern}  ->  {dst_dir.relative_to(ROOT)}")

    print(f"\nStaged {total} files into {DEST.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
