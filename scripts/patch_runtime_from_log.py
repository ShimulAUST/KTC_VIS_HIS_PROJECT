"""Patch real Docker runtimes from runtime_log.md into results.h5.

Reads the per-sample wall-clock times recorded in data/cache/runtime_log.md
(produced by benchmark_runtime_*.py docker runs) and overwrites the `runtime`
scalar in each matching HDF5 group.

CUQI8 has no Docker runtime data in the log — its `runtime` key is removed from
the cache so the dashboard shows "—" rather than a misleading file-I/O time.

Usage:
    python scripts/patch_runtime_from_log.py
    python scripts/patch_runtime_from_log.py --dry-run   # print without writing
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import h5py

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CACHE_PATH   = _PROJECT_ROOT / "data" / "cache" / "results.h5"
_LOG_PATH     = _PROJECT_ROOT / "data" / "cache" / "runtime_log.md"
_SAMPLES      = ["a", "b", "c", "d"]

# Maps markdown section heading -> HDF5 algorithm key
_ALG_MAP = {"ABC1": "abc1", "PNPE2E": "pnpe2e"}

# Row pattern: |   L1 |  28.14 s |  21.92 s |  21.42 s |  22.09 s |  ...
_ROW_RE = re.compile(
    r"\|\s*L(\d)\s*\|"
    r"\s*([\d.]+)\s*s\s*\|"
    r"\s*([\d.]+)\s*s\s*\|"
    r"\s*([\d.]+)\s*s\s*\|"
    r"\s*([\d.]+)\s*s\s*\|"
)


def _parse_log(log_text: str) -> dict[tuple[str, int, str], float]:
    """Return {(alg, level, sample): seconds} from the markdown table."""
    runtimes: dict[tuple[str, int, str], float] = {}
    for heading, alg_key in _ALG_MAP.items():
        start = log_text.find(f"## {heading}")
        if start == -1:
            continue
        # find the start of the next ## section (or end of file)
        next_section = log_text.find("## ", start + 3)
        chunk = log_text[start: next_section if next_section != -1 else len(log_text)]
        for m in _ROW_RE.finditer(chunk):
            level = int(m.group(1))
            for col, sample in enumerate(_SAMPLES, start=2):
                runtimes[(alg_key, level, sample)] = float(m.group(col))
    return runtimes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be written without modifying the cache.")
    args = parser.parse_args()

    if not _LOG_PATH.exists():
        print(f"✗  runtime log not found: {_LOG_PATH}")
        sys.exit(1)
    if not _CACHE_PATH.exists():
        print(f"✗  cache not found: {_CACHE_PATH}")
        sys.exit(1)

    runtimes = _parse_log(_LOG_PATH.read_text())
    if not runtimes:
        print("✗  no runtime rows parsed from log — check the file format.")
        sys.exit(1)

    print(f"Parsed {len(runtimes)} runtime entries from {_LOG_PATH.name}")
    print(f"Cache: {_CACHE_PATH}")
    print(f"Dry-run: {args.dry_run}")
    print("-" * 60)

    patched = removed = skipped = 0

    with h5py.File(str(_CACHE_PATH), "a") as f:
        # ── Patch ABC1 and PNPE2E from log ───────────────────────────────────
        for (alg, level, sample), seconds in sorted(runtimes.items()):
            group = f"results/{alg}/{level}/{sample}"
            if group not in f:
                print(f"  skip  {alg}/L{level}/{sample}  (not in cache)")
                skipped += 1
                continue
            if args.dry_run:
                print(f"  patch {alg}/L{level}/{sample}  ->  {seconds:.2f}s")
                patched += 1
                continue
            grp = f[group]
            if "runtime" in grp:
                del grp["runtime"]
            grp.create_dataset("runtime", data=float(seconds))
            print(f"  OK  {alg}/L{level}/{sample}  ->  {seconds:.2f}s")
            patched += 1

        # ── Remove stale file-I/O runtime from CUQI8 ─────────────────────────
        print()
        print("CUQI8: no Docker timing in log — removing file-I/O placeholder.")
        for level in range(1, 8):
            for sample in _SAMPLES:
                group = f"results/cuqi8/{level}/{sample}"
                if group not in f:
                    continue
                grp = f[group]
                if "runtime" not in grp:
                    continue
                if args.dry_run:
                    v = float(grp["runtime"][()])
                    print(f"  remove cuqi8/L{level}/{sample}  (was {v:.6f}s)")
                    removed += 1
                    continue
                del grp["runtime"]
                print(f"  OK  removed cuqi8/L{level}/{sample}/runtime")
                removed += 1

    print()
    print("-" * 60)
    action = "Would patch" if args.dry_run else "Patched"
    print(f"{action} {patched} entries  |  removed {removed} CUQI8 placeholders"
          + (f"  |  skipped {skipped}" if skipped else ""))


if __name__ == "__main__":
    main()
