"""Regenerate data/cache/runtime_log.md from data/cache/results.h5.
Usage:
    python scripts/dump_runtime_log.py
"""

from __future__ import annotations

import datetime
import statistics
from pathlib import Path

import h5py

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CACHE_PATH = _PROJECT_ROOT / "data" / "cache" / "results.h5"
_OUT_PATH = _PROJECT_ROOT / "data" / "cache" / "runtime_log.md"


def _fmt(s: float) -> str:
    if s < 60:
        return f"{s:.2f} s"
    m, sec = divmod(int(s), 60)
    if m < 60:
        return f"{m}m {sec:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h {m:02d}m"


_ALG_SOURCE = {
    "abc1": (
        "_Source: per-sample docker runs of `ktc2023-abc-python` via "
        "`scripts/benchmark_runtime_abc1.py --per-sample`._"
    ),
    "pnpe2e": (
        "_Source: per-sample docker runs of `ktc2023-pnpe2e` (CPU mode) via "
        "`scripts/benchmark_runtime_pnpe2e.py --per-sample`._"
    ),
    "cuqi8": (
        "_Source: **estimated** from an observed 48h batch run of all 7 levels × 4 samples._  \n"
        "_Distribution: base = 48h / 28 = 6171 s per sample, with mild L1→L7 decline "
        "(fewer voltage measurements)_  \n"
        "_and ±70 s deterministic scatter (seed=42). Not fresh docker measurements._"
    ),
}


def main() -> None:
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines: list[str] = []
    lines.append("# Reconstruction Runtime Log")
    lines.append("")
    lines.append(f"_Generated: {ts}_")
    lines.append("")
    lines.append("Wall-clock time per reconstruction, one row per level, one column per sample.")
    lines.append("")

    with h5py.File(str(_CACHE_PATH), "r") as f:
        for alg in ["abc1", "cuqi8", "pnpe2e"]:
            lines.append(f"## {alg.upper()}")
            lines.append("")
            lines.append(_ALG_SOURCE[alg])
            lines.append("")
            lines.append("| Level | Sample A | Sample B | Sample C | Sample D | Level mean |")
            lines.append("|------:|---------:|---------:|---------:|---------:|-----------:|")

            all_vals: list[float] = []
            for lv in range(1, 8):
                row_vals: list[float] = []
                cells: list[str] = []
                for s in ["a", "b", "c", "d"]:
                    key = f"results/{alg}/{lv}/{s}/runtime"
                    if key in f:
                        rt = float(f[key][()])
                        if rt < 0.01:
                            cells.append("_n/a_")
                        else:
                            cells.append(_fmt(rt))
                            row_vals.append(rt)
                            all_vals.append(rt)
                    else:
                        cells.append("-")
                mean_str = _fmt(statistics.mean(row_vals)) if row_vals else "-"
                lines.append(
                    f"|   L{lv} | {cells[0]:>9} | {cells[1]:>9} | {cells[2]:>9} | "
                    f"{cells[3]:>9} | {mean_str:>10} |"
                )

            if all_vals:
                lines.append("")
                lines.append(
                    f"**Overall mean across L1-L7:** {_fmt(statistics.mean(all_vals))} "
                    f"(min {_fmt(min(all_vals))}, max {_fmt(max(all_vals))})"
                )
            lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("## Notes")
    lines.append(
        "- ABC1 and PNPE2E numbers are measured in per-sample mode: each cell = one full "
        "docker invocation processing a single `data<idx>.mat`. Model/framework startup "
        "cost is paid every cell."
    )
    lines.append(
        "- CUQI8 numbers are derived from an observed 48-hour batch benchmark; per-sample "
        "scatter is deterministic (seed=42) so teammates re-generating the log will see "
        "identical values."
    )
    lines.append(
        "- Cache location: `data/cache/results.h5`, key `results/<alg>/<level>/<sample>/runtime`. "
        "Provenance for CUQI8 is stored in the dataset attribute of the same name."
    )
    lines.append("")

    _OUT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {_OUT_PATH}")


if __name__ == "__main__":
    main()
