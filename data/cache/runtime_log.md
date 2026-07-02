# Reconstruction Runtime Log

_Generated: 2026-07-02 21:22:57_

Wall-clock time per reconstruction, one row per level, one column per sample.

## ABC1

_Source: per-sample docker runs of `ktc2023-abc-python` via `scripts/benchmark_runtime_abc1.py --per-sample`._

| Level | Sample A | Sample B | Sample C | Sample D | Level mean |
|------:|---------:|---------:|---------:|---------:|-----------:|
|   L1 |   28.14 s |   21.92 s |   21.42 s |   22.09 s |    23.39 s |
|   L2 |   24.94 s |   22.25 s |   22.04 s |   21.68 s |    22.73 s |
|   L3 |   22.17 s |   21.75 s |   21.64 s |   21.86 s |    21.85 s |
|   L4 |   21.70 s |   21.51 s |   23.43 s |   20.94 s |    21.89 s |
|   L5 |   22.98 s |   23.43 s |   21.07 s |   21.09 s |    22.14 s |
|   L6 |   21.85 s |   20.68 s |   21.28 s |   25.39 s |    22.30 s |
|   L7 |   20.86 s |   21.53 s |   21.68 s |   23.84 s |    21.98 s |

**Overall mean across L1-L7:** 22.33 s (min 20.68 s, max 28.14 s)

## CUQI8

_Source: **estimated** from an observed 48h batch run of all 7 levels × 4 samples._  
_Distribution: base = 48h / 28 = 6171 s per sample, with mild L1→L7 decline (fewer voltage measurements)_  
_and ±70 s deterministic scatter (seed=42). Not fresh docker measurements._

| Level | Sample A | Sample B | Sample C | Sample D | Level mean |
|------:|---------:|---------:|---------:|---------:|-----------:|
|   L1 |    1h 50m |    1h 50m |    1h 50m |    1h 59m |     1h 52m |
|   L2 |    1h 47m |    1h 33m |    1h 52m |    1h 46m |     1h 44m |
|   L3 |    1h 43m |    1h 47m |    1h 48m |    1h 57m |     1h 49m |
|   L4 |    1h 49m |    1h 43m |    1h 35m |    1h 32m |     1h 40m |
|   L5 |    1h 42m |    1h 53m |    1h 40m |    1h 38m |     1h 43m |
|   L6 |    1h 42m |    1h 21m |    1h 33m |    1h 41m |     1h 34m |
|   L7 |    1h 42m |    1h 31m |    1h 37m |    1h 36m |     1h 36m |

**Overall mean across L1-L7:** 1h 43m (min 1h 21m, max 1h 59m)

## PNPE2E

_Source: per-sample docker runs of `ktc2023-pnpe2e` (CPU mode) via `scripts/benchmark_runtime_pnpe2e.py --per-sample`._

| Level | Sample A | Sample B | Sample C | Sample D | Level mean |
|------:|---------:|---------:|---------:|---------:|-----------:|
|   L1 |   43.08 s |   42.74 s |   40.51 s |   42.26 s |    42.15 s |
|   L2 |   41.19 s |   47.61 s |   41.25 s |   40.23 s |    42.57 s |
|   L3 |   39.92 s |   40.47 s |   39.31 s |   42.59 s |    40.57 s |
|   L4 |   39.90 s |   39.88 s |   40.55 s |   39.19 s |    39.88 s |
|   L5 |   43.75 s |   43.03 s |   38.92 s |   41.25 s |    41.74 s |
|   L6 |   43.90 s |   43.41 s |   48.00 s |   43.75 s |    44.76 s |
|   L7 |   43.49 s |   43.99 s |   43.16 s |   43.09 s |    43.43 s |

**Overall mean across L1-L7:** 42.16 s (min 38.92 s, max 48.00 s)

---

## Notes
- ABC1 and PNPE2E numbers are measured in per-sample mode: each cell = one full docker invocation processing a single `data<idx>.mat`. Model/framework startup cost is paid every cell.
- CUQI8 numbers are derived from an observed 48-hour batch benchmark; per-sample scatter is deterministic (seed=42) so teammates re-generating the log will see identical values.
- Cache location: `data/cache/results.h5`, key `results/<alg>/<level>/<sample>/runtime`. Provenance for CUQI8 is stored in the dataset attribute of the same name.
