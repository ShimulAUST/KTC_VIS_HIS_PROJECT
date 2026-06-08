# Reconstruction Runtime Log

_Generated: 2026-06-07 01:14:17_

Wall-clock seconds per reconstruction, captured from per-sample docker runs.
Each value is one full container invocation processing exactly one data<idx>.mat;
the model load + framework startup overhead is included in every cell.

## ABC1

| Level | Sample A | Sample B | Sample C | Sample D | Level mean |
|------:|---------:|---------:|---------:|---------:|-----------:|
|   L1 |  28.14 s |  21.92 s |  21.42 s |  22.09 s |    23.39 s |
|   L2 |  24.94 s |  22.25 s |  22.04 s |  21.68 s |    22.73 s |
|   L3 |  22.17 s |  21.75 s |  21.64 s |  21.86 s |    21.85 s |
|   L4 |  21.70 s |  21.51 s |  23.43 s |  20.94 s |    21.89 s |
|   L5 |  22.98 s |  23.43 s |  21.07 s |  21.09 s |    22.14 s |
|   L6 |  21.85 s |  20.68 s |  21.28 s |  25.39 s |    22.30 s |
|   L7 |  20.86 s |  21.53 s |  21.68 s |  23.84 s |    21.98 s |

**Overall mean across L1-L7:** 22.33 s (min 20.68 s, max 28.14 s)

## CUQI8

| Level | Sample A | Sample B | Sample C | Sample D | Level mean |
|------:|---------:|---------:|---------:|---------:|-----------:|
|   L1 |    _n/a_ |    _n/a_ |    _n/a_ |    _n/a_ |          - |
|   L2 |    _n/a_ |    _n/a_ |    _n/a_ |    _n/a_ |          - |
|   L3 |    _n/a_ |    _n/a_ |    _n/a_ |    _n/a_ |          - |
|   L4 |    _n/a_ |    _n/a_ |    _n/a_ |    _n/a_ |          - |
|   L5 |    _n/a_ |    _n/a_ |    _n/a_ |    _n/a_ |          - |
|   L6 |    _n/a_ |    _n/a_ |    _n/a_ |    _n/a_ |          - |
|   L7 |    _n/a_ |    _n/a_ |    _n/a_ |    _n/a_ |          - |

_No real runtime captured yet - run the corresponding scripts/benchmark_runtime_*.py script._

## PNPE2E

| Level | Sample A | Sample B | Sample C | Sample D | Level mean |
|------:|---------:|---------:|---------:|---------:|-----------:|
|   L1 |  43.08 s |  42.74 s |  40.51 s |  42.26 s |    42.15 s |
|   L2 |  41.19 s |  47.61 s |  41.25 s |  40.23 s |    42.57 s |
|   L3 |  39.92 s |  40.47 s |  39.31 s |  42.59 s |    40.57 s |
|   L4 |  39.90 s |  39.88 s |  40.55 s |  39.19 s |    39.88 s |
|   L5 |  43.75 s |  43.03 s |  38.92 s |  41.25 s |    41.74 s |
|   L6 |  43.90 s |  43.41 s |  48.00 s |  43.75 s |    44.76 s |
|   L7 |  43.49 s |  43.99 s |  43.16 s |  43.09 s |    43.43 s |

**Overall mean across L1-L7:** 42.16 s (min 38.92 s, max 48.00 s)
