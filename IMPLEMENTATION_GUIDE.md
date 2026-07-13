# KTC-Vis — Implementation Guide

## 0. Course & Team

**Course:** Project HIS
**Institution:** Frankfurt University of Applied Sciences, Germany
**Guided by:** Prof. Dr. Martin Simon & Emanuele Pepe

| Name | Student ID | Role | Modules Owned |
|------|------------|------|----------------|
| Muhammad Muzammal | 1541353 | EIT & Data Specialist / Backend Engineer | M1, M5 |
| Smit Savani | 1420825 | Metrics & Backend Engineer | M3, M6 |
| Asmita Bhuva | 1541650 | Viz & Frontend Developer | M4, M5 |
| Shimul Paul | 1441927 | Integration & DevOps Lead | M2, M6 |

M5 and M6 are jointly owned (Muzammal + Asmita on M5; Smit + Shimul on M6). This table is the authoritative ownership map — some in-code module docstrings still carry an older single-owner label from the original task split; defer to this table over those comments.

## 1. What This Project Is

KTC-Vis is a Dash dashboard for benchmarking three EIT (Electrical Impedance Tomography) reconstruction algorithms — **ABC1**, **CUQI8**, **PNPE2E** — against the KTC2023 competition dataset, across 7 difficulty levels (32 → 20 electrodes) and 4 phantom samples (a–d).

Two parallel data paths feed the dashboard:

1. **Reference-output path** (fast, no Docker) — `ReferenceOutputAdapter` reads pre-computed `.mat` reconstructions staged from each algorithm's original repo under `data/raw/ktc2023/reference_outputs/<algo>/level<N>/`. This is what powers M1 and M3 by default and what `scripts/populate_cache_from_reference.py` uses to fill the HDF5 cache quickly.
2. **Live algorithm path** (slow, Docker-based) — `ABC1Adapter`, `CUQI8Adapter`, `PNPE2EAdapter` each shell out to a published Docker image, write subsampled measurement `.mat` files, run the container, and read back a reconstruction. `scripts/run_benchmark.py` drives this path to (re)populate the cache with real reconstructions and real runtimes.

Both paths converge on the same `MetricsEngine` and the same HDF5 cache format, so the dashboard does not depend on which path produced a given cached result.

## 2. Repository Layout

```
KTC_VIS_HIS_PROJECT/
├── app.py                              # Dash entry point (reads configs/experiment.yaml)
├── environment.yml                     # Conda env (dash/plotly/torch/pyeit/mat73/…)
├── requirements.txt                    # pip-only equivalent (no FEniCS)
├── pyproject.toml                      # setuptools + pytest + coverage + flake8 config
├── Dockerfile                          # App image (miniconda3 base + Docker CLI)
├── docker-compose.yml                  # Mounts data/ + host Docker socket
├── .dockerignore
├── .github/workflows/ci.yml            # Conda + mamba CI: flake8, pytest, coverage
├── configs/
│   └── experiment.yaml                 # Master config: data paths, benchmark scope, metric flags
├── data/
│   ├── raw/ktc2023/                    # Staged dataset (NOT committed) — see §4
│   └── cache/
│       ├── results.h5                  # HDF5 results cache (metrics + reconstructions)
│       └── runtime_log.md              # Notes from runtime benchmarking sessions
├── ktc_vis/
│   ├── data/
│   │   ├── loader.py                   # KTCDataLoader — .mat → KTCMeasurement
│   │   └── subsampler.py               # subsample_electrodes() / subsample_measurement()
│   ├── adapters/
│   │   ├── base.py                     # KTCMeasurement dataclass + AlgorithmAdapter ABC
│   │   ├── reference_adapter.py        # ReferenceOutputAdapter — precomputed .mat, no Docker
│   │   ├── abc1_adapter.py             # Docker: muzammal5566/ktc2023-abc-python
│   │   ├── cuqi8_adapter.py            # Docker: muzammal5566/ktc2023-cuqi8 (level-batched)
│   │   └── pnpe2e_adapter.py           # Docker: muzammal5566/ktc2023-pnpe2e (level-batched)
│   ├── metrics/
│   │   ├── engine.py                   # MetricsEngine.compute_all() → 18-key metrics dict
│   │   ├── image_quality.py            # SSIM + spatial SSIM map
│   │   ├── shape_matching.py           # Hausdorff, position error, resolution
│   │   ├── class_metrics.py            # Per-class IoU/Dice, confusion matrix
│   │   ├── measurement.py              # Voltage residual, resistance consistency, current sensitivity
│   │   ├── efficiency.py               # measure_runtime()
│   │   └── _voltage_surrogate.py       # Surrogate forward model for measurement-domain metrics
│   ├── cache/
│   │   └── hdf5_store.py               # save_result / load_result / is_cached / CacheMiss
│   ├── dashboard/
│   │   ├── layout.py                   # create_layout() — sidebar + 6-tab dcc.Tabs
│   │   ├── theme.py                    # Shared color tokens + CARD_STYLE etc.
│   │   ├── components/
│   │   │   └── sidebar.py              # Algorithm dropdown, level slider, sample radio
│   │   └── modules/
│   │       ├── m1_reconstruction_explorer.py
│   │       ├── m2_difficulty_animator.py
│   │       ├── m3_comparison_grid.py
│   │       ├── m4_fingerprint_radar.py
│   │       ├── m5_failure_autopsy.py
│   │       └── m6_measurement_viewer.py
│   └── utils/
│       └── figures.py                  # segmentation/reconstruction/error-overlay figure builders
├── scripts/
│   ├── stage_dataset.py                 # Consolidate external repo data → data/raw/ktc2023/
│   ├── validate_data.py                 # Verify staged .mat files per level
│   ├── run_benchmark.py                 # Live Docker benchmark → populates HDF5 cache
│   ├── populate_cache_from_reference.py # Fast cache population from reference outputs
│   ├── benchmark_runtime_abc1.py        # Times ABC1 docker runs, patches runtime into cache
│   ├── benchmark_runtime_pnpe2e.py      # Times PNPE2E docker runs, patches runtime into cache
│   └── setup_third_party.sh             # Clones/prepares the external algorithm repos
├── tests/
│   ├── conftest.py                      # Shared fixtures (currently empty — see §10)
│   ├── test_loader.py, test_adapters.py, test_cache.py, test_metrics.py
│   └── test_modules/test_m1.py … test_m6.py, test_glossary.py
├── docs/
│   ├── adding_new_algorithm.md
│   └── interpreting_dashboard.md
└── IMPLEMENTATION_GUIDE.md              # This file
```

## 3. Data Pipeline

### `ktc_vis/adapters/base.py`

`KTCMeasurement` dataclass: `current_matrix` (n_inj × n_electrodes), `voltage_matrix` (n_inj × n_meas), `resistance_matrix` (= V/I, same shape as voltage), `ground_truth` (256×256 uint8, {0=water, 1=resistive, 2=conductive}), `level`, `sample`.

`AlgorithmAdapter` ABC: every adapter implements `reconstruct(measurement) -> np.ndarray` (256×256 uint8). Adapters that can process a whole level in one container invocation (CUQI8, PNPE2E) set `supports_level_batching = True` and override `reconstruct_level(level, samples)`.

### `ktc_vis/data/loader.py` — `KTCDataLoader`

Only **Level 1** is stored on disk (`measurements/data{1..4}.mat`, `measurements/ref.mat`, `ground_truth/true{1..4}.mat`). `load(level, sample)` reads Level-1 data directly and derives Levels 2–7 at runtime via `subsample_measurement`. Sample letters map to file index: a→1, b→2, c→3, d→4.

### `ktc_vis/data/subsampler.py`

Electrode counts per level (KTC2023 paper Table 1, 0-based indices):

| Level | Electrodes | Measurements |
|-------|-----------|--------------|
| 1 | 32 | 2356 |
| 2 | 30 | 1624 |
| 3 | 28 | 1404 |
| 4 | 26 | 1200 |
| 5 | 24 | 1012 |
| 6 | 22 | 630 |
| 7 | 20 | 513 |

`subsample_measurement()` drops injection rows and `Mpat`-derived voltage columns whose electrodes fall outside the active set for that level, and re-derives resistance (`R = V / I`) on the reduced matrices.

### `ktc_vis/cache/hdf5_store.py`

Flat HDF5 layout: `results/{algorithm}/{level}/{sample}/{metric_name | reconstruction}`. `save_result()` opens in append mode and replaces the group if it exists. `load_result()` returns `(metrics_dict, reconstruction_array)` or raises `CacheMiss`. `is_cached()` is a cheap existence check used by the benchmark scripts to skip already-computed combinations.

## 4. Adapters

| Adapter | File | Mechanism |
|---|---|---|
| `ReferenceOutputAdapter` | `reference_adapter.py` | Reads `data/raw/ktc2023/reference_outputs/<algo>/level<N>/<file>.mat` (no Docker). Falls back to the legacy flat layout at Level 1. Filename convention differs per algo: `abc1` → `data{idx}.mat`, `cuqi8`/`pnpe2e` → `{idx}.mat`. |
| `ABC1Adapter` | `abc1_adapter.py` | Docker image `muzammal5566/ktc2023-abc-python:latest`. Writes subsampled `data{idx}.mat` + `ref.mat` to a temp `TrainingData/` dir, runs `python main_python.py TrainingData Outputs <level>`, reads back the first 2-D array in the output `.mat`. |
| `CUQI8Adapter` | `cuqi8_adapter.py` | Docker image `muzammal5566/ktc2023-cuqi8:latest`. `supports_level_batching = True` — one container run processes all samples for a level via `conda run -n env python main.py TrainingData Output <level>`. |
| `PNPE2EAdapter` | `pnpe2e_adapter.py` | Docker image `muzammal5566/ktc2023-pnpe2e:latest`. Also level-batched. Monkey-patches `torch.load` to `map_location='cpu'` before importing `main` so CUDA-trained weights load on CPU-only hosts. |

All three live adapters require Docker on the host and `--platform linux/amd64` (the images are x86_64-only; Apple Silicon runs them under Rosetta 2 — `run_benchmark.py` prints a warning when it detects `arm64`).

## 5. Metrics Engine

`MetricsEngine(adapter, cache_path).compute_all(measurement)` checks the cache first via `is_cached`, and if the result is missing it calls `adapter.reconstruct()`, times it, and computes the full metric set via `_compute_metrics_from_reconstruction()`. Every call writes the result back to the HDF5 cache.

18 keys are computed per `(algorithm, level, sample)`:

| Group | Keys | Notes |
|---|---|---|
| Image quality | `ssim`, `ssim_min` | `skimage.metrics.structural_similarity`, `data_range=2.0`; `ssim_min` = worst local window from the same spatial map used by M5. |
| Shape matching | `hausdorff`, `position_error`, `resolution` | Symmetric Hausdorff over foreground points; centroid Euclidean offset; diameter of the smallest pred∩gt connected region. |
| Class-specific | `iou_water`, `iou_resistive`, `iou_conductive`, `iou_mean`, `dice_water`, `dice_resistive`, `dice_conductive`, `dice_mean`, `confusion_accuracy` | Per-class IoU/Dice from `class_metrics.py`; `confusion_accuracy` = mean of the confusion matrix diagonal. |
| Measurement domain | `voltage_residual`, `resistance_consistency`, `current_sensitivity` | Computed from the measurement alone (no reconstruction needed) — see `metrics/measurement.py`. `voltage_residual` = normalised RMS deviation from the per-injection mean; `resistance_consistency` and `current_sensitivity` are `1 − clip(CoV, 0, 1)` homogeneity/balance scores. |
| Efficiency | `runtime` | Wall-clock seconds for `adapter.reconstruct()` (or the reference-output read cost, which is near-zero and treated as "not real" by M2/M4 — see the `_RUNTIME_REAL_THRESHOLD_S` guard). |

M3's scorecard (`_METRIC_KEYS`) is the single source of truth for display labels, format specs, and higher-is-better flags for all 18 metrics plus `dice_mean`/`dice_*` duplicates.

## 6. Dashboard Modules

Every module is a `layout()` + `register_callbacks(app)` pair registered from `ktc_vis/dashboard/layout.py`. All component IDs are prefixed `m{N}-`. Figures use `uirevision="constant"` (or a module-specific string) so zoom and pan state survives data re-renders. The three sidebar controls — `sidebar-algorithm-dropdown`, `sidebar-level-slider`, `sidebar-sample-radio` — are shared `Input`s across every module.

The descriptions below follow the same structure for each panel: what the figure shows, where the data comes from and how the figure is constructed, and what analytical question it is intended to answer.

### M1 · Reconstruction Explorer (`m1_reconstruction_explorer.py`) — owner: Muzammal

A two-row layout for one `(algorithm, level, sample)`, backed by `ReferenceOutputAdapter`. Its purpose is to give a complete single-case view: what did the algorithm output, and where exactly did it go wrong.

**1. Ground Truth heatmap** (`m1-gt-image`) — 256×256 categorical heatmap of the true phantom (water / resistive / conductive), rendered via `segmentation_figure(gt)` from `utils/figures.py` with a hard-stepped colorscale at 0.33/0.66, hidden axes, locked aspect ratio, and a reversed y-axis to match image convention. This panel is the reference against which everything else in the row is compared.

**2. Reconstruction heatmap** (`m1-recon-image`) — continuous Viridis heatmap of the algorithm's raw output via `reconstruction_figure(recon)`, with zmin/zmax from the array's own range and a colorbar. Showing the raw output before thresholding reveals smoothing, blur, or soft-boundary artefacts that a categorical view would hide.

**3. Segmentation heatmap** (`m1-seg-image`) — the same reconstruction re-rendered on the identical 3-class colorscale used for the ground truth via `segmentation_figure(recon)`. This panel enables a direct class-by-class comparison with panel 1.

**4. Error Overlay heatmap** (`m1-error-image`) — a single map encoding four per-pixel outcomes (green = correct, red = false positive, blue = false negative, yellow = wrong class), built by `error_overlay_figure(recon, gt)` from a 0–4 integer code derived from boolean masks, with a legend drawn directly as a figure annotation. The aggregate correct-pixel percentage feeds the chip row above.

The chip row (algorithm, level, sample, electrode/injection/voltage-sample counts, pixel agreement %) provides a concise numerical summary without requiring the user to read individual panels.

### M2 · Difficulty Animator (`m2_difficulty_animator.py`) — owner: Shimul Paul

A level 1–7 slider with play/pause auto-advance at 1.2 s per level, wrapping from level 7 back to 1. The purpose of this module is to show whether an algorithm's quality declines gradually or collapses suddenly as electrodes are removed, and at which specific level that transition occurs.

**1. Ground Truth image** (`m2-gt-graph`) — the true phantom loaded directly from `true{idx}.mat` via `scipy.io`, using a module-local 3-class colorscale. This panel is intentionally static: because it never changes, the viewer can focus attention on the panels that do.

**2. Segmentation image** (`m2-recon-graph`) — heatmap of the cached reconstruction at the current level, loaded via `load_result()` from the HDF5 cache. This is the primary animated element; visible changes (blurring, dropped inclusions, shifted boundaries) as the level advances are the central observation the module is designed to produce.

**3. Error Overlay image** (`m2-error-graph`) — the same four-class error encoding as M1, recomputed per level by `_recon_and_error_figures()`. Watching the red, blue, and orange pixel area grow across levels gives a concrete visual representation of "less data = worse reconstruction."

**4–10. Degradation curves** (SSIM, Mean IoU, Dice, Hausdorff, Position Error, Resolution, Runtime) — line charts with x = level 1–7, y = metric value, for the selected algorithm and sample. A dashed vertical marker indicates the currently displayed level. `_curve_figure()` reads cached metrics for all 7 levels; Dice is derived from IoU on the fly (`2·IoU/(1+IoU)`); SSIM, IoU, Dice, and Runtime use fixed y-axis ranges to prevent auto-scaling from making a poorly-performing algorithm appear falsely stable. Runtime is suppressed when the cached value falls below `_RUNTIME_REAL_THRESHOLD_S`, since near-zero values indicate a cache read rather than a live Docker run.

The level context progress bar and the auto-generated commentary panel translate the curves into plain language — noting, for example, the level at which SSIM first drops below 0.60 — for readers who want a summary rather than seven separate charts.

### M3 · Side-by-Side Comparison Grid (`m3_comparison_grid.py`) — owner: Smit Savani

Shows all three algorithms for the same `(level, sample)` simultaneously, removing the need to flip the sidebar algorithm dropdown for each comparison.

**1–3. Algorithm reconstruction heatmaps** (ABC1 / CUQI8 / PNPE2E) — three 3-class segmentation heatmaps via `_recon_figure()`, using the shared `CLASS_COLORS` colorscale. Each panel loads from the cache first and falls back to the live adapter if the result is not cached. Placing three outputs in the same coordinate space makes subtle differences in inclusion size, shape, or boundary placement easier to detect than sequential comparison.

**4–6. Pairwise difference heatmaps** (ABC1−CUQI8, ABC1−PNPE2E, CUQI8−PNPE2E) — pixel-wise class-index subtraction in the range −2…+2 via `_diff_figure()`, rendered with a diverging purple-to-gold colorscale where the dark midpoint indicates pixel-level agreement. The figure title shows the percentage of pixels that disagree, which helps quantify whether two algorithms produce "close but different" or "fundamentally different" reads of the same phantom.

**7. Voltage Measurement Pattern** (grouped bar chart) — x = measurement channel, one bar series per injection (up to 8 evenly sampled), y = voltage, via `_voltage_figure()`. A fixed categorical palette ensures readability on dark backgrounds; a double-click triggers a clientside callback that toggles a full-viewport overlay. This chart shows the raw electrical signal all three algorithms received — if all reconstructions look poor, a flat or low-variance signal here suggests the measurement data itself is the limiting factor, not the algorithms.

**8. Metrics Scorecard** (table) — all 18 cached metrics for all three algorithms, grouped into Image Quality, Shape Matching, Class Specific, Data Efficiency, and Measurement Domain sections. The best value per row is starred, and the sidebar-selected algorithm's column is highlighted. Built by `_scorecard_children()`, which silently backfills missing or stale keys without triggering a new adapter run.

### M4 · Fingerprint Radar (`m4_fingerprint_radar.py`) — owner: Asmita Bhuva

A 9-axis radar chart giving a compact view of each algorithm's overall performance profile rather than any individual metric.

**1. Fingerprint Radar** (`Scatterpolar`, 9 axes) — one filled polygon per algorithm across Total SSIM, Easy SSIM (levels 1–3), Hard SSIM (levels 5–7), IoU Conductive, IoU Resistive, Speed, Robustness, Voltage Residual, and Current Sensitivity. `_compute_axes()` aggregates cached metrics differently per axis: SSIM axes average across the relevant level range; IoU and Speed use the exact selected `(level, sample)`; Robustness is the standard deviation of SSIM across samples at the selected level. All values are then min-max normalised to [0, 1] across the three algorithms. Axes where lower values indicate better performance (Speed, Robustness, Voltage Residual) are inverted, so a larger polygon always corresponds to better overall performance.

**2. Algorithm profile mini-bars** — three cards showing each algorithm's three highest-scoring normalised axes as horizontal bars. These provide a concise summary of each algorithm's strengths without requiring the reader to decode the radar geometry.

**3. Performance Heatmap table** — an axis-by-algorithm grid where each cell shows a mini progress bar, the normalised [0, 1] score, and the raw metric value. The best value per row is starred. This table complements the radar: the polygon is useful for comparing shapes, but reading exact values from a polar axis is unreliable, and this table gives the actual numbers.

### M5 · Failure Autopsy (`m5_failure_autopsy.py`) — owners: Muzammal & Asmita Bhuva

Lists all cached `(algorithm, level, sample)` combinations ranked by ascending SSIM, with an algorithm filter and a top-N slider (5–25). Clicking a row snaps the shared sidebar to that case and loads four diagnostic panels. The goal is to identify the root cause of a specific failure, not just flag that the SSIM is low.

**1. Ranked worst-case list** — rows sorted by ascending SSIM via `_load_all_cases()`, each showing the algorithm, level/sample, an SSIM mini-bar, and an auto-assigned failure badge. Rather than stepping through 84 combinations manually, the analyst can navigate directly to the worst-performing cases.

**2. Spatial SSIM heatmap** (RdYlGn) — a 256×256 map of local SSIM values via `compute_spatial_ssim_map()`. This identifies which spatial region of the image is responsible for dragging the aggregate score down.

**3. Confusion Matrix heatmap** (3×3) — rows = ground truth class, columns = predicted class, cells = row-normalised percentages, via `compute_confusion_matrix()`. This is the direct evidence for which specific misclassification occurred, and it is also the primary input to the automatic failure classifier.

**4. Boundary Error polar histogram** — a 36-bin angular histogram of mismatched pixels around the image centroid, built from error-pixel (dy, dx) offsets via `arctan2`, with active and removed electrode positions overlaid as coloured markers. Error clusters that align with removed electrode sectors can often be attributed to reduced coverage rather than algorithmic failure.

**5. Measurement Perturbation polar** — per-injection `‖V − V_ref‖` RMS energy as a polar bar chart via `_per_injection_perturbation()`, with a dashed threshold ring at the 25th percentile. Injections below the threshold (red bars) carried little inclusion signal to begin with, which helps distinguish a data-limited case from a case where the signal was present but the algorithm failed to use it.

**6. Signal breakdown bars** — one horizontal bar per failure code A–E showing its raw classifier score from `_classify_failure()`, with the winning code highlighted. This makes the auto-assigned badge auditable: the analyst can see the margin between the winning and runner-up codes and judge how confident the classification is.

**7. Ground Truth vs. Prediction thumbnails** — two small segmentation heatmaps side by side, as a quick visual reference without requiring the user to switch to M1.

Failure taxonomy:

| Code | Name | Signal |
|---|---|---|
| A | Ghost inclusion | FP-dominant (predicted non-water over true water) |
| B | Missing inclusion | FN-dominant (predicted water over true inclusion) |
| C | Class flip | Resistive↔conductive swap concentrated off-diagonal |
| D | Boundary erosion | Reasonable IoU but predicted footprint materially smaller than GT |
| E | Mask suppression | PNPE2E-specific: FN-heavy with balanced inclusion-class confusion |

### M6 · Measurement Domain Viewer (`m6_measurement_viewer.py`) — owners: Smit Savani & Shimul Paul

Explores the raw electrical measurements independently of any reconstruction (the algorithm selector has no effect on this module). An injection-step slider (0–75) with play animation at 0.9 s per step and a CSV export button drives the injection-dependent panels. Two separate callbacks split fast injection-dependent updates from slower level/sample updates so the play animation remains smooth.

**1. Current Pattern** (polar bar) — one bar per electrode via `current_polar_figure()`, with red = source, blue = sink, radius = current magnitude, and grey crosses for removed electrodes. This shows the physical injection configuration for the selected step.

**2. Electrode Voltages** (polar overlay) — measured voltage and empty-tank reference as solid and dashed lines respectively via `voltage_polar_figure()`, with angles from `_pair_angles()` using the `Mpat` matrix. Divergence between the two lines indicates where an inclusion is disturbing the current path.

**3. Voltage Difference ΔV** (bar) — `V_measured − V_reference` per measurement pair via `voltage_diff_figure()`. This is the input signal the reconstruction algorithms work from; pairs with large |ΔV| carry the most information about the inclusion location.

**4. R per Pair** (bar) — `R = V/I` per pair via `resistance_bar_figure()`, coloured green for positive and red for negative values. A sign reversal can indicate an electrode contact problem or strong current deflection.

**5. ΔR = R − R_ref** (bar) — the resistance-domain equivalent of ΔV, via `resistance_delta_figure()`. Large bars indicate where the inclusion perturbs the current field in the same units that CUQI8's forward model uses internally.

**6. Signal-to-Noise per Pair** (bar, YlOrRd colorscale) — `|ΔV| / |V_ref|` per pair via `snr_per_pair_figure()`. Low SNR pairs contribute noise rather than information, which can explain poor reconstructions that are not obviously caused by the algorithm itself.

**7. All Injections Overlay** (resistance) — every injection's R-profile as a faint background line with the selected injection highlighted in yellow, via `resistance_overlay_figure()`. Background traces are merged into a single `None`-separated `Scatter` for animation performance. A highlighted line that falls well outside the cluster indicates that injection is particularly sensitive to the inclusions.

**8. Mean ± Std Summary** (resistance) — mean R per pair with a shaded ±1 standard deviation band and variability-coloured markers, via `resistance_summary_figure()`. This panel is static per `(level, sample)` and shows which pairs are most informative across the entire injection protocol rather than for a single step.

**9. Anomaly Score per Injection** (bar) — `Σ|ΔR|` per injection with the current step highlighted, via `anomaly_score_figure()`. This ranks injection steps by inclusion information content and can also identify outlier injections that may indicate hardware issues.

**10. Measurement Stability / CV** (bar) — coefficient of variation (`std/|mean|`) per pair across all injections, with a 2× median threshold line, via `measurement_stability_figure()`. High CV can indicate either genuine sensitivity to inclusions (useful) or measurement instability (a data-quality concern); the threshold helps distinguish the two.

**11. Electrode Contact Quality** (bar) — mean `|R|` per electrode across all its pairs, with a 2× median threshold, via `electrode_impedance_figure()`. Electrodes consistently above the threshold may have poor galvanic contact, which can explain spatially localised reconstruction errors.

**12. Coverage per Level** (dual-axis bar + line) — active electrode count per level (bars) and measurement pair count (line, right axis), with the current level highlighted, via `level_coverage_figure()`. This ties the measurement reduction directly to M2's degradation curves by showing the data-availability root cause.

**13. Electrode Coverage Polar Map** — a full-circle electrode layout showing active (blue) and removed (grey ×) electrodes, current source/sink positions, and sampled measurement-pair arcs, via `coverage_polar_figure()`. This updates during injection play and gives the most complete picture of how the measurement protocol changes as injection steps progress and as difficulty levels increase.

Every M6 panel includes a short inline interpretation note. Panels 3, 5, 6, and 9 fall back to a "reference unavailable" placeholder if `measurements/ref.mat` is missing.

## 7. Shared UI Building Blocks

`ktc_vis/dashboard/theme.py` is the single source of truth for the dark colour palette (`BG`, `SURFACE`, `CARD`, `BORDER`, `TEXT`, `MUTED`, `ACCENT`, `SUCCESS`, `WARN`, `DANGER`) and the shared `CARD_STYLE` and `SECTION_LABEL_STYLE` dictionaries used by every module.

`ktc_vis/utils/figures.py` provides `segmentation_figure`, `reconstruction_figure`, `error_overlay_figure`, and `empty_figure`. It centralises the class colorscale (`CLASS_COLORS`/`CLASS_LABELS`) and the error-overlay palette (`ERROR_COLORS`) so that M1's legend and figure colours cannot diverge from the other modules that use the same utilities.

`ktc_vis/dashboard/components/sidebar.py` defines the three shared selectors. Their component IDs — `sidebar-algorithm-dropdown`, `sidebar-level-slider`, `sidebar-sample-radio` — are effectively a contract; every module's callbacks depend on them by name and any rename would break all six modules simultaneously.

## 8. Scripts

| Script | Purpose |
|---|---|
| `scripts/stage_dataset.py` | Consolidates `ktc_vis/external/<repo>/...` into the canonical `data/raw/ktc2023/{measurements,ground_truth,reference_outputs}` layout. Run once after cloning the external algorithm repos. |
| `scripts/validate_data.py` | Checks per-level `.mat` files exist, have the right keys, and ground truth is `(256,256)` with only `{0,1,2}` values. |
| `scripts/run_benchmark.py` | Live Docker benchmark. Supports `--algorithms/--levels/--samples`, `--overwrite`, `--stream`, `--timeout`, `--no-batch`. Uses per-level batching automatically for adapters with `supports_level_batching = True`, with an ETA display and Apple Silicon warning. |
| `scripts/populate_cache_from_reference.py` | Fast path — computes metrics against `ReferenceOutputAdapter` outputs instead of running Docker. This is the recommended first step to get the dashboard populated. |
| `scripts/benchmark_runtime_abc1.py` / `benchmark_runtime_pnpe2e.py` | Time real Docker runs per level and patch just the `runtime` key into existing cache entries (supports `--per-sample` for isolated per-sample timing, `--dry-run`). |
| `scripts/setup_third_party.sh` | Prepares/clones the external algorithm repositories under `ktc_vis/external/`. |

Typical bring-up order: `setup_third_party.sh` → `stage_dataset.py` → `validate_data.py` → `populate_cache_from_reference.py` → `run_benchmark.py` (only when Docker images are available and real runtimes are needed).

## 9. Environment, Docker & CI

`environment.yml` specifies Python 3.10.13 with `dash 2.14.2`, `plotly 5.18.0`, `numpy 1.26.2`, `scipy 1.11.4`, `scikit-image 0.22.0`, `scikit-learn 1.3.2`, and `pytorch 2.0.1` (CPU-only via `cpuonly`), plus pip extras: `pyeit`, `mat73`, `torch-geometric 2.4.0`, `deepinv 0.1.0`, `dash-bootstrap-components`. FEniCS/dolfinx is not installed here — it runs inside the separate `ktc2023-cuqi8` Docker image and is invoked only through `CUQI8Adapter`.

`requirements.txt` is a pip-only equivalent for development without Conda.

The `Dockerfile` uses a `continuumio/miniconda3` base with a static Docker CLI binary (so the app container can launch the algorithm images via the mounted host socket), builds the `ktc-vis` conda env, installs the package with `pip install -e . --no-deps`, and exposes port 8050 with a healthcheck against `/`.

`docker-compose.yml` mounts `./data` (shared cache), `/var/run/docker.sock` (so adapters can launch sibling containers), and hot-reloads `ktc_vis/`, `app.py`, and `configs/` for development.

`.github/workflows/ci.yml` uses `conda-incubator/setup-miniconda` with mamba, runs `flake8 ktc_vis/ --max-line-length=100` (no ignore flags — all violations are fixed at source), then `pytest tests/ --cov=ktc_vis` excluding `test_adapters.py` (which requires the real dataset and Docker and is intended to run locally). Coverage is uploaded via `codecov-action`.

## 10. Testing — Current State

`pyproject.toml` configures `pytest` with `pytest-cov` (source=`ktc_vis`, excluding `dashboard/modules/*`) and a `requires_data` marker for tests that need the real KTC2023 dataset.

`tests/test_modules/test_glossary.py` contains real assertions covering glossary term lookup, case-insensitive access, and the `with_tooltip`, `info_pill`, and `glossary_term` helper functions.

All other test files — `test_loader.py`, `test_cache.py`, `test_metrics.py`, and `test_modules/test_m1.py` through `test_m6.py` — currently contain placeholder stubs (`def test_placeholder(): assert True`). `test_adapters.py` requires Docker and the staged dataset and is excluded from the default CI job by design. Until the stub files are completed, a green CI run indicates that the code is importable and style-compliant, but does not verify correctness of the data pipeline, metrics computation, or module callbacks. Completing these tests is the highest-priority remaining gap.

## 11. Coding Standards

All Python code targets version 3.10 and follows PEP 8 with a maximum line length of 100 characters, enforced by flake8 without any ignore exceptions. Long lines are fixed in source rather than suppressed. Public functions and methods carry type hints. Module docstrings identify an "Owner" in most files, reflecting the original team assignment rather than a hard ownership boundary; the authoritative ownership map is the table in §0.

Domain constants (electrode counts, level-to-measurement tables, sample mappings) are kept in `subsampler.py`, `loader.py`, and `configs/experiment.yaml` rather than scattered as magic numbers through module code.

For Dash and Plotly: every component ID is unique and module-prefixed (`m{N}-...`); callbacks declare explicit `Output`, `Input`, and `State` lists; figures set `uirevision` to preserve zoom and pan state across data updates.

Git conventions: branch names follow `feature/<name>`, `fix/<name>`, or `docs/<name>`; commit messages use `[scope] short description` format; direct commits to `main` are not permitted.

## 12. Contacts

| Topic | Contact |
|-------|---------|
| Data loading / physics / adapters / M1, M5 | Muhammad Muzammal (1541353) |
| Metrics engine / cache / CUQI8 & PNPE2E adapters / M3, M6 | Smit Savani (1420825) |
| Dashboard UI design / M4, M5 | Asmita Bhuva (1541650) |
| Docker / CI / environment / M2, M6 | Shimul Paul (1441927) |
| Academic supervision | Prof. Dr. Martin Simon & Emanuele Pepe |

See §0 for the full team and course context, and §6 for per-module ownership headers.
