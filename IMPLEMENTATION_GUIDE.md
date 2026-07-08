# KTC-Vis — Implementation Guide

---

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

---

## 1. What This Project Is

KTC-Vis is a Dash dashboard for benchmarking three EIT (Electrical Impedance Tomography) reconstruction algorithms — **ABC1**, **CUQI8**, **PNPE2E** — against the KTC2023 competition dataset, across 7 difficulty levels (32 → 20 electrodes) and 4 phantom samples (a–d).

Two parallel data paths feed the dashboard:

1. **Reference-output path** (fast, no Docker) — `ReferenceOutputAdapter` reads pre-computed `.mat` reconstructions staged from each algorithm's original repo under `data/raw/ktc2023/reference_outputs/<algo>/level<N>/`. This is what powers M1 and M3 by default and what `scripts/populate_cache_from_reference.py` uses to fill the HDF5 cache quickly.
2. **Live algorithm path** (slow, Docker-based) — `ABC1Adapter`, `CUQI8Adapter`, `PNPE2EAdapter` each shell out to a published Docker image, write subsampled measurement `.mat` files, run the container, and read back a reconstruction. `scripts/run_benchmark.py` drives this path to (re)populate the cache with real reconstructions and real runtimes.

Both paths converge on the same `MetricsEngine` and the same HDF5 cache format, so the dashboard doesn't care which one produced a given cached result.

---

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
│   │   └── efficiency.py               # measure_runtime()
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
│   └── test_modules/test_m1.py … test_m6.py
├── docs/
│   ├── adding_new_algorithm.md
│   └── interpreting_dashboard.md
└── IMPLEMENTATION_GUIDE.md              # This file
```

---

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

---

## 4. Adapters

| Adapter | File | Mechanism |
|---|---|---|
| `ReferenceOutputAdapter` | `reference_adapter.py` | Reads `data/raw/ktc2023/reference_outputs/<algo>/level<N>/<file>.mat` (no Docker). Falls back to the legacy flat layout at Level 1. Filename convention differs per algo: `abc1` → `data{idx}.mat`, `cuqi8`/`pnpe2e` → `{idx}.mat`. |
| `ABC1Adapter` | `abc1_adapter.py` | Docker image `muzammal5566/ktc2023-abc-python:latest`. Writes subsampled `data{idx}.mat` + `ref.mat` to a temp `TrainingData/` dir, runs `python main_python.py TrainingData Outputs <level>`, reads back the first 2-D array in the output `.mat`. |
| `CUQI8Adapter` | `cuqi8_adapter.py` | Docker image `muzammal5566/ktc2023-cuqi8:latest`. `supports_level_batching = True` — one container run processes all samples for a level via `conda run -n env python main.py TrainingData Output <level>`. |
| `PNPE2EAdapter` | `pnpe2e_adapter.py` | Docker image `muzammal5566/ktc2023-pnpe2e:latest`. Also level-batched. Monkey-patches `torch.load` to `map_location='cpu'` before importing `main` so CUDA-trained weights load on CPU-only hosts. |

All three live adapters need Docker on the host and `--platform linux/amd64` (the images are x86_64-only; Apple Silicon runs them under Rosetta 2 — `run_benchmark.py` prints a warning when it detects `arm64`).

---

## 5. Metrics Engine

`MetricsEngine(adapter, cache_path).compute_all(measurement)` — checks the cache first (`is_cached`), otherwise calls `adapter.reconstruct()`, times it, and computes the full metric battery via `_compute_metrics_from_reconstruction()`. Every call writes the result back to the HDF5 cache.

18 keys are computed per `(algorithm, level, sample)`:

| Group | Keys | Notes |
|---|---|---|
| Image quality | `ssim`, `ssim_min` | `skimage.metrics.structural_similarity`, `data_range=2.0`; `ssim_min` = worst local window from the same spatial map used by M5. |
| Shape matching | `hausdorff`, `position_error`, `resolution` | Symmetric Hausdorff over foreground points; centroid Euclidean offset; diameter of the smallest pred∩gt connected region. |
| Class-specific | `iou_water`, `iou_resistive`, `iou_conductive`, `iou_mean`, `dice_water`, `dice_resistive`, `dice_conductive`, `dice_mean`, `confusion_accuracy` | Per-class IoU/Dice from `class_metrics.py`; `confusion_accuracy` = mean of the confusion matrix diagonal. |
| Measurement domain | `voltage_residual`, `resistance_consistency`, `current_sensitivity` | Computed from the measurement alone (no reconstruction needed) — see `metrics/measurement.py`. `voltage_residual` = normalised RMS deviation from the per-injection mean; `resistance_consistency` and `current_sensitivity` are `1 − clip(CoV, 0, 1)` homogeneity/balance scores. |
| Efficiency | `runtime` | Wall-clock seconds for `adapter.reconstruct()` (or the reference-output read cost, which is near-zero and treated as "not real" by M2/M4 — see the `_RUNTIME_REAL_THRESHOLD_S` guard). |

M3's scorecard (`_METRIC_KEYS`) is the single source of truth for display labels/format specs/higher-is-better flags for all 18 metrics plus `dice_mean`/`dice_*` duplicates.

---

## 6. Dashboard Modules

Shared conventions: every module is a `layout()` + `register_callbacks(app)` pair registered from `ktc_vis/dashboard/layout.py`; all component IDs are prefixed `m{N}-`; all figures use `uirevision="constant"` (or a module-specific string) so zoom/pan state survives re-renders; the three sidebar controls (`sidebar-algorithm-dropdown`, `sidebar-level-slider`, `sidebar-sample-radio`) are shared `Input`s across every module.

For every graph below: **What** = what's on screen, **How** = where the data comes from and how the figure is built, **Why** = the decision or insight the graph exists to support.

### M1 · Reconstruction Explorer (`m1_reconstruction_explorer.py`) — owner: Muzammal

Two-row layout for one `(algorithm, level, sample)`, backed by `ReferenceOutputAdapter`. Purpose: the single-case "is this reconstruction any good, and why/why not" view.

**1. Ground Truth heatmap** (`m1-gt-image`)
- *What:* 256×256 categorical heatmap of the true phantom (water / resistive / conductive), 3-color scale.
- *How:* `segmentation_figure(gt)` (`utils/figures.py`) — hard-stepped colorscale at 0.33/0.66, axes hidden, aspect locked, y reversed to match image convention.
- *Why:* the answer key. Every other panel in the row is judged against this one, so it anchors the user's eye before they look at the reconstruction.

**2. Reconstruction heatmap** (`m1-recon-image`)
- *What:* continuous Viridis heatmap of the algorithm's raw output for the same case.
- *How:* `reconstruction_figure(recon)` — zmin/zmax set from the array's own min/max, colorbar shown.
- *Why:* a "photographic" view before the output is hardened into classes — reveals smoothing, blur, or soft-boundary artifacts that the categorical Segmentation panel would hide.

**3. Segmentation heatmap** (`m1-seg-image`)
- *What:* the same reconstruction re-rendered on the identical 3-color scale used for Ground Truth.
- *How:* `segmentation_figure(recon)`.
- *Why:* apples-to-apples visual diff against panel 1 — isolates "did it find the right regions" from "what does the raw output look like" (panel 2's job).

**4. Error Overlay heatmap** (`m1-error-image`)
- *What:* one map encoding four outcomes per pixel — green=correct, red=false positive, blue=false negative, yellow=wrong class.
- *How:* `error_overlay_figure(recon, gt)` builds a 0–4 integer code from boolean masks and a matching stepped colorscale; a legend is drawn directly onto the figure as an annotation.
- *Why:* the highest-signal debugging view in M1 — shows *where* the algorithm is wrong and *how*, and its aggregate (correct-pixel %) feeds the chip row's pixel-agreement figure.

*Supporting element:* the chip row (algorithm, level, sample, electrode/injection/voltage-sample counts, pixel agreement %) is a KPI strip, not a chart, but it's what tells you at a glance whether a low SSIM elsewhere in the app is worth investigating here.

### M2 · Difficulty Animator (`m2_difficulty_animator.py`) — owner: Shimul Paul

Level 1–7 slider with ▶ Play/Pause auto-advance (1.2 s/level, wraps 7→1). Purpose: show *how* — gradually or suddenly — an algorithm's quality collapses as electrodes are removed.

**1. Ground Truth image** (`m2-gt-graph`)
- *What:* the true phantom, fixed across the whole animation.
- *How:* loaded directly from `true{idx}.mat` via `scipy.io`, independent of `utils/figures.py` (module keeps its own 3-color scale).
- *Why:* a deliberately non-moving reference — because this panel never changes, the user's eye is free to focus on how the other two panels degrade as the level slider moves.

**2. Segmentation image** (`m2-recon-graph`)
- *What:* heatmap of the cached reconstruction at the current level.
- *How:* `load_result()` from the HDF5 cache.
- *Why:* the "moving part" of the animation — this is what visibly degrades (blurs, drops inclusions, shifts boundaries) as the level increases.

**3. Error Overlay image** (`m2-error-graph`)
- *What:* the same 4-way error coding as M1, recomputed per level.
- *How:* `_recon_and_error_figures()` (module-local, mirrors `utils/figures.error_overlay_figure` logic).
- *Why:* the animation's payoff frame — watching red/blue/orange pixel area visibly grow as L increases is the clearest possible demonstration of "less data = worse reconstruction."

**4–10. Degradation curves** (SSIM, Mean IoU, Dice, Hausdorff, Position Error, Resolution, Runtime — 7 line charts)
- *What:* x = level 1–7, y = metric value, one point per level for the selected algorithm+sample, with a vertical dashed marker at whichever level is currently selected.
- *How:* `_curve_figure()` reads `results/{alg}/{level}/{sample}/{metric}` from the HDF5 cache for all 7 levels; Dice is derived on the fly from cached IoU (`2·IoU/(1+IoU)`, no separate cache key); SSIM/IoU/Dice/Runtime use fixed y-axis ranges so the chart never auto-rescales in a way that makes a bad algorithm look falsely stable; Runtime is suppressed unless the cached value exceeds a 0.01 s "is this a real timing, not a cache-read artifact" threshold.
- *Why:* this is the actual "difficulty curve" the module is named for — it answers "does this algorithm fail gracefully or fall off a cliff, and at which level?" The dashed marker ties the abstract curve back to whatever concrete level the images above are showing.

*Supporting elements (not charts):* the **level context progress bar** (% of Level-1 measurements remaining, with a mild/moderate/severe description) and the **Level Analysis commentary panel** (auto-generated per-metric labels like "Excellent"/"Poor", plus a synthesized sentence flagging the level where SSIM first drops below 0.60) translate the curves above into plain language — useful for a viewer who wants the conclusion without reading 7 charts.

### M3 · Side-by-Side Comparison Grid (`m3_comparison_grid.py`) — owner: Smit Savani

All three algorithms for one `(level, sample)` at once. Purpose: head-to-head comparison without repeatedly flipping the sidebar's algorithm dropdown.

**1–3. Algorithm reconstruction heatmaps** (ABC1 / CUQI8 / PNPE2E)
- *What:* three 3-class segmentation heatmaps, one per algorithm, same case.
- *How:* `_recon_figure()`, categorical colorscale matching `CLASS_COLORS`; each panel loads cache-first, live-adapter-fallback via `_load_reconstruction()`.
- *Why:* the primary "which algorithm looks best on this exact phantom" view — side-by-side beats sequential because subtle differences (a slightly smaller inclusion, a rounder boundary) are much easier to spot in direct comparison.

**4–6. Pairwise difference heatmaps** (ABC1−CUQI8, ABC1−PNPE2E, CUQI8−PNPE2E)
- *What:* pixel-wise class-index subtraction, range −2…+2, diverging purple↔gold colorscale (dark middle = agreement).
- *How:* `_diff_figure()` computes `recon_a.astype(int16) - recon_b`, counts nonzero pixels for a "% disagree" figure title.
- *Why:* quantifies *where* and *how much* two algorithms disagree — distinguishes "close SSIM scores but boundary jitter only" from "fundamentally different read of the phantom," which a side-by-side glance alone can't reliably tell apart.

**7. Voltage Measurement Pattern** (grouped bar chart)
- *What:* x = measurement channel, one colored bar-series per injection (up to 8 evenly sampled for readability), y = voltage.
- *How:* `_voltage_figure()` — fixed categorical palette (`_INJECTION_COLORS`, validated for dark-background CVD separation), range-slider appears above 32 channels, double-click triggers a clientside callback that toggles a full-viewport overlay.
- *Why:* shows the raw electrical signal every algorithm is working from, independent of any reconstruction — if the reconstructions all look poor, this chart tells you whether the *data itself* was weak (flat, low-variance channels) rather than the algorithms being at fault.

**8. Metrics Scorecard** (table)
- *What:* all 18 cached metrics for all 3 algorithms, organized into Image Quality / Shape Matching / Class Specific / Data Efficiency / Measurement Domain sections, best-per-row starred, sidebar-selected algorithm's column highlighted.
- *How:* `_scorecard_children()` pulls per-algorithm cached metrics and silently backfills missing/stale keys (including auto-correcting legacy saturated measurement-domain values) without re-running the adapter.
- *Why:* the numeric ground truth behind the visual panels — answers "which algorithm actually wins here, and by how much" instead of relying on eyeballing heatmaps.

### M4 · Fingerprint Radar (`m4_fingerprint_radar.py`) — owner: Asmita Bhuva

Purpose: a single glance at each algorithm's overall performance *shape*, not just a number.

**1. Fingerprint Radar** (`Scatterpolar`, 9 axes)
- *What:* one filled polygon per algorithm across Total SSIM, Easy SSIM (L1–3), Hard SSIM (L5–7), IoU Conductive, IoU Resistive, Speed, Robustness, Voltage Residual, Curr. Sensitivity.
- *How:* `_compute_axes()` aggregates cached metrics differently per axis — SSIM axes average across a level range for the selected sample; IoU/Speed use the exact selected `(level, sample)`; Robustness = std of SSIM across samples at the selected level. All axes are then min-max normalised to [0,1] across the three algorithms, and "lower-is-better" axes (Speed, Robustness, Voltage Residual) are inverted so *outward always means better* on every axis.
- *Why:* this is the tool's namesake view — a wide/low polygon vs. a narrow/tall one visually is the algorithm-comparison tradeoff table from the README (fast-but-fragile vs. slow-but-robust) turned into one shape you can compare at a glance, rather than reading three separate numbers.

**2. Algorithm profile mini-bars** (3 cards, top-3 axes each)
- *What:* a short horizontal bar per algorithm's 3 highest-scoring axes.
- *How:* sorts each algorithm's normalised axis dict descending, takes the top 3.
- *Why:* a one-line textual/visual summary of "what is this algorithm actually best at" without requiring the reader to parse the full 9-axis radar first.

**3. Performance Heatmap table**
- *What:* axis × algorithm grid, each cell showing a mini progress bar, the normalised [0,1] score, and the raw metric value; best-per-row starred; a "Measurement Domain" section divider precedes the last two axes.
- *How:* reuses the same normalised + raw dictionaries computed for the radar.
- *Why:* the radar is good for shape but bad for reading exact values off a polar axis — this table is the precise companion view for when a user needs to cite an actual number rather than "PNPE2E's polygon is bigger here."

### M5 · Failure Autopsy (`m5_failure_autopsy.py`) — owners: Muzammal & Asmita Bhuva

Ranks every cached `(algorithm, level, sample)` by ascending SSIM (filterable by algorithm, top-N slider 5–25); clicking a row snaps the shared sidebar to that case. Purpose: root-cause a specific bad result, not just flag that it's bad.

**1. Ranked worst-case list**
- *What:* rows sorted by SSIM ascending, each showing algorithm, level/sample, an SSIM mini-bar, and the auto-assigned failure badge.
- *How:* `_load_all_cases()` loads every cached combination, sorts, truncates to the top-N filter.
- *Why:* the entry point for failure hunting — instead of manually stepping through 84 combinations, jump straight to the worst ones.

**2. Spatial SSIM heatmap** (RdYlGn)
- *What:* 256×256 map of local SSIM values (−1…1) for the selected case.
- *How:* `compute_spatial_ssim_map()`, the same underlying `structural_similarity` call used for the scalar score.
- *Why:* pinpoints exactly *which region* of the image is dragging the aggregate score down, instead of only knowing the number is low.

**3. Confusion Matrix heatmap** (3×3)
- *What:* rows = ground truth class, columns = predicted class, cell text = row-normalised percentage.
- *How:* `compute_confusion_matrix()`.
- *Why:* the direct evidence for *which* misclassification happened (e.g., resistive predicted as conductive) — and it's literally the input to the failure classifier below, so it's what explains the assigned badge.

**4. Boundary Error polar histogram**
- *What:* 36-bin angular histogram of mismatched pixels around the image centroid, overlaid with green dots (active electrodes) and gray ✕ marks (removed electrodes) for the selected level.
- *How:* error-pixel `(dy, dx)` offsets converted to angle via `arctan2` and binned; electrode angles from `subsample_electrodes(level)`.
- *Why:* reveals whether errors cluster in a specific angular sector — and whether that sector lines up with removed electrodes, turning a statistical anomaly into a physically explainable one (fewer electrodes there → less information → worse reconstruction there).

**5. Measurement Perturbation polar**
- *What:* per-injection `‖V − V_ref‖` RMS energy as a polar bar chart, with a dashed ring at the 25th-percentile "weak signal" threshold.
- *How:* `_per_injection_perturbation()`; injections below the threshold are colored red.
- *Why:* distinguishes a "hard measurement" failure (most injections carried little inclusion signal to begin with — a data problem) from an "algorithm" failure (signal was present but still not reconstructed correctly).

**6. Signal breakdown bars** ("why this badge?")
- *What:* one horizontal bar per failure code A–E showing its raw classifier score, winning code highlighted.
- *How:* `_classify_failure()`'s returned `signal_scores` dict.
- *Why:* makes the auto-assigned failure badge auditable instead of a black box — shows the runner-up codes and by how much the winner beat them, so a skeptical user can judge whether the classification is confident or a close call.

**7. Ground Truth vs. Prediction thumbnails**
- *What:* two small heatmaps side by side.
- *How:* lightweight versions of the M1 segmentation figure.
- *Why:* a fast visual sanity check alongside the numeric diagnostics, without needing to flip over to M1 for the same case.

Failure taxonomy (what the classifier in panel 6 is choosing between):

| Code | Name | Signal |
|---|---|---|
| A | Ghost inclusion | FP-dominant (predicted non-water over true water) |
| B | Missing inclusion | FN-dominant (predicted water over true inclusion) |
| C | Class flip | Resistive↔conductive swap concentrated off-diagonal |
| D | Boundary erosion | Reasonable IoU but predicted footprint materially smaller than GT |
| E | Mask suppression | PNPE2E-specific: FN-heavy with balanced inclusion-class confusion |

### M6 · Measurement Domain Viewer (`m6_measurement_viewer.py`) — owners: Smit Savani & Shimul Paul

Explores raw electrical measurements independent of any reconstruction (the algorithm selector has no effect here — the header shows a UI note to that effect). Injection-step slider (0–75) with ▶ Play (0.9 s/step) and a CSV export button. Two callbacks split fast (injection-dependent) vs. slow (level/sample-only) figures so play animation stays smooth. Purpose: understand the *data* an algorithm receives, before judging what it did with it.

**1. Current Pattern** (polar bar)
- *What:* one bar per electrode; red = source (+), blue = sink (−), radius = current magnitude, gray ✕ = removed at this level.
- *How:* `current_polar_figure()` reads `measurement.current_matrix[inj_idx]`.
- *Why:* shows exactly which two electrodes drive current for the selected injection step — the physical starting point of the entire measurement.

**2. Electrode Voltages** (polar overlay)
- *What:* solid line = measured voltage (with inclusions) per pair at its electrode angle; dashed line = empty-tank reference.
- *How:* `voltage_polar_figure()`; angles from `_pair_angles()` using the `Mpat` matrix.
- *Why:* where the two lines diverge is where an inclusion is disturbing the current path — the qualitative, whole-injection version of the ΔV bar chart below.

**3. Voltage Difference ΔV** (bar)
- *What:* `V_measured − V_reference` per measurement pair; green = increase, red = decrease.
- *How:* `voltage_diff_figure()`.
- *Why:* the literal input signal used by the reconstruction algorithms — large `|ΔV|` pairs are the ones actually carrying information about the inclusion.

**4. R per Pair** (bar)
- *What:* `R = V/I` per pair; green = positive (expected), red = negative (sign reversal).
- *How:* `resistance_bar_figure()`.
- *Why:* a sign reversal can flag an electrode contact problem or strong current deflection — an early red flag before even looking at reconstruction quality.

**5. ΔR = R − R_ref** (bar)
- *What:* the resistance-domain counterpart of ΔV.
- *How:* `resistance_delta_figure()`.
- *Why:* pre-reconstruction anomaly signal — large bars show exactly where the inclusion perturbs the current field, in the same units CUQI8's forward model reasons in.

**6. Signal-to-Noise per Pair** (bar, YlOrRd colorscale)
- *What:* `|ΔV| / |V_ref|` per pair.
- *How:* `snr_per_pair_figure()`.
- *Why:* identifies which pairs are actually informative vs. noise-dominated — useful for judging whether a poor reconstruction is a data-quality problem rather than an algorithm problem.

**7. All Injections Overlay** (resistance)
- *What:* every injection's R-profile as a faint background line, selected injection highlighted in yellow.
- *How:* `resistance_overlay_figure()` merges all background traces into one `None`-separated `Scatter` for smooth play-animation performance.
- *Why:* shows how typical (or atypical) the selected injection's resistance profile is relative to the whole protocol — a highlighted line far from the cluster signals that injection is especially sensitive to the inclusions.

**8. Mean ± Std Summary** (resistance)
- *What:* mean R per pair with a shaded ±1 std band, marker color = variability.
- *How:* `resistance_summary_figure()`; static per `(level, sample)` — doesn't change during injection play.
- *Why:* identifies the most/least informative pairs in aggregate across the whole protocol, not just for one injection.

**9. Anomaly Score per Injection** (bar)
- *What:* `Σ|ΔR|` per injection, current injection highlighted.
- *How:* `anomaly_score_figure()`.
- *Why:* ranks injection steps by how much inclusion information they carry — useful both for finding the most diagnostic injections and for spotting hardware-fault outliers.

**10. Measurement Stability / CV** (bar)
- *What:* coefficient of variation (`std/|mean|`) per pair across all injections, red above a 2×-median dashed threshold.
- *How:* `measurement_stability_figure()`.
- *Why:* separates "legitimately sensitive to inclusions" (high CV, useful) from "noisy/unstable electrode" (high CV, a data-quality concern) — a diagnostic on the data itself, independent of any algorithm.

**11. Electrode Contact Quality** (bar)
- *What:* mean `|R|` per electrode across all its pairs, red above a 2×-median threshold.
- *How:* `electrode_impedance_figure()`.
- *Why:* flags individual electrodes with likely poor galvanic contact — a hardware-level explanation for reconstruction problems localized to one part of the tank.

**12. Coverage per Level** (dual-axis bar + line)
- *What:* bars = active electrode count per level 1–7; line = measurement-pair count (right axis); current level highlighted.
- *How:* `level_coverage_figure()`.
- *Why:* visually justifies *why* quality degrades with level, tying directly back to M2's degradation curves by showing their root cause — shrinking data availability.

**13. Electrode Coverage Polar Map**
- *What:* full-circle electrode layout — blue = active, gray ✕ = removed, large red/blue dots = source/sink for the current injection, faint purple arcs = sampled measurement-pair connections; updates every injection step during play.
- *How:* `coverage_polar_figure()`.
- *Why:* the single most complete physical picture of the measurement protocol — builds intuition for how the injection pattern rotates around the tank and where coverage gaps open up at higher difficulty levels.

Every M6 panel carries an inline plain-English interpretation line explaining what to look for, and falls back to a "reference unavailable" placeholder on panels 3, 5, 6, and 9 when `measurements/ref.mat` is missing.

---

## 7. Shared UI Building Blocks

- **`ktc_vis/dashboard/theme.py`** — single source of truth for the dark palette (`BG`, `SURFACE`, `CARD`, `BORDER`, `TEXT`, `MUTED`, `ACCENT`, `SUCCESS`, `WARN`, `DANGER`) and `CARD_STYLE`/`SECTION_LABEL_STYLE` dicts reused by every module.
- **`ktc_vis/utils/figures.py`** — `segmentation_figure`, `reconstruction_figure`, `error_overlay_figure`, `empty_figure`; centralises the class colorscale (`CLASS_COLORS`/`CLASS_LABELS`) and the error-overlay palette (`ERROR_COLORS`) so M1's legend and figures never drift apart.
- **`ktc_vis/dashboard/components/sidebar.py`** — the three shared selectors; IDs (`sidebar-algorithm-dropdown`, `sidebar-level-slider`, `sidebar-sample-radio`) are contractual — every module's callbacks depend on them by name.

---

## 8. Scripts

| Script | Purpose |
|---|---|
| `scripts/stage_dataset.py` | Consolidates `ktc_vis/external/<repo>/...` into the canonical `data/raw/ktc2023/{measurements,ground_truth,reference_outputs}` layout. Run once after cloning the external algorithm repos. |
| `scripts/validate_data.py` | Checks per-level `.mat` files exist, have the right keys, and ground truth is `(256,256)` with only `{0,1,2}` values. |
| `scripts/run_benchmark.py` | Live Docker benchmark. Supports `--algorithms/--levels/--samples`, `--overwrite`, `--stream`, `--timeout`, `--no-batch`. Uses per-level batching automatically for adapters with `supports_level_batching = True`, with an ETA display and Apple Silicon warning. |
| `scripts/populate_cache_from_reference.py` | Fast path — computes metrics against `ReferenceOutputAdapter` outputs instead of running Docker. What you run first to get the dashboard populated. |
| `scripts/benchmark_runtime_abc1.py` / `benchmark_runtime_pnpe2e.py` | Time real Docker runs per level and patch just the `runtime` key into existing cache entries (supports `--per-sample` for isolated per-sample timing, `--dry-run`). |
| `scripts/setup_third_party.sh` | Prepares/clones the external algorithm repositories under `ktc_vis/external/`. |

Typical bring-up order: `setup_third_party.sh` → `stage_dataset.py` → `validate_data.py` → `populate_cache_from_reference.py` (instant dashboard data) → `run_benchmark.py` (real reconstructions/runtimes when Docker images are available).

---

## 9. Environment, Docker & CI

- **`environment.yml`** — Python 3.10.13, `dash 2.14.2` / `plotly 5.18.0`, `numpy 1.26.2` / `scipy 1.11.4` / `scikit-image 0.22.0` / `scikit-learn 1.3.2`, `pytorch 2.0.1` (CPU-only via `cpuonly`), pip extras: `pyeit`, `mat73`, `torch-geometric 2.4.0`, `deepinv 0.1.0`, `dash-bootstrap-components`. FEniCS/dolfinx is **not** installed in this environment — it lives inside the separate `ktc2023-cuqi8` Docker image, invoked at arm's length by `CUQI8Adapter`.
- **`requirements.txt`** — pip-only equivalent for quick local dev without Conda.
- **`Dockerfile`** — `continuumio/miniconda3` base + a static Docker CLI binary (so the app container can `docker run` the algorithm images via the mounted host socket), builds the `ktc-vis` conda env, installs the package with `pip install -e . --no-deps`, exposes 8050 with a healthcheck against `/`.
- **`docker-compose.yml`** — mounts `./data` (shared cache), `/var/run/docker.sock` (so adapters can launch sibling containers), and hot-reloads `ktc_vis/`, `app.py`, `configs/` for development.
- **`.github/workflows/ci.yml`** — `conda-incubator/setup-miniconda` + mamba, `flake8 ktc_vis/ --max-line-length=100 --ignore=E501,W503`, `pytest tests/ --cov=ktc_vis` (excludes `test_adapters.py`, which needs the real dataset/Docker and is meant to run locally), uploads coverage via `codecov-action`.

---

## 10. Testing — Current State

`tests/conftest.py` and every `tests/test_*.py` / `tests/test_modules/test_m*.py` file are currently **placeholder stubs** (`def test_placeholder(): assert True`). `pyproject.toml` already wires up `pytest` + `pytest-cov` (source=`ktc_vis`, excluding `dashboard/modules/*`) and a `requires_data` marker for tests needing the real KTC2023 dataset — the scaffolding is ready, the actual assertions are not written yet. Before relying on CI green as a correctness signal, treat this as the highest-priority gap:

- `test_loader.py` / `test_cache.py` / `test_metrics.py` — should assert against known-good values (e.g. SSIM within the paper's tolerance) once fixtures load real or synthetic `.mat` data.
- `test_adapters.py` — needs Docker + the staged dataset; correctly excluded from the default CI job.
- `test_modules/test_m*.py` — should assert that each module's callbacks return valid `Figure` objects for edge-case inputs (level 1, level 7, all three algorithms, missing cache).

---

## 11. Coding Standards

- **Python 3.10**, PEP 8, max line length 100 (flake8 config in `pyproject.toml` ignores `E501`/`W503` — line length is soft-enforced by convention, not the linter).
- Type hints on public functions/methods; module docstrings note an "Owner" in most files (historical — reflects original team assignment, not a hard ownership boundary).
- No magic numbers for domain constants — level/electrode tables live in `subsampler.py` / `loader.py` / `configs/experiment.yaml`.
- **Dash/Plotly:** every component ID is unique and module-prefixed (`m{N}-...`); callbacks declare explicit `Output`/`Input`/`State`; figures set `uirevision` to preserve zoom/pan across data updates.
- **Git:** branch naming `feature/<name>` / `fix/<name>` / `docs/<name>`; commit messages `[scope] short description`; no direct commits to `main`.

---

## 12. Contacts

| Topic | Contact |
|-------|---------|
| Data loading / physics / adapters / M1, M5 | Muhammad Muzammal (1541353) |
| Metrics engine / cache / CUQI8 & PNPE2E adapters / M3, M6 | Smit Savani (1420825) |
| Dashboard UI design / M4, M5 | Asmita Bhuva (1541650) |
| Docker / CI / environment / M2, M6 | Shimul Paul (1441927) |
| Academic supervision | Prof. Dr. Martin Simon & Emanuele Pepe |

See §0 for the full team/course context and §6 for per-module ownership headers.
