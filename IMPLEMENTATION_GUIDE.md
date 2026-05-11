# KTC-Vis — Implementation Guide & Task Assignments

> This document is the authoritative reference for **what each team member must build**, the **coding standards** to follow, and the **acceptance criteria** that define "done". Read this before writing a single line of code.

---

## 1. Team Roles & Ownership

### Role A: Muzammal & Smit — EIT/Data Specialist & Backend Engineer

**Owns:** Data loading pipeline, method adapter API, metrics engine implementation, backend tests, results cache organization.

### Role B: Asmita & Shimul — Viz/Frontend Developer & DevOps Lead

**Owns:** All Dash layouts, callbacks, visual design of 6 modules, Conda environment, Dockerfile, GitHub Actions CI, final handover archive.

> **Rule:** No one merges their own PR. Role A reviews Role B's visual PRs; Role B reviews Role A's backend PRs.

---

## 2. Repository Layout

```
ktc-vis/
├── app.py                        # Dash entry point — run this to launch dashboard
├── environment.yml               # Conda environment (all deps including FEniCS)
├── Dockerfile                    # Multi-stage Docker image
├── .github/
│   └── workflows/
│       └── ci.yml                # GitHub Actions CI pipeline
├── configs/
│   └── experiment.yaml           # Master experiment config (YAML)
├── data/
│   ├── raw/
│   │   └── ktc2023/              # .mat files from Zenodo (NOT committed to Git)
│   └── cache/
│       └── results.h5            # HDF5 results cache (NOT committed to Git)
├── ktc_vis/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py             # KTCDataLoader: parse .mat → current/voltage/resistance
│   │   └── subsampler.py        # Electrode subsampling per level
│   ├── adapters/
│   │   ├── __init__.py
│   │   ├── base.py               # Abstract AlgorithmAdapter base class
│   │   ├── abc1_adapter.py       # ABC1 wrapper
│   │   ├── cuqi8_adapter.py      # CUQI8 wrapper (FEniCS dependency)
│   │   └── pnpe2e_adapter.py     # PNPE2E wrapper (DeepInverse + torch-geometric)
│   ├── metrics/
│   │   ├── __init__.py
│   │   ├── engine.py             # MetricsEngine: computes all 14 metrics
│   │   ├── image_quality.py      # SSIM, spatial SSIM map
│   │   ├── shape_matching.py     # Hausdorff distance, position error, resolution
│   │   ├── class_metrics.py      # Per-class IoU, confusion matrix, F1
│   │   ├── measurement.py        # Voltage residual, resistance consistency
│   │   └── efficiency.py         # Runtime measurement
│   ├── cache/
│   │   ├── __init__.py
│   │   └── hdf5_store.py         # Read/write HDF5 results cache
│   ├── dashboard/
│   │   ├── __init__.py
│   │   ├── layout.py             # Top-level Dash app layout + sidebar
│   │   ├── components/
│   │   │   └── sidebar.py        # Shared algorithm/level/sample selectors
│   │   └── modules/
│   │       ├── m1_reconstruction_explorer.py
│   │       ├── m2_difficulty_animator.py
│   │       ├── m3_comparison_grid.py
│   │       ├── m4_fingerprint_radar.py
│   │       ├── m5_failure_autopsy.py
│   │       └── m6_measurement_viewer.py
│   └── utils/
│       ├── __init__.py
│       └── figures.py            # Shared Plotly figure helpers
├── scripts/
│   ├── run_benchmark.py          # CLI: run full benchmark and populate HDF5 cache
│   └── validate_data.py          # CLI: verify .mat files are correctly downloaded
├── tests/
│   ├── conftest.py               # Shared pytest fixtures (loaded data, mock cache)
│   ├── test_loader.py
│   ├── test_adapters.py
│   ├── test_metrics.py
│   ├── test_cache.py
│   └── test_modules/
│       ├── test_m1.py
│       ├── test_m2.py
│       ├── test_m3.py
│       ├── test_m4.py
│       ├── test_m5.py
│       └── test_m6.py
├── docs/
│   ├── adding_new_algorithm.md   # Guide for extending to a 4th algorithm
│   └── interpreting_dashboard.md # User guide for dashboard outputs
├── notebooks/
│   └── exploration.ipynb         # Data exploration scratch notebook
├── .gitignore
├── README.md
└── IMPLEMENTATION_GUIDE.md       # This file
```

---

## 3. Task Assignments: Per-File Ownership

### 3.1 Muhammad Muzammal (1541353) — EIT & Data Specialist

**Primary deliverables:**

#### `ktc_vis/data/loader.py`
- Class `KTCDataLoader` with method `load(level, sample)` returning a dataclass `KTCMeasurement` containing:
  - `current_matrix`: ndarray (n_injections × n_electrodes)
  - `voltage_matrix`: ndarray (n_injections × n_measurements)
  - `resistance_matrix`: ndarray derived as R = V / I
  - `ground_truth`: ndarray (256 × 256) with values {0=water, 1=resistive, 2=conductive}
  - `level`: int 1–7
  - `sample`: str "a" | "b" | "c"
- Validates electrode subsampling matches the official KTC dataset exactly:
  - Level 1: 32 electrodes, 2356 measurements
  - Level 2: 30 electrodes, 1624 measurements
  - Level 3: 28 electrodes, 1404 measurements
  - Level 4: 26 electrodes, 1200 measurements
  - Level 5: 24 electrodes, 1012 measurements
  - Level 6: 22 electrodes, 630 measurements
  - Level 7: 20 electrodes, 513 measurements
- Unit-tests: `tests/test_loader.py` — test each level/sample produces correct matrix shapes.

#### `ktc_vis/data/subsampler.py`
- Function `subsample_electrodes(level)` returning the exact electrode indices used per level.
- Must match Table 1 of the KTC2023 paper exactly.

#### `ktc_vis/adapters/base.py`
- Abstract base class `AlgorithmAdapter`:
  ```python
  class AlgorithmAdapter(ABC):
      name: str
      @abstractmethod
      def reconstruct(self, measurement: KTCMeasurement) -> np.ndarray:
          """Returns 256×256 uint8 array with values {0,1,2}."""
  ```

#### `ktc_vis/adapters/abc1_adapter.py`
- Wraps the ABC1 repository; loads CNN weights from Zenodo; returns 256×256 segmentation.
- Validates against published SSIM scores in the KTC2023 paper.

#### `ktc_vis/metrics/image_quality.py`
- `compute_ssim(pred, gt)` → float — must match published KTC organiser scores (tolerance ±0.01).
- `compute_spatial_ssim_map(pred, gt)` → ndarray 256×256 — used by M5 Failure Autopsy.

#### `ktc_vis/metrics/shape_matching.py`
- `compute_hausdorff(pred, gt)` → float
- `compute_position_error(pred, gt)` → float (centroid offset in pixels)
- `compute_resolution(pred, gt)` → float (smallest detected inclusion diameter)

#### Review responsibility:
- Review all `ktc_vis/adapters/cuqi8_adapter.py` and `pnpe2e_adapter.py` for physics correctness.
- Review Module 4 radar axes definitions and Module 6 measurement panel physical plausibility.

---

### 3.2 Smit Savani (1420825) — Metrics & Backend Engineer

**Primary deliverables:**

#### `ktc_vis/adapters/cuqi8_adapter.py`
- Wraps CUQI8 repository including FEniCS/dolfinx dependency.
- Handles per-level regularization weight adaptation.
- On Windows: must work inside WSL2; Dockerfile must install dolfinx correctly.

#### `ktc_vis/adapters/pnpe2e_adapter.py`
- Wraps PNPE2E including DeepInverse + torch-geometric.
- Must load per-level weight files correctly.

#### `ktc_vis/metrics/engine.py`
- Class `MetricsEngine` with method `compute_all(algorithm, level, sample)` → dict of 14 metrics.
- Runs all metrics, stores results to HDF5 cache, returns from cache on second call.

#### `ktc_vis/metrics/class_metrics.py`
- `compute_per_class_iou(pred, gt)` → dict with keys `water`, `resistive`, `conductive`
- `compute_confusion_matrix(pred, gt)` → 3×3 ndarray (normalized, as percentages)

#### `ktc_vis/metrics/measurement.py`
- `compute_voltage_residual(measurement, adapter)` → float — predicted vs. measured voltages using forward model.
- `compute_resistance_consistency(measurement, pred)` → float

#### `ktc_vis/metrics/efficiency.py`
- `measure_runtime(adapter, measurement)` → float (seconds)

#### `ktc_vis/cache/hdf5_store.py`
- `save_result(algorithm, level, sample, metrics_dict, reconstruction)` → writes to `data/cache/results.h5`
- `load_result(algorithm, level, sample)` → reads from cache; raises `CacheMiss` if not found.

#### `tests/test_metrics.py`
- Test each metric against reference values (SSIM from KTC2023 paper).
- Test that cache saves and loads correctly.

#### `scripts/run_benchmark.py`
- CLI tool: iterates all 3 algorithms × 7 levels × 3 samples; calls MetricsEngine; populates HDF5.
- Progress bar; skip if already cached.

---

### 3.3 Asmita Bhuva (1541650) — Viz & Frontend Developer

**Primary deliverables:**

#### `ktc_vis/dashboard/layout.py`
- Top-level `app.layout` with:
  - Shared sidebar (algorithm selector, level slider 1–7, sample radio).
  - Tab navigation for 6 modules.
  - Sidebar state is preserved when switching modules.

#### `ktc_vis/dashboard/components/sidebar.py`
- Reusable Dash components: `AlgorithmDropdown`, `LevelSlider`, `SampleRadio`.
- All IDs must be unique and prefixed (e.g., `sidebar-algorithm-dropdown`).

#### `ktc_vis/dashboard/modules/m1_reconstruction_explorer.py`
- 4-panel layout: Ground Truth | Reconstruction | Segmentation | Error Overlay.
- Shared callback on (algorithm, level, sample) → updates all 4 panels simultaneously.
- Error overlay: green = correct, red = false positive, blue = false negative.

#### `ktc_vis/dashboard/modules/m2_difficulty_animator.py`
- Level 1–7 slider with Play/Pause animation.
- Reconstruction animation panels + degradation curves (SSIM, IoU, Hausdorff) as Plotly line charts.
- Data from HDF5 cache.

#### `ktc_vis/dashboard/modules/m3_comparison_grid.py`
- 3-column grid: ABC1 | CUQI8 | PNPE2E at the same (level, sample).
- Pairwise difference images (pixel-wise subtraction).
- Score table at the bottom (SSIM per algorithm).

#### `ktc_vis/dashboard/modules/m4_fingerprint_radar.py`
- 9-axis radar chart with algorithm toggle checkboxes.
- Voltage residual axis is inverted (lower value → better → shown as higher on chart).
- Score summary table below the radar.

#### `ktc_vis/dashboard/modules/m5_failure_autopsy.py`
- Ranked list of worst-performing cases by SSIM (collapsible rows).
- Clicking a row opens 4 diagnostic panels: spatial SSIM heatmap, 3×3 confusion matrix, boundary radial plot, measurement residual polar plot.
- Failure type badge (A/B/C/D/E) automatically assigned based on diagnostic signals.

#### `ktc_vis/dashboard/modules/m6_measurement_viewer.py`
- 3 sub-panels: Current (polar bar), Voltage (polar + difference), Resistance (scatter + 2D heatmap).
- All panels update on (level, sample) selection.

#### `tests/test_modules/`
- For each module: test that callbacks return valid Plotly figure objects.
- Test edge cases: level=1, level=7, all 3 algorithms toggled.

---

### 3.4 Shimul Paul (1441927) — Integration & DevOps Lead

**Primary deliverables:**

#### `environment.yml`
- Conda environment with pinned versions for: Python 3.10, Dash, Plotly, PyTorch, FEniCS/dolfinx, DeepInverse, torch-geometric, SciPy, h5py, NumPy, scikit-image, scikit-learn, PyYAML, pytest, Matplotlib.
- Must install on Linux (Docker base) and macOS (dev machines).

#### `Dockerfile`
- Base image: `dolfinx/dolfinx:v0.7.0` (for FEniCS compatibility).
- Install Conda env; copy source; expose port 8050.
- `CMD ["python", "app.py"]`

#### `.github/workflows/ci.yml`
- Trigger: push and pull_request on `main`.
- Steps: checkout → setup Conda → install env → run `pytest tests/` → report coverage.
- Must not run FEniCS-heavy tests in CI (use mock adapter for CUQI8).

#### `configs/experiment.yaml`
- Defines: data path, cache path, algorithms to run, levels to run, samples to run, metric flags.
- Documented inline with comments.

#### `app.py`
- Initializes Dash app; imports all 6 module layouts; registers all callbacks; sets up server.
- Startup banner prints loaded algorithms and cache status.

#### `docs/adding_new_algorithm.md`
- Step-by-step guide: define an `AlgorithmAdapter` subclass → register in `configs/experiment.yaml` → run benchmark → view in dashboard.

#### `docs/interpreting_dashboard.md`
- User guide explaining each module, each metric, and the failure taxonomy.

#### Review responsibility:
- All PRs go through Shimul for final merge approval.
- Ensures Docker build succeeds end-to-end before any merge to `main`.

---

## 4. Coding Standards

### Python
- **Version:** Python 3.10 (pinned in `environment.yml`)
- **Style:** PEP 8; max line length 100; enforced via `flake8` in CI.
- **Type hints:** Required on all public functions and class methods.
- **Docstrings:** Google-style docstrings on all public classes and methods.
- **No magic numbers:** All constants in `configs/experiment.yaml` or a `constants.py` module.

### Dash/Plotly
- Every Dash component ID must be **unique** and **descriptive** (use module prefix, e.g., `m1-gt-image`).
- All callbacks must have explicit `Output`, `Input`, `State` — no dynamic callback registration.
- Figures must set `uirevision` to preserve zoom state across updates.

### Testing
- Every public function must have at least one unit test.
- Integration tests must use shared fixtures from `conftest.py` (pre-loaded data, mock cache).
- CI must pass before any PR is merged.
- Coverage target: ≥ 80% on `ktc_vis/` (excluding `dashboard/modules/`).

### Git Workflow
- Branch naming: `feature/<name>`, `fix/<name>`, `docs/<name>`
- Commit messages: `[scope] short description` (e.g., `[loader] add voltage matrix parsing`)
- No direct commits to `main` — all changes via Pull Request.
- PR must include: what changed, how to test it, screenshots for UI changes.

---

## 5. Acceptance Criteria (Definition of Done)

### Data Layer
- [ ] `KTCDataLoader.load(level, sample)` returns correct matrix shapes for all 7 levels and 3 samples.
- [ ] Electrode subsampling matches KTC2023 paper Table 1 exactly.

### Adapters
- [ ] All 3 adapters return a 256×256 uint8 array with values only in {0, 1, 2}.
- [ ] ABC1 SSIM score matches published KTC2023 score within ±0.01.
- [ ] All 3 adapters run inside Docker without code changes.

### Metrics Engine
- [ ] SSIM implementation matches published values from KTC2023 organiser scores.
- [ ] All 14 metrics computed and cached correctly for all 3 × 7 × 3 = 63 combinations.
- [ ] Second call to `MetricsEngine.compute_all()` reads from cache (no recomputation).

### Dashboard Modules
- [ ] All 6 modules render without errors for any valid (algorithm, level, sample) combination.
- [ ] Module 4 radar chart normalizes all axes to [0, 1]; voltage residual axis inverted.
- [ ] Module 5 failure type badges (A–E) are automatically assigned.
- [ ] Sidebar state (algorithm, level, sample) is preserved when switching between modules.

### Infrastructure
- [ ] `docker build` completes without errors.
- [ ] `docker run -p 8050:8050 ktc-vis` launches dashboard accessible in browser.
- [ ] GitHub Actions CI runs `pytest` and reports pass/fail on every PR.
- [ ] Full benchmark (all 63 combinations) runs end-to-end and populates HDF5 cache.

---

## 6. Weekly Checkpoints

| Week | Muzammal | Smit | Asmita | Shimul |
|------|----------|------|--------|--------|
| W1–2 | Read KTC2023 paper; inspect .mat files | Audit CUQI8 + PNPE2E repos | Set up Dash project skeleton | Init Git repo; create environment.yml skeleton |
| W3–4 | Implement `loader.py`; unit tests | Implement CUQI8 adapter | Design sidebar component | Write Dockerfile base; set up CI |
| W5–6 | ABC1 adapter + metrics foundation | Full metrics engine; PNPE2E adapter | M1 Reconstruction Explorer | HDF5 cache integration; CI metrics tests |
| W7 | Validate all 3 algorithms on all levels | Cache all metrics | M2 Difficulty Animator | run_benchmark.py script |
| W8–9 | Backend for M4, M5, M6 panels | Backend for M3, M4, M5 | M3, M4, M5, M6 visual design | Docker final build; CI full suite |
| W10 | Validate SSIM vs published values | Reproducibility check on second machine | UI bug fixes; animation polish | Documentation: adding_new_algorithm.md |
| W11–12 | Final report contribution (algorithms section) | Final report contribution (metrics section) | Final report contribution (dashboard section) | Final report + demo prep + handover archive |

---

## 7. Key Dependencies & Gotchas

### FEniCS (for CUQI8)
- **Must use dolfinx ≥ 0.7** — incompatible API in older versions.
- On Windows dev machines: **use WSL2**. Docker image handles this for CI/deployment.
- Do not install via pip — use `conda install -c conda-forge fenics-dolfinx`.

### DeepInverse + torch-geometric (for PNPE2E)
- Install order matters: PyTorch → torch-geometric → DeepInverse.
- Pin `torch==2.0.1` to match PNPE2E's original environment.
- Per-level weights must be downloaded from the PNPE2E repo and placed in `data/raw/weights/pnpe2e/`.

### KTC Dataset
- Use **Zenodo v3** (v1 and v2 have a ground-truth cropping bug).
- `.mat` files use MATLAB v7.3 format — use `scipy.io.loadmat` with `mat73` fallback.
- File naming convention: `level{L}_target{T:02d}.mat` (e.g., `level1_target01.mat`).

### HDF5 Cache
- Group structure: `/results/{algorithm}/{level}/{sample}/` containing datasets: `reconstruction`, `ssim`, `hausdorff`, `iou_water`, `iou_resistive`, `iou_conductive`, `voltage_residual`, `runtime`, etc.
- Always open with `h5py.File(..., "a")` (append mode) to avoid overwriting existing results.

---

## 8. Contact & Review

| Topic | Contact |
|-------|---------|
| Data loading / physics questions | Muzammal |
| Metrics / algorithm correctness | Smit Savani |
| Dashboard layout / UI design | Asmita Bhuva |
| Docker / CI / environment issues | Shimul Paul |
| PR merge approval | Shimul Paul |
