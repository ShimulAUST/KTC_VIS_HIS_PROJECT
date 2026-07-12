# KTC-Vis: Interactive EIT Algorithm Benchmarking Dashboard

> **Kuopio Tomography Challenge 2023 — Unified Visualization Framework**
> **Course:** Project HIS
> **Institution:** Frankfurt University of Applied Sciences, Germany
> **Guided By:** Prof. Dr. Martin Simon & Emanuele Pepe

## What is KTC-Vis?

Electrical Impedance Tomography (EIT) is a non-invasive imaging technique that estimates the internal conductivity distribution of an object by injecting small currents through surface electrodes and measuring the resulting voltages at other electrodes. It is radiation-free, low-cost, and has potential applications in medical imaging, industrial process monitoring, and geophysical sensing — but producing reliable reconstructions from the measured data remains an active research problem.

The Kuopio Tomography Challenge 2023 (KTC2023) is an international EIT benchmarking competition that released a standardised open dataset of phantom tank measurements across seven difficulty levels, ranging from 32 electrodes (full measurement coverage) down to 20 (heavily subsampled). Several research groups submitted open-source reconstruction algorithms as part of the competition, making KTC2023 one of the few publicly available resources for comparing EIT methods on identical data.

KTC-Vis was developed as part of the Project HIS course at Frankfurt University of Applied Sciences. The goal was to build an interactive dashboard that loads three of those submitted algorithms — ABC1, CUQI8, and PNPE2E — and allows researchers and students to compare their outputs directly, inspect the underlying measurement data, and understand where and why each algorithm performs well or poorly. The dashboard covers all seven difficulty levels, four phantom samples, and 18 computed metrics per combination, without requiring any code to be written by the user.

## Team Members

| Name | Student ID | Role | Modules |
|------|------------|------|---------|
| Muhammad Muzammal | 1541353 | EIT & Data Specialist / Backend Engineer | M1, M5 |
| Smit Savani | 1420825 | Metrics & Backend Engineer | M3, M6 |
| Asmita Bhuva | 1541650 | Viz & Frontend Developer | M4, M5 |
| Shimul Paul | 1441927 | Integration & DevOps Lead | M2, M6 |

## Motivation

When we began this project, no existing open-source tool allowed all three KTC2023 algorithms to be run on the same data and compared interactively. The closest alternatives were pyEIT, which provides classical EIT reconstruction methods (back-projection, Gauss-Newton, FEM mesh tooling) but has no benchmark comparison mode and no support for the KTC2023 dataset format, and OpenEIT, which is oriented toward real-time imaging from live hardware and does not support offline dataset analysis at all. The individual algorithm repositories from the competition each work in isolation — there is no shared metrics framework, no unified coordinate space, and no way to place two outputs side-by-side without writing custom code.

This left a practical gap for anyone trying to study the relative behaviour of these methods: which algorithm handles sparse electrode configurations better? How does reconstruction quality degrade as measurement data is removed level by level? When an algorithm fails, is it a data problem or a model problem? These questions require comparing outputs across algorithms, difficulty levels, and raw measurement signals simultaneously.

KTC-Vis addresses this by treating all three algorithms as interchangeable components behind a common adapter interface and routing their outputs through a shared metrics engine and HDF5 cache. The dashboard then exposes the results through six purpose-built modules. A few concrete capabilities this enables that were not available before:

- All three algorithms can be compared on identical (level, sample) inputs in the same coordinate space, with pixel-level difference maps showing where they disagree.
- The raw current/voltage/resistance signals an algorithm received can be inspected alongside the reconstruction it produced, in the same session.
- Bad reconstructions are automatically classified by failure type (ghost inclusion, missing inclusion, class flip, boundary erosion, mask suppression) rather than just flagged by a low SSIM score.
- Reconstruction quality across all seven difficulty levels can be animated directly, making the degradation pattern observable rather than inferred from a summary table.
- All metrics are stored in an HDF5 cache keyed by (algorithm, level, sample), so every chart in the dashboard is reproducible from the same cache file.

## Algorithm Comparison

| Criterion | ABC1 [3] | CUQI8 [4] | PNPE2E [5] |
|-----------|----------|------------|-------------|
| **Approach** | Smoothness prior + CNN repair | Level-set + TV forward model | E2E U-Net + PnP Graph U-Net |
| **Learning enters** | After physics reconstruction | Never — pure optimization | Before & during refinement |
| **Neural network** | CNN postprocessor | None | 4-scale U-Net + Graph U-Net |
| **Uses forward model** | Linearised Jacobian only | Full CEM at every iteration | Via PnP Gauss-Newton step |
| **Pretrained weights** | CNN weights on Zenodo | None needed | Separate weights per level |
| **FEniCS needed** | No | Yes (runs in Docker/WSL2) | No |
| **Strongest at levels** | 1–3 (fast, artefact-clean) | 5–7 (stable, geometry-aware) | All (difficulty-adaptive) |
| **Expected failure mode** | Ghost inclusions / boundary erosion | Missing inclusion (wrong region count) | Class flip / mask suppression |
| **Runtime** | Fast | Slow (FEM per iteration) | Medium |

## The KTC2023 Dataset

The dataset uses a shallow circular water tank (23 cm diameter) with 32 stainless-steel electrodes and plastic or metal cylindrical inclusions. The reconstruction task is to produce a 256×256-pixel segmented image with three tissue classes: water, resistive inclusion, and conductive inclusion. The seven difficulty levels are created by progressively removing electrodes from the measurement protocol:

| Level | Electrodes | Measurements |
|-------|------------|--------------|
| 1 | 32 | 2356 |
| 2 | 30 | 1624 |
| 3 | 28 | 1404 |
| 4 | 26 | 1200 |
| 5 | 24 | 1012 |
| 6 | 22 | 630 |
| 7 | 20 | 513 |

The dataset is available on Zenodo (v3 recommended) at doi: 10.5281/zenodo.10986692.

## Dashboard: 6 Modules

### Module 1 — Reconstruction Explorer
> **Owner: Muhammad Muzammal**

Provides a single-case view of any (algorithm, level, sample) combination. The layout shows the ground truth phantom alongside the algorithm's raw reconstruction output, its thresholded 3-class segmentation, and a pixel-level error overlay (green = correct, red = false positive, blue = false negative, yellow = wrong class). A chip row above the panels reports the electrode count, injection count, voltage sample count, and overall pixel agreement for the selected case.

### Module 2 — Difficulty Animator
> **Owner: Shimul Paul**

A level slider (1–7) with play/pause animation (1.2 s per level) that steps through all difficulty levels for the selected algorithm and sample. Three image panels — ground truth, segmentation, and error overlay — update at each level, making the degradation in reconstruction quality directly visible. Seven metric curves (SSIM, mean IoU, Dice, Hausdorff distance, position error, resolution, runtime) plot quality against difficulty level, with a marker tracking whichever level is currently displayed.

### Module 3 — Side-by-Side Comparison Grid
> **Owner: Smit Savani**

Shows all three algorithms for the same (level, sample) simultaneously in a three-row layout. The first row contains the three reconstruction heatmaps; the second shows pixel-wise class-index difference images for each algorithm pair (ABC1−CUQI8, ABC1−PNPE2E, CUQI8−PNPE2E), with a diverging colorscale where the dark midpoint indicates pixel-level agreement. The third row contains the raw voltage measurement chart and a full 18-metric scorecard for all three algorithms.

### Module 4 — Fingerprint Radar
> **Owner: Asmita Bhuva**

A 9-axis radar chart comparing the three algorithms on normalised performance profiles. Axes cover Total SSIM, Easy SSIM (levels 1–3), Hard SSIM (levels 5–7), IoU Conductive, IoU Resistive, Speed, Robustness, Voltage Residual, and Current Sensitivity Balance. All axes are min-max normalised to [0, 1] across the three algorithms; axes where lower values are better (Speed, Robustness, Voltage Residual) are inverted so that a larger polygon always indicates better overall performance. A numerical summary table accompanies the chart.

| Axis | Domain |
|------|--------|
| Total SSIM | Image |
| Easy SSIM (levels 1–3) | Image |
| Hard SSIM (levels 5–7) | Image |
| IoU Conductive | Image |
| IoU Resistive | Image |
| Speed | Image |
| Robustness | Image |
| Voltage Residual (inverted) | Measurement |
| Current Sensitivity Balance | Measurement |

### Module 5 — Failure Autopsy
> **Owners: Muhammad Muzammal & Asmita Bhuva**

Lists all cached (algorithm, level, sample) combinations ranked by ascending SSIM, filterable by algorithm and top-N count. Selecting a row loads four diagnostic panels: a spatial SSIM heatmap showing which image regions drove the score down, a 3×3 confusion matrix showing class-level misclassification, a boundary-error polar histogram showing whether errors concentrate near specific electrode positions, and a per-injection measurement perturbation plot showing whether the raw signal was weak to begin with. A failure-type badge is automatically assigned based on these signals.

| Type | Label | Likely Algorithm | Diagnostic Signal |
|------|-------|-----------------|-------------------|
| A | Ghost inclusion | ABC1 | False positive; IoU near zero |
| B | Missing inclusion | CUQI8 | Wrong region count; recall ≈ 0 |
| C | Class flip | PNPE2E | Conductivity sign wrong; off-diagonal confusion high |
| D | Boundary erosion | ABC1 | Hausdorff high; IoU moderate |
| E | Mask suppression | PNPE2E | Recall near zero |

### Module 6 — Measurement Domain Viewer
> **Owners: Smit Savani & Shimul Paul**

Explores the raw electrical measurements independently of any reconstruction — the algorithm selector has no effect here, since all three algorithms receive identical inputs for a given (level, sample) combination. An injection-step slider with play animation (0.9 s per step) steps through each injection pattern, updating a current polar chart (showing which electrode pair drives current), a voltage polar overlay (measured voltage against the empty-tank reference), a ΔV bar chart (the actual anomaly signal the algorithms receive), and an anomaly score bar (Σ|ΔR| per injection). A parallel set of static panels — resistance per pair, all-injections overlay, mean ± std summary, ΔR difference, SNR per pair, coefficient of variation, electrode contact quality estimate, and a cross-level coverage chart — give aggregate views that do not change with the injection step. A polar coverage map shows active versus removed electrodes and sampled measurement-pair connections, updated each injection step during playback. A CSV export button is available for the selected injection's R, V, and I matrices. All panels carry a brief plain-language note describing what to look for in typical and problematic cases.

## Metrics

For each (algorithm, level, sample) combination, 18 metrics are computed and stored in the HDF5 cache:

| Category | Metric | What It Measures |
|----------|--------|-----------------|
| Image Quality | SSIM Score | Overall structural similarity to ground truth |
| Image Quality | Spatial SSIM Map | Pixel-level quality distribution |
| Shape Matching | Resolution | Smallest detectable inclusion diameter |
| Shape Matching | Hausdorff Distance | Worst-case boundary error |
| Shape Matching | Position Error | Centroid offset between predicted and true inclusion |
| Class Specific | Confusion Matrix | Class-level misclassification rates |
| Class Specific | Per-class IoU | Detection accuracy per material type |
| Data Efficiency | Runtime | Wall-clock reconstruction time |
| Data Efficiency | Position Error | Localisation accuracy across levels |
| Measurement | Voltage Residual | RMS deviation of measured vs. reference voltages |
| Measurement | Resistance Consistency | Homogeneity of the R=V/I map |

## System Architecture

```
KTC .mat files (Zenodo)
        │
        ▼
┌─────────────────┐
│   Data Layer    │  scipy.io.loadmat → current/voltage/resistance matrices
│  (HDF5 cache)   │  Organized by level (1–7) and sample (A/B/C)
└────────┬────────┘
         │
         ▼
┌─────────────────────┐
│  Method Adapter API │  Uniform interface → 256×256 segmentation output
│  ABC1 | CUQI8 |     │  Per-algorithm wrappers with dependency management
│  PNPE2E             │
└────────┬────────────┘
         │
         ▼
┌─────────────────┐
│  Metrics Engine │  18 metrics × 3 algorithms × 7 levels × 4 samples
│  (HDF5 cache)   │
└────────┬────────┘
         │
         ▼
┌────────────────────────────────────────┐
│  Plotly Dash Dashboard (6 Modules)     │
│  M1 | M2 | M3 | M4 | M5 | M6         │
└────────────────────────────────────────┘
```

## Technology Stack

| Layer | Tool | Purpose |
|-------|------|---------|
| Dashboard | Plotly Dash | All 6 modules; layout & callbacks |
| Visualization | Plotly | Radar, polar, SSIM surface, image panels |
| Static exports | Matplotlib | Publication-quality PNG figures |
| EIT Baseline | pyEIT | Back-projection, Gauss-Newton, FEM mesh |
| Algorithm | PyTorch | CNN inference (ABC1), Graph U-Net (PNPE2E) |
| Algorithm | torch-geometric | Graph U-Net denoiser for PNPE2E |
| Algorithm | DeepInverse | Plug-and-play solver for PNPE2E |
| Algorithm | FEniCS/dolfinx | FEM forward model for CUQI8 |
| Data I/O | SciPy | Read KTC `.mat` files |
| Data I/O | h5py | HDF5 results cache |
| Numerics | NumPy | Array operations, R=V/I computation |
| Metrics | scikit-image | SSIM (spatial), Hausdorff distance |
| Metrics | scikit-learn | Confusion matrix, per-class F1 |
| Config | PyYAML | Experiment configuration files |
| Environment | Conda | Full environment including non-pip deps |
| Container | Docker | Reproducible cross-platform packaging |
| Testing | pytest | Unit + integration tests |
| CI | GitHub Actions | Auto-run tests on every push |
| Version Control | Git + GitHub | Source, config, docs |

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/ShimulAUST/KTC_VIS_HIS_PROJECT.git
cd KTC_VIS_HIS_PROJECT

# 2. Set up the Conda environment
conda env create -f environment.yml
conda activate KTC_VIS_HIS_PROJECT

# 3. Download the KTC2023 dataset (Zenodo v3)
#    Place .mat files under data/raw/ktc2023/

# 4. Populate the HDF5 cache (choose one):
#    Fast — use pre-computed reference outputs (no Docker needed):
python scripts/populate_cache_from_reference.py
#    Full — run live algorithms via Docker (takes hours):
python scripts/run_benchmark.py --config configs/experiment.yaml

# 5. Launch the dashboard
python app.py
# Open http://localhost:8050 in your browser
```

## 12-Week Timeline

| Week | Phase | Key Deliverables |
|------|-------|-----------------|
| W1–2 | Study & Setup | Repo skeleton, acceptance criteria, dependency audit |
| W3–4 | Core Backend | Data loader (current/voltage/resistance), adapter API, ABC1 integrated & validated |
| W5–6 | Algorithm Integration | CUQI8 (FEM), PNPE2E (DeepInverse + per-level weights), full metrics engine, HDF5 cache |
| W7 | Foundation Modules | M1 (Reconstruction Explorer) + M2 (Difficulty Animator) functional |
| W8–9 | Advanced Modules | M3 (Comparison Grid), M4 (Fingerprint Radar), M5 (Failure Autopsy), M6 (Measurement Viewer) |
| W10 | Validation | Full benchmark: 21 targets × 3 algorithms; reproducibility check; codebase cleanup |
| W11–12 | Report & Demo | Final report + live demo across all 6 modules |

## What We Have Achieved

Over the course of this project, all six dashboard modules were implemented and made functional against pre-computed reference outputs. The data layer handles all seven difficulty levels through runtime subsampling of the Level 1 KTC2023 measurement files, deriving the reduced electrode and injection configurations for Levels 2–7 according to the competition protocol. Both data paths are operational: the `ReferenceOutputAdapter` allows the dashboard to be populated immediately from staged reference outputs without requiring Docker, while the three live adapters (`ABC1Adapter`, `CUQI8Adapter`, `PNPE2EAdapter`) wrap their respective Docker images and support level-batched execution where the algorithm allows it.

The metrics engine computes 18 keys per (algorithm, level, sample) combination — covering image quality, shape matching, class-specific accuracy, measurement domain diagnostics, and runtime — and writes results to a shared HDF5 cache that all six modules read from. The failure autopsy in Module 5 includes an automatic classifier that assigns one of five failure types (ghost inclusion, missing inclusion, class flip, boundary erosion, mask suppression) based on the confusion matrix, boundary error distribution, and measurement perturbation signals. Module 6 contains 13 independent visualisation panels covering every stage of the measurement pipeline from raw current injection through to electrode contact quality and cross-level coverage. A continuous integration pipeline runs flake8 linting and pytest on every push via GitHub Actions, and the full application can be deployed in a Docker container with docker-compose.

## Limitations

The most significant gap in the current implementation is the test suite. Aside from `test_glossary.py`, which verifies that all required terms are present in the shared glossary component, every other test file contains only a placeholder assertion. This means the CI pipeline reports a passing build without actually validating any data loading, metric computation, adapter output, or callback behaviour. The test scaffolding is in place — pytest, pytest-cov, fixture structure, and a `requires_data` marker — but the assertions themselves have not been written.

On the deployment side, live algorithm execution requires Docker and the KTC2023 dataset, neither of which is bundled with the repository. CUQI8 additionally depends on FEniCS/dolfinx, which is not installed in the main Conda environment and runs only inside a separate Docker image invoked via the host socket. As a result, the dashboard was developed and validated against pre-computed reference outputs rather than live Docker reconstructions, and timing values in the cache may reflect reference-read latency rather than true reconstruction runtimes unless the runtime benchmarking scripts (`benchmark_runtime_abc1.py`, `benchmark_runtime_pnpe2e.py`) have been run separately. The data pipeline also derives Levels 2–7 at runtime by subsampling the single Level 1 `.mat` file; while this matches the KTC2023 paper's specification, it means all difficulty levels share the same underlying phantom measurement rather than independently acquired data.

## Future Work

The most immediate priority is completing the test suite with real assertions: `test_loader.py` and `test_cache.py` should validate load and round-trip behaviour on small synthetic `.mat` files; `test_metrics.py` should check known metric values for a controlled input pair; and the `test_modules/test_m*.py` files should verify that each module's Dash callbacks return valid figure objects for typical and edge-case inputs (missing cache, level 7, all three algorithms). Adding the dataset to a CI fixture or using synthetic data would make this possible without requiring the full Zenodo download in the CI environment.

Beyond testing, several extensions follow naturally from the current architecture. The adapter interface is already defined as an abstract base class, so adding support for new EIT algorithms would require writing one new adapter file and registering it — a workflow that is documented but not yet demonstrated with a fourth algorithm. Real-time measurement stream support, which OpenEIT provides for live hardware but KTC-Vis does not, could be added as a streaming data path without changing the dashboard layout. The current results are stored locally in HDF5; exposing them as a lightweight API or generating a structured benchmark report (PDF or LaTeX table) would make the outputs easier to cite in a paper. Finally, the dashboard is currently deployed on localhost; making it accessible via a public URL with authentication would broaden its use as a shared research tool.

## References

1. Räsänen et al., "KTC2023 — EIT Competition and Open Dataset," *Applied Mathematics for Modern Challenges*, 2024. doi:10.3934/ammc.2024009
2. KTC2023 Dataset on Zenodo v3. doi:10.5281/zenodo.10986692
3. Beraldo et al. (ABC1), KTC2023 submission, UFABC. https://github.com/robert-abc/KTC2023-ABC1
4. Carøe et al. (CUQI8), KTC2023 submission, DTU. https://github.com/CUQI-DTU/KTC2023-CUQI8
5. Santacesaria et al. (PNPE2E), KTC2023 submission. https://github.com/msantacesaria/KTC2023_PNPE2E
6. Tachella et al., DeepInverse. doi:10.5281/zenodo.7982256
7. Liu et al., pyEIT. *SoftwareX*, 2018. doi:10.1016/j.softx.2018.09.005
8. Wang et al., SSIM. *IEEE TIP*, 2004. doi:10.1109/TIP.2003.819861
