# KTC-Vis: Interactive EIT Algorithm Benchmarking Dashboard

> **Kuopio Tomography Challenge 2023 — Unified Visualization Framework**
> **Course:** Project HIS
> **Institution:** Frankfurt University of Applied Sciences, Germany
> **Guided By:** Prof. Dr. Martin Simon & Emanuele Pepe

KTC-Vis integrates three open-source EIT reconstruction algorithms from KTC2023 — **ABC1**, **CUQI8**, and **PNPE2E** — into a single, reproducible, interactive Plotly Dash dashboard. It connects the physical measurement domain (currents, voltages, resistance) with the image-domain reconstruction quality across all 7 official KTC difficulty levels.

---

## Team Members

| Name | Student ID | Role | Modules |
|------|------------|------|---------|
| Muhammad Muzammal | 1541353 | EIT & Data Specialist / Backend Engineer | M1, M5 |
| Smit Savani | 1420825 | Metrics & Backend Engineer | M3, M6 |
| Asmita Bhuva | 1541650 | Viz & Frontend Developer | M4, M5 |
| Shimul Paul | 1441927 | Integration & DevOps Lead | M2, M6 |

---

## Project Goal

There is no open tool that loads all KTC2023 algorithms, runs them on the same data, and interactively compares them side-by-side across all difficulty levels. While existing tools like **pyEIT** focus on classical EIT workflows and **OpenEIT** emphasizes live hardware streams, KTC-Vis fills a crucial gap for offline benchmark comparisons by providing:

1. **Reproducibility** — All experiments are defined in a YAML config and cached in HDF5; any run is exactly repeatable.
2. **Extensibility** — Add a new algorithm by defining one adapter function; add a measurement subset by adding one config line.
3. **Multi-dimensional evaluation** — 14 metrics per (algorithm × level × sample) covering image quality, shape matching, class-specific accuracy, measurement fit, and data efficiency.
4. **Visual explainability** — Every metric is linked to a visual (e.g., low SSIM → spatial quality map; high voltage residual → measurement error plot).
5. **Measurement-domain visibility** — Raw current sequences, voltage responses, and resistance estimates are all explorable.

---

## Algorithm Comparison at a Glance

| Criterion | ABC1 [9] | CUQI8 [10] | PNPE2E [12] |
|-----------|----------|------------|-------------|
| **Approach** | Smoothness prior + CNN repair | Level-set + TV forward model | E2E U-Net + PnP Graph U-Net |
| **Learning enters** | After physics reconstruction | Never — pure optimization | Before & during refinement |
| **Neural network** | CNN postprocessor | None | 4-scale U-Net + Graph U-Net |
| **Uses forward model**| Linearised Jacobian only | Full CEM at every iteration | Via PnP Gauss-Newton step |
| **Pretrained weights**| CNN weights on Zenodo | None needed | Separate weights per level |
| **FEniCS needed** | No | Yes (runs in Docker/WSL2) | No |
| **Strongest at levels**| 1–3 (fast, artefact-clean) | 5–7 (stable, geometry-aware) | All (difficulty-adaptive) |
| **Expected failure mode**| Ghost inclusions / boundary erosion | Missing inclusion (wrong region count) | Class flip / mask suppression |
| **Runtime** | Fast | Slow (FEM per iteration) | Medium |

---

## The KTC2023 Dataset

- **Setup:** Shallow circular water tank (23 cm diameter), 32 stainless-steel electrodes, plastic/metal cylindrical inclusions.
- **Task:** Reconstruct a 256×256-pixel segmented image (water / resistive inclusion / conductive inclusion).
- **7 Difficulty Levels:**

| Level | Electrodes | Measurements |
|-------|------------|--------------|
| 1 | 32 | 2356 |
| 2 | 30 | 1624 |
| 3 | 28 | 1404 |
| 4 | 26 | 1200 |
| 5 | 24 | 1012 |
| 6 | 22 | 630 |
| 7 | 20 | 513 |

- **Data:** Available on Zenodo (v3 recommended) — doi: 10.5281/zenodo.10986692

---

## Dashboard: 6 Modules

### Module 1 — Reconstruction Explorer
> **Owner: Muhammad Muzammal**

Inspect any (algorithm, level, sample) combination across 4 side-by-side panels:
- Ground Truth Segmentation
- Reconstructed Conductivity / Heatmap
- Predicted 3-Class Segmentation
- Error Overlay

### Module 2 — Difficulty Animator
> **Owner: Shimul Paul**

Level 1–7 slider animates how each algorithm degrades. Shows degradation curves for SSIM, IoU, Dice, Hausdorff distance, and runtime.

### Module 3 — Side-by-Side Comparison Grid
> **Owner: Smit Savani**

Three-column grid showing all algorithms at the same level/sample simultaneously. Includes pairwise difference images and voltage residual chart.

### Module 4 — Fingerprint Radar
> **Owner: Asmita Bhuva**

9-axis radar chart comparing algorithm performance profiles:

| Axis | Domain |
|------|--------|
| Total SSIM | Image |
| Easy SSIM (levels 1–3) | Image |
| Hard SSIM (levels 5–7) | Image |
| IoU Conductive | Image |
| IoU Resistive | Image |
| Speed | Image |
| Robustness | Image |
| Voltage Residual (inverted — lower is better) | Measurement |
| Current Sensitivity Balance | Measurement |

Expected shapes: CUQI8 → narrow/tall; ABC1 → wide/low; PNPE2E → balanced/intermediate.

### Module 5 — Failure Autopsy
> **Owners: Muhammad Muzammal & Asmita Bhuva**

Ranked worst-case list by SSIM. Click any case to open 4 diagnostic panels:
- Spatial SSIM heatmap
- 3×3 material confusion matrix
- Boundary-error radial plot
- Measurement residual polar plot

**Failure taxonomy (Types A–F):**

| Type | Label | Likely Algorithm | Diagnostic Signal |
|------|-------|-----------------|-------------------|
| A | Ghost inclusion | ABC1 | False positive; IoU near zero |
| B | Missing inclusion | CUQI8 | Wrong region count; recall ≈ 0 |
| C | Class flip | PNPE2E | Conductivity sign wrong; off-diagonal confusion high |
| D | Boundary erosion | ABC1 | Hausdorff high; IoU moderate |
| E | Mask suppression | PNPE2E | Recall near zero |

### Module 6 — Measurement Domain Viewer
> **Owners: Smit Savani & Shimul Paul**

Explore raw electrical measurements independent of reconstruction:
- **Current panel:** Polar plot of injected current patterns (electrodes 1–32, magnitude = bar height).
- **Voltage panel:** Polar measured voltages + difference plot (inclusion vs. empty tank).
- **Resistance panel:** R=V/I at one electrode pair + 2D resistance map (rows=injection patterns, columns=measurement pairs).

---

## 14 Metrics Engine

| Category | Metric | What It Measures |
|----------|--------|-----------------|
| Image Quality | SSIM Score | Overall image quality |
| Image Quality | Spatial SSIM Map | Pixel-by-pixel quality heatmap |
| Shape Matching | Resolution | Smallest detectable inclusion |
| Shape Matching | Hausdorff Distance | Worst-case boundary error |
| Shape Matching | Position Error | Inclusion center offset |
| Class Specific | Confusion Matrix | Mislabeling between materials |
| Class Specific | Per-class IoU | Detection quality per material |
| Data Efficiency | Runtime | Algorithm speed |
| Data Efficiency | Position Error | Localization accuracy |
| Measurement | Voltage Residual | Measured vs. predicted voltages |
| Measurement | Resistance Consistency | R=V/I map vs. ground truth |

---

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
│  Metrics Engine │  14 metrics × 3 algorithms × 7 levels × 4 samples
│  (HDF5 cache)   │
└────────┬────────┘
         │
         ▼
┌────────────────────────────────────────┐
│  Plotly Dash Dashboard (6 Modules)     │
│  M1 | M2 | M3 | M4 | M5 | M6         │
└────────────────────────────────────────┘
```

---

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

---

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

---

## 12-Week Timeline

| Week | Phase | Key Deliverables |
|------|-------|-----------------|
| W1–2 | Study & Setup | Repo skeleton, acceptance criteria, dependency audit |
| W3–4 | Core Backend | Data loader (current/voltage/resistance), adapter API, ABC1 integrated & validated |
| W5–6 | Algorithm Integration | CUQI8 (FEM), PNPE2E (DeepInverse + per-level weights), full 14-metric engine, HDF5 cache |
| W7 | Foundation Modules | M1 (Reconstruction Explorer) + M2 (Difficulty Animator) functional |
| W8–9 | Advanced Modules | M3 (Comparison Grid), M4 (Fingerprint Radar), M5 (Failure Autopsy), M6 (Measurement Viewer) |
| W10 | Validation | Full benchmark: 21 targets × 3 algorithms; reproducibility check; codebase cleanup |
| W11–12 | Report & Demo | Final report + live demo across all 6 modules |

---

## References

1. Räsänen et al., "KTC2023 — EIT Competition and Open Dataset," *Applied Mathematics for Modern Challenges*, 2024. doi:10.3934/ammc.2024009
2. KTC2023 Dataset on Zenodo v3. doi:10.5281/zenodo.10986692
3. Beraldo et al. (ABC1), KTC2023 submission, UFABC. https://github.com/robert-abc/KTC2023-ABC1
4. Carøe et al. (CUQI8), KTC2023 submission, DTU. https://github.com/CUQI-DTU/KTC2023-CUQI8
5. Santacesaria et al. (PNPE2E), KTC2023 submission. https://github.com/msantacesaria/KTC2023_PNPE2E
6. Tachella et al., DeepInverse. doi:10.5281/zenodo.7982256
7. Liu et al., pyEIT. *SoftwareX*, 2018. doi:10.1016/j.softx.2018.09.005
8. Wang et al., SSIM. *IEEE TIP*, 2004. doi:10.1109/TIP.2003.819861
