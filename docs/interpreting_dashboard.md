# Interpreting the KTC-Vis Dashboard

## The Sidebar

The sidebar is shared across all modules. Changing the **Algorithm**, **Level**, or **Sample** updates the currently open module instantly.

- **Algorithm:** ABC1, CUQI8, or PNPE2E (see README for descriptions).
- **Difficulty Level (1–7):** Level 1 = easiest (32 electrodes, 2356 measurements); Level 7 = hardest (27 electrodes, 513 measurements).
- **Sample (A/B/C):** Three different physical inclusion configurations per level.

---

## Module 1 — Reconstruction Explorer

**What it shows:** Four side-by-side panels for any (algorithm, level, sample):
1. **Ground Truth** — The photographed actual inclusion layout (reference).
2. **Reconstruction** — Raw conductivity or heatmap from the algorithm.
3. **Predicted Segmentation** — 3-class output: water (blue), resistive (red), conductive (green).
4. **Error Overlay** — Pixel-level errors: green=correct, red=false positive, blue=false negative.

**How to use:** Select a specific case to understand what the algorithm got right and wrong. The error overlay shows the spatial distribution of mistakes.

---

## Module 2 — Difficulty Animator

**What it shows:** How each algorithm degrades as the level increases from 1 → 7.
- **Animation panels:** Reconstruction changes frame-by-frame as level increases.
- **Degradation curves:** Line plots of SSIM, IoU, Hausdorff distance, and runtime across levels.

**Key insight:** An algorithm whose SSIM drops steeply from level 3 to 4 is sensitive to electrode removal. CUQI8 is expected to be stable; ABC1 is expected to degrade faster.

---

## Module 3 — Side-by-Side Comparison Grid

**What it shows:** All three algorithms at the **same level and sample** simultaneously.
- **3-column grid:** ABC1 | CUQI8 | PNPE2E
- **Pairwise difference images:** If errors cluster in the same region for two algorithms, it's a data problem (not algorithm-specific).
- **Score table:** SSIM, IoU, and runtime for all three algorithms.

---

## Module 4 — Fingerprint Radar

**What it shows:** A 9-axis radar chart summarizing each algorithm's performance profile.

| Axis | Good direction | What it means |
|------|---------------|----------------|
| Total SSIM | Outward | Higher overall image quality |
| Easy SSIM | Outward | Strong at levels 1–3 |
| Hard SSIM | Outward | Strong at levels 5–7 |
| IoU Conductive | Outward | Better at detecting conductive inclusions |
| IoU Resistive | Outward | Better at detecting resistive inclusions |
| Speed | Outward | Faster reconstruction |
| Robustness | Outward | Consistent across samples |
| Voltage Residual | Outward | **Inverted** — outward = lower residual = better measurement fit |
| Current Sensitivity Balance | Outward | Uses all injection patterns equally |

**Expected shapes:** CUQI8 → narrow/tall (robust + good residual); ABC1 → wide/low (fast + strong for easy); PNPE2E → balanced.

---

## Module 5 — Failure Autopsy

**What it shows:** The worst-performing (algorithm, level, sample) cases ranked by SSIM.

Clicking a row opens 4 diagnostic panels:
1. **Spatial SSIM heatmap** — Which pixels hurt the score most (red = bad).
2. **3×3 Confusion matrix** — What materials are being misclassified (e.g., conductive predicted as water).
3. **Boundary-error radial plot** — Whether errors cluster near electrode gaps (data problem) or are spread around (algorithm problem).
4. **Measurement residual polar plot** — Whether the algorithm fails because of poor measurements or poor learning.

**Failure types:**

| Type | Name | What happened |
|------|------|----------------|
| A | Ghost inclusion | ABC1 CNN sees electrode artifact as a real inclusion |
| B | Missing inclusion | CUQI8 level-set converges to wrong number of regions |
| C | Class flip | PNPE2E gets the location right but the conductivity sign wrong |
| D | Boundary erosion | ABC1 smoothness prior blurs inclusion edges |
| E | Mask suppression | PNPE2E E2E stage masks out a real inclusion |

---

## Module 6 — Measurement Domain Viewer

**What it shows:** The raw electrical measurements that were given to the algorithms — independent of any reconstruction.

- **Current panel:** Polar bar chart of injected current patterns (electrode positions = angle, magnitude = bar height). Shows which electrodes were active.
- **Voltage panel (top):** Polar plot of measured voltages per injection. Shows voltage distribution around the tank boundary.
- **Voltage panel (bottom):** Difference between inclusion case and empty tank. This difference is the key input to ABC1 and the loss comparison for CUQI8.
- **Resistance panel:** R = V/I values. Left: scatter plot for one electrode pair. Right: 2D heatmap (rows=injection, columns=measurement pairs). Bright regions = high-information measurement combinations.

**Key insight:** If the voltage difference in Level 7 is noisy or flat, all algorithms struggle — this is a data limitation, not an algorithm flaw.
