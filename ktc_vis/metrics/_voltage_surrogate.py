"""Lightweight EIT forward surrogate for algorithm-aware measurement metrics.

We need three measurement-domain metrics in Module 3 that actually depend on the
reconstruction (not just on the raw measurement). A full FEM forward solver is
overkill for a dashboard score, so this module implements a **Born / linearised
sensitivity approximation** on a circular tank, using Geselowitz's sensitivity
formula from bioimpedance theory (1971):

  δV_pred[k, j] ≈ -∫ ∇φ_inj_k · ∇φ_meas_kj · δσ(x) dx

where ``φ_e(x) = -1/(2π) log|x - e|`` is the free-space Green's function of the
2D Laplace equation — the logarithmic potential of a unit point source at
electrode ``e``, evaluated on a 256×256 pixel grid. The reconstruction's label
map is converted to a signed conductivity perturbation ``δσ ∈ {-1, 0, +1}`` for
(resistive, water, conductive).

This is enough to discriminate algorithms: a good reconstruction produces a
δV_pred whose per-pattern profile correlates with the measured δV; a degenerate
one does not.

Pure numpy/scipy, no FEM, no external KTC modules. Per-electrode gradient
fields are cached so the cost is amortised across all (alg, level, sample)
calls in one process.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

# ── Tank geometry (KTC2023) ────────────────────────────────────────────────────
_TANK_RADIUS_M = 0.115
_TANK_DIAMETER_M = 2 * _TANK_RADIUS_M
_N_ELECTRODES_FULL = 32
_PIX = 256
_PIXEL_WIDTH_M = _TANK_DIAMETER_M / _PIX
_PIXEL_AREA_M2 = _PIXEL_WIDTH_M ** 2

# Soft regulariser used when an electrode point falls on a pixel — prevents
# log(0) and 1/0 in the gradient.
_REG_EPS_M = 0.5 * _PIXEL_WIDTH_M


def _electrode_positions(n_el: int = _N_ELECTRODES_FULL) -> np.ndarray:
    """Return (n_el, 2) electrode positions on the tank boundary.

    Electrodes are evenly spaced around the disk; electrode 0 sits on the +x
    axis. This matches the KTC2023 convention used by KTCAux.setMeasurementPattern.
    """
    theta = 2 * np.pi * np.arange(n_el) / n_el
    return _TANK_RADIUS_M * np.stack([np.cos(theta), np.sin(theta)], axis=1)


@lru_cache(maxsize=4)
def _per_electrode_gradient_field(
    n_el: int = _N_ELECTRODES_FULL,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return per-electrode log-potential and its 2D gradient on the pixel grid.

    Returns:
        Tuple ``(disk_mask, dphi_dx, dphi_dy)`` where ``disk_mask`` is the
        (256, 256) boolean mask of pixels inside the circular tank and
        ``dphi_dx``, ``dphi_dy`` are (n_el, 256, 256) gradients of
        ``φ_e(x) = -1/(2π) log|x - e|`` at every pixel for every electrode.
    """
    half = _TANK_RADIUS_M - 0.5 * _PIXEL_WIDTH_M
    coords = np.linspace(-half, half, _PIX)
    X, Y = np.meshgrid(coords, coords)  # both (256, 256)
    r2 = X * X + Y * Y
    disk_mask = r2 <= _TANK_RADIUS_M ** 2

    electrodes = _electrode_positions(n_el)  # (n_el, 2)
    # vectors x - e for every pixel and every electrode → (n_el, 256, 256, 2)
    dx = X[None, :, :] - electrodes[:, 0, None, None]
    dy = Y[None, :, :] - electrodes[:, 1, None, None]
    r2_e = dx * dx + dy * dy + _REG_EPS_M ** 2

    # ∇φ_e = -(1 / (2π)) · (x - e) / |x - e|²
    coeff = -1.0 / (2.0 * np.pi)
    dphi_dx = coeff * dx / r2_e
    dphi_dy = coeff * dy / r2_e

    # Zero contribution outside the tank so it cannot leak into integrals.
    dphi_dx = dphi_dx * disk_mask[None, :, :]
    dphi_dy = dphi_dy * disk_mask[None, :, :]
    return disk_mask, dphi_dx, dphi_dy


def _label_to_contrast(reconstruction: np.ndarray) -> np.ndarray:  # translate labels into physics
    """Map label image {0=water, 1=resistive, 2=conductive} → δσ ∈ {-1, 0, +1}."""
    r = np.asarray(reconstruction)
    out = np.zeros(r.shape, dtype=np.float64)
    out[r == 1] = -1.0  # resistive (lower conductivity)
    out[r == 2] = +1.0  # conductive (higher conductivity)
    return out


def _active_electrode_indices(
    current_matrix: np.ndarray,
    mpat: np.ndarray | None,
    n_full: int = _N_ELECTRODES_FULL,
) -> np.ndarray:
    """Recover the original 0..31 electrode indices that survived sub-sampling.

    After ``subsample_measurement`` the current matrix and mpat are restricted
    to the active electrodes (e.g. shape (n_inj, 30) at level 2). The columns
    appear in the original order, so the active set is ``[0..n_active-1]``
    when the sub-sampler uses prefix selection — which it does (see
    ``ktc_vis.data.subsampler._ELECTRODE_INDICES``).
    """
    n_active = current_matrix.shape[1]
    # Defensive: only the contiguous prefix scheme is supported here. If
    # mpat width or current width disagrees with that, fall back to all-active.
    if n_active > n_full:
        return np.arange(n_full)
    return np.arange(n_active)


def predict_voltage_perturbation(
    reconstruction: np.ndarray,
    current_matrix: np.ndarray,
    mpat: np.ndarray | None,
) -> np.ndarray:
    """Predict δV (voltage perturbation from a homogeneous-tank baseline).

    Args:
        reconstruction: (256, 256) label image with values in {0, 1, 2}.
        current_matrix: (n_inj, n_active) injected currents per pattern.
        mpat: (n_active, n_meas_per_inj) measurement pattern matrix. Each
            column has two non-zero entries (+1, -1) marking the electrode
            pair whose voltage difference is read out. If ``None`` we assume
            the standard adjacent pattern ``V[j] - V[j+1]``.

    Returns:
        (n_inj, n_meas_per_inj) predicted voltage perturbation array, in the
        same orientation as ``measurement.voltage_matrix``.
    """
    delta_sigma = _label_to_contrast(reconstruction)  # (256, 256)
    _, dphi_dx, dphi_dy = _per_electrode_gradient_field(_N_ELECTRODES_FULL)
    # (n_full, 256, 256) — already zero outside the disk.

    active = _active_electrode_indices(current_matrix, mpat)
    n_inj = current_matrix.shape[0]
    n_active = active.size

    # Map active currents back into a 32-vector per pattern so we can index
    # straight into the precomputed full-electrode field.
    I_full = np.zeros((n_inj, _N_ELECTRODES_FULL), dtype=np.float64)
    I_full[:, active] = current_matrix.astype(np.float64)

    # Injection field per pattern: ∇φ_inj_k = Σ_e I[k,e] ∇φ_e
    inj_dx = np.tensordot(I_full, dphi_dx, axes=(1, 0))  # (n_inj, 256, 256)
    inj_dy = np.tensordot(I_full, dphi_dy, axes=(1, 0))

    # Build the measurement field for every (pattern, channel) on the fly.
    if mpat is None:
        n_meas = n_active - 1
        # Standard adjacent pattern: channel j → φ_j - φ_{j+1}
        m_dx = dphi_dx[active[:n_meas]] - dphi_dx[active[1:n_meas + 1]]
        m_dy = dphi_dy[active[:n_meas]] - dphi_dy[active[1:n_meas + 1]]
    else:
        mpat = np.asarray(mpat, dtype=np.float64)
        # mpat shape (n_active, n_meas). Project to full-electrode basis.
        M_full = np.zeros((_N_ELECTRODES_FULL, mpat.shape[1]), dtype=np.float64)
        M_full[active, :] = mpat
        # Σ_e mpat[e, j] · ∇φ_e → (n_meas, 256, 256)
        m_dx = np.tensordot(M_full.T, dphi_dx, axes=(1, 0))
        m_dy = np.tensordot(M_full.T, dphi_dy, axes=(1, 0))

    n_meas = m_dx.shape[0]
    weighted = delta_sigma[None, :, :] * _PIXEL_AREA_M2  # broadcast over channels

    # δV[k, j] = -Σ_xy (inj_dx[k]·m_dx[j] + inj_dy[k]·m_dy[j]) · δσ(xy)
    # Vectorised: collapse the spatial axes once.
    inj_flat_x = inj_dx.reshape(n_inj, -1)
    inj_flat_y = inj_dy.reshape(n_inj, -1)
    m_flat_x = (m_dx * weighted).reshape(n_meas, -1)
    m_flat_y = (m_dy * weighted).reshape(n_meas, -1)

    delta_v = -(inj_flat_x @ m_flat_x.T + inj_flat_y @ m_flat_y.T)
    return delta_v  # (n_inj, n_meas)
