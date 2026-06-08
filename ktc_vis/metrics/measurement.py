"""Measurement-domain metrics. Owner: Smit Savani."""

from __future__ import annotations

import numpy as np


def compute_voltage_residual(measurement) -> float:  # noqa: ANN001
    """Mean per-injection RMS voltage deviation from the cross-injection baseline.

    Subtracts the mean voltage pattern across all injections (proxy for the
    empty-tank response) from each injection row, then computes the RMS of the
    residual and averages across injections.

    Args:
        measurement: KTCMeasurement with a populated voltage_matrix
                     of shape (n_injections, n_channels).

    Returns:
        Mean RMS residual (float). Lower = weaker inclusion signal.
    """
    V = np.asarray(measurement.voltage_matrix, dtype=np.float64)
    baseline = V.mean(axis=0, keepdims=True)
    delta = V - baseline
    per_inj_rms = np.sqrt(np.mean(delta ** 2, axis=1))
    return float(per_inj_rms.mean())


def compute_resistance_consistency(measurement) -> float:  # noqa: ANN001
    """Consistency of the R = V/I resistance map, in [0, 1].

    Computes the coefficient of variation (std / |mean|) of the full
    resistance matrix and maps it to a [0, 1] score via 1 / (1 + CV).

    A value near 1 means the resistance field is spatially uniform (consistent
    with a simple homogeneous background). A value near 0 means high spatial
    variability — strong inclusion perturbation or noisy measurements.

    Args:
        measurement: KTCMeasurement with a populated resistance_matrix
                     of shape (n_injections, n_measurements).

    Returns:
        Consistency score in [0, 1]. Higher = more consistent.
    """
    R = np.asarray(measurement.resistance_matrix, dtype=np.float64)
    mean_abs = float(np.abs(R).mean())
    if mean_abs < 1e-10:
        return 0.0
    cv = float(np.std(R) / mean_abs)
    return float(1.0 / (1.0 + cv))


def compute_current_sensitivity_balance(measurement) -> float:  # noqa: ANN001
    """How evenly distributed the injected current magnitudes are across injections.

    Computes the L1 norm (total absolute current) per injection pattern, then
    applies the same CV-to-score mapping as resistance_consistency:
    score = 1 / (1 + CV), where CV = std / |mean| of per-injection totals.

    A score near 1 means every injection pattern carries roughly the same total
    current (balanced protocol). A score near 0 means a few injections dominate.

    Args:
        measurement: KTCMeasurement with a populated current_matrix
                     of shape (n_injections, n_electrodes).

    Returns:
        Balance score in [0, 1]. Higher = more evenly balanced.
    """
    I = np.asarray(measurement.current_matrix, dtype=np.float64)
    per_inj_magnitude = np.abs(I).sum(axis=1)   # (n_injections,)
    mean_mag = float(per_inj_magnitude.mean())
    if mean_mag < 1e-10:
        return 0.0
    cv = float(per_inj_magnitude.std() / mean_mag)
    return float(1.0 / (1.0 + cv))
