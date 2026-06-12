"""Measurement-domain metrics. Owner: Smit Savani.

Scalar summaries derived purely from a `KTCMeasurement` (currents, voltages,
resistance) — no forward-model evaluation required, so these can be computed
even for algorithms that don't expose predicted voltages.
"""

from __future__ import annotations

import numpy as np

from ktc_vis.adapters.base import KTCMeasurement

_EPS = 1e-12


def compute_voltage_residual(measurement: KTCMeasurement) -> float:
    """RMS deviation of measured voltages from a per-injection mean baseline.

    Each row of the voltage matrix is one injection pattern. Subtracting the
    row mean removes the DC offset; the remaining RMS, normalised by the RMS
    of the original signal, gives a dimensionless residual in [0, 1].

    Higher values indicate stronger spatial structure (a richer measurement
    that distinguishes inclusions from background); lower values indicate a
    nearly uniform response.

    Args:
        measurement: Loaded KTC2023 measurement.

    Returns:
        Scalar residual in [0, 1]. Returns 0.0 if the voltage matrix is empty.
    """
    V = np.asarray(measurement.voltage_matrix, dtype=np.float64)
    if V.size == 0:
        return 0.0
    baseline = V.mean(axis=1, keepdims=True)
    residual = V - baseline
    num = float(np.sqrt(np.mean(residual ** 2)))
    denom = float(np.sqrt(np.mean(V ** 2))) + _EPS
    return float(num / denom)


def compute_resistance_consistency(measurement: KTCMeasurement) -> float:
    """Homogeneity score of the R = V / I matrix in [0, 1].

    For a homogeneous tank the resistance values are nearly constant, so the
    coefficient of variation ``std(R) / |mean(R)|`` is small. We return
    ``1 - clip(CoV, 0, 1)`` so that higher = more consistent / more homogeneous.

    Args:
        measurement: Loaded KTC2023 measurement.

    Returns:
        Consistency score in [0, 1]. Higher is better.
    """
    R = np.asarray(measurement.resistance_matrix, dtype=np.float64)
    R = R[np.isfinite(R)]
    if R.size == 0:
        return 0.0
    mean = float(np.mean(R))
    if abs(mean) < _EPS:
        return 0.0
    cov = float(np.std(R) / abs(mean))
    return float(1.0 - min(max(cov, 0.0), 1.0))


def compute_current_sensitivity_balance(measurement: KTCMeasurement) -> float:
    """Balance of injected current magnitude across electrodes, in [0, 1].

    For each electrode, sum the absolute current contribution across all
    injection patterns. A perfectly balanced protocol drives every electrode
    with equal total current. We measure deviation from that ideal as the
    coefficient of variation of the per-electrode totals and return
    ``1 - clip(CoV, 0, 1)`` so higher = more balanced.

    Args:
        measurement: Loaded KTC2023 measurement.

    Returns:
        Balance score in [0, 1]. Higher means more uniform electrode usage.
    """
    I = np.asarray(measurement.current_matrix, dtype=np.float64)
    if I.size == 0:
        return 0.0
    per_electrode = np.abs(I).sum(axis=0)
    mean = float(per_electrode.mean())
    if mean < _EPS:
        return 0.0
    cov = float(per_electrode.std() / mean)
    return float(1.0 - min(max(cov, 0.0), 1.0))
