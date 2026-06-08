"""Measurement-domain metrics. Owner: Smit Savani."""

from __future__ import annotations

import numpy as np


def compute_voltage_residual(measurement) -> float:  # noqa: ANN001
    """Mean per-injection RMS voltage deviation from the cross-injection baseline.

    Subtracts the mean voltage pattern across all injections (proxy for the
    empty-tank response) from each injection row, then computes the RMS of the
    residual and averages across injections.

    A large value means the inclusion strongly perturbs the voltage field.

    Args:
        measurement: KTCMeasurement with a populated voltage_matrix
                     of shape (n_injections, n_channels).

    Returns:
        Mean RMS residual (float). Lower = weaker inclusion signal.
    """
    V = np.asarray(measurement.voltage_matrix, dtype=np.float64)
    baseline = V.mean(axis=0, keepdims=True)      # average over all injections
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
