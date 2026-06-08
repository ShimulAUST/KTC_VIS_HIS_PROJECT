"""Measurement-domain metrics. Owner: Smit Savani."""

import numpy as np


def compute_voltage_residual(measurement) -> float:
    """RMS deviation of each injection pattern from the mean voltage pattern."""
    V = np.asarray(measurement.voltage_matrix, dtype=np.float64)
    baseline = V.mean(axis=0, keepdims=True)
    delta = V - baseline
    per_inj_rms = np.sqrt(np.mean(delta ** 2, axis=1))
    return float(per_inj_rms.mean())


def compute_resistance_consistency(measurement) -> float:
    """Inverse coefficient of variation of the resistance matrix (1 = perfectly consistent)."""
    R = np.asarray(measurement.resistance_matrix, dtype=np.float64)
    mean_abs = float(np.abs(R).mean())
    if mean_abs < 1e-10:
        return 0.0
    cv = float(np.std(R) / mean_abs)
    return float(1.0 / (1.0 + cv))


def compute_current_sensitivity_balance(measurement) -> float:
    """Inverse coefficient of variation of per-injection current magnitudes (1 = perfectly balanced)."""
    I = np.asarray(measurement.current_matrix, dtype=np.float64)
    per_inj_magnitude = np.abs(I).sum(axis=1)
    mean_mag = float(per_inj_magnitude.mean())
    if mean_mag < 1e-10:
        return 0.0
    cv = float(per_inj_magnitude.std() / mean_mag)
    return float(1.0 / (1.0 + cv))
