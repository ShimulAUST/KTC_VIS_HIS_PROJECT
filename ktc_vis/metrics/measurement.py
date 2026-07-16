"""These three scalars summarise how well an algorithm's reconstruction *explains*
the measured voltages, evaluated with a lightweight Born / linearised forward
surrogate (see :mod:`ktc_vis.metrics._voltage_surrogate`). Because they depend
on the reconstruction they vary per algorithm — distinguishing ABC1, CUQI8 and
PNPE2E on the Module 3 scorecard.
"""

from __future__ import annotations

import numpy as np

from ktc_vis.adapters.base import KTCMeasurement
from ktc_vis.metrics._voltage_surrogate import predict_voltage_perturbation

_EPS = 1e-12


def _measured_perturbation(measurement: KTCMeasurement) -> np.ndarray:
    """Per-injection zero-mean voltage perturbation (n_inj, n_meas_per_inj)."""
    V = np.asarray(measurement.voltage_matrix, dtype=np.float64)
    if V.size == 0:
        return V
    return V - V.mean(axis=1, keepdims=True)


def _predicted_perturbation(
    measurement: KTCMeasurement,
    reconstruction: np.ndarray,
) -> np.ndarray:
    """Forward-surrogate prediction of the per-injection voltage perturbation.

    Returns an array shaped like ``measurement.voltage_matrix`` (zero-mean per
    row so it is directly comparable to :func:`_measured_perturbation`).
    """
    mpat = measurement.__dict__.get("mpat")
    pred = predict_voltage_perturbation(
        reconstruction,
        np.asarray(measurement.current_matrix, dtype=np.float64),
        np.asarray(mpat) if mpat is not None else None,
    )
    if pred.size == 0:
        return pred
    return pred - pred.mean(axis=1, keepdims=True)


def _row_pearson(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pearson correlation per row; returns shape (n_rows,) with NaN if undefined."""
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)
    num = (a * b).sum(axis=1)
    den = np.sqrt((a * a).sum(axis=1) * (b * b).sum(axis=1))
    out = np.full(a.shape[0], np.nan, dtype=np.float64)
    ok = den > _EPS
    out[ok] = num[ok] / den[ok]
    return out


def compute_voltage_residual(
    measurement: KTCMeasurement,
    reconstruction: np.ndarray | None = None,
) -> float:
    """RMS error between measured voltages and the surrogate forward prediction.

    The reconstruction is converted to a signed conductivity contrast and pushed
    through a Born approximation of the EIT forward operator. The predicted
    voltage perturbation is best-fit-scaled to the measured perturbation (one
    scalar α per dataset) and the residual RMS is reported relative to the
    measured RMS.

    Args:
        measurement: Loaded KTC2023 measurement.
        reconstruction: 256×256 label image with values in {0, 1, 2}. If
            ``None``, falls back to the legacy measurement-only residual so
            tests that only have a measurement keep working.

    Returns:
        Scalar residual in [0, 1] — lower is better.
    """
    dv_meas = _measured_perturbation(measurement)
    if dv_meas.size == 0:
        return 0.0
    denom = float(np.sqrt(np.mean(dv_meas ** 2))) + _EPS

    if reconstruction is None:
        V = np.asarray(measurement.voltage_matrix, dtype=np.float64)
        full_rms = float(np.sqrt(np.mean(V ** 2))) + _EPS
        return float(np.clip(denom / full_rms, 0.0, 1.0))

    dv_pred = _predicted_perturbation(measurement, reconstruction)
    num = float((dv_meas * dv_pred).sum())
    den = float((dv_pred * dv_pred).sum()) + _EPS
    alpha = num / den
    residual = dv_meas - alpha * dv_pred
    rms = float(np.sqrt(np.mean(residual ** 2)))
    return float(np.clip(rms / denom, 0.0, 1.0))


def compute_resistance_consistency(
    measurement: KTCMeasurement,
    reconstruction: np.ndarray | None = None,
) -> float:
    """Per-pattern fit consistency in [0, 1].

    For each injection pattern we compute the Pearson correlation between the
    measured and the surrogate-predicted voltage profile across channels.
    The mean correlation is rescaled from ``[-1, +1]`` to ``[0, 1]`` via
    ``(r + 1) / 2`` so the score never collapses to exactly 0 when the Born
    approximation produces a sign-flipped match:

      * 1.0  → predictions track measurements perfectly,
      * 0.5  → no linear relationship,
      * 0.0  → predictions anti-correlate with measurements.

    Args:
        measurement: Loaded KTC2023 measurement.
        reconstruction: 256×256 label image. If ``None`` returns 0.5
            (no information).

    Returns:
        Score in [0, 1] — higher is better.
    """
    if reconstruction is None:
        return 0.5
    dv_meas = _measured_perturbation(measurement)
    if dv_meas.size == 0:
        return 0.5
    dv_pred = _predicted_perturbation(measurement, reconstruction)
    r = _row_pearson(dv_meas, dv_pred) #_row_pearson() returns an array of Pearson correlation coefficients — one number per injection pattern
    r = r[np.isfinite(r)]
    if r.size == 0:
        return 0.5
    return float(np.clip(0.5 * (r.mean() + 1.0), 0.0, 1.0))


def compute_current_sensitivity_balance(
    measurement: KTCMeasurement,
    reconstruction: np.ndarray | None = None,
) -> float:
    """How uniformly the reconstruction explains every active electrode, in [0, 1].

    For each active electrode the Pearson correlation between measured and
    predicted voltages — over all measurement channels involving that electrode
    and all injection patterns — is computed. The mean correlation is rescaled
    from ``[-1, +1]`` to ``[0, 1]`` via ``(r + 1) / 2`` (same convention as
    :func:`compute_resistance_consistency`).

    Args:
        measurement: Loaded KTC2023 measurement.
        reconstruction: 256×256 label image. If ``None`` returns 0.5.

    Returns:
        Score in [0, 1] — higher is better.
    """
    if reconstruction is None:
        return 0.5

    dv_meas = _measured_perturbation(measurement)
    if dv_meas.size == 0:
        return 0.5
    dv_pred = _predicted_perturbation(measurement, reconstruction)

    mpat = measurement.__dict__.get("mpat")
    if mpat is None:
        return 0.5
    mpat = np.asarray(mpat)
    n_active = mpat.shape[0]
    if n_active == 0:
        return 0.5

    scores: list[float] = []
    for e in range(n_active):
        ch_mask = mpat[e] != 0
        if not np.any(ch_mask):
            continue
        meas_slice = dv_meas[:, ch_mask].ravel()
        pred_slice = dv_pred[:, ch_mask].ravel()
        meas_slice = meas_slice - meas_slice.mean()
        pred_slice = pred_slice - pred_slice.mean()
        denom = float(np.linalg.norm(meas_slice) * np.linalg.norm(pred_slice))
        if denom < _EPS:
            continue
        scores.append(float(meas_slice @ pred_slice) / denom)

    if not scores:
        return 0.5
    return float(np.clip(0.5 * (np.mean(scores) + 1.0), 0.0, 1.0))
