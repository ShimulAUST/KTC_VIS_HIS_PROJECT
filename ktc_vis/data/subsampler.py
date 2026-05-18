"""Electrode subsampling per KTC2023 difficulty level. Owner: Muzammal."""

from __future__ import annotations

import numpy as np

from ktc_vis.adapters.base import KTCMeasurement

# KTC2023 paper Table 1: electrodes used per difficulty level.
# Level 1 uses all 32 electrodes; each higher level drops 2 more.
# Indices are 0-based (electrode 1 in the paper = index 0 here).
_ELECTRODE_INDICES: dict[int, list[int]] = {
    1: list(range(32)),   # all 32 electrodes
    2: list(range(30)),   # electrodes 1–30
    3: list(range(28)),   # electrodes 1–28
    4: list(range(26)),   # electrodes 1–26
    5: list(range(24)),   # electrodes 1–24
    6: list(range(22)),   # electrodes 1–22
    7: list(range(20)),   # electrodes 1–20
}

# Expected electrode and measurement counts — used for validation.
LEVEL_SPECS: dict[int, dict[str, int]] = {
    1: {"n_electrodes": 32, "n_measurements": 2356},
    2: {"n_electrodes": 30, "n_measurements": 1624},
    3: {"n_electrodes": 28, "n_measurements": 1404},
    4: {"n_electrodes": 26, "n_measurements": 1200},
    5: {"n_electrodes": 24, "n_measurements": 1012},
    6: {"n_electrodes": 22, "n_measurements": 630},
    7: {"n_electrodes": 20, "n_measurements": 513},
}


def subsample_electrodes(level: int) -> list[int]:
    """Return the 0-based electrode indices used at the given difficulty level.

    Args:
        level: KTC2023 difficulty level, 1–7.

    Returns:
        List of 0-based electrode indices matching KTC2023 paper Table 1.

    Raises:
        ValueError: If level is outside the valid range.
    """
    if level not in _ELECTRODE_INDICES:
        raise ValueError(f"level must be 1–7, got {level}")
    return _ELECTRODE_INDICES[level]


def subsample_measurement(measurement: KTCMeasurement, level: int) -> KTCMeasurement:
    """Return a copy of ``measurement`` reduced to the electrodes of ``level``.

    Applies two filters:
      1. Drops **injection rows** whose nonzero current electrodes are not all
         within the active electrode set (an injection that touches a removed
         electrode is invalid at this level).
      2. Drops **voltage columns** (per the measurement pattern ``Mpat``) whose
         non-zero entries are not all within the active set.

    Args:
        measurement: Level-1 measurement (32 electrodes). Must carry an
            ``mpat`` attribute attached by the loader.
        level: Target difficulty level, 1–7.

    Returns:
        New :class:`KTCMeasurement` with reduced matrices and ``level`` set.
    """
    if level == measurement.level or level == 1:
        return measurement

    active = subsample_electrodes(level)
    active_set = set(active)

    # 1) Filter injections
    inj = measurement.current_matrix          # (n_injections, 32)
    keep_inj_mask = np.array(
        [all(e in active_set for e in np.nonzero(row)[0]) for row in inj],
        dtype=bool,
    )

    # 2) Filter voltage columns by Mpat (32 × 31): each column is a measurement
    #    pair indicated by its two non-zero entries (±1).
    mpat = measurement.__dict__.get("mpat")
    if mpat is not None:
        col_mask = np.array(
            [all(e in active_set for e in np.nonzero(mpat[:, c])[0])
             for c in range(mpat.shape[1])],
            dtype=bool,
        )
    else:
        col_mask = np.ones(measurement.voltage_matrix.shape[1], dtype=bool)

    new_current = inj[keep_inj_mask][:, active]
    new_voltage = measurement.voltage_matrix[keep_inj_mask][:, col_mask]
    new_resistance = measurement.resistance_matrix[keep_inj_mask][:, col_mask]

    new_meas = KTCMeasurement(
        current_matrix=new_current,
        voltage_matrix=new_voltage,
        resistance_matrix=new_resistance,
        ground_truth=measurement.ground_truth,
        level=level,
        sample=measurement.sample,
    )
    if mpat is not None:
        new_meas.__dict__["mpat"] = mpat[:, col_mask][active, :]
    return new_meas
