"""Electrode subsampling per KTC2023 difficulty level. Owner: Muzammal."""

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
