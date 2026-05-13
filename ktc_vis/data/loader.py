"""KTC2023 dataset loader. Owner: Muzammal."""

from pathlib import Path

import numpy as np
import scipy.io

from ktc_vis.adapters.base import KTCMeasurement

# Maps sample letter → data file index
_SAMPLE_INDEX = {"a": 1, "b": 2, "c": 3}

# Expected measurement counts per level (from KTC2023 paper Table 1)
EXPECTED_MEASUREMENTS = {
    1: 2356,
    2: 1624,
    3: 1404,
    4: 1200,
    5: 1012,
    6: 630,
    7: 513,
}


class KTCDataLoader:
    """Loads KTC2023 .mat files into KTCMeasurement dataclasses.

    Directory layout expected under raw_dir:
        level{L}/data{1|2|3}.mat   — measurement data (Inj, Mpat, Uel)
        level{L}/ref.mat           — background reference (Injref, Mpat, Uelref)
        level{L}/{1|2|3}_true.mat  — ground truth segmentation (truth)

    Args:
        raw_dir: Path to the ktc2023 root directory (contains level1/ … level7/).
    """

    def __init__(self, raw_dir: str | Path = "data/raw/ktc2023") -> None:
        self.raw_dir = Path(raw_dir)

    def load(self, level: int, sample: str) -> KTCMeasurement:
        """Load one (level, sample) combination from disk.

        Args:
            level: Difficulty level, 1–7.
            sample: Sample identifier, one of "a", "b", "c".

        Returns:
            KTCMeasurement with all matrices populated.

        Raises:
            ValueError: If level or sample is invalid.
            FileNotFoundError: If the expected .mat files are missing.
        """
        if level not in range(1, 8):
            raise ValueError(f"level must be 1–7, got {level}")
        if sample not in _SAMPLE_INDEX:
            raise ValueError(f"sample must be 'a', 'b', or 'c', got '{sample}'")

        idx = _SAMPLE_INDEX[sample]
        level_dir = self.raw_dir / f"level{level}"

        data = self._load_mat(level_dir / f"data{idx}.mat")
        gt_data = self._load_mat(level_dir / f"{idx}_true.mat")

        # Uel is stored as (n_measurements, 1) — flatten to 1-D then reshape
        uel_flat = data["Uel"].flatten()           # (n_measurements,)
        inj = data["Inj"]                          # (32, n_injections)
        n_inj = inj.shape[1]
        n_meas_per_inj = len(uel_flat) // n_inj

        voltage_matrix = uel_flat.reshape(n_inj, n_meas_per_inj)

        # Build a current matrix: each column of Inj is one injection pattern
        current_matrix = inj.T                     # (n_injections, 32)

        # R = V / I — use mean current magnitude per injection to avoid div-by-zero
        current_magnitude = np.abs(current_matrix).max(axis=1, keepdims=True)
        current_magnitude = np.where(current_magnitude == 0, 1.0, current_magnitude)
        resistance_matrix = voltage_matrix / current_magnitude

        ground_truth = gt_data["truth"].astype(np.uint8)

        return KTCMeasurement(
            current_matrix=current_matrix,
            voltage_matrix=voltage_matrix,
            resistance_matrix=resistance_matrix,
            ground_truth=ground_truth,
            level=level,
            sample=sample,
        )

    @staticmethod
    def _load_mat(path: Path) -> dict:
        """Load a .mat file, raising FileNotFoundError with a clear message."""
        if not path.exists():
            raise FileNotFoundError(
                f"KTC2023 data file not found: {path}\n"
                "Run 'bash scripts/setup_third_party.sh' or download the dataset."
            )
        return scipy.io.loadmat(str(path))
