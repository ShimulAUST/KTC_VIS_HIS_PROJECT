"""KTC2023 dataset loader. Owner: Muzammal.

Reads the canonical staged dataset under ``data/raw/ktc2023/`` (populated by
``scripts/stage_dataset.py``). The native dataset is captured at Level 1
(32 electrodes); lower levels are produced at runtime via electrode
subsampling — see :mod:`ktc_vis.data.subsampler`.
"""

from pathlib import Path

import numpy as np
import scipy.io

from ktc_vis.adapters.base import KTCMeasurement

# Maps sample letter → data file index. KTC2023 ships 4 samples (1..4).
_SAMPLE_INDEX = {"a": 1, "b": 2, "c": 3, "d": 4}

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
    """Loads KTC2023 .mat files into :class:`KTCMeasurement` dataclasses.

    Expected directory layout under ``raw_dir``::

        measurements/data{1..4}.mat   — measurement (Inj, Mpat, Uel)
        measurements/ref.mat          — empty-tank reference (Injref, Mpat, Uelref)
        ground_truth/true{1..4}.mat   — segmentation (key: truth, 256×256 uint8)

    Only level=1 is stored on disk; higher levels are realised by subsampling
    the Level-1 measurement at load time.
    """

    def __init__(self, raw_dir: str | Path = "data/raw/ktc2023") -> None:
        self.raw_dir = Path(raw_dir)
        self.measurements_dir = self.raw_dir / "measurements"
        self.ground_truth_dir = self.raw_dir / "ground_truth"

    def load(self, level: int, sample: str) -> KTCMeasurement:
        """Load one ``(level, sample)`` combination from disk.

        Args:
            level: Difficulty level, 1–7. (Level > 1 not yet implemented.)
            sample: Sample identifier, one of ``"a"``, ``"b"``, ``"c"``, ``"d"``.

        Returns:
            :class:`KTCMeasurement` with all matrices populated.

        Raises:
            ValueError: If ``level`` or ``sample`` is invalid.
            FileNotFoundError: If the expected ``.mat`` files are missing.
            NotImplementedError: If ``level > 1`` (subsampling not yet wired in).
        """
        if level not in range(1, 8):
            raise ValueError(f"level must be 1–7, got {level}")
        if sample not in _SAMPLE_INDEX:
            raise ValueError(
                f"sample must be one of {sorted(_SAMPLE_INDEX)}, got '{sample}'"
            )

        idx = _SAMPLE_INDEX[sample]

        data = self._load_mat(self.measurements_dir / f"data{idx}.mat")
        gt_data = self._load_mat(self.ground_truth_dir / f"true{idx}.mat")

        # Uel is stored as (n_measurements, 1) — flatten to 1-D then reshape
        uel_flat = data["Uel"].flatten()           # (n_measurements,)
        inj = data["Inj"]                          # (32, n_injections)
        mpat = data["Mpat"].astype(np.int16)       # (32, 31) measurement pattern
        n_inj = inj.shape[1]
        n_meas_per_inj = len(uel_flat) // n_inj

        voltage_matrix = uel_flat.reshape(n_inj, n_meas_per_inj)

        # Each column of Inj is one injection pattern across 32 electrodes
        current_matrix = inj.T                     # (n_injections, 32)

        # R = V / I — use peak current magnitude per injection (avoids /0)
        current_magnitude = np.abs(current_matrix).max(axis=1, keepdims=True)
        current_magnitude = np.where(current_magnitude == 0, 1.0, current_magnitude)
        resistance_matrix = voltage_matrix / current_magnitude

        ground_truth = gt_data["truth"].astype(np.uint8)

        measurement = KTCMeasurement(
            current_matrix=current_matrix,
            voltage_matrix=voltage_matrix,
            resistance_matrix=resistance_matrix,
            ground_truth=ground_truth,
            level=1,
            sample=sample,
        )
        # Attach Mpat for downstream subsampling (not part of the public dataclass)
        measurement.__dict__["mpat"] = mpat

        if level != 1:
            from ktc_vis.data.subsampler import subsample_measurement
            measurement = subsample_measurement(measurement, level)

        return measurement

    @staticmethod
    def _load_mat(path: Path) -> dict:
        """Load a .mat file, raising :class:`FileNotFoundError` with guidance."""
        if not path.exists():
            raise FileNotFoundError(
                f"KTC2023 data file not found: {path}\n"
                "Run 'python scripts/stage_dataset.py' to populate data/raw/ktc2023/."
            )
        return scipy.io.loadmat(str(path))
