"""Abstract adapter interface. Owner: Muzammal."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class KTCMeasurement:
    """Container for one KTC2023 measurement loaded from a .mat file."""

    current_matrix: np.ndarray    # (n_injections, n_electrodes)
    voltage_matrix: np.ndarray    # (n_injections, n_measurements)
    resistance_matrix: np.ndarray # (n_injections, n_measurements)  R = V / I
    ground_truth: np.ndarray      # (256, 256) uint8, values in {0, 1, 2}
    level: int                    # 1–7
    sample: str                   # "a" | "b" | "c"


class AlgorithmAdapter(ABC):
    """Base class for all EIT reconstruction algorithm adapters.

    Each concrete adapter wraps a Docker container that runs one of the
    KTC2023 competition algorithms.  The adapter is responsible for:
      1. Writing the measurement data to a temporary host directory.
      2. Invoking `docker run` with the correct volume mounts and command.
      3. Reading the output reconstruction back from the host directory.
      4. Returning a 256×256 uint8 array with values in {0, 1, 2}.
    """

    name: str  # Short identifier used in HDF5 cache keys and UI labels

    @abstractmethod
    def reconstruct(self, measurement: KTCMeasurement) -> np.ndarray:
        """Run reconstruction for a single measurement.

        Args:
            measurement: Loaded KTC2023 measurement for one (level, sample).

        Returns:
            256×256 uint8 ndarray with pixel values in {0=water, 1=resistive,
            2=conductive}.
        """
