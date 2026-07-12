# Abstract adapter interface.

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class KTCMeasurement:
    # Container for one KTC2023 measurement loaded from a .mat file.

    current_matrix: np.ndarray    # (n_injections, n_electrodes)
    voltage_matrix: np.ndarray    # (n_injections, n_measurements)
    resistance_matrix: np.ndarray  # (n_injections, n_measurements)  R = V / I
    ground_truth: np.ndarray      # (256, 256) uint8, values in {0, 1, 2}
    level: int                    # 1–7
    sample: str                   # "a" | "b" | "c" | "d"


class AlgorithmAdapter(ABC):
    """Base class for all EIT reconstruction algorithm adapters.

      The adapter is responsible for:
        1. Writing the measurement data to a temporary host directory.
        2. Invoking `docker run` with the correct volume mounts and command.
        3. Reading the output reconstruction back from the host directory.
        4. Returning a 256×256 uint8 array with values in {0, 1, 2}.
    """

    name: str  # Short identifier used in HDF5 cache keys and UI labels
    supports_level_batching: bool = False  # Override in adapters that can process a whole level at once

    def __init__(self, timeout: int = 600, stream: bool = False) -> None:
        self.timeout = timeout
        self.stream = stream

    @abstractmethod
    def reconstruct(self, measurement: KTCMeasurement) -> np.ndarray:
        """ Run reconstruction for a single measurement.
            Returns: 256×256 uint8 ndarray with pixel values in {0=water, 1=resistive, 2=conductive}.
        """

    def reconstruct_level(
        self, level: int, samples: list[str]
    ) -> dict[str, np.ndarray]:
        # Run reconstruction for all samples at a given level in one docker call.
        # Returns: Dict mapping sample identifier → 256×256 uint8 ndarray.
        raise NotImplementedError("This adapter does not support level batching.")
