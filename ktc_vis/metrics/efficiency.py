"""Runtime metrics. Owner: Smit Savani."""

import time

import numpy as np

from ktc_vis.adapters.base import AlgorithmAdapter, KTCMeasurement


def measure_runtime(adapter: AlgorithmAdapter, measurement: KTCMeasurement) -> float:
    """Measure wall-clock reconstruction time in seconds.

    Args:
        adapter: Any AlgorithmAdapter instance.
        measurement: The measurement to reconstruct.

    Returns:
        Elapsed time in seconds.
    """
    start = time.perf_counter()
    adapter.reconstruct(measurement)
    return time.perf_counter() - start
