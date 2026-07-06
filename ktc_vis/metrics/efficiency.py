# Runtime metrics.

import time

from ktc_vis.adapters.base import AlgorithmAdapter, KTCMeasurement


def measure_runtime(adapter: AlgorithmAdapter, measurement: KTCMeasurement) -> float:
    # Measure wall-clock reconstruction time in seconds.
    # Returns:  Elapsed time in seconds.

    start = time.perf_counter()
    adapter.reconstruct(measurement)
    return time.perf_counter() - start
