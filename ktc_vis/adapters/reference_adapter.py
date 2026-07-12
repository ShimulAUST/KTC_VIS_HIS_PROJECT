"""Reference-output adapter: serves pre-computed reconstructions from disk.

This adapter does **not** run any reconstruction algorithm — it simply loads
the published reconstruction ``.mat`` files staged under
``data/raw/ktc2023/reference_outputs/<algo>/``.

It exists so that Module 1 (Reconstruction Explorer) can render results
immediately without first wiring up the Docker-based live adapters.
Replace with the real algorithm adapters in a later sprint.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.io

from ktc_vis.adapters.base import AlgorithmAdapter, KTCMeasurement

# Each algorithm uses a slightly different filename convention in its
# original output folder.  ``stage_dataset.py`` preserves these names.
_FILENAME_FORMAT: dict[str, str] = {
    "abc1": "data{idx}.mat",
    "cuqi8": "{idx}.mat",
    "pnpe2e": "{idx}.mat",
}

_SAMPLE_INDEX = {"a": 1, "b": 2, "c": 3, "d": 4}
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_OUTPUTS_ROOT = _PROJECT_ROOT / "data" / "raw" / "ktc2023" / "reference_outputs"


class ReferenceOutputAdapter(AlgorithmAdapter):
    """Loads a cached reconstruction ``.mat`` file produced by the original repo."""

    def __init__(
        self,
        algorithm: str,
        outputs_root: str | Path = _DEFAULT_OUTPUTS_ROOT,
    ) -> None:
        if algorithm not in _FILENAME_FORMAT:
            raise ValueError(
                f"unknown algorithm '{algorithm}'; expected one of "
                f"{sorted(_FILENAME_FORMAT)}"
            )
        self.name = algorithm
        self._dir = Path(outputs_root) / algorithm
        self._pattern = _FILENAME_FORMAT[algorithm]

    def reconstruct(self, measurement: KTCMeasurement) -> np.ndarray:
        """Return the published reconstruction for ``measurement.sample``.

        Reads ``reference_outputs/<algo>/level<N>/<file>.mat`` for the level
        carried by ``measurement``. Falls back to the legacy flat layout
        (``reference_outputs/<algo>/<file>.mat``) at Level 1 if a per-level
        directory has not yet been staged.
        """
        idx = _SAMPLE_INDEX.get(measurement.sample)
        if idx is None:
            raise ValueError(f"unsupported sample '{measurement.sample}'")

        level = int(measurement.level)
        if level not in range(1, 8):
            raise ValueError(f"level must be 1–7, got {level}")

        filename = self._pattern.format(idx=idx)
        leveled = self._dir / f"level{level}" / filename
        legacy = self._dir / filename  # flat layout (Level 1 only)

        if leveled.exists():
            path = leveled
        elif level == 1 and legacy.exists():
            path = legacy
        else:
            raise FileNotFoundError(
                f"reference reconstruction not found: {leveled}\n"
                "To use pre-computed reference outputs: run 'python scripts/stage_dataset.py'\n"
                "To run algorithms yourself and populate the cache: "
                "run 'python scripts/run_benchmark.py'"
            )

        mat = scipy.io.loadmat(str(path))
        recon = mat["reconstruction"]
        return recon.astype(np.uint8)
