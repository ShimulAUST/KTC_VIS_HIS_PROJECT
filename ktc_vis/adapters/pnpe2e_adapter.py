"""PNPE2E algorithm adapter. Owner: Smit Savani.

Wraps Docker image: muzammal5566/ktc2023-pnpe2e:latest

Container run command:
    docker run --rm --platform linux/amd64 \
      -v "<level_dir>:/app/TrainingData" \
      -v "<output_dir>:/app/Output" \
      muzammal5566/ktc2023-pnpe2e:latest \
      python -c "from main import main; main('TrainingData', 'Output', <level>)"

We mount only TrainingData and Output — the container keeps its own
model weights, helper code, and environment intact.
"""

import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import scipy.io

from ktc_vis.adapters.base import AlgorithmAdapter, KTCMeasurement

DOCKER_IMAGE = "muzammal5566/ktc2023-pnpe2e:latest"
RAW_DIR = Path("data/raw/ktc2023")

_SAMPLE_INDEX = {"a": 1, "b": 2, "c": 3}


class PNPE2EAdapter(AlgorithmAdapter):
    """Adapter for the PNPE2E (physics-informed end-to-end) reconstruction algorithm."""

    name = "pnpe2e"

    def reconstruct(self, measurement: KTCMeasurement) -> np.ndarray:
        """Run PNPE2E reconstruction via Docker and return 256×256 segmentation.

        Args:
            measurement: Loaded KTC2023 measurement.

        Returns:
            256×256 uint8 ndarray with values in {0, 1, 2}.
        """
        level_dir = (RAW_DIR / f"level{measurement.level}").resolve()
        sample_idx = _SAMPLE_INDEX[measurement.sample]

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            training_dir = tmp_path / "TrainingData"
            output_dir = tmp_path / "Output"
            training_dir.mkdir()
            output_dir.mkdir()

            # Copy only measurement files — exclude ground truth (*_true.mat)
            for fname in ("data1.mat", "data2.mat", "data3.mat", "ref.mat"):
                shutil.copy2(level_dir / fname, training_dir / fname)

            python_call = (
                f"from main import main; main('TrainingData', 'Output', {measurement.level})"
            )
            cmd = [
                "docker", "run", "--rm",
                "--platform", "linux/amd64",
                "-v", f"{training_dir}:/app/TrainingData",
                "-v", f"{output_dir}:/app/Output",
                DOCKER_IMAGE,
                "python", "-c", python_call,
            ]
            subprocess.run(cmd, check=True, capture_output=True)

            return _load_reconstruction(output_dir, sample_idx)


def _load_reconstruction(output_dir: Path, sample_idx: int) -> np.ndarray:
    """Read the reconstruction output for the given sample index.

    Args:
        output_dir: Directory where the container wrote its results.
        sample_idx: 1, 2, or 3 corresponding to samples a, b, c.

    Returns:
        256×256 uint8 ndarray with values in {0, 1, 2}.

    Raises:
        FileNotFoundError: If the expected output file is missing.
    """
    out_path = output_dir / f"data{sample_idx}.mat"
    if not out_path.exists():
        mat_files = list(output_dir.glob("*.mat"))
        if not mat_files:
            raise FileNotFoundError(f"PNPE2E container produced no output in {output_dir}")
        out_path = mat_files[0]

    data = scipy.io.loadmat(str(out_path))
    for value in data.values():
        if isinstance(value, np.ndarray) and value.ndim == 2:
            return value.astype(np.uint8)

    raise ValueError(f"No 2-D array found in PNPE2E output: {out_path}")
