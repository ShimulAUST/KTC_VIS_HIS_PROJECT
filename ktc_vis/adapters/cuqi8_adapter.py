"""CUQI8 algorithm adapter. Owner: Smit Savani.

Wraps Docker image: muzammal5566/ktc2023-cuqi8:latest

Container run command:
    docker run --rm --platform linux/amd64 \
      -v "<level_dir>:/app/TrainingData" \
      -v "<output_dir>:/app/Output" \
      muzammal5566/ktc2023-cuqi8:latest \
      conda run --no-capture-output -n env python main.py TrainingData Output <level>
"""

import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import scipy.io

from ktc_vis.adapters.base import AlgorithmAdapter, KTCMeasurement

DOCKER_IMAGE = "muzammal5566/ktc2023-cuqi8:latest"
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DIR = _PROJECT_ROOT / "data" / "raw" / "ktc2023"

_SAMPLE_INDEX = {"a": 1, "b": 2, "c": 3}


class CUQI8Adapter(AlgorithmAdapter):
    """Adapter for the CUQI8 (Bayesian/FEniCS-based) reconstruction algorithm."""

    name = "cuqi8"
    supports_level_batching = True

    def reconstruct(self, measurement: KTCMeasurement) -> np.ndarray:
        results = self.reconstruct_level(measurement.level, [measurement.sample])
        return results[measurement.sample]

    def reconstruct_level(
        self, level: int, samples: list[str]
    ) -> dict[str, np.ndarray]:
        """Run CUQI8 once for the whole level — one docker run covers all samples."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            training_dir = tmp_path / "TrainingData"
            output_dir = tmp_path / "Output"
            training_dir.mkdir()
            output_dir.mkdir()

            _write_training_data(training_dir, level, list(_SAMPLE_INDEX.keys()))

            cmd = [
                "docker", "run", "--rm",
                "--platform", "linux/amd64",
                "-v", f"{training_dir}:/app/TrainingData",
                "-v", f"{output_dir}:/app/Output",
                DOCKER_IMAGE,
                "conda", "run", "--no-capture-output", "-n", "env",
                "python", "main.py",
                "TrainingData", "Output", str(level),
            ]
            _docker_run(cmd, timeout=self.timeout, stream=self.stream)

            return {
                sample: _load_reconstruction(output_dir, _SAMPLE_INDEX[sample])
                for sample in samples
            }


def _write_training_data(training_dir: Path, level: int, samples: list[str]) -> None:
    """Write subsampled measurement .mat files into training_dir for Docker."""
    from ktc_vis.data.loader import KTCDataLoader
    from ktc_vis.data.subsampler import subsample_measurement
    loader = KTCDataLoader()

    ref_src = RAW_DIR / "measurements" / "ref.mat"
    shutil.copy2(ref_src, training_dir / "ref.mat")

    for sample in samples:
        idx = _SAMPLE_INDEX[sample]
        m = loader.load(1, sample)
        if level != 1:
            m = subsample_measurement(m, level)
        scipy.io.savemat(
            str(training_dir / f"data{idx}.mat"),
            {"Inj": m.current_matrix.T, "Mpat": m.__dict__.get("mpat", np.zeros((32, 31))), "Uel": m.voltage_matrix.flatten()},
        )


def _load_reconstruction(output_dir: Path, sample_idx: int) -> np.ndarray:
    out_path = output_dir / f"data{sample_idx}.mat"
    if not out_path.exists():
        mat_files = list(output_dir.glob("*.mat"))
        if not mat_files:
            raise FileNotFoundError(f"CUQI8 container produced no output in {output_dir}")
        out_path = mat_files[0]

    data = scipy.io.loadmat(str(out_path))
    for value in data.values():
        if isinstance(value, np.ndarray) and value.ndim == 2:
            return value.astype(np.uint8)

    raise ValueError(f"No 2-D array found in CUQI8 output: {out_path}")


def _docker_run(cmd: list[str], timeout: int, stream: bool) -> None:
    if stream:
        subprocess.run(cmd, check=True, timeout=timeout)
    else:
        result = subprocess.run(
            cmd, check=False, capture_output=True, timeout=timeout
        )
        if result.returncode != 0:
            stderr = result.stderr.decode(errors="replace").strip()
            raise subprocess.CalledProcessError(result.returncode, cmd, stderr=stderr)
