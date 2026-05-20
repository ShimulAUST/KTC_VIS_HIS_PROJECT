"""PNPE2E algorithm adapter. Owner: Smit Savani.

Wraps Docker image: muzammal5566/ktc2023-pnpe2e:latest

Container run command:
    docker run --rm --platform linux/amd64 \
      -v "<level_dir>:/app/TrainingData" \
      -v "<output_dir>:/app/Output" \
      muzammal5566/ktc2023-pnpe2e:latest \
      python -c "from main import main; main('TrainingData', 'Output', <level>)"
"""

import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import scipy.io

from ktc_vis.adapters.base import AlgorithmAdapter, KTCMeasurement

DOCKER_IMAGE = "muzammal5566/ktc2023-pnpe2e:latest"
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DIR = _PROJECT_ROOT / "data" / "raw" / "ktc2023"

_SAMPLE_INDEX = {"a": 1, "b": 2, "c": 3}

# Weights were saved on CUDA; patch torch.load before importing main so it
# works on CPU-only machines (Mac, any server without a GPU).
_CPU_PATCH = (
    "import torch; _orig=torch.load; "
    "torch.load=lambda *a,**kw: _orig(*a,**{**kw,'map_location':'cpu'}); "
)


class PNPE2EAdapter(AlgorithmAdapter):
    """Adapter for the PNPE2E (physics-informed end-to-end) reconstruction algorithm."""

    name = "pnpe2e"
    supports_level_batching = True

    def reconstruct(self, measurement: KTCMeasurement) -> np.ndarray:
        results = self.reconstruct_level(measurement.level, [measurement.sample])
        return results[measurement.sample]

    def reconstruct_level(
        self, level: int, samples: list[str]
    ) -> dict[str, np.ndarray]:
        """Run PNPE2E once for the whole level and return results for all samples."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            training_dir = tmp_path / "TrainingData"
            output_dir = tmp_path / "Output"
            training_dir.mkdir()
            output_dir.mkdir()

            _write_training_data(training_dir, level, list(_SAMPLE_INDEX.keys()))

            python_call = (
                _CPU_PATCH
                + f"from main import main; main('TrainingData', 'Output', {level})"
            )
            cmd = [
                "docker", "run", "--rm",
                "--platform", "linux/amd64",
                "-v", f"{training_dir}:/app/TrainingData",
                "-v", f"{output_dir}:/app/Output",
                DOCKER_IMAGE,
                "python", "-c", python_call,
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
            raise FileNotFoundError(f"PNPE2E container produced no output in {output_dir}")
        out_path = mat_files[0]

    data = scipy.io.loadmat(str(out_path))
    for value in data.values():
        if isinstance(value, np.ndarray) and value.ndim == 2:
            return value.astype(np.uint8)

    raise ValueError(f"No 2-D array found in PNPE2E output: {out_path}")


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
