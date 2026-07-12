# Metrics engine.

import time
from pathlib import Path

import numpy as np
import scipy.io

from ktc_vis.adapters.base import AlgorithmAdapter, KTCMeasurement
from ktc_vis.cache.hdf5_store import is_cached, load_result, save_result
from ktc_vis.metrics.class_metrics import (
    compute_confusion_matrix,
    compute_mean_dice,
    compute_mean_iou,
    compute_per_class_dice,
    compute_per_class_iou,
)
from ktc_vis.metrics.image_quality import compute_spatial_ssim_map, compute_ssim
from ktc_vis.metrics.measurement import (
    compute_current_sensitivity_balance,
    compute_resistance_consistency,
    compute_voltage_residual,
)
from ktc_vis.metrics.shape_matching import (
    compute_hausdorff, compute_position_error, compute_resolution,
)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DIR = _PROJECT_ROOT / "data" / "raw" / "ktc2023"
_SAMPLE_INDEX = {"a": 1, "b": 2, "c": 3, "d": 4}


def _nan_to_zero(value: float) -> float:
    return 0.0 if np.isnan(value) else float(value)


class MetricsEngine:
    # Computes all metrics for a given (algorithm, level, sample) combination.

    def __init__(
        self,
        adapter: AlgorithmAdapter,
        cache_path: str | Path = "data/cache/results.h5",
    ) -> None:
        self.adapter = adapter
        self.cache_path = Path(cache_path)

    def compute_all(
        self,
        measurement: KTCMeasurement,
        overwrite: bool = False,
    ) -> dict:
        # Compute (or load from cache) all metrics for one measurement.

        alg = self.adapter.name
        level = measurement.level
        sample = measurement.sample

        if not overwrite and is_cached(alg, level, sample, self.cache_path):
            metrics, _ = load_result(alg, level, sample, self.cache_path)
            return metrics

        t0 = time.perf_counter()
        reconstruction = self.adapter.reconstruct(measurement)
        runtime = time.perf_counter() - t0

        metrics = self._compute_metrics_from_reconstruction(
            measurement, reconstruction, runtime
        )
        save_result(alg, level, sample, metrics, reconstruction, self.cache_path)
        return metrics

    def _compute_metrics_from_reconstruction(
        self,
        measurement: KTCMeasurement,
        reconstruction: np.ndarray,
        runtime: float,
    ) -> dict:
        # Compute the full metric battery given a pre-computed reconstruction.
        gt = self._load_ground_truth(measurement.level, measurement.sample)

        # Image quality
        ssim = compute_ssim(reconstruction, gt)
        ssim_map = compute_spatial_ssim_map(reconstruction, gt)
        ssim_min = float(np.min(ssim_map))

        # Class specific
        iou = compute_per_class_iou(reconstruction, gt)
        iou_mean = compute_mean_iou(reconstruction, gt)
        dice = compute_per_class_dice(reconstruction, gt)
        dice_mean = compute_mean_dice(reconstruction, gt)
        cm = compute_confusion_matrix(reconstruction, gt)
        confusion_accuracy = float(np.mean(np.diag(cm)))

        # Shape matching
        hausdorff = compute_hausdorff(reconstruction, gt)
        position_error = compute_position_error(reconstruction, gt)
        resolution = compute_resolution(reconstruction, gt)

        # Measurement domain — surrogate forward model uses the reconstruction
        voltage_residual = compute_voltage_residual(measurement, reconstruction)
        resistance_consistency = compute_resistance_consistency(
            measurement, reconstruction
        )
        current_sensitivity = compute_current_sensitivity_balance(
            measurement, reconstruction
        )

        return {
            "ssim": ssim,
            "ssim_min": ssim_min,
            "iou_water": _nan_to_zero(iou["water"]),
            "iou_resistive": _nan_to_zero(iou["resistive"]),
            "iou_conductive": _nan_to_zero(iou["conductive"]),
            "iou_mean": iou_mean,
            "dice_water": _nan_to_zero(dice["water"]),
            "dice_resistive": _nan_to_zero(dice["resistive"]),
            "dice_conductive": _nan_to_zero(dice["conductive"]),
            "dice_mean": dice_mean,
            "confusion_accuracy": confusion_accuracy,
            "hausdorff": hausdorff,
            "position_error": position_error,
            "resolution": resolution,
            "voltage_residual": voltage_residual,
            "resistance_consistency": resistance_consistency,
            "current_sensitivity": current_sensitivity,
            "runtime": runtime,
        }

    @staticmethod
    def _load_ground_truth(level: int, sample: str) -> np.ndarray:
        idx = _SAMPLE_INDEX[sample]
        gt_path = RAW_DIR / "ground_truth" / f"true{idx}.mat"
        return scipy.io.loadmat(str(gt_path))["truth"].astype(np.uint8)
