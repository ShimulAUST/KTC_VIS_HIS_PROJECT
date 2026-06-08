"""Metrics engine. Owner: Smit Savani."""

import time
from pathlib import Path

import numpy as np
import scipy.io

from ktc_vis.adapters.base import AlgorithmAdapter, KTCMeasurement
from ktc_vis.cache.hdf5_store import is_cached, load_result, save_result
from ktc_vis.metrics.class_metrics import compute_confusion_matrix, compute_mean_iou, compute_per_class_iou
from ktc_vis.metrics.image_quality import compute_ssim_stats
from ktc_vis.metrics.measurement import compute_resistance_consistency, compute_voltage_residual
from ktc_vis.metrics.shape_matching import compute_hausdorff, compute_position_error, compute_resolution

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DIR = _PROJECT_ROOT / "data" / "raw" / "ktc2023"
_SAMPLE_INDEX = {"a": 1, "b": 2, "c": 3, "d": 4}


class MetricsEngine:
    """Computes all metrics for a given (algorithm, level, sample) combination.

    Results are cached in HDF5 — second call returns from cache instantly.

    Args:
        adapter: The algorithm adapter to use for reconstruction.
        cache_path: Path to the HDF5 cache file.
    """

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
        """Compute (or load from cache) all metrics for one measurement.

        Args:
            measurement: Loaded KTC2023 measurement.
            overwrite: If True, recompute even if a cached result exists.

        Returns:
            Dict of metric_name → float value.
        """
        alg = self.adapter.name
        level = measurement.level
        sample = measurement.sample

        if not overwrite and is_cached(alg, level, sample, self.cache_path):
            metrics, _ = load_result(alg, level, sample, self.cache_path)
            return metrics

        gt = self._load_ground_truth(level, sample)

        # ── Reconstruct and time it ───────────────────────────────────────────
        t0 = time.perf_counter()
        reconstruction = self.adapter.reconstruct(measurement)
        runtime = time.perf_counter() - t0

        # ── Image quality ─────────────────────────────────────────────────────
        ssim, ssim_min = compute_ssim_stats(reconstruction, gt)

        # ── Class metrics ─────────────────────────────────────────────────────
        iou = compute_per_class_iou(reconstruction, gt)
        iou_mean = compute_mean_iou(reconstruction, gt)
        cm = compute_confusion_matrix(reconstruction, gt)
        confusion_accuracy = float(np.mean(np.diag(cm)))

        # ── Shape matching ────────────────────────────────────────────────────
        hausdorff = compute_hausdorff(reconstruction, gt)
        position_error = compute_position_error(reconstruction, gt)
        resolution = compute_resolution(reconstruction, gt)

        # ── Measurement domain ────────────────────────────────────────────────
        voltage_residual = compute_voltage_residual(measurement)
        resistance_consistency = compute_resistance_consistency(measurement)

        metrics = {
            "ssim": ssim,
            "ssim_min": ssim_min,
            "iou_water": iou["water"] if not np.isnan(iou["water"]) else 0.0,
            "iou_resistive": iou["resistive"] if not np.isnan(iou["resistive"]) else 0.0,
            "iou_conductive": iou["conductive"] if not np.isnan(iou["conductive"]) else 0.0,
            "iou_mean": iou_mean,
            "confusion_accuracy": confusion_accuracy,
            "hausdorff": hausdorff,
            "position_error": position_error,
            "resolution": resolution,
            "voltage_residual": voltage_residual,
            "resistance_consistency": resistance_consistency,
            "runtime": runtime,
        }

        # ── Save to cache ─────────────────────────────────────────────────────
        save_result(alg, level, sample, metrics, reconstruction, self.cache_path)

        return metrics

    def _compute_metrics_from_reconstruction(
        self,
        measurement: KTCMeasurement,
        reconstruction: np.ndarray,
        runtime: float,
    ) -> dict:
        """Compute all metrics given a pre-computed reconstruction array.

        Used by the batch benchmark path where reconstruct_level() already ran
        the algorithm — avoids calling the adapter a second time.
        """
        gt = self._load_ground_truth(measurement.level, measurement.sample)

        ssim, ssim_min = compute_ssim_stats(reconstruction, gt)
        iou = compute_per_class_iou(reconstruction, gt)
        iou_mean = compute_mean_iou(reconstruction, gt)
        cm = compute_confusion_matrix(reconstruction, gt)
        confusion_accuracy = float(np.mean(np.diag(cm)))
        hausdorff = compute_hausdorff(reconstruction, gt)
        position_error = compute_position_error(reconstruction, gt)
        resolution = compute_resolution(reconstruction, gt)
        voltage_residual = compute_voltage_residual(measurement)
        resistance_consistency = compute_resistance_consistency(measurement)

        return {
            "ssim": ssim,
            "ssim_min": ssim_min,
            "iou_water": iou["water"] if not np.isnan(iou["water"]) else 0.0,
            "iou_resistive": iou["resistive"] if not np.isnan(iou["resistive"]) else 0.0,
            "iou_conductive": iou["conductive"] if not np.isnan(iou["conductive"]) else 0.0,
            "iou_mean": iou_mean,
            "confusion_accuracy": confusion_accuracy,
            "hausdorff": hausdorff,
            "position_error": position_error,
            "resolution": resolution,
            "voltage_residual": voltage_residual,
            "resistance_consistency": resistance_consistency,
            "runtime": runtime,
        }

    @staticmethod
    def _load_ground_truth(level: int, sample: str) -> np.ndarray:
        idx = _SAMPLE_INDEX[sample]
        gt_path = RAW_DIR / "ground_truth" / f"true{idx}.mat"
        return scipy.io.loadmat(str(gt_path))["truth"].astype(np.uint8)
