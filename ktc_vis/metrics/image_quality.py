"""SSIM metrics. Owner: Muzammal / Smit."""

import numpy as np
from skimage.metrics import structural_similarity


def compute_ssim(pred: np.ndarray, gt: np.ndarray) -> float:
    """Compute SSIM between predicted and ground-truth segmentation maps.

    Args:
        pred: 256×256 uint8 array with values in {0, 1, 2}.
        gt:   256×256 uint8 array with values in {0, 1, 2}.

    Returns:
        SSIM score in [0, 1]. Higher is better.
    """
    p = pred.astype(np.float64)
    g = gt.astype(np.float64)
    score, _ = structural_similarity(p, g, full=True, data_range=2.0)
    return float(score)


def compute_ssim_stats(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float]:
    """Compute mean SSIM and minimum local SSIM in a single pass.

    Reuses the same structural_similarity call to avoid computing the full map
    twice (once for the scalar, once for the spatial map).

    Args:
        pred: 256×256 uint8 array with values in {0, 1, 2}.
        gt:   256×256 uint8 array with values in {0, 1, 2}.

    Returns:
        (mean_ssim, min_ssim) — mean is the overall score; min is the worst
        local window, indicating where the reconstruction fails most.
    """
    p = pred.astype(np.float64)
    g = gt.astype(np.float64)
    score, ssim_map = structural_similarity(p, g, full=True, data_range=2.0)
    return float(score), float(ssim_map.min())


def compute_spatial_ssim_map(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Compute pixel-level SSIM map between pred and ground truth.

    Args:
        pred: 256×256 uint8 array.
        gt:   256×256 uint8 array.

    Returns:
        256×256 float64 array of local SSIM values in [-1, 1].
    """
    p = pred.astype(np.float64)
    g = gt.astype(np.float64)
    _, ssim_map = structural_similarity(p, g, full=True, data_range=2.0)
    return ssim_map.astype(np.float64)
