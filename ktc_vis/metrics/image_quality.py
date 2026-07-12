# SSIM metrics.

import numpy as np
from skimage.metrics import structural_similarity


def compute_ssim(pred: np.ndarray, gt: np.ndarray) -> float:
    # Compute SSIM between predicted and ground-truth segmentation maps.
    p = pred.astype(np.float64)
    g = gt.astype(np.float64)
    score, _ = structural_similarity(p, g, full=True, data_range=2.0)
    return float(score)


def compute_spatial_ssim_map(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    # Compute pixel-level SSIM map between pred and ground truth.
    p = pred.astype(np.float64)
    g = gt.astype(np.float64)
    _, ssim_map = structural_similarity(p, g, full=True, data_range=2.0)
    return ssim_map.astype(np.float64)
