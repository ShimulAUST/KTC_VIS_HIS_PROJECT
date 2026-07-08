# Shape matching metrics.

import numpy as np
from scipy.spatial.distance import directed_hausdorff


def compute_hausdorff(pred: np.ndarray, gt: np.ndarray) -> float:
    # Compute symmetric Hausdorff distance between foreground boundaries.

    pred_pts = np.argwhere(pred > 0)
    gt_pts = np.argwhere(gt > 0)

    if len(pred_pts) == 0 and len(gt_pts) == 0:
        return 0.0
    if len(pred_pts) == 0 or len(gt_pts) == 0:
        return float(max(pred.shape))

    d1 = directed_hausdorff(pred_pts, gt_pts)[0]
    d2 = directed_hausdorff(gt_pts, pred_pts)[0]
    return float(max(d1, d2))


def compute_position_error(pred: np.ndarray, gt: np.ndarray) -> float:
    # Compute centroid offset between predicted and ground-truth foreground.

    pred_pts = np.argwhere(pred > 0)
    gt_pts = np.argwhere(gt > 0)

    if len(pred_pts) == 0 or len(gt_pts) == 0:
        return float(max(pred.shape))

    pred_centroid = pred_pts.mean(axis=0)
    gt_centroid = gt_pts.mean(axis=0)
    return float(np.linalg.norm(pred_centroid - gt_centroid))


def compute_resolution(pred: np.ndarray, gt: np.ndarray) -> float:
    # Estimate the smallest detected inclusion diameter in pixels.
    # Returns: Estimated diameter in pixels. Lower means finer resolution.

    from skimage.measure import label, regionprops

    overlap = np.logical_and(pred > 0, gt > 0).astype(np.uint8)
    labeled = label(overlap)
    props = regionprops(labeled)
    if not props:
        return float(max(pred.shape))
    areas = [r.area for r in props]
    min_area = min(areas)
    return float(2 * np.sqrt(min_area / np.pi))
