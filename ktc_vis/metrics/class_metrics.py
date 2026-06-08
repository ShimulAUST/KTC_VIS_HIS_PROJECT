"""Class-specific metrics. Owner: Smit Savani."""

import numpy as np

CLASSES = {0: "water", 1: "resistive", 2: "conductive"}


def compute_per_class_iou(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    """Compute per-class Intersection over Union.

    Args:
        pred: 256×256 uint8 array with values in {0, 1, 2}.
        gt:   256×256 uint8 array with values in {0, 1, 2}.

    Returns:
        Dict with keys "water", "resistive", "conductive" → IoU float in [0, 1].
        Classes absent from both pred and gt are returned as NaN.
    """
    result = {}
    for cls_id, cls_name in CLASSES.items():
        p = pred == cls_id
        g = gt == cls_id
        intersection = np.logical_and(p, g).sum()
        union = np.logical_or(p, g).sum()
        result[cls_name] = float(intersection / union) if union > 0 else float("nan")
    return result


def compute_mean_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    """Compute mean IoU across classes present in the ground truth.

    Args:
        pred: 256×256 uint8 array.
        gt:   256×256 uint8 array.

    Returns:
        Mean IoU over present classes, in [0, 1].
    """
    per_class = compute_per_class_iou(pred, gt)
    valid = [v for v in per_class.values() if not np.isnan(v)]
    return float(np.mean(valid)) if valid else 0.0


def compute_per_class_dice(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    """Compute per-class Dice coefficient (F1 for segmentation).

    Dice = 2·TP / (2·TP + FP + FN) per class.

    Args:
        pred: 256×256 uint8 array with values in {0, 1, 2}.
        gt:   256×256 uint8 array with values in {0, 1, 2}.

    Returns:
        Dict with keys "water", "resistive", "conductive" → Dice float in [0, 1].
        Classes absent from both pred and gt are returned as NaN.
    """
    result = {}
    for cls_id, cls_name in CLASSES.items():
        p = pred == cls_id
        g = gt == cls_id
        tp = np.logical_and(p, g).sum()
        denom = p.sum() + g.sum()
        result[cls_name] = float(2 * tp / denom) if denom > 0 else float("nan")
    return result


def compute_mean_dice(pred: np.ndarray, gt: np.ndarray) -> float:
    """Compute mean Dice across classes present in pred or gt.

    Args:
        pred: 256×256 uint8 array.
        gt:   256×256 uint8 array.

    Returns:
        Mean Dice over present classes, in [0, 1].
    """
    per_class = compute_per_class_dice(pred, gt)
    valid = [v for v in per_class.values() if not np.isnan(v)]
    return float(np.mean(valid)) if valid else 0.0


def compute_confusion_matrix(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Compute 3×3 normalized confusion matrix (rows = GT, cols = pred).

    Args:
        pred: 256×256 uint8 array.
        gt:   256×256 uint8 array.

    Returns:
        3×3 float64 array where each row sums to 1 (percentage).
    """
    matrix = np.zeros((3, 3), dtype=np.float64)
    for gt_cls in range(3):
        mask = gt == gt_cls
        total = mask.sum()
        if total == 0:
            continue
        for pred_cls in range(3):
            matrix[gt_cls, pred_cls] = np.logical_and(mask, pred == pred_cls).sum() / total
    return matrix
