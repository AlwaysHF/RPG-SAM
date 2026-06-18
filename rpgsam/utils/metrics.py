from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import average_precision_score


def iou_and_dice(mask1: np.ndarray, mask2: np.ndarray) -> tuple[float, float]:
    a = np.asarray(mask1).astype(bool).squeeze()
    b = np.asarray(mask2).astype(bool).squeeze()
    if a.shape != b.shape:
        raise ValueError(f"Mask shape mismatch: {a.shape} vs {b.shape}")
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    total = a.sum() + b.sum()
    iou = float(inter / union) if union else 1.0
    dice = float((2.0 * inter) / total) if total else 1.0
    return iou, dice


def coverage(pred_mask: np.ndarray, target_mask: np.ndarray) -> float:
    pred = np.asarray(pred_mask).astype(bool)
    target = np.asarray(target_mask).astype(bool)
    target_area = target.sum()
    if target_area == 0:
        return 1.0
    return float(np.logical_and(pred, target).sum() / target_area)


def mask_metrics(pred_mask: np.ndarray, target_mask: np.ndarray) -> tuple[float, float]:
    pred = np.asarray(pred_mask).astype(bool)
    target = np.asarray(target_mask).astype(bool)
    inter = np.logical_and(pred, target).sum()
    union = pred.sum() + target.sum() - inter
    cov = float(inter / target.sum()) if target.sum() else 1.0
    iou = float(inter / union) if union else 1.0
    return iou, cov


def auc_pr(pred_heatmap: torch.Tensor | np.ndarray, gt_mask: torch.Tensor | np.ndarray) -> float:
    if torch.is_tensor(pred_heatmap):
        y_score = pred_heatmap.detach().cpu().numpy().flatten()
    else:
        y_score = np.asarray(pred_heatmap).flatten()
    if torch.is_tensor(gt_mask):
        y_true = gt_mask.detach().cpu().numpy().flatten()
    else:
        y_true = np.asarray(gt_mask).flatten()
    y_true = (y_true > 0.5).astype(np.int32)
    if y_true.max() == y_true.min():
        return float(y_true.max())
    return float(average_precision_score(y_true, y_score))
