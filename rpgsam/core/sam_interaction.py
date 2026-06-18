from __future__ import annotations

import cv2
import numpy as np
import torch

from rpgsam.utils.image_io import tensor_image_to_uint8_hwc
from rpgsam.utils.metrics import mask_metrics


def edt_center(mask: np.ndarray) -> tuple[int, int] | None:
    mask_u8 = np.asarray(mask).astype(np.uint8)
    if not np.any(mask_u8):
        return None
    dist = cv2.distanceTransform(mask_u8, cv2.DIST_L2, 5)
    _, _, _, max_loc = cv2.minMaxLoc(dist)
    return (int(max_loc[1]), int(max_loc[0]))


def epe_segmentation_sam(
    query_image: torch.Tensor,
    predictor,
    prior_mask: np.ndarray,
    max_iterations: int,
    cov_reth: float,
    iou_reth: float,
    sam_image_size: int = 1024,
) -> tuple[np.ndarray, list[dict]]:
    image_np = tensor_image_to_uint8_hwc(query_image, out_size=sam_image_size)
    predictor.set_image(image_np)
    prior = cv2.resize(
        prior_mask.squeeze().astype(np.uint8),
        dsize=(sam_image_size, sam_image_size),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)

    points_yx: list[tuple[int, int]] = []
    labels: list[int] = []
    history: list[np.ndarray] = []
    logs: list[dict] = []
    mask_i = np.zeros_like(prior, dtype=bool)
    iou_i = 0.0
    cov_i = 0.0

    for iteration in range(1, int(max_iterations) + 1):
        if cov_i < cov_reth:
            target = (~mask_i) & prior
            center = edt_center(target)
            label = 1
            step = "positive_prompt"
        else:
            target = mask_i & (~prior)
            center = edt_center(target)
            label = 0
            step = "negative_correction"
        if center is None:
            break

        points_yx.append(center)
        labels.append(label)
        point_coords = np.asarray(points_yx)[:, ::-1]
        point_labels = np.asarray(labels)
        masks, scores, _ = predictor.predict_points(point_coords, point_labels, multimask_output=False)
        mask_i = masks[0].astype(bool)
        iou_i, cov_i = mask_metrics(mask_i, prior)
        history.append(mask_i)
        logs.append(
            {
                "iteration": iteration,
                "step_type": step,
                "added_point_yx": center,
                "added_point_label": label,
                "score": float(scores[0]),
                "predicted_iou": iou_i,
                "coverage": cov_i,
                "output_mask": mask_i,
            }
        )
        if cov_i >= cov_reth and iou_i >= iou_reth:
            break

    if not history:
        return np.zeros_like(prior, dtype=bool), logs
    best = max(logs, key=lambda item: item["predicted_iou"])
    return best["output_mask"], logs
