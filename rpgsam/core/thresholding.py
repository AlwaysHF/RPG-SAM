from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn.functional as F


def generate_prior_slices(pred: torch.Tensor, min_thre: float, max_thre: float, step: float, out_size: tuple[int, int]):
    if pred.dim() == 2:
        pred = pred.unsqueeze(0).unsqueeze(0)
    elif pred.dim() == 3:
        pred = pred.unsqueeze(0)
    thresholds = torch.arange(min_thre, max_thre + 1e-6, step, device=pred.device)
    priors = {}
    for thre in thresholds:
        prior = F.interpolate((pred > thre).float(), size=out_size, mode="nearest").bool().squeeze()
        priors[float(thre.item())] = prior
    return priors


def select_best_mask_sodality_area(mask_list: dict[float, torch.Tensor], area_ref_perc: float = 50, area_alpha: float = 0.3):
    if not mask_list:
        return np.zeros((1024, 1024), dtype=np.uint8)

    candidates = []
    areas_for_ref = []
    for thre in sorted(mask_list):
        mask_np = mask_list[thre].detach().cpu().numpy().astype(np.uint8)
        mask_u8 = mask_np * 255
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        areas = [cv2.contourArea(c) for c in contours]
        max_area = max(areas)
        if max_area <= 0:
            continue
        valid = [i for i, area in enumerate(areas) if area >= 0.2 * max_area]
        total_area = 0.0
        weighted_solidity = 0.0
        valid_contours = []
        for i in valid:
            contour = contours[i]
            area = float(areas[i])
            hull_area = float(cv2.contourArea(cv2.convexHull(contour)))
            solidity = area / hull_area if hull_area > 0 else 0.0
            weighted_solidity += solidity * area
            total_area += area
            valid_contours.append(contour)
        if total_area <= 0:
            continue
        clean = np.zeros_like(mask_u8)
        cv2.drawContours(clean, valid_contours, -1, 255, thickness=cv2.FILLED)
        candidates.append((float(thre), weighted_solidity / total_area, total_area, clean))
        areas_for_ref.append(total_area)

    if not candidates:
        any_mask = next(iter(mask_list.values()))
        h, w = any_mask.shape[-2], any_mask.shape[-1]
        return np.zeros((h, w), dtype=np.uint8)

    area_ref = float(np.percentile(areas_for_ref, area_ref_perc)) if areas_for_ref else 1.0
    area_ref = max(area_ref, 1.0)
    best_score = -1.0
    best_mask = candidates[0][3]
    for _, solidity, total_area, clean in candidates:
        area_factor = min(1.0, total_area / (area_ref + 1e-6)) ** float(area_alpha)
        score = float(solidity * area_factor)
        if score > best_score:
            best_score = score
            best_mask = clean
    return best_mask
