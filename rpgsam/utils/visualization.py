from __future__ import annotations

import numpy as np
from PIL import Image


def overlay_mask(image: Image.Image, mask: np.ndarray, color=(255, 0, 0), alpha: float = 0.45) -> Image.Image:
    base = image.convert("RGBA")
    mask_bool = np.asarray(mask).astype(bool)
    overlay = np.zeros((mask_bool.shape[0], mask_bool.shape[1], 4), dtype=np.uint8)
    overlay[mask_bool] = [color[0], color[1], color[2], int(255 * alpha)]
    overlay_img = Image.fromarray(overlay, mode="RGBA").resize(base.size, Image.NEAREST)
    return Image.alpha_composite(base, overlay_img).convert("RGB")
