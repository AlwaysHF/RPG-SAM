from __future__ import annotations

import torch


def support_fg_flat(support_mask: torch.Tensor, img_size: int, patch_size: int) -> torch.Tensor:
    feat_h = img_size // patch_size
    feat_w = img_size // patch_size
    pooled = torch.nn.functional.adaptive_avg_pool2d(
        support_mask.float().unsqueeze(0).unsqueeze(0), (feat_h, feat_w)
    ).squeeze()
    return (pooled > 0.5).flatten()
