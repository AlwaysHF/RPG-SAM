from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from skimage.segmentation import slic


def extract_slic_region_means(
    image_tensor: torch.Tensor,
    support_mask: torch.Tensor,
    features_norm: torch.Tensor,
    n_segments: int,
    patch_size: int = 14,
    compactness: float = 20.0,
    sigma: float = 1.0,
) -> torch.Tensor | None:
    if image_tensor.dim() == 3:
        image_tensor = image_tensor.unsqueeze(0)
    _, _, h, w = image_tensor.shape
    feat_h, feat_w = h // patch_size, w // patch_size
    if features_norm.shape[0] != feat_h * feat_w:
        raise ValueError(f"Feature count {features_norm.shape[0]} does not match {feat_h}x{feat_w}.")

    image_np = image_tensor.squeeze(0).permute(1, 2, 0).detach().cpu().numpy().astype(np.float32)
    image_np = image_np / 255.0 if image_np.max() > 1.0 else image_np
    mask_np = support_mask.detach().cpu().squeeze().numpy() > 0.5
    if mask_np.sum() == 0:
        return None

    segments = slic(
        image_np,
        n_segments=int(n_segments),
        compactness=float(compactness),
        mask=mask_np,
        start_label=1,
        sigma=float(sigma),
        enforce_connectivity=True,
    )
    seg_t = torch.from_numpy(segments).unsqueeze(0).unsqueeze(0).float()
    seg_small = F.interpolate(seg_t, size=(feat_h, feat_w), mode="nearest").squeeze().long()

    device = features_norm.device
    seg_flat = seg_small.to(device=device).flatten()
    valid = seg_flat > 0
    if valid.sum() == 0:
        return None

    labels, inverse = torch.unique(seg_flat[valid], sorted=True, return_inverse=True)
    feats = features_norm[valid]
    out = torch.zeros((labels.numel(), feats.shape[1]), device=device, dtype=feats.dtype)
    counts = torch.zeros((labels.numel(), 1), device=device, dtype=feats.dtype)
    out.scatter_add_(0, inverse.unsqueeze(1).expand(-1, feats.shape[1]), feats)
    counts.scatter_add_(0, inverse.unsqueeze(1), torch.ones((inverse.numel(), 1), device=device, dtype=feats.dtype))
    return out / counts.clamp_min(1.0)


def extract_kmeans_centers(
    features_norm: torch.Tensor,
    support_mask: torch.Tensor,
    good_k: int,
    bad_k: int,
    img_size: int,
    patch_size: int,
    seed: int = 40,
    num_iters: int = 100,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    k = int(good_k + bad_k)
    feat_h = img_size // patch_size
    feat_w = img_size // patch_size
    mask_small = F.adaptive_max_pool2d(support_mask.float().unsqueeze(0).unsqueeze(0), (feat_h, feat_w)).squeeze()
    fg = features_norm[(mask_small > 0.5).flatten()]
    if fg.shape[0] == 0:
        raise ValueError("No foreground pixels available for KMeans.")

    n = fg.shape[0]
    centroids = fg[torch.randperm(n, device=fg.device)[:k]]
    for _ in range(num_iters):
        labels = torch.argmin(torch.cdist(fg.unsqueeze(0), centroids.unsqueeze(0)).squeeze(0), dim=1)
        new_centroids = []
        for i in range(k):
            pts = fg[labels == i]
            new_centroids.append(pts.mean(0) if pts.numel() else fg[torch.randint(0, n, (1,), device=fg.device)].squeeze(0))
        centroids = torch.stack(new_centroids, dim=0)

    mean_sim = (fg @ centroids.T).mean(dim=0)
    order = torch.argsort(mean_sim, descending=True)
    centroids = centroids[order]
    good = centroids[:good_k]
    bad = centroids[k - bad_k :] if bad_k > 0 else centroids[:0]
    return centroids, good, bad
