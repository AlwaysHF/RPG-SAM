from __future__ import annotations

import torch
import torch.nn.functional as F


def proto_weights_cr(
    centroids: torch.Tensor,
    support_features: torch.Tensor,
    support_fg_flat: torch.Tensor,
    query_features: torch.Tensor,
    q_top_ratio: float = 0.10,
    rev_top_ratio: float = 0.10,
    temp: float = 1.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    if centroids.dim() == 1:
        centroids = centroids.unsqueeze(0)
    device = support_features.device
    centroids = centroids.to(device=device, dtype=support_features.dtype)
    support_fg_flat = support_fg_flat.to(device=device).bool()
    support_bg_flat = ~support_fg_flat

    if support_fg_flat.sum() == 0 or support_bg_flat.sum() == 0:
        contrast = torch.ones(centroids.shape[0], device=device)
    else:
        sim_support = (support_features @ centroids.T) / float(temp)
        mu_fg = sim_support[support_fg_flat].mean(dim=0)
        mu_bg = sim_support[support_bg_flat].mean(dim=0)
        std = sim_support.std(dim=0, unbiased=False).clamp_min(eps)
        contrast = torch.relu((mu_fg - mu_bg) / std)

    n_query = query_features.shape[0]
    n_support = support_features.shape[0]
    top_q = max(1, min(int(n_query * q_top_ratio), n_query))
    top_r = max(1, min(int(n_support * rev_top_ratio), n_support))
    sim_query = (query_features @ centroids.T) / float(temp)
    idx_q = torch.topk(sim_query, k=top_q, dim=0).indices
    query_mu = query_features[idx_q.reshape(-1)].reshape(top_q, centroids.shape[0], -1).mean(dim=0)
    query_mu = F.normalize(query_mu, dim=1)

    sim_rev = (support_features @ query_mu.T) / float(temp)
    idx_r = torch.topk(sim_rev, k=top_r, dim=0).indices
    area_fg = support_fg_flat.float().mean()
    purity = support_fg_flat.float()[idx_r].mean(dim=0)
    reverse = ((purity - area_fg) / (1.0 - area_fg + eps)).clamp(0.0, 1.0)

    weights = (contrast * reverse).clamp_min(0.0)
    if weights.sum() <= eps:
        return torch.full_like(weights, 1.0 / max(1, weights.numel()))
    return weights / weights.sum()


def new_cpg_weighted(
    fg_prototypes: torch.Tensor,
    bg_prototypes: torch.Tensor,
    query_features: torch.Tensor,
    p_runs: int,
    temperature: float,
    good_weights: torch.Tensor,
    img_size: int,
    patch_size: int,
    lamda: float = 1.0,
    weight_gamma: float = 1.0,
) -> torch.Tensor:
    device = query_features.device
    hw = query_features.shape[0]
    feat_h = img_size // patch_size
    feat_w = img_size // patch_size

    weights = good_weights.to(device=device, dtype=torch.float32).flatten().clamp_min(0.0)
    if weights.numel() != fg_prototypes.shape[0]:
        raise ValueError("Prototype weight count does not match foreground prototype count.")
    if weights.sum() <= 1e-12:
        weights = torch.ones_like(weights)
    if weight_gamma != 1.0:
        weights = (weights + 1e-8) ** float(weight_gamma)
    weights = weights / (weights.sum() + 1e-9) * float(weights.numel())

    logits_sum = torch.zeros((hw,), device=device)
    for i, proto in enumerate(fg_prototypes):
        logits_sum += weights[i] * (query_features @ proto).flatten()
    prior = torch.softmax(logits_sum, dim=0).unsqueeze(1)

    for proto in bg_prototypes:
        sim = torch.softmax((query_features @ proto).flatten(), dim=0).view(hw, 1)
        prior -= float(lamda) * sim

    mn, mx = prior.min(), prior.max()
    prior = (prior - mn) / (mx - mn + 1e-6)

    self_corr = query_features @ query_features.T
    self_corr = F.softmax(self_corr * float(temperature), dim=1)
    for _ in range(int(p_runs)):
        prior = self_corr @ prior

    prior = prior.view(1, 1, feat_h, feat_w)
    prior = F.interpolate(prior, size=(img_size, img_size), mode="bilinear", align_corners=False)
    mn, mx = prior.min(), prior.max()
    return (prior - mn) / (mx - mn + 1e-6)
