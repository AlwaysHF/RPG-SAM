from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from rpgsam.config import load_config, require_path
from rpgsam.core.features import support_fg_flat
from rpgsam.core.prior import new_cpg_weighted, proto_weights_cr
from rpgsam.core.prototypes import extract_kmeans_centers, extract_slic_region_means
from rpgsam.core.sam_interaction import epe_segmentation_sam
from rpgsam.core.thresholding import generate_prior_slices, select_best_mask_sodality_area
from rpgsam.models import DINOv2Encoder, SAM2Wrapper
from rpgsam.utils.device import cuda_autocast, resolve_device
from rpgsam.utils.image_io import image_to_tensor, load_mask_pil, load_rgb_pil, mask_to_tensor


class RPGSAMSegmentor:
    def __init__(self, cfg: dict[str, Any], device: str | torch.device = "auto"):
        self.cfg = cfg
        self.device = resolve_device(device)
        self.use_cuda_autocast = bool(cfg.get("runtime", {}).get("cuda_autocast", True))

        model_cfg = cfg["model"]
        paths = cfg["paths"]
        self.img_size = int(model_cfg["img_size"])
        self.patch_size = int(model_cfg["patch_size"])
        self.sam_image_size = int(model_cfg.get("sam_image_size", 1024))

        self.dinov2 = DINOv2Encoder(
            checkpoint_path=require_path(cfg, "paths.dinov2_checkpoint"),
            device=self.device,
            img_size=self.img_size,
            patch_size=self.patch_size,
            arch=model_cfg.get("dinov2_arch", "vit_large"),
            compile_model=bool(model_cfg.get("compile_dinov2", False)),
        )
        self.sam2 = SAM2Wrapper(
            config_path=require_path(cfg, "paths.sam2_config"),
            checkpoint_path=require_path(cfg, "paths.sam2_checkpoint"),
            device=self.device,
        )
        self.last_prior: torch.Tensor | None = None

    @classmethod
    def from_config(
        cls,
        config_path: str | Path,
        device: str = "auto",
        overrides: dict[str, Any] | None = None,
    ) -> "RPGSAMSegmentor":
        cfg = load_config(config_path, overrides=overrides)
        return cls(cfg, device=device)

    def _prototypes(
        self,
        support_image: torch.Tensor,
        support_mask: torch.Tensor,
        support_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        method = self.cfg["method"]
        good = int(method["good"])
        bad = int(method["bad"])
        if method["clusters"] == "slic":
            fg = extract_slic_region_means(
                support_image,
                support_mask,
                support_features,
                n_segments=good + bad,
                patch_size=self.patch_size,
            )
            bg = extract_slic_region_means(
                support_image,
                1.0 - support_mask,
                support_features,
                n_segments=good + bad,
                patch_size=self.patch_size,
            )
            if fg is None:
                raise ValueError("Support mask has no foreground region after SLIC.")
            if bg is None:
                bg = support_features[:0]
            return fg, bg[:bad] if bad > 0 else bg[:0]

        if method["clusters"] == "kmeans":
            _, fg, _ = extract_kmeans_centers(
                support_features,
                support_mask,
                good_k=good,
                bad_k=bad,
                img_size=self.img_size,
                patch_size=self.patch_size,
                seed=int(method.get("kmeans_seed", 40)),
            )
            _, bg, _ = extract_kmeans_centers(
                support_features,
                1.0 - support_mask,
                good_k=good,
                bad_k=bad,
                img_size=self.img_size,
                patch_size=self.patch_size,
                seed=int(method.get("kmeans_seed", 40)),
            )
            return fg, bg[:bad] if bad > 0 else bg[:0]
        raise ValueError(f"Unsupported cluster method: {method['clusters']}")

    @torch.no_grad()
    def predict(
        self,
        support_image: str | Path,
        support_mask: str | Path,
        query_image: str | Path,
    ) -> np.ndarray:
        support_img_t = image_to_tensor(load_rgb_pil(support_image), self.img_size)
        support_mask_t = mask_to_tensor(load_mask_pil(support_mask), self.img_size)
        query_img_t = image_to_tensor(load_rgb_pil(query_image), self.img_size)

        with cuda_autocast(self.device, enabled=self.use_cuda_autocast):
            support_features = self.dinov2.extract(support_img_t)
            query_features = self.dinov2.extract(query_img_t)
            support_mask_dev = support_mask_t.to(self.device)

            fg_prototypes, bg_prototypes = self._prototypes(support_img_t, support_mask_dev, support_features)
            fg_flat = support_fg_flat(support_mask_dev, self.img_size, self.patch_size)
            weights = proto_weights_cr(fg_prototypes, support_features, fg_flat, query_features)

            method = self.cfg["method"]
            prior = new_cpg_weighted(
                fg_prototypes,
                bg_prototypes,
                query_features,
                p_runs=int(method["p_runs"]),
                temperature=float(method["temperture"]),
                good_weights=weights,
                img_size=self.img_size,
                patch_size=self.patch_size,
                lamda=float(method["lamda"]),
                weight_gamma=float(method["weight_gamma"]),
            )
            self.last_prior = prior.detach().cpu()

        if method["thre"] == "one_thre_with_sodality_area":
            prior_slices = generate_prior_slices(
                prior,
                min_thre=float(method["thre_l"]),
                max_thre=float(method["thre_h"]),
                step=float(method["thre_step"]),
                out_size=(self.sam_image_size, self.sam_image_size),
            )
            prior_mask = select_best_mask_sodality_area(
                prior_slices,
                area_ref_perc=float(method["area_ref_perc"]),
                area_alpha=float(method["area_alpha"]),
            )
        else:
            prior_mask = (prior.squeeze().detach().cpu().numpy() > float(method["thre_h"])).astype(np.uint8) * 255

        final_mask, _ = epe_segmentation_sam(
            query_img_t,
            predictor=self.sam2,
            prior_mask=prior_mask,
            max_iterations=int(method["max_iter"]),
            cov_reth=float(method["cov_reth"]),
            iou_reth=float(method["iou_reth"]),
            sam_image_size=self.sam_image_size,
        )
        return final_mask.astype(np.uint8) * 255
