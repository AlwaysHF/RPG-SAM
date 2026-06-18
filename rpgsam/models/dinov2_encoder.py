from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision import transforms


def interpolate_pos_embed(model: torch.nn.Module, checkpoint_pos_embed: torch.Tensor) -> torch.Tensor:
    cls_pos = checkpoint_pos_embed[:, :1, :]
    patch_pos = checkpoint_pos_embed[:, 1:, :]
    dim = patch_pos.shape[-1]

    old_grid = int(patch_pos.shape[1] ** 0.5)
    patch_pos = patch_pos.reshape(1, old_grid, old_grid, dim).permute(0, 3, 1, 2)

    new_grid = int(model.patch_embed.num_patches ** 0.5)
    patch_pos = F.interpolate(patch_pos, size=(new_grid, new_grid), mode="bicubic", align_corners=False)
    patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, new_grid * new_grid, dim)
    return torch.cat([cls_pos, patch_pos], dim=1)


class DINOv2Encoder:
    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str | torch.device,
        img_size: int = 560,
        patch_size: int = 14,
        arch: str = "vit_large",
        compile_model: bool = False,
    ):
        self.device = torch.device(device)
        self.img_size = int(img_size)
        self.patch_size = int(patch_size)
        self.normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

        try:
            from dinov2.models import vision_transformer as vits
        except Exception as exc:
            raise ImportError(
                "DINOv2 is not installed. Install the official DINOv2 repository or make it importable on PYTHONPATH."
            ) from exc

        if arch != "vit_large":
            raise ValueError("The public RPG-SAM config currently supports model.dinov2_arch=vit_large.")

        model = vits.vit_large(patch_size=self.patch_size, img_size=self.img_size, block_chunks=0, init_values=1.0)
        state_dict = torch.load(str(checkpoint_path), map_location="cpu")
        if "model" in state_dict and isinstance(state_dict["model"], dict):
            state_dict = state_dict["model"]
        if "pos_embed" in state_dict:
            state_dict["pos_embed"] = interpolate_pos_embed(model, state_dict["pos_embed"])
        model.load_state_dict(state_dict, strict=False)
        model.eval().to(self.device)
        if compile_model:
            model = torch.compile(model)
        self.model = model

    @torch.no_grad()
    def extract(self, image: torch.Tensor) -> torch.Tensor:
        if image.dim() == 3:
            image = image.unsqueeze(0)
        image = self.normalize(image.squeeze(0)).unsqueeze(0).to(self.device)
        features = self.model.get_intermediate_layers(image, n=1)[0].squeeze(0)
        return F.normalize(features, p=2, dim=1)
