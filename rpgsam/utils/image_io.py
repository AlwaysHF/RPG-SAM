from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms


def load_rgb_pil(path: str | Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def load_mask_pil(path: str | Path) -> Image.Image:
    return Image.open(path).convert("L")


def image_to_tensor(image: Image.Image, img_size: int) -> torch.Tensor:
    transform = transforms.Compose(
        [
            transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
        ]
    )
    return transform(image)


def mask_to_tensor(mask: Image.Image, img_size: int) -> torch.Tensor:
    transform = transforms.Compose(
        [
            transforms.Resize(img_size, interpolation=transforms.InterpolationMode.NEAREST),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
        ]
    )
    return (transform(mask).squeeze(0) > 0.5).float()


def tensor_image_to_uint8_hwc(image: torch.Tensor, out_size: int = 1024) -> np.ndarray:
    if image.dim() == 4:
        image = image.squeeze(0)
    arr = (image.detach().cpu().clamp(0, 1) * 255.0).permute(1, 2, 0).numpy()
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return cv2.resize(arr, dsize=(out_size, out_size), interpolation=cv2.INTER_CUBIC)


def resize_mask_bool(mask: np.ndarray | torch.Tensor, out_size: tuple[int, int]) -> np.ndarray:
    if torch.is_tensor(mask):
        arr = mask.detach().cpu().numpy()
    else:
        arr = np.asarray(mask)
    arr = arr.squeeze().astype(np.uint8)
    return cv2.resize(arr, dsize=(out_size[1], out_size[0]), interpolation=cv2.INTER_NEAREST).astype(bool)


def save_mask(mask: np.ndarray | torch.Tensor, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if torch.is_tensor(mask):
        mask = mask.detach().cpu().numpy()
    arr = (np.asarray(mask).squeeze() > 0).astype(np.uint8) * 255
    Image.fromarray(arr, mode="L").save(path)
