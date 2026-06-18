from __future__ import annotations

import os
from contextlib import nullcontext

import torch


def resolve_device(device: str | torch.device = "auto") -> torch.device:
    if isinstance(device, torch.device):
        requested = str(device)
    else:
        requested = str(device)

    if requested == "auto":
        requested = "cuda:0" if torch.cuda.is_available() else "cpu"

    resolved = torch.device(requested)
    if resolved.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"Requested device '{requested}', but torch.cuda.is_available() is False. "
                "Check CUDA/PyTorch visibility or run with --device auto/--device cpu."
            )
        torch.cuda.set_device(resolved)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    else:
        # Local DINOv2 uses xFormers when importable; xFormers attention is CUDA-only here.
        os.environ.setdefault("XFORMERS_DISABLED", "1")

    return resolved


def cuda_autocast(device: torch.device, enabled: bool = True):
    if enabled and device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()
