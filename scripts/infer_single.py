#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rpgsam import RPGSAMSegmentor
from rpgsam.utils.image_io import save_mask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RPG-SAM on one support/query image pair.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--support-image", required=True, help="Support RGB image path.")
    parser.add_argument("--support-mask", required=True, help="Support binary mask path.")
    parser.add_argument("--query-image", required=True, help="Query RGB image path.")
    parser.add_argument("--output", required=True, help="Output mask path.")
    parser.add_argument("--device", default="auto", help="auto, cuda:0, or cpu. auto prefers cuda:0 when available.")
    parser.add_argument("--dinov2-checkpoint", default=None, help="Override DINOv2 checkpoint path.")
    parser.add_argument("--sam2-checkpoint", default=None, help="Override SAM2 checkpoint path.")
    parser.add_argument("--sam2-config", default=None, help="Override SAM2 model config path.")
    parser.add_argument("--compile-dinov2", action="store_true", help="Enable torch.compile for DINOv2.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    overrides = {"paths": {}, "model": {}}
    if args.dinov2_checkpoint:
        overrides["paths"]["dinov2_checkpoint"] = args.dinov2_checkpoint
    if args.sam2_checkpoint:
        overrides["paths"]["sam2_checkpoint"] = args.sam2_checkpoint
    if args.sam2_config:
        overrides["paths"]["sam2_config"] = args.sam2_config
    if args.compile_dinov2:
        overrides["model"]["compile_dinov2"] = True

    segmentor = RPGSAMSegmentor.from_config(args.config, device=args.device, overrides=overrides)
    print(f"Using device: {segmentor.device}")
    mask = segmentor.predict(args.support_image, args.support_mask, args.query_image)
    save_mask(mask, args.output)
    print(f"Saved prediction to {args.output}")


if __name__ == "__main__":
    main()
