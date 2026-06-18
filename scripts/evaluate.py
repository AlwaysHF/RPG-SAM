#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rpgsam import RPGSAMSegmentor
from rpgsam.config import load_config
from rpgsam.data import PolypDataset, dataset_spec
from rpgsam.utils.image_io import load_mask_pil, mask_to_tensor, resize_mask_bool, save_mask
from rpgsam.utils.metrics import auc_pr, iou_and_dice
from rpgsam.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate RPG-SAM on a polyp segmentation dataset.")
    parser.add_argument("--config", required=True, help="Evaluation YAML config.")
    parser.add_argument("--support-id", required=True, help="Manual support sample id.")
    parser.add_argument("--dataset-root", default=None, help="Override dataset root.")
    parser.add_argument("--output-dir", default=None, help="Override output directory.")
    parser.add_argument("--device", default="auto", help="auto, cuda:0, or cpu. auto prefers cuda:0 when available.")
    parser.add_argument("--dinov2-checkpoint", default=None, help="Override DINOv2 checkpoint path.")
    parser.add_argument("--sam2-checkpoint", default=None, help="Override SAM2 checkpoint path.")
    parser.add_argument("--sam2-config", default=None, help="Override SAM2 model config path.")
    parser.add_argument("--compile-dinov2", action="store_true", help="Enable torch.compile for DINOv2.")
    parser.add_argument("--save-masks", action="store_true", help="Save predicted masks.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    overrides = {"paths": {}, "model": {}}
    if args.output_dir:
        overrides["paths"]["output_dir"] = args.output_dir
    if args.dinov2_checkpoint:
        overrides["paths"]["dinov2_checkpoint"] = args.dinov2_checkpoint
    if args.sam2_checkpoint:
        overrides["paths"]["sam2_checkpoint"] = args.sam2_checkpoint
    if args.sam2_config:
        overrides["paths"]["sam2_config"] = args.sam2_config
    if args.compile_dinov2:
        overrides["model"]["compile_dinov2"] = True

    cfg = load_config(args.config, overrides=overrides)
    set_seed(cfg.get("runtime", {}).get("seed"))

    key = cfg["dataset"]["key"]
    spec = dataset_spec(cfg, key)
    if args.dataset_root:
        spec["root"] = args.dataset_root
    dataset = PolypDataset(spec, max_samples=cfg["dataset"].get("max_samples"))
    support = dataset.get_by_id(args.support_id)

    output_dir = Path(cfg["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    masks_dir = output_dir / "masks"
    if args.save_masks:
        masks_dir.mkdir(parents=True, exist_ok=True)

    segmentor = RPGSAMSegmentor(cfg, device=args.device)
    print(f"Using device: {segmentor.device}")
    rows = []
    for sample in tqdm(dataset.samples, desc=f"Evaluating {spec['name']}"):
        if sample.sample_id == args.support_id:
            continue
        pred = segmentor.predict(support.image_path, support.mask_path, sample.image_path)
        gt = resize_mask_bool(mask_to_tensor(load_mask_pil(sample.mask_path), cfg["model"]["img_size"]), pred.shape)
        pred_bool = pred > 0
        iou, dice = iou_and_dice(pred_bool, gt)

        ap = None
        if segmentor.last_prior is not None:
            gt_560 = mask_to_tensor(load_mask_pil(sample.mask_path), cfg["model"]["img_size"])
            ap = auc_pr(segmentor.last_prior.squeeze(), gt_560)

        if args.save_masks:
            save_mask(pred, masks_dir / f"{sample.sample_id}.png")
        rows.append({"sample_id": sample.sample_id, "dice": dice, "iou": iou, "auc_pr": ap})

    if not rows:
        raise RuntimeError("No query samples evaluated. Check dataset size and support id.")

    auc_values = [r["auc_pr"] for r in rows if r["auc_pr"] is not None]
    summary = {
        "support_id": args.support_id,
        "dataset": spec["name"],
        "num_queries": len(rows),
        "mean_dice": sum(r["dice"] for r in rows) / len(rows),
        "mean_iou": sum(r["iou"] for r in rows) / len(rows),
        "mean_auc_pr": sum(auc_values) / len(auc_values) if auc_values else 0.0,
    }

    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["sample_id", "dice", "iou", "auc_pr"])
        writer.writeheader()
        writer.writerows(rows)

    with (output_dir / "summary.txt").open("w", encoding="utf-8") as f:
        for key_, value in summary.items():
            f.write(f"{key_}: {value}\n")

    print(
        f"{summary['dataset']} | support={summary['support_id']} | "
        f"Dice={summary['mean_dice']:.4f} | IoU={summary['mean_iou']:.4f} | AUC-PR={summary['mean_auc_pr']:.4f}"
    )


if __name__ == "__main__":
    main()
