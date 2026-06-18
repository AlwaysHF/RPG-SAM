from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


class SAM2Wrapper:
    def __init__(self, config_path: str | Path, checkpoint_path: str | Path, device: str | torch.device):
        self.device = torch.device(device)
        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except Exception as exc:
            raise ImportError(
                "SAM2 is not installed. Install the official SAM2 repository or make it importable on PYTHONPATH."
            ) from exc

        model = build_sam2(config_file=str(config_path), ckpt_path=str(checkpoint_path), device=self.device)
        self.predictor = SAM2ImagePredictor(model)

    def set_image(self, image_uint8: np.ndarray) -> None:
        self.predictor.set_image(image_uint8)

    def predict_points(
        self,
        point_coords_xy: np.ndarray,
        point_labels: np.ndarray,
        multimask_output: bool = False,
    ):
        return self.predictor.predict(
            point_coords=point_coords_xy.copy(),
            point_labels=point_labels.copy(),
            multimask_output=multimask_output,
        )
