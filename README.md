# RPG-SAM

RPG-SAM is a training-free one-shot polyp segmentation repository built around DINOv2 visual features, region prototypes, prior-mask generation, and SAM2 point interaction.

The method uses one manually selected support image and support mask to build foreground/background prototypes. A query image is encoded by DINOv2, matched to weighted support prototypes, converted into a prior mask by threshold scanning with solidity-area selection, and refined with SAM2 interactive prompts. This public release only includes 2D polyp segmentation code.

## Installation

Tested environment:

- Python 3.10.0
- PyTorch 2.4.1+cu121
- torchvision 0.19.1+cu121
- CUDA 12.1
- GPU tested on NVIDIA RTX 3090

Use `torch.cuda.is_available()` as the source of truth for CUDA availability. On some servers, `nvidia-smi` or NVML may be unavailable even when PyTorch can use CUDA.

The CLI defaults to `--device auto`, which uses `cuda:0` when CUDA is visible and falls back to CPU only when no CUDA device is available. To force the same GPU style as the original experiment script, pass `--device cuda:0`.

```bash
git clone https://github.com/AlwaysHF/RPG-SAM.git
cd RPG-SAM
pip install -r requirements.txt
```

## External Dependencies

DINOv2 and SAM2 are external projects. This repository does not redistribute their full source code or model weights. Install the official repositories following their own instructions, or otherwise make their Python packages importable.

During development, DINOv2 and SAM2 were used as locally installed external dependencies. 
This repository does not depend on a specific vendored copy of either project. 
For public use, please install DINOv2 and SAM2 from their official repositories and provide the corresponding checkpoint/config paths.

Download the required DINOv2 and SAM2 checkpoints from their official release channels, then set:

- `paths.dinov2_checkpoint`
- `paths.sam2_checkpoint`
- `paths.sam2_config`

in YAML config files or pass them through CLI arguments.

## Dataset Layout

No medical datasets are redistributed. Place datasets locally and update `configs/datasets.yaml` or pass `--dataset-root`.

```text
Kvasir-SEG/
  images/*.jpg
  masks/*.jpg

CVC-ClinicDB/
  images/*.png
  masks/*.png

CVC-ColonDB/
  images/*.png
  masks/*.png

PolypGen2021_MultiCenterData_v3/
  data_C1/images/*.jpg
  data_C1/masks/*_mask.jpg
  data_C2/images/*.jpg
  data_C2/masks/*_mask.jpg
  data_C3/images/*.jpg
  data_C3/masks/*_mask.jpg
```

## Single Image Inference

```bash
python scripts/infer_single.py \
  --config configs/default.yaml \
  --support-image /path/to/support.jpg \
  --support-mask /path/to/support_mask.jpg \
  --query-image /path/to/query.jpg \
  --output outputs/pred_mask.png \
  --device cuda:0 \
  --dinov2-checkpoint /path/to/dinov2_vitl14_pretrain.pth \
  --sam2-checkpoint /path/to/sam2_hiera_large.pt \
  --sam2-config /path/to/sam2_hiera_l.yaml
```

`torch.compile` is disabled by default. To opt in:

```bash
python scripts/infer_single.py ... --compile-dinov2
```

## Dataset Evaluation

Support selection is manual and required. Random support selection and hard-coded top-1/top-10 support lists are not part of the public interface.

```bash
python scripts/evaluate.py \
  --config configs/eval_kvasir.yaml \
  --support-id cju0qkwl35piu0993l0dewei2 \
  --dataset-root /path/to/Kvasir-SEG \
  --device cuda:0 \
  --dinov2-checkpoint /path/to/dinov2_vitl14_pretrain.pth \
  --sam2-checkpoint /path/to/sam2_hiera_large.pt \
  --sam2-config /path/to/sam2_hiera_l.yaml
```

Evaluation writes `metrics.csv` and `summary.txt` under `paths.output_dir` or `--output-dir`.

## Python API

```python
from rpgsam import RPGSAMSegmentor

segmentor = RPGSAMSegmentor.from_config("configs/default.yaml", device="cuda:0")
mask = segmentor.predict(
    support_image="/path/to/support.jpg",
    support_mask="/path/to/support_mask.jpg",
    query_image="/path/to/query.jpg",
)
```

## Configuration

The main result configuration

- `clusters: slic`
- `pred: mine_with_weights`
- `thre: one_thre_with_sodality_area`
- `opsam_interact: mine_use_one_mask`
- `good: 10`, `bad: 0`, `p_runs: 10`
- `lamda: 1.0`, `temperture: 20`, `weight_gamma: 1`
- `thre_l: 0.4`, `thre_h: 0.7`, `thre_step: 0.05`
- `area_ref_perc: 50`, `area_alpha: 0.3`
- `cov_reth: 0.9`, `iou_reth: 0.8`, `max_iter: 5`
- `img_size: 560`, `patch_size: 14`, `batch_size: 1`

All paths are configured through YAML or CLI arguments. The code does not set `CUDA_VISIBLE_DEVICES`, does not reset CUDA memory stats at import time, and does not force CPU thread counts by default. When a CUDA device is selected, RPG-SAM sets the current CUDA device, enables TF32, and uses CUDA bfloat16 autocast for the inference path.

## Citation

```bibtex
@inproceedings{lin2026rpgsam,
  title     = {RPG-SAM: Reliability-Weighted Prototypes and Geometric Adaptive Threshold Selection for Training-Free One-Shot Polyp Segmentation},
  author    = {Lin, Weikun and Bai, Yunhao and Wang, Yan},
  booktitle = {International Conference on Medical Image Computing and Computer-Assisted Intervention},
  year      = {2026}
}
```

## License

This project is released under the Apache-2.0 License.

This repository does not redistribute medical datasets, SAM2 weights, DINOv2 weights, or external project source code.
