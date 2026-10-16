# FGPT: Fine-Grained Prompt Tuning for Zero-Shot Anomaly Detection

Official implementation of **FGPT: Fine-Grained Prompt Tuning for Zero-Shot Anomaly Detection**.

**Yuexin Wang, Xiaoyang Wang, Haocheng Li, Jiejie Liu, Tammam Tillo, and Jimin Xiao**  
*IEEE Transactions on Artificial Intelligence*, vol. 7, no. 8, pp. 4340–4352, 2026.

[Paper](https://ieeexplore.ieee.org/abstract/document/11373225) · [DOI](https://doi.org/10.1109/TAI.2026.3661382) · [Repository](https://github.com/JasonKyng/FGPT)

## Overview

FGPT learns fine-grained prompts for zero-shot anomaly detection and localization. A shared prompt can struggle to describe the diverse visual patterns within an image, resulting in false activations. FGPT maintains normal and abnormal prompt pools and adapts their use to individual images and patches.

The framework consists of:

- **Image-level Prompt Identification:** dynamically selects expert prompts for each image and combines them with a general prompt.
- **Patch-level Prompt Matching:** matches the selected prompts to local visual features using **Entropy-aware Optimal Transport (EOT)**, which gives greater weight to informative image patches.
- **Prompt diversity regularization:** encourages expert prompts to capture distinct patterns through an orthogonality loss.

<img width="2278" height="1314" alt="image" src="https://github.com/user-attachments/assets/f8503257-c7c4-49f0-a659-9f9679172867" />

The pretrained vision-language backbone is frozen while the prompt learner is optimized using image classification, pixel segmentation, and orthogonality losses. Prompts use the object-agnostic descriptions `object` and `damaged object` to support transfer to unseen categories.

## Code layout

```text
FGPT/
├── CustomCLIP/
│   ├── clip.py                    # CLIP model loading
│   ├── model.py                   # CLIP with prompt learning and VV attention
│   ├── prompt_learner.py          # CLIP prompt pool
│   ├── prompt_learner_siglip.py   # SigLIP-family prompt pool
│   └── tokenizer.py
├── open_clip/                     # Bundled OpenCLIP implementation and model configs
├── utils/
│   ├── dataset.py                 # Metadata-based dataset loader
│   ├── entropy.py                 # Patch entropy estimation
│   ├── eot.py                     # Entropy-aware optimal transport and Sinkhorn solver
│   ├── loss.py                    # Orthogonality, focal, and Dice losses
│   ├── transform.py
│   └── utils.py
├── train.py                       # CLIP training entry point
├── train.sh                       # Example CLIP launch script
├── train_siglip.py                # SigLIP-family training entry point
├── train_siglip.sh                # Example SigLIP-family launch script
├── requirements.txt
└── LICENSE
```

## Environment

The paper reports experiments with **PyTorch 2.1.0** on a single **NVIDIA RTX 4090 (24 GB)**.

Python 3.10 is a suggested starting point for the older scientific Python dependencies. From the repository root, create an environment:

```bash
conda create -n fgpt python=3.10 -y
conda activate fgpt
```

For example, with a CUDA 11.8-compatible NVIDIA driver:

```bash
python -m pip install torch==2.1.0 torchvision==0.16.0 \
  --index-url https://download.pytorch.org/whl/cu118
```

## Data preparation

Download and extract the datasets separately. Each dataset root must contain a **`meta.json`** file, with image and mask paths relative to that root. No dataset or metadata-generation script is included in this snapshot.

The main source datasets use the following case-sensitive identifiers:

| Dataset | `--dataset` |
| --- | --- |
| MVTec AD | `mvtec` |
| VisA | `visa` |
| MVTec AD 2 | `MVTec2` |
| MPDD | `mpdd` |
| BTAD | `btad` |
| DAGM | `DAGM_KaggleUpload` |
| KolektorSDD | `SDD` |
| DTD | `DTD` |

Additional medical and Real-IAD class mappings are defined in `utils/dataset.py`. These mappings define accepted class names; they do not provide dataset conversion or evaluation pipelines.

An illustrative layout is:

```text
/path/to/dataset/
├── meta.json
└── bottle/
    ├── test/
    │   ├── good/000.png
    │   └── broken_large/000.png
    └── ground_truth/
        └── broken_large/000_mask.png
```

The corresponding minimal `meta.json` is:

```json
{
  "test": {
    "bottle": [
      {
        "img_path": "bottle/test/good/000.png",
        "mask_path": "",
        "cls_name": "bottle",
        "specie_name": "good",
        "anomaly": 0
      },
      {
        "img_path": "bottle/test/broken_large/000.png",
        "mask_path": "bottle/ground_truth/broken_large/000_mask.png",
        "cls_name": "bottle",
        "specie_name": "broken_large",
        "anomaly": 1
      }
    ]
  }
}
```

- `anomaly` is `0` for normal images and `1` for anomalous images.
- Normal masks are generated as zeros; anomalous samples should supply valid pixel masks for segmentation training. Positive mask pixels are interpreted as anomalies.
- All five record fields are read by the loader, including `specie_name`.
- Each `cls_name` must match the mapping for the selected dataset.
- For MVTec AD 2, the loader selects `test_public` when the dataset root's final path component is exactly `mvtec_ad_2`. Avoid a trailing slash for this special case.

### Zero-shot protocol

The supplied training scripts instantiate `Dataset` without overriding its default `mode='test'`, so they read **`meta.json["test"]` even during training**. This follows the auxiliary-data protocol described in the paper:

| Target evaluation dataset | Auxiliary prompt-training data |
| --- | --- |
| MVTec AD | Labeled test split of VisA |
| Other evaluated datasets | Labeled test split of MVTec AD |

Zero-shot refers to the **target datasets/categories**: their images and labels must not be used to train the prompt pool. Training still uses normal/anomalous labels and segmentation masks from the separate auxiliary dataset.

## Training

Run commands from the repository root after addressing the implementation issues below.

### CLIP backbone

`train.py` loads **CLIP ViT-L/14@336px** and uses a default input resolution of **518 × 518**. Its loader downloads pretrained backbone weights into `~/.cache/clip` when needed; these are distinct from trained FGPT prompt checkpoints.

Train on VisA for subsequent zero-shot evaluation on MVTec AD:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --dataset visa \
  --train_data_path /path/to/visa \
  --save_path ./checkpoints/fgpt_clip_visa \
  --num_n 5 \
  --num_a 5 \
  --top_k 2 \
  --epoch 30
```

Train on MVTec AD for transfer to the other target datasets:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --dataset mvtec \
  --train_data_path /path/to/mvtec_anomaly_detection \
  --save_path ./checkpoints/fgpt_clip_mvtec \
  --num_n 5 \
  --num_a 5 \
  --top_k 2 \
  --epoch 30
```

Alternatively, edit the dataset path and device in `train.sh`, then run `bash train.sh`.

### SigLIP-family backbone

Despite its filename, `train_siglip.py` currently selects **`ViT-L-16-SigLIP2-512`**, with `pretrained='webli'`. The backbone name is set inside the script, not through a command-line option. `ViT-L-16-SigLIP-384` appears as a commented alternative.

The intended SigLIP2 configuration is:

```bash
CUDA_VISIBLE_DEVICES=0 python train_siglip.py \
  --dataset visa \
  --train_data_path /path/to/visa \
  --save_path ./checkpoints/fgpt_siglip2_visa \
  --image_size 512 \
  --num_n 5 \
  --num_a 5 \
  --top_k 2 \
  --epoch 30
```

Use `--image_size 512` explicitly: the script inherits a default of `518`, which is not divisible by its 16-pixel patch size. The bundled `train_siglip.sh` also requires this adjustment. This branch needs additional API and preprocessing reconciliation before it can run; changing the resolution alone is insufficient.

## Citation

If you use FGPT in your research, please cite:

```bibtex
@article{wang2026fgpt,
  author  = {Wang, Yuexin and Wang, Xiaoyang and Li, Haocheng and Liu, Jiejie and Tillo, Tammam and Xiao, Jimin},
  title   = {{FGPT}: Fine-Grained Prompt Tuning for Zero-Shot Anomaly Detection},
  journal = {IEEE Transactions on Artificial Intelligence},
  year    = {2026},
  volume  = {7},
  number  = {8},
  pages   = {4340--4352},
  doi     = {10.1109/TAI.2026.3661382}
}
```

## Acknowledgements and license

FGPT builds on pretrained vision-language models and prompt-learning research, including CLIP, OpenCLIP, and AnomalyCLIP. We thank the authors of these methods and the dataset contributors.

This repository includes an [MIT license](LICENSE). Third-party code, pretrained weights, and datasets remain subject to their respective licenses and terms.
