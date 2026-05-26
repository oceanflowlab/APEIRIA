# <img src="favicon.ico" style="height: 1.2em; width: auto; vertical-align: text-bottom;" alt="APEIRIA"> APEIRIA: Distilling Neuro-Symbolic Programs into 3D Multi-modal LLMs

[![Project Page](https://img.shields.io/badge/Project-Page-2f6f73)](https://matthewdm0816.github.io/Apeiria_Open)
[![Paper](https://img.shields.io/badge/Paper-PDF-b31b1b)](https://matthewdm0816.github.io/Apeiria_Open/assets/paper.pdf)
[![Code](https://img.shields.io/badge/Code-GitHub-181717)](./)
[![Model](https://img.shields.io/badge/Model-HuggingFace-ffcc4d)](https://huggingface.co/kmichiru/OpenApeiria)


This repository provides the official implementation of **APEIRIA** (ἄπειρον, *unlimited* in Greek), a 3D multi-modal LLM framework that distills neuro-symbolic execution traces into transparent 3D chain-of-thought reasoning. 

This work is accepted by ICML 2026.

## Contents

1. [Getting Started](#getting-started)
2. [Training](#training)
3. [Offline Rollouts And Inference](#offline-rollouts-and-inference)
4. [Release TODO](#release-todo)
5. [Citation](#citation)
6. [License](#license)



## Getting Started

This section walks through the environment and data setup needed to run APEIRIA.

### 1. Environment Setup

Configure the environment with the following commands, adjusting the PyTorch, FlashAttention, and SGLang installation for your platform as needed. SGLang is necessary for RL training and accelerated rollout generation. 

The recommended installation order is: create the environment, install `uv`, install PyTorch following the official PyTorch selector, install SGLang following the official SGLang guide, then install any remaining libraries.

```bash
conda create -n sgl055 python=3.12
conda activate sgl055

# Use uv for package management.
python -m pip install --upgrade pip
python -m pip install uv

# Example only; choose the command that matches your platform.
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Install SGLang using the official installation guide:
# https://docs.sglang.io/docs/get-started/install
uv pip install sglang==0.5.5.post3

# Install FlashAttention.
uv pip install flash-attn --no-build-isolation

# Install remaining APEIRIA dependencies if not already pulled by above commands.
uv pip install transformers accelerate peft hydra-core omegaconf datasets wandb
uv pip install numpy scipy pandas scikit-learn tqdm h5py pillow filelock nltk
uv pip install lark lark-cython pretty-errors transforms3d sparsemax
uv pip install open-clip-torch tensordict packaging requests psutil nest-asyncio
```

For exact package versions observed in the development environment, see `requirements-reference.txt`. Use that file only when you intentionally want to inspect or mirror the reference environment.

#### SGLang Patch

APEIRIA uses SGLang for feature-input MLLM generation during RL rollout and accelerated inference. The original experiments used SGLang `0.5.5.post3` with a small local patch for arbitrary feature/input-embedding generation at `sglang_0.5.5.post3.patch`. Without this patch, SGLang will raise [issue](https://github.com/sgl-project/sglang/issues/14109) when retracting decoding requests that would exceed reserved KV memory.

If you reproduce the exact environment, install `sglang==0.5.5.post3` and apply the patch to the matching SGLang source tree. 
Newer SGLang versions should have already contain an equivalent fix through a different implementation path; prefer upstream SGLang when it works with APEIRIA's feature-input generation.

### 2. Data Setup

#### Data, Model and Features Download

| Component | Link | Description |
| --- | --- | --- |
| APEIRIA checkpoint | [Download](https://huggingface.co/kmichiru/OpenApeiria) | Released APEIRIA model checkpoint (LoRA-only). |
| Qwen3-VL-4B/8B | [4B Download](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct)/[8B Download](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) | Base MLLM model. |
| 3D features | [Download](https://huggingface.co/kmichiru/OpenApeiria/blob/main/pc_features.zip) | Pre-extracted 3D features adapted from Chat-Scene |
| Data Compilation | [Download](https://huggingface.co/kmichiru/OpenApeiria/blob/main/data.zip) | Precompiled task annotations and scene JSON files. |
| ScanNet (Optional) | [Download](https://www.scan-net.org/) | Raw ScanNet data. Follow the official ScanNet terms of use. |

#### Organize Files

After downloading, unzip 3D features (`data.zip`) and data compilation (`pc_features.zip`). The code expects external datasets and precomputed features to be available through the following paths:

- `data/` to contain task annotations and scene JSON files.
- `data/pc_features/` for precomputed 3D proposal features:
  - `scannetv2-vote2cap-feature_box_features_281d.pkl`
  - `chatscene_features/scannet_gt_trainval_feat+bbox_feats_200obj2d3d.pt`
  - `chatscene_features/scannet_mask3d_trainval_feat+bbox_feats_200obj2d3d_nms0.975_noinvalid_combined.pt`
  - Move the downloaded 3D features to `data/pc_features/`. (Update the paths in the config if needed)
- (Optional) `data/scannet` to point to the ScanNet release directory .



The expected high-level layout is:

```text
<WORKSPACE_ROOT>/
|-- apeiria_open/
|   |-- README.md
|   |-- configs/
|   |-- data/
|   |   |-- scannet -> <SCANNET_ROOT>
|   |   |   |-- <scene0000_00> <scene0001_00>/ ... # raw ScanNet data for each scene
|   |   |-- meta_data/
|   |   |-- scannet_data
|   |   |-- apeiria_scannet_w_caption_gpt4o_and_corners_and_nyu_names_fixed_prec4
|   |   |   |-- <scene0000_00> <scene0001_00> ... .json
apeiria_scannet_w_caption_gpt4o_and_corners_and_nyu_names_fixed_prec4
|   |   |   |-- <scene0000_00> <scene0001_00> ... .json
|   |   |-- *.json              # task annotations
|   |   |-- pc_features/        # precomputed 3D features
|   |   |   |-- chatscene_features/
|   |   |   |   |-- scannet_gt_trainval_feat+bbox_feats_200obj2d3d.pt
|   |   |   |   |-- scannet_mask3d_trainval_feat+bbox_feats_200obj2d3d_nms0.975_noinvalid_combined.pt
|   |-- scannet/
|   |   |-- scans_fixed/  # preprocessed ScanNet point clouds and point labels
|   |-- train_mllm.sh
|   |-- train_grpo_mllm.sh
|-- Qwen3-VL-4B-Instruct/ # base MLLM
|-- Qwen3-VL-8B-Instruct/ # base MLLM
```

#### Scene JSON Generation

APEIRIA uses object-centric scene JSON files containing object IDs, boxes, positions, classes, and captions. 
To pre-process on your own or regenerate them from ScanNet bbox files:

```bash
python batch_generate_scene_json.py
```

Before running, check the paths at the top of `batch_generate_scene_json.py`, especially `scannet_dir`, `scannet_test_dir`, `output_dir`, and `data_dir`.


## Training

> **Note**: Training typically requires GPUs with at least 40GB of VRAM and 300-600GB of system RAM for 4-8 GPU runs. Results are saved to `<REPO_PARENT>/apeiria-output` by default. 
If no sufficient memory, SGLang worker might **quit siliently** and cause the whole training/inference to stall. 
Please log in to `wandb` to track metrics, or disable it with `wandb disabled`.

### Run Stage 1 and 2

Perception Alignment and CoT-SFT (Stage 1 and 2) is launched through Hydra. The default config is `configs/apeiria_mllm.yaml`.

```bash
./train_mllm.sh weight_decay=1e-3 optimizer=adamw \
  load_from_cache=false gradient_accumulation_steps=1 \
  object_embedding_type="discrete_location_separate" \
  eval_batch_size=32 batch_size=4 warmup_ratio=0.01 \
  no_save=false eval_use_sglang=false
```

Replace arguments with your desired settings such as model/datasets/batch_size/learning schedule.

### Run Stage 3

The CoT-RL stage (Stage 3) uses asynchronous generation workers and use SGLang for faster rollout generation. The default config is `configs/multimodal_grpo.yaml`.

```bash
./train_grpo_mllm.sh learning_rate=5e-6 optimizer_type=muon
```

Replace arguments with your desired settings such as model/datasets/batch_size/learning schedule.

## Offline Rollouts And Inference

Use `generate_trace_rollouts_offline_multigpu.py` to generate traces or evaluate a checkpoint:

```bash
python generate_trace_rollouts_offline_multigpu.py \
  resume_from_checkpoint=<CHECKPOINT_DIR> \
  dataset_type=scanrefer \
  output_dir=results/apeiria_rollouts \
  do_sample=false num_inference_passes=1 split=val \
  batch_size=256 no_save=false location_precision=4 \
  gt_scene_data_path="" previous_results_path="" \
```

`dataset_type` can be changed to `multi3drefer` or `scanrefer`.

#### Modular Enhancement

In APEIRIA, `scene()` execution in CoT can be replaced in-place by better perception modules. To do this, first generate SegDINO3D-based object info (we also have provided the pre-processed files):

```bash
# TODO
```

Then, run base and enhanced inference:

```bash
python generate_trace_rollouts_offline_multigpu.py \
  resume_from_checkpoint=<CHECKPOINT_DIR> \
  dataset_type=scanrefer \
  output_dir=results/apeiria_rollouts \
  do_sample=false num_inference_passes=1 split=val \
  batch_size=256 no_save=false location_precision=4 \
  gt_scene_data_path="" previous_results_path="" \
```

This step will run APEIRIA inference and save output to `results/apeiria_rollouts/<run_timestamp>/`. Then, update `previous_results_path` to point to the base APEIRIA inference results, and run again with the same command to generate enhanced rollouts with SegDINO3D object info.

```bash
python generate_trace_rollouts_offline_multigpu.py \
  resume_from_checkpoint=<CHECKPOINT_DIR> \
  dataset_type=scanrefer \
  output_dir=results/apeiria_rollouts \
  do_sample=false num_inference_passes=1 split=val \
  batch_size=256 no_save=false location_precision=4 \
  gt_scene_data_path="data/apeiria_scannet_w_caption_gpt4o_and_corners_and_nyu_names_fixed_mask3d_box_segdino3d_pred" \
  previous_results_path="results/apeiria_rollouts/<run_timestamp>/all_results.json" \
```

#### Replicate Results

| Model | ScanRefer (Acc@0.25/@0.50) | Multi3DRefer (F1@0.25/@0.50) |
| :--- | :--- | :--- |
| APEIRIA (reported) | 58.4/51.2 | 59.2/53.8 |
| APEIRIA (this code) | 58.1/51.1 | 59.5/54.1 |
| APEIRIA w/ Modular Enhancement (reported) | 60.5/53.2 | 60.9/55.2 |
| APEIRIA w/ Modular Enhancement (this code) | 60.5/53.1 | 61.0/55.4 |

## Release TODO

- [x] Upload the final public checkpoints and keep the Hugging Face model card synchronized with the training and inference configs.
- [x] Remove unused legacy code paths and simplify modules that still contain experiment-only utilities.
- [x] Add an exact-version reference dependency file for the development environment.
- [x] Run cleaned code to test checkpoint inference.
- [x] Add modular enhancement instruction.
- [ ] Add SegDINO3D-based object info generation code and instructions.
- [x] Add the final repository license and any third-party attribution notes required by bundled evaluation utilities.

## Citation

If you find this work useful, please consider cite us:

```bibtex
@inproceedings{mo2026,
  title={Distilling Neuro-Symbolic Programs into 3D Multi-modal LLMs},
  author={Mo, Wentao and Liu, Yang},
  booktitle={International Conference on Machine Learning},
  year={2026}
}
```

## Acknowledgements

This code builds upon the great work from previous 3D MLLMs and 3D foundation models, including but not limited to [Chat-Scene](github.com/ZzZZCHS/Chat-Scene), [SegDINO3D](https://github.com/IDEA-Research/SegDINO3D), [Mask3D](https://github.com/jonasschult/mask3d). Special thanks to the [SGLang]() library for fast multi-modal generation support to make the RL training possible. 

## License

This code repository and datasets are licensed under a [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/) license.
Please also follow the licenses and terms of the underlying datasets and pretrained models.

Copyright (c) 2026 Wentao Mo.
