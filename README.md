# <img src="icon2.png" style="height: 1.2em; width: auto; vertical-align: text-bottom;" alt="APEIRIA"> APEIRIA: Distilling Neuro-Symbolic Programs into 3D Multi-modal LLMs

[![Project Page](https://img.shields.io/badge/Project-Page-2f6f73)](https://matthewdm0816.github.io/Apeiria_Open)
[![Paper](https://img.shields.io/badge/Paper-PDF-b31b1b)](assets/paper.pdf)
[![Code](https://img.shields.io/badge/Code-GitHub-181717)](./)
[![Model](https://img.shields.io/badge/Model-HuggingFace-ffcc4d)](https://huggingface.co/kmichiru/OpenApeiria)


This repository provides the official implementation of **APEIRIA** (ἄπειρον, *unlimited* in Greek), a 3D multi-modal LLM framework that distills neuro-symbolic execution traces into transparent 3D chain-of-thought reasoning. 

This work is accepted by ICML 2026.

## Contents

1. [Repository Contents](#repository-contents)
2. [Getting Started](#getting-started)
3. [Scene JSON Generation](#scene-json-generation)
4. [Training](#training)
5. [Offline Rollouts And Inference](#offline-rollouts-and-inference)
6. [Release TODO](#release-todo)
7. [Citation](#citation)
8. [License](#license)

## Repository Contents

The repository is organized around the core APEIRIA training, inference, reward, and preprocessing components.

| Path | Purpose |
| --- | --- |
| `apeiria_mllm.py` | Main 3D MLLM wrapper, object-feature projection, generation, and SGLang integration. |
| `apeiria_mllm_config_schema.py` | Hydra config schema for SFT and inference. |
| `apeiria_lm_prog_to_thinking.py` | Real 3D dataset construction, program-to-CoT conversion, task formatting, and evaluation parsing. |
| `apeiria_lm_utils.py`, `apeiria_parser.py` | Synthetic data helpers, DSL parsing, and program utilities. |
| `train_apeiria_mllm.py` | SFT training entry point. |
| `train_apeiria_mllm_cot_rl_async.py` | Async GRPO/RL training entry point with SGLang generation workers. |
| `train_apeiria_mllm_cot_rl.py` | Shared RL utilities used by the async trainer. |
| `generate_trace_rollouts_offline_multigpu.py` | Multi-GPU inference and rollout generation for trained checkpoints. |
| `simple_filter_dataset_grpo.py` | Response parsing, grounding reward, format reward, and pass-at-k utilities. |
| `generate_scene_json_from_bbox_list.py`, `batch_generate_scene_json.py` | ScanNet bbox-to-scene-JSON preprocessing. |
| `configs/` | Hydra configs for SFT, inference, and GRPO. |
| `train_mllm.sh`, `train_grpo_mllm.sh` | Convenience launch scripts. |
| `requirements-reference.txt` | Curated exact package versions observed in the `sgl055` development environment, provided as reference only. |
| `sglang_0.5.5.post3.patch` | Reference patch for the SGLang version used in the original experiments. |
| `libs/capeval/` | Caption-evaluation utilities used by the dataset and evaluation code. |
| `data/` | Task annotations and scene metadata. Configure external dataset paths as described below. |

## Getting Started

This section walks through the environment and data setup needed to run APEIRIA.

### 1. Environment Setup

Configure the environment with the following commands, adjusting the PyTorch, FlashAttention, and SGLang installation for your platform as needed. SGLang is necessary for RL training and accelerated rollout generation. The recommended installation order is: create the environment, install `uv`, install PyTorch following the official PyTorch selector, install SGLang following the official SGLang guide, then install any remaining libraries.

`requirements-reference.txt` records exact versions for APEIRIA-relevant packages observed in the `sgl055` development environment. It is provided for reference and debugging only; these exact versions are not required for all users or platforms. Prefer the official installation commands for platform-sensitive packages.

```bash
conda create -n sgl055 python=3.10
conda activate sgl055

# Use uv for package management.
python -m pip install --upgrade pip
python -m pip install uv

# Install PyTorch, torchvision, and torchaudio using the official selector:
# https://pytorch.org/get-started/locally/
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

APEIRIA uses SGLang for feature-input MLLM generation during RL rollout and accelerated inference. The original experiments used SGLang `0.5.5.post3` with a small local patch for arbitrary feature/input-embedding generation:

```bash
sglang_0.5.5.post3.patch
```
Without this patch, SGLang will raise [issue](https://github.com/sgl-project/sglang/issues/14109) when retracting decoding requests that would exceed reserved KV memory.
If you reproduce the exact environment, install `sglang==0.5.5.post3` and apply the patch to the matching SGLang source tree. 
Newer SGLang versions should have already contain an equivalent fix through a different implementation path; prefer upstream SGLang when it works with APEIRIA's feature-input generation.

### 2. Data Setup

#### Download Files

Download the required components and arrange them using the structure below.

| Component | Link | Description |
| --- | --- | --- |
| APEIRIA checkpoint | [Download](https://huggingface.co/kmichiru/OpenApeiria) | Released APEIRIA model checkpoint. |
| Qwen3-VL-4B/8B-Instruct | [4B Download](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct)/[8B Download](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) | Base MLLM model for SFT/RL. |
| SVC data and features | [Download](https://huggingface.co/datasets/kmichiru/SVC) | Preprocessed 3D features, task annotations, and auxiliary files. |
| ScanNet | [Apply / Download](https://www.scan-net.org/) | Raw ScanNet data. Follow the official ScanNet terms of use. |

#### Organize Files

The code expects external datasets and precomputed features to be available through the following paths:

- `data/` to contain task annotations and scene JSON files.
- `data/scannet` to point to the ScanNet release directory.
- `../SVC` to exist as a sibling directory of this repository. It stores large 3D features and task data used by the dataset code.
- Proposal features under `../SVC/pc_features/chatscene_features/`, including:
  - `scannet_gt_trainval_feat+bbox_feats_200obj2d3d.pt`
  - `scannet_mask3d_trainval_feat+bbox_feats_200obj2d3d_nms0.975_noinvalid_combined.pt`

For a new machine, recreate the same layout with symlinks:

```bash
ln -s <SCANNET_ROOT> data/scannet
ln -s <SVC_ROOT> ../SVC
```

If you prefer a different layout, update `DATA_PATH` and `SVC_PATH` in `apeiria_lm_prog_to_thinking.py`.

The expected high-level layout is:

```text
<WORKSPACE_ROOT>/
|-- apeiria_open/
|   |-- README.md
|   |-- configs/
|   |-- data/
|   |   |-- scannet -> <SCANNET_ROOT>
|   |   |-- meta_data/
|   |   |-- scannet_data/
|   |   |-- multi3drefer/
|   |   |-- mmscan-obj-desc/
|   |   |-- msqa/
|   |   |-- apeiria_scannet_w_caption_gpt4o_and_corners_and_nyu_names_fixed_prec4/
|   |   |-- *.json
|   |-- scannet/
|   |   |-- scans -> <SCANNET_SCANS>
|   |   |-- scans_fixed/
|   |-- train_mllm.sh
|   |-- train_grpo_mllm.sh
|-- SVC/
|   |-- pc_features/
|   |   |-- chatscene_features/
|   |   |   |-- scannet_gt_trainval_feat+bbox_feats_200obj2d3d.pt
|   |   |   |-- scannet_mask3d_trainval_feat+bbox_feats_200obj2d3d_nms0.975_noinvalid_combined.pt
|   |-- ...
|-- Qwen3-VL-4B-Instruct/
|-- Qwen3-VL-8B-Instruct/
```

## Scene JSON Generation

APEIRIA uses object-centric scene JSON files containing object IDs, boxes, positions, classes, and captions. To regenerate them from ScanNet bbox files:

```bash
python batch_generate_scene_json.py
```

Before running, check the paths at the top of `batch_generate_scene_json.py`, especially `scannet_dir`, `scannet_test_dir`, `output_dir`, and `data_dir`.

For a single scene/bbox file:

```bash
python generate_scene_json_from_bbox_list.py \
  -i scannet/scans_fixed/scene0000_00_aligned_bbox.npy \
  -o data/apeiria_scannet_w_caption_gpt4o_and_corners_and_nyu_names_fixed_prec4/scene0000_00.json \
  -c data/scene_object_top_captions_from_gpt4o.json \
  -p 4
```

## Training

> **Note**: Training typically requires GPUs with at least 40GB of VRAM and 300-600GB of system RAM for 4-8 GPU runs. Results are saved to `<REPO_PARENT>/apeiria-output` by default. Please log in to `wandb` to track metrics, or disable it with `wandb disabled`.

### Supervised Fine-Tuning

SFT is launched through Hydra. The default config is `configs/apeiria_mllm.yaml`.

```bash
bash train_mllm.sh \
  model_name=<QWEN3_VL_4B_MODEL> \
  resume_from_checkpoint=null \
  output_dir=outputs/sft \
  dataset_type='[scanrefer_nocot,nr3d_nocot,multi3drefer_nocot,sr3d_nocot]' \
  no_save=false
```

The provided config files are executable examples. For a fresh run, override at least `model_name`, `resume_from_checkpoint`, `output_dir`, `dataset_type`, and the batch-size settings.

### GRPO/RL Training

The RL stage uses asynchronous generation workers and can use SGLang for faster rollout generation. The default config is `configs/multimodal_grpo.yaml`.

```bash
bash train_grpo_mllm.sh \
  model_name=<QWEN3_VL_8B_MODEL> \
  load_checkpoint=<SFT_CHECKPOINT_DIR> \
  dataset_type=scanrefer \
  num_inference_gpus=4 \
  num_training_gpus=4 \
  use_sglang_for_generation=true \
  hydra.run.dir=outputs/grpo/run1
```

`simple_filter_dataset_grpo.py` contains the response parser and reward components used by the GRPO trainer. The most important knobs are `num_generations`, `rollout_batch_size`, `generation_micro_batch_size`, `max_completion_length`, `temperature`, and the reward shaping fields in `configs/multimodal_grpo.yaml`.

## Offline Rollouts And Inference

Use `generate_trace_rollouts_offline_multigpu.py` to generate traces or evaluate a checkpoint:

```bash
python generate_trace_rollouts_offline_multigpu.py \
  model_name=<QWEN3_VL_4B_MODEL> \
  resume_from_checkpoint=<CHECKPOINT_DIR> \
  dataset_type=nr3d \
  output_dir=results/nr3d_rollouts \
  batch_size=16 \
  num_inference_passes=16 \
  use_sglang_for_generation=true
```

Supported dataset names are defined in `DATASET_CLSMAP` inside `train_apeiria_mllm.py` and the rollout script.

## Release TODO

- [ ] Upload the final public checkpoints and keep the Hugging Face model card synchronized with the training and inference configs.
- [x] Remove unused legacy code paths and simplify modules that still contain experiment-only utilities.
- [x] Add an exact-version reference dependency file for the development environment.
- [ ] Run cleaned code to test checkpoint inference.
- [ ] Add modular enhancement instruction.
- [x] Add the final repository license and any third-party attribution notes required by bundled evaluation utilities.

## Citation

If you find this work useful, please consider cite us:

```bibtex
@inproceedings{mo2026,
  title={Distilling Neuro-Symbolic Programs into 3D Multi-modal LLMs},
  author={Mo, Wentao and Liu, Yang},
  % booktitle={International Conference on Machine Learning},
  year={2026}
}
```

## Acknowledgements

This code builds on the open-source ecosystems around PyTorch, Hugging Face Transformers, PEFT, Hydra, SGLang, Qwen, ScanNet, ScanRefer, ReferIt3D, Multi3DRefer, ScanQA, and SQA3D. Please also follow the licenses and terms of the underlying datasets and pretrained models.

## License

This code repository and datasets are licensed under a [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/) license.

Copyright (c) 2026 Wentao Mo.
