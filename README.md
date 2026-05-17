# APEIRIA: Distilling Neuro-Symbolic Programs into 3D Multi-modal LLMs

[![Project Page](https://img.shields.io/badge/Project-Page-2f6f73)](https://matthewdm0816.github.io/Apeiria_Open/)
[![Paper](https://img.shields.io/badge/Paper-PDF-b31b1b)](https://matthewdm0816.github.io/Apeiria_Open/assets/paper.pdf)
[![Code](https://img.shields.io/badge/Code-GitHub-181717)](https://github.com/matthewdm0816/Apeiria_Open)
[![Model](https://img.shields.io/badge/Model-HuggingFace-ffcc4d)](https://huggingface.co/kmichiru/OpenApeiria)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776ab)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c)](https://pytorch.org/)
[![SGLang](https://img.shields.io/badge/SGLang-0.5.5%2B-6f42c1)](https://github.com/sgl-project/sglang)

This repository provides the official implementation of **APEIRIA**, a 3D multi-modal LLM framework that distills neuro-symbolic execution traces into transparent 3D chain-of-thought reasoning. The code supports supervised fine-tuning (SFT), GRPO-style reinforcement learning, SGLang-accelerated rollout generation, and ScanNet scene JSON preprocessing.

The project page is available at [matthewdm0816.github.io/Apeiria_Open](https://matthewdm0816.github.io/Apeiria_Open/). The released model checkpoint is hosted at [huggingface.co/kmichiru/OpenApeiria](https://huggingface.co/kmichiru/OpenApeiria).

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
| `sglang_0.5.5.post3.patch` | Reference patch for the SGLang version used in the original experiments. |
| `libs/capeval/` | Caption-evaluation utilities used by the dataset and evaluation code. |
| `data/` | Task annotations and scene metadata. Configure external dataset paths as described below. |

## Installation

The original experiments were run in an environment named `sgl055`. A typical setup is:

```bash
conda create -n sgl055 python=3.10
conda activate sgl055

# Install a CUDA-compatible PyTorch build first.
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

pip install transformers accelerate peft hydra-core omegaconf datasets wandb
pip install numpy scipy pandas scikit-learn tqdm h5py pillow filelock nltk
pip install lark lark-cython pretty-errors transforms3d sparsemax
pip install open-clip-torch tensordict packaging requests psutil nest-asyncio

# Optional but recommended for training/inference throughput.
pip install flash-attn sglang
```

Install `flash-attn` and SGLang with versions compatible with your CUDA, PyTorch, and GPU driver stack. Some clusters require site-specific builds.

## Data And Features

The code expects external datasets and precomputed features to be available through the following paths:

- `data/` to contain task annotations and scene JSON files.
- `data/scannet` to point to the ScanNet release directory. For example, on a shared cluster this may be `/network_space/server129/shared_dataset/scannet`.
- `../SVC` to exist as a sibling directory of this repository. It stores large 3D features and task data used by the dataset code.
- Proposal features under `../SVC/pc_features/chatscene_features/`, including:
  - `scannet_gt_trainval_feat+bbox_feats_200obj2d3d.pt`
  - `scannet_mask3d_trainval_feat+bbox_feats_200obj2d3d_nms0.975_noinvalid_combined.pt`

For a new machine, recreate the same layout with symlinks:

```bash
ln -s /path/to/scannet data/scannet
ln -s /path/to/SVC ../SVC
```

If you prefer a different layout, update `DATA_PATH` and `SVC_PATH` in `apeiria_lm_prog_to_thinking.py`.

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

## Supervised Fine-Tuning

SFT is launched through Hydra. The default config is `configs/apeiria_mllm.yaml`.

```bash
bash train_mllm.sh \
  model_name=/path/to/Qwen3-VL-4B-Instruct \
  resume_from_checkpoint=null \
  output_dir=outputs/sft \
  dataset_type='[scanrefer_nocot,nr3d_nocot,multi3drefer_nocot,sr3d_nocot]' \
  no_save=false
```

The provided config files are executable examples. For a fresh run, override at least `model_name`, `resume_from_checkpoint`, `output_dir`, `dataset_type`, and the batch-size settings.

## GRPO/RL Training

The RL stage uses asynchronous generation workers and can use SGLang for faster rollout generation. The default config is `configs/multimodal_grpo.yaml`.

```bash
bash train_grpo_mllm.sh \
  model_name=/path/to/Qwen3-VL-8B-Instruct \
  load_checkpoint=/path/to/sft_checkpoint \
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
  model_name=/path/to/Qwen3-VL-4B-Instruct \
  resume_from_checkpoint=/path/to/checkpoint \
  dataset_type=nr3d \
  output_dir=results/nr3d_rollouts \
  batch_size=16 \
  num_inference_passes=16 \
  use_sglang_for_generation=true
```

Supported dataset names are defined in `DATASET_CLSMAP` inside `train_apeiria_mllm.py` and the rollout script.

## SGLang Notes

The original experiments used a patched SGLang `0.5.5.post3` to support arbitrary feature/input-embedding MLLM inference. The patch is included as:

```bash
sglang_0.5.5.post3.patch
```

If you reproduce the exact environment, apply the patch to the matching SGLang source tree. Newer SGLang versions may already contain an equivalent fix through a different implementation path; prefer upstream SGLang when it works with APEIRIA's feature-input generation.

## Citation

If you find this work useful, please cite:

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

This repository is intended for academic research release. Please check the final public repository for the license file and dataset/model usage terms.
