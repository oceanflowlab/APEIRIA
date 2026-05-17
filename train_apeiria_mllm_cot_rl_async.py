import warnings

# 过滤FutureWarning
warnings.filterwarnings('ignore', category=FutureWarning)

# 过滤pydantic的警告
warnings.filterwarnings('ignore', message='.*UnsupportedFieldAttributeWarning.*')

import os
import random
import copy
import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.utils.rnn import pad_sequence
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup, Adafactor
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from accelerate import Accelerator, DistributedType, InitProcessGroupKwargs, DistributedDataParallelKwargs
from accelerate.logging import get_logger
from accelerate.utils import set_seed, gather_object
import wandb
import logging
import datetime
from icecream import ic
import contextlib
from tqdm.auto import tqdm
import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.core.config_store import ConfigStore
from dataclasses import dataclass, field
from typing import List, Dict, Union, Optional, Tuple, Callable
from omegaconf import MISSING, OmegaConf
import sys
import shutil
import json
from traceback import print_exc
from multiprocessing import set_start_method
from datetime import timedelta
import transformers
import gc
from types import SimpleNamespace
import pandas as pd
import time
import gc
from muon import Muon

set_start_method("spawn", force=True)

# Import the multimodal model and dataset classes
from dist_tools import all_gather_vlen, all_gather_vdim, model_to_device
from apeiria_mllm import MultimodalLanguageModelDecoderOnly, find_free_port, NEW_SGLANG
from apeiria_lm_utils import MergedDataset
from apeiria_lm_prog_to_thinking import Real3DDataset, Real3DObjectInfoDataset, Real3DFilterDataset
import simple_filter_dataset_grpo
from simple_filter_dataset_grpo import combined_reward, parse_response, calculate_position_similarity, calculate_size_similarity, CombinedReward, pass_at_k
from qwen_helpers import apply_qwen_template, count_example_tokens, batch_count_tokens
from train_apeiria_mllm import create_dataset
from cot_rl_config import MultimodalGRPOConfig

from train_apeiria_mllm_cot_rl import (
    DATASET_CLSMAP,
    print_once,
    set_random_seed,
    recursive_to_device,
    collate_fn_simple,
    collate_fn,
    prepare_model_inputs,
    setup_distributed,
    evaluate_model,
)

from liger import LIGER_KERNEL_AVAILABLE, apply_liger_kernel_to_qwen3, apply_liger_kernel_to_qwen2, apply_liger_kernel_to_qwen3_vl

# logger = get_logger(__name__)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def print_mem(step_name):
    torch.cuda.synchronize()
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    max_allocated = torch.cuda.max_memory_allocated() / 1024**3
    print(f"[{step_name}] Alloc: {allocated:.2f} GB | Reserved: {reserved:.2f} GB | Max Alloc: {max_allocated:.2f} GB")

def apply_selective_checkpointing(model, step=2):
    """
    Enables gradient checkpointing for every 'step' layers.
    step=1: Full checkpointing (slowest, lowest memory)
    step=2: Checkpoint every 2nd layer (faster, medium memory)
    """
    # Assuming Qwen2/3-VL structure: model.model.layers
    layers = model.model.language_model.layers 
    
    # 1. Enable GC generally (sets flags)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    
    # 2. Manually disable it for specific layers
    for i, layer in enumerate(layers):
        if i % step != 0:
            # Disable GC for this specific layer instance
            # Note: This relies on the layer class having a `gradient_checkpointing` attribute 
            # which transformers models usually respect.
            if hasattr(layer, "gradient_checkpointing"):
                layer.gradient_checkpointing = False
            
    logger.info(f"Applied selective gradient checkpointing with step {step}")

# code from trl
def entropy_from_logits(logits: torch.Tensor, chunk_size: int = 128) -> torch.Tensor:
    """
    Compute the Shannon entropy (in nats) for each row of *logits* in a memory-efficient way.

    Instead of materializing the full softmax for all rows at once, the logits are flattened to shape (N, num_classes),
    where N is the product of all leading dimensions. Computation is then performed in chunks of size `chunk_size`
    along this flattened dimension, reducing peak memory usage. The result is reshaped back to match the input's
    leading dimensions.

    Args:
        logits (`torch.Tensor`):
            Logits tensor of shape `(..., num_classes)`. Entropy is taken along the last axis; all leading dimensions
            are preserved in the output.
        chunk_size (`int`, *optional*, defaults to `128`):
            Number of rows from the flattened logits to process per iteration. Smaller values reduce memory usage at
            the cost of more iterations.

    Returns:
        `torch.Tensor`:
            Entropy values with shape `logits.shape[:-1]`.
    """
    original_shape = logits.shape[:-1]  # all dims except num_classes
    num_classes = logits.shape[-1]

    # Flatten all leading dimensions into one
    flat_logits = logits.reshape(-1, num_classes)

    entropies = []
    for chunk in flat_logits.split(chunk_size, dim=0):
        logps = F.log_softmax(chunk, dim=-1)
        chunk_entropy = -(torch.exp(logps) * logps).sum(-1)
        entropies.append(chunk_entropy)

    entropies = torch.cat(entropies, dim=0)
    return entropies.reshape(original_shape)

class MultimodalDistributedGRPO:
    def __init__(self, config: MultimodalGRPOConfig, accelerator: Accelerator):
        self.config = config
        self.accelerator = accelerator
        

        # Determine process role
        self.world_size = self.accelerator.num_processes
        self.process_idx = self.accelerator.process_index
        self.training_process_idx = self.process_idx - config.num_inference_gpus
        self.is_inference_process = self.process_idx < config.num_inference_gpus
        self.is_training_process = not self.is_inference_process
        self.is_main_process = self.accelerator.is_main_process
        self.is_main_inference_process = self.process_idx == 0
        self.is_main_training_process = self.process_idx == config.num_inference_gpus

        print(f"Rank {config.rank} - Process index: {self.process_idx}, World size: {self.world_size}, Inference process: {self.is_inference_process}, Training process: {self.is_training_process}, Main process: {self.is_main_process}, device: {self.accelerator.device}")

        # self.prompt_continue("Before initializing model and tokenizer...")

        # Create inference and training process groups
        self.inference_ranks = list(range(config.num_inference_gpus))
        self.training_ranks = list(range(config.num_inference_gpus, 
                                        config.num_inference_gpus + config.num_training_gpus))
        
        self.inference_group = dist.new_group(ranks=self.inference_ranks)
        self.training_group = dist.new_group(ranks=self.training_ranks)
        self.group = self.inference_group if self.is_inference_process else self.training_group
        
        if self.world_size != (config.num_inference_gpus + config.num_training_gpus):
            raise ValueError(f"Total processes ({self.world_size}) must equal inference GPUs ({config.num_inference_gpus}) + training GPUs ({config.num_training_gpus})")

        logger.debug(f"Process index: {self.process_idx}, Inference process: {self.is_inference_process}, Training process: {self.is_training_process}, Main process: {self.is_main_process}")
        
        # Setup logging levels
        self.setup_logging()
        
        # Initialize model and tokenizer
        self.setup_model_and_tokenizer()
        
        # Initialize optimizer and scheduler if this is a training process
        if self.is_training_process:
            self.setup_optimizer_and_scheduler()

        # self.prompt_continue()
        
        if self.is_training_process:
            self.model = self.model.to(self.accelerator.device)
            self.model.language_encoder.config.use_cache = False  # we don't generate, so we don't need cache in training

            if self.config.compile_train_model:
                logger.info(f"Rank {self.process_idx} - Compiling model")
                self.model = torch.compile(self.model)

            self.model = DDP(self.model, process_group=self.training_group, find_unused_parameters=True)
            self.model.train()

            # Setup reference model
            self.update_ref_model()

        else:
            if not self.config.use_sglang_for_generation:
                # Use DDP-based inference
                self.model = self.model.to(self.accelerator.device)
                self.model.language_encoder.config.use_cache = True  # we need cache in inference
                if self.config.compile_model:
                    self.lm.generation_config.cache_implementation = "static"

                self.model = DDP(self.model, process_group=self.inference_group, find_unused_parameters=True)

                if self.config.compile_model:
                    logger.info(f"Rank {self.process_idx} - Compiling model")
                    # self.model.forward = torch.compile(self.model.forward, mode="reduce-overhead", fullgraph=True)
                    self.lm.forward = torch.compile(self.lm.forward, mode="reduce-overhead", fullgraph=True)

            else:
                # move non-lm part to GPU 
                # for name, model in self.model.named_modules():
                #     if "language_encoder" not in name:
                #         model.to(self.accelerator.device)
                self.model.object_proj = self.model.object_proj.to(self.accelerator.device)

        # self.prompt_continue()
        self.accelerator.wait_for_everyone()

        # Initialize wandb if this is the main process
        if self.is_main_training_process:
            wandb.init(
                project=self.config.wandb_project,
                name=self.config.wandb_run_name,
                # config=OmegaConf.to_container(self.config, resolve=True),
                config=self.config,
            )

    def prompt_continue(self, prompt: Optional[str]=None):
        if self.is_main_process:
            input(prompt or "Press Enter to continue...")
        
        self.accelerator.wait_for_everyone()

    def setup_logging(self):
        """Setup logging levels based on process role"""
        level_map = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL,
        }
        
        transformers_verbosity_map_func = {
            "debug": transformers.utils.logging.set_verbosity_debug,
            "info": transformers.utils.logging.set_verbosity_info,
            "warning": transformers.utils.logging.set_verbosity_warning,
            "error": transformers.utils.logging.set_verbosity_error,
        }

        if self.is_main_inference_process or self.is_main_training_process:
            # Set logging level for main train/inference process
            log_level_name = self.config.log_level_main.upper()
            transformers_verbosity_name = self.config.transformers_verbosity_main.lower()
            
            log_level = level_map.get(log_level_name, logging.INFO)
            transformers_verbosity_func = transformers_verbosity_map_func.get(
                transformers_verbosity_name, transformers.utils.logging.set_verbosity_info
            )
            
            logging.basicConfig(
                format=self.config.log_format, 
                level=log_level, 
                datefmt=self.config.log_datefmt,
                force=True
            )
            transformers_verbosity_func()
        else:
            log_level_name = self.config.log_level_other.upper()
            transformers_verbosity_name = self.config.transformers_verbosity_other.lower()

            log_level = level_map.get(log_level_name, logging.WARNING)
            transformers_verbosity_func = transformers_verbosity_map_func.get(
                transformers_verbosity_name, transformers.utils.logging.set_verbosity_warning
            )

            logging.basicConfig(
                format=self.config.log_format, 
                level=log_level, 
                datefmt=self.config.log_datefmt,
                force=True
            )
            transformers_verbosity_func()

    def apply_chat_template(self, prompt: str):
        """
        Apply chat template to the input string.
        """
        prompt_messages = [
            {"role": "system", "content": self.config.system_prompt},
            {"role": "user", "content": prompt}
        ]
        prompt = self.tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
        return prompt


    @property
    def unwrapped_model(self) -> MultimodalLanguageModelDecoderOnly:
        # return self.model.module
        return self.accelerator.unwrap_model(self.model)

    def update_ref_model(self):
        """
        Initialize reference model (only for training processes)
        """
        if self.is_training_process:
            logger.debug(f"Rank {self.process_idx} - Updating reference model")
            # if no need to use ref_model - beta=0, just skip
            if self.config.beta == 0.0:
                logger.info(f"Rank {self.process_idx} - Skipping reference model update since beta=0.0")
                self.ref_model = None
                return

            self.ref_model = copy.deepcopy(self.unwrapped_model)
            self.ref_model.eval()
            for param in self.ref_model.parameters():
                param.requires_grad = False

            if self.config.offload_reference_model:
                # Offload reference model to CPU, and load to GPU when used, and move back when not used
                logger.info(f"Rank {self.process_idx} - Offloading reference model to CPU")
                self.ref_model = self.ref_model.to("cpu")
                gc.collect()
                torch.cuda.empty_cache()

            logger.debug(f"Rank {self.process_idx} - Reference model updated")

    def setup_model_and_tokenizer(self):
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name, padding_side="left")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load language model
        if "Qwen3-VL" in self.config.model_name:
            # For models like Qwen3-VL, use AutoModelForConditionalGeneration.from_pretrained
            language_model = transformers.Qwen3VLForConditionalGeneration.from_pretrained(
                self.config.model_name,
                torch_dtype=torch.bfloat16,
                attn_implementation=self.config.attn_implementation,
                device_map=self.process_idx,
            )
        else:
            language_model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                torch_dtype=torch.bfloat16,
                attn_implementation=self.config.attn_implementation,
                device_map=self.process_idx,
            )

        # shall set fused_linear_cross_entropy=False => otherwise logits is not outputted, and is directly used to compute loss
        # but we don't input label here, so won't trigger fused loss computation.
        if LIGER_KERNEL_AVAILABLE:
            if "qwen3-vl" in self.config.model_name.lower():
                logger.info(f"Applying liger kernel to Qwen3-VL model")
                apply_liger_kernel_to_qwen3_vl(language_model)

            elif "qwen3" in self.config.model_name.lower():
                logger.info(f"Applying liger kernel to Qwen3 model")
                apply_liger_kernel_to_qwen3(language_model)

            elif "qwen2" in self.config.model_name.lower():
                logger.info(f"Applying liger kernel to Qwen2/Qwen2.5 model")
                apply_liger_kernel_to_qwen2(language_model)

        language_model.config.pad_token_id = self.tokenizer.eos_token_id
        language_model.config.eos_token_id = self.tokenizer.eos_token_id

        # Apply LoRA if enabled
        if self.config.use_lora:
            lora_config = LoraConfig(
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                # target_modules=OmegaConf.to_container(self.config.lora_target_modules, resolve=True),
                target_modules=self.config.lora_target_modules,
                bias="none",
                task_type="CAUSAL_LM"
            )
            if self.is_training_process:
                language_model = get_peft_model(language_model, lora_config)

                # self.prompt_continue("Before loading LoRA checkpoint...")
                
                if self.config.load_checkpoint:
                    # FIXME: loading on GPU maybe cause all process take up GPU memory on main rank
                    #   Possible fix: load on CPU, then move to GPU after all adapter is loaded
                    #   However, pissa init is much much faster on GPU, but if pissa init, no need to load adapter, separate the code.

                    logger.info(f"Loading LoRA checkpoint from {self.config.load_checkpoint}...")
                    # Only load adapter on training processes. For inference processes, we will load adapter in vLLM or SGLang
                    message = language_model.load_adapter(
                        model_id=self.config.load_checkpoint,
                        adapter_name="default",
                        is_trainable=True,
                        torch_device="cpu",
                        low_cpu_mem_usage=True,
                    )
                    logger.info(message)

            # put the adapter to GPU
            language_model = language_model.to("cpu").to(self.accelerator.device)

            logger.critical(f"Rank {self.process_idx} - language model is on device: {next(language_model.parameters()).device}")
            
            if self.is_main_training_process:
                language_model.print_trainable_parameters()

            # self.prompt_continue("After loading LoRA checkpoint...")

        # self.lm = self.model.language_encoder
        self.lm = language_model
        
        # Create the multimodal model
        # First, determine the feature dimension from the dataset
        # temp_dataset = create_dataset(self.config.dataset_type, self.config, tokenizer=self.tokenizer, split="train")
        # feature_dim = temp_dataset.feature_dim

        if self.config.use_sglang_for_generation and self.is_inference_process:
            sglang_kwargs = {
                "use_sglang": True,
                "sglang_model_path": self.config.model_name,
                "sglang_lora_paths": [self.config.lora_update_path],
                "sglang_port": self.config.sglang_port,
            }
        else:
            sglang_kwargs = {
                "use_sglang": False,
            }

        # save LoRA first 
        if self.config.use_lora and self.config.use_sglang_for_generation:
            if self.is_main_process:
                self.initial_copy_lora_for_sglang()
            
            self.accelerator.wait_for_everyone()
        
        # Initialize the multimodal model
        self.model = MultimodalLanguageModelDecoderOnly(
            language_model=language_model,
            tokenizer=self.tokenizer,
            object_feature_dim=self.config.feature_dim,
            max_objects=self.config.max_objects,
            no_object_in_language_model=self.config.no_object_in_language_model,
            object_embedding_type=self.config.object_embedding_type,
            discrete_location_bins=self.config.discrete_location_bins,
            discrete_location_decay_kernel=self.config.discrete_location_decay_kernel,
            discrete_location_bin_range=self.config.discrete_location_bin_range,
            discrete_location_decay_kernel_size=self.config.discrete_location_decay_kernel_size,
            separate_location_embedding=self.config.separate_location_embedding,
            dtype=torch.bfloat16,
            modality_dims=self.config.modality_dims,
            modality_order=self.config.modality_order,
            sglang_log_level=self.config.sglang_log_level,
            **sglang_kwargs,
        )


        if self.config.load_checkpoint:
            # Load the model checkpoint, the non_lm part
            self.model.load_non_lm_parameters(self.config.load_checkpoint)

        if self.is_main_process:
            num_params = 0
            for name, param in self.model.named_parameters():
                if param.requires_grad and "language_encoder" not in name:
                    num_params += param.numel()
            logger.info(f"Trainable parameters outside language model: {num_params:,d}")
            logger.info(str(self.model))
        
        # Optimize model memory usage in training processes
        if self.is_training_process:
            self.model.language_encoder.config.use_cache = False
            if self.config.use_gradient_checkpointing:
                self.model.language_encoder.gradient_checkpointing_enable({"use_reentrant": False})
                # apply_selective_checkpointing(self.model.language_encoder, step=2)
                if hasattr(self.lm, "enable_input_require_grads"):
                    self.model.language_encoder.enable_input_require_grads()
                else:
                    def make_inputs_require_grad(module, input, output):
                        output.requires_grad_(True)
                    self.model.language_encoder.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

            # for Qwen3-VL, offload vision encoder - since we don't use them.
            if "qwen3-vl" in self.config.model_name.lower():
                logger.info(f"Rank {self.process_idx} - Offloading vision encoder for Qwen3-VL")
                self.model.language_encoder.model.visual = self.model.language_encoder.model.visual.to("cpu")

    def setup_optimizer_and_scheduler(self):
        # Create optimizer
        if self.config.optimizer_type == 'adamw':
            # 创建AdamW优化器
            self.optimizer = torch.optim.AdamW(
                [params for params in self.model.parameters() if params.requires_grad],
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                fused=True,
            )
        
        elif self.config.optimizer_type == 'adafactor':
            # 创建Adafactor优化器 - 注意Adafactor不需要scheduler
            self.optimizer = Adafactor(
                [params for params in self.model.parameters() if params.requires_grad],
                scale_parameter=False, relative_step=False, warmup_init=False,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )

        elif self.config.optimizer_type == "muon":
            lm_params: dict[str, nn.Parameter] = {}
            non_lm_params: dict[str, nn.Parameter] = {}
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    if "language_encoder" in name:
                        lm_params[name] = param
                    else:
                        non_lm_params[name] = param

            muon_params = [
                (name, p)
                for name, p in lm_params.items()
                if p.ndim >= 2 and "embed_tokens" not in name and "lm_head" not in name
            ]
            adamw_params = [
                (name, p)
                for name, p in lm_params.items()
                if not (
                    p.ndim >= 2 and "embed_tokens" not in name and "lm_head" not in name
                )
            ] + [
                (name, p) for name, p in non_lm_params.items()
            ]

            # show what are optimized with Muon, what with adamw:
            if self.is_main_training_process:
                logger.info("Parameters optimized with Muon:".center(50, "="))
                for name, p in muon_params:
                    logger.info(f"{name}: {p.shape}")
                logger.info("Parameters optimized with AdamW:".center(50, "="))
                for name, p in adamw_params:
                    logger.info(f"{name}: {p.shape}")

            # take only the parameters
            muon_params = [p for name, p in muon_params]
            adamw_params = [p for name, p in adamw_params]

            self.optimizer = Muon(
                lr=self.config.learning_rate,
                wd=self.config.weight_decay,
                muon_params=muon_params,
                adamw_params=adamw_params,
            )
        
        else:
            raise ValueError(f"不支持的优化器类型: {self.config.optimizer_type}")
        
        # Create learning rate scheduler
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=self.config.num_steps * self.config.num_iterations // self.config.gradient_accumulation_steps
        ) # we don't use prepare, so we need to manually set the total steps
        
        # Prepare optimizer with accelerator
        self.optimizer = self.accelerator.prepare(self.optimizer)
    
    def create_completion_mask(self, completion_ids, eos_token_id):
        """
        Creates a mask for completion tokens that excludes tokens after the EOS token.

        Args:
            completion_ids (torch.Tensor): Token IDs of the generated completions.
            eos_token_id (int): The ID of the end-of-sequence token.

        Returns:
            torch.Tensor: A binary mask with 1s for valid tokens and 0s after the EOS token.

        Explanation:
            1. Identifies positions where EOS tokens occur in each sequence.
            2. Finds the index of the first EOS token in each sequence.
            3. Creates a mask where positions before and including the first EOS are 1, others are 0.
            4. If no EOS token is found in a sequence, all positions are set to 1.
        """
        is_eos = completion_ids == eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=completion_ids.device)
        mask_exists = is_eos.any(dim=1)
        eos_idx[mask_exists] = is_eos.int().argmax(dim=1)[mask_exists]
        sequence_indices = torch.arange(is_eos.size(1), device=completion_ids.device).expand(is_eos.size(0), -1)
        # NOTE: using long() to avoid dtype mismatch in all_gather and the hang from it!!!
        return (sequence_indices <= eos_idx.unsqueeze(1)).long()
    
    def generate_completions(self, batch, num_generations, max_completion_length):
        """
        Generate completions for a batch of multimodal inputs.
        Each input is repeated `num_generations` times to generate multiple completions.
        Therefore, each GPU will generate `rollout_batch_size * num_generations // num_inference_gpus` completions.
        """
        # Only inference processes generate completions
        if not self.is_inference_process:
            logger.debug(f"Rank {self.process_idx} - Returning empty completions")
            return (
                torch.zeros((len(batch["instructions"]), 1), dtype=torch.bfloat16, device=self.accelerator.device), 
                torch.zeros((len(batch["instructions"]), 1), dtype=torch.long, device=self.accelerator.device), 
                torch.zeros((len(batch["instructions"]), 1), dtype=torch.long, device=self.accelerator.device), 
                torch.zeros((len(batch["instructions"]), 1), dtype=torch.long, device=self.accelerator.device),
                torch.zeros((len(batch["instructions"]), 1), dtype=torch.float32, device=self.accelerator.device),  # seq_logprobs_sum
                torch.zeros((len(batch["instructions"]), 1), dtype=torch.float32, device=self.accelerator.device),  # seq_logprobs_mean
            )
        
        # Calculate how many samples each inference process should handle
        prompts: list[str] = batch["instructions"]
        object_features: torch.Tensor = batch["object_features"]
        object_masks = batch["object_masks"]
        
        # Repeat each prompt and object features num_generations times
        repeated_prompts = []
        repeated_object_features = []
        repeated_object_masks = []
        
        for i in range(len(prompts)):
            for _ in range(num_generations):
                repeated_prompts.append(prompts[i])
                repeated_object_features.append(object_features[i])
                repeated_object_masks.append(object_masks[i])
        
        assert len(repeated_prompts) % self.config.num_inference_gpus == 0, f"Number of prompts ({len(repeated_prompts)}) must be divisible by the number of inference GPUs ({self.config.num_inference_gpus})"
        local_batch_size = len(repeated_prompts) // self.config.num_inference_gpus
        start_idx = self.process_idx * local_batch_size
        end_idx = min(start_idx + local_batch_size, len(repeated_prompts))
        
        # Create local batch
        local_prompts = repeated_prompts[start_idx:end_idx]
        local_object_features = repeated_object_features[start_idx:end_idx]
        local_object_masks = repeated_object_masks[start_idx:end_idx]
        
        
        # prompt_length = prompt_ids.size(1)
        all_completion_ids = []
        all_inputs_embeds = []
        all_inputs_mask = []

        # Prepare object features for the model
        object_set_embeds = []
        for features, mask in zip(local_object_features, local_object_masks):
            valid_features = features[mask]
            object_set_embeds.append([valid_features])

        # Use micro-batching to reduce memory usage
        for i in range(0, len(local_prompts), self.config.generation_micro_batch_size):
            batch_end = min(i + self.config.generation_micro_batch_size, len(local_prompts))
            
            batch_prompts = local_prompts[i:batch_end]
            # batch_prompt_ids = prompt_ids[i:batch_end]
            # batch_prompt_mask = prompt_mask[i:batch_end]
            batch_object_embeds = object_set_embeds[i:batch_end]

            # Generate completions
            with torch.no_grad():
                outputs = self.unwrapped_model.generate(
                    instructions=batch_prompts,
                    object_set_embeds=batch_object_embeds,
                    max_length=max_completion_length,
                    do_sample=True,
                    temperature=self.config.temperature,
                    top_p=1.0,
                    top_k=-1, # 200?
                    return_dict=True,
                    return_logprob=True,
                )

                # calculate sequence level logprobs if available
                token_logprobs_sgl = outputs.get("logprobs", [])  # 是一个 List[Tuple[logprob, token_id, decoded_text]] (结构随版本略有不同)
                seq_logprobs_sum = []
                seq_logprobs_mean = []
                if token_logprobs_sgl != []:
                    for logprob_tuples in token_logprobs_sgl:
                        logprobs = np.array([tup[0] for tup in logprob_tuples])  # 提取每个token的logprob
                        seq_logprobs_sum.append(logprobs.sum().item())
                        seq_logprobs_mean.append(logprobs.mean().item())

                    logger.debug(f"Rank {self.process_idx} - Generated sequence logprobs sum: {seq_logprobs_sum}")
                    logger.debug(f"Rank {self.process_idx} - Generated sequence logprobs mean: {seq_logprobs_mean}")


                completion_ids = outputs["completion_ids"]
                inputs_embeds = outputs["inputs_embeds"]
                inputs_mask = outputs["attention_masks"]

                if isinstance(completion_ids, list):
                    logger.critical(f"Rank {self.process_idx} - completion_ids is a list, len: {[len(cid) for cid in completion_ids]}")
                    # show if any exceeds max_completion_length
                    for cid in completion_ids:
                        if len(cid) > max_completion_length:
                            logger.critical(f"Rank {self.process_idx} - completion_id length {len(cid)} exceeds max_completion_length {max_completion_length}")
                            # show response
                            response = self.tokenizer.decode(cid, skip_special_tokens=True)
                            logger.critical(f"Rank {self.process_idx} - completion_id response: {response}")
                    all_completion_ids.extend([torch.tensor(completion_id, dtype=torch.long, device=self.accelerator.device).unsqueeze(0) for completion_id in completion_ids])
                else:
                    logger.critical(f"Rank {self.process_idx} - completion_ids is a tensor, shape: {completion_ids.shape}")
                    all_completion_ids.append(completion_ids) # already a tensor
                all_inputs_embeds.append(inputs_embeds) # [B, L, D]
                all_inputs_mask.append(inputs_mask) # [B, L]

        # Pad right side of completion_ids
        # logger.debug(f"{all_completion_ids=}")
        completion_lens = [completion_id.size(-1) for completion_id in all_completion_ids]
        max_completion_len = max([completion_id.size(-1) for completion_id in all_completion_ids])
        logger.critical(f"Rank {self.process_idx} - Completion lengths: {completion_lens}, Max completion length: {max_completion_len}, max_new_tokens: {max_completion_length}")

        completion_ids = [
            F.pad(completion_id, (0, max_completion_len - completion_id.size(-1)), value=self.tokenizer.pad_token_id)
            for completion_id in all_completion_ids
        ]
        completion_ids = torch.cat(completion_ids, dim=0)
        logger.debug(f"{completion_ids.shape=}")
        completion_mask = self.create_completion_mask(completion_ids, self.tokenizer.eos_token_id)

        # Pad left side of inputs_embeds and inputs_mask
        max_inputs_len = max([inputs_embed.size(1) for inputs_embed in all_inputs_embeds])
        inputs_embeds = [
            F.pad(inputs_embed, (0, 0, max_inputs_len - inputs_embed.size(1), 0), value=0)
            for inputs_embed in all_inputs_embeds
        ]
        inputs_embeds = torch.cat(inputs_embeds, dim=0)
        inputs_mask = [
            F.pad(inputs_mask, (max_inputs_len - inputs_mask.size(1), 0), value=0)
            for inputs_mask in all_inputs_mask
        ]
        inputs_mask = torch.cat(inputs_mask, dim=0)

        # tensorize seq_logprobs
        seq_logprobs_sum = torch.tensor(seq_logprobs_sum, dtype=torch.float32, device=self.accelerator.device)
        seq_logprobs_mean = torch.tensor(seq_logprobs_mean, dtype=torch.float32, device=self.accelerator.device)
        
        # return prompt_ids, prompt_mask, completion_ids, completion_mask
        return inputs_embeds, inputs_mask.long(), completion_ids.long(), completion_mask.long(), seq_logprobs_sum, seq_logprobs_mean

    def initial_copy_lora_for_sglang(self):
        # copy load_checkpoint to lora_update_path
        if self.config.load_checkpoint:
            logger.debug(f"Rank {self.process_idx} - Copying LoRA checkpoint to update path")
            shutil.copytree(self.config.load_checkpoint, self.config.lora_update_path, dirs_exist_ok=True)
            # List all files in the update path
            files = os.listdir(self.config.lora_update_path)
            logger.debug(f"Rank {self.process_idx} - Files in update path: {files}")
        else:
            # Dump new init LoRA
            logger.debug(f"Rank {self.process_idx} - Dumping new LoRA to update path")
            self.lm.save_pretrained(self.config.lora_update_path)
            

    def update_lora_for_sglang(self, model: Optional[MultimodalLanguageModelDecoderOnly]=None):
        """Update LoRA for SGLang to be reloaded from disk"""
        logger.debug(f"Rank {self.process_idx} - Updating LoRA for SGLang")
        
        unwrapped_model = model if model else self.unwrapped_model
        unwrapped_model.language_encoder.save_pretrained(self.config.lora_update_path)
        unwrapped_model.save_non_lm_parameters(self.config.lora_update_path)
        # List all files in the update path
        files = os.listdir(self.config.lora_update_path)
        logger.debug(f"Rank {self.process_idx} - Files in update path: {files}")

    def selective_log_softmax(self, logits, input_ids):
        """
        Computes log probabilities for specific tokens in the vocabulary.
        """
        log_probs = nn.functional.log_softmax(logits, dim=-1)
        return log_probs.gather(dim=-1, index=input_ids.unsqueeze(-1)).squeeze(-1)

    def compute_log_probs_mm(self, model, inputs_embeds, attention_mask, completion_ids, completion_mask, logits_to_keep, calculate_entropy=False):
        """
        Computes the log probabilities for a batch of tokens with multimodal inputs.
        """
        # For multimodal model, we need to prepare the inputs differently
        batch_size = inputs_embeds.size(0)
        
        # Prepare inputs for forward pass
        model_inputs = {
            "inputs_embeds": inputs_embeds,
            "attention_masks": attention_mask,
            "response_ids": completion_ids,
            "response_mask": completion_mask,
            # "return_dict": True,
        }

        # logger.info(f"inputs_embeds: {inputs_embeds.size()}, attention_mask: {attention_mask.size()}, completion_ids: {completion_ids.size()}, completion_mask: {completion_mask.size()}")

        # show use_cache status
        # assert self.unwrapped_model.language_encoder.config.use_cache == False, "use_cache should be False during log prob computation"
        # # check gradient checkpointing status
        # assert self.unwrapped_model.language_encoder.is_gradient_checkpointing, "gradient checkpointing should be enabled during log prob computation"
        # # report batch size
        # logger.critical(f"Rank {self.process_idx} - Computing log probs for batch size: {batch_size}")
        # logger.critical(f"Rank {self.process_idx} - inputs_embeds device: {inputs_embeds.device}, attention_mask device: {attention_mask.device}, completion_ids device: {completion_ids.device}, completion_mask device: {completion_mask.device}")
        
        # Get logits from the model
        logits = model(**model_inputs).logits[:, :-1, :]
        # completion_ids = completion_ids[:, 1:]  # Shift right to align with logits
        
        # Select only the last 'logits_to_keep' tokens
        completion_ids = completion_ids[:, -logits_to_keep:].contiguous()
        logits = logits[:, -logits_to_keep:, :].contiguous()

        if calculate_entropy:
            # Calculate entropy bonus
            with torch.no_grad():
                entropies = entropy_from_logits(logits) # token-wise entropy, [batch_size, logits_to_keep]

        if calculate_entropy:
            return self.selective_log_softmax(logits, completion_ids), entropies
        else:
            return self.selective_log_softmax(logits, completion_ids)

    def compute_log_probs_dist_mm(self, model, inputs_embeds, attention_mask, completion_ids, completion_mask, logits_to_keep, process_idx, world_size, micro_batch_size=None, group=None):
        """
        Computes log probabilities distributedly for multimodal inputs
        """
        batch_size = inputs_embeds.size(0)
        assert batch_size % world_size == 0, f"Number of texts ({batch_size}) must be divisible by the world size ({world_size})"
        local_batch_size = batch_size // world_size
        start_idx = process_idx * local_batch_size
        end_idx = min(start_idx + local_batch_size, batch_size)

        local_inputs_embeds = inputs_embeds[start_idx:end_idx] #.detach()?
        local_attention_mask = attention_mask[start_idx:end_idx]
        local_completion_ids = completion_ids[start_idx:end_idx]
        local_completion_mask = completion_mask[start_idx:end_idx]
        
        if micro_batch_size is None or micro_batch_size >= local_batch_size:
            log_probs = self.compute_log_probs_mm(model, local_inputs_embeds, local_attention_mask, local_completion_ids, local_completion_mask, logits_to_keep)
        else:
            log_probs = []
            for i in range(0, local_batch_size, micro_batch_size):
                end = min(i + micro_batch_size, batch_size)
                log_probs.append(
                    self.compute_log_probs_mm(
                        model, 
                        local_inputs_embeds[i:end],
                        local_attention_mask[i:end],
                        local_completion_ids[i:end],
                        local_completion_mask[i:end],
                        logits_to_keep
                    )
                )
            log_probs = torch.cat(log_probs, dim=0)
        
        log_probs = all_gather_vdim(log_probs, group=group)
        return torch.cat(log_probs, dim=0)
    

    def generate_and_sync_completions(self, batch_samples, num_generations, max_completion_length, group=None):
        """
        Generate completions for a batch of multimodal inputs and synchronize across processes.
        This method is used to generate rollout data for training.
        Returns batch_samples as-is for convenience.
        FIXME: now it returns all_inputs_embeds on ALL processes, may waste memory.
            E.g., [num_rollouts * batch_size * embedding_dim * input_seq_len] would be stored on all processes.
            For num_rollouts=8, batch_size=128, embedding_dim=4096, input_seq_len=512, FP32, it would take 8GB memory.
        """
        batch = collate_fn(batch_samples, self.accelerator.device)

        # format instruction with chat template
        # batch["instructions"] = [self.apply_chat_template(inst) for inst in batch["instructions"]]
        batch["instructions"] = [apply_qwen_template(inst, tokenizer=self.tokenizer)[0] for inst in batch["instructions"]]

        
        # Generate completions (only on inference processes)
        logger.debug(f"Rank {self.process_idx} - Starting generating rollout traces")
        
        
        inputs_embeds, inputs_mask, completion_ids, completion_mask, seq_logprobs_sum, seq_logprobs_mean = self.generate_completions(
            batch, num_generations, max_completion_length
        )

        # logger.info(f"inputs_mask shape: {inputs_mask.size()}, inputs_embeds shape: {inputs_embeds.size()}, completion_ids shape: {completion_ids.size()}, completion_mask shape: {completion_mask.size()}")
        
        # self.accelerator.wait_for_everyone()
        
        logger.debug(f"Rank {self.process_idx} - Gathering rollout data")

        if group is not None:
            logger.warning(f"Rank {self.process_idx} - Using custom group {group} for all_gather_vdim, may have bug!")
        
        # Gather data from all inference processes, discard dummy data
        all_inputs_embeds = all_gather_vdim(inputs_embeds, group=group)
        all_inputs_mask = all_gather_vdim(inputs_mask, group=group)
        all_completion_ids = all_gather_vdim(completion_ids, group=group)
        all_completion_mask = all_gather_vdim(completion_mask, group=group)
        
        all_completion_logprobs_sum = all_gather_vdim(torch.tensor(seq_logprobs_sum, device=self.accelerator.device), group=group)
        all_completion_logprobs_mean = all_gather_vdim(torch.tensor(seq_logprobs_mean, device=self.accelerator.device), group=group)
        
        return all_inputs_embeds, all_inputs_mask, all_completion_ids, all_completion_mask, all_completion_logprobs_sum, all_completion_logprobs_mean, batch_samples

    def generate_rollout_data_from_completions(self, all_inputs_embeds, all_inputs_mask, all_completion_ids, all_completion_mask,
                                                    all_completion_logprobs_sum, all_completion_logprobs_mean,
                                                    batch_samples, num_generations, need_old_logits=True, need_ref_logits=True):
        """
        Generate rollout data from pre-generated completions.
        """
        # Pad left completion_ids to the same length
        logits_to_keep = max([completion_id.size(1) for completion_id in all_completion_ids])
        logger.debug(f"Rank {self.process_idx} - Padding rollout data, logits_to_keep: {logits_to_keep}")
        all_completion_ids = [
            F.pad(completion_id, (0, logits_to_keep - completion_id.size(1)), value=self.tokenizer.pad_token_id)
            for completion_id in all_completion_ids
        ]
        all_completion_mask = [
            F.pad(mask, (0, logits_to_keep - mask.size(1)), value=0)
            for mask in all_completion_mask
        ]

        all_completion_ids = torch.cat(all_completion_ids, dim=0) # [batch_size, max_completion_length]
        all_completion_mask = torch.cat(all_completion_mask, dim=0) # [batch_size, max_completion_length]

        # Pad right inputs_embeds and inputs_mask
        max_inputs_len = max([inputs_embed.size(1) for inputs_embed in all_inputs_embeds])
        all_inputs_embeds = [
            F.pad(inputs_embed, (0, 0, max_inputs_len - inputs_embed.size(1), 0), value=0)
            for inputs_embed in all_inputs_embeds
        ]
        all_inputs_mask = [
            F.pad(inputs_mask, (max_inputs_len - inputs_mask.size(1), 0), value=0)
            for inputs_mask in all_inputs_mask
        ]

        all_inputs_embeds = torch.cat(all_inputs_embeds, dim=0) # [batch_size, max_inputs_length, feature_dim]
        all_inputs_mask = torch.cat(all_inputs_mask, dim=0) # [batch_size, max_inputs_length]

        logger.info(f"all_inputs_embeds shape: {all_inputs_embeds.size()}, all_inputs_mask shape: {all_inputs_mask.size()}, all_completion_ids shape: {all_completion_ids.size()}, all_completion_mask shape: {all_completion_mask.size()}")

        completion_lengths = all_completion_mask.sum(dim=1)

        # Compute log probabilities for completions
        logger.debug(f"Rank {self.process_idx} - Calculating reference and old log probs")
        with torch.no_grad():
            old_log_probs = None
            ref_log_probs = None
            with contextlib.nullcontext():
                # NOTE: instead, we use results in first step.
                # if need_old_logits:
                #     old_log_probs = self.compute_log_probs_dist_mm(
                #         self.model, 
                #         all_inputs_embeds,
                #         all_inputs_mask,
                #         all_completion_ids,
                #         all_completion_mask,
                #         logits_to_keep, 
                #         self.training_process_idx, 
                #         self.config.num_training_gpus, 
                #         micro_batch_size=self.config.ref_model_micro_batch_size, 
                #         group=self.training_group
                #     )
                if need_ref_logits:
                    with model_to_device(self.ref_model, self.accelerator.device, empty_cache=True): # empty CUDA cache to free ref-model completely
                        # make ref_model eval - already in eval mode
                        ref_log_probs = self.compute_log_probs_dist_mm(
                            self.ref_model, 
                            all_inputs_embeds,
                            all_inputs_mask,
                            all_completion_ids,
                            all_completion_mask,
                            logits_to_keep, 
                            self.training_process_idx, 
                            self.config.num_training_gpus, 
                            micro_batch_size=self.config.ref_model_micro_batch_size, 
                            group=self.training_group
                        )
        
        logger.debug(f"Rank {self.process_idx} - Rollout data generated")
        
        # Format completions for reward calculation
        formatted_completions = [[{'content': self.tokenizer.decode(ids, skip_special_tokens=True)}] 
                                for ids in all_completion_ids]
        
        # Repeat prompts to match the number of generated completions
        repeated_prompts = []
        repeated_answers = []
        repeated_batch_samples = []
        
        for i, sample in enumerate(batch_samples):
            for _ in range(num_generations):
                repeated_prompts.append(sample["description"])
                repeated_answers.append(sample["expected_response"])
                repeated_batch_samples.append(sample)
        
        return {
            # "input_ids": input_ids,
            # "attention_mask": attention_mask,
            # "instructions": all_prompts,
            # "object_set_embeds": all_object_set_embeds,
            "inputs_embeds": all_inputs_embeds,
            "attention_mask": all_inputs_mask,
            "completion_ids": all_completion_ids,
            "completion_mask": all_completion_mask,
            "old_log_probs": old_log_probs,
            "ref_log_probs": ref_log_probs,
            "formatted_completions": formatted_completions,
            "repeated_prompts": repeated_prompts,
            "repeated_answers": repeated_answers,
            "logits_to_keep": logits_to_keep,
            "batch_size": len(batch_samples),
            "num_generations": num_generations,
            "repeated_batch_samples": repeated_batch_samples,
            "completion_lengths": completion_lengths,
            "all_completion_logprobs_sum": all_completion_logprobs_sum,
            "all_completion_logprobs_mean": all_completion_logprobs_mean,
            # "object_set_embeds": object_set_embeds,
        }
    
    def generate_rollout_data(self, batch_samples, num_generations, max_completion_length, need_old_logits=True, need_ref_logits=True):
        """
        Generate rollout data from multimodal inputs.
        """
        all_inputs_embeds, all_inputs_mask, all_completion_ids, all_completion_mask, all_completion_logprobs_sum, all_completion_logprobs_mean, batch_samples = self.generate_and_sync_completions(
            batch_samples, 
            num_generations, 
            max_completion_length
        )

        if not self.is_training_process:
            return None
        
        return self.generate_rollout_data_from_completions(
            all_inputs_embeds[:self.config.num_inference_gpus], 
            all_inputs_mask[:self.config.num_inference_gpus], 
            all_completion_ids[:self.config.num_inference_gpus],
            all_completion_mask[:self.config.num_inference_gpus],
            batch_samples, 
            num_generations, 
            need_old_logits=need_old_logits, 
            need_ref_logits=need_ref_logits
        )
    
    def compute_reward_advantage(self, rollout_data, reward_function):
        """
        Compute rewards and advantages for a batch of rollout data.
        """
        # TODO: how to pre-compute logps to calibrate reward model scores?
        #   maybe use SGLang's return_logprobs=True - need to check the flush_state implementation for logprobs recording.

        logger.debug(f"Rank {self.process_idx} - Computing rewards")
        rewards = torch.tensor(
            reward_function(
                prompts=rollout_data["repeated_prompts"],
                completions=rollout_data["formatted_completions"],
                answer_data=rollout_data["repeated_batch_samples"],
            ),
            dtype=torch.float32,
            device=self.accelerator.device
        )

        correctness = reward_function.compute_correctness(
            prompts=rollout_data["repeated_prompts"],
            completions=rollout_data["formatted_completions"],
            answer_data=rollout_data["repeated_batch_samples"],
        )
        correctness = torch.tensor(correctness, dtype=torch.float32, device=self.accelerator.device)
        
        num_generations = rollout_data["num_generations"]
        rewards = rewards.view(-1, num_generations)
        correctness = correctness.view(-1, num_generations)
        avg_reward = rewards.mean().item()
        avgmax_reward = rewards.max(dim=1).values.mean().item()
        
        # Compute advantages
        mean_rewards = rewards.mean(dim=1).repeat_interleave(rollout_data["num_generations"])
        std_rewards = rewards.std(dim=1).repeat_interleave(rollout_data["num_generations"])
        advantages = ((rewards.view(-1) - mean_rewards) / (std_rewards + 1e-4)).unsqueeze(1)

        # conduct DS-GRPO
        # TODO: move to grpo_loss, that use more accurate logps, but that need ALL logps calculated first to decide group normalization
        if self.config.logp_factor_correct > 0 or self.config.logp_factor_wrong > 0:
            if self.config.logp_length_normalize:
                logp = rollout_data["all_completion_logprobs_mean"]
            else:
                logp = rollout_data["all_completion_logprobs_sum"]

            # logp = torch.tensor(logp, dtype=torch.float32, device=self.accelerator.device).view(-1, num_generations) # [batch_size, num_generations]
            # logp = logp.view(-1, num_generations)  # [batch_size, num_generations]
            # stack logp, which is a list of tensors
            # logger.warning(f"Rank {self.process_idx} - logp is {logp}")
            logp = torch.stack(logp).view(-1, num_generations)  # [batch_size, num_generations]

            # show logp
            if self.is_main_training_process:
                logger.info(f"Rank {self.process_idx} - logp is {logp}")

            # normalize correct and wrong logp separately, and the mean, std shall be computed only within the group
            if self.config.logp_group_normalize:
                # normalized_logp = torch.zeros_like(logp)
                # correct_mask = (correctness == 1)
                # wrong_mask = (correctness == 0)
                # assert correct_mask.sum() + wrong_mask.sum() == logp.numel(), "correctness mask error in logp group normalize"

                # for i in range(logp.size(0)):
                #     if correct_mask[i].sum() > 0:
                #         mean_logp_correct = logp[i][correct_mask[i]].mean()
                #         std_logp_correct = logp[i][correct_mask[i]].std() + 1e-4
                #         normalized_logp[i][correct_mask[i]] = (logp[i][correct_mask[i]] - mean_logp_correct) / std_logp_correct
                #     if wrong_mask[i].sum() > 0:
                #         mean_logp_wrong = logp[i][wrong_mask[i]].mean()
                #         std_logp_wrong = logp[i][wrong_mask[i]].std() + 1e-4
                #         normalized_logp[i][wrong_mask[i]] = (logp[i][wrong_mask[i]] - mean_logp_wrong) / std_logp_wrong

                # logp = normalized_logp

                # just normalize ignoring correctness
                mean_logp = logp.mean(dim=1, keepdim=True)
                std_logp = logp.std(dim=1, keepdim=True) + 1e-4
                logp = (logp - mean_logp) / std_logp

            # apply to advantages
            logp_adjustment = torch.zeros_like(logp)
            logp_adjustment[correctness == 1] = -self.config.logp_factor_correct * logp[correctness == 1]
            logp_adjustment[correctness == 0] = self.config.logp_factor_wrong * logp[correctness == 0]

            logp_adjustment = logp_adjustment.view(-1, 1) # same shape as advantages

            if self.config.logp_clip_ratio > 0:
                # clip logp_adjustment to be within +/- logp_clip_ratio * advantages
                logp_adjustment = torch.clamp(
                    logp_adjustment, 
                    -self.config.logp_clip_ratio * advantages, 
                    self.config.logp_clip_ratio * advantages,
                )

            advantages += logp_adjustment

        mean_advantages = advantages.mean()
        std_advantages = advantages.std() + 1e-4
        
        if self.config.normalize_advantages:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-4)
        
        return rewards, advantages, avg_reward, avgmax_reward, correctness, mean_rewards, std_rewards, mean_advantages, std_advantages
    
    def grpo_loss(self, rollout_data, reward_function, beta, epsilon_min, epsilon_max, calculate_entropy=True):
        """
        Compute GRPO loss for multimodal inputs.
        """
        self.model.train() # Ensure model is in training mode, necessary for gradient checkpointing
        # Only training processes compute loss
        if not self.is_training_process:
            return None, 0.0
        
        logger.debug(f"Rank {self.process_idx} - Computing GRPO loss, full/micro batch size: {rollout_data['inputs_embeds'].size(0)}")
        
        inputs_embeds = rollout_data["inputs_embeds"]
        attention_mask = rollout_data["attention_mask"]
        completion_ids = rollout_data["completion_ids"]
        completion_mask = rollout_data["completion_mask"]
        logits_to_keep = rollout_data["logits_to_keep"]
        old_log_probs = rollout_data["old_log_probs"]
        ref_log_probs = rollout_data["ref_log_probs"]
        
        # Split local batch
        assert inputs_embeds.size(0) % self.config.num_training_gpus == 0, f"Number of input_ids ({inputs_embeds.size(0)}) must be divisible by the number of training GPUs ({self.config.num_training_gpus})"
        local_batch_size = inputs_embeds.size(0) // self.config.num_training_gpus
        start_idx = self.training_process_idx * local_batch_size
        end_idx = min(start_idx + local_batch_size, inputs_embeds.size(0))
        
        # inputs_embeds = inputs_embeds[start_idx:end_idx]
        # load inputs_embeds to device
        if not inputs_embeds.device == self.accelerator.device:
            inputs_embeds = inputs_embeds[start_idx:end_idx].detach().to(self.accelerator.device)
        else:
            inputs_embeds = inputs_embeds[start_idx:end_idx].detach()
            
        attention_mask = attention_mask[start_idx:end_idx].detach()
        completion_ids = completion_ids[start_idx:end_idx].detach()
        completion_mask = completion_mask[start_idx:end_idx].detach()
        
        if old_log_probs is not None:
            old_log_probs = old_log_probs[start_idx:end_idx].clone().detach()
        if ref_log_probs is not None:
            ref_log_probs = ref_log_probs[start_idx:end_idx].clone().detach()

        # logger.info(f"inputs_embeds shape: {inputs_embeds.size()}, attention_mask shape: {attention_mask.size()}, completion_ids shape: {completion_ids.size()}, completion_mask shape: {completion_mask.size()}")
        
        # Compute current token log probabilities
        logger.debug(f"Rank {self.process_idx} - Computing token log probs")
        token_log_probs, entropies = self.compute_log_probs_mm(self.model, inputs_embeds, attention_mask, 
                                                    completion_ids, completion_mask, logits_to_keep, calculate_entropy=calculate_entropy)

        if calculate_entropy:
            # Calculate entropy bonus
            with torch.no_grad():
                # Average entropy per sequence
                entropies = (entropies * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)
        else:
            entropies = None

        
        # Calculate probability ratio
        # 如果没有传入 old_log_probs，说明这是第一轮 (iter 0)
        # 此时当前的 policy 就是 behavior policy
        if old_log_probs is None:
            # mu=1, always on policy
            old_log_probs = token_log_probs.detach().clone()
        ratio = torch.exp(token_log_probs - old_log_probs) # the importance sampling weights?
        
        advantages_local = rollout_data["advantage"][start_idx:end_idx]
        
        # Compute PPO surrogate loss
        surr1 = ratio * advantages_local
        surr2 = torch.clamp(ratio, 1 - epsilon_max, 1 + epsilon_min) * advantages_local
        surrogate_loss = torch.min(surr1, surr2)
        
        # Compute KL divergence
        if ref_log_probs is None:
            kl = torch.zeros_like(token_log_probs)
        else:
            kl = torch.exp(ref_log_probs - token_log_probs) - (ref_log_probs - token_log_probs) - 1
        
        # Combine losses
        per_token_loss = surrogate_loss - beta * kl

        if self.config.token_level_pg_loss:
            # Token-level loss, as proposed in DAPO
            # NOTE: since it is micro-batch, shall we put all large batch's completion_mask sum in the denominator?
            loss = -((per_token_loss * completion_mask).sum() / completion_mask.sum())
        else:
            # Sample-level loss
            loss = -((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        
        # Sample-wise kl and ratio
        with torch.no_grad():
            rollout_ratio = ((ratio * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).detach()
            rollout_kl = ((kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).detach()
        
        return loss, rollout_ratio, rollout_kl, old_log_probs, start_idx, end_idx, entropies

    def split_rollout_data(self, rollout_data, micro_batch_size):
        """
        Split rollout data into microbatches for grpo_loss function.
        """
        rollout_batch_size = rollout_data["inputs_embeds"].size(0)
        num_micro_batches = rollout_batch_size // micro_batch_size
        assert rollout_batch_size % micro_batch_size == 0, f"Rollout batch size ({rollout_batch_size}) must be divisible by micro batch size ({micro_batch_size})"

        logger.debug(f"Rank {self.process_idx} - Splitting rollout data into {num_micro_batches} microbatches")
        for i in range(num_micro_batches):
            start_idx = i * micro_batch_size
            end_idx = min(start_idx + micro_batch_size, rollout_batch_size)
            yield {
                "inputs_embeds": rollout_data["inputs_embeds"][start_idx:end_idx],
                "attention_mask": rollout_data["attention_mask"][start_idx:end_idx],
                "completion_ids": rollout_data["completion_ids"][start_idx:end_idx],
                "completion_mask": rollout_data["completion_mask"][start_idx:end_idx],
                "old_log_probs": rollout_data["old_log_probs"][start_idx:end_idx] if rollout_data["old_log_probs"] is not None else None,
                "ref_log_probs": rollout_data["ref_log_probs"][start_idx:end_idx] if rollout_data["ref_log_probs"] is not None else None,
                "formatted_completions": rollout_data["formatted_completions"][start_idx:end_idx],
                "repeated_prompts": rollout_data["repeated_prompts"][start_idx:end_idx],
                "repeated_answers": rollout_data["repeated_answers"][start_idx:end_idx],
                "logits_to_keep": rollout_data["logits_to_keep"],
                "batch_size": rollout_data["batch_size"],
                "num_generations": rollout_data["num_generations"],
                "repeated_batch_samples": rollout_data["repeated_batch_samples"][start_idx:end_idx],
                # rewards and advantages and avg_reward
                "reward": rollout_data["reward"][start_idx:end_idx],
                "advantage": rollout_data["advantage"][start_idx:end_idx],
                "avg_reward": rollout_data["avg_reward"],
            }

    def save_checkpoint(self, output_dir: str, postfix: str | int):
        """
        Saves the current state of the model, optimizer, and scheduler to the specified directory.
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        output_subdir = os.path.join(output_dir, f"checkpoint-{postfix}")
        os.makedirs(output_subdir, exist_ok=True)

        logger.info(f"Saving checkpoint to {output_subdir}")

        # Use accelerator's save_state for a comprehensive checkpoint.
        # This saves the model, optimizer, scheduler, and random states.
        # self.accelerator.save_state(output_subdir)

        # It's also good practice to save the tokenizer and the unwrapped model's
        # configuration for easier reloading and inference.
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        
        # For PEFT models, save_pretrained saves the adapter and its config.
        # For full models, it saves the entire model.
        unwrapped_model.save_pretrained(output_subdir)
        
        logger.info(f"Checkpoint saved successfully to {output_subdir}")
    
    def train(self, train_data, reward_function):
        logger.debug(f"Rank {self.process_idx} - Training process started")

        # init a generator, so that the random orders are the same across all processes
        generator = torch.Generator("cpu")
        generator.manual_seed(self.config.seed)
        dataloader = DataLoader(
            train_data,
            batch_size=self.config.rollout_batch_size,
            # sampler=RandomSampler(train_data, generator=generator), # drop last batch if not full
            shuffle=True,
            collate_fn=collate_fn_simple, # just put samples in a list
            # num_workers=self.config.num_workers,
            num_workers=0, # since restart sglang will kill all subprocesses - it will also kill dataloader workers, so we don't use them
            pin_memory=True,
            # prefetch_factor=24,
            drop_last=True,
            generator=generator,
            # enable_memory_saver=True,
        )

        # Main training loop
        update_counter = 0
        for iteration in range(self.config.num_iterations):
            if self.is_main_process:
                logger.info(f"\nIteration {iteration+1}/{self.config.num_iterations}")
            
            self.update_ref_model()
            
            # Ensure all processes are synchronized
            self.accelerator.wait_for_everyone()

            # Re-create the iterator for each iteration/epoch
            data_iterator = iter(dataloader)
            
            # === Pipeline Priming Step for the current iteration ===
            logger.debug(f"Rank {self.process_idx} - Priming the pipeline for iteration {iteration+1}")
            self.accelerator.wait_for_everyone()
            try:
                batch_samples = next(data_iterator)
                # this will conduct inference on inference processes and broadcast the results to training processes
                completion_data = self.generate_and_sync_completions(
                    batch_samples, 
                    self.config.num_generations, 
                    self.config.max_completion_length
                )
            except StopIteration:
                logger.warning("Dataloader is empty, skipping iteration.")
                continue # Skip to the next iteration if the dataloader is empty
            
            self.accelerator.wait_for_everyone()
            
            # Progress bar only on main training process
            # We iterate up to len(dataloader) - 1 because we already processed the first batch
            progress_bar = tqdm(range(len(dataloader) - 1),
                                desc=f"Iteration {iteration+1}/{self.config.num_iterations}", 
                                disable=not self.is_main_training_process)
            
            for step in progress_bar:
                if self.config.start_iters > 0 and iteration == 0:
                    if step < self.config.start_iters:
                        next(data_iterator)
                        continue

                # The training processes already have `rollout_data` from the previous iteration (or priming)
                # The inference processes will generate the *next* batch's data in parallel.
                
                # --- Generation Step (Inference Processes) ---
                # if self.is_inference_process:
                # clean completion_data on inference processes to save memory
                if self.is_inference_process:
                    del completion_data
                    gc.collect()
                    torch.cuda.empty_cache()

                gen_start_time = time.time()
                try:
                    next_batch_samples = next(data_iterator)
                    gen_start_time = time.time()
                    # Generate completions for the next batch of samples
                    # for train process, this will be a dummy data, since the data is already generated in the priming step
                    next_completion_data = self.generate_and_sync_completions(
                        next_batch_samples, 
                        self.config.num_generations, 
                        self.config.max_completion_length,
                        group=self.group
                    )
                except StopIteration:
                    # No more data to generate
                    next_completion_data = None

                # Measure generation time on inference processes and broadcast it
                if self.is_inference_process:
                    generation_time = time.time() - gen_start_time
                else:
                    generation_time = 0.0

                # --- Training Step (Training Processes) ---
                if self.is_training_process:
                    train_start_time = time.time()
                    # if rollout_data is None: # Skip if no data was received
                    if completion_data is None:
                        logger.warning(f"Rank {self.process_idx} has no completion data to process, skipping training step.")
                        raise Exception("No completion data to process, must be a bug in the data generation step.")
                    
                    all_inputs_embeds, all_inputs_mask, all_completion_ids, all_completion_mask, all_completion_logprobs_sum, all_completion_logprobs_mean, batch_samples = completion_data

                    # calculate rollout data from the generated completions
                    rollout_data = self.generate_rollout_data_from_completions(
                        all_inputs_embeds[:self.config.num_inference_gpus],
                        all_inputs_mask[:self.config.num_inference_gpus],
                        all_completion_ids[:self.config.num_inference_gpus],
                        all_completion_mask[:self.config.num_inference_gpus],
                        all_completion_logprobs_sum[:self.config.num_inference_gpus],
                        all_completion_logprobs_mean[:self.config.num_inference_gpus],
                        batch_samples,
                        num_generations=self.config.num_generations,
                        need_old_logits=(self.config.mu > 1),
                        need_ref_logits=(self.config.beta > 0),
                    )

                    # save memory, especially in all_inputs_embeds
                    del completion_data, all_inputs_embeds, all_inputs_mask, all_completion_ids, all_completion_mask, all_completion_logprobs_sum, all_completion_logprobs_mean, batch_samples

                    # Compute rewards and advantages
                    (
                        rewards, advantages, avg_reward, avgmax_reward, correctness,
                        mean_rewards, std_rewards, mean_advantages, std_advantages,
                    ) = self.compute_reward_advantage(rollout_data, reward_function)
                    rollout_data["reward"] = rewards
                    rollout_data["advantage"] = advantages
                    rollout_data["avg_reward"] = avg_reward

                    rollout_data["mean_rewards"] = mean_rewards
                    rollout_data["std_rewards"] = std_rewards
                    rollout_data["mean_advantages"] = mean_advantages
                    rollout_data["std_advantages"] = std_advantages

                    # Compute pass@k
                    N_passes = rewards.shape[-1]
                    pass_k = [1, 2, 3, 5, 8, 12]
                    pass_k = [k for k in pass_k if k <= N_passes]
                    # max_reward = reward_function.reward_considered_correct # not necessarily the reward_function.max_reward
                    # correct = (rewards >= max_reward).sum(dim=-1)
                    correct = correctness.sum(dim=-1)
                    pass_at_k_acc = [[pass_at_k(N_passes, c.item(), k) for c in correct] for k in pass_k] # [N_k, B]
                    pass_at_k_acc = np.array(pass_at_k_acc).mean(axis=1)

                    if self.is_main_training_process:
                        logger.info(f"Rewards: {rewards}")
                        logger.info(f"Rewards (Mean/Std): {rewards.mean():.2f} ± {rewards.std():.2f}")
                        logger.info(f"Advantages: {advantages.squeeze().mean():.2f} ± {advantages.squeeze().std():.2f}")
                        logger.info(f"Average Reward: {avg_reward}, Average Max Reward: {avgmax_reward}")
                        _info = []
                        for idx, k in enumerate(pass_k):
                            _info.append(f"Pass@{k}: {pass_at_k_acc[idx]*100:.2f}%")
                        logger.info(", ".join(_info))

                    # rollout_data["inputs_embeds"] = rollout_data["inputs_embeds"].clone().detach()
                    rollout_data["inputs_embeds"].requires_grad = False
                    if self.config.offload_input_embeds:
                        # Offload inputs_embeds to CPU to save GPU memory
                        logger.info(f"Offloading inputs_embeds to CPU to save GPU memory")
                        rollout_data["inputs_embeds"] = rollout_data["inputs_embeds"].to("cpu")

                    # create old_log_probs buffer if mu > 1
                    if self.config.mu > 1 and rollout_data["old_log_probs"] is None:
                        rollout_data["old_log_probs"] = [None] * rollout_data["inputs_embeds"].size(0)
                            # a list, each element corresponds to a prompt+rollout
                    gc.collect() 
                    torch.cuda.empty_cache()

                    for grpo_iter in range(self.config.mu):
                        # Split rollout data into microbatches, backward and accumulate gradients
                        num_micro_batches = len(rollout_data["inputs_embeds"]) // self.config.train_micro_batch_size
                        train_stats = []

                        
                        for micro_batch_idx, micro_batch_data in enumerate(self.split_rollout_data(rollout_data, self.config.train_micro_batch_size)):
                            # Compute loss
                            logger.critical(f"Rank {self.process_idx} - Microbatch {micro_batch_idx+1}/{num_micro_batches}, GRPO Step {grpo_iter+1}/{self.config.mu}")
                            # print_mem(f"Microbatch {micro_batch_idx+1} Before GRPO Loss Computation")

                            # LOGIC: Only sync gradients on the very last micro-batch
                            is_last_microbatch = (micro_batch_idx == num_micro_batches - 1)
                            
                            # If it is NOT the last microbatch, use no_sync to prevent DDP communication
                            sync_context = contextlib.nullcontext() if is_last_microbatch else self.accelerator.no_sync(self.model)

                            with sync_context:
                                loss, ratio, kl, old_log_probs, start_idx, end_idx, entropies = self.grpo_loss(
                                    micro_batch_data,
                                    reward_function,
                                    self.config.beta,
                                    self.config.epsilon_min,
                                    self.config.epsilon_max
                                )

                                # put into rollout_data for old_log_probs, if first step
                                if grpo_iter == 0 and self.config.mu > 1:
                                    rollout_data["old_log_probs"][start_idx:end_idx] = old_log_probs.detach().cpu()
                                    # NOTE: this would not cover all prompts, since we split by training process
                                    #   but since each process only computes its own part, it's fine. maybe?
                                
                                loss = loss / num_micro_batches
                                
                                # Backward pass and optimization
                                self.accelerator.backward(loss)
                            
                            train_stats.append({
                                "loss": loss.item(),
                                "kl": kl.detach().cpu(),
                                "ratio": ratio.detach().cpu(),
                                "entropy": entropies.mean().item() if entropies is not None else 0.0,
                            })
                        
                        # Update weights
                        # gradient clip
                        if self.config.max_grad_norm > 0 and self.config.optimizer_type != "adafactor":
                            grad_norm = self.accelerator.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                        else:
                            grad_norm = None

                        self.optimizer.step()
                        self.scheduler.step()
                        self.optimizer.zero_grad()
                        
                        # Log metrics
                        loss = torch.tensor([stat["loss"] for stat in train_stats]).mean().to(self.accelerator.device)
                        kl = torch.cat([stat["kl"] for stat in train_stats]).to(self.accelerator.device)
                        ratio = torch.cat([stat["ratio"] for stat in train_stats]).to(self.accelerator.device)
                        entropy = torch.tensor([stat["entropy"] for stat in train_stats]).mean().to(self.accelerator.device)
                        
                        all_losses = all_gather_vdim(loss.unsqueeze(0), group=self.training_group)
                        loss = torch.cat(all_losses, dim=0).sum().item()
                        
                        kl = all_gather_vdim(kl, group=self.training_group)
                        ratio = all_gather_vdim(ratio, group=self.training_group)
                        
                        kl = torch.cat(kl, dim=0).view(-1).cpu()
                        ratio = torch.cat(ratio, dim=0).view(-1).cpu()

                        entropy = all_gather_vdim(entropy.unsqueeze(0), group=self.training_group)
                        entropy = torch.cat(entropy, dim=0).mean().item()
                        
                        if self.is_main_training_process:
                            logger.info(f"KL: {kl}")
                            logger.info(f"Ratio: {ratio}")
                        
                        lr = self.scheduler.get_last_lr()[0]
                        response_length = rollout_data["completion_lengths"].float().mean().item()
                        max_response_length = rollout_data["completion_lengths"].max().item()
                        
                        if self.is_main_training_process:
                            global_step = step + 1 + len(dataloader) * iteration
                            training_time = time.time() - train_start_time
                            
                            log_dict = {
                                "loss": loss,
                                "average_entropy": entropy,
                                "average_reward": avg_reward,
                                "average_max_reward": avgmax_reward,
                                "learning_rate": lr,
                                "iteration": iteration + 1,
                                "step": global_step,
                                "grpo_iter": grpo_iter + 1,
                                "kl": kl.mean().item(),
                                "ratio": ratio.mean().item(),
                                "response_length": response_length,
                                "max_response_length": max_response_length,
                                "training_time": training_time,
                                # pass@k
                                **{
                                    f"pass@{k}": pass_at_k_acc[idx].item()
                                    for idx, k in enumerate(pass_k)
                                }
                            }

                            if grad_norm is not None:
                                log_dict["grad_norm"] = grad_norm.item()

                            wandb.log(log_dict)
                            
                            progress_bar.set_postfix({
                                "loss": loss,
                                "ent": entropy,
                                "avg_r": avg_reward,
                                "avgmax_r": avgmax_reward,
                                "lr": lr,
                                "len_resp": response_length,
                            })
                            
                            # Find a sample with max reward
                            max_reward_idx = rewards.view(-1).argmax().item()
                            logger.info("".center(50, "="))
                            logger.info(f"Max Reward: {rewards.view(-1)[max_reward_idx].item()}, Ratio: {ratio[max_reward_idx].item()}, KL: {kl[max_reward_idx].item()}")
                            logger.info(f"Max Reward Prompt: {rollout_data['repeated_prompts'][max_reward_idx]}")
                            logger.info(f"Max Reward Completion: {rollout_data['formatted_completions'][max_reward_idx][0]['content']}")
                            logger.info(f"Expected Answer: {rollout_data['repeated_answers'][max_reward_idx]}")
                            
                            # Also log a sample at 95% percentile and non-max
                            max_reward_value = rewards.view(-1)[max_reward_idx].item()
                            rewards_mod = rewards.view(-1).clone()
                            rewards_mod[rewards_mod == max_reward_value] = -float('inf')
                            perc95_idx = torch.quantile(rewards_mod, 0.95, interpolation='nearest').long().item()
                            logger.info(f"95th Percentile Reward: {rewards_mod[perc95_idx].item()}, Ratio: {ratio[perc95_idx].item()}, KL: {kl[perc95_idx].item()}")
                            logger.info(f"95th Percentile Prompt: {rollout_data['repeated_prompts'][perc95_idx]}")
                            logger.info(f"95th Percentile Completion: {rollout_data['formatted_completions'][perc95_idx][0]['content']}")
                            logger.info(f"Expected Answer: {rollout_data['repeated_answers'][perc95_idx]}")
                            logger.info("".center(50, "=") + "\n")

                            # log training time and entropy
                            logger.info(f"Batch Training time: {training_time:.2f} seconds")
                            logger.info(f"Batch Entropy: {entropy:.4f}")
                            logger.info("".center(50, "=") + "\n")
                        
                        update_counter += 1
                
                else:
                    update_counter += self.config.mu
                
                # Synchronize model weights between training and inference processes

                # Broadcast generation_time from rank 0 (main inference) to all other processes
                # This shall be done after the training step to avoid blocking the training processes
                generation_time_tensor = torch.tensor([generation_time], dtype=torch.float32, device=self.accelerator.device)
                dist.broadcast(generation_time_tensor, src=0)
                generation_time = generation_time_tensor.item()
                if self.is_main_training_process:
                    logger.info(f"Generation time for step {step + 1}: {generation_time:.2f} seconds")
                    # log to wandb
                    wandb.log({
                        "generation_time": generation_time,
                        "step": step + 1 + len(dataloader) * iteration,
                        "iteration": iteration + 1,
                    })

                # --- Data Transfer for Next Iteration ---
                # The main training process gathers the completion data from the inference processes for next batch
                if self.is_inference_process:
                    # Wait for the training processes to finish before proceeding
                    self.accelerator.wait_for_everyone()
                    logger.warning(f"Rank {self.process_idx} - Inference process completed, waiting for training processes")
                elif self.is_training_process:
                    # The main training process will wait for the inference processes to finish
                    self.accelerator.wait_for_everyone()
                    logger.warning(f"Rank {self.process_idx} - Main training process completed, waiting for inference processes")

                # --- Retrieve Completions to Train ---
                # currently, inference processes holds next batch completions in `next_completion_data`
                #  training processes holds dummy data

                # stop if no more data from last generation step's next_completion_data, which means the dataloader is exhausted
                if next_completion_data is None:
                    logger.warning(f"Rank {self.process_idx} - No more completion data for the next step.")
                    completion_data = None
                    break # Exit the inner step loop

                logger.warning(f"Rank {self.process_idx} - Retrieving/sending next batch completions")
                all_inputs_embeds, all_inputs_mask, all_completion_ids, all_completion_mask, all_completion_logprobs_sum, all_completion_logprobs_mean, _ = next_completion_data

                logger.warning(f"Rank {self.process_idx} - all_inputs_embeds shape: {all_inputs_embeds[0].size()}, all_inputs_mask shape: {all_inputs_mask[0].size()}, all_completion_ids shape: {all_completion_ids[0].size()}, all_completion_mask shape: {all_completion_mask[0].size()}")

                if self.is_training_process:
                    this_index = self.training_process_idx
                else:
                    this_index = self.process_idx
                
                all_inputs_embeds = all_gather_vdim(all_inputs_embeds[this_index])
                all_inputs_mask = all_gather_vdim(all_inputs_mask[this_index])
                all_completion_ids = all_gather_vdim(all_completion_ids[this_index])
                all_completion_mask = all_gather_vdim(all_completion_mask[this_index])
                
                all_completion_logprobs_sum = all_gather_vdim(all_completion_logprobs_sum[this_index])
                all_completion_logprobs_mean = all_gather_vdim(all_completion_logprobs_mean[this_index])

                completion_data = (
                    all_inputs_embeds, # list of N_gpus tensors
                    all_inputs_mask, 
                    all_completion_ids, 
                    all_completion_mask, 
                    all_completion_logprobs_sum, 
                    all_completion_logprobs_mean,
                    next_batch_samples
                )


                # --- Model Synchronization ---
                if update_counter % self.config.update_iters == 0:
                    logger.debug(f"Rank {self.process_idx} - Synchronizing model weights")
                    if self.config.use_sglang_for_generation:
                        # Push the model to SGLang from training processes
                        if self.is_main_training_process:
                            self.update_lora_for_sglang()

                        self.accelerator.wait_for_everyone()
                        if self.config.use_sglang_for_generation and self.is_inference_process:
                            # restart sglang engine. this is for reloading the LoRA and non-lm parameters
                            # FIXME: this is a hack, need to find a better way if we can load LoRA inplace
                            #   But SGLang currently only supports update weights of main model
                            if NEW_SGLANG: 
                                # use SGLang's new LoRA reloading functionality
                                self.unwrapped_model._non_restart_reload_sglang_weights()
                            else:
                                self.unwrapped_model._init_sglang_engine()
                            # self.unwrapped_model._init_sglang_subprocess()
                            # self.unwrapped_model._init_sglang_server()
                        logger.debug(f"Rank {self.process_idx} - Reached here, after restarting sglang engine")
                    else: # for huggingface
                        self.broadcast_model_from_train_to_all()

                logger.debug(f"Rank {self.process_idx} - Reached here, before next batch")
                self.accelerator.wait_for_everyone()

                # save
                if (step + 1) % self.config.save_iters == 0:
                    if self.is_main_training_process:
                        logger.info(f"[Rank {self.process_idx}] Saving model checkpoint at step {step + 1}")
                        self.save_checkpoint(self.config.output_dir, f"step-e{iteration + 1}-{step + 1}")
                    
                    self.accelerator.wait_for_everyone()
        
            # Save the model after each iteration/epoch
            if self.is_main_training_process:
                logger.info(f"[Rank {self.process_idx}] Saving model checkpoint at the end of iteration {iteration + 1}")
                self.save_checkpoint(self.config.output_dir, f"epoch-{iteration + 1}")
            self.accelerator.wait_for_everyone()


        wandb.finish()
        return self.model

    def broadcast_model_from_train_to_all(self):
        self.accelerator.wait_for_everyone()
        logger.debug(f"Rank {self.process_idx} - Synchronizing model weights")
        if self.world_size > 1:
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    broadcast_src = self.config.num_inference_gpus
                    dist.broadcast(param.data, src=broadcast_src)
        self.accelerator.wait_for_everyone()
        logger.debug(f"Rank {self.process_idx} - Model weights synchronized")

def init_wandb(config: MultimodalGRPOConfig):
    wandb.init(
        project=config.wandb_project,
        name=config.wandb_run_name,
        # config=OmegaConf.to_container(self.config, resolve=True),
        config=config,
    )

def main(rank: int, config: MultimodalGRPOConfig):
    torch.set_float32_matmul_precision('high')
    OmegaConf.set_struct(config, False)

    # logger.info(os.environ)
    # avoid SGLang hanging in NCCL init
    setup_distributed(rank, config)

    # setup logging
    if rank == 0:
        logging.basicConfig(
            format="[%(asctime)s %(name)s %(levelname)s] %(message)s",
            level=logging.INFO,
            datefmt="%I:%M:%S",
        )
    else:
        logging.basicConfig(
            format="[%(asctime)s %(name)s %(levelname)s] %(message)s",
            level=logging.WARNING,
            datefmt="%I:%M:%S",
        )
    
    # Set random seed for reproducibility
    set_random_seed(config.seed)
    config.rank = rank
    config.world_size = dist.get_world_size()
    config.local_rank = rank
    config.sglang_port = config.sglang_ports[rank]

    # Set L_min, L_max... in simple_filter_dataset_grpo
    simple_filter_dataset_grpo.L_min = config.L_min
    simple_filter_dataset_grpo.L_min_cache = config.L_min_cache
    simple_filter_dataset_grpo.L_max = config.L_max
    simple_filter_dataset_grpo.L_max_cache = config.L_max_cache


    # Initialize Accelerator with custom DDP kwargs
    ddp_kwargs = DistributedDataParallelKwargs(
        find_unused_parameters=True
    )
    dist_kwargs = InitProcessGroupKwargs(
        timeout=timedelta(seconds=60000),
        init_method=f"tcp://{config.master_addr}:{config.master_port}",
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        mixed_precision=config.mixed_precision,
        kwargs_handlers=[ddp_kwargs, dist_kwargs],
    )
    config.rank = dist.get_rank()

    # show config
    logger.info(f"Configuration: {OmegaConf.to_yaml(config)}")

    resolved_dict = OmegaConf.to_container(config, resolve=True, throw_on_missing=False)
    config: MultimodalGRPOConfig = SimpleNamespace(**resolved_dict)

    # get a temp tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    config.tokenizer = tokenizer # FIXME: is it possible to set arbitrary type?
    
    # Prepare dataset
    if "," in config.dataset_type:
        dataset_types = config.dataset_type.split(",")
    else:
        dataset_types = [config.dataset_type]

    train_datasets = []
    eval_datasets = []
    for ds_type in dataset_types:
        this_train_dataset = create_dataset(ds_type, config, dataset_class=Real3DDataset, split="train", sft=False)
        this_eval_dataset = create_dataset(ds_type, config, dataset_class=Real3DDataset, split="val", sft=False)

        train_datasets.append(this_train_dataset)
        eval_datasets.append(this_eval_dataset)

        

    # 获取特征维度
    feature_dim = train_datasets[0].feature_dim
    modality_dims = train_datasets[0].modality_dims
    modality_order = train_datasets[0].modality_order

    train_dataset = MergedDataset(train_datasets)
    eval_dataset = MergedDataset(eval_datasets)

    # train_dataset = create_dataset(config.dataset_type, config, dataset_class=Real3DDataset, split="train")
    # eval_dataset = create_dataset(config.dataset_type, config, dataset_class=Real3DDataset, split="val")

    config.modality_order = modality_order
    config.modality_dims = modality_dims
    config.feature_dim = feature_dim

    if rank == 0:
        print(f"Modality order: {config.modality_order}"
            f"\nModality dims: {config.modality_dims}"
            f"\nFeature dim: {config.feature_dim}")

    config.num_steps = len(train_dataset) // config.rollout_batch_size

    
    # Initialize distributed GRPO trainer
    trainer = MultimodalDistributedGRPO(config, accelerator)

    # update configs to wandb
    if trainer.is_main_training_process:
        wandb.config.update({
            "modality_order": config.modality_order,
            "modality_dims": config.modality_dims,
            "feature_dim": config.feature_dim,
            "num_steps": config.num_steps,
        }, allow_val_change=True)

    # log some training examples
    for ds_type, this_train_dataset in zip(dataset_types, train_datasets):
        if trainer.is_main_training_process:
            print(f"Training examples for [{ds_type}]:".center(50, "="))
            for i in range(5):
                example = random.choice(this_train_dataset)
                print(f"Scene ID: {example['scene_id']}, Objects: {example['object_ids']}")
                print(f"Description: {example['description']}")
                print(f"Expected Response: {example['expected_response']}")
                # estimate token length
                print(f"Token length: {count_example_tokens(example, tokenizer)}")
                print("=" * 50)

            # log to wandb artifact
            artifact = wandb.Artifact(
                name=f"train_dataset_{ds_type}",
                type="dataset",
                description=f"Training dataset example for {ds_type}",
            )

            # create a pandas dataframe with 5 examples
            examples = []
            for i in range(10):
                example = this_train_dataset[i] # fix to first 10 examples - for better comparability
                examples.append({
                    "scene_id": example['scene_id'],
                    "object_ids": example['object_ids'],
                    "description": example['description'],
                    "expected_response": example['expected_response'],
                })
            df = pd.DataFrame(examples)
            table = wandb.Table(dataframe=df)
            artifact.add(table, "examples")
            wandb.log_artifact(artifact)

    
    
    # Convert datasets to list format for easier handling
    # train_data = [train_dataset[i] for i in range(len(train_dataset))]
    # eval_data = [eval_dataset[i] for i in range(len(eval_dataset))]
    
    # Evaluate model before training (main process only)
    if trainer.is_main_process and False:
        logger.info("\nInitial model evaluation before finetuning:")
        pre_metrics = evaluate_model(
            trainer.unwrapped_model,
            trainer.tokenizer,
            eval_dataset[:config.eval_size],
            trainer.accelerator.device,
            max_new_tokens=config.max_completion_length,
        )
    
    # Train model
    try:
        logger.info("Starting RL fine-tuning using distributed GRPO...")
        # trainer.train(train_dataset, CombinedReward(do_max_reward_normalize=False))
        trainer.train(train_dataset, CombinedReward(
            do_max_reward_normalize=False, 
            logp_factor_correct=config.logp_factor_correct,
            logp_factor_wrong=config.logp_factor_wrong,
        ))
    except Exception as e:
        logger.error(f"Error during training: {e}")
        print_exc(file=sys.stdout)
        raise e
    
    # Evaluate model after training (main process only)
    if trainer.is_main_process and False:
        logger.info("Final model evaluation after GRPO RL fine-tuning:")
        post_metrics = evaluate_model(
            trainer.unwrapped_model,
            trainer.tokenizer,
            eval_dataset[:config.eval_size],
            trainer.accelerator.device,
            max_new_tokens=config.max_completion_length,
        )
        
        # Log improvement
        improvement = {
            "id_accuracy_improvement": post_metrics["avg_id_accuracy"] - pre_metrics["avg_id_accuracy"],
            "position_accuracy_improvement": post_metrics["avg_position_accuracy"] - pre_metrics["avg_position_accuracy"],
            "size_accuracy_improvement": post_metrics["avg_size_accuracy"] - pre_metrics["avg_size_accuracy"],
            "overall_accuracy_improvement": post_metrics["avg_overall_accuracy"] - pre_metrics["avg_overall_accuracy"],
            "perfect_match_improvement": post_metrics["perfect_match_rate"] - pre_metrics["perfect_match_rate"]
        }
        
        for metric, value in improvement.items():
            logger.info(f"{metric}: {value:.4f}")
        
        logger.info(f"Training complete! Model saved to {config.output_dir}")


@hydra.main(config_path="configs", config_name="multimodal_grpo", version_base=None)
def entry(config: MultimodalGRPOConfig):
    """
    Entry point for the script.
    """
    OmegaConf.set_struct(config, False)

    num_gpus = torch.cuda.device_count()

    config.output_dir = HydraConfig.get().runtime.output_dir

    # pre-allocate sglang ports, duplicate N times, N = num_inference_gpus
    config.sglang_ports = [find_free_port() for _ in range(num_gpus)]
    print(f"Pre-allocated sglang ports: {config.sglang_ports}")

    # Run the main function
    mp.spawn(main, args=(config,), nprocs=num_gpus, join=True)

if __name__ == "__main__":
    # log_format = "[%(asctime)s %(name)s %(levelname)s] %(message)s"
    # logging.basicConfig(format=log_format, level=logging.INFO, datefmt="%I:%M:%S")
    # mp.set_start_method("spawn")
    entry()
