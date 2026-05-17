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
from icecream import ic
import contextlib
from tqdm.auto import tqdm
import hydra
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


set_start_method("spawn", force=True)

# Import the multimodal model and dataset classes
from dist_tools import all_gather_vlen, all_gather_vdim, model_to_device
from apeiria_mllm import MultimodalLanguageModelDecoderOnly, find_free_port
from apeiria_lm_prog_to_thinking import Real3DDataset, Real3DObjectInfoDataset, Real3DFilterDataset
from simple_filter_dataset_grpo import combined_reward, parse_response, calculate_position_similarity, calculate_size_similarity
from qwen_helpers import apply_qwen_template, count_example_tokens, batch_count_tokens
from train_apeiria_mllm import create_dataset
from cot_rl_config import MultimodalGRPOConfig


# logger = get_logger(__name__)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# define a typevar for dataset class
Synthetic3DDatasetType = Union[
    Real3DDataset, Real3DObjectInfoDataset, Real3DFilterDataset
]

DATASET_CLSMAP: Dict[str, Callable[..., Synthetic3DDatasetType]] = {
    "sr3d": Real3DDataset,
    "sr3d_object_info": Real3DObjectInfoDataset,
    "sr3d_filter": Real3DFilterDataset,
    "nr3d": Real3DDataset,
}


def print_once(string):
    try:
        print_once.printed
    except AttributeError:
        print_once.printed = set()

    if string not in print_once.printed:
        print(string)
        print_once.printed.add(string)

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def recursive_to_device(obj, device):
    """Recursively move tensors in a nested structure to the specified device."""
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    elif isinstance(obj, list):
        return [recursive_to_device(item, device) for item in obj]
    elif isinstance(obj, dict):
        return {key: recursive_to_device(value, device) for key, value in obj.items()}
    else:
        logger.warning(f"Unsupported type {type(obj)} in recursive_to_device")
        return obj

def collate_fn_simple(examples):
    return examples

def collate_fn(examples, device):
    """Collate function for DataLoader"""
    # Extract object features and masks
    object_features = [example["object_feature"] for example in examples]
    object_masks = [example["object_mask"] for example in examples]

    # tensorize
    if isinstance(object_features[0], np.ndarray):
        object_features = [torch.from_numpy(f).float() for f in object_features]
    if isinstance(object_masks[0], np.ndarray):
        object_masks = [torch.from_numpy(m) for m in object_masks]
    
    # Extract instructions and responses
    instructions = [example["description"] for example in examples]
    responses = [example["expected_response"] for example in examples]

    if "image_embeds" in examples[0]:
        image_embeds = [example["image_embeds"] for example in examples]
    
    # Create a batch
    batch = {
        "object_features": recursive_to_device(object_features, device),
        "object_masks": recursive_to_device(object_masks, device),
        "instructions": instructions,
        "responses": responses,
        "scene_ids": [example["scene_id"] for example in examples],
        "object_ids": [example["object_ids"] for example in examples],
        "scanrefer_id": [example.get("scanrefer_id", f"scene_{i}") for i, example in enumerate(examples)],
        **({"image_embeds": image_embeds} if "image_embeds" in examples[0] else {}),
    }

    # add all other fields
    # for key in examples[0].keys():
    #     if key not in batch:
    #         batch[key] = [example[key] for example in examples]
    
    return batch


def prepare_model_inputs(batch, model, tokenizer):
    """Prepare inputs for the model from batch data"""
    # Process object features
    object_set_embeds = []
    for features, mask in zip(batch["object_features"], batch["object_masks"]):
        # Apply mask to features
        valid_features = features[mask]
        object_set_embeds.append([valid_features])

    # For Qwen, add chat template
    instructions = batch["instructions"]
    responses = batch["responses"] if "responses" in batch else [None] * len(instructions)
    if "qwen" in tokenizer.__class__.__name__.lower():
        print_once("Applying Qwen template to instructions...")
        instructions_responses = [apply_qwen_template(inst, tokenizer, resp) for inst, resp in zip(instructions, batch["responses"])]
        instructions = [inst_resp[0] for inst_resp in instructions_responses]

    
    # Prepare inputs for the model
    model_inputs = {
        "instructions": instructions,
        "object_set_embeds": object_set_embeds,
    }

    # Add responses for training
    if "responses" in batch and batch["responses"][0] is not None:
        model_inputs["responses"] = responses

    return model_inputs


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
            self.lm.config.use_cache = False  # we don't generate, so we don't need cache in training

            if self.config.compile_train_model:
                logger.info(f"Rank {self.process_idx} - Compiling model")
                self.model = torch.compile(self.model)

            self.model = DDP(self.model, process_group=self.training_group, find_unused_parameters=True)

            # Setup reference model
            self.update_ref_model()

        else:
            if not self.config.use_sglang_for_generation:
                # Use DDP-based inference
                self.model = self.model.to(self.accelerator.device)
                self.lm.config.use_cache = True  # we need cache in inference
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
            self.ref_model = copy.deepcopy(self.unwrapped_model)
            if self.config.offload_reference_model:
                # Offload reference model to CPU, and load to GPU when used, and move back when not used
                self.ref_model = self.ref_model.to("cpu")
            self.ref_model.eval()
            for param in self.ref_model.parameters():
                param.requires_grad = False

            logger.debug(f"Rank {self.process_idx} - Reference model updated")

    def setup_model_and_tokenizer(self):
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name, padding_side="left")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load language model
        language_model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation=self.config.attn_implementation,
            device_map=self.process_idx,
        )
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
            language_model = get_peft_model(language_model, lora_config)

            # self.prompt_continue("Before loading LoRA checkpoint...")
            
            if self.config.load_checkpoint:
                # FIXME: loading on GPU maybe cause all process take up GPU memory on main rank
                #   Possible fix: load on CPU, then move to GPU after all adapter is loaded
                #   However, pissa init is much much faster on GPU, but if pissa init, no need to load adapter, separate the code.

                logger.info(f"Loading LoRA checkpoint from {self.config.load_checkpoint}...")
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
            
            if self.is_main_process:
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
            self.lm.config.use_cache = False
            if self.config.use_gradient_checkpointing:
                self.lm.gradient_checkpointing_enable()
    
    def setup_optimizer_and_scheduler(self):
        # Create optimizer
        if self.config.optimizer_type == 'adamw':
            # 创建AdamW优化器
            self.optimizer = torch.optim.AdamW(
                [params for params in self.model.parameters() if params.requires_grad],
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        
        elif self.config.optimizer_type == 'adafactor':
            # 创建Adafactor优化器 - 注意Adafactor不需要scheduler
            self.optimizer = Adafactor(
                [params for params in self.model.parameters() if params.requires_grad],
                scale_parameter=False, relative_step=False, warmup_init=False,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
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
                torch.zeros((len(batch["instructions"]), 1), dtype=torch.long, device=self.accelerator.device)
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
                    top_k=-1,
                    return_dict=True,
                )
                
                completion_ids = outputs["completion_ids"]
                inputs_embeds = outputs["inputs_embeds"]
                inputs_mask = outputs["attention_masks"]

                if isinstance(completion_ids, list):
                    all_completion_ids.extend([torch.tensor(completion_id, dtype=torch.long, device=self.accelerator.device).unsqueeze(0) for completion_id in completion_ids])
                else:
                    all_completion_ids.append(completion_ids) # already a tensor
                all_inputs_embeds.append(inputs_embeds) # [B, L, D]
                all_inputs_mask.append(inputs_mask) # [B, L]

        # Pad right side of completion_ids
        # logger.debug(f"{all_completion_ids=}")
        max_completion_len = max([completion_id.size(-1) for completion_id in all_completion_ids])
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
        
        # return prompt_ids, prompt_mask, completion_ids, completion_mask
        return inputs_embeds, inputs_mask.long(), completion_ids.long(), completion_mask.long()

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

    def compute_log_probs_mm(self, model, inputs_embeds, attention_mask, completion_ids, completion_mask, logits_to_keep):
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
        
        # Get logits from the model
        logits = model(**model_inputs).logits[:, :-1, :]
        # completion_ids = completion_ids[:, 1:]  # Shift right to align with logits
        
        # Select only the last 'logits_to_keep' tokens
        completion_ids = completion_ids[:, -logits_to_keep:]
        logits = logits[:, -logits_to_keep:, :]
        
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

        local_inputs_embeds = inputs_embeds[start_idx:end_idx]
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

    def generate_rollout_data(self, batch_samples, num_generations, max_completion_length, need_old_logits=True, need_ref_logits=True):
        """
        Generate rollout data from multimodal inputs.
        """
        # Prepare batch data
        batch = collate_fn(batch_samples, self.accelerator.device)

        # format instruction with chat template
        batch["instructions"] = [self.apply_chat_template(inst) for inst in batch["instructions"]]
        
        # Generate completions (only on inference processes)
        logger.debug(f"Rank {self.process_idx} - Starting generating rollout traces")
        
        
        inputs_embeds, inputs_mask, completion_ids, completion_mask = self.generate_completions(
            batch, num_generations, max_completion_length
        )

        # logger.info(f"inputs_mask shape: {inputs_mask.size()}, inputs_embeds shape: {inputs_embeds.size()}, completion_ids shape: {completion_ids.size()}, completion_mask shape: {completion_mask.size()}")
        
        self.accelerator.wait_for_everyone()
        
        logger.debug(f"Rank {self.process_idx} - Gathering rollout data")
        
        # Gather data from all inference processes, discard dummy data
        all_inputs_embeds = all_gather_vdim(inputs_embeds,)[:self.config.num_inference_gpus]
        all_inputs_mask = all_gather_vdim(inputs_mask,)[:self.config.num_inference_gpus]
        all_completion_ids = all_gather_vdim(completion_ids,)[:self.config.num_inference_gpus]
        all_completion_mask = all_gather_vdim(completion_mask,)[:self.config.num_inference_gpus]
        
        # Only training processes need to compute log probs and rewards
        if not self.is_training_process:
            return None
        
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
                if need_old_logits:
                    old_log_probs = self.compute_log_probs_dist_mm(
                        self.model, 
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
                if need_ref_logits:
                    with model_to_device(self.ref_model, self.accelerator.device, empty_cache=True): # empty CUDA cache to free ref-model completely
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
            # "object_set_embeds": object_set_embeds,
        }
    
    def compute_reward_advantage(self, rollout_data, reward_function):
        """
        Compute rewards and advantages for a batch of rollout data.
        """
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
        
        num_generations = rollout_data["num_generations"]
        rewards = rewards.view(-1, num_generations)
        avg_reward = rewards.mean().item()
        avgmax_reward = rewards.max(dim=1).values.mean().item()
        
        # Compute advantages
        mean_rewards = rewards.mean(dim=1).repeat_interleave(rollout_data["num_generations"])
        std_rewards = rewards.std(dim=1).repeat_interleave(rollout_data["num_generations"])
        advantages = ((rewards.view(-1) - mean_rewards) / (std_rewards + 1e-4)).unsqueeze(1)
        
        if self.config.normalize_advantages:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-4)
        
        return rewards, advantages, avg_reward, avgmax_reward
    
    def grpo_loss(self, rollout_data, reward_function, beta, epsilon_min, epsilon_max):
        """
        Compute GRPO loss for multimodal inputs.
        """
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
        
        inputs_embeds = inputs_embeds[start_idx:end_idx]
        attention_mask = attention_mask[start_idx:end_idx]
        completion_ids = completion_ids[start_idx:end_idx]
        completion_mask = completion_mask[start_idx:end_idx]
        
        if old_log_probs is not None:
            old_log_probs = old_log_probs[start_idx:end_idx]
        if ref_log_probs is not None:
            ref_log_probs = ref_log_probs[start_idx:end_idx]

        # logger.info(f"inputs_embeds shape: {inputs_embeds.size()}, attention_mask shape: {attention_mask.size()}, completion_ids shape: {completion_ids.size()}, completion_mask shape: {completion_mask.size()}")
        
        # Compute current token log probabilities
        logger.debug(f"Rank {self.process_idx} - Computing token log probs")
        token_log_probs = self.compute_log_probs_mm(self.model, inputs_embeds, attention_mask, 
                                                    completion_ids, completion_mask, logits_to_keep)

        
        # Calculate probability ratio
        if old_log_probs is None:
            # mu=1, always on policy
            old_log_probs = token_log_probs.detach().clone()
        ratio = torch.exp(token_log_probs - old_log_probs)
        
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
        loss = -((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        
        # Sample-wise kl and ratio
        with torch.no_grad():
            rollout_ratio = ((ratio * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).detach()
            rollout_kl = ((kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).detach()
        
        return loss, rollout_ratio, rollout_kl

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
    
    def train(self, train_data, reward_function):
        # Initialize wandb if this is the main process
        if self.is_main_training_process:
            wandb.init(
                project=self.config.wandb_project,
                name=self.config.wandb_run_name,
                # config=OmegaConf.to_container(self.config, resolve=True),
                config=self.config,
            )

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
            
            # Progress bar only on main training process
            progress_bar = tqdm(dataloader,
                                total=len(dataloader),
                                desc=f"Iteration {iteration+1}/{len(dataloader)}", 
                                disable=not self.is_main_training_process)
            
            for step, batch_samples in enumerate(progress_bar):
                # Generate rollout data
                rollout_data = self.generate_rollout_data(
                    batch_samples,
                    self.config.num_generations,
                    self.config.max_completion_length,
                    need_old_logits=(self.config.mu > 1),
                    need_ref_logits=(self.config.beta > 0),
                )
                
                # Ensure all processes are synchronized after rollout generation
                self.accelerator.wait_for_everyone()
                
                # Training processes perform updates
                if self.is_training_process:
                    # Compute rewards and advantages
                    rewards, advantages, avg_reward, avgmax_reward = self.compute_reward_advantage(rollout_data, reward_function)
                    rollout_data["reward"] = rewards
                    rollout_data["advantage"] = advantages
                    rollout_data["avg_reward"] = avg_reward
                    
                    if self.is_main_training_process:
                        logger.info(f"Rewards: {rewards}")
                        logger.info(f"Advantages: {advantages.squeeze()}")
                        logger.info(f"Average Reward: {avg_reward}, Average Max Reward: {avgmax_reward}")
                    
                    for grpo_iter in range(self.config.mu):
                        # Split rollout data into microbatches, backward and accumulate gradients
                        num_micro_batches = len(rollout_data["inputs_embeds"]) // self.config.train_micro_batch_size
                        train_stats = []
                        
                        for micro_batch_data in self.split_rollout_data(rollout_data, self.config.train_micro_batch_size):
                            # Compute loss
                            loss, ratio, kl = self.grpo_loss(
                                micro_batch_data,
                                reward_function,
                                self.config.beta,
                                self.config.epsilon_min,
                                self.config.epsilon_max
                            )
                            
                            loss = loss / num_micro_batches
                            
                            # Backward pass and optimization
                            self.accelerator.backward(loss)
                            train_stats.append({
                                "loss": loss.item(),
                                "kl": kl.detach().cpu(),
                                "ratio": ratio.detach().cpu(),
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
                        
                        all_losses = all_gather_vdim(loss.unsqueeze(0), group=self.training_group)
                        loss = torch.cat(all_losses, dim=0).sum().item()
                        
                        kl = all_gather_vdim(kl, group=self.training_group)
                        ratio = all_gather_vdim(ratio, group=self.training_group)
                        
                        kl = torch.cat(kl, dim=0).view(-1).cpu()
                        ratio = torch.cat(ratio, dim=0).view(-1).cpu()
                        
                        if self.is_main_training_process:
                            logger.info(f"KL: {kl}")
                            logger.info(f"Ratio: {ratio}")
                        
                        lr = self.scheduler.get_last_lr()[0]
                        response_length = rollout_data["completion_lengths"].float().mean().item()
                        
                        if self.is_main_training_process:
                            global_step = step + 1 + len(dataloader) * iteration
                            
                            log_dict = {
                                "loss": loss,
                                "average_reward": avg_reward,
                                "average_max_reward": avgmax_reward,
                                "learning_rate": lr,
                                "iteration": iteration + 1,
                                "step": global_step,
                                "grpo_iter": grpo_iter + 1,
                                "kl": kl.mean().item(),
                                "ratio": ratio.mean().item(),
                                "response_length": response_length,
                            }

                            if grad_norm is not None:
                                log_dict["grad_norm"] = grad_norm.item()

                            wandb.log(log_dict)
                            
                            progress_bar.set_postfix({
                                "loss": loss,
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
                            logger.info("".center(50, "=") + "\n")
                        
                        update_counter += 1
                
                else:
                    update_counter += self.config.mu
                
                # Synchronize model weights between training and inference processes
                if update_counter % self.config.update_iters == 0:
                    logger.debug(f"Rank {self.process_idx} - Synchronizing model weights")
                    if self.config.use_sglang_for_generation:
                        # Push the model to SGLang from training processes
                        if self.is_main_training_process:
                            self.update_lora_for_sglang()

                        self.accelerator.wait_for_everyone()
                        if self.config.use_sglang_for_generation and self.is_inference_process:
                            # restart sglang engine. this is for reloading the LoRA.
                            # FIXME: this is a hack, need to find a better way if we can load LoRA inplace
                            #   But SGLang currently only supports update weights of main model
                            self.unwrapped_model._init_sglang_engine()
                            # self.unwrapped_model._init_sglang_subprocess()
                            # self.unwrapped_model._init_sglang_server()
                        logger.debug(f"Rank {self.process_idx} - Reached here, after restarting sglang engine")
                    else: # for huggingface
                        self.broadcast_model_from_train_to_all()

                logger.debug(f"Rank {self.process_idx} - Reached here, before next batch")
                self.accelerator.wait_for_everyone()

            # save
            if self.is_main_process and (iteration + 1) % self.config.save_iters == 0:
                logger.info(f"Saving model checkpoint at iteration {iteration + 1}")
                self.accelerator.wait_for_everyone()
                unwrapped_model = self.accelerator.unwrap_model(self.model)
                unwrapped_model.save_pretrained(self.config.output_dir, safe_serialization=True)
                
        
        # Save the final model (main process only)
        if self.is_main_process:
            self.accelerator.wait_for_everyone()
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            unwrapped_model.save_pretrained(self.config.output_dir)
            self.tokenizer.save_pretrained(self.config.output_dir)
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




def evaluate_model(model, tokenizer, eval_examples, device, max_new_tokens=200, num_examples=None):
    """
    Evaluates the multimodal model on scene object identification tasks.
    """
    model.eval()
    
    # Select subset of examples if specified
    if num_examples is not None:
        eval_subset = eval_examples[:num_examples]
    else:
        eval_subset = eval_examples
    
    total_examples = len(eval_subset)
    print(f"\n{'='*50}\nEVALUATING ON {total_examples} EXAMPLES\n{'='*50}")
    
    # Metrics to track
    metrics = {
        "total_examples": total_examples,
        "correct_format_count": 0,
        "object_id_accuracy": [],
        "position_accuracy": [],
        "size_accuracy": [],
        "overall_accuracy": [],
        "perfect_matches": 0
    }
    
    for i, example in enumerate(eval_subset):
        print(f"\nEvaluating example {i+1}/{total_examples}")
        
        # Get prompt and ground truth
        prompt = example["description"]
        object_features = example["object_feature"]
        object_mask = example["object_mask"]
        
        # Convert to tensors if needed
        if isinstance(object_features, np.ndarray):
            object_features = torch.from_numpy(object_features).float()
        if isinstance(object_mask, np.ndarray):
            object_mask = torch.from_numpy(object_mask)
        
        # Apply mask to features
        valid_features = object_features[object_mask]
        object_set_embeds = [[valid_features.to(device)]]
        
        # Generate response
        with torch.no_grad():
            response = model.generate(
                instructions=[prompt],
                object_set_embeds=object_set_embeds,
                max_length=max_new_tokens,
                do_sample=True,
                temperature=0.7,
            )[0]
        
        # Check format correctness
        has_thinks = "[APEIRIA THINKS]" in response
        has_speaks = "[APEIRIA SPEAKS]" in response
        correct_format = has_thinks and has_speaks
        
        if correct_format:
            metrics["correct_format_count"] += 1
        
        # Parse the response
        predicted_objects = parse_response(response)
        
        # Get ground truth objects
        true_objects = example.get("objects", [])
        
        # Format true objects for comparison
        formatted_true_objects = []
        for obj in true_objects:
            formatted_true_objects.append({
                "id": obj["id"],
                "x": obj["position"][0],
                "y": obj["position"][1],
                "z": obj["position"][2],
                "width": obj["size"][0],
                "height": obj["size"][1],
                "depth": obj["size"][2]
            })
        
        # Handle case where no objects of the category exist
        if not true_objects:
            category = example.get("category", "object")
            no_objects_reported = (
                "didn't find any" in response.lower() or 
                f"no {category}" in response.lower()
            )
            
            if no_objects_reported:
                metrics["object_id_accuracy"].append(1.0)
                metrics["position_accuracy"].append(1.0)
                metrics["size_accuracy"].append(1.0)
                metrics["overall_accuracy"].append(1.0)
                metrics["perfect_matches"] += 1
            else:
                metrics["object_id_accuracy"].append(0.0)
                metrics["position_accuracy"].append(0.0)
                metrics["size_accuracy"].append(0.0)
                metrics["overall_accuracy"].append(0.0)
            
            print(f"Ground truth: No {category}s in the scene")
            print(f"Model correctly reported no objects: {no_objects_reported}")
            continue
        
        # Create dictionaries for quick lookup
        true_obj_dict = {obj["id"]: obj for obj in formatted_true_objects}
        pred_obj_dict = {obj["id"]: obj for obj in predicted_objects}
        
        # Calculate ID accuracy
        correct_ids = set(true_obj_dict.keys()) & set(pred_obj_dict.keys())
        id_accuracy = len(correct_ids) / len(true_obj_dict) if true_obj_dict else 1.0
        metrics["object_id_accuracy"].append(id_accuracy)
        
        # Calculate position and size accuracy for correctly identified objects
        position_similarities = []
        size_similarities = []
        
        for obj_id in correct_ids:
            true_obj = true_obj_dict[obj_id]
            pred_obj = pred_obj_dict[obj_id]
            
            # Position similarity
            pos_sim = calculate_position_similarity(
                [pred_obj["x"], pred_obj["y"], pred_obj["z"]],
                [true_obj["x"], true_obj["y"], true_obj["z"]]
            )
            position_similarities.append(pos_sim)
            
            # Size similarity
            size_sim = calculate_size_similarity(
                [pred_obj["width"], pred_obj["height"], pred_obj["depth"]],
                [true_obj["width"], true_obj["height"], true_obj["depth"]]
            )
            size_similarities.append(size_sim)
        
        # Average position and size accuracy
        avg_position_accuracy = sum(position_similarities) / len(position_similarities) if position_similarities else 0.0
        avg_size_accuracy = sum(size_similarities) / len(size_similarities) if size_similarities else 0.0
        
        metrics["position_accuracy"].append(avg_position_accuracy)
        metrics["size_accuracy"].append(avg_size_accuracy)
        
        # Calculate overall accuracy (weighted average of ID, position, and size accuracy)
        overall_accuracy = 0.4 * id_accuracy + 0.3 * avg_position_accuracy + 0.3 * avg_size_accuracy
        metrics["overall_accuracy"].append(overall_accuracy)
        
        # Check for perfect match (all objects correctly identified with high accuracy)
        is_perfect = (
            id_accuracy == 1.0 and 
            avg_position_accuracy > 0.9 and 
            avg_size_accuracy > 0.9
        )
        
        if is_perfect:
            metrics["perfect_matches"] += 1
        
        # Print detailed results for this example
        print(f"\nPrompt: {prompt}")
        print(f"Ground truth objects: {len(true_objects)}")
        print(f"Predicted objects: {len(predicted_objects)}")
        print(f"Correctly identified objects: {len(correct_ids)}/{len(true_objects)}")
        print(f"ID accuracy: {id_accuracy:.2f}")
        print(f"Position accuracy: {avg_position_accuracy:.2f}")
        print(f"Size accuracy: {avg_size_accuracy:.2f}")
        print(f"Overall accuracy: {overall_accuracy:.2f}")
        print(f"Perfect match: {'✓' if is_perfect else '✗'}")
        
        # Print the first part of the response
        print("\nModel response (truncated):")
        print(response[:500] + "..." if len(response) > 500 else response)
        print("-" * 50)
    
    # Calculate aggregate metrics
    metrics["format_accuracy"] = metrics["correct_format_count"] / total_examples
    metrics["avg_id_accuracy"] = sum(metrics["object_id_accuracy"]) / total_examples
    metrics["avg_position_accuracy"] = sum(metrics["position_accuracy"]) / total_examples
    metrics["avg_size_accuracy"] = sum(metrics["size_accuracy"]) / total_examples
    metrics["avg_overall_accuracy"] = sum(metrics["overall_accuracy"]) / total_examples
    metrics["perfect_match_rate"] = metrics["perfect_matches"] / total_examples
    
    # Print summary
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    print(f"Format accuracy: {metrics['format_accuracy']:.2f}")
    print(f"Average ID accuracy: {metrics['avg_id_accuracy']:.2f}")
    print(f"Average position accuracy: {metrics['avg_position_accuracy']:.2f}")
    print(f"Average size accuracy: {metrics['avg_size_accuracy']:.2f}")
    print(f"Average overall accuracy: {metrics['avg_overall_accuracy']:.2f}")
    print(f"Perfect match rate: {metrics['perfect_match_rate']:.2f} ({metrics['perfect_matches']}/{total_examples})")
    print("="*50)
    
    model.train()
    return metrics

def setup_distributed(rank: int, config: MultimodalGRPOConfig) -> None:
    """Initialize the distributed environment.

    Args:
        rank: Current process rank
        config: Configuration parameters
    """
    # Determine world size
    world_size = os.environ.get("WORLD_SIZE", None) or torch.cuda.device_count()
    world_size = int(world_size)

    # Set environment variables
    os.environ["MASTER_ADDR"] = config.master_addr
    os.environ["MASTER_PORT"] = config.master_port
    # os.environ["NCCL_DEBUG"] = "INFO"
    os.environ["NCCL_BLOCKING_WAIT"] = "1"
    
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_WORLD_SIZE"] = str(world_size)

    # Set device for this process
    torch.cuda.set_device(rank)

    # Initialize process group
    print(f"[Rank {rank}] Initializing process group")
    try:
        dist.init_process_group(
            backend="nccl",
            init_method=f"tcp://{config.master_addr}:{config.master_port}",
            world_size=world_size,
            rank=rank,
            timeout=timedelta(seconds=60000),
        )
        print(f"[Rank {rank}] Process group initialized successfully")
    except Exception as e:
        print(f"[Rank {rank}] Failed to initialize process group: {e}")
        raise e

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
    train_dataset = create_dataset(config.dataset_type, config, dataset_class=Real3DDataset, split="train")
    eval_dataset = create_dataset(config.dataset_type, config, dataset_class=Real3DDataset, split="val")

    config.modality_order = train_dataset.modality_order
    config.modality_dims = train_dataset.modality_dims
    config.feature_dim = train_dataset.feature_dim

    if rank == 0:
        print(f"Modality order: {config.modality_order}"
            f"\nModality dims: {config.modality_dims}"
            f"\nFeature dim: {config.feature_dim}")

    config.num_steps = len(train_dataset) // config.rollout_batch_size

    

    # Initialize distributed GRPO trainer
    trainer = MultimodalDistributedGRPO(config, accelerator)
    
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
        trainer.train(train_dataset, combined_reward)
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
