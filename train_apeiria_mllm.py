import warnings

# 过滤FutureWarning
warnings.filterwarnings('ignore', category=FutureWarning)

# 过滤pydantic的警告
warnings.filterwarnings('ignore', message='.*UnsupportedFieldAttributeWarning.*')

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import transformers
from transformers import AutoModel, AutoTokenizer, PreTrainedModel, Adafactor
from peft import get_peft_model, LoraConfig, PeftModel
import logging
import argparse
import wandb
import numpy as np
from datetime import datetime
from datetime import timedelta
import random
import re
from tqdm.auto import tqdm
from accelerate import Accelerator, DistributedType, DistributedDataParallelKwargs, InitProcessGroupKwargs
from accelerate.logging import get_logger
from accelerate.utils import set_seed, gather_object
from accelerate import FullyShardedDataParallelPlugin
from torch.distributed.fsdp.fully_sharded_data_parallel import FullOptimStateDictConfig, FullStateDictConfig
from torch.distributed.fsdp import ShardingStrategy, BackwardPrefetch
from typing import List, Dict, Union, Optional, Tuple, Callable
import json
from copy import deepcopy
from icecream import ic
from argparse import Namespace
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from torch.distributed import get_rank, get_world_size
from types import SimpleNamespace
import torch.distributed as dist
import torch.multiprocessing as mp
import shutil
import torch.distributed as dist
from peft import PeftModel # 用于检查和保存LoRA
import pretty_errors

try:
    from galore_torch import GaLoreAdamW, GaLoreAdamW8bit, GaLoreAdafactor
except ImportError:
    # print("GaLore not found, if you want to use GaLore optimizers, please install it.")
    pass

from muon import Muon

from apeiria_mllm_config_schema import Config
from apeiria_mllm import MultimodalLanguageModelDecoderOnly
from apeiria_lm_utils import Synthetic3DDataset, Synthetic3DObjectInfoDataset, MergedDataset, Synthetic3DRelationalDataset
from apeiria_lm_prog_to_thinking import (
    Real3DDataset, 
    Real3DObjectInfoDataset, 
    Real3DFilterDataset, 
    Real3DDatasetWithExternalTrace, 
    Synthetic3DDatasetType, 
    Real3DDatasetFreeformThinking,
    Real3DDatasetWithAttributes,
    Real3DDatasetWithAttributesNew,
    Real3DDenseCaptioningDataset,
    Real3DQADataset,
    Real3DFilterByAttributeDataset,
    Real3DGroundingWithCaptionCoTDataset,
    Templates,
)
from qwen_helpers import apply_qwen_template, count_example_tokens, batch_count_tokens, batch_count_tokens_fast

from liger import LIGER_KERNEL_AVAILABLE, apply_liger_kernel_to_qwen3, apply_liger_kernel_to_qwen2, apply_liger_kernel_to_qwen3_vl



# Setup logging
logger = get_logger(__name__)

Cfg = Namespace()

def find_free_port():
    import socket
    from contextlib import closing
    """动态寻找一个可用的空闲端口"""
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))  # 绑定到一个随机空闲端口
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]  # 返回分配的端口号

# define a typevar for dataset class
# Synthetic3DDatasetType = Union[
#     Synthetic3DDataset, Synthetic3DObjectInfoDataset, Synthetic3DRelationalDataset, 
#     Real3DDataset, Real3DObjectInfoDataset, Real3DFilterDataset
# ]

DATASET_CLSMAP: Dict[str, Callable[..., Synthetic3DDatasetType]] = {
    "synthetic3d": Synthetic3DDataset,
    "synthetic3d_object_info": Synthetic3DObjectInfoDataset,
    "synthetic3d_relational": Synthetic3DRelationalDataset,
    "sr3d": Real3DDataset,
    "sr3d_object_info": Real3DObjectInfoDataset,
    "sr3d_filter": Real3DFilterDataset,
    # "nr3d": Real3DDatasetWithExternalTrace,
    "scanrefer_nocot": Real3DDataset,
    "nr3d_nocot": Real3DDataset,
    "multi3drefer_nocot": Real3DDataset,
    "sr3d_nocot": Real3DDataset,
    "scanrefer": Real3DDataset,
    "nr3d": Real3DDataset,
    "scene-r1": Real3DDatasetFreeformThinking,
    "scannet_attributes": Real3DDatasetWithAttributesNew,
    "scanrefer_dense": Real3DDenseCaptioningDataset,
    "nr3d_dense": Real3DDenseCaptioningDataset,
    "sr3d_dense": Real3DDenseCaptioningDataset,
    "scanqa": Real3DQADataset,
    "sqa3d": Real3DQADataset,
    "msqa": Real3DQADataset,
    "sr3d_filter_attribute": Real3DFilterByAttributeDataset,

    "nr3d_caption_cot": Real3DGroundingWithCaptionCoTDataset,
    "scanrefer_caption_cot": Real3DGroundingWithCaptionCoTDataset,
    "sr3d_caption_cot": Real3DGroundingWithCaptionCoTDataset,
    "multi3drefer_caption_cot": Real3DGroundingWithCaptionCoTDataset,
}

DATASETS_NO_EVAL = ["scene-r1"] # no need to evaluate these datasets

def print_once(string):
    try:
        print_once.printed
    except AttributeError:
        print_once.printed = set()

    if string not in print_once.printed:
        print(string)
        print_once.printed.add(string)


def collate_fn(examples):
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
    
    # if "objects" in examples[0]:
    if all("objects" in example for example in examples): # make sure all examples have "objects" key
        objects = [example["objects"] for example in examples]
    else:
        objects = None
    
    # Create a batch
    batch = {
        "object_features": object_features,
        "object_masks": object_masks,
        "instructions": instructions,
        "responses": responses,
        "scene_ids": [example["scene_id"] for example in examples],
        "object_ids": [example["object_ids"] for example in examples],
        "scanrefer_id": [example["scanrefer_id"] for example in examples],
        "objects": objects,
        **({"image_embeds": image_embeds} if "image_embeds" in examples[0] else {}),
    }
    
    return batch


def prepare_model_inputs(batch, model, tokenizer):
    """Prepare inputs for the model from batch data"""
    # Process object features
    object_set_embeds = []
    for features, mask in zip(batch["object_features"], batch["object_masks"]):
        # Convert to tensor if not already, but shall be tensorized in collate_fn
        # if isinstance(features, np.ndarray):
        #     features = torch.from_numpy(features).float()
        # if isinstance(mask, np.ndarray):
        #     mask = torch.from_numpy(mask).bool()
            
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

    # 2D image embeds
    if "image_embeds" in batch:
        image_embeds = batch["image_embeds"]
        model_inputs["image_embeds"] = image_embeds

    # Add responses for training
    if "responses" in batch and batch["responses"][0] is not None:
        model_inputs["responses"] = responses
        if "qwen" in tokenizer.__class__.__name__.lower():
            model_inputs["responses"] = [inst_resp[1] for inst_resp in instructions_responses]

    # Add grounding targets for training grounding loss
    if "objects" in batch and batch["objects"] is not None and len(batch["objects"]) > 0 and batch["objects"][0] is not None:
        grounding_targets = []
        for obj_list in batch["objects"]:
            # obj_list is a list of dict with keys: id, location, size
            targets = {}
            for obj in obj_list:
                obj_id = obj["id"]
                x, y, z = obj["location"]
                h, w, l = obj["size"]
                targets[obj_id] = torch.tensor([x, y, z, h, w, l], dtype=torch.float32)
            
            # sort by object id and make into tensor
            targets = sorted(targets.items(), key=lambda x: x[0]) # sort by object id
            targets = torch.stack([t[1] for t in targets], dim=0) if len(targets) > 0 else torch.zeros((0, 6), dtype=torch.float32)
            grounding_targets.append(targets)

        model_inputs["grounding_targets"] = grounding_targets # list of [N_obj, 6] tensors
    
    else:
        # raise ValueError("Batch does not contain 'objects' key for grounding targets.")
        model_inputs["grounding_targets"] = None

    return model_inputs

def train_epoch(model, dataloader, optimizer, scheduler, accelerator, args, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    
    progress_bar = tqdm(
        dataloader, 
        desc=f"Training Epoch {epoch}", 
        disable=not accelerator.is_local_main_process
    )
    
    for step, batch in enumerate(progress_bar):
        with accelerator.accumulate(model):
            # Prepare inputs
            loss_type = args.loss_type if hasattr(args, 'loss_type') else "sft"
            model_inputs = prepare_model_inputs(batch, model, args.tokenizer)
            model_inputs['loss_type'] = loss_type
            
            with accelerator.autocast():
                # Forward pass
                outputs = model(**model_inputs)
                loss = outputs.loss
            
            # Backward pass
            accelerator.backward(loss)
            
            if accelerator.sync_gradients and args.max_grad_norm > 0 and args.optimizer != "adafactor":
                # accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                grad_norm = accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            else:
                grad_norm = None
                
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            # Log loss
            loss_value = loss.detach()
            # gather loss
            loss_value = accelerator.gather(loss_value.unsqueeze(0)).mean().float()
            total_loss += loss_value

            grounding_loss = outputs.grounding_loss.item() if hasattr(outputs, 'grounding_loss') else 0
            
            # Update progress bar
            progress_bar.set_postfix({"loss": loss_value.item(), "lr": f"{scheduler.get_last_lr()[0]:.2e}"})
            
            # Log to wandb
            if accelerator.is_local_main_process and step % args.logging_steps == 0:
                log_dict = {
                    "train/loss": loss_value.item(),
                    "train/grounding_loss": grounding_loss,
                    "train/learning_rate": scheduler.get_last_lr()[0],
                    # "train/step": step + epoch * len(dataloader),
                    "train/step": step + args.global_step,
                    "train/epoch": epoch,
                }
                if grad_norm is not None:
                    log_dict["train/grad_norm"] = grad_norm
                wandb.log(log_dict)
            
            # Evaluate
            if step > 0 and step % args.eval_steps == 0:
                evaluate(model, accelerator.unwrap_model(model), args.val_dataloaders, accelerator, args, epoch, step)
                model.train()
            
            # Save checkpoint
            if step > 0 and step % args.save_steps == 0:
                save_checkpoint(model, optimizer, scheduler, args, epoch, step)
    
    return total_loss / len(dataloader)

def evaluate(model, unwrapped_model: MultimodalLanguageModelDecoderOnly, dataloaders: Dict[str, DataLoader], accelerator, args, epoch, step=None):
    """Evaluate the model on multiple datasets"""
    accelerator.wait_for_everyone() # 确保所有进程同步开始评估
    logger.info(f"Rank {accelerator.process_index}: Starting evaluation for epoch {epoch}" + (f", step {step}" if step else ""))


    # clear cache
    torch.cuda.empty_cache()
    model.eval()

    original_device = accelerator.device
    sglang_active = False

    # --- SGLang Setup ---
    if args.eval_use_sglang:
        logger.info(f"Rank {accelerator.process_index}: Attempting to use SGLang for evaluation.")
        
        # 1. Save current LoRA weights
        if accelerator.is_main_process:
            os.makedirs(args.temp_lora_path_eval, exist_ok=True)
            logger.info(f"Rank {accelerator.process_index} - Updating LoRA for vLLM/SGLang")
        
            unwrapped_model.language_encoder.save_pretrained(args.temp_lora_path_eval)
            unwrapped_model.save_non_lm_parameters(args.temp_lora_path_eval)
            # List all files in the update path
            files = os.listdir(args.temp_lora_path_eval)
            logger.info(f"Rank {accelerator.process_index} - Files in update path: {files}")

        accelerator.wait_for_everyone() # wait for main process saving model
        
        # 2. Activate SGLang
        unwrapped_model.activate_sglang()
        sglang_active = True
        logger.info(f"Rank {accelerator.process_index}: SGLang activated.")
    
    # Create a timestamped log file
    log_dir = os.path.join(args.output_dir, "eval_logs")
    if accelerator.is_local_main_process:
        os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_file = os.path.join(log_dir, f"eval_log_epoch{epoch}" + (f"_step{step}" if step is not None else "") + f"_{timestamp}.txt")
    
    all_metrics = {}
    
    # Evaluate on each dataset
    for dataset_name, dataloader in dataloaders.items():
        all_predictions = {}
        all_references = {}
        
        progress_bar = tqdm(
            dataloader, 
            desc=f"Evaluating {dataset_name}", 
            disable=not accelerator.is_local_main_process
        )

        validation_loss, validation_loss_count = torch.tensor(0.0).to(accelerator.device), 0
        
        for batch_idx, batch in enumerate(progress_bar):
            # logger.info(f"Rank {accelerator.process_index}: Before putting batch to GPU | batch {batch_idx} for dataset {dataset_name}")
            # Prepare inputs
            model_inputs = prepare_model_inputs(batch, model, args.tokenizer)

            # logger.info(f"Rank {accelerator.process_index}: Put to GPU done. Starting generate() | batch {batch_idx} for dataset {dataset_name}")
            
            # Generate predictions
            with torch.no_grad():
                with accelerator.autocast():
                    # calculate validation loss
                    # outputs = model(**model_inputs)
                    # loss = outputs.loss
                    # loss = accelerator.gather(loss.unsqueeze(0)).detach().float()
                    # validation_loss = loss + validation_loss # make it tensor and on device
                    # logger.info(f"Rank {accelerator.process_index}: Inside autocast() | batch {batch_idx} for dataset {dataset_name}")

                    validation_loss_count += 1

                    predictions = unwrapped_model.generate(
                        instructions=model_inputs["instructions"],
                        object_set_embeds=model_inputs["object_set_embeds"],
                        image_embeds=model_inputs.get("image_embeds", None),
                        max_length=args.max_new_tokens,
                        num_beams=args.num_beams,
                        do_sample=args.do_sample,
                        top_k=args.top_k,
                        top_p=args.top_p,
                        temperature=args.temperature,
                        use_static_cache=args.compile_model, # torch.compile must use static cache
                    )

            # Store predictions and references
            for i, (pred, scene_id, obj_ids, scanrefer_id) in enumerate(zip(predictions, batch["scene_ids"], batch["object_ids"], batch["scanrefer_id"])):
                key = scanrefer_id
                all_predictions[key] = pred
                all_references[key] = {
                    "scene_id": scene_id,
                    "object_ids": obj_ids,
                    "expected_response": batch["responses"][i] if "responses" in batch else None,
                }

            # Log examples to file at specified frequency
            if accelerator.is_local_main_process and batch_idx % args.eval_logging_frequency == 0:
                with open(log_file, "a") as f:
                    f.write(f"Dataset: {dataset_name}, Batch {batch_idx}, Epoch {epoch}" + (f", Step {step}" if step is not None else "") + "\n")
                    f.write("=" * 80 + "\n\n")
                    
                    for i, (instruction, pred) in enumerate(zip(model_inputs["instructions"], predictions)):
                        scene_id = batch["scene_ids"][i]
                        obj_ids = batch["object_ids"][i]
                        expected = batch["responses"][i] if "responses" in batch else "N/A"
                        
                        f.write(f"Example {i+1} (Scene: {scene_id}, Target Objects: {obj_ids}):\n")
                        f.write(f"Instruction: {instruction}\n\n")
                        f.write(f"Expected Response:\n{expected}\n\n")
                        f.write(f"Model Prediction:\n{pred}\n\n")
                        f.write("-" * 80 + "\n\n")
                    
                    f.write("\n" + "=" * 80 + "\n\n")
        
        # Gather predictions and references from all processes
        all_predictions = gather_object([(k, v) for k, v in all_predictions.items()])
        all_predictions = {k: v for k, v in all_predictions}
        
        all_references = gather_object([(k, v) for k, v in all_references.items()])
        all_references = {k: v for k, v in all_references}
        
        # Calculate metrics
        val_dataset: Synthetic3DDatasetType = dataloader.dataset
        all_reference_text = {k: v["expected_response"] for k, v in all_references.items()}
        log_message, metrics = val_dataset.evaluate(all_predictions, all_reference_text, iou_threshold=args.iou_threshold)

        if accelerator.is_local_main_process:
            print("\n")
            print(f"Evaluated on {dataset_name} dataset:".center(50, "="))
            print(log_message)

        # calculate validation loss, then gather
        validation_loss /= validation_loss_count if validation_loss_count > 0 else 1
        validation_loss = accelerator.gather(validation_loss.unsqueeze(0)).mean().item()
        
        # Add prefix to metrics
        prefixed_metrics = {f"val/{dataset_name}/{k}": v for k, v in metrics.items()}
        prefixed_metrics[f"val/{dataset_name}/loss"] = validation_loss
        
        # Store metrics for this dataset
        all_metrics.update(prefixed_metrics)
        
        # Print some examples for this dataset
        if accelerator.is_local_main_process:
            print(f"\n{dataset_name} Examples:")
            # for i, (key, pred) in enumerate(list(all_predictions.items())[:3]):
            for i in range(3):
                key = random.choice(list(all_predictions.keys()))
                pred = all_predictions[key]
                ref = all_references[key]
                print(f"Example {i+1}:")
                print(f"Scene: {ref['scene_id']}, Objects: {ref['object_ids']}")
                print(f"Expected: {ref['expected_response']}")
                print()
                print(f"\nPredicted: {pred}")
                print("\n" + ("-" * 50))

            print("\n" + f"Evaluation on {dataset_name} complete.".center(50, "="))
    
    # Log all metrics
    if accelerator.is_local_main_process:
        if step is not None:
            all_metrics["val/step"] = step + epoch * len(args.train_dataloader)
        
        all_metrics["val/epoch"] = epoch
        wandb.log(all_metrics)

    if sglang_active:
        # 4. Deactivate SGLang
        logger.info(f"Rank {accelerator.process_index}: Deactivating SGLang.")
        unwrapped_model.deactivate_sglang()
        torch.cuda.empty_cache()
        
        # 5. Move model back to GPU
        unwrapped_model.to(original_device)
        torch.cuda.empty_cache()
        logger.info(f"Rank {accelerator.process_index}: Model moved back to GPU.")
    
    return all_metrics


def save_checkpoint(model, optimizer, scheduler, args, epoch, step=None):
    """Save model checkpoint"""
    if args.no_save:
        logger.info("Skipping save checkpoint as no_save is set.")
        return

    if not args.accelerator.is_local_main_process:
        return
    
    # Create checkpoint directory
    checkpoint_dir = os.path.join(
        args.output_dir, 
        f"checkpoint-epoch{epoch}" + (f"-step{step}" if step is not None else "")
    )
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Unwrap model
    if args.accelerator.distributed_type == DistributedType.FSDP:
        args.accelerator.save_state(checkpoint_dir)
    else:
        unwrapped_model: MultimodalLanguageModelDecoderOnly = args.accelerator.unwrap_model(model)
        
        # Save model parameters
        unwrapped_model.save_pretrained(
            checkpoint_dir # , accelerator=args.accelerator
        )
        
        # Save optimizer and scheduler
        torch.save(optimizer.state_dict(), os.path.join(checkpoint_dir, "optimizer.pt"))
        torch.save(scheduler.state_dict(), os.path.join(checkpoint_dir, "scheduler.pt"))
    
    # Save training state information
    training_state = {
        "epoch": epoch,
        "step": step,
        "global_step": args.global_step,
        "current_stage_index": getattr(args, 'current_stage_index', 0),
        "curriculum_stages": getattr(args, 'curriculum_stages', []),
        "random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_random_state": torch.get_rng_state(),
        "cuda_random_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    
    with open(os.path.join(checkpoint_dir, "training_state.json"), "w") as f:
        # Convert numpy arrays and torch tensors to lists for JSON serialization
        state_to_save = {}
        for key, value in training_state.items():
            if key == "numpy_random_state":
                state_to_save[key] = {
                    "method": value[0],
                    "state": value[1],
                    "pos": int(value[2]),
                    "has_gauss": int(value[3]),
                    "cached_gaussian": float(value[4])
                }
            elif key == "torch_random_state":
                state_to_save[key] = value.tolist()
            elif key == "cuda_random_state":
                state_to_save[key] = [state.tolist() for state in value] if value else None
            elif key == "curriculum_stages":
                # Convert curriculum stages to serializable format
                state_to_save[key] = [{"epoch": stage.epoch, "datasets": stage.datasets} for stage in value]
            else:
                state_to_save[key] = value
        json.dump(state_to_save, f, indent=4, default=str)
    
    # Save training arguments
    with open(os.path.join(checkpoint_dir, "training_args.json"), "w") as f:
        json.dump(args.run_config, f, indent=4, default=str)
    
    print(f"Saved checkpoint to {checkpoint_dir}")

def load_checkpoint(model, optimizer, scheduler, args, checkpoint_path):
    """Load model checkpoint and training state"""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    def tuplize_random_state(state_list):
        # it is saved as list in json, convert back to tuple - and all nested contents also need to be tuples
        tuple_types = (list, tuple)
        if isinstance(state_list, list):
            return tuple(tuplize_random_state(item) if isinstance(item, tuple_types) else item for item in state_list)

    def parse_json_numpy_array(state_str: str) -> np.ndarray:
        # return np.array(json.loads(obj))
        numbers_str = state_str.replace('[', '').replace(']', '').replace('\n', ' ')
        state_array = np.fromstring(numbers_str, dtype=int, sep=' ')
        return state_array
    
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    
    # Load training state
    training_state_path = os.path.join(checkpoint_path, "training_state.json")
    if os.path.exists(training_state_path):
        with open(training_state_path, "r") as f:
            training_state = json.load(f)
        
        # Restore random states
        if "random_state" in training_state:
            # random.setstate(tuple(training_state["random_state"]))
            random.setstate(tuplize_random_state(training_state["random_state"]))
        
        if "numpy_random_state" in training_state:
            np_state = training_state["numpy_random_state"]
            np.random.set_state((
                np_state["method"],
                # np_state["state"],
                parse_json_numpy_array(np_state["state"]),
                np_state["pos"],
                np_state["has_gauss"],
                np_state["cached_gaussian"]
            ))
        
        # Due to version changes of PyTorch, we comment out torch random state restoration for now
        # if "torch_random_state" in training_state:
        #     torch.set_rng_state(torch.tensor(training_state["torch_random_state"], dtype=torch.uint8))
        
        # if "cuda_random_state" in training_state and training_state["cuda_random_state"]:
        #     cuda_states = [torch.tensor(state, dtype=torch.uint8) for state in training_state["cuda_random_state"]]
        #     torch.cuda.set_rng_state_all(cuda_states)
        
        # Store training state info
        resume_epoch = training_state.get("epoch", None)
        resume_step = training_state.get("step", None)
        global_step = training_state.get("global_step", 0)
        resume_stage_index = training_state.get("current_stage_index", 0)
        
        logger.info(f"Resuming from epoch {resume_epoch}, global step {global_step}, curriculum stage {resume_stage_index}")
    else:
        logger.warning(f"Training state file not found: {training_state_path}")
        resume_epoch = None
        resume_step = None
        global_step = 0
        resume_stage_index = 0

    # try restore resume_epoch from filename
    if resume_epoch is None:
        match = re.search(r"checkpoint-epoch(\d+)", os.path.basename(checkpoint_path))
        if match:
            resume_epoch = int(match.group(1))
        else:
            resume_epoch = 0

    if resume_epoch is None:
        resume_epoch = 0
    
    # Load model, optimizer, and scheduler
    if args.accelerator.distributed_type == DistributedType.FSDP:
        args.accelerator.load_state(checkpoint_path)
    else:
        # NOTE: Load model parameters - should be done previously in get_lm()
        # unwrapped_model: MultimodalLanguageModelDecoderOnly = args.accelerator.unwrap_model(model)
        # lm = unwrapped_model.language_encoder
        # lm.load_pretrained(checkpoint_path)
        # unwrapped_model.load_non_lm_parameters(checkpoint_path)
        # logger.info(f"Loaded model parameters from {checkpoint_path}")
        
        # Load optimizer state
        try:
            optimizer_path = os.path.join(checkpoint_path, "optimizer.pt")
            if os.path.exists(optimizer_path):
                optimizer.load_state_dict(torch.load(optimizer_path, map_location="cpu"))
                logger.info("Loaded optimizer state")
            else:
                logger.warning("Optimizer state file not found")
        except Exception as e:
            logger.error(f"Error loading optimizer state: {e}, using fresh optimizer state")
        
        # Load scheduler state
        # scheduler_path = os.path.join(checkpoint_path, "scheduler.pt")
        # if os.path.exists(scheduler_path):
        #     scheduler.load_state_dict(torch.load(scheduler_path, map_location="cpu"))
        #     logger.info("Loaded scheduler state")
        # else:
        #     logger.warning("Scheduler state file not found - will manually advance scheduler")
        # since we will manually advance scheduler later, we skip loading scheduler state
    
    return resume_epoch, resume_step, global_step, resume_stage_index


def manually_advance_scheduler(scheduler, target_steps):
    """Manually advance scheduler to target steps for old checkpoints without scheduler state"""
    logger.info(f"Manually advancing scheduler to step {target_steps}")
    for _ in range(target_steps):
        scheduler.step()

def get_lm(args) -> Tuple[PreTrainedModel, transformers.PreTrainedTokenizer]:
    model_id = args.model_name
    logger.info(f"Loading model {model_id}...")

    if "deberta" in model_id:
        model = transformers.DebertaV2Model.from_pretrained(model_id)
        tokenizer = transformers.DebertaV2Tokenizer.from_pretrained(model_id)
        args.lora_target_modules = "dense,query_proj,key_proj,value_proj" # for debertav2

    elif "bert" in model_id:
        model = transformers.BertModel.from_pretrained(model_id)
        tokenizer = transformers.BertTokenizer.from_pretrained(model_id)
        args.lora_target_modules = "dense,decoder,query,key,value" # for bert

    elif "Qwen2.5" in model_id:
        model = transformers.Qwen2ForCausalLM.from_pretrained(
            model_id, 
            torch_dtype=torch.bfloat16,
            # attn_implementation="flash_attention_2",
            attn_implementation=args.attn_implementation,
        )
        # model = transformers.AutoModelForCausalLM.from_pretrained(
        #     model_id,
        #     torch_dtype=torch.bfloat16,
        #     attn_implementation="flash_attention_2",
        # )
        tokenizer = transformers.Qwen2TokenizerFast.from_pretrained(model_id)
        tokenizer.padding_side = "left"
        args.lora_target_modules = "down_proj,up_proj,gate_proj,q_proj,k_proj,v_proj,o_proj" # for qwen2

        if LIGER_KERNEL_AVAILABLE:
            logger.info("Applying LIGER kernel to Qwen2.5 model...")
            apply_liger_kernel_to_qwen2(model)

    elif "Qwen3" in model_id:
        if "Qwen3-VL" in model_id:
            model = transformers.Qwen3VLForConditionalGeneration.from_pretrained(
                model_id, 
                torch_dtype=torch.bfloat16,
                # attn_implementation="flash_attention_2",
                attn_implementation=args.attn_implementation,
            )
            if LIGER_KERNEL_AVAILABLE:
                logger.info("Applying LIGER kernel to Qwen3-VL model...")
                apply_liger_kernel_to_qwen3_vl(model)

            logger.info(f"Offloading vision encoder for Qwen3-VL")
            model.model.visual = model.model.visual.to("cpu")
            torch.cuda.empty_cache()

        else:
            model = transformers.Qwen3ForCausalLM.from_pretrained(
                model_id, 
                torch_dtype=torch.bfloat16,
                # attn_implementation="flash_attention_2",
                attn_implementation=args.attn_implementation,
            )
            if LIGER_KERNEL_AVAILABLE:
                logger.info("Applying LIGER kernel to Qwen3 model...")
                apply_liger_kernel_to_qwen3(model)


        tokenizer = transformers.Qwen2TokenizerFast.from_pretrained(model_id)
        tokenizer.padding_side = "left"
        args.lora_target_modules = "down_proj,up_proj,gate_proj,q_proj,k_proj,v_proj,o_proj"

    else:
        raise ValueError(f"Model {model_id} not supported.")

    # load model
    if args.checkpoint_path != "":
        if args.lora_rank < 0:
            # load full model
            logger.info(f"Loading model from {args.checkpoint_path}...")
            lm = model.language_encoder
            lm.load_pretrained(args.checkpoint_path)

        # load non-lm parameters
        # model.load_non_lm_parameters(args.checkpoint_path)

    # set sync batchnorm
    if args.sync_bn:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model) 

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs = {"use_reentrant": False})
    
    if args.compile_model:
        # use static cache
        model.generation_config.cache_implementation = "static"

    # apply LoRA
    model = get_peft_lm(model, args)

    # move to device, since load lora can be on CPU
    if args.accelerator.distributed_type != DistributedType.FSDP:
        model = model.to(args.accelerator.device)

    if args.accelerator.is_local_main_process and args.lora_rank > 0:
        # print("Trainable parameters in LVLM:")
        model.print_trainable_parameters()

    if args.compile_model:
        logger.info("Compiling model...")
        # model = torch.compile(model)
        # model = torch.compile(model, mode="reduce-overhead")
        # model = torch.compile(model, mode="max-autotune")
        model.generate = torch.compile(model.generate)

    return model, tokenizer

def get_peft_lm(model: PreTrainedModel, args):
    lm = model

    if args.lora_rank == 0:
        logger.info(f"No LoRA applied as lora_rank == {args.lora_rank}.")
        logger.info("Freezing all LVLM parameters...")
        for p in lm.parameters():
            p.requires_grad = False
        return model

    elif args.lora_rank == -1:
        logger.info("Full fine-tuning.")
        for p in lm.parameters():
            p.requires_grad = True
        return model

    modules_to_save = [
        # currently, No, all params inside the LM
    ]
    if args.unfreeze_word_embedding:
        if args.lora_word_embedding:
            args.lora_target_modules = "word_embeddings,embed_tokens,lm_head," + args.lora_target_modules
        else:
            modules_to_save.extend([
                "embed_tokens",
                "lm_head",
                "word_embeddings",
            ])
    
    peft_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        inference_mode=False, 
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=args.lora_target_modules.split(","),
        modules_to_save=modules_to_save,
        init_lora_weights="pissa_niter_48" if args.use_pissa else True,
        use_rslora=args.use_rslora,
        use_dora=args.use_dora,
    )

    # apply LORA to the LLM
    logger.info(f"Applying LoRA in {type(lm)} submodel...")
    lm = get_peft_model(lm, peft_config)

    # load checkpoint
    if args.checkpoint_path != "" and not args.only_load_adapter:
        logger.info(f"Loading LoRA checkpoint from {args.checkpoint_path}...")
        message = lm.load_adapter(
            model_id=args.checkpoint_path,
            adapter_name="default",
            torch_device="cpu",
            # is_trainable=args.trainable_lora_in_finetune,
            is_trainable=True,
        )
        logger.info(message)
        torch.cuda.empty_cache()


    return lm


def get_optmizer(optimizer_name: str, model: nn.Module, args):
    if optimizer_name == "galore":
        assert args.lora_rank == -1, "GaLore optimizer is not supported with LoRA."

    lm_params: dict[str, nn.Parameter] = {}
    non_lm_params: dict[str, nn.Parameter] = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            if "language_encoder" in name:
                lm_params[name] = param
            else:
                non_lm_params[name] = param

    if optimizer_name == "galore":
        # split all Linear layers in LM to be optimized by GaLore
        galore_params = []
        non_galore_lm_params = []

        module_dict = dict(model.named_modules())

        for name, param in lm_params.items():
            module_name = ".".join(name.split(".")[:-1])
            if module_name in module_dict and isinstance(module_dict[module_name], nn.Linear):
                if "embed_tokens" in name or "lm_head" in name:
                    param.requires_grad = False
                    logger.info(f"Freezing {name}.")
                else:
                    galore_params.append(param)
                    logger.info(f"Optimizing {name} with GaLore.")
            else:
                non_galore_lm_params.append(param)
                logger.info(f"Optimizing {name} without GaLore.")

        param_list_with_lr = [
            {"params": non_galore_lm_params, "lr": args.lr},
            {"params": galore_params, "lr": args.lr, "rank": args.galore_rank, "update_proj_gap": 200, "scale": 2.0, "proj_type": 'std'},
            {"params": list(non_lm_params.values()), "lr": args.lr_non_lm}, # e.g., adapters, spatial embeddings
        ]

    elif optimizer_name == "muon":
        muon_params = [
            p
            for name, p in lm_params.items()
            if p.ndim >= 2 and "embed_tokens" not in name and "lm_head" not in name
        ]
        adamw_params = [
            p
            for name, p in lm_params.items()
            if not (
                p.ndim >= 2 and "embed_tokens" not in name and "lm_head" not in name
            )
        ] + list(non_lm_params.values())

    else:
        param_list_with_lr = [
            {"params": list(lm_params.values()), "lr": args.lr},
            {"params": list(non_lm_params.values()), "lr": args.lr_non_lm},
        ]
    
    # Create optimizer
    if optimizer_name == 'adamw':
        optimizer = torch.optim.AdamW(
            param_list_with_lr,
            weight_decay=args.weight_decay
        )
    elif optimizer_name == 'adafactor':
        optimizer = Adafactor(
            param_list_with_lr,
            weight_decay=args.weight_decay,
            scale_parameter=False, relative_step=False, warmup_init=False,
        )
    elif optimizer_name == 'galore':
        optimizer = GaLoreAdamW(
            param_list_with_lr,
            weight_decay=args.weight_decay,
        )

    elif optimizer_name == 'muon':
        return Muon(
            lr=args.lr,
            wd=args.weight_decay,
            muon_params=muon_params,
            adamw_params=adamw_params,
        )

    return optimizer

def create_dataset(dataset_type, args, dataset_class=None, split="train", **kwargs):
    """创建指定类型和分割的数据集"""
    if dataset_class is None:
        dataset_class = DATASET_CLSMAP[dataset_type]

    enforce_nocot = False
    if "nocot" in dataset_type:
        enforce_nocot = True
        dataset_type = dataset_type.replace("_nocot", "")

    num_scenes = args.num_scenes
    if split == "val":
        num_scenes = args.num_scenes // 20  # 验证集更小, only for synthetic datasets
    
    dataset = dataset_class(
        name=dataset_type,
        split=split,
        shuffle_objects=args.shuffle_objects if split == "train" else False,
        num_scenes=num_scenes,
        objects_per_scene=args.objects_per_scene,
        room_size=args.room_size,
        max_objects=args.max_objects,
        min_objects_per_class=args.min_objects_per_class,
        max_objects_per_class=args.max_objects_per_class,
        seed=args.seed if split == "train" else args.seed + 1,
        fix_template=args.fix_template,
        add_thinking_trace=args.add_thinking_trace and not enforce_nocot,
        adjust_scene_layouts=args.adjust_scene_layouts,
        relational_data_ratio=args.relational_data_ratio,
        add_full_thinking_trace_for_relational=args.add_full_thinking_trace_for_relational,
        add_partial_full_thinking_trace_for_relational=args.add_partial_full_thinking_trace_for_relational,
        add_partial_full_thinking_trace_for_filter=args.add_partial_full_thinking_trace_for_filter,
        add_full_thinking_trace_for_filter_in_relational=args.add_full_thinking_trace_for_filter_in_relational,
        only_add_positive_relations=args.only_add_positive_relations,
        pre_filter_objects=args.pre_filter_objects,
        ratio=args.ratio,
        max_filter_objects=args.max_filter_objects,
        use_clip_class_embedding=args.use_clip_class_embedding,
        clip_model_name=args.clip_model_name,
        cuda_device=dist.get_rank(),
        use_proposal_feature=args.use_proposal_feature,
        proposal_type=args.proposal_type,
        load_from_cache=args.load_from_cache,
        normalize_proposal_feature=args.normalize_proposal_feature,
        no_object_id_input=args.no_object_id_input,
        use_2d_proposal_feature=args.use_2d_proposal_feature,
        add_plans_first=args.add_plans_first,
        tokenizer=args.tokenizer,
        max_traces_per_sample=args.max_traces_per_sample,
        external_traces_path=args.external_traces_path,
        shuffle_traces=args.shuffle_traces,
        # sft=True,
        sft=kwargs.get("sft", True),
        image_encoder=args.image_encoder,
        image_feature_type=args.image_feature_type,
        n_views_in_m_views=args.n_views_in_m_views,
        # use_trainval=False,
        use_trainval=kwargs.get("use_trainval", False),
        add_thinking_trace_prompt=args.add_thinking_trace_prompt,
        add_bracket_in_object_detail=args.add_bracket_in_object_detail,
    )
    
    return dataset


def setup_distributed(rank: int, config: Config) -> None:
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

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    
def main(rank: int, cfg: Config):
    # logger.info(os.environ)
    # avoid SGLang hanging in NCCL init
    setup_distributed(rank, cfg)
    
    # Set random seed for reproducibility
    set_random_seed(cfg.seed)

    # configure torch.cuda, expandable_segment
    # torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')
    # torch._dynamo.config.capture_scalar_outputs = True

    OmegaConf.set_struct(cfg, False)
    cfg.rank = rank
    cfg.world_size = dist.get_world_size()
    cfg.local_rank = rank
    cfg.sglang_port = cfg.sglang_ports[rank]

    # --- Handle Dynamic/Post-processing Defaults ---
    # Accessing Hydra's output directory if you want to use it
    # hydra_output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    # print(f"Hydra output directory: {hydra_output_dir}")

    # # If you *strictly* want the output_dir logic from argparse:
    # if cfg.output_dir == "":
    #     # Note: cfg.wandb_run_name is already resolved by Hydra here
    #     cfg.output_dir = hydra_output_dir

    os.makedirs(cfg.output_dir, exist_ok=True)

    # Handle discrete_location_bin_range default based on room_size
    if cfg.discrete_location_bin_range is None:
        cfg.discrete_location_bin_range = [0.0, cfg.room_size] 

    # NOTE: resolve=True ensures interpolations like ${...} are evaluated
    # throw_on_missing=False prevents errors if some keys are missing (shouldn't happen with schema)
    resolved_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False)
    args: Config | SimpleNamespace = SimpleNamespace(**resolved_dict)
    Cfg.args = args

    # set Templates.DEFAULT_PRECISION
    Templates.DEFAULT_PRECISION = args.location_precision

    # Initialize accelerator
    ddp_kwargs = DistributedDataParallelKwargs(
        broadcast_buffers=False,
        find_unused_parameters=cfg.find_unused_parameters
    )
    dist_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=60000))

    if getattr(cfg, "use_fsdp", False):
        fsdp_plugin = FullyShardedDataParallelPlugin(
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
            limit_all_gathers=True,
            use_orig_params=True,
            activation_checkpointing=args.gradient_checkpointing,
        )
        accelerator = Accelerator(
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            # kwargs_handlers=[ddp_kwargs, dist_kwargs]
            kwargs_handlers=[dist_kwargs],
            fsdp_plugin=fsdp_plugin,
        )
    else:
        fsdp_plugin = None
        accelerator = Accelerator(
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            kwargs_handlers=[ddp_kwargs, dist_kwargs]
        )

    args.accelerator = accelerator

    if accelerator.is_local_main_process:
        print("Arguments:".center(50, "="))
        for k, v in vars(args).items():
            print(f"{k}: {v}")
        print("=" * 50)

    args.run_config = deepcopy(vars(args))
    
    level_map = { # Map string names to logging constants
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }
    transformers_verbosity_map_func = { # Map string names to transformers functions
        "debug": transformers.utils.logging.set_verbosity_debug,
        "info": transformers.utils.logging.set_verbosity_info,
        "warning": transformers.utils.logging.set_verbosity_warning,
        "error": transformers.utils.logging.set_verbosity_error,
        # "passive": transformers.utils.logging.set_verbosity_passive,
    }

    if accelerator.is_local_main_process:
        log_level_name = cfg.log_level_main.upper()
        transformers_verbosity_name = cfg.transformers_verbosity_main.lower()
        
        log_level = level_map.get(log_level_name, logging.INFO) # Default to INFO
        transformers_verbosity_func = transformers_verbosity_map_func.get(
            transformers_verbosity_name, transformers.utils.logging.set_verbosity_info # Default to info
        )
        
        logging.basicConfig(
            format=cfg.log_format, 
            level=log_level, 
            datefmt=cfg.log_datefmt,
            force=True # Override potential previous basicConfig calls
        )
        transformers_verbosity_func()

    else:
        log_level_name = cfg.log_level_other.upper()
        transformers_verbosity_name = cfg.transformers_verbosity_other.lower()

        log_level = level_map.get(log_level_name, logging.WARNING) # Default to WARNING
        transformers_verbosity_func = transformers_verbosity_map_func.get(
            transformers_verbosity_name, transformers.utils.logging.set_verbosity_warning # Default to warning
        )

        logging.basicConfig(
            format=cfg.log_format, 
            level=log_level, 
            datefmt=cfg.log_datefmt,
            force=True # Override potential previous basicConfig calls
        )
        transformers_verbosity_func()

    if accelerator.is_local_main_process:
        print("\nOmegaConf structure:")
        print(OmegaConf.to_yaml(cfg))

    if args.seed is not None:
        set_seed(args.seed)
    
    # Initialize wandb if main process
    if accelerator.is_local_main_process:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=OmegaConf.to_container(cfg, resolve=True),
            # mode="offline",
        )
    
    # --- Load model ---
    # Load tokenizer and language model
    # if resume_from_checkpoint, set args.checkpoint_path
    if args.resume_from_checkpoint:
        if args.checkpoint_path == "" or args.checkpoint_path is None or args.checkpoint_path == args.resume_from_checkpoint:
            args.checkpoint_path = args.resume_from_checkpoint
        else:
            logger.warning(f"resume_from_checkpoint is set to {args.resume_from_checkpoint}, but checkpoint_path is set to {args.checkpoint_path}. Using checkpoint_path instead.")

    language_model, tokenizer = get_lm(args)

    args.tokenizer = tokenizer

    # Ensure tokenizer has pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    

    # --- Load dataset ---
    # 解析课程学习阶段
    if hasattr(cfg, 'curriculum_stages') and cfg.enable_curriculum:
        curriculum_stages = cfg.curriculum_stages
        # 按照epoch排序
        curriculum_stages = sorted(curriculum_stages, key=lambda x: x.epoch)
        
        # 验证第一个阶段从epoch 0开始
        if curriculum_stages[0].epoch != 0:
            raise ValueError("第一个课程学习阶段必须从epoch 0开始")
        
        # 获取所有阶段使用的数据集类型
        all_dataset_types = set()
        for stage in curriculum_stages:
            all_dataset_types.update(stage.datasets)
    else:
        # 默认：所有epoch使用同一组数据集
        dataset_types = cfg.dataset_type.strip("[]").split(",") if "[" in cfg.dataset_type else [cfg.dataset_type]
        dataset_types = [dt.strip() for dt in dataset_types]
        curriculum_stages = [SimpleNamespace(epoch=0, datasets=dataset_types)]
        all_dataset_types = set(dataset_types)
    
    if accelerator.is_local_main_process:
        print("课程学习阶段:".center(50, "="))
        for i, stage in enumerate(curriculum_stages):
            print(f"阶段 {i+1}: 从Epoch {stage.epoch}开始")
            print(f"   数据集: {', '.join(stage.datasets)}")
        print("=" * 50)

    # 创建所有数据集
    all_train_datasets: Dict[str, Synthetic3DDatasetType] = {}
    all_val_datasets = {}
    
    for dtype in all_dataset_types:
        # 检查数据集类型是否已注册
        if dtype not in DATASET_CLSMAP:
            raise ValueError(f"未知的数据集类型: {dtype}")
        
        # 创建训练数据集
        all_train_datasets[dtype] = create_dataset(dtype, args, split="train")

        if dtype in DATASETS_NO_EVAL:
            if accelerator.is_local_main_process:
                print(f"跳过无需评估的数据集类型: {dtype}")
        else:
            # 创建验证数据集
            all_val_datasets[dtype] = create_dataset(dtype, args, split="val")
    
    # 获取特征维度
    first_dtype = next(iter(all_dataset_types))
    feature_dim = all_train_datasets[first_dtype].feature_dim

    modality_dims = all_train_datasets[first_dtype].modality_dims
    modality_order = all_train_datasets[first_dtype].modality_order

    # show some examples
    train_dataset = MergedDataset(list(all_train_datasets.values()))
    if accelerator.is_local_main_process:
        print("Training examples:".center(50, "="))
        if not args.validation_only:
            for i in range(5):
                # example = train_dataset[i]
                example = random.choice(train_dataset)
                print(f"Scene ID: {example['scene_id']}, Objects: {example['object_ids']}")
                print(f"Description: {example['description']}")
                print(f"Expected Response: {example['expected_response']}")
                # estimate token length
                print(f"Token length: {count_example_tokens(example, tokenizer)}")
                print("=" * 50)

            for dtype, dataset in all_train_datasets.items():
                # log to wandb artifact
                artifact = wandb.Artifact(
                    name=f"train_dataset_{dtype}",
                    type="dataset",
                    description=f"Training dataset example for {dtype}",
                )
                # create a pandas dataframe with 5 examples
                import pandas as pd
                examples = []
                for i in range(10):
                    example = dataset[i] # fix to first 10 examples - for better comparability
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

        # find longest example
        if do_count_length := False and not args.validation_only:
            token_lengths = []
            print("Calculating max token length in training set...")
            for i in tqdm(range(0, len(train_dataset), 100)):
                batch = [train_dataset[j] for j in range(i, min(i+100, len(train_dataset)))]
                batch_tokens = batch_count_tokens_fast(batch, tokenizer)
                token_lengths.extend(batch_tokens['token_count'])

            # print a stat of token length
            token_lengths = np.array(token_lengths)
            print(f"Max token length: {token_lengths.max().item()}, min token length: {token_lengths.min().item()}")
            print(f"Mean token length: {token_lengths.mean().item():.2f}, median token length: {np.median(token_lengths).item()}")
            print(f"Token length 95% percentile: {np.percentile(token_lengths, 95).item():.2f}")
            print(f"Token length 99% percentile: {np.percentile(token_lengths, 99).item():.2f}")
            print(f"Token length STD: {token_lengths.std().item():.2f}")

            # Log to wandb
            wandb.log({
                "train/token_length_max": token_lengths.max().item(),
                "train/token_length_mean": token_lengths.mean().item(),
                "train/token_length_median": np.median(token_lengths).item(),
                "train/token_length_95_percentile": np.percentile(token_lengths, 95).item(),
                "train/token_length_99_percentile": np.percentile(token_lengths, 99).item(),
                "train/token_length_std": token_lengths.std().item(),
            })

            # show longest example
            max_token_idx = token_lengths.argmax()
            example = train_dataset[max_token_idx]
            print("Longest example".center(50, "="))
            print(f"Scene ID: {example['scene_id']}, Objects: {example['object_ids']}")
            print(f"Description: {example['description']}")
            print(f"Expected Response: {example['expected_response']}")
            print("=" * 50)
        
        
        for name, val_dataset in all_val_datasets.items():
            print(f"Validation {name} examples:".center(50, "="))
            for i in range(3):
                # example = val_dataset[i]
                example = random.choice(val_dataset)
                print(f"Scene ID: {example['scene_id']}, Objects: {example['object_ids']}")
                print(f"Description: {example['description']}")
                print(f"Expected Response: {example['expected_response']}")
                print("=" * 50)
    
    # Initialize model
    if cfg.eval_use_sglang:
        sglang_kwargs = {
            "use_sglang": False, # not init for now.
            "sglang_model_path": cfg.model_name,
            "sglang_lora_paths": [cfg.temp_lora_path_eval],
            "sglang_port": cfg.sglang_port,
        }
    else:
        sglang_kwargs = {
            "use_sglang": False,
        }

    model = MultimodalLanguageModelDecoderOnly(
        language_model=language_model,
        tokenizer=tokenizer,
        object_feature_dim=feature_dim,
        max_objects=args.max_objects,
        no_object_in_language_model=args.no_object_in_language_model,
        object_embedding_type=args.object_embedding_type,
        discrete_location_bins=args.discrete_location_bins,
        discrete_location_decay_kernel=args.discrete_location_decay_kernel,
        discrete_location_bin_range=args.discrete_location_bin_range,
        discrete_location_decay_kernel_size=args.discrete_location_decay_kernel_size,
        separate_location_embedding=args.separate_location_embedding,
        dtype=torch.bfloat16,
        modality_dims=modality_dims,
        modality_order=modality_order,
        image_embedding_dim=MultimodalLanguageModelDecoderOnly.image_encoder_to_embedding_dim_map[args.image_encoder] if args.image_encoder else None,
        coeff_grounding_loss=args.coeff_grounding_loss,
        **sglang_kwargs,
    )

    if args.checkpoint_path != "":
        logger.info(f"Loading non-lm part of model from {args.checkpoint_path}...")
        model.load_non_lm_parameters(args.checkpoint_path, device="cpu")

    num_params = 0
    for name, param in model.named_parameters():
        if param.requires_grad and "language_encoder" not in name:
            # print(name, param.numel())
            num_params += param.numel()

    print(f"Trainable parameters outside LVLM: {num_params:,d}")

    args.model = model

    if args.accelerator.is_local_main_process:
        print(model)
    
    def create_stage_dataloader(stage_datasets, is_train=True):
        datasets = [all_train_datasets[dt] for dt in stage_datasets] if is_train else [all_val_datasets[dt] for dt in stage_datasets]
        merged_dataset = MergedDataset(datasets)
        batch_size = args.batch_size if is_train else args.eval_batch_size
        
        dataloader = DataLoader(
            merged_dataset,
            batch_size=batch_size,
            shuffle=is_train,
            collate_fn=collate_fn,
            num_workers=0 if not cfg.eval_use_sglang else 0, 
            # num_workers=4,
            # FIXME: sglang shutdown will kill all subprocesses, make a bug here if num_workers > 0
            #   Potential fix: re-create dataloader after each epoch (after sglang shutdown)
            # UPDATE: fixed by specifying current subproc PIDs before SGLang init for .shutdown()
            #   to not to kill.
            # persistent_workers=True,
        ) 
        return merged_dataset, dataloader
    
    # 计算学习率调度器的总步数
    total_steps = 0
    for i, stage in enumerate(curriculum_stages):
        next_epoch = args.epochs if i == len(curriculum_stages) - 1 else curriculum_stages[i + 1].epoch
        stage_epochs = next_epoch - stage.epoch
        
        # 创建临时数据加载器以获取其长度
        _, temp_dataloader = create_stage_dataloader(stage.datasets, is_train=True)
        steps_per_epoch = len(temp_dataloader) // args.gradient_accumulation_steps

        # Log training steps info
        if accelerator.is_local_main_process:
            print(f"阶段 {i + 1}:")
            print(f"  数据集: {', '.join(stage.datasets)}")
            print(f"  Epochs: {stage_epochs}")
            print(f"  Steps per epoch: {steps_per_epoch}")
        
        total_steps += steps_per_epoch * stage_epochs
    
    # 初始化优化器和调度器
    optimizer = get_optmizer(args.optimizer, model, args)
    
    lr_scheduler = transformers.get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=int(total_steps * args.warmup_ratio), # // cfg.world_size,
        num_training_steps=total_steps, # // cfg.world_size,
    )
    
    
    # 初始化第一个阶段
    current_stage_index = 0
    current_stage = curriculum_stages[current_stage_index]
    
    train_dataset, train_dataloader = create_stage_dataloader(current_stage.datasets, is_train=True)
    val_dataloaders = {}
    
    for dt in all_val_datasets:
        val_dataloader = DataLoader(
            all_val_datasets[dt],
            batch_size=args.eval_batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0 if not cfg.eval_use_sglang else 0,
            # persistent_workers=True,
            # num_workers=4,
        )
        val_dataloaders[dt] = val_dataloader
    
    # Prepare model, optimizer, and dataloaders with accelerator
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    # try offload vision tower if possible
    if "qwen3-vl" in cfg.model_name.lower():
        logger.info(f"Offloading vision encoder for Qwen3-VL")
        unwrapped_model = accelerator.unwrap_model(model)

        for param in unwrapped_model.language_encoder.model.visual.parameters():
            param.requires_grad = False
        unwrapped_model.language_encoder.model.visual.to("cpu")

        torch.cuda.empty_cache()

    # show each module in model's GPU/CPU placement
    # if accelerator.is_local_main_process:
    #     print("Model module device placement:".center(50, "="))
    #     for name, module in model.named_modules():
    #         print(f"{name}: {next(module.parameters()).device}")
    #     print("=" * 50)

    # optimizer, train_dataloader = accelerator.prepare(
    #     optimizer, train_dataloader
    # )
    # model = DDP(model, find_unused_parameters=True)
    
    for dt in val_dataloaders:
        val_dataloaders[dt] = accelerator.prepare(val_dataloaders[dt])

    args.model = model
    args.tokenizer = tokenizer
    args.train_dataloader = train_dataloader
    # args.val_dataloaders = {dt: val_dataloaders[dt] for dt in current_stage.datasets} # the used datasets in current stage
    args.val_dataloaders = {dt: val_dataloaders[dt] for dt in current_stage.datasets if dt in val_dataloaders} # the used datasets in current stage, since some datasets may not have val sets, so we need to check first.

    
    # Print training information
    if accelerator.is_local_main_process:
        print("Training information:".center(50, "="))
        
        # 打印每个数据集的样本数量
        print("Dataset statistics:".center(40, "-"))
        for dtype in all_dataset_types:
            train_count = len(all_train_datasets[dtype])
            val_count = len(all_val_datasets[dtype]) if dtype in all_val_datasets else "no"
            print(f"Dataset '{dtype}': {train_count} training samples, {val_count} validation samples")
        
        # 打印每个课程学习阶段的信息
        print("Curriculum stages:".center(40, "-"))
        for i, stage in enumerate(curriculum_stages):
            # 计算该阶段的epoch范围
            next_epoch = args.epochs if i == len(curriculum_stages) - 1 else curriculum_stages[i + 1].epoch
            stage_epochs = next_epoch - stage.epoch
            
            # 计算该阶段的样本总数
            stage_samples = sum(len(all_train_datasets[dt]) for dt in stage.datasets)
            
            # 创建临时数据加载器以获取其长度
            _, temp_dataloader = create_stage_dataloader(stage.datasets, is_train=True)
            steps_per_epoch = len(temp_dataloader) // args.gradient_accumulation_steps // cfg.world_size 
            # NOTE: here dataloaders are already prepared by accelerator, so actual len is 1/world_size to temp dataloader
            total_stage_steps = steps_per_epoch * stage_epochs
            
            print(f"Stage {i+1} (Epochs {stage.epoch}-{next_epoch-1}):")
            print(f"   Datasets: {', '.join(stage.datasets)}")
            print(f"   Total samples: {stage_samples}")
            print(f"   Steps per epoch: {steps_per_epoch}")
            print(f"   Total steps: {total_stage_steps}")
        
        # 打印总体训练信息
        print("General training settings:".center(40, "-"))
        print(f"Number of epochs: {args.epochs}")
        print(f"Batch size: {args.batch_size}")
        print(f"Gradient accumulation steps: {args.gradient_accumulation_steps}")
        print(f"Total optimization steps: {total_steps // cfg.world_size // args.gradient_accumulation_steps}")
        print(f"Warmup steps: {int(total_steps * args.warmup_ratio // cfg.world_size)}")

    
    # Training loop
    args.curriculum_stages = curriculum_stages
    args.current_stage_index = current_stage_index

    # Initialize training variables
    start_epoch = 0
    # Load checkpoint if specified
    if hasattr(cfg, 'resume_from_checkpoint') and cfg.resume_from_checkpoint:
        try:
            resume_epoch, resume_step, _, _ = load_checkpoint(
                model, optimizer, lr_scheduler, args, cfg.resume_from_checkpoint
            )

            if not cfg.resume_from_checkpoint_epoch:
                resume_epoch = 0 # start over from epoch 0

            # calculate resume_stage_index
            resume_stage_index = 0
            for i, stage in enumerate(curriculum_stages):
                if resume_epoch >= stage.epoch:
                    resume_stage_index = i

            logger.info(f"Resuming from epoch {resume_epoch}, stage {resume_stage_index + 1}")
            
            start_epoch = resume_epoch + 1  # Start from next epoch

            # args.global_step = global_step # TODO
            # current_stage_index = resume_stage_index
            # args.current_stage_index = current_stage_index
            
            # Update to correct curriculum stage NOTE: will be done at the start of each epoch
            # if current_stage_index < len(curriculum_stages):
            #     current_stage = curriculum_stages[current_stage_index]
                
            #     # Recreate dataloader for current stage if needed
            #     if set(current_stage.datasets) != set(curriculum_stages[0].datasets):
            #         train_dataset, train_dataloader = create_stage_dataloader(current_stage.datasets, is_train=True)
            #         train_dataloader = accelerator.prepare(train_dataloader)
            #         args.train_dataloader = train_dataloader
            #         args.val_dataloaders = {dt: val_dataloaders[dt] for dt in current_stage.datasets}
                
            #     if accelerator.is_local_main_process:
            #         print(f"Resumed at curriculum stage {current_stage_index + 1}")
            #         print(f"Current stage datasets: {', '.join(current_stage.datasets)}")
            
            # For old checkpoints without scheduler state, manually advance scheduler
            _global_step = 0
            # scheduler_path = os.path.join(cfg.resume_from_checkpoint, "scheduler.pt")
            # if not os.path.exists(scheduler_path):

            # Calculate total steps up to current point
            steps_completed = 0
            for i, stage in enumerate(curriculum_stages[:resume_stage_index + 1]):
                if i < resume_stage_index:
                    # Completed stages
                    next_epoch = curriculum_stages[i + 1].epoch if i + 1 < len(curriculum_stages) else args.epochs
                    stage_epochs = next_epoch - stage.epoch
                else:
                    # Current stage up to resume epoch
                    stage_epochs = resume_epoch - stage.epoch + 1
                
                _, temp_dataloader = create_stage_dataloader(stage.datasets, is_train=True) # this epoch's train dataloader
                
                steps_per_epoch = len(temp_dataloader) // args.gradient_accumulation_steps // cfg.world_size 
                # NOTE: here dataloaders are already prepared by accelerator, so actual len is 1/world_size to temp dataloader
                steps_completed += steps_per_epoch * stage_epochs
                _global_step += len(temp_dataloader) * stage_epochs
            
            manually_advance_scheduler(lr_scheduler, steps_completed)
            args.global_step = _global_step
            
            if accelerator.is_local_main_process:
                print(f"Successfully resumed training from epoch {start_epoch}")
                
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            raise
            logger.info("Starting training from scratch")
            start_epoch = 0
            args.global_step = 0
            current_stage_index = 0
            args.current_stage_index = current_stage_index

    # delete all not needed dataset?

    args.global_step = 0 if not hasattr(args, 'global_step') else args.global_step
    # for epoch in range(args.epochs):
    for epoch in range(start_epoch, args.epochs):
        # 检查是否需要切换到新的课程学习阶段
        while current_stage_index + 1 < len(curriculum_stages) and epoch >= curriculum_stages[current_stage_index + 1].epoch:
            current_stage_index += 1
            current_stage = curriculum_stages[current_stage_index]
            
            if accelerator.is_local_main_process:
                print(f"切换到课程学习阶段 {current_stage_index + 1}，当前epoch {epoch}")
                print(f"使用数据集: {', '.join(current_stage.datasets)}")
            
            # 创建新的训练数据加载器
            train_dataset, train_dataloader = create_stage_dataloader(current_stage.datasets, is_train=True)
            
            # 使用accelerator准备
            train_dataloader = accelerator.prepare(train_dataloader)
            
            # 更新args
            args.train_dataloader = train_dataloader
            args.val_dataloaders = {dt: val_dataloaders[dt] for dt in current_stage.datasets if dt not in DATASETS_NO_EVAL}
        

        # Train for one epoch
        # comment to test the eval only
        if not args.validation_only:
            train_loss = train_epoch(
                model, train_dataloader, optimizer, lr_scheduler, 
                accelerator, args, epoch
            )

        # Save checkpoint
        save_checkpoint(model, optimizer, lr_scheduler, args, epoch)
        
        # Evaluate
        metrics = evaluate(
            model, accelerator.unwrap_model(model), 
            args.val_dataloaders, accelerator, args, epoch
        )

        # 更新全局步数
        args.global_step += len(train_dataloader)
        
        
        # Log epoch results
        if accelerator.is_local_main_process:
            print(f"Epoch {epoch} completed. Train loss: {train_loss:.4f}")
            print(f"Validation metrics: {metrics}")
    
    # Close wandb
    if accelerator.is_local_main_process:
        wandb.finish()

@hydra.main(version_base=None, config_path="configs", config_name="apeiria_mllm")
def entry(config: Config):
    """
    Entry point for the script.
    """
    OmegaConf.set_struct(config, False)

    num_gpus = torch.cuda.device_count()

    # pre-allocate sglang ports, duplicate N times, N = num_inference_gpus
    config.sglang_ports = [find_free_port() for _ in range(num_gpus)]
    print(f"Pre-allocated sglang ports: {config.sglang_ports}")

    hydra_output_dir = HydraConfig.get().runtime.output_dir
    print(f"Hydra output directory detected: {hydra_output_dir}")

    # 如果用户没有指定 output_dir，就使用 Hydra 的目录
    if config.output_dir == "": # 或者检查是否为 None，取决于你的 config 默认值
        print(f"config.output_dir is empty, defaulting to Hydra output directory.")
        config.output_dir = hydra_output_dir
    else:
        print(f"Using specified output directory: {config.output_dir}")

    # Run the main function
    mp.spawn(main, args=(config,), nprocs=num_gpus, join=True)

if __name__ == "__main__":
    entry()
