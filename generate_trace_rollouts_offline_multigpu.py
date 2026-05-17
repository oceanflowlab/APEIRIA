import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import transformers
from transformers import AutoModel, AutoTokenizer
from peft import PeftModel
import logging
import argparse
import numpy as np
from datetime import datetime
import random
import json
from tqdm.auto import tqdm
from typing import List, Dict, Union, Optional, Tuple, Callable
from copy import deepcopy
from icecream import ic
from collections import defaultdict
import hydra
from omegaconf import DictConfig, OmegaConf
from types import SimpleNamespace
import torch.multiprocessing as mp
# import multiprocessing as mp
from multiprocessing import Process, Queue, Event
import signal
import sys
import time
import tempfile
import sys
import time
import queue
import traceback
import socket
from contextlib import closing
import glob
from scipy.optimize import linear_sum_assignment
import re

from apeiria_mllm_config_schema import Config
from apeiria_mllm import MultimodalLanguageModelDecoderOnly
from apeiria_lm_utils import Synthetic3DDataset, Synthetic3DObjectInfoDataset, MergedDataset, Synthetic3DRelationalDataset
from apeiria_lm_prog_to_thinking import Real3DDataset, Real3DObjectInfoDataset, Real3DFilterDataset, parse_response
from qwen_helpers import apply_qwen_template

# Set multiprocessing start method
mp.set_start_method('spawn', force=True)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global shutdown event for graceful exit
shutdown_event = mp.Event()

# Dataset class mapping
DATASET_CLSMAP = {
    "synthetic3d": Synthetic3DDataset,
    "synthetic3d_object_info": Synthetic3DObjectInfoDataset,
    "synthetic3d_relational": Synthetic3DRelationalDataset,
    "sr3d": Real3DDataset,
    "nr3d": Real3DDataset, 
    "nr3d-gemini2.5pro": Real3DDataset, 
    "sr3d_object_info": Real3DObjectInfoDataset,
    "sr3d_filter": Real3DFilterDataset,
    "scanrefer_nocot": Real3DDataset,
    "nr3d_nocot": Real3DDataset,
    "multi3drefer_nocot": Real3DDataset,
    "scanrefer": Real3DDataset,
    "nr3d": Real3DDataset,
    "multi3drefer": Real3DDataset,
}

def build_hash_from_object_id_list(object_id_list: List[Union[int, str]]) -> str:
    """Build a unique hash string from a list of object IDs"""
    sorted_ids = sorted([int(oid) for oid in object_id_list])
    hash_str = "|".join([str(oid) for oid in sorted_ids])
    return hash_str

def robust_save_json(data_to_save: dict, filepath: str, max_retries: int = 5, delay_seconds: int = 5):
    """
    Robustly saves a dictionary to a JSON file with retries and atomic move,
    using tempfile for safer temporary file creation.
    This is useful for unstable file systems like NFS.
    """
    # Ensure the destination directory exists
    dest_dir = os.path.dirname(filepath)
    os.makedirs(dest_dir, exist_ok=True)

    for attempt in range(max_retries):
        temp_file = None
        try:
            # 1. Create a temporary file in the *same directory* as the final file.
            #    This is crucial for an atomic os.rename() on most filesystems (like NFS).
            #    'delete=False' is necessary because we will rename it ourselves.
            with tempfile.NamedTemporaryFile(mode='w', delete=False, dir=dest_dir, suffix=".tmp") as f:
                temp_filepath = f.name
                json.dump(data_to_save, f, indent=2)
                # 2. Ensure data is written to disk
                f.flush()
                os.fsync(f.fileno())

            # 3. Atomically rename the temp file to the final destination
            os.rename(temp_filepath, filepath)
            
            # Success
            return True
        except (IOError, OSError) as e:
            logger.warning(f"Attempt {attempt + 1}/{max_retries} to save {filepath} failed: {e}. Retrying in {delay_seconds}s...")
            # Clean up the temporary file if it still exists
            if temp_filepath and os.path.exists(temp_filepath):
                try:
                    os.remove(temp_filepath)
                except OSError as remove_err:
                    logger.error(f"Failed to remove temporary file {temp_filepath}: {remove_err}")
            time.sleep(delay_seconds)
    
    logger.error(f"Failed to save file {filepath} after {max_retries} attempts.")
    return False


def find_free_port():
    """动态寻找一个可用的空闲端口"""
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(('', 0))  # 绑定到一个随机空闲端口
        return s.getsockname()[1]  # 返回分配的端口号

class InferenceWorker(Process):
    """Worker process for running inference on a single GPU"""
    
    def __init__(self, 
                 worker_id: int,
                 gpu_id: int,
                 args: SimpleNamespace,
                 input_queue: Queue,
                 output_queue: Queue,
                 shutdown_event: Event,
                 feature_dim: int,
                 modality_dims: Dict,
                 modality_order: List[str],
                 external_plans: Dict = None): # Added external_plans
        super().__init__()
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)  # Set visible GPU for this worker
        self.worker_id = worker_id
        self.gpu_id = gpu_id
        self.args = args
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.shutdown_event = shutdown_event
        self.feature_dim = feature_dim
        self.modality_dims = modality_dims
        self.modality_order = modality_order
        self.external_plans = external_plans # Store external plans
        self.scene_objects_cache = {} # Cache for ground truth scene objects string
        self.model = None
        self.tokenizer = None

    @property
    def is_rank0_worker(self):
        """Check if this is the rank 0 worker"""
        return self.worker_id == 0
        
    def setup_model(self):
        """Initialize model on the assigned GPU"""
        # Set CUDA device
        # torch.cuda.set_device(self.gpu_id)
        # device = torch.device(f"cuda:{self.gpu_id}")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # only one GPU visible

        # ==== SGLang参数适配 ====
        use_sglang = getattr(self.args, "use_sglang_for_generation", False)
        sglang_kwargs = {}
        if use_sglang:
            sglang_kwargs = {
                "use_sglang": True,
                "sglang_model_path": self.args.model_name,
                "sglang_lora_paths": [self.args.checkpoint_path],
                "sglang_port": self.args.sglang_port[self.worker_id],
            }
        else:
            sglang_kwargs = {"use_sglang": False}
        
        logger.info(f"Worker {self.worker_id}: Loading model on GPU {self.gpu_id}")
        
        # Load model and tokenizer
        language_model, tokenizer = load_model_and_tokenizer(self.args, device)

        # move to device
        if not use_sglang:
            language_model = language_model.to(device)
        # else:
        #     # only move embeddings to device, the rest will be handled by SGLang
        #     language_model.get_input_embeddings().to(device)

        torch.cuda.empty_cache()

        # Initialize multimodal model
        model = MultimodalLanguageModelDecoderOnly(
            language_model=language_model,
            tokenizer=tokenizer,
            object_feature_dim=self.feature_dim,
            max_objects=self.args.max_objects,
            no_object_in_language_model=self.args.no_object_in_language_model,
            object_embedding_type=self.args.object_embedding_type,
            discrete_location_bins=self.args.discrete_location_bins,
            discrete_location_decay_kernel=self.args.discrete_location_decay_kernel,
            discrete_location_bin_range=[0.0, self.args.room_size],
            discrete_location_decay_kernel_size=self.args.discrete_location_decay_kernel_size,
            separate_location_embedding=self.args.separate_location_embedding,
            dtype=torch.bfloat16,
            modality_dims=self.modality_dims,
            modality_order=self.modality_order,
            delete_model_from_cpu=True, # save CPU memory
            sglang_log_level=self.args.sglang_log_level,
            **sglang_kwargs,
        )
        
        if not use_sglang:
            # Load non-LM checkpoint if provided
            if self.args.checkpoint_path:
                logger.info(f"Worker {self.worker_id}: Loading non-LM part of checkpoint from {self.args.checkpoint_path}")
                model.load_non_lm_parameters(self.args.checkpoint_path)
            model = model.to(device)
        else:
            # move embedding layers to GPU
            model.object_proj = model.object_proj.to(device)
            model.language_encoder.get_input_embeddings().to(device)

        # print a summary of the model, and each parameter's device
        # for name, param in model.named_parameters():
        #     logger.info(f"Worker {self.worker_id}: Parameter {name} on device {param.device}")
        
        model.eval()
        
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        
        logger.info(f"Worker {self.worker_id}: Model setup complete")
        
    def process_batch(self, batch_data):
        """Process a single batch and return results"""
        try:
            batch = batch_data['batch']
            num_passes = batch_data['num_passes']
            batch_idx = batch_data['batch_idx']

            if self.args.enforce_single_prediction:
                single_prediction_prompt = getattr(self.args, "single_prediction_prompt", "")
                if single_prediction_prompt:
                    batch["instructions"] = [inst + single_prediction_prompt for inst in batch["instructions"]]
            
            # Prepare model inputs
            model_inputs = prepare_model_inputs(batch, self.device, self.tokenizer)
            
            forced_prefixes = [""] * len(batch["scanrefer_id"])
            if self.external_plans:
                new_instructions = []
                for i in range(len(batch["scanrefer_id"])):
                    scene_id = batch["scene_ids"][i]
                    # Handle object_ids conversion to string key
                    # In ScanRefer, object_ids is usually [int]
                    obj_list = batch["object_ids"][i]
                    # if not obj_list:
                    # NOTE: multi3drefer can have 0, 1, or more object ids
                    if obj_list is None:
                        raise ValueError(f"No object_id found for sample {batch['scanrefer_id'][i]}")
                    elif isinstance(obj_list, list):
                        if len(obj_list) == 0:
                            object_id = "<none>"
                        else:
                            object_id = str(obj_list[0])
                    else:
                        object_id = str(obj_list)

                    description = batch["raw_description"][i]
                    
                    # Lookup key matches JSON loader in main
                    # key = (scene_id, object_id, description)
                    key = (scene_id, build_hash_from_object_id_list(obj_list), description)
                    
                    if key not in self.external_plans:
                        # Fallback try? No, strict mode requested.
                        # raise ValueError(f"CRITICAL: External plan not found for scene={scene_id}, obj={object_id}. Description: '{description[:50]}...'")
                        logger.warning(f"CRITICAL: External plan not found for scene={scene_id}, obj={object_id}. Description: '{description}', will use no plan and no object info injection.")
                        # set no external plan and no object info injection
                        prefix = ""
                    else:
                        plan = self.external_plans[key]
                        
                        # Pre-fill response with thinking header and plan
                        prefix = f"[APEIRIA THINKS]\n{plan}\n"
                        
                        # Inject GT object list if configured
                        gt_scene_path = getattr(self.args, "gt_scene_data_path", None)
                        if gt_scene_path:
                            if scene_id not in self.scene_objects_cache:
                                try:
                                    json_path = os.path.join(gt_scene_path, f"{scene_id}.json")
                                    if os.path.exists(json_path):
                                        with open(json_path, 'r') as f:
                                            data = json.load(f)
                                            objs = data.get("objects", [])
                                            # Sort by ID to ensure deterministic order
                                            objs.sort(key=lambda x: int(x["id"]) if isinstance(x["id"], int) else int(x["id"]))
                                            
                                            obj_items = []
                                            for o in objs:
                                                # Prioritize name, then nyu40_name, then default to "object"
                                                # if have predicted_label, first use it: it is detector sourced prediction
                                                name = o.get("predicted_label") or o.get("name") or o.get("nyu40_name") or "object"
                                                obj_items.append(f"{o['id']} ({name})")
                                            
                                            count = len(objs)
                                            list_str = ", ".join(obj_items)
                                            
                                            # Format: "First, I'll examine all N objects... I see N object(s)...: list"
                                            cache_entry = f"First, I'll examine all {count} objects in the scene.\nI see {count} object(s) in the scene: {list_str}\n"
                                            self.scene_objects_cache[scene_id] = cache_entry
                                    else:
                                        logger.warning(f"GT scene file not found for injection: {json_path}")
                                        self.scene_objects_cache[scene_id] = ""
                                except Exception as e:
                                    logger.error(f"Error loading GT scene data for {scene_id}: {e}")
                                    self.scene_objects_cache[scene_id] = ""
                            
                            prefix += self.scene_objects_cache.get(scene_id, "")

                    forced_prefixes[i] = prefix
                    
                    # Append to instruction prompt. This assumes instruction ends with assistant start token
                    # (which prepare_model_inputs/apply_qwen_template does)
                    # The model will then continue generating from this plan.
                    new_instructions.append(model_inputs["instructions"][i] + prefix)
                
                model_inputs["instructions"] = new_instructions

            if self.args.use_nr3d_plan_from_program: 
                # use NR3D plan from its program written by LLM instead of generating it
                ic(model_inputs["instructions"][0])
                ic(batch["prompt_with_plan"][0])
                model_inputs["instructions"] = batch["prompt_with_plan"]
            
            batch_size = len(batch["scanrefer_id"])
            results = []
            
            # Initialize results for each sample
            for i in range(batch_size):
                results.append({
                    "scanrefer_id": batch["scanrefer_id"][i],
                    "scene_id": batch["scene_ids"][i],
                    "object_ids": batch["object_ids"][i],
                    "expected_response": batch["responses"][i],
                    "instruction": batch["instructions"][i],
                    "raw_description": batch["raw_description"][i],
                    "passes": []
                })
            
            # Run multiple passes
            # for pass_idx in range(num_passes):
            pbar = tqdm(range(num_passes), desc=f"Worker {self.worker_id} - Processing batch {batch_idx}", leave=False, disable=not self.is_rank0_worker)

            for pass_idx in pbar:
                # Rejection sampling logic: retry if the number of grounded objects is incorrect
                # Default to 5 retries if enforcing single prediction
                max_retries = getattr(self.args, "max_retries", 5) if self.args.enforce_single_prediction else 0
                active_indices = list(range(batch_size))
                predictions = [""] * batch_size

                for attempt in range(max_retries + 1):
                    # TODO: in future, we can add into prompt that only one object should be selected if retrying, and enforce_single_prediction is on.
                    if not active_indices:
                        break
                    
                    # Prepare inputs for the current subset of the batch
                    current_instructions = [model_inputs["instructions"][i] for i in active_indices]
                    current_object_embeds = [model_inputs["object_set_embeds"][i] for i in active_indices]

                    init_temperature = self.args.temperature if self.args.do_sample else 0.0
                    current_temperature = init_temperature + attempt * 0.2  # Increase temperature on each retry
                    with torch.no_grad():
                        # Generate predictions
                        current_preds = self.model.generate(
                            instructions=current_instructions,
                            object_set_embeds=current_object_embeds,
                            max_length=self.args.max_new_tokens,
                            num_beams=1,
                            do_sample=self.args.do_sample if attempt == 0 else True,  # Enable sampling on retries, to get diversity
                            top_k=self.args.top_k,
                            top_p=self.args.top_p,
                            temperature=current_temperature,
                        )
                    
                    next_active_indices = []
                    for local_idx, pred in enumerate(current_preds):
                        original_idx = active_indices[local_idx]
                        
                        is_valid = True
                        if self.args.enforce_single_prediction:
                            # Reconstruct full response to check object count (mimics post-processing logic)
                            temp_response = pred
                            if self.external_plans and forced_prefixes[original_idx]:
                                if not temp_response.strip().startswith("[APEIRIA THINKS]"):
                                    temp_response = forced_prefixes[original_idx] + temp_response
                            
                            # Count unique grounded objects
                            parsed_objs = parse_response(temp_response)
                            unique_ids = set(o["id"] for o in parsed_objs)
                            if len(unique_ids) != 1:
                                is_valid = False
                        #     else:
                        #         # For non-enforced cases, just check if any object is present
                        #         if len(parsed_objs) == 0 or (len(parsed_objs) == 1 and parsed_objs[0]["id"] == ""):
                        #             is_valid = False
                        
                        if is_valid or attempt == max_retries:
                            predictions[original_idx] = pred
                        else:
                            next_active_indices.append(original_idx)
                    
                    active_indices = next_active_indices
                    # report if some are still active
                    if active_indices:
                        logger.info(f"Worker {self.worker_id} - Batch {batch_idx} - Pass {pass_idx} - Retry attempt {attempt + 1}: {len(active_indices)} samples still invalid, retrying...")
                
                # Process results
                for i in range(batch_size):
                    response = predictions[i]
                    
                    # If using external plan, reconstruct full response if generator returned only new tokens
                    # or partial output. We force the prefix into the response for evaluation/extraction.
                    if self.external_plans and forced_prefixes[i]:
                        # Check if response already contains the plan (some generators return full text)
                        # We use a loose check (startswith) or simple inclusion
                        if not response.strip().startswith("[APEIRIA THINKS]"):
                            response = forced_prefixes[i] + response

                    # Extract thinking trace
                    thinking_trace, trace_parts = extract_thinking_trace_from_response(response)
                    
                    # Evaluate correctness
                    # Pass object_lookup from batch for precise IoU
                    pred_lookup = batch.get("pred_object_id_to_box", [None]*batch_size)[i]
                    gt_lookup = batch.get("gt_object_id_to_box", [None]*batch_size)[i]

                    is_correct, eval_details = evaluate_response(
                        response, batch["responses"][i], None, 
                        enforce_single_prediction=self.args.enforce_single_prediction,
                        pred_object_lookup=pred_lookup,
                        gt_object_lookup=gt_lookup
                    )
                    
                    # Create pass result
                    pass_result = {
                        "pass_idx": pass_idx,
                        "response": response,
                        "thinking_trace": thinking_trace,
                        "thinking_trace_parts": trace_parts,
                        "is_correct": is_correct,
                        "evaluation_details": eval_details,
                    }
                    
                    results[i]["passes"].append(pass_result)

            # show some examples from this batch, first 2 and one wrong
            to_show_ids = []
            try:
                for i in range(batch_size):
                    for pass_result in results[i]['passes']:
                        if not pass_result['is_correct']:
                            to_show_ids.append(i)
                            raise StopIteration
            except StopIteration:
                pass
            
            to_show_ids.extend(list(range(min(2, batch_size))))
            to_show_ids = sorted(set(to_show_ids))

            for i in to_show_ids:
                logger.info(f"Worker {self.worker_id} - Batch {batch_idx} - Sample {i}:")
                logger.info(f"  ScanRefer ID: {results[i]['scanrefer_id']}")
                logger.info(f"  Instruction: {results[i]['instruction']}")
                for pass_result in results[i]['passes']:
                    logger.info(f"    Pass {pass_result['pass_idx']}:")
                    logger.info(f"      Response: {pass_result['response']}")
                    logger.info(f"      Is Correct: {pass_result['is_correct']}")
                    logger.info(f"      Evaluation Details: {pass_result['evaluation_details']}")
                    logger.info(f"      Thinking Trace: {pass_result['thinking_trace']}")
            
            return {
                'batch_idx': batch_idx,
                'results': results,
                'worker_id': self.worker_id,
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Worker {self.worker_id}: Error processing batch: {str(e)}")
            logger.error(traceback.format_exc())
            return {
                'batch_idx': batch_idx,
                'results': None,
                'worker_id': self.worker_id,
                'success': False,
                'error': str(e)
            }
    
    def run(self):
        """Main worker loop"""
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)

        try:
            # Setup model
            self.setup_model()
            
            # Signal ready
            self.output_queue.put({
                'type': 'ready',
                'worker_id': self.worker_id
            })
            
            # Process batches
            while not self.shutdown_event.is_set():
                try:
                    # Get batch from queue with timeout
                    batch_data = self.input_queue.get(timeout=1.0)
                    
                    if batch_data is None:  # Poison pill
                        break
                    
                    # Process batch
                    result = self.process_batch(batch_data)
                    
                    # Send result back
                    self.output_queue.put({
                        'type': 'result',
                        'data': result
                    })
                    
                except queue.Empty:
                    continue
                except Exception as e:
                    logger.error(f"Worker {self.worker_id}: Error in main loop: {str(e)}")
                    logger.error(traceback.format_exc())
                    
        except Exception as e:
            logger.error(f"Worker {self.worker_id}: Fatal error: {str(e)}")
            logger.error(traceback.format_exc())
        finally:
            logger.info(f"Worker {self.worker_id}: Shutting down")
            self.output_queue.put({
                'type': 'shutdown',
                'worker_id': self.worker_id
            })

def signal_handler(signum, frame):
    """Handle interrupt signals gracefully"""
    logger.info("\nReceived interrupt signal. Shutting down gracefully...")
    shutdown_event.set()

def load_partial_results(partial_results_dir: str, dataset: torch.utils.data.Dataset) -> Tuple[Dict, set]:
    """Load and validate partial results from a previous run."""
    if not os.path.isdir(partial_results_dir):
        logger.warning(f"Partial results directory not found: {partial_results_dir}")
        return {}, set()

    logger.info("Loading scanrefer_id to instruction map for validation...")
    scanrefer_id_to_instruction = {item['scanrefer_id']: item['raw_description'] for item in dataset}
    logger.info("Map created.")

    loaded_data = {
        'correct_traces': defaultdict(list),
        'incorrect_traces': defaultdict(list),
        'all_results': [],
        'sample_correct_counts': defaultdict(int),
        'sample_total_counts': defaultdict(int),
        'sample_iou_scores': defaultdict(list),  # Track IoU scores for resume
        'sample_f1_scores': defaultdict(lambda: defaultdict(list)), # Track F1 scores
    }
    completed_scanrefer_ids = set()

    partial_files = sorted(glob.glob(os.path.join(partial_results_dir, "batch_*.json")))
    logger.info(f"Found {len(partial_files)} partial result files.")

    for f_path in tqdm(partial_files, desc="Loading partial results"):
        try:
            with open(f_path, 'r') as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Could not read or parse {f_path}, skipping. Error: {e}")
            continue
        
        batch_idx = data['batch_idx']
        
        # Validate instructions
        batch_scanrefer_ids = []
        for sample_result in data.get('results', []):
            scanrefer_id = sample_result.get('scanrefer_id')
            if not scanrefer_id:
                logger.warning(f"Skipping sample without scanrefer_id in {f_path}")
                continue

            if scanrefer_id not in scanrefer_id_to_instruction:
                logger.warning(f"scanrefer_id {scanrefer_id} from partial results not found in current dataset. Skipping sample from batch {batch_idx}.")
                continue
            
            if 'raw_description' in sample_result and sample_result['raw_description'] != scanrefer_id_to_instruction[scanrefer_id]:
                logger.warning(f"Instruction mismatch for scanrefer_id {scanrefer_id} in batch {batch_idx}. Skipping sample.")
                continue

            # If validation passes, process this sample
            loaded_data['all_results'].append(sample_result)
            
            for pass_result in sample_result['passes']:
                loaded_data['sample_total_counts'][scanrefer_id] += 1
                if pass_result['is_correct']:
                    loaded_data['sample_correct_counts'][scanrefer_id] += 1

                # Store IoU score
                mean_iou = pass_result['evaluation_details'].get('mean_iou', 0.0)
                loaded_data['sample_iou_scores'][scanrefer_id].append(mean_iou)

                # Store F1 scores
                for key, value in pass_result['evaluation_details'].items():
                    if key.startswith('f1@'):
                        loaded_data['sample_f1_scores'][scanrefer_id][key].append(value)

                trace_entry = {
                    "thinking_trace": pass_result['thinking_trace'],
                    "thinking_trace_parts": pass_result['thinking_trace_parts'],
                    "metadata": {
                        "scene_id": sample_result['scene_id'],
                        "object_ids": sample_result['object_ids'],
                        "pass_idx": pass_result['pass_idx'],
                        "is_correct": pass_result['is_correct'],
                        "evaluation_details": pass_result['evaluation_details'],
                    }
                }
                if pass_result['is_correct']:
                    loaded_data['correct_traces'][scanrefer_id].append(trace_entry)
                else:
                    loaded_data['incorrect_traces'][scanrefer_id].append(trace_entry)
            
            batch_scanrefer_ids.append(scanrefer_id)
        
        completed_scanrefer_ids.update(batch_scanrefer_ids)

    logger.info(f"Loaded and validated {len(completed_scanrefer_ids)} samples from partial results.")
    return loaded_data, completed_scanrefer_ids


def run_multiprocessing_inference(
    dataloader,
    args: SimpleNamespace,
    num_passes: int,
    save_dir: str,
    feature_dim: int,
    modality_dims: Dict,
    modality_order: List[str],
    preloaded_data: Dict,
    num_gpus: Optional[int] = None,
    available_gpus: Optional[List[int]] = None,
    external_plans: Dict = None,  # Added argument
):
    """Run inference using multiple GPU workers"""
    
    # Determine number of GPUs
    if num_gpus is None:
        num_gpus = torch.cuda.device_count()
    num_gpus = min(num_gpus, torch.cuda.device_count())
    
    if num_gpus == 0:
        raise RuntimeError("No GPUs available for inference")
    
    logger.info(f"Using {num_gpus} GPUs for inference")
    
    # Create queues
    input_queue = mp.Queue()  # Limit queue size
    output_queue = mp.Queue()
    
    # Create and start workers
    workers = []
    for i in range(num_gpus):
        worker = InferenceWorker(
            worker_id=i,
            gpu_id=int(available_gpus[i]) if available_gpus else i,
            args=args,
            input_queue=input_queue,
            output_queue=output_queue,
            shutdown_event=shutdown_event,
            feature_dim=feature_dim,
            modality_dims=modality_dims,
            modality_order=modality_order,
            external_plans=external_plans,  # Pass external plans to worker
        )
        worker.start()
        workers.append(worker)
    
    # Wait for workers to be ready
    ready_count = 0
    while ready_count < num_gpus:
        msg = output_queue.get()
        if msg['type'] == 'ready':
            ready_count += 1
            logger.info(f"Worker {msg['worker_id']} ready ({ready_count}/{num_gpus})")
    
    # Initialize result containers from preloaded_data
    correct_traces = preloaded_data.get('correct_traces', defaultdict(list))
    incorrect_traces = preloaded_data.get('incorrect_traces', defaultdict(list))
    all_results = preloaded_data.get('all_results', [])
    
    # Statistics
    total_samples = 0
    total_correct = 0
    
    # Pass@k tracking - 为每个sample追踪正确/错误次数
    sample_correct_counts = preloaded_data.get('sample_correct_counts', defaultdict(int))
    sample_total_counts = preloaded_data.get('sample_total_counts', defaultdict(int))

    # IoU Pass@k tracking - track per-sample IoU scores
    sample_iou_scores = preloaded_data.get('sample_iou_scores', defaultdict(list))  # scanrefer_id -> [iou_per_pass]
    
    # F1 scores tracking
    sample_f1_scores = preloaded_data.get('sample_f1_scores', defaultdict(lambda: defaultdict(list)))

    # Update stats from preloaded data
    if preloaded_data:
        total_samples = len(sample_total_counts)
        total_correct = sum(sample_correct_counts.values())
        logger.info(f"Resumed state: {total_samples} samples, {total_correct} correct inferences.")

    # Create progress bar
    total_batches = len(dataloader)
    pbar = tqdm(total=total_batches, desc="Processing batches")
    
    # Submit all batches to queue
    batch_futures = {}
    submitted_batches = 0
    completed_batches = 0
    
    # 用于保存部分结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create a dedicated directory for this run
    run_dir = os.path.join(save_dir, timestamp)
    partial_results_dir = os.path.join(run_dir, "partial_results")
    
    # Always create run_dir to save summary, regardless of no_save
    os.makedirs(run_dir, exist_ok=True)

    if not getattr(args, 'no_save', False):
        os.makedirs(partial_results_dir, exist_ok=True)
    
    try:
        # Start submitting batches
        for batch_idx, batch in enumerate(dataloader):
            if shutdown_event.is_set():
                break
                
            # Submit batch to queue
            batch_data = {
                'batch': batch,
                'num_passes': num_passes,
                'batch_idx': batch_idx
            }
            
            input_queue.put(batch_data)
            batch_futures[batch_idx] = None
            submitted_batches += 1

        logger.info(f"Submitted {submitted_batches} batches for processing")
        
        # Collect results
        while completed_batches < submitted_batches and not shutdown_event.is_set():
            try:
                msg = output_queue.get(timeout=10.0)
                
                if msg['type'] == 'result':
                    logger.warning(f"Received result from worker {msg['data']['worker_id']} for batch {msg['data']['batch_idx']}, success: {msg['data']['success']}")


                    result = msg['data']
                    
                    if result['success']:
                        batch_results = []
                        
                        # Process results
                        for sample_result in result['results']:
                            scanrefer_id = sample_result['scanrefer_id']
                            
                            for pass_result in sample_result['passes']:
                                # Update pass@k tracking
                                sample_total_counts[scanrefer_id] += 1
                                if pass_result['is_correct']:
                                    sample_correct_counts[scanrefer_id] += 1

                                # Store IoU score for this pass
                                mean_iou = pass_result['evaluation_details'].get('mean_iou', 0.0)
                                sample_iou_scores[scanrefer_id].append(mean_iou)

                                # Store F1 scores
                                for key, value in pass_result['evaluation_details'].items():
                                    # FIXME: if no, we simply skip in calculation, which maybe the wrong eval.
                                    if key.startswith('f1@'):
                                        sample_f1_scores[scanrefer_id][key].append(value)

                                # Store trace
                                trace_entry = {
                                    "thinking_trace": pass_result['thinking_trace'],
                                    "thinking_trace_parts": pass_result['thinking_trace_parts'],
                                    "metadata": {
                                        "scene_id": sample_result['scene_id'],
                                        "object_ids": sample_result['object_ids'],
                                        "pass_idx": pass_result['pass_idx'],
                                        "is_correct": pass_result['is_correct'],
                                        "evaluation_details": pass_result['evaluation_details'],
                                    }
                                }
                                
                                if pass_result['is_correct']:
                                    correct_traces[scanrefer_id].append(trace_entry)
                                    total_correct += 1
                                else:
                                    incorrect_traces[scanrefer_id].append(trace_entry)
                            
                            all_results.append(sample_result)
                            batch_results.append(sample_result)
                            total_samples += 1 # This should be sample_result['passes'] length
                        
                        # 保存当前批次的部分结果
                        if not getattr(args, 'no_save', False):
                            batch_file = os.path.join(partial_results_dir, f"batch_{result['batch_idx']:04d}.json")
                            data_to_save = {
                                'batch_idx': result['batch_idx'],
                                'results': batch_results,
                                'timestamp': datetime.now().isoformat()
                            }

                            # with open(batch_file, 'w') as f:
                            #     json.dump({
                            #         'batch_idx': result['batch_idx'],
                            #         'results': batch_results,
                            #         'timestamp': datetime.now().isoformat()
                            #     }, f, indent=2)
                            #     # 强制将缓冲区刷新到磁盘
                            #     f.flush()
                            #     os.fsync(f.fileno())
                            # logger.info(f"Saved partial results for batch {result['batch_idx']} to {batch_file}")

                            save_successful = robust_save_json(data_to_save, batch_file)
                            if save_successful:
                                logger.info(f"Saved partial results for batch {result['batch_idx']} to {batch_file}")
                            else:
                                logger.error(f"CRITICAL: Failed to save partial results for batch {result['batch_idx']}. Data might be lost.")
                                # Optionally, you could trigger a graceful shutdown here if saving is critical
                                # shutdown_event.set()
                    
                    
                    completed_batches += 1
                    pbar.update(1)

                    # calculate a ETA
                    eta_dict = {
                        'completed': completed_batches,
                        'total': submitted_batches,
                        'remaining': (submitted_batches - completed_batches) * (pbar.format_dict['elapsed'] / completed_batches) if completed_batches > 0 else 0
                    }
                    # pbar.set_postfix(eta_dict)

                    logger.info(f"ETA: {eta_dict['remaining'] / 3600:.2f} hours, completed {completed_batches}/{submitted_batches} batches")
                    
                    # 计算当前的pass@k指标
                    if len(sample_total_counts) > 0:
                        pass_at_k_results = {}
                        for k in range(1, min(9, num_passes + 1)):  # k from 1 to 8
                            pass_k_scores = []
                            for scanrefer_id in sample_total_counts:
                                n = sample_total_counts[scanrefer_id]
                                c = sample_correct_counts[scanrefer_id]
                                if n >= k:  # 只有当样本有足够的passes时才计算
                                    pass_k = pass_at_k(n, c, k)
                                    pass_k_scores.append(pass_k)
                            
                            if pass_k_scores:
                                pass_at_k_results[f"pass@{k}"] = np.mean(pass_k_scores)
                        
                        # Calculate current mean F1 scores
                        current_f1_stats = {}
                        all_f1_keys = set()
                        for sid in sample_f1_scores:
                            all_f1_keys.update(sample_f1_scores[sid].keys())
                        
                        for key in all_f1_keys:
                            all_scores = []
                            for sid in sample_f1_scores:
                                if key in sample_f1_scores[sid]:
                                    all_scores.extend(sample_f1_scores[sid][key])
                            if all_scores:
                                current_f1_stats[key] = np.mean(all_scores)

                        # Calculate current IoU@threshold (Pass@1)
                        current_iou_stats = {}
                        for threshold in [0.25, 0.5]:
                            pass1_scores = []
                            for sid in sample_iou_scores:
                                if sample_iou_scores[sid]:
                                    pass1_scores.append(1.0 if sample_iou_scores[sid][0] >= threshold else 0.0)
                            if pass1_scores:
                                current_iou_stats[f"iou@{threshold}_pass@1"] = np.mean(pass1_scores)

                        # Log progress with pass@k metrics
                        accuracy = total_correct / (total_samples * num_passes) if total_samples > 0 else 0
                        logger.info(f"Processed {total_samples} samples, accuracy: {accuracy:.4f}")
                        logger.info(f"Pass@k metrics: {pass_at_k_results}")
                        if current_f1_stats:
                            logger.info(f"Current Mean F1 scores: {current_f1_stats}")
                        if current_iou_stats:
                            logger.info(f"Current IoU@threshold (Pass@1): {current_iou_stats}")
                        
            except queue.Empty:
                continue
                
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
        shutdown_event.set()
    except:
        # print traceback
        logger.error("Unexpected error during inference:")
        logger.error(traceback.format_exc())

        raise # let it crash
    finally:
        # Close progress bar
        pbar.close()
        
        # Ensure shutdown event is set so workers act on it
        shutdown_event.set()
        
        # Send poison pills to workers
        for _ in workers:
            try:
                input_queue.put(None)
            except:
                pass
        
        # Wait for workers to finish
        logger.info("Waiting for workers to finish...")
        
        # Drain output queue to prevent workers from blocking on put()
        # Allows background threads to flush data to pipe
        start_wait = time.time()
        while any(w.is_alive() for w in workers) and (time.time() - start_wait < 30):
            try:
                # Continuously drain the queue
                while True:
                    output_queue.get_nowait()
            except queue.Empty:
                pass
            except Exception:
                pass
            time.sleep(0.1)

        for worker in workers:
            worker.join(timeout=2)
            if worker.is_alive():
                logger.warning(f"Worker {worker.worker_id} did not terminate gracefully, forcing termination")
                try:
                    worker.terminate()
                    worker.join(timeout=1)
                    if worker.is_alive() and hasattr(worker, 'kill'):
                        worker.kill()
                except Exception as e:
                    logger.error(f"Error terminating worker {worker.worker_id}: {e}")
        
        # 最终计算所有的pass@k指标
        final_pass_at_k_results = {}
        for k in range(1, min(9, num_passes + 1)):
            pass_k_scores = []
            for scanrefer_id in sample_total_counts:
                n = sample_total_counts[scanrefer_id]
                c = sample_correct_counts[scanrefer_id]
                if n >= k:
                    pass_k = pass_at_k(n, c, k)
                    pass_k_scores.append(pass_k)
            
            if pass_k_scores:
                final_pass_at_k_results[f"pass@{k}"] = {
                    "mean": np.mean(pass_k_scores),
                    "std": np.std(pass_k_scores),
                    "num_samples": len(pass_k_scores)
                }
        
        # Save results
        if not getattr(args, 'no_save', False):
            logger.info(f"Saving final results to {run_dir}...")
            
            # Save all results
            all_results_file = os.path.join(run_dir, "all_results.json")
            with open(all_results_file, 'w') as f:
                json.dump(all_results, f, indent=2)
            
            # Save traces
            correct_traces_file = os.path.join(run_dir, "correct_traces.json")
            with open(correct_traces_file, 'w') as f:
                json.dump(dict(correct_traces), f, indent=2)
            
            incorrect_traces_file = os.path.join(run_dir, "incorrect_traces.json")
            with open(incorrect_traces_file, 'w') as f:
                json.dump(dict(incorrect_traces), f, indent=2)
            
            # Save per-sample statistics
            sample_stats = {}
            for scanrefer_id in sample_total_counts:
                sample_stats[scanrefer_id] = {
                    "correct": sample_correct_counts[scanrefer_id],
                    "total": sample_total_counts[scanrefer_id],
                    "accuracy": sample_correct_counts[scanrefer_id] / sample_total_counts[scanrefer_id],
                    "iou_scores": sample_iou_scores[scanrefer_id],  # Add per-pass IoU scores
                }
            
            sample_stats_file = os.path.join(run_dir, "sample_stats.json")
            with open(sample_stats_file, 'w') as f:
                json.dump(sample_stats, f, indent=2)

        # Calculate IoU pass@k results
        iou_thresholds = [0.25, 0.5]
        iou_pass_at_k_results = {}
        for threshold in iou_thresholds:
            iou_pass_at_k_results[f"iou@{threshold}"] = {}
            for k in range(1, min(9, num_passes + 1)):  # k from 1 to 8
                iou_pass_k_scores = []
                for scanrefer_id in sample_iou_scores:
                    ious = sample_iou_scores[scanrefer_id]
                    if len(ious) >= k:  # Only calculate if enough passes
                        # Count how many IoU scores meet or exceed threshold
                        c = sum(1 for iou in ious if iou >= threshold)
                        pass_k = pass_at_k(len(ious), c, k)
                        iou_pass_k_scores.append(pass_k)

                if iou_pass_k_scores:
                    iou_pass_at_k_results[f"iou@{threshold}"][f"pass@{k}"] = {
                        "mean": np.mean(iou_pass_k_scores),
                        "std": np.std(iou_pass_k_scores),
                        "num_samples": len(iou_pass_k_scores)
                    }

        # Calculate F1 statistics
        f1_results = {}
        all_f1_keys = set()
        for sid in sample_f1_scores:
            all_f1_keys.update(sample_f1_scores[sid].keys())
        
        for key in all_f1_keys:
            # Mean F1 across all passes
            all_scores = []
            # Max F1 per sample (best of k)
            max_scores = []
            
            for sid in sample_f1_scores:
                if key in sample_f1_scores[sid]:
                    scores = sample_f1_scores[sid][key]
                    all_scores.extend(scores)
                    if scores:
                        max_scores.append(max(scores))
            
            f1_results[key] = {
                "mean_all_passes": np.mean(all_scores) if all_scores else 0.0,
                "mean_best_pass": np.mean(max_scores) if max_scores else 0.0,
                "std_all_passes": np.std(all_scores) if all_scores else 0.0,
            }

        # Helper to recursively sanitize dictionary for JSON dump (handles Tokenizers, numpy types, etc.)
        def sanitize_for_json(obj):
            if isinstance(obj, (str, int, float, bool, type(None))):
                return obj
            elif isinstance(obj, (list, tuple)):
                return [sanitize_for_json(i) for i in obj]
            elif isinstance(obj, dict):
                return {k: sanitize_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, set):
                return [sanitize_for_json(i) for i in list(obj)]
            elif isinstance(obj, (SimpleNamespace, argparse.Namespace)):
                return sanitize_for_json(vars(obj))
            elif hasattr(obj, 'tolist'):  # numpy arrays
                return sanitize_for_json(obj.tolist())
            elif hasattr(obj, 'item'):  # numpy scalars
                return obj.item()
            else:
                return str(obj)

        # Save summary with pass@k results
        summary = sanitize_for_json({
            "total_samples": total_samples,
            "num_passes_per_sample": num_passes,
            "total_inferences": total_samples * num_passes,
            "total_correct": total_correct,
            "total_incorrect": total_samples * num_passes - total_correct,
            "overall_accuracy": total_correct / (total_samples * num_passes) if total_samples > 0 else 0,
            "samples_with_at_least_one_correct": len(correct_traces),
            "samples_with_all_incorrect": len([k for k in incorrect_traces if k not in correct_traces]),
            "pass_at_k_results": final_pass_at_k_results,
            "iou_pass_at_k_results": iou_pass_at_k_results,  # Add IoU pass@k results
            "f1_results": f1_results, # Add F1 results
            "timestamp": timestamp,
            "num_gpus": num_gpus,
            "args": vars(args),
            "partial_results_dir": partial_results_dir,
        })
        
        summary_file = os.path.join(run_dir, "summary.json")
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # 打印最终的pass@k结果
        if not getattr(args, 'no_save', False):
            logger.info(f"Results saved to {run_dir}")
        logger.info(f"Overall accuracy: {summary['overall_accuracy']:.4f}")
        logger.info("Final Pass@k results:")
        for k_metric, k_stats in final_pass_at_k_results.items():
            logger.info(f"  {k_metric}: {k_stats['mean']:.4f} (±{k_stats['std']:.4f}, n={k_stats['num_samples']})")

        # Print IoU pass@k results
        logger.info("IoU Pass@k results:")
        for threshold_key, threshold_results in iou_pass_at_k_results.items():
            logger.info(f"  {threshold_key.upper()}:")
            for k_metric, k_stats in threshold_results.items():
                logger.info(f"    {k_metric}: {k_stats['mean']:.4f} (±{k_stats['std']:.4f}, n={k_stats['num_samples']})")
        
        # Print F1 results
        logger.info("F1 results:")
        for key, stats in f1_results.items():
            logger.info(f"  {key}: Mean (all passes): {stats['mean_all_passes']:.4f}, Mean (best pass): {stats['mean_best_pass']:.4f}")
        
        return all_results, correct_traces, incorrect_traces, summary


# Keep the original functions unchanged
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
    
    # NEW: Extract precise object boxes map for IoU evaluation
    # Split into Prediction (Context/Proposal) lookup and GT lookup
    pred_object_id_to_box = []
    gt_object_id_to_box = []

    for ex in examples:
        # Check if new dictionary keys exist (from modified Real3DDataset)
        if "proposal_id2box" in ex:
            def convert_to_box_dict(box, oid):
                 return {
                    "x": float(box[0]), "y": float(box[1]), "z": float(box[2]),
                    "width": float(box[3]), "height": float(box[4]), "depth": float(box[5]),
                    "id": str(oid)
                }
            
            pred_map = {}
            if ex["proposal_id2box"]:
                for oid, box in ex["proposal_id2box"].items():
                    pred_map[str(oid)] = convert_to_box_dict(box, oid)
            
            gt_map = {}
            if "gt_id2box" in ex and ex["gt_id2box"]:
                for oid, box in ex["gt_id2box"].items():
                    gt_map[str(oid)] = convert_to_box_dict(box, oid)
            else:
                gt_map = pred_map.copy()


        pred_object_id_to_box.append(pred_map)
        gt_object_id_to_box.append(gt_map)

    # Extract instructions and responses
    instructions = [example["description"] for example in examples]
    responses = [example["expected_response"] for example in examples]
    
    # Create a batch
    batch = {
        "object_features": object_features,
        "object_masks": object_masks,
        "instructions": instructions,
        "responses": responses,
        "scene_ids": [example["scene_id"] for example in examples],
        "object_ids": [example["object_ids"] for example in examples],
        "scanrefer_id": [example["scanrefer_id"] for example in examples],
        "hash_id": [example["hash_id"] for example in examples],
        "raw_description": [example["raw_description"] for example in examples],
        "prompt_with_plan": [example["prompt_with_plan"] for example in examples],
        "pred_object_id_to_box": pred_object_id_to_box,
        "gt_object_id_to_box": gt_object_id_to_box,
    }
    
    return batch

def prepare_model_inputs(batch, device, tokenizer):
    """Prepare inputs for the model from batch data"""
    # Process object features
    object_set_embeds = []

    # move to device
    # device = model.device if hasattr(model, 'device') else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for features, mask in zip(batch["object_features"], batch["object_masks"]):
        # Apply mask to features
        valid_features = features[mask].to(device)
        object_set_embeds.append([valid_features])

    # For Qwen, add chat template
    instructions = batch["instructions"]
    if "qwen" in tokenizer.__class__.__name__.lower():
        instructions = [apply_qwen_template(inst, tokenizer)[0] for inst in instructions]

    # Prepare inputs for the model
    model_inputs = {
        "instructions": instructions,
        "object_set_embeds": object_set_embeds,
    }

    return model_inputs

def load_tokenizer(model_id: str):
    """Load tokenizer based on model ID"""
    if "Qwen2.5" in model_id or "Qwen2-" in model_id:
        tokenizer = transformers.Qwen2TokenizerFast.from_pretrained(model_id)
        tokenizer.padding_side = "left"
    elif "Qwen3" in model_id:
        # Qwen3 uses the same tokenizer as Qwen2
        tokenizer = transformers.Qwen2TokenizerFast.from_pretrained(model_id)
        tokenizer.padding_side = "left"
    else:
        raise ValueError(f"Model {model_id} not supported.")
    
    # Ensure tokenizer has pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return tokenizer

def load_model_and_tokenizer(args, device):
    """Load model and tokenizer from checkpoint"""
    model_id = args.model_name
    logger.info(f"Loading base model {model_id}...")

    # # Load tokenizer
    # if "Qwen2.5" in model_id or "Qwen2-" in model_id:
    #     tokenizer = transformers.Qwen2TokenizerFast.from_pretrained(model_id)
    #     tokenizer.padding_side = "left"
    # elif "Qwen3" in model_id:
    #     # Qwen3 uses the same tokenizer as Qwen2
    #     tokenizer = transformers.Qwen2TokenizerFast.from_pretrained(model_id)
    #     tokenizer.padding_side = "left"
    # else:
    #     raise ValueError(f"Model {model_id} not supported.")
    tokenizer = load_tokenizer(model_id)

    # Ensure tokenizer has pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load language model
    if "Qwen2.5" in model_id or "Qwen2-" in model_id:
        language_model = transformers.Qwen2ForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            attn_implementation=args.attn_implementation,
            device_map=device,
        )
    elif "Qwen3" in model_id:
        # Import Qwen3 model class if available
        try:
            from transformers import Qwen3ForCausalLM, Qwen3VLForConditionalGeneration
            if "vl" in model_id.lower():
                logger.info("Loading Qwen3VLForConditionalGeneration model for Qwen3 VL model")
                language_model = Qwen3VLForConditionalGeneration.from_pretrained(
                    model_id,
                    torch_dtype=torch.bfloat16,
                    attn_implementation=args.attn_implementation,
                    device_map=device,
                )
            else:
                logger.info("Loading Qwen3ForCausalLM model for Qwen3 model")
                language_model = Qwen3ForCausalLM.from_pretrained(
                    model_id,
                    torch_dtype=torch.bfloat16,
                    attn_implementation=args.attn_implementation,
                    device_map=device,
                )
        except ImportError:
            # Fallback: Qwen3 might use the same architecture as Qwen2
            logger.warning("Qwen3ForCausalLM not found, trying Qwen2ForCausalLM for Qwen3 model")
            raise
            language_model = transformers.Qwen2ForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
                attn_implementation=args.attn_implementation,
                device_map="auto",
            )
    else:
        raise ValueError(f"Model {model_id} not supported.")

    # Load LoRA if checkpoint path is provided
    # NOTE: loras will be served in SGLang - not here
    if args.checkpoint_path != "" and False:
        logger.info(f"Loading LoRA checkpoint from {args.checkpoint_path}...")
        
        # First, convert to PeftModel if not already
        if not isinstance(language_model, PeftModel):
            # Need to apply PEFT wrapper first
            language_model = PeftModel.from_pretrained(
                language_model,
                args.checkpoint_path,
                is_trainable=False,
                torch_device="cpu",
            )
        else:
            # Already a PeftModel, just load adapter
            message = language_model.load_adapter(
                model_id=args.checkpoint_path,
                adapter_name="default",
                is_trainable=False,
                torch_device="cpu",
            )
            logger.info(message)
        
        # Merge and unload LoRA weights
        language_model = language_model.to(device).merge_and_unload().to("cpu")
        logger.info("LoRA weights merged and unloaded")

    else:
        language_model = language_model.to("cpu")
        logger.info("LoRA to be served by SGLang, not loaded here.")

    return language_model, tokenizer

def create_dataset(dataset_type, args, split="val"):
    """Create dataset for inference"""
    dataset_class = DATASET_CLSMAP[dataset_type]

    enforce_nocot = False
    if "nocot" in dataset_type:
        enforce_nocot = True
        dataset_type = dataset_type.replace("_nocot", "")
    
    dataset = dataset_class(
        name=dataset_type,
        split=split,
        shuffle_objects=False,
        num_scenes=args.num_scenes, # no use??
        objects_per_scene=args.objects_per_scene,
        room_size=args.room_size,
        max_objects=args.max_objects,
        seed=args.seed,
        fix_template=args.fix_template,
        add_thinking_trace=args.add_thinking_trace and not enforce_nocot,
        pre_filter_objects=args.pre_filter_objects,
        ratio=args.ratio,
        use_clip_class_embedding=args.use_clip_class_embedding,
        clip_model_name=args.clip_model_name,
        cuda_device=0,
        use_proposal_feature=args.use_proposal_feature,
        proposal_type=args.proposal_type,
        normalize_proposal_feature=args.normalize_proposal_feature,
        use_2d_proposal_feature=args.use_2d_proposal_feature,
        load_from_cache=False,
        tokenizer=args.tokenizer,
        only_plans=args.use_nr3d_plan_from_program, # use NR3D plan from its program written by LLM
        image_encoder=args.image_encoder,
        image_feature_type=args.image_feature_type,
        n_views_in_m_views=args.n_views_in_m_views,
        add_thinking_trace_prompt=args.add_thinking_trace_prompt,
        sft=False,
    )
    
    return dataset

def extract_internal_plan_text(trace: str) -> str:
    """Extract the internal plan sentence from a thinking trace."""
    if not trace: 
        return ""
    # Extract the first sentence starting with 'Let's plan...' until newline
    match = re.search(r"(Let's plan my next steps[:\s]*.*?)(?:\n|$)", trace, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return ""

def extract_thinking_trace_from_response(response: str) -> Tuple[str, Dict[str, str]]:
    """
    Extract thinking trace and its parts from model response.
    
    Returns:
        thinking_trace: The complete thinking trace text
        parts: Dictionary containing different parts of the trace
    """
    parts = {
        "all": "",
        "plan": "",
        "execution": "",
        "header": "",
    }
    
    thinking_trace = ""
    
    # Check if response contains thinking trace
    # if "[APEIRIA THINKS]" in response and "[APEIRIA SPEAKS]" in response:
    if "[APEIRIA SPEAKS]" in response:
        # Extract complete thinking trace
        start_idx = response.find("[APEIRIA THINKS]")
        if start_idx == -1:
            start_idx = 0 # if no THINKS, start from beginning
        end_idx = response.find("[APEIRIA SPEAKS]") + len("[APEIRIA SPEAKS]")
        thinking_trace = response[start_idx:end_idx]
        
        # Split into lines for detailed parsing
        lines = thinking_trace.split("\n")
        parts["all"] = thinking_trace
        
        # Extract header (first few lines)
        header_lines = []
        for line in lines:
            header_lines.append(line)
            if "described as:" in line:
                break
        parts["header"] = "\n".join(header_lines)
        
        # Extract plan
        plan_start = False
        plan_lines = []
        for line in lines:
            if "plan" in line.lower() and ("steps" in line.lower() or "next" in line.lower()):
                plan_start = True
            elif plan_start and (line.strip() == "" or "First" in line or "Now" in line):
                break
            elif plan_start:
                plan_lines.append(line)
        
        if plan_lines:
            parts["plan"] = "\n".join(plan_lines)
        
        # The rest is execution
        execution_start = False
        execution_lines = []
        for line in lines:
            if ("First" in line or "Now" in line) and "[APEIRIA SPEAKS]" not in line:
                execution_start = True
            if execution_start and "[APEIRIA SPEAKS]" not in line:
                execution_lines.append(line)
        
        parts["execution"] = "\n".join(execution_lines)
    
    return thinking_trace, parts

def pass_at_k(n, c, k):
    """
    :param n: total number of samples
    :param c: number of correct samples
    :param k: k in pass@$k$
    """
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

def iou_pass_at_k(n, ious, k, threshold=0.5):
    """
    计算从n个样本中抽k个，至少有一个IoU>=threshold的概率
    :param n: 总样本数
    :param ious: 每个样本的IoU分数列表
    :param k: k in pass@k
    :param threshold: IoU阈值
    """
    c = sum(1 for iou in ious if iou >= threshold)

    return pass_at_k(n, c, k)

def box3d_overlap(boxes1, boxes2):
    """
    计算两组3D box的IoU
    Input:
        boxes1: (N, 6) [x, y, z, size_x, size_y, size_z]
        boxes2: (M, 6)
    Output:
        iou: (N, M)
    """
    N = boxes1.shape[0]
    M = boxes2.shape[0]
    
    vol1 = boxes1[:, 3] * boxes1[:, 4] * boxes1[:, 5]  # (N,)
    vol2 = boxes2[:, 3] * boxes2[:, 4] * boxes2[:, 5]  # (M,)
    
    boxes1_expanded = boxes1.unsqueeze(1).expand(N, M, 6)
    boxes2_expanded = boxes2.unsqueeze(0).expand(N, M, 6)
    
    min_corner = torch.maximum(
        boxes1_expanded[:, :, :3] - boxes1_expanded[:, :, 3:] / 2,
        boxes2_expanded[:, :, :3] - boxes2_expanded[:, :, 3:] / 2
    )
    
    max_corner = torch.minimum(
        boxes1_expanded[:, :, :3] + boxes1_expanded[:, :, 3:] / 2,
        boxes2_expanded[:, :, :3] + boxes2_expanded[:, :, 3:] / 2
    )
    
    inter_dims = torch.clamp(max_corner - min_corner, min=0)  # (N, M, 3)
    inter_vol = inter_dims[:, :, 0] * inter_dims[:, :, 1] * inter_dims[:, :, 2]  # (N, M)
    
    union_vol = vol1.unsqueeze(1) + vol2.unsqueeze(0) - inter_vol  # (N, M)
    iou = inter_vol / (union_vol + 1e-8)
    
    return iou

def evaluate_response(response: str, expected_response: str, dataset, iou_thresholds: List[float]=[0.25, 0.5], enforce_single_prediction: bool=True, pred_object_lookup: Dict = None, gt_object_lookup: Dict = None) -> Tuple[bool, Dict]:
    """
    Evaluate if the response is correct by comparing with expected response.
    
    Returns:
        is_correct: Boolean indicating if the response is correct
        details: Dictionary containing evaluation details
    """
    # Parse both responses to extract object information
    pred_objects = parse_response(response)
    expected_objects = parse_response(expected_response)

    # Helper to lookup precise box info
    def enhance_objects_with_lookup(objects_list, lookup_map):
        if not lookup_map:
            return objects_list
        enhanced = []
        for obj in objects_list:
            oid = str(obj["id"])
            if oid in lookup_map:
                # Create new obj dict with precise coordinates from lookup
                precise_obj = obj.copy()
                gt = lookup_map[oid]
                precise_obj.update({
                    "x": gt["x"], "y": gt["y"], "z": gt["z"],
                    "width": gt["width"], "height": gt["height"], "depth": gt["depth"]
                })
                enhanced.append(precise_obj)
            else:
                enhanced.append(obj)
        return enhanced

    # Apply lookup if available: Separate lookups for prediction and GT
    if pred_object_lookup:
        pred_objects = enhance_objects_with_lookup(pred_objects, pred_object_lookup)
    
    if gt_object_lookup:
        expected_objects = enhance_objects_with_lookup(expected_objects, gt_object_lookup)

    # Helper to deduplicate objects by ID to avoid inflating metrics logic
    def deduplicate_objects(objects):
        seen_ids = set()
        unique_objects = []
        for obj in objects:
            if obj["id"] not in seen_ids:
                seen_ids.add(obj["id"])
                unique_objects.append(obj)
            else:
                logger.warning(f"Duplicate object ID {obj['id']} found in response, all ids: {[o['id'] for o in objects]}")
        return unique_objects
    
    # Deduplicate expected objects (GT should be unique)
    expected_objects = deduplicate_objects(expected_objects)
    # Deduplicate predicted objects
    pred_objects = deduplicate_objects(pred_objects)
    
    # Extract object IDs
    pred_ids = set([obj["id"] for obj in pred_objects])
    if enforce_single_prediction:
        # always take first prediction if multiple
        # SYNC FIX: Filter pred_objects directly to match the pred_ids logic
        if len(pred_objects) > 0:
            pred_objects = pred_objects[:1]
            pred_ids = set([pred_objects[0]["id"]])
        else:
            pred_ids = set()

    expected_ids = set([obj["id"] for obj in expected_objects])
    
    # Check if predicted IDs match expected IDs
    is_correct = pred_ids == expected_ids
    
    # Calculate IoU if both have objects
    ious = {} # iou_threshold -> {recall, precision, f1}
    mious = []

    # for zero-target cases
    if len(expected_objects) == 0:
        if len(pred_objects) == 0:
            # both empty, perfect match
            for thresh in iou_thresholds:
                ious[thresh] = {
                    "recall": 1.0,
                    "precision": 1.0,
                    "f1": 1.0,
                }
            mean_iou = 1.0

        elif len(pred_objects) > 0:
            # predictions but no expected, all wrong
            for thresh in iou_thresholds:
                ious[thresh] = {
                    "recall": 0.0,
                    "precision": 0.0,
                    "f1": 0.0,
                }
            mean_iou = 0.0

    # no predictions, zero scores
    elif len(pred_objects) == 0 and len(expected_objects) > 0:
        for thresh in iou_thresholds:
            ious[thresh] = {
                "recall": 0.0,
                "precision": 0.0,
                "f1": 0.0,
            }
        mean_iou = 0.0

    elif pred_objects and expected_objects:
        pred_boxes = np.array([[obj["x"], obj["y"], obj["z"], obj["width"], obj["height"], obj["depth"]] for obj in pred_objects])
        expected_boxes = np.array([[obj["x"], obj["y"], obj["z"], obj["width"], obj["height"], obj["depth"]] for obj in expected_objects])

        iou = box3d_overlap(torch.tensor(pred_boxes), torch.tensor(expected_boxes))  # (num_pred, num_exp)
        
        # Hungarian matching
        max_dim = max(iou.shape)
        padded_iou_matrix = np.zeros((max_dim, max_dim))
        padded_iou_matrix[:iou.shape[0], :iou.shape[1]] = iou

        row_idx, col_idx = linear_sum_assignment(-padded_iou_matrix)  # maximize IoU
        
        for thr in iou_thresholds:
            _tp = 0
            for i in range(len(pred_ids)):
                this_iou = padded_iou_matrix[row_idx[i], col_idx[i]]
                if this_iou >= thr:
                    _tp += 1  # True positive
            ious[thr] = {
                "recall": _tp / len(expected_objects) if len(expected_objects) > 0 else 0.0,
                "precision": _tp / len(pred_ids) if len(pred_ids) > 0 else 0.0,
                "f1": 2 * _tp / (len(pred_ids) + len(expected_objects)) if (len(pred_ids) + len(expected_objects)) > 0 else 0.0,
            }

        # Max-match strategy
        # calculate recall, precision, f1 @ different thresholds
        # for thresh in iou_thresholds:
        #     matched_pred = set()
        #     matched_exp = set()
        #     for i in range(iou.shape[0]):
        #         for j in range(iou.shape[1]):
        #             if iou[i, j] >= thresh:
        #                 matched_pred.add(i)
        #                 matched_exp.add(j)
        #     recall = len(matched_exp) / len(expected_objects) if expected_objects else 0.0
        #     precision = len(matched_pred) / len(pred_objects) if pred_objects else 0.0
        #     f1 = 2 * (precision * recall) / (precision + recall + 1e-8) if (precision + recall) > 0 else 0.0
        #     ious[thresh] = {
        #         "recall": recall,
        #         "precision": precision,
        #         "f1": f1,
        #     }

        mious = iou.max(dim=1).values.numpy().tolist()  # max IoU for each predicted box

    iou_result_dict = {
        f"{metric}@{thresh}": scores[metric]
        for thresh, scores in ious.items()
        for metric in scores
    } # "recall@0.5", "precision@0.5", "f1@0.5", ...
        
    
    mean_iou = np.mean(mious) if mious else 0.0
    
    details = {
        "predicted_ids": list(pred_ids),
        "expected_ids": list(expected_ids),
        "id_match": is_correct,
        "mean_iou": mean_iou,
        **iou_result_dict,
        "num_predicted": len(pred_ids),
        "num_expected": len(expected_ids),
    }
    
    return is_correct, details

@hydra.main(version_base=None, config_path="configs", config_name="apeiria_mllm_inference")
def main(cfg: Config):
    """Main inference function"""
    logger.info("Starting inference with the following configuration:")
    logger.info(OmegaConf.to_yaml(cfg))

    # num_gpus = getattr(args, 'num_gpus', None)
    # get num_gpus from CUDA_VISIBLE_DEVICES if set, otherwise use all available GPUs
    num_gpus = len(os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")) if "CUDA_VISIBLE_DEVICES" in os.environ else torch.cuda.device_count()
    logger.info(f"Number of GPUs available: {num_gpus}")

    available_gpus = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if "CUDA_VISIBLE_DEVICES" in os.environ else list(range(num_gpus))
    logger.info(f"Available GPUs: {available_gpus}")

    # Set up signal handler
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Convert config to args
    args = SimpleNamespace(**OmegaConf.to_container(cfg, resolve=True))

    # for nr3d, scanrefer, sr3d, enforce_single_prediction is always True; for m3dref, might be many predictions
    if args.dataset_type in ["nr3d", "scanrefer", "sr3d", "nr3d_nocot", "scanrefer_nocot", "sr3d_nocot"]:
        args.enforce_single_prediction = True
    else:
        args.enforce_single_prediction = False

    # merge train args into args
    if hasattr(cfg, "train"):
        train_args = SimpleNamespace(**OmegaConf.to_container(cfg.train, resolve=True))
        for key, value in vars(train_args).items():
            if key not in vars(args): # avoid overwriting existing args
                setattr(args, key, value)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    if args.resume_from_checkpoint:
        if args.checkpoint_path == "" or args.checkpoint_path is None or args.checkpoint_path == args.resume_from_checkpoint:
            args.checkpoint_path = args.resume_from_checkpoint
        else:
            logger.warning(f"resume_from_checkpoint is set to {args.resume_from_checkpoint}, but checkpoint_path is set to {args.checkpoint_path}. Using checkpoint_path instead.")
    
    # Add resume from partial results dir to args
    args.resume_from_partial_results_dir = getattr(cfg, "resume_from_partial_results_dir", None)
    args.no_save = getattr(cfg, "no_save", False)
    
    # Load external plans if configured
    args.external_plan_path = getattr(cfg, "external_plan_path", None)
    args.previous_results_path = getattr(cfg, "previous_results_path", None) # Path to previous results for extraction
    args.gt_scene_data_path = getattr(cfg, "gt_scene_data_path", None) # Path to ground truth scene jsons for injection
    
    external_plans = None
    if args.external_plan_path:
        logger.info(f"Loading external plans from {args.external_plan_path}...")
        try:
            with open(args.external_plan_path, 'r') as f:
                plans_list = json.load(f)
            external_plans = {}
            for item in plans_list:
                # Key: (scene_id, object_id, description)
                # Ensure object_id is converted to string to match batch processing
                # key = (item['scene_id'], str(item['object_id']), item['description'])
                # use object_ids to support multiple or non object ids
                key = (item['scene_id'], build_hash_from_object_id_list(item['object_ids']), item['description'])
                external_plans[key] = item['generated_plan']
            logger.info(f"Loaded {len(external_plans)} external plans.")
        except Exception as e:
            logger.error(f"Failed to load external plans: {e}")
            raise e
    elif args.previous_results_path:
        # This is required when: using external perception results.
        # Load plans from previous inference results
        logger.info(f"Loading internal plans from previous results: {args.previous_results_path}...")
        try:
            with open(args.previous_results_path, 'r') as f:
                prev_results = json.load(f)
            
            external_plans = {}
            count_extracted = 0
            
            # Determine list to iterate
            assert isinstance(prev_results, (list, dict)), "Previous results should be a list or a dict with 'all_results' key."
            results_list = prev_results if isinstance(prev_results, list) else prev_results.get('all_results', [])
            
            for item in results_list:
                # Extract plan from first pass
                passes = item.get('passes', [])
                if not passes: continue
                
                trace = passes[0].get('thinking_trace', '')
                response = passes[0].get('response', '')
                extracted_plan = extract_internal_plan_text(trace)
                if extracted_plan == "":
                    # Try extracting from response if not found in trace
                    extracted_plan = extract_internal_plan_text(response)
                
                if extracted_plan:
                    scene_id = item['scene_id']
                    obj_ids = item['object_ids']
                    raw_description = item.get('raw_description', '')
                    
                    # Handle object_id list
                    # if isinstance(obj_ids, list) and len(obj_ids) > 0:
                    #     obj_id_key = str(obj_ids[0])
                    # else:
                    #     obj_id_key = str(obj_ids)
                        
                    # key = (scene_id, obj_id_key, raw_description)
                    key = (scene_id, build_hash_from_object_id_list(item['object_ids']), raw_description)
                    external_plans[key] = extracted_plan
                    count_extracted += 1

                else:
                    logger.warning(f"No internal plan extracted for scene {item['scene_id']} and object_ids {item['object_ids']}, raw_description: {item.get('raw_description', '')}")
            
            logger.info(f"Extracted {count_extracted} internal plans from {len(results_list)} previous results.")
            
        except Exception as e:
            logger.error(f"Failed to load previous results for plan extraction: {e}")
            raise e

    # ==== SGLang参数适配 ====
    args.use_sglang_for_generation = getattr(cfg, "use_sglang_for_generation", False)
    args.sglang_lora_paths = [args.checkpoint_path]
    args.sglang_port = [find_free_port() for _ in range(num_gpus)]

    # Load dataset
    dataset_types = args.dataset_type.strip("[]").split(",") if "[" in args.dataset_type else [args.dataset_type]
    dataset_types = [dt.strip() for dt in dataset_types]
    
    all_datasets = []
    assert len(dataset_types) == 1, "Currently only one dataset type is supported for inference"
    tokenizer = load_tokenizer(args.model_name)
    args.tokenizer = tokenizer

    for dtype in dataset_types:
        dataset = create_dataset(dtype, args, split=args.split)
        print(dataset[0])

        all_datasets.append(dataset)
        logger.info(f"Loaded {dtype} dataset with {len(dataset)} samples")
    
    # Merge datasets if multiple
    # if len(all_datasets) > 1:
    #     dataset = MergedDataset(all_datasets)
    # else:

    dataset = all_datasets[0]
    
    # Resume from partial results by filtering the dataset
    preloaded_data = {}
    if args.resume_from_partial_results_dir:
        logger.info(f"Attempting to resume from partial results in: {args.resume_from_partial_results_dir}")
        preloaded_data, completed_scanrefer_ids = load_partial_results(args.resume_from_partial_results_dir, dataset)
        
        dataset.exclude_sample_by_scanrefer_id(completed_scanrefer_ids)


    # Get feature dimensions
    feature_dim = dataset.feature_dim
    modality_dims = dataset.modality_dims
    modality_order = dataset.modality_order
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
    )
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run multiprocessing inference
    results = run_multiprocessing_inference(
        dataloader=dataloader,
        args=args,
        num_passes=args.num_inference_passes,
        save_dir=args.output_dir,
        feature_dim=feature_dim,
        modality_dims=modality_dims,
        modality_order=modality_order,
        preloaded_data=preloaded_data,
        num_gpus=num_gpus,
        available_gpus=available_gpus,
        external_plans=external_plans,  # Pass external_plans
    )

    logger.info("Inference completed!")

if __name__ == "__main__":
    main()
