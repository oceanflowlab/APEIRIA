import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import Sampler
import torch.distributed as dist
import torch.nn.functional as F


import json
import os
import pickle
from PIL import Image
import math
import numpy as np
import logging
from typing import Union, Optional, List, Callable, Tuple, Dict, Any, Set, Iterable
from enum import Enum
import accelerate
import random
from collections import OrderedDict, defaultdict
import hashlib
from icecream import ic
from copy import deepcopy
import re
from functools import wraps
from dataclasses import dataclass
import uuid
import base64
import pretty_errors
from datetime import datetime
from tqdm.auto import tqdm

from general_utils import Singleton, AverageMeter, TimingMeter, print_once, softmax_focal_loss
from apeiria_parser import LEGAL_FUNCTIONS, PARSER, parse_program_string, is_valid_program, extract_function_names, is_valid_legal_type_valid_program, get_default_signatures, get_grounding_signatures, get_call_depth, analyze_function_calls, analyze_programs_attributes

logger = logging.getLogger(__name__)

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
# SVC_PATH = "/home/mwt/hdd/SVC"
SVC_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../SVC")

MEAN_COLOR_RGB = np.array([109.8, 97.2, 83.8])

def gather_scalar(accelerator, scalar):
    scalar = torch.tensor(scalar).float().to(accelerator.device)
    return accelerator.gather(scalar).mean().item()

def find_params_wo_grad(model):
    for name, param in model.named_parameters():
        if param.grad is None:
            yield name, param

def scanrefer_to_hash_id(scanrefer_id: str) -> str:
    """
    Convert scanrefer_id to hash_id, and in base64
    """
    hash_id = hashlib.md5(scanrefer_id.encode()).hexdigest()
    return base64.b64encode(hash_id.encode()).decode()[:8] # 10 characters == 64^10 slots


class MergedDataset(Dataset):
    """
    wrapper for multiple datasets.
    get from each dataset by index
    """

    def __init__(
        self,
        datasets: Iterable[Dataset],
        sample_with_sqrt_freq: bool = False,
        annealing_schedule: list[tuple[float, float]] = None,
        seed: int = 42,
        shuffle: bool = True,
        curriculum_learning: bool = False,  # Add new parameter
        curriculum_steps: int = None,  
    ):
        self.datasets = list(datasets)
        self.lengths = [len(d) for d in datasets]
        self.seed = seed
        self.shuffle = shuffle
        self.curriculum_learning = curriculum_learning
        self.curriculum_steps = curriculum_steps

        if curriculum_learning:
            logger.info("Initializing curriculum learning based on program complexity")
            self._setup_curriculum()

        # sample with sqrt frequency
        self.sample_with_sqrt_freq = sample_with_sqrt_freq
        if sample_with_sqrt_freq:
            logger.info("Sample with sqrt frequency")
            self.freqs = [np.sqrt(len(d)) for d in datasets]
            self.freqs = np.array(self.freqs)
            self.freqs = self.freqs / self.freqs.sum()

        self.cum_lengths = np.cumsum(self.lengths)
        logger.info(f"Total merged dataset length: {self.cum_lengths[-1]}")
        if annealing_schedule is not None:
            self.annealing_schedule = annealing_schedule
            # logger.info(f"Generating indices with annealing schedule: {annealing_schedule}")
            # log each dataset name and ratio start.end
            for i, d in enumerate(datasets):
                logger.info(f"Dataset {d.name}: {annealing_schedule[i]}")
            logger.info(f"Generating indices with annealing schedule.")
            self._check_annealing_schedule()
            self._generate_indices()
            logger.info(f"Generating indices with annealing schedule done.")

    def __len__(self):
        return self.cum_lengths[-1]
    
    def _setup_curriculum(self):
        """Setup curriculum learning by sorting all samples based on complexity"""
        # g = torch.Generator()
        # g.manual_seed(self.seed)
        
        # Collect all samples with their complexities and original indices
        all_samples = []
        for dataset_idx, dataset in enumerate(self.datasets):
            for sample_idx in range(len(dataset)):
                sample = dataset[sample_idx]
                complexity = sample['program_complexity']
                all_samples.append((complexity, dataset_idx, sample_idx))
        
        # Sort by complexity
        all_samples.sort(key=lambda x: x[0])

        self.all_samples = all_samples
        self.reseed()

    def reseed(self, new_seed=None):
        if getattr(self, "all_samples", None) is None:
            logger.info("Curriculum learning not set up, skipping reseed")
            return
        
        seed = self.seed
        if new_seed is not None:
            seed = new_seed + self.seed
        logger.info(f"Reseeding curriculum learning with seed {seed}")
        # Store sorted indices
        # self.curriculum_indices = [(d_idx, s_idx) for _, d_idx, s_idx in all_samples]
        # Store complexity-grouped indices
        self.curriculum_indices = defaultdict(list)
        for i, (complexity, d_idx, s_idx) in enumerate(self.all_samples):
            self.curriculum_indices[complexity].append((d_idx, s_idx))

        # shuffle inside each complexity group, deterministic
        g = torch.Generator()
        g.manual_seed(seed)
        for complexity in self.curriculum_indices:
            perm = torch.randperm(len(self.curriculum_indices[complexity]), generator=g) #.numpy()
            # self.curriculum_indices[complexity] = np.random.permutation(self.curriculum_indices[complexity])
            self.curriculum_indices[complexity] = [self.curriculum_indices[complexity][i] for i in perm]

        # log complexity distribution
        complexity_counts = {complexity: len(idxs) for complexity, idxs in self.curriculum_indices.items()}
        complexity_counts = sorted(complexity_counts.items(), key=lambda x: x[0])

        # Flatten the list, from small to large complexity
        self.curriculum_indices = [idxs for complexity in sorted(self.curriculum_indices.keys()) for idxs in self.curriculum_indices[complexity]]

        
        logger.info(f"Curriculum setup complete. Complexity range: "
                   f"{self.all_samples[0][0]} to {self.all_samples[-1][0]}")
        logger.info(f"Complexity distribution: {complexity_counts}")
        

    def _check_annealing_schedule(self):
        # check sum-1 property, normalize
        r_start_sum = sum([r_start for r_start, r_end in self.annealing_schedule])
        r_end_sum = sum([r_end for r_start, r_end in self.annealing_schedule])
        
        logger.info(f"Annealing schedule sum: {r_start_sum}, {r_end_sum}, normalizing to 1")
        self.annealing_schedule = [(r_start / r_start_sum, r_end / r_end_sum) for r_start, r_end in self.annealing_schedule]

    def _generate_indices(self):
        # generate deterministic indices
        g = torch.Generator()
        g.manual_seed(self.seed)

        self.indices = []
        self.dataset_indices = []
        # sample one by one
        ratios = [
            torch.linspace(r_start, r_end, steps=len(self)) for r_start, r_end in self.annealing_schedule
        ] # [N, L], N is the number of datasets, L is the length of the merged dataset
        ratios = torch.stack(ratios, dim=1) # [L, N]

        remaining_count = torch.tensor([len(d) for d in self.datasets]) # [N]
        if self.shuffle:
            remaining_indices = [
                torch.randperm(len(d), generator=g).tolist() for d in self.datasets
            ]
        else:
            remaining_indices = [list(range(len(d))) for d in self.datasets]

        for i in range(len(self)):
            # sample dataset 
            current_ratio = ratios[i] * remaining_count 
            current_ratio = current_ratio / current_ratio.sum()
            dataset_idx = torch.multinomial(current_ratio, 1, generator=g).item()
            self.dataset_indices.append(dataset_idx)

            # sample one from the dataset
            remaining_count[dataset_idx] -= 1
            idx = remaining_indices[dataset_idx][remaining_count[dataset_idx]] # take the last one
            self.indices.append(idx)


    def __getitem__(self, idx):
        if self.curriculum_learning:
            # Calculate how many samples to make available based on training progress
            # progress = idx / self.curriculum_steps
            progress = idx / len(self)
            progress = min(1.0, progress)  # Cap at 1.0
            
            # Calculate available samples (gradually increase from 10% to 100%)
            # available_samples = int(0.1 * len(self) + progress * 0.9 * len(self))
            
            # Get actual index from curriculum sorted indices
            # actual_idx = idx % available_samples
            # dataset_idx, sample_idx = self.curriculum_indices[actual_idx]
            # return self.datasets[dataset_idx][sample_idx]

            # sample from the curriculum indices
            dataset_idx, sample_idx = self.curriculum_indices[idx]
            return self.datasets[dataset_idx][sample_idx]
     

        if hasattr(self, "annealing_schedule"):
            # annealing schedule, to combine without shuffle
            return self.datasets[self.dataset_indices[idx]][self.indices[idx]]
        elif not self.sample_with_sqrt_freq:
            # normal sampling
            for i, cum_len in enumerate(self.cum_lengths):
                if idx < cum_len:
                    return self.datasets[i][idx - (cum_len - self.lengths[i])]
        else:
            # sample with sqrt frequency
            dataset_idx = np.random.choice(len(self.datasets), p=self.freqs)
            return self.datasets[dataset_idx][idx % self.lengths[dataset_idx]]

        raise IndexError("Index out of range")
    
def get_optimizer_param_groups_by_names_dict(
    model: nn.Module,
    names_dict: OrderedDict[str, List[str]],
    lr_dict: Dict[str, float],
    weight_decay_dict: Dict[str, float],
    lr_default: Optional[float] = None,
    weight_decay_default: Optional[float] = None,
) -> Tuple[Dict[str, Dict[str, Union[List[torch.nn.Parameter], float]]], Dict[str, List[str]]]:
    """
    Get optimizer parameter groups by names dict - it shall be OrderedDict to represent the priority.
    unspecifed parameters will be assigned with default values.
    unspecified lr/weight_decay will be assigned with default values or lr_dict["default"]/weight_decay_dict["default"], if the previous is not specified.
    ignores parameters that do not require grad.
    """
    assert (
        lr_default is not None or "default" in lr_dict
    ), "lr_default must be specified or lr_dict must contain a 'default' key"
    assert (
        weight_decay_default is not None or "default" in weight_decay_dict
    ), "weight_decay_default must be specified or weight_decay_dict must contain a 'default' key"
    lr_default = lr_default if lr_default is not None else lr_dict["default"]
    weight_decay_default = (
        weight_decay_default
        if weight_decay_default is not None
        else weight_decay_dict["default"]
    )

    param_groups = {
        param_group_name: {
            "params": [],
            "lr": lr_dict.get(param_group_name, lr_default),
            "weight_decay": weight_decay_dict.get(param_group_name, weight_decay_default),
        }
        for param_group_name in names_dict.keys()
    }
    # add default param group
    param_groups["default"] = {
        "params": [],
        "lr": lr_dict.get("default", lr_default),
        "weight_decay": weight_decay_dict.get("default", weight_decay_default),
    }
    param_names_groups = {param_group_name: [] for param_group_name in param_groups.keys()}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        selected = False
        for param_group_name, param_names in names_dict.items():
            if any([param_name in name for param_name in param_names]):
                param_groups[param_group_name]["params"].append(param)
                param_names_groups[param_group_name].append(name)
                selected = True
                break
        if not selected:
            param_groups["default"]["params"].append(param)
            param_names_groups["default"].append(name)

    # remove empty param groups
    param_groups = {
        param_group_name: param_group
        for param_group_name, param_group in param_groups.items()
        if len(param_group["params"]) > 0
    }

    return param_groups, param_names_groups

def box3d_iou_orthogonal(xyzlwh1, xyzlwh2):
    """
    Compute 3D bounding box IoU. 
    Assume the boxes are aligned with the axis.
    Don't use triangles/convex hulls, they are *very slow*.
    """
    # normalize sizes
    xyzlwh1[..., 3:] = np.abs(xyzlwh1[..., 3:])
    xyzlwh2[..., 3:] = np.abs(xyzlwh2[..., 3:])

    x1, y1, z1, l1, w1, h1 = xyzlwh1
    x2, y2, z2, l2, w2, h2 = xyzlwh2

    def box3d_vol_orthogonal(l, w, h):
        return l*w*h

    def overlap_1d(x1, x2, y1, y2): 
        """
        return the overlap of 1d segment [x1, x2] and [y1, y2]
        """
        return max(0, min(x2, y2) - max(x1, y1))
    
    vol1 = box3d_vol_orthogonal(l1, w1, h1)
    vol2 = box3d_vol_orthogonal(l2, w2, h2)


    overlap_x = overlap_1d(x1 - l1/2, x1 + l1/2, x2 - l2/2, x2 + l2/2)
    overlap_y = overlap_1d(y1 - w1/2, y1 + w1/2, y2 - w2/2, y2 + w2/2)
    overlap_z = overlap_1d(z1 - h1/2, z1 + h1/2, z2 - h2/2, z2 + h2/2)

    inter_vol = box3d_vol_orthogonal(overlap_x, overlap_y, overlap_z)
    iou = inter_vol / (vol1 + vol2 - inter_vol)
    # replace nan with 0
    iou = np.nan_to_num(iou, nan=0.0)

    return iou

def mutual_iou(predictions, gts) -> np.ndarray:
    """
    predictions ~ (K1, 6), xyzhwl
    gts ~ (K2, 6)
    """
    iou_matrix = np.zeros((len(predictions), len(gts)))
    for i, pred in enumerate(predictions):
        for j, gt in enumerate(gts):
            # iou_matrix[i, j], _ = box3d_iou(
            #     get_3d_box(pred[3:], 0, pred[:3]), get_3d_box(gt[3:], 0, gt[:3])
            # )
            iou_matrix[i, j] = box3d_iou_orthogonal(
                pred[:6], gt[:6]
            )

    return iou_matrix

def mutual_iou_vectorized(predictions: np.ndarray, gts: np.ndarray) -> np.ndarray:
    """
    Vectorized calculation of 3D IoU matrix for axis-aligned boxes.

    Args:
        predictions (np.ndarray): Predicted boxes (K1, 6) -> (x, y, z, l, w, h)
        gts (np.ndarray): Ground truth boxes (K2, 6) -> (x, y, z, l, w, h)

    Returns:
        np.ndarray: IoU matrix (K1, K2)
    """
    K1 = predictions.shape[0]
    K2 = gts.shape[0]

    # Ensure dimensions (l, w, h) are positive
    # (K1, 6) -> (K1, 1, 6)
    preds_expanded = np.expand_dims(predictions, axis=1)
    preds_xyz = preds_expanded[..., :3]
    preds_lwh = np.abs(preds_expanded[..., 3:]) # Shape (K1, 1, 3)

    # (K2, 6) -> (1, K2, 6)
    gts_expanded = np.expand_dims(gts, axis=0)
    gts_xyz = gts_expanded[..., :3]
    gts_lwh = np.abs(gts_expanded[..., 3:])     # Shape (1, K2, 3)

    # Calculate min and max coordinates for all boxes
    # Broadcasting happens here: (K1, 1, 3) op (1, K2, 3) -> (K1, K2, 3)
    preds_min = preds_xyz - preds_lwh / 2.0
    preds_max = preds_xyz + preds_lwh / 2.0
    gts_min = gts_xyz - gts_lwh / 2.0
    gts_max = gts_xyz + gts_lwh / 2.0

    # Calculate intersection corners
    inter_min = np.maximum(preds_min, gts_min) # Shape (K1, K2, 3)
    inter_max = np.minimum(preds_max, gts_max) # Shape (K1, K2, 3)

    # Calculate intersection lengths, widths, heights
    # Ensure non-negative dimensions (if no overlap, dimension is 0)
    inter_lwh = np.maximum(0.0, inter_max - inter_min) # Shape (K1, K2, 3)

    # Calculate intersection volume
    inter_vol = np.prod(inter_lwh, axis=2) # Shape (K1, K2)

    # Calculate individual box volumes
    preds_vol = np.prod(preds_lwh, axis=2) # Shape (K1, 1)
    gts_vol = np.prod(gts_lwh, axis=2)     # Shape (1, K2)

    # Calculate union volume
    # Broadcasting: (K1, 1) + (1, K2) - (K1, K2) -> (K1, K2)
    union_vol = preds_vol + gts_vol - inter_vol # Shape (K1, K2)

    # Calculate IoU
    # Add epsilon to avoid division by zero
    iou_matrix = inter_vol / np.maximum(union_vol, 1e-8) # Shape (K1, K2)

    return iou_matrix
    

def assign_preds_to_gts(predictions, gts) -> Tuple[np.ndarray, np.ndarray]:
    """
    predictions ~ (K1, 6), (x,y,z, size_x, size_y, size_z)
    gts ~ (K2, 6)
    """
    iou_matrix = mutual_iou(predictions, gts)

    # no need of hungarian algorithm, since we can have duplicate assignments
    pred_idx_assigned, pred_iou = np.argmax(iou_matrix, axis=0), np.max(
        iou_matrix, axis=0
    )  # (K2, ), (K2, )
    return pred_idx_assigned, pred_iou

def assign_gts_to_preds(predictions, gts) -> Tuple[np.ndarray, np.ndarray]:
    """
    predictions ~ (K1, 6), xyzhwl
    gts ~ (K2, 6)
    """
    iou_matrix = mutual_iou(predictions, gts)

    # no need of hungarian algorithm, since we can have duplicate assignments
    gt_idx_assigned, gt_iou = np.argmax(iou_matrix, axis=1), np.max(
        iou_matrix, axis=1
    )  # (K1, ), (K1, )
    return gt_idx_assigned, gt_iou



@dataclass
class NYU40Object(metaclass=Singleton):
    # num_classes: int = 40
    # dummy_class: int = 40

    def __init__(self):
        with open(f"{DATA_PATH}/scannetv2-labels.combined.tsv") as f:
            lines = f.readlines()
        # self.object_classes = [line.split("\t")[1] for line in lines[1:]]
        self.object_classes = [line.split("\t")[7] for line in lines]
        self.object_classes = list(map(lambda x: x.strip(), self.object_classes))
        self.object_classes = sorted(self.object_classes) + ["unknown"]  # make sure no randomness in class index

        self.name_to_id = {name: i for i, name in enumerate(self.object_classes)}

        self.num_classes = len(self.object_classes)
        self.dummy_class = self.num_classes - 1 # add one dummy class

        logger.info(f"Loaded {self.num_classes} ScanNet NYU40 object classes")

# @dataclass(init=False)
class ScanNetRawObject(metaclass=Singleton):
    # name_to_id: Dict[str, int]
    # object_classes: List[str]
    # num_classes: int = 607
    # dummy_class: int = 607

    def __init__(self):
        with open(f"{DATA_PATH}/meta_data/scannetv2-labels.combined.tsv") as f:
            lines = f.readlines()
        self.object_classes = [line.split("\t")[1] for line in lines[1:]]
        self.object_classes = list(map(lambda x: x.strip(), self.object_classes))
        self.object_classes = sorted(self.object_classes) + ["unknown"] # make sure no randomness in class index

        self.name_to_id = {name: i for i, name in enumerate(self.object_classes)}

        self.num_classes = len(self.object_classes)
        self.dummy_class = self.num_classes - 1 # add one dummy class

        logger.info(f"Loaded {self.num_classes} ScanNet raw object classes")

# --- Synthetic Simple Dataset ---

def get_spatial_relation(obj1_pos: np.ndarray, obj2_pos: np.ndarray, threshold: float = 2.0, add_none_relation: bool = False) -> List[SpatialRelation]:
    """Determine spatial relations between two objects based on their positions"""
    relations = []
    
    # Extract positions
    x1, y1, z1 = obj1_pos[:3]
    x2, y2, z2 = obj2_pos[:3]
    
    # Horizontal relations (x-axis)
    if x1 < x2 - threshold:
        relations.append(SpatialRelation.LEFT)
    elif x1 > x2 + threshold:
        relations.append(SpatialRelation.RIGHT)
    
    # Vertical relations (y-axis)
    if y1 < y2 - threshold:
        relations.append(SpatialRelation.BELOW)
    elif y1 > y2 + threshold:
        relations.append(SpatialRelation.ABOVE)
    
    # Depth relations (z-axis)
    if z1 < z2 - threshold:
        relations.append(SpatialRelation.IN_FRONT_OF)
    elif z1 > z2 + threshold:
        relations.append(SpatialRelation.BEHIND)
    
    # Distance relation
    distance = np.sqrt((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)
    if distance < threshold * 1.5:
        relations.append(SpatialRelation.NEAR)
    elif distance > threshold * 3:
        relations.append(SpatialRelation.FAR)

    if add_none_relation and len(relations) == 0:
        relations.append(SpatialRelation.NONE)

    return relations


def format_multiple_predicates(predicates: List[str]):
    # use "and" for the last predicate, and ", " for the rest
    if len(predicates) == 1:
        return predicates[0]

    try:
        return ", ".join(predicates[:-1]) + " and " + predicates[-1]
    except Exception as e:
        ic(e)
        ic(predicates)
        raise e

def calculate_gradient_norms(model):
    """Calculate gradient norms for different parameter groups"""
    grad_norms = {}
    
    # Overall model gradient norm
    total_norm = 0.0
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            param_norm = param.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
            
            # Group parameters by their prefix (e.g., transformer, lm, etc.)
            prefix = name.split('.')[0]
            if prefix not in grad_norms:
                grad_norms[prefix] = 0.0
            grad_norms[prefix] += param_norm.item() ** 2
    
    # Calculate final norms
    total_norm = total_norm ** 0.5
    grad_norms = {k: v ** 0.5 for k, v in grad_norms.items()}
    grad_norms['total'] = total_norm
    
    return grad_norms
