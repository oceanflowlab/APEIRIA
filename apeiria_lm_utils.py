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

class Program3DDataset(Dataset):
    def get_annotation_file(self, split="train") -> str:
        SR3D_ANNO = {
            "train": f"{DATA_PATH}/sr3d_with_programs_train.json",
            "val": f"{DATA_PATH}/sr3d_with_programs_val.json",
        }
        NR3D_ANNO = {
            "train": f"{DATA_PATH}/nr3d_train_with_program.json",
            "val": f"{DATA_PATH}/nr3d_val_with_program.json",
        }
        SCANQA_ANNO = {
            "train": f"{DATA_PATH}/ScanQA_v1.0_train_with_program.json",
            "val": f"{DATA_PATH}/ScanQA_v1.0_val_with_program.json",
        }
        SCANREFER_ANNO = {
            "train": f"{DATA_PATH}/ScanRefer_filtered_train_with_program.json",
            "val": f"{DATA_PATH}/ScanRefer_filtered_val_with_program.json",
        }


        if "sr3d" in self.name.lower():
            result = SR3D_ANNO[split]
        elif "nr3d" in self.name.lower():
            result = NR3D_ANNO[split]
        elif "scanqa" in self.name.lower():
            result = SCANQA_ANNO[split]
        elif "scanrefer" in self.name.lower():
            result = SCANREFER_ANNO[split]
        else:
            raise ValueError(f"Unknown dataset name: {self.name}")

        if not os.path.exists(result):
            raise FileNotFoundError(f"Annotation file {result} not found, for dataset {self.get_dataset_description()}")
        
        return result

    @staticmethod
    def get_scene_path() -> str:
        # return f"{DATA_PATH}/scannet_data"
        return f"scannet/scans_fixed" # full scene without any filter (wall, ceiling, floor)

    @staticmethod
    def get_multiview_path() -> str:
        return f"{DATA_PATH}/scannet/scannet_data/enet_feats_maxpool"

    @staticmethod
    def get_frozen_object_feature_path(type: str="pnpp") -> str:
        if type == "pnpp":
            return f"{DATA_PATH}/scannetv2-pnpp-feature.pkl"
        elif type == "pnpp-vote2cap-box":
            return f"{SVC_PATH}/pc_features/scannetv2-vote2cap-feature_box_features_281d.pkl" # its box need flip!
            # return f"{SVC_PATH}/pc_features/scannetv2-vote2cap-feature-new-2_box_features_281d.pkl" # this don't
        elif type == "uni3d-mask3d-box":
            return f"{SVC_PATH}/pc_features/chatscene_features/scannet_mask3d_trainval_feat+bbox_feats.pt" # 1030d
        elif type == "uni3d-mask3d-gt":
            return f"{SVC_PATH}/pc_features/chatscene_features/scannet_gt_trainval_feat+bbox_feats.pt" # 1030d
        elif type == "pnpp-vote2cap-enc":
            return f"{DATA_PATH}/scannetv2-vote2cap-feature_enc_features_259d.pkl"
        else:
            raise ValueError(f"Unknown frozen object feature type: {type}")

    def __len__(self):
        return len(self.annotation)

    @staticmethod
    def count_program_complexity(program: str) -> int:
        """count the number of steps in the program, by counting parentheses"""
        simple_functions = ["scene", "intersection", "union", "intersect", "exclude"]
        # return program.count("(") - program.count("scene(")
        parentheses = program.count("(")
        for func in simple_functions:
            parentheses -= program.count(f"{func}(")

        return parentheses

    def get_all_scene_ids(self):
        if not hasattr(self, "scene_ids"):
            import glob

            # load all scene_id in scene_path
            scene_path = self.get_scene_path()
            filenames = glob.glob(
                os.path.join(scene_path, "*_aligned_vert.npy")
            )  # .../scene0804_00_aligned_vert.npy
            self.scene_ids = [
                os.path.basename(f)[: -len("_aligned_vert.npy")] for f in filenames
            ]
            logger.info(f"Loaded {len(self.scene_ids)} scene ids")
            return self.scene_ids
        else:
            return self.scene_ids
        
    def get_dataset_description(self):
        return f"{self.__class__.__name__}-{self.name}-{self.split}"
    
    def __init__(
        self,
        name: str,
        split: str,
        ratio: float,
        shuffle_objects: bool = False,
        start_from_last: bool = False,
        frozen_object_type: str = "pnpp-vote2cap-box",
        pc_tokenizer_type: str = "frozen",
        object_label_type: type = NYU40Object,
        max_objects: int = -1,
        seed: int = 0,
        enforce_shuffle_objects: bool = False,
        min_freq: int = 0,
        max_calls: int = 10,
        pad_objects: bool = False,
        no_build_vocab: bool = False,
        prebuilt_vocab: bool = True,
        num_distractor_centric: int = -1, # if > 0, then sample all same-target label distractor objects first, then sample from all objects rest
        **kwargs,
    ):
        self.name: str = name
        self.split: str = split
        self.annotation = self.get_annotation_file(split)
        self.start_from_last = start_from_last
        self.accessed_times = defaultdict(int)
        self.shuffle_objects = shuffle_objects
        self.frozen_object_type = frozen_object_type
        self.pc_tokenizer_type = pc_tokenizer_type
        self.object_label_type = object_label_type
        self.max_objects = max_objects
        self.enforce_shuffle_objects = enforce_shuffle_objects
        self.min_freq = min_freq    
        self.max_calls = max_calls
        self.pad_objects = pad_objects
        self.num_distractor_centric = num_distractor_centric

        logger.info(f"Loading {self.get_dataset_description()}, {ratio=}, {seed=}, {frozen_object_type=}, {pc_tokenizer_type=}, {object_label_type=}, {max_objects=}, {min_freq=}, {max_calls=}, {pad_objects=}, {no_build_vocab=}, {num_distractor_centric=}")

        self.scanrefer_id_to_hash_id = {}
        self.hash_id_to_scanrefer_id = {}

        self.seed = seed

        if isinstance(self.annotation, str):
            self.annotation = json.load(open(self.get_annotation_file(split)))
        else:
            assert isinstance(self.annotation, list) # already a (loaded from JSON) list
        if ratio < 1.0:
            self._take_partial_data(ratio)

        self._preprocess_annotation()  # preprocess annotation
        self._load()  # load scene data

        self._load_frozen_features(self.frozen_object_type)
        self._get_input_predicted_bbox()

        self._trim_too_many_objects(self.max_objects)

        # after removing some objects, we then compute the closest gt bbox
        self._compute_closest_predicted_bbox()
        self._compute_closest_gt_bbox()

        self._filter_invalid_illegal_programs()

        if not no_build_vocab:
            self._build_concept_vocabularies(prebuilt_vocab=prebuilt_vocab)


    def _build_concept_vocabularies(self, prebuilt_vocab: bool = True):
        """
        check all concepts used in programs, and build a vocabulary for them
        """
        logger.info("Building concept vocabularies...")

        if prebuilt_vocab:
            # use prebuilt vocabularies
            logger.info("Using prebuilt vocabularies...")
            self.object_classes = ScanNetRawObject().object_classes
            self.pair_classes = ['on','left', 'right', 'front', 'behind', 'above', 'below',
                'beside', 'over', 'under', 'beneath', 'underneath', 'lying', 'next',
                'back', 'top', 'supporting','with',
                'near', 'close', 'closer', 'closest', 'far', 'farthest']
            self.triplet_classes = ['between', 'center', 'middle', 'facing', 'looking', 'left', 'right', 'behind', 'back']
            
        else:
            all_programs = [data["program"] for data in self.annotation]
            result = analyze_programs_attributes(all_programs) 

            # returns like:
            # result = {
            #     'unary_attributes': Counter(...),
            #     'binary_attributes': Counter(...),
            #     'ternary_attributes': Counter(...),
            #     'query_attributes': Counter(...)
            # }
        
            # Filter by minimum frequency
            self.object_classes = [c for c, count in result["unary_attributes"].items() 
                                if count >= self.min_freq]
            self.pair_classes = [c for c, count in result["binary_attributes"].items()
                                if count >= self.min_freq]
            self.triplet_classes = [c for c, count in result["ternary_attributes"].items()
                                if count >= self.min_freq]

        # replace "_" with " "
        self.object_classes = [c.replace("_", " ") for c in self.object_classes]
        self.pair_classes = [c.replace("_", " ") for c in self.pair_classes]
        self.triplet_classes = [c.replace("_", " ") for c in self.triplet_classes]
        
        # Create mappings
        self.object_class_to_id = {c: i for i, c in enumerate(self.object_classes)}
        self.pair_class_to_id = {c: i for i, c in enumerate(self.pair_classes)}
        self.triplet_class_to_id = {c: i for i, c in enumerate(self.triplet_classes)}

        logger.info(f"Built concept vocabularies: {len(self.object_classes)} unary, {len(self.pair_classes)} binary, {len(self.triplet_classes)} ternary")

    def _set_class_maps(self, object_classes: List[str], pair_classes: List[str], triplet_classes: List[str]):
        self.object_classes = object_classes
        self.pair_classes = pair_classes
        self.triplet_classes = triplet_classes

        self.object_class_to_id = {c: i for i, c in enumerate(self.object_classes)}
        self.pair_class_to_id = {c: i for i, c in enumerate(self.pair_classes)}
        self.triplet_class_to_id = {c: i for i, c in enumerate(self.triplet_classes)}

        logger.info(f"Set concept vocabularies: {len(self.object_classes)} unary, {len(self.pair_classes)} binary, {len(self.triplet_classes)} ternary")
    
    @staticmethod
    def merge_concept_vocabularies(datasets: List["Program3DDataset"]):
        """
        Merge concept vocabularies from multiple datasets
        """
        object_classes = set()
        pair_classes = set()
        triplet_classes = set()

        for dataset in datasets:
            object_classes.update(dataset.object_classes)
            pair_classes.update(dataset.pair_classes)
            triplet_classes.update(dataset.triplet_classes)

        object_classes = sorted(object_classes)
        pair_classes = sorted(pair_classes)
        triplet_classes = sorted(triplet_classes)

        logger.info("Merged concept vocabularies:")
        logger.info(f"Object classes: {object_classes}")
        logger.info(f"Pair classes: {pair_classes}")
        logger.info(f"Triplet classes: {triplet_classes}")

        for dataset in datasets:
            dataset._set_class_maps(object_classes, pair_classes, triplet_classes)

    def _filter_invalid_illegal_programs(self):
        """
        Filter out invalid programs, by the way stats call depth of programs
        """
        if "sr3d" in self.name.lower():
            return # NOTE: we assume all programs for sr3d are valid

        logger.info("Filtering out invalid programs...")
        annotations = []
        call_depths = []
        call_count = []
        # for idx, data in enumerate(self.annotation):
        for idx, data in tqdm(enumerate(self.annotation), desc="Filtering out invalid programs", total=len(self.annotation), leave=False):
            program = data["program"]
            legal_signatures = get_default_signatures() if "qa" in self.name.lower() else get_grounding_signatures()
            tree = is_valid_legal_type_valid_program(program, signatures=legal_signatures)
            # if not is_valid_legal_type_valid_program(program, signatures=legal_signatures):
            if tree is False:
                logger.info(f"Invalid program: {program}")
                continue

            annotations.append(data)
            call_depths.append(get_call_depth(tree)[0] - 1) # -1 because scene() is not counted (as an LM invoke/forward)
            stats = analyze_function_calls(tree, exclude={'scene', 'intersection', 'union', 'intersect', 'exclude'})
            # filter by call count
            program_call_count = sum(stats.values())
            if program_call_count > self.max_calls:
                logger.info(f"Program {program} has too many calls: {program_call_count}")
                continue

            call_count.append(program_call_count)

        logger.info(f"Filtered out {len(self.annotation) - len(annotations)} invalid/illegal programs, {len(annotations)} left, filtered ratio: {len(annotations) / len(self.annotation)}")
        logger.info(f"Call depth stats: {np.mean(call_depths):.2f} mean, {np.median(call_depths)} median, {np.max(call_depths)} max")
        logger.info(f"Call count stats: {np.mean(call_count):.2f} mean, {np.median(call_count)} median, {np.max(call_count)} max")

        self.annotation = annotations

    def _trim_too_many_objects(self, max_objects: int):
        """
        修剪每个场景中的物体数量,保留最多max_objects个物体
        """
        if max_objects < 0:
            logger.info("不需要修剪物体数量")
            return
        
        logger.info(f"修剪物体数量到最多{max_objects}个...")
        for scene_id in self.scene_list:
            # 获取当前场景的物体特征
            object_feature = self.frozen_features[scene_id][0] 
            object_mask = self.frozen_features[scene_id][1]
            predicted_bbox_corners = self.frozen_features[scene_id][2] if len(self.frozen_features[scene_id]) == 3 else None
            
            num_objects = len(object_feature)
            
            if num_objects > max_objects:
                # 只保留前max_objects个物体
                self.frozen_features[scene_id][0] = object_feature[:max_objects]
                self.frozen_features[scene_id][1] = object_mask[:max_objects] 
                
                if predicted_bbox_corners is not None:
                    self.frozen_features[scene_id][2] = predicted_bbox_corners[:max_objects]

                logger.info(f"场景 {scene_id} 的物体数量从 {num_objects} 修剪到 {max_objects}")

                    
            # 同时修剪预测的边界框
            if scene_id in self.input_predicted_bboxes:
                # num_input_predicted_bboxes = len(self.input_predicted_bboxes[scene_id])
                if (num_input_predicted_bboxes := len(self.input_predicted_bboxes[scene_id])) > max_objects:
                    self.input_predicted_bboxes[scene_id] = self.input_predicted_bboxes[scene_id][:max_objects]

                    logger.info(f"场景 {scene_id} 的输入物体框从 {num_input_predicted_bboxes} 修剪到 {max_objects}")

                    
                # # 更新最近的GT边界框信息
                # # NOTE: 不需要，因为裁剪之后才计算最近的GT边界框
                # if "closest_gt_bbox" in self.scene_data[scene_id]:
                #     new_closest_gt = {}
                #     for pred_id in range(max_objects):
                #         if pred_id in self.scene_data[scene_id]["closest_gt_bbox"]:
                #             new_closest_gt[pred_id] = self.scene_data[scene_id]["closest_gt_bbox"][pred_id]
                #     self.scene_data[scene_id]["closest_gt_bbox"] = new_closest_gt

                    

    def set_pc_tokenizer_type(self, pc_tokenizer_type: str):
        self.pc_tokenizer_type = pc_tokenizer_type
        if pc_tokenizer_type == "frozen":
            self._load_frozen_features(self.frozen_object_type)

    def _take_partial_data(self, ratio: float):
        """
        Take partial data from the annotation
        """
        # self.annotation = self.annotation[: int(len(self.annotation) * ratio)]
        if self.start_from_last:
            # take from last X percent
            self.annotation = self.annotation[-int(len(self.annotation) * ratio) :]
        else:
            self.annotation = self.annotation[: int(len(self.annotation) * ratio)]

    def _preprocess_annotation(self):
        # add scanrefer_id and hash_id for each annotation
        for idx, data in enumerate(self.annotation):
            scene_id = data["scene_id"]

            scanrefer_id = f"{scene_id}|{idx}"
            hash_id = scanrefer_to_hash_id(scanrefer_id)
            data["scanrefer_id"] = scanrefer_id
            data["hash_id"] = hash_id

            self.scanrefer_id_to_hash_id[scanrefer_id] = hash_id
            self.hash_id_to_scanrefer_id[hash_id] = scanrefer_id

    def _load(self):
        """
        Load 3D scene, instance and object information
        """
        logger.info("Loading scene (ScanNet) data...")
        # add scannet data
        self.scene_list = sorted(list(set([data["scene_id"] for data in self.annotation])))
        logger.info(f"Loaded {len(self.scene_list)} scenes")
        # logger.info(self.scene_list)
        # self.scene_list = self.get_all_scene_ids()

        # load scene data
        self.scene_data = {}
        scene_path = self.get_scene_path()
        for scene_id in self.scene_list:
            self.scene_data[scene_id] = {}
            self.scene_data[scene_id]["mesh_vertices"] = np.load(
                os.path.join(scene_path, scene_id) + "_aligned_vert.npy"
            )  # axis-aligned
            self.scene_data[scene_id]["instance_labels"] = np.load(
                os.path.join(scene_path, scene_id) + "_ins_label.npy"
            )
            self.scene_data[scene_id]["semantic_labels"] = np.load(
                os.path.join(scene_path, scene_id) + "_sem_label.npy"
            )
            self.scene_data[scene_id]["instance_bboxes"] = np.load(
                os.path.join(scene_path, scene_id) + "_aligned_bbox.npy"
            )
            self.scene_data[scene_id]["raw_categories"] = json.load(
                open(os.path.join(scene_path, scene_id) + "_categories.json", "r")
            ) # -> N strings of object categories
            try:
                axis_align_matrix = json.load(open(f"{SVC_PATH}/alignments.json", "r"))[scene_id]
                axis_align_matrix = np.array(axis_align_matrix).reshape((4,4))
            except KeyError:
                axis_align_matrix = np.eye(4) # for test scenes
            self.scene_data[scene_id]["axis_align_matrix"] = axis_align_matrix

            # load raw label
            if self.object_label_type == ScanNetRawObject:
                raw_scene_info = f"data/apeiria_scannet_w_caption_gpt4o_and_corners_and_nyu_names_fixed/{scene_id}.json"
                raw_scene_info = json.load(open(raw_scene_info, "r"))["objects"]
                # a list of dict, each dict has keys: "name", "id"..., we need to map "name" to "id"
                # raw_label_info = {item["id"]: {
                #     "name": item["name"],
                #     "object_label": ScanNetRawObject().name_to_id.get(item["name"], ScanNetRawObject().dummy_class)
                # } for item in raw_label_info}
                raw_label_info = {}
                for item in raw_scene_info:
                    if item["name"] not in ScanNetRawObject().name_to_id:
                        logger.warning(f"Unknown object name: {item['name']}")

                    object_label = ScanNetRawObject().name_to_id.get(item["name"], ScanNetRawObject().dummy_class)
                    raw_label_info[item["id"]] = {
                        "name": item["name"],
                        "object_label": object_label
                    }

                self.scene_data[scene_id]["raw_label_info"] = raw_label_info


    def _get_input_predicted_bbox(self) -> Dict[str, np.ndarray]:
        """
        assumed to return a scene_id -> list of predicted bboxes mapping
        no masked predicted bboxes!
        """
        if self.frozen_object_type == "pnpp-vote2cap-box":
            # predicted_bbox_file = f"{SVC_PATH}/pc_features/scene_bbox_info_for_valtest_vote2cap_detr_latest.pkl"
            predicted_bbox_file = f"{SVC_PATH}/i2t/scene_bbox_info_for_valtest_vote2cap_detr_latest.pkl"
            logger.info(f"Loading predicted bboxes from {predicted_bbox_file}...")

            predicted_bbox = pickle.load(open(predicted_bbox_file, "rb"))
            # scene_id -> {bbox_id: {"bbox": ..., "is_valid": True/False}, ...}
            # NOTE: the predicted bboxes are already trimmed, so the index is the index among valid predicted bboxes
            for scene_id in predicted_bbox:
                # predicted_bbox[scene_id] = np.stack(
                #     [np.array(item["bbox"]) for item in predicted_bbox[scene_id] if item["is_valid"]]
                # )
                num_bboxes = len(predicted_bbox[scene_id])
                bboxes = np.zeros((num_bboxes, 6))
                for bbox_id, item in predicted_bbox[scene_id].items():
                    # print(item)
                    assert item["is_valid"]
                    if item["is_valid"]:
                        bboxes[bbox_id] = np.array(item["bbox"])

                predicted_bbox[scene_id] = bboxes

        elif self.frozen_object_type == "uni3d-mask3d-box" or self.frozen_object_type == "uni3d-mask3d-gt":
            if self.frozen_object_type == "uni3d-mask3d-box":
                predicted_bbox_file = f"{SVC_PATH}/pc_features/chatscene_features/scannet_mask3d_trainval_feat+bbox_feats.pt" # 1030d
            elif self.frozen_object_type == "uni3d-mask3d-gt":
                # predicted_bbox_file = f"{SVC_PATH}/pc_features/chatscene_features/scannet_gt_trainval_feat+bbox_feats.pt"
                predicted_bbox_file = f"{SVC_PATH}/pc_features/chatscene_features/scannet_gt_trainval_feat+bbox_feats_200obj.pt"
            else:
                raise ValueError(f"Unknown frozen object feature type: {self.frozen_object_type}")
            
            logger.info(f"Loading predicted bboxes from {predicted_bbox_file}...")
            predicted_bbox_anno = torch.load(predicted_bbox_file, map_location="cpu") # a list of dict

            # note that all of them are considered valid
            predicted_bbox = {}
            for item in predicted_bbox_anno:
                scene_id = item["scene_id"]
                bboxes = item["bbox"]
                if (object_mask := item.get("mask", None)) is not None:
                    bboxes = bboxes[object_mask.bool()[:len(bboxes)]] # sometimes bboxes are more than mask
                
                predicted_bbox[scene_id] = bboxes.numpy() # [N_bboxes, 6]


            for scene_id in self.scene_list:
                if scene_id not in predicted_bbox:
                    logger.warning(f"Scene {scene_id} has no predicted bboxes!")
                    # predicted_bbox[scene_id] = np.zeros((100, 6))
                    max_objects = self.max_objects if self.max_objects > 0 else 100
                    predicted_bbox[scene_id] = np.random.rand(max_objects, 6) # random bboxes

        else:
            raise NotImplementedError(f"Invalid frozen object type: {self.frozen_object_type} that have no predicted bboxes")
        
        self.input_predicted_bboxes = predicted_bbox
        return predicted_bbox

    def _compute_closest_predicted_bbox(self):
        """
        calculate all IoU3D between GT bbox and predicted bboxes, find the closest one for each GT bbox
        """
        # stat overlap between predicted bboxes and GT bboxes
        ious = []
        for scene_id in self.scene_list:
            pred_bboxes = self.input_predicted_bboxes[scene_id][:, :6]  # (K1, 6)
            # pred_bboxes_object_ids = self.input_predicted_bboxes[scene_id][:, -1]  # (K1,)
            pred_bboxes_object_ids = np.arange(len(pred_bboxes))  # (K1,)
            gt_bboxes = self.scene_data[scene_id]["instance_bboxes"][:, :6]  # (K2, 6)
            gt_object_ids = self.scene_data[scene_id]["instance_bboxes"][:, -1]  # (K2,)
            
            
            pred_idx_assigned, pred_iou = assign_preds_to_gts(pred_bboxes, gt_bboxes)  # (K2,), (K2,)
            for i, gt_object_id in enumerate(gt_object_ids):
                gt_object_id = int(gt_object_id)
                pred_id = int(pred_bboxes_object_ids[pred_idx_assigned[i]]) # assumably pred_idx_assigned[i], since pred_bboxes_object_ids is a range
                iou = pred_iou[i]
                ious.append(iou)
                
                # Store the closest predicted bbox and its IoU for each GT bbox
                if "closest_pred_bbox" not in self.scene_data[scene_id]:
                    self.scene_data[scene_id]["closest_pred_bbox"] = {}
                self.scene_data[scene_id]["closest_pred_bbox"][gt_object_id] = {
                    "pred_id": pred_id,
                    "iou": iou
                }

        # print a simple stat
        logger.info(f"IoU3D of predicted bboxes and GT bboxes: {np.mean(ious):.4f} mean, {np.median(ious):.4f} median, {np.max(ious):.4f} max, {np.min(ious):.4f} min")

    def _compute_closest_gt_bbox(self):
        """
        Calculate all IoU3D between predicted bboxes and GT bboxes, find the closest GT bbox for each predicted bbox
        """
        for scene_id in self.scene_list:
            pred_bboxes = self.input_predicted_bboxes[scene_id][:, :6]  # (K1, 6)
            pred_bboxes_object_ids = np.arange(len(pred_bboxes))  # (K1,)
            gt_bboxes = self.scene_data[scene_id]["instance_bboxes"][:, :6]  # (K2, 6)
            gt_object_ids = self.scene_data[scene_id]["instance_bboxes"][:, -1]  # (K2,)
            gt_object_ids_in_array = np.arange(len(gt_bboxes)) # NOTE: easy to index


            gt_idx_assigned, gt_iou = assign_gts_to_preds(pred_bboxes, gt_bboxes)  # (K1,), (K1,)
            
            if "closest_gt_bbox" not in self.scene_data[scene_id]:
                self.scene_data[scene_id]["closest_gt_bbox"] = {}
                
            for i, pred_id in enumerate(pred_bboxes_object_ids):
                pred_id = int(pred_id)
                gt_id = int(gt_object_ids[gt_idx_assigned[i]])
                gt_id_in_array = int(gt_object_ids_in_array[gt_idx_assigned[i]])
                iou = gt_iou[i]
                
                self.scene_data[scene_id]["closest_gt_bbox"][pred_id] = {
                    "gt_id": gt_id,
                    "gt_id_in_array": gt_id_in_array,
                    "iou": iou,
                }


    def _load_frozen_features(self, frozen_object_type: str):
        if not hasattr(self, "frozen_features"):
            frozen_features_path = self.get_frozen_object_feature_path(frozen_object_type)
            logger.info(f"Loading frozen features from {frozen_features_path}...")
            self.frozen_features = torch.load(frozen_features_path, map_location="cpu")
            self.frozen_features = {
                item["scene_id"]: [item["feature"], item["mask"], item["box_corners"]] if "box_corners" in item else [item["feature"], item["mask"]]
                for item in self.frozen_features
            }
            self.frozen_in_channels = next(iter(self.frozen_features.values()))[0].shape[1]

            logger.info(f"loaded {frozen_object_type} object features of in_channels {self.frozen_in_channels}")

            self._trim_frozen_features()

    def _trim_frozen_features(self):
        # Trim masked objects, and input predicted bboxes correspondingly
        logger.info("Trimming frozen features...")
        for scene_id in self.scene_list:
            # print_once("Trimming frozen features...")
            object_feature = self.frozen_features[scene_id][0]
            object_mask = self.frozen_features[scene_id][1]

            # don't trim objects that have no objects at all - it will cause error because frozen_features[scene_id][0] is empty
            if object_mask.sum() == 0:
                logger.warning(f"Empty mask for {scene_id}, setting all to True, and skipping trimming. Note: the feature shall be all-zero.")
                object_mask = torch.ones(len(object_feature), dtype=bool)

            self.frozen_features[scene_id][0] = object_feature[object_mask.bool()]
            self.frozen_features[scene_id][1] = torch.ones(len(self.frozen_features[scene_id][0]), dtype=bool)

            # FIXME: assumingly, now the feature shall align with the input predicted bboxes

    @property
    def need_shuffle_objects(self):
        return (self.shuffle_objects and self.split == "train") or self.enforce_shuffle_objects


    def __getitem__(self, idx):
        self.accessed_times[idx] += 1
        # --- get textual data and ids ---
        data_type = self.name
        split = self.split
        scene_id = self.annotation[idx]["scene_id"]
        ann_id = self.annotation[idx].get("ann_id", "-1")
        question_id = f"{scene_id}_{ann_id}"  # for scanrefer/scan2cap

        if "sr3d" in self.name.lower():
            question_id = f"{scene_id}_{ann_id}"
            _description = self.annotation[idx]["description"]
            # hash description, take first 6 characters, since there are multiple descriptions for single ann_id
            question_id += hashlib.md5(_description.encode()).hexdigest()[:6]

        raw_question_id = self.annotation[idx].get("question_id", question_id)
        raw_question_id = str(raw_question_id)

        description = None
        if "description" in self.annotation[idx]:
            description = self.annotation[idx]["description"]
        elif "question" in self.annotation[idx]:
            description = self.annotation[idx]["question"]
        else:
            raise ValueError(f"Unknown description field in {self.name}")

        program = self.annotation[idx]["program"]
        program_complexity = self.count_program_complexity(program)

        # get scan2cap corpus id
        try:
            object_name = self.annotation[idx]["object_name"]
            object_id = int(self.annotation[idx]["object_id"])
        except KeyError as e:
            try:
                object_name = self.annotation[idx]["object_names"][0]
                object_id = int(self.annotation[idx]["object_ids"][0])
            except (KeyError, IndexError) as e:
                object_name = "unknown"  # sqa3d has no object name
                object_id = 0

        object_id = int(object_id) if object_id is not None else 0
        scan2cap_id = f"{scene_id}|{object_id}|{object_name}"

        object_name = object_name.replace("_", " ")  # for instruction, replace _ with space

        scanrefer_id = self.annotation[idx].get("scanrefer_id", None)
        hash_id = self.annotation[idx].get("hash_id", None)


        # --- get scene data ---
        # TODO: load scene, and break up into object-wise point clouds
        # mesh_vertices = self.scene_data[scene_id]['mesh_vertices'].copy()
        # instance_labels = self.scene_data[scene_id]['instance_labels'].copy()
        # semantic_labels = self.scene_data[scene_id]['semantic_labels'].copy()
        # instance_bboxes = self.scene_data[scene_id]['instance_bboxes'].copy()
        # instance_raw_categories = self.scene_data[scene_id]['raw_categories'].copy()
        # # axis_align_matrix = self.scene_data[scene_id]['axis_align_matrix'].copy()

        # # break down the mesh vertices by instance labels
        # instance_mesh_vertices = {}

        # # use color
        # point_cloud = mesh_vertices[:,0:6] 
        # point_cloud[:,3:6] = (point_cloud[:,3:6]-MEAN_COLOR_RGB)/256.0

        instance_bboxes = self.scene_data[scene_id]["instance_bboxes"].copy()  # xyzhwl
        instance_box_labels = instance_bboxes[:, 6].copy()
        if self.object_label_type == ScanNetRawObject:
            # replace the label with raw label
            print_once("Replacing instance box nyu40 labels with raw labels...")
            instance_box_labels = np.zeros_like(instance_box_labels)
            for i, gt_object_id in enumerate(instance_bboxes[:, -1]):
                instance_box_labels[i] = self.scene_data[scene_id]["raw_label_info"][int(gt_object_id)]["object_label"]

        instance_box_gt_ids = instance_bboxes[:, -1].copy()

        # --- get target bbox ---
        unique_instance_ids = instance_bboxes[:, -1]
        target_id = 0
        for i, unique_instance_id in enumerate(unique_instance_ids):
            if int(unique_instance_id) == int(object_id):
                target_id = i
                break

        # --- get instruction ---
        #   |-- get target bbox text ---
        target_bbox = instance_bboxes[target_id, 0:6].copy()  # xyzhwl
        # target_corners = get_3d_box(target_bbox[3:6], 0, target_bbox[0:3])
        if object_id in self.scene_data[scene_id]["closest_pred_bbox"]:
            target_pred_id = self.scene_data[scene_id]["closest_pred_bbox"][object_id]["pred_id"] # in unshuffled indices
        else:
            target_pred_id = None

        if self.pc_tokenizer_type == "frozen":
            # self._load_frozen_features(self.frozen_object_type)
            # object_feature, object_mask = self.frozen_features[scene_id]
            object_feature = self.frozen_features[scene_id][0]
            object_mask = self.frozen_features[scene_id][1]
            if object_mask.sum() == 0:
                logger.warning(f"Empty mask for {scene_id}")
                object_mask = torch.ones_like(object_mask)

            object_feature = torch.tensor(object_feature)
            object_mask = torch.tensor(object_mask)

            predicted_bbox_corners = self.frozen_features[scene_id][2] if len(self.frozen_features[scene_id]) == 3 else None
            input_predicted_bbox = self.input_predicted_bboxes[scene_id][:, :6]  # (K1, 6)

            predicted_bbox_corners = torch.tensor(predicted_bbox_corners) if predicted_bbox_corners is not None else None
            input_predicted_bbox = torch.tensor(input_predicted_bbox)

        # get each gt box's closest predicted box, and assign label to the predicted box
        # Get labels for each predicted box based on closest GT box
        num_objects = len(object_feature)
        object_labels = np.full(num_objects, self.object_label_type().dummy_class, dtype=np.int64)
        object_ious = np.zeros(num_objects, dtype=np.float32)  # store IoUs

        object_boxes_gt = np.zeros((num_objects, 6))
        
        # Assign labels based on closest GT boxes
        closest_gt_data = self.scene_data[scene_id]["closest_gt_bbox"]
        for pred_idx in range(num_objects):
            # Get original pred_idx before shuffling
            # orig_pred_idx = pred_idx if not self.need_shuffle_objects else shuffle_indices[pred_idx] 
            orig_pred_idx = pred_idx

            if object_mask[pred_idx] == 0:
                continue

            if orig_pred_idx in closest_gt_data:
                gt_info = closest_gt_data[orig_pred_idx]
                if gt_info["iou"] < 0.25:
                    continue # ignore low IoU boxes
                
                object_labels[pred_idx] = instance_box_labels[gt_info["gt_id_in_array"]]  # assign GT object label as label
                assert instance_box_gt_ids[gt_info["gt_id_in_array"]] == gt_info["gt_id"], f"GT ID mismatch: {instance_box_gt_ids[gt_info['gt_id_in_array']]} vs {gt_info['gt_id']}"
                object_ious[pred_idx] = gt_info["iou"]
                object_boxes_gt[pred_idx] = instance_bboxes[gt_info["gt_id_in_array"], :6] # xyzhwl

            else:
                pass
                # logger.warning(f"No closest GT box for predicted box {pred_idx} in scene {scene_id}")

        if self.pc_tokenizer_type == "frozen":
            if self.num_distractor_centric > 0 and self.split == "train":
                print_once(f"Using distractor-centric sampling with {self.num_distractor_centric} distractors")
                valid_indices = torch.where(object_mask)[0]

                context_indices = [target_pred_id] # target is always in context
                # target_instance_label = instance_box_labels[target_id]
                target_instance_label = object_labels[target_pred_id]
                # for i, instance_label in enumerate(instance_box_labels):
                for i, instance_label in enumerate(object_labels):
                    if instance_label == target_instance_label and i != target_pred_id and object_mask[i]:
                        context_indices.append(i)

                # context_indices = np.random.permutation(context_indices).tolist()

                rest_indices = np.setdiff1d(valid_indices.numpy(), context_indices)
                context_indices += np.random.permutation(rest_indices).tolist()
                context_indices = context_indices[:self.num_distractor_centric]
                
                # target_pred_id = context_indices.index(target_pred_id) # shall be 0, since target is always in context
                
                # mask-off the rest
                object_centric_mask = np.zeros(len(object_feature), dtype=bool)
                object_centric_mask[context_indices] = True
                object_mask = object_mask & torch.tensor(object_centric_mask)

                # trim pc_dict
                # object_feature = object_feature[object_mask]
                # predicted_bbox_corners = predicted_bbox_corners[object_mask] if predicted_bbox_corners is not None else None
                # input_predicted_bbox = input_predicted_bbox[object_mask]
                # object_mask = object_mask[object_mask] # which is all True then
                
            

            object_mask_np = object_mask.bool().numpy() if isinstance(object_mask, torch.Tensor) else object_mask
            if object_mask_np.sum() == 0:
                logger.warning(f"Object mask is all-false in scene {scene_id}!")
                object_mask_np = np.ones(len(object_feature), dtype=bool)


            if self.need_shuffle_objects:
                # shuffle object features
                # FIXME: how to let target label also shuffle?
                # 1, 2, 3, .., => 3, 1, 2, ... (shuffle) => 3, ..., 17, ..., 8 (shuffle, valid) => 1, .., 3, ... 2 (shuffle, valid, in trimmed objects)
                generator = np.random.default_rng(seed=idx + self.accessed_times[idx] + self.seed)
                    # ensure different seed for same index at different time
                # shuffle_indices = np.random.permutation(len(object_feature))
                shuffle_indices = generator.permutation(len(object_feature))

                # shuffled_indices_trimmed = np.argsort(shuffle_indices[object_mask_np]) #　0-th goes to shuffled_indices_trimmed[0]-th, 1-th goes to shuffled_indices_trimmed[1]-th, ...
                revert_indices = np.argsort(shuffle_indices) # 0-th goes to revert_indices[0]-th, 1-th goes to revert_indices[1]-th, ...
                # ic(object_feature.shape, object_mask.shape, predicted_bbox_corners.shape)
                object_feature = object_feature[shuffle_indices] 
                    # shuffle_indices[0]-th -> 0-th, shuffle_indices[1]-th -> 1-th, ...
                    # so 0-th -> argsort(shuffle_indices)[0]-th, 1-th -> argsort(shuffle_indices)[1]-th, ...
                object_mask = object_mask[shuffle_indices]
                predicted_bbox_corners = predicted_bbox_corners[shuffle_indices]
                input_predicted_bbox = input_predicted_bbox[shuffle_indices]

                target_pred_id = revert_indices[target_pred_id].item() if target_pred_id is not None else None

                object_labels = object_labels[shuffle_indices]
                object_ious = object_ious[shuffle_indices]

            else:
                shuffle_indices = np.arange(len(object_feature)) # no shuffle, 0, 1, ..., N_obj-1
                revert_indices = np.arange(len(object_feature)) # no shuffle, 0, 1, ..., N_obj-1

            if self.pad_objects and self.max_objects > 0:
                # pad objects to max_objects
                num_objects = len(object_feature)
                if num_objects < self.max_objects:
                    # pad with zeros
                    num_pad = self.max_objects - num_objects
                    object_feature = torch.cat([object_feature, torch.zeros((num_pad, object_feature.shape[1]), dtype=object_feature.dtype)], dim=0)
                    object_mask = torch.cat([object_mask, torch.zeros(num_pad, dtype=object_mask.dtype)], dim=0)
                    predicted_bbox_corners = torch.cat([predicted_bbox_corners, torch.zeros((num_pad, *predicted_bbox_corners.shape[1:]), dtype=predicted_bbox_corners.dtype)], dim=0) if predicted_bbox_corners is not None else None
                    
                    
                    shuffle_indices = np.concatenate([shuffle_indices, np.arange(len(object_feature) - num_pad, len(object_feature))], axis=0)
                    revert_indices = np.concatenate([revert_indices, np.arange(len(object_feature) - num_pad, len(object_feature))], axis=0)

                    object_labels = np.concatenate([object_labels, np.full(num_pad, self.object_label_type().dummy_class, dtype=np.int64)], axis=0)
                    object_ious = np.concatenate([object_ious, np.zeros(num_pad, dtype=np.float32)], axis=0)

                num_pad_input_predicted_bbox = self.max_objects - len(input_predicted_bbox)
                if num_pad_input_predicted_bbox > 0:
                    input_predicted_bbox = torch.cat([input_predicted_bbox, torch.zeros((num_pad_input_predicted_bbox, *input_predicted_bbox.shape[1:]), dtype=input_predicted_bbox.dtype)], dim=0)


            pc_dict = {
                "scene_id": scene_id,
                "object_feature": object_feature,
                "object_mask": object_mask,
                "predicted_bbox_corners": predicted_bbox_corners,
                "input_predicted_bbox": input_predicted_bbox, # TODO: use corners instead of bbox?
                "object_labels": object_labels,
                "object_ious": object_ious,
            }

        
        

        # # --- get instruction ---
        # #   |-- get target bbox text ---
        # target_bbox = instance_bboxes[target_id, 0:6].copy()  # xyzhwl
        # # target_corners = get_3d_box(target_bbox[3:6], 0, target_bbox[0:3])
        # if object_id in self.scene_data[scene_id]["closest_pred_bbox"]:
        #     target_pred_id = self.scene_data[scene_id]["closest_pred_bbox"][object_id]["pred_id"] # in unshuffled indices
        #     if self.need_shuffle_objects:
        #         # ic(len(shuffle_indices), target_pred_id, object_id)
        #         target_pred_id = revert_indices[target_pred_id].item() # revert to original index
        # else:
        #     target_pred_id = None # will be assigned later
            
        # target_pred_id = self.scene_data[scene_id]["closest_pred_bbox"][object_id]["pred_id"]

        # # get each gt box's closest predicted box, and assign label to the predicted box
        # # Get labels for each predicted box based on closest GT box
        # num_objects = len(object_feature)
        # # object_labels = np.zeros(num_objects, dtype=np.int64)  # default label 0
        # # object_labels = np.full(num_objects, NYU40Object.dummy_class, dtype=np.int64)
        # object_labels = np.full(num_objects, self.object_label_type().dummy_class, dtype=np.int64)
        # object_ious = np.zeros(num_objects, dtype=np.float32)  # store IoUs

        # object_boxes_gt = np.zeros((num_objects, 6))
        
        # # Assign labels based on closest GT boxes
        # closest_gt_data = self.scene_data[scene_id]["closest_gt_bbox"]
        # for pred_idx in range(num_objects):
        #     # Get original pred_idx before shuffling
        #     orig_pred_idx = pred_idx if not self.need_shuffle_objects else shuffle_indices[pred_idx] 

        #     if object_mask[pred_idx] == 0:
        #         continue

        #     if orig_pred_idx in closest_gt_data:
        #         gt_info = closest_gt_data[orig_pred_idx]
        #         if gt_info["iou"] < 0.25:
        #             continue # ignore low IoU boxes
                
        #         object_labels[pred_idx] = instance_box_labels[gt_info["gt_id_in_array"]]  # assign GT object label as label
        #         assert instance_box_gt_ids[gt_info["gt_id_in_array"]] == gt_info["gt_id"], f"GT ID mismatch: {instance_box_gt_ids[gt_info['gt_id_in_array']]} vs {gt_info['gt_id']}"
        #         object_ious[pred_idx] = gt_info["iou"]
        #         object_boxes_gt[pred_idx] = instance_bboxes[gt_info["gt_id_in_array"], :6] # xyzhwl

        #     else:
        #         pass
        #         # logger.warning(f"No closest GT box for predicted box {pred_idx} in scene {scene_id}")

        

        # # Add labels and IoUs to pc_dict
        # pc_dict.update({
        #     "object_labels": object_labels,  # labels for each object feature
        #     "object_ious": object_ious,      # IoU scores for each object
        # })


        return {
            # 2D instruction, image, target
            "question_id": question_id,
            "raw_question_id": raw_question_id,
            "scan2cap_id": scan2cap_id,
            "scene_id": scene_id,
            "scanrefer_id": scanrefer_id,
            "hash_id": hash_id,
            "target_id": target_id, # index in GT bboxes
            "target_pred_id": target_pred_id, # index in predicted bboxes
            "object_id": object_id, # index in GT bbox index (not a continuous range)
            "data_type": data_type,
            "split": split,
            "target_bbox": target_bbox,
            "program": program,
            "program_complexity": program_complexity,
            "description": description,
            "shuffle_indices": shuffle_indices,
            "revert_indices": revert_indices,
            # 3D
            **pc_dict,
        }

    def process_instructions(self, instructions: Dict, **kwargs) -> Dict:
        """
        Process instructions, e.g. remove template, add prompt, etc.
        """
        return instructions

    def evaluate(self, preds, gt_indices, iou_threshold=0.25, hash_id_index: bool=False, use_closest_gt: bool=False) -> Tuple[str, Dict]:
        """
        preds: Map from scanrefer_id to predicted object index (index of input predicted bbox (ignored masked ones))
        gt_indices: Map from scanrefer_id to GT object index (in array, not real object id)

        returns: iou message and iou score
        """
        # correct, total = 0, 0
        correct_by_index = 0
        correct_iou = 0
        ious = []
        common_keys = set(preds.keys()) & set(gt_indices.keys())
        logger.info(f"Common keys: {len(common_keys)}")
        logger.info(f"Total predictions: {len(preds)}")
        logger.info(f"Total GT: {len(gt_indices)}")
        if len(common_keys) != len(preds) or len(common_keys) != len(gt_indices):
            logger.warning("Some keys are missing in GT or predictions!")
        for scanrefer_id in common_keys:
            # pred_caption = preds[scanrefer_id] 
            # shall be integer
            # pred_id: int = self._parse_object_index(pred_caption) 
            pred_id: int = preds[scanrefer_id]
            gt_id: int = gt_indices[scanrefer_id]
            # scene_id, _, _ = scanrefer_id.split("|")
            if hash_id_index:
                scanrefer_id = self.hash_id_to_scanrefer_id[scanrefer_id]
            
            # scene_id = scanrefer_id.split("|")[1]
            scene_id = scanrefer_id.split("|")[0]

            assert scene_id in self.scene_list, f"Invalid scene ID: {scene_id}, current scene list: {self.scene_list}"

            gt_bbox = self.scene_data[scene_id]["instance_bboxes"][gt_id, :6]

            pred_bboxes = self.input_predicted_bboxes[scene_id]
            if pred_id >= len(pred_bboxes):
                logger.warning(f"Invalid predicted bbox index: {pred_id}, but only {len(pred_bboxes)} predicted bboxes")
                pred_id = 0

            pred_bbox = self.input_predicted_bboxes[scene_id][pred_id, :6]
            if use_closest_gt:
                # replace pred_bbox with the closest GT bbox
                if pred_id in self.scene_data[scene_id]["closest_gt_bbox"]:
                    closest_gt_data = self.scene_data[scene_id]["closest_gt_bbox"][pred_id]
                    closest_gt_id = closest_gt_data["gt_id_in_array"]
                    pred_bbox = self.scene_data[scene_id]["instance_bboxes"][closest_gt_id, :6].copy()
                else:
                    pass
                    # logger.warning(f"No closest GT box for predicted box {pred_id} in scene {scene_id}")

            iou = box3d_iou_orthogonal(gt_bbox, pred_bbox)
            ious.append(iou)

            if iou >= iou_threshold:
                correct_iou += 1

            # if gt_id == pred_id:
                
        
        ious = np.array(ious)
        accuracy = correct_iou / len(common_keys)
        message = f"[Acc@{iou_threshold:.2f}] Mean: {np.mean(ious):.4f}, Max: {np.max(ious):.4f}, Min: {np.min(ious):.4f}, Acc: {correct_iou}/{len(common_keys)}={accuracy:.4f}"
        
        return message, {"accuracy": accuracy}


class Program3DDatasetFilter(Program3DDataset):
    """
    Compose examples for `filter` task - finding all objects of a given class in a scene
    """
    def get_annotation_file(self, split) -> List[str]:
        path = "data/apeiria_scannet_w_caption_gpt4o_and_corners_and_nyu_names_fixed"
        import glob
        filenames = glob.glob(f"{path}/*.json")
        return filenames
    
    def _filter_invalid_illegal_programs(self):
        pass # no need to filter

    def __init__(self, object_name_field: str="name", **kwargs):
        self.object_name_field = object_name_field

        super().__init__(**kwargs)
        
        logger.info(f"Created Filter dataset with object name field: {object_name_field}")

        self.num_distractor_centric = -1 # no distractor-centric sampling

    def _preprocess_annotation(self):
        """
        Process raw annotations to create filter program samples
        For each scene and unique object class, create a sample to find all objects of that class
        """
        # super()._preprocess_annotation()

        # load JSONs
        processed_annotations = {}
        for filename in self.annotation:
            scene_id = os.path.basename(filename).split(".")[0]
            with open(filename, "r") as f:
                data = json.load(f)
                processed_annotations[scene_id] = data["objects"] # list of objects

        self.annotation = processed_annotations

        processed_annotations = []
        # filter train/val/test scenes by id
        metadata_path = f"data/meta_data/scannetv2_{self.split}.txt"
        metadata = open(metadata_path, "r").readlines()
        split_scene_ids = [line.strip() for line in metadata]
        logger.info(f"Loaded {len(split_scene_ids)} {self.split} scenes")


        # Create filter samples for each scene and object class
        for scene_id, objects in self.annotation.items():
            if scene_id not in split_scene_ids:
                continue

            # Group by object class
            class_objects = defaultdict(list)
            for obj in objects:
                # class_objects[obj["name"]].append(obj)
                class_objects[obj[self.object_name_field]].append(obj)
                
            # Create one sample per class
            for obj_class, class_objs in class_objects.items():
                sample = {
                    "scene_id": scene_id,
                    "description": f"find all {obj_class} objects in the scene",
                    "ann_id": str(uuid.uuid4())[:6],
                    "program": f"filter(scene(), {obj_class.replace(' ', '_')})", # kitchen_table => kitchen table 
                    "object_ids": [obj["id"] for obj in class_objs],
                    "object_name": obj_class,
                    # Keep first object's ID for compatibility with parent class
                    "object_id": class_objs[0]["id"]
                }
                processed_annotations.append(sample)
                
        self.annotation = processed_annotations
        logger.info(f"Created {len(self.annotation)} filter samples")

        super()._preprocess_annotation()

    def __getitem__(self, idx):
        data = super().__getitem__(idx)

        # supplement target_pred_id with multiple object ids
        scene_id = data["scene_id"]
        object_ids = self.annotation[idx]["object_ids"]
        object_ids = [int(obj_id) for obj_id in object_ids]
        
        data["target_id"] = object_ids

        target_pred_ids = []
        for object_id in object_ids:
            if object_id in self.scene_data[scene_id]["closest_pred_bbox"]:
                target_pred_id = self.scene_data[scene_id]["closest_pred_bbox"][object_id]["pred_id"]
                if self.need_shuffle_objects:
                    target_pred_id = data["revert_indices"][target_pred_id].item() # revert to original index
                target_pred_ids.append(target_pred_id)

            # NOTE: it is ok to have some missing, since some gt boxes of certain classes (e.g., wall, ceiling, floor) are excluded.

        data["target_pred_id"] = target_pred_ids # some would be empty, since their GT boxes are excluded

        return data

    # not tested, and not used
    def evaluate(self, preds, gt_indices):
        """
        Evaluate filter predictions by calculating precision, recall, and F1 score
        
        Args:
            preds (dict): Mapping from sample ID to list of predicted object indices
            gt_indices (dict): Mapping from sample ID to list of ground truth object indices
        
        Returns:
            dict: Dictionary containing precision, recall, and F1 metrics
        """
        total_precision = 0
        total_recall = 0
        total_f1 = 0
        total_samples = 0

        common_keys = set(preds.keys()) & set(gt_indices.keys())
        logger.info(f"Common keys: {len(common_keys)}")
        logger.info(f"Total predictions: {len(preds)}")
        logger.info(f"Total GT: {len(gt_indices)}")

        for sample_id in common_keys:
                
            pred_set = set(preds[sample_id])
            gt_set = set(gt_indices[sample_id])
            
            # Handle empty predictions or ground truth
            if len(pred_set) == 0 and len(gt_set) == 0:
                precision = 1.0
                recall = 1.0
            elif len(pred_set) == 0 or len(gt_set) == 0:
                precision = 0.0
                recall = 0.0
            else:
                # Calculate true positives
                true_positives = len(pred_set.intersection(gt_set))
                
                # Calculate precision and recall
                precision = true_positives / len(pred_set)
                recall = true_positives / len(gt_set)
            
            # Calculate F1 score
            if precision + recall > 0:
                f1 = 2 * (precision * recall) / (precision + recall)
            else:
                f1 = 0.0
                
            total_precision += precision
            total_recall += recall
            total_f1 += f1
            total_samples += 1
        
        # Calculate averages
        avg_precision = total_precision / total_samples if total_samples > 0 else 0
        avg_recall = total_recall / total_samples if total_samples > 0 else 0
        avg_f1 = total_f1 / total_samples if total_samples > 0 else 0

        message = f"Precision: {avg_precision:.4f}, Recall: {avg_recall:.4f}, F1: {avg_f1:.4f}"
        
        metrics = {
            "precision": avg_precision,
            "recall": avg_recall,
            "f1": avg_f1
        }

        return message, metrics

# --- Synthetic Simple Dataset ---

class SpatialRelation(Enum):
    """Enum for spatial relations between objects"""
    LEFT = "left to"
    RIGHT = "right to"
    ABOVE = "above"
    BELOW = "below"
    IN_FRONT_OF = "in front of"
    BEHIND = "behind"
    NEAR = "near"
    FAR = "far from"
    NONE = "no relation"

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


class SyntheticObject(metaclass=Singleton):
    """Class to define object categories for synthetic data"""
    def __init__(self):
        self.object_classes = [
            "chair", "table", "sofa", "lamp", "computer", "book", "vase", "plant",
            "cup", "bottle", "clock", "painting", "shelf", "rug", "pillow", "box",
            # "television", "phone", "bowl", "plate", "fork", "knife", "spoon", "mug",
            # "pen", "pencil", "notebook", "keyboard", "mouse", "speaker", "headphones", "camera",
            # "bag", "guitar", "bicycle", "airplane model", "toy car", "doll", "ball", "teddy bear",
            "unknown"  # Add unknown class as the last one
        ] # total 41 classes?
        self.name_to_id = {name: i for i, name in enumerate(self.object_classes)}
        self.num_classes = len(self.object_classes)
        self.dummy_class = self.num_classes - 1  # Last class (unknown) is the dummy class
        
        logger.info(f"Loaded {self.num_classes} synthetic object classes")

class Synthetic3DDataset(Dataset):
    """
    Synthetic 3D dataset that generates scenes with objects and creates instructions
    to identify objects of specific classes with their IDs.
    """
    
    def __init__(
        self,
        name: str = "synthetic3d",
        split: str = "train",
        ratio: float = 1.0,
        shuffle_objects: bool = False,
        frozen_object_type: str = "synthetic",
        pc_tokenizer_type: str = "frozen",
        object_label_type: Any = SyntheticObject,
        max_objects: int = -1,
        seed: int = 42,
        num_scenes: int = 100,
        objects_per_scene: int = 10,
        room_size: float = 10.0,
        # feature_dim: int = 128,
        min_objects_per_class: int = 1,
        max_objects_per_class: int = 3,
        instruction_templates: Optional[List[str]] = None,
        response_templates: Optional[List[str]] = None,
        object_detail_templates: Optional[List[str]] = None,
        object_templates: Optional[str] = None,
        fix_template: bool = False,
        add_thinking_trace: bool = False,
        **kwargs  # Additional kwargs are ignored
    ):
        # Set random seed for reproducibility
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Store parameters
        self.name: str = name
        self.split: str = split
        self.shuffle_objects: bool = shuffle_objects
        self.frozen_object_type = frozen_object_type
        self.pc_tokenizer_type = pc_tokenizer_type
        self.object_label_type = object_label_type() if isinstance(object_label_type, type) else object_label_type
        self.max_objects: int = max_objects
        self.seed: int = seed
        self.num_scenes: int = num_scenes
        self.objects_per_scene: int = min(objects_per_scene, max_objects) if max_objects > 0 else objects_per_scene
        self.room_size: float = room_size
        # feature = one-hot class + object ID + 3D position + 3D size, so feature_dim = num_classes + 6
        self.feature_dim: int = self.object_label_type.num_classes + self.objects_per_scene + 6
        self.min_objects_per_class: int = min_objects_per_class
        self.max_objects_per_class: int = max_objects_per_class
        self.ratio: float = ratio
        self.add_thinking_trace: bool = add_thinking_trace
        
        # Add instruction templates
        self.instruction_templates = instruction_templates or [
            "Identify all {class_name}s in the scene and provide their IDs, locations, and sizes.\n",
            "Find all {class_name}s in this scene. For each one, provide its ID, position, and dimensions.\n",
            "List all {class_name}s with their IDs, coordinates, and sizes.\n",
            "Where are all the {class_name}s in this scene? Give me their IDs, positions, and dimensions.\n",
            "Locate all {class_name}s in the scene. For each one, specify its ID, location, and size.\n"
        ]

        self.require_thinking_templates = [
            "Think about the scene first. ",
            "Analyze the scene first before answering. ",
            "Consider the scene layout first. ",
            "Take a moment to understand the scene before proceeding. ",
        ]

        if fix_template:
            self.require_thinking_templates = [self.require_thinking_templates[0]]

        if add_thinking_trace:
            # in the end of instruction, add a prompt to require the model to think about the scene
            # NOTE: add as prefix, so the model will not confuse between instruction with and without thinking trace
            self.instruction_templates = [random.choice(self.require_thinking_templates) + inst for inst in self.instruction_templates]

        self.object_templates = object_templates or "These are all objects in the scene: |object_set| \n"
        self.instruction_templates = [self.object_templates + inst for inst in self.instruction_templates]

        # Add response templates
        self.response_templates = response_templates or [
            "Apeiria found {count} {class_name}(s) in the scene:\n{object_details}",
            "Roger. There are {count} {class_name}(s) in this scene:\n{object_details}",
            "Roger. The scene contains {count} {class_name}(s):\n{object_details}",
            "{count} {class_name}(s) identified:\n{object_details}",
            "Apeiria has located {count} {class_name}(s):\n{object_details}"
        ]
        
        # Object detail templates (for each individual object in the response)
        self.object_detail_templates = object_detail_templates or [
            "Object {id}: At ({x:.2f}, {y:.2f}, {z:.2f}), size: {width:.2f} x {height:.2f} x {depth:.2f}",
            "ID {id}: Position ({x:.2f}, {y:.2f}, {z:.2f}), size {width:.2f} x {height:.2f} x {depth:.2f}",
            "{id}: Coordinates ({x:.2f}, {y:.2f}, {z:.2f}), dimensions {width:.2f} x {height:.2f} x {depth:.2f}",
            "Object {id}: ({x:.2f}, {y:.2f}, {z:.2f}), {width:.2f} x {height:.2f} x {depth:.2f}"
        ]

        self.thinking_trace_template = [
            "[APEIRIA THINKS]\n"
            "Apeiria will now analyze the scene and identify the requested object.\n"
            "First, let me list all objects and their details:\n"
            "{object_details_with_class}\n"
            "Now, Apeiria need to identify all {class_name}(s) in the scene. "
            "According to the above analyzed object details, those objects are:\n"
            "Object {object_ids_with_class}\n"
            "Now, Apeiria will formulate the response based on the identified objects.\n"
            "[APEIRIA SPEAKS]\n"
        ]
        self.object_detail_with_class_templates = [
            "Object {id}: {object_name} at ({x:.2f}, {y:.2f}, {z:.2f}), size: {width:.2f} x {height:.2f} x {depth:.2f}",
            "ID {id}: {object_name}. Position ({x:.2f}, {y:.2f}, {z:.2f}), size {width:.2f} x {height:.2f} x {depth:.2f}",
            "{id}: {object_name}. Coordinates ({x:.2f}, {y:.2f}, {z:.2f}), dimensions {width:.2f} x {height:.2f} x {depth:.2f}",
            "Object {id}({object_name}): ({x:.2f}, {y:.2f}, {z:.2f}), {width:.2f} x {height:.2f} x {depth:.2f}"
        ]
        self.object_id_with_class_templates = "{id}({object_name})"

        if fix_template:
            self.instruction_templates = [self.instruction_templates[0]]
            self.response_templates = [self.response_templates[0]]
            self.object_detail_templates = [self.object_detail_templates[0]]
            self.thinking_trace_template = [self.thinking_trace_template[0]]
            self.object_detail_with_class_templates = [self.object_detail_with_class_templates[0]]

        
        # Initialize tracking variables
        self.scene_list = []
        self.scene_data = {}
        self.frozen_features = {}
        self.input_predicted_bboxes = {}
        self.annotation = []
        self.accessed_times = defaultdict(int)
        
        # Build concept vocabularies
        self.object_classes = self.object_label_type.object_classes
        self.pair_classes = ['on', 'left', 'right', 'front', 'behind', 'above', 'below', 'near', 'far']
        self.triplet_classes = ['between', 'center', 'middle']
        
        self.object_class_to_id = {c: i for i, c in enumerate(self.object_classes)}
        self.pair_class_to_id = {c: i for i, c in enumerate(self.pair_classes)}
        self.triplet_class_to_id = {c: i for i, c in enumerate(self.triplet_classes)}
        
        # Generate data
        # self._generate_data()
        self._generate_scene_layouts()
        self._generate_data_from_layouts()
        
        # Apply ratio if needed
        if ratio < 1.0:
            self._take_partial_data(ratio)

        # build scanrefer_id to annotation index mapping
        self.scanrefer_id_to_idx = {data["scanrefer_id"]: idx for idx, data in enumerate(self.annotation)}

    def _generate_scene_layouts(self):
        """Generate synthetic 3D scene layouts with objects"""
        logger.info(f"Generating {self.num_scenes} synthetic 3D scene layouts...")
        
        # Generate scenes
        for scene_idx in range(self.num_scenes):
            scene_id = f"scene{scene_idx:04d}"
            self.scene_list.append(scene_id)
            
            # Generate object classes for this scene
            # Select a subset of classes to appear in this scene
            available_classes = list(range(len(self.object_classes) - 1))  # Exclude unknown class
            num_classes_in_scene = min(len(available_classes), self.objects_per_scene // self.min_objects_per_class)
            selected_class_indices = random.sample(available_classes, num_classes_in_scene)
            
            # Determine how many objects of each class
            objects_per_class = {}
            remaining_objects = self.objects_per_scene
            
            for class_idx in selected_class_indices[:-1]:  # All but the last class
                if remaining_objects <= 0:
                    break
                count = min(random.randint(self.min_objects_per_class, self.max_objects_per_class), remaining_objects)
                objects_per_class[class_idx] = count
                remaining_objects -= count
            
            # Assign remaining objects to the last class
            if remaining_objects > 0 and selected_class_indices:
                objects_per_class[selected_class_indices[-1]] = remaining_objects
            
            # Create object instances
            object_classes = []
            for class_idx, count in objects_per_class.items():
                object_classes.extend([class_idx] * count)
            
            # Shuffle object order
            random.shuffle(object_classes)
            
            # Generate object locations and sizes
            num_objects = len(object_classes)
            locations = np.zeros((num_objects, 6))  # [x, y, z, h, w, l]
            
            # Assign random positions and sizes
            for i in range(num_objects):
                # Position (x, y, z)
                locations[i, :3] = np.random.uniform(0, self.room_size, 3)
                
                # Size (h, w, l)
                locations[i, 3:] = np.random.uniform(0.5, 2.0, 3)
            
            # Create instance bboxes with class labels
            instance_bboxes = np.zeros((num_objects, 8))  # [x, y, z, h, w, l, class_id, object_id]
            instance_bboxes[:, :6] = locations
            instance_bboxes[:, 6] = np.array(object_classes) # class IDs
            instance_bboxes[:, 7] = np.arange(num_objects)  # object IDs
            
            # Store scene data
            self.scene_data[scene_id] = {
                "instance_bboxes": instance_bboxes,
                "raw_categories": [self.object_classes[int(class_id)] for class_id in object_classes],
                "axis_align_matrix": np.eye(4),  # Identity matrix for synthetic data
                "objects_per_class": objects_per_class
            }
            
            # Generate object features
            object_features = np.zeros((num_objects, self.feature_dim))
            # object_features is a one-hot encoding of class ID, object ID, and 3D location and size
            for i, class_idx in enumerate(object_classes):
                object_features[i, class_idx] = 1.0
                object_features[i, self.object_label_type.num_classes + i] = 1.0
                object_features[i, -6:] = locations[i]

            object_mask = torch.ones(num_objects, dtype=torch.bool)  # All objects are valid
            
            # Generate bbox corners (8 corners per box)
            bbox_corners = np.zeros((num_objects, 8, 3))
            for i in range(num_objects):
                center = locations[i, :3]
                size = locations[i, 3:]
                
                # Generate 8 corners of the box
                for j, (x, y, z) in enumerate([(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1),
                                              (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)]):
                    bbox_corners[i, j] = center + (np.array([x, y, z]) - 0.5) * size
            
            # Store frozen features
            self.frozen_features[scene_id] = [
                object_features,
                object_mask,
                torch.tensor(bbox_corners, dtype=torch.float32)
            ]
            
            # Store predicted bboxes (same as ground truth for synthetic data)
            self.input_predicted_bboxes[scene_id] = locations
            
            # Add closest_gt_bbox and closest_pred_bbox mappings
            self.scene_data[scene_id]["closest_gt_bbox"] = {}
            self.scene_data[scene_id]["closest_pred_bbox"] = {}
            
            for i in range(num_objects):
                # Each predicted box maps to itself in GT
                self.scene_data[scene_id]["closest_gt_bbox"][i] = {
                    "gt_id": i,
                    "gt_id_in_array": i,
                    "iou": 1.0  # Perfect IoU with itself
                }
                
                # Each GT box maps to itself in predictions
                self.scene_data[scene_id]["closest_pred_bbox"][i] = {
                    "pred_id": i,
                    "iou": 1.0  # Perfect IoU with itself
                }

    def _generate_data_from_layouts(self):
        """Generate data (annotations, questions, expected responses) from scene layouts"""
        logger.info(f"Generating annotations from {len(self.scene_list)} scene layouts...")
        
        # Create annotations for each class in each scene
        for scene_id in self.scene_list:
            objects_per_class = self.scene_data[scene_id].get("objects_per_class", {})
            instance_bboxes = self.scene_data[scene_id]["instance_bboxes"]
            locations = instance_bboxes[:, :6]
            object_classes = instance_bboxes[:, 6].astype(int)
            
            for class_idx in objects_per_class.keys():
                class_name = self.object_classes[class_idx]
                
                # Find all objects of this class
                object_indices = [i for i, c in enumerate(object_classes) if c == class_idx]
                
                if not object_indices:
                    continue
                
                # Create a program that filters objects of this class
                program = f"filter(scene(), {class_name})"
                
                # Create a description/question
                description = random.choice(self.instruction_templates).format(class_name=class_name)
                
                # Format the object set for the instruction
                # object_set = ", ".join([f"{i}: {self.object_classes[int(object_classes[i])]}" for i in range(len(object_classes))])
                # description = description.replace("|object_set|", object_set)
                
                # Create the expected response
                object_details = []
                for i in object_indices:
                    obj_detail = random.choice(self.object_detail_templates).format(
                        id=i,
                        x=locations[i, 0],
                        y=locations[i, 1],
                        z=locations[i, 2],
                        width=locations[i, 3],
                        height=locations[i, 4],
                        depth=locations[i, 5]
                    )
                    object_details.append(obj_detail)
                
                object_details_str = "\n".join(object_details)
                
                response_template = random.choice(self.response_templates)
                
                expected_response = response_template.format(
                    count=len(object_indices),
                    class_name=class_name,
                    object_details=object_details_str
                )

                # Add thinking trace if enabled
                if self.add_thinking_trace:
                    # Create detailed object descriptions with class names for all objects
                    object_details_with_class = []
                    for i in range(len(object_classes)):
                        obj_name = self.object_classes[int(object_classes[i])]
                        obj_detail = random.choice(self.object_detail_with_class_templates).format(
                            id=i,
                            object_name=obj_name,
                            x=locations[i, 0],
                            y=locations[i, 1],
                            z=locations[i, 2],
                            width=locations[i, 3],
                            height=locations[i, 4],
                            depth=locations[i, 5]
                        )
                        object_details_with_class.append(obj_detail)
                    
                    # Create list of objects of the target class
                    object_ids_with_class = []
                    for i in object_indices:
                        obj_id_with_class = self.object_id_with_class_templates.format(
                            id=i,
                            object_name=class_name
                        )
                        object_ids_with_class.append(obj_id_with_class)
                    
                    # Format the thinking trace
                    # ic(self.thinking_trace_template)
                    thinking_trace = random.choice(self.thinking_trace_template).format(
                        object_details_with_class="\n".join(object_details_with_class),
                        class_name=class_name,
                        object_ids_with_class=", ".join(object_ids_with_class)
                    )
                    
                    # Combine thinking trace with expected response
                    expected_response = thinking_trace + expected_response
                    
                
                # Add annotation
                ann_id = len(self.annotation)
                scanrefer_id = f"{scene_id}|{ann_id}"
                hash_id = f"synthetic_{scene_id}_{ann_id}"
                
                self.annotation.append({
                    "scene_id": scene_id,
                    "ann_id": str(ann_id),
                    "description": description,
                    "program": program,
                    "object_name": class_name,
                    "object_id": object_indices[0],  # Use first object as primary target
                    "object_ids": object_indices,    # All objects of this class
                    "scanrefer_id": scanrefer_id,
                    "hash_id": hash_id,
                    "expected_response": expected_response
                })
        
        logger.info(f"Generated {len(self.annotation)} annotations across {self.num_scenes} scenes")
    
    def _generate_data(self):
        """Legacy method that combines scene layout generation and data generation"""
        self._generate_scene_layouts()
        self._generate_data_from_layouts()
    
    def _take_partial_data(self, ratio: float):
        """Take partial data from the annotation"""
        self.annotation = self.annotation[:int(len(self.annotation) * ratio)]
        logger.info(f"Taking {ratio:.2f} of data: {len(self.annotation)} annotations")
    
    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self, idx):
        self.accessed_times[idx] += 1
        
        # Get annotation data
        data = self.annotation[idx]
        scene_id = data["scene_id"]
        ann_id = data["ann_id"]
        question_id = f"{scene_id}_{ann_id}"
        raw_question_id = question_id
        scanrefer_id = data["scanrefer_id"]
        hash_id = data["hash_id"]
        description = data["description"]
        program = data["program"]
        program_complexity = 1  # Simple filter program
        
        # Get object information
        object_name = data["object_name"]
        object_id = data["object_id"]
        object_ids = data["object_ids"]
        
        # Get scene data
        instance_bboxes = self.scene_data[scene_id]["instance_bboxes"].copy()
        target_id = object_ids[0]  # Use first object as primary target
        target_bbox = instance_bboxes[target_id, 0:6].copy()
        
        # Get target predicted ID
        target_pred_id = self.scene_data[scene_id]["closest_pred_bbox"][object_id]["pred_id"]
        
        # Get object features
        object_feature = self.frozen_features[scene_id][0]
        object_mask = self.frozen_features[scene_id][1]
        predicted_bbox_corners = self.frozen_features[scene_id][2]
        input_predicted_bbox = torch.tensor(self.input_predicted_bboxes[scene_id])
        
        # Get object labels
        object_labels = instance_bboxes[:, 6].astype(np.int64)
        object_ious = np.ones(len(object_feature), dtype=np.float32)  # All 1.0 for synthetic data
        
        # Apply shuffling if needed
        if self.shuffle_objects and self.split == "train":
            generator = np.random.default_rng(seed=idx + self.accessed_times[idx] + self.seed)
            shuffle_indices = generator.permutation(len(object_feature))
            revert_indices = np.argsort(shuffle_indices)
            
            object_feature = object_feature[shuffle_indices]
            object_mask = object_mask[shuffle_indices]
            predicted_bbox_corners = predicted_bbox_corners[shuffle_indices]
            input_predicted_bbox = input_predicted_bbox[shuffle_indices]
            object_labels = object_labels[shuffle_indices]
            object_ious = object_ious[shuffle_indices]
            
            # Update target_pred_id after shuffling
            target_pred_id = revert_indices[target_pred_id].item()
        else:
            shuffle_indices = np.arange(len(object_feature))
            revert_indices = np.arange(len(object_feature))
        
        # Create PC dictionary
        pc_dict = {
            "object_feature": object_feature,
            "object_mask": object_mask,
            "predicted_bbox_corners": predicted_bbox_corners,
            "input_predicted_bbox": input_predicted_bbox,
            "object_labels": object_labels,
            "object_ious": object_ious,
        }
        
        # Create scan2cap_id
        scan2cap_id = f"{scene_id}|{object_id}|{object_name}"
        
        return {
            # 2D instruction, image, target
            "question_id": question_id,
            "raw_question_id": raw_question_id,
            "scan2cap_id": scan2cap_id,
            "scene_id": scene_id,
            "scanrefer_id": scanrefer_id,
            "hash_id": hash_id,
            "target_id": target_id,  # index in GT bboxes
            "target_pred_id": target_pred_id,  # index in predicted bboxes
            "object_id": object_id,  # index in GT bbox index
            "object_ids": object_ids,  # all objects of this class
            "data_type": self.name,
            "split": self.split,
            "target_bbox": target_bbox,
            "program": program,
            "program_complexity": program_complexity,
            "description": description,
            "expected_response": data["expected_response"],
            "shuffle_indices": shuffle_indices,
            "revert_indices": revert_indices,
            # 3D
            **pc_dict,
        }
    
    def process_instructions(self, instructions: Dict, **kwargs) -> Dict:
        """Process instructions, e.g. remove template, add prompt, etc."""
        return instructions
    
    def _parse_response(self, response: str):
        """
        Parse the model's response to extract object IDs and locations.
        
        Args:
            response: String response from the model
            
        Returns:
            List of dicts with keys: id, x, y, z, width, height, depth
        """
        parsed_objects = []

        # remove thinking trace if present, i.e., remove all contents before [APEIRIA SPEAKS]
        #  if not detected, but [APEIRIA THINKS] is detected, then the trace is truncated
        #  therefore, the response is invalid, return empty list
        if re.search(r"\[APEIRIA THINKS\]", response, flags=re.IGNORECASE) is not None:
            if re.search(r"\[APEIRIA SPEAKS\]", response, flags=re.IGNORECASE) is None:
                logger.warning("Thinking trace detected without response, skipping...")
                return parsed_objects

            else:
                # remove all contents before [APEIRIA SPEAKS]
                response = re.split(r"\[APEIRIA SPEAKS\]", response, flags=re.IGNORECASE)[-1].strip()
        
        # Regular expressions for different response formats
        patterns = [
            # Pattern 1: Object X: At (x, y, z), size: w x h x d
            r"Object\s+(\d+):\s+At\s+\(([+-]?\d+\.?\d*),\s*([+-]?\d+\.?\d*),\s*([+-]?\d+\.?\d*)\),\s*size:\s*([+-]?\d+\.?\d*)\s*x\s*([+-]?\d+\.?\d*)\s*x\s*([+-]?\d+\.?\d*)",
            
            # Pattern 2: ID X: Position (x, y, z), size w x h x d
            r"ID\s+(\d+):\s+Position\s+\(([+-]?\d+\.?\d*),\s*([+-]?\d+\.?\d*),\s*([+-]?\d+\.?\d*)\),\s*size\s+([+-]?\d+\.?\d*)\s*x\s*([+-]?\d+\.?\d*)\s*x\s*([+-]?\d+\.?\d*)",
            
            # Pattern 3: X: Coordinates (x, y, z), dimensions w x h x d
            r"(\d+):\s+(?:Coordinates\s+)?\(([+-]?\d+\.?\d*),\s*([+-]?\d+\.?\d*),\s*([+-]?\d+\.?\d*)\),\s*(?:dimensions|size)?\s*([+-]?\d+\.?\d*)\s*x\s*([+-]?\d+\.?\d*)\s*x\s*([+-]?\d+\.?\d*)",
            
            # Pattern 4: Object X: (x, y, z), w x h x d
            r"(?:Object\s+)?(\d+):\s+\(([+-]?\d+\.?\d*),\s*([+-]?\d+\.?\d*),\s*([+-]?\d+\.?\d*)\),\s*([+-]?\d+\.?\d*)\s*x\s*([+-]?\d+\.?\d*)\s*x\s*([+-]?\d+\.?\d*)"
        ]
        
        # Try each pattern
        for pattern in patterns:
            # let it be case-insensitive
            matches = re.finditer(pattern, response, re.IGNORECASE)
            for match in matches:
                try:
                    obj_id = int(match.group(1))
                    x = float(match.group(2))
                    y = float(match.group(3))
                    z = float(match.group(4))
                    width = float(match.group(5))
                    height = float(match.group(6))
                    depth = float(match.group(7))
                    
                    parsed_objects.append({
                        "id": obj_id,
                        "x": x, "y": y, "z": z,
                        "width": width, "height": height, "depth": depth
                    })
                except (ValueError, IndexError):
                    continue
        
        return parsed_objects
    
    def evaluate(self, preds, gt_indices, iou_threshold=0.25, hash_id_index: bool=False, use_closest_gt: bool=False) -> Tuple[str, Dict]:
        """
        Evaluate predictions against ground truth.
        
        Args:
            preds: Dict mapping scanrefer_id to predicted object indices or response text
            gt_indices: Dict mapping scanrefer_id to GT object indices
            iou_threshold: IoU threshold for considering a prediction correct
            hash_id_index: Whether keys in preds are hash_ids instead of scanrefer_ids
            use_closest_gt: Whether to use closest GT bbox for evaluation
            
        Returns:
            message: Evaluation message
            metrics: Dictionary of evaluation metrics
        """
        correct_iou = 0
        correct_class = 0
        correct_ids = 0
        total = 0
        ious = []
        
        # Metrics for precision and recall
        total_pred_boxes = 0
        total_gt_boxes = 0
        true_positives = 0  # Predicted boxes that match a GT box
        detected_gt_boxes = 0  # GT boxes that are detected by any predicted box
        
        # Process predictions to extract object IDs
        processed_preds = {}
        for key, pred in preds.items():
            if isinstance(pred, (int, np.integer)):
                # Already an object index
                processed_preds[key] = [pred]
            else:
                # Parse response text to extract object IDs
                parsed_objects = self._parse_response(pred)
                processed_preds[key] = parsed_objects
        
        # Evaluate predictions
        common_keys = set(processed_preds.keys()) & set(gt_indices.keys())
        logger.info(f"Common keys: {len(common_keys)}")
        logger.info(f"Total predictions: {len(processed_preds)}")
        logger.info(f"Total GT: {len(gt_indices)}")
        
        for key in common_keys:
            scanrefer_id = key
            if hash_id_index:
                # Convert hash_id to scanrefer_id if needed
                for ann in self.annotation:
                    if ann["hash_id"] == key:
                        scanrefer_id = ann["scanrefer_id"]
                        break
            
            scene_id = scanrefer_id.split("|")[0]
            ann_id = int(scanrefer_id.split("|")[1])
            
            # Get annotation data
            annotation = None
            if scanrefer_id in self.scanrefer_id_to_idx:
                annotation = self.annotation[self.scanrefer_id_to_idx[scanrefer_id]]
            
            if annotation is None:
                logger.warning(f"Annotation not found for {scanrefer_id}")
                continue
            
            # Get predicted and ground truth object IDs
            pred_ids = [obj["id"] for obj in processed_preds[key]]
            gt_ids = annotation["object_ids"]
            
            # Count total boxes for precision/recall
            total_pred_boxes += len(pred_ids)
            total_gt_boxes += len(gt_ids)
            
            # Check if at least one predicted ID matches a ground truth ID
            has_match = any(pred_id in gt_ids for pred_id in pred_ids)
            
            if has_match:
                correct_ids += 1
            
            # Evaluate IoU for all predicted boxes against all GT boxes
            # pred_boxes = []
            pred_boxes = [
                [obj["x"], obj["y"], obj["z"], obj["width"], obj["height"], obj["depth"]]
                for obj in processed_preds[key]
            ]
            pred_boxes = np.array(pred_boxes)
            # for pred_id in pred_ids:
            #     if pred_id >= len(self.input_predicted_bboxes[scene_id]):
            #         logger.warning(f"Invalid predicted bbox index: {pred_id}, skipping")
            #         continue
            #     pred_boxes.append(self.input_predicted_bboxes[scene_id][pred_id, :6])
            
            gt_boxes = []
            for gt_id in gt_ids:
                gt_boxes.append(self.scene_data[scene_id]["instance_bboxes"][gt_id, :6])
            
            # Calculate IoU matrix between all pred_boxes and gt_boxes
            iou_matrix = np.zeros((len(pred_boxes), len(gt_boxes)))
            for i, pred_box in enumerate(pred_boxes):
                for j, gt_box in enumerate(gt_boxes):
                    iou_matrix[i, j] = box3d_iou_orthogonal(pred_box, gt_box)
            
            # For precision: check if each predicted box has IoU > threshold with any GT box
            pred_matches = (iou_matrix.max(axis=1) > iou_threshold).sum() if len(gt_boxes) > 0 else 0
            true_positives += pred_matches
            
            # For recall: check if each GT box has IoU > threshold with any predicted box
            gt_matches = (iou_matrix.max(axis=0) > iou_threshold).sum() if len(pred_boxes) > 0 else 0
            detected_gt_boxes += gt_matches
            
            # Calculate mean IoU for this sample
            if len(iou_matrix) > 0:
                # For each predicted box, find its max IoU with any GT box
                max_ious = iou_matrix.max(axis=1) if len(gt_boxes) > 0 else np.zeros(len(pred_boxes))
                ious.extend(max_ious.tolist())
            
            
            total += 1
        
        # Calculate metrics
        ious = np.array(ious)
        # correct_iou = (ious > iou_threshold).sum()
        # iou_accuracy = correct_iou / total if total > 0 else 0
        # class_accuracy = correct_class / total if total > 0 else 0
        id_accuracy = correct_ids / total if total > 0 else 0
        
        # Calculate precision, recall, and F1
        precision = true_positives / total_pred_boxes if total_pred_boxes > 0 else 0
        recall = detected_gt_boxes / total_gt_boxes if total_gt_boxes > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        mean_iou = np.mean(ious) if len(ious) > 0 else 0
        
        # Create evaluation message
        message = (
            "Object Identification Task:\n"
            f"[Acc@{iou_threshold:.2f}] "
            f"Mean IoU: {mean_iou:.4f}, "
            # f"Class: {class_accuracy:.4f}, "
            f"ID: {id_accuracy:.4f}, "
            f"Precision: {precision:.4f}, "
            f"Recall: {recall:.4f}, "
            f"F1: {f1_score:.4f}, "
            f"Total: {total}"
        )
        
        metrics = {
            # "iou_accuracy": iou_accuracy,
            # "class_accuracy": class_accuracy,
            "id_accuracy": id_accuracy,
            "mean_iou": mean_iou.item() if isinstance(mean_iou, np.ndarray) else mean_iou,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "total_pred_boxes": total_pred_boxes,
            "total_gt_boxes": total_gt_boxes,
            "true_positives": true_positives,
            "detected_gt_boxes": detected_gt_boxes,
            "total": total,
        }
        
        return message, metrics
    
    @staticmethod
    def count_program_complexity(program: str) -> int:
        """Count the number of steps in the program, by counting parentheses"""
        simple_functions = ["scene", "intersection", "union", "intersect", "exclude"]
        parentheses = program.count("(")
        for func in simple_functions:
            parentheses -= program.count(f"{func}(")
        return parentheses
    
    def get_all_scene_ids(self):
        """Get all scene IDs in the dataset"""
        return self.scene_list
    
    def get_dataset_description(self):
        """Get a description of the dataset"""
        return f"{self.__class__.__name__}-{self.name}-{self.split}"
    
class Synthetic3DObjectInfoDataset(Synthetic3DDataset):
    """
    A simplified version of the Synthetic3DDataset that focuses on a single task:
    For each object in each scene, output its ID, class, location, and size.
    """
    
    def __init__(
        self,
        name: str = "synthetic3d_object_info",
        **kwargs
    ):
        # Override prompt templates
        kwargs["instruction_templates"] = kwargs.get("instruction_templates", None) or [
            "Describe the object with ID {object_id} in the scene. Provide its class, position, and dimensions.\n",
            "For object {object_id} in the scene, specify its class, location, and size.\n",
            "Tell me about object {object_id} in the scene. What is its class, position, and dimensions?\n",
            "Provide details for object {object_id} in the scene. Include category, position, and size.\n",
            "What can you tell me about object {object_id} in the scene? Its category, location, and dimensions, please.\n"
        ]

        # kwargs["object_templates"] = kwargs.get("object_templates", None) or "This is the object in the scene: |object_set| \n" # shall we use a unified one?
        
        # Override response templates to work with object_detail_templates
        kwargs["response_templates"] = kwargs.get("response_templates", None) or [
            "Roger. The object is {object_class}. {object_details}",
            "Apeiria found {object_class}: {object_details}",
            "Object {object_class} details: {object_details}",
            "Here are the details for {object_class}: {object_details}",
            "Apeiria has analyzed the {object_class}: {object_details}"
        ]

        kwargs["add_thinking_trace"] = False # for single object, no need to add thinking trace

        # Override object detail templates, add class name
        # kwargs["object_detail_templates"] = kwargs.get("object_detail_templates", None) or [
        #     "Object ID {id}:  located at ({x:.2f}, {y:.2f}, {z:.2f}), dimensions: {width:.2f} x {height:.2f} x {depth:.2f}",
        #     "ID {id}: {object_class}, position ({x:.2f}, {y:.2f}, {z:.2f}), size {width:.2f} x {height:.2f} x {depth:.2f}",
        #     "{id}: {object_class}, coordinates ({x:.2f}, {y:.2f}, {z:.2f}), dimensions {width:.2f} x {height:.2f} x {depth:.2f}",
        #     "Object {id} ({object_class}): ({x:.2f}, {y:.2f}, {z:.2f}), {width:.2f} x {height:.2f} x {depth:.2f}"
        # ]

        # Initialize with parent class but override name
        super().__init__(name=name, **kwargs)

    def _generate_data_from_layouts(self):
        """
        Generate data for the object information task.
        For each object in each scene, create a single annotation that asks about that object.
        """
        logger.info(f"Generating object info annotations from {len(self.scene_list)} scene layouts...")
        
        # Create one annotation per object in each scene
        for scene_id in self.scene_list:
            instance_bboxes = self.scene_data[scene_id]["instance_bboxes"]
            locations = instance_bboxes[:, :6]
            object_classes = instance_bboxes[:, 6].astype(int)
            
            # For each object in the scene
            for obj_idx in range(len(object_classes)):
                class_name = self.object_classes[int(object_classes[obj_idx])]
                
                # Create a description/question asking about this specific object
                description = random.choice(self.instruction_templates).format(object_id=obj_idx)
                
                # Create the object detail using the existing templates
                obj_detail = random.choice(self.object_detail_templates).format(
                    id=obj_idx,
                    object_class=class_name,
                    x=locations[obj_idx, 0],
                    y=locations[obj_idx, 1],
                    z=locations[obj_idx, 2],
                    width=locations[obj_idx, 3],
                    height=locations[obj_idx, 4],
                    depth=locations[obj_idx, 5]
                )
                
                # Create the expected response
                expected_response = random.choice(self.response_templates).format(
                    object_id=obj_idx,
                    object_class=class_name,
                    object_details=obj_detail
                )
                
                # Add annotation
                ann_id = len(self.annotation)
                scanrefer_id = f"{scene_id}|{ann_id}"
                hash_id = f"synthetic_{scene_id}_{ann_id}"
                
                self.annotation.append({
                    "scene_id": scene_id,
                    "ann_id": str(ann_id),
                    "description": description,
                    "program": f"get_object(scene(), {obj_idx})",  # Simple program to get this object
                    "object_name": class_name,
                    "object_id": obj_idx,  # This specific object
                    "object_ids": [obj_idx],  # Just this one object
                    "scanrefer_id": scanrefer_id,
                    "hash_id": hash_id,
                    "expected_response": expected_response
                })
        
        logger.info(f"Generated {len(self.annotation)} object info annotations across {self.num_scenes} scenes")
    
    def __getitem__(self, idx):
        """
        Override the __getitem__ method to focus on a single object.
        """
        self.accessed_times[idx] += 1
        
        # Get annotation data
        data = self.annotation[idx]
        scene_id = data["scene_id"]
        ann_id = data["ann_id"]
        question_id = f"{scene_id}_{ann_id}"
        raw_question_id = question_id
        scanrefer_id = data["scanrefer_id"]
        hash_id = data["hash_id"]
        description = data["description"]
        program = data["program"]
        program_complexity = 1  # Simple program
        
        # Get object information - this is for a single object
        object_name = data["object_name"]
        object_id = data["object_id"]
        object_ids = data["object_ids"]  # Should be a list with just one ID
        
        # Get scene data
        instance_bboxes = self.scene_data[scene_id]["instance_bboxes"].copy()
        target_id = object_id
        target_bbox = instance_bboxes[target_id, 0:6].copy()
        
        # Get target predicted ID (same as object_id for synthetic data)
        target_pred_id = object_id
        
        # Get object features - but only for this specific object
        all_object_features = self.frozen_features[scene_id][0]
        all_object_mask = self.frozen_features[scene_id][1]
        all_predicted_bbox_corners = self.frozen_features[scene_id][2]
        all_input_predicted_bbox = torch.tensor(self.input_predicted_bboxes[scene_id])
        all_object_labels = instance_bboxes[:, 6].astype(np.int64)
        
        # Create a mask that only includes this object
        single_object_mask = torch.zeros_like(all_object_mask)
        single_object_mask[object_id] = 1
        
        # Create PC dictionary with just this object
        pc_dict = {
            "object_feature": all_object_features,  # Keep all features but use mask
            "object_mask": single_object_mask,      # Only this object is visible
            "predicted_bbox_corners": all_predicted_bbox_corners,
            "input_predicted_bbox": all_input_predicted_bbox,
            "object_labels": all_object_labels,
            "object_ious": np.ones(len(all_object_features), dtype=np.float32),  # All 1.0 for synthetic data
            "target_object_id": object_id,  # Add the target object ID explicitly
        }
        
        # Create scan2cap_id
        scan2cap_id = f"{scene_id}|{object_id}|{object_name}"
        
        return {
            # 2D instruction, image, target
            "question_id": question_id,
            "raw_question_id": raw_question_id,
            "scan2cap_id": scan2cap_id,
            "scene_id": scene_id,
            "scanrefer_id": scanrefer_id,
            "hash_id": hash_id,
            "target_id": target_id,  # index in GT bboxes
            "target_pred_id": target_pred_id,  # index in predicted bboxes
            "object_id": object_id,  # index in GT bbox index
            "object_ids": object_ids,  # just this one object
            "data_type": self.name,
            "split": self.split,
            "target_bbox": target_bbox,
            "program": program,
            "program_complexity": program_complexity,
            "description": description,
            "expected_response": data["expected_response"],
            "shuffle_indices": np.arange(len(all_object_features)),  # No shuffling needed
            "revert_indices": np.arange(len(all_object_features)),
            # 3D
            **pc_dict,
        }
    
    def evaluate(self, preds, gt_indices, iou_threshold=0.25, hash_id_index: bool=False, use_closest_gt: bool=False) -> Tuple[str, Dict]:
        """
        Evaluate predictions for the single object information task.
        
        Args:
            preds: Dict mapping scanrefer_id to predicted response text
            gt_indices: Dict mapping scanrefer_id to GT object indices (single object per sample)
            iou_threshold: IoU threshold for considering a prediction correct
            hash_id_index: Whether keys in preds are hash_ids instead of scanrefer_ids
            use_closest_gt: Whether to use closest GT bbox for evaluation
            
        Returns:
            message: Evaluation message
            metrics: Dictionary of evaluation metrics
        """
        total_objects = 0
        correctly_identified_objects = 0
        correctly_classified_objects = 0
        position_errors = []
        dimension_errors = []
        ious = []

        # Process predictions to extract object information
        for key, pred in preds.items():
            scanrefer_id = key
            if hash_id_index:
                # Convert hash_id to scanrefer_id if needed
                for ann in self.annotation:
                    if ann["hash_id"] == key:
                        scanrefer_id = ann["scanrefer_id"]
                        break
            
            # Find the corresponding annotation
            annotation = None
            if scanrefer_id in self.scanrefer_id_to_idx:
                annotation = self.annotation[self.scanrefer_id_to_idx[scanrefer_id]]
            
            if annotation is None:
                logger.warning(f"Annotation not found for {scanrefer_id}")
                continue
            
            scene_id = annotation["scene_id"]
            object_id = annotation["object_id"]
            
            # Get ground truth data for this object
            instance_bboxes = self.scene_data[scene_id]["instance_bboxes"]
            gt_class = self.object_classes[int(instance_bboxes[object_id, 6])]
            gt_position = instance_bboxes[object_id, :3]
            gt_dimensions = instance_bboxes[object_id, 3:6]
            
            # Parse the response to extract object information
            parsed_objects = self._parse_response(pred)
            # logger.info(f"Pred: {pred}, Parsed: {parsed_objects}")
            
            total_objects += 1
            
            # Check if the object was correctly identified
            for obj in parsed_objects:
                if obj["id"] == object_id:
                    correctly_identified_objects += 1
                    
                    # Check if class is mentioned in the response
                    if gt_class.lower() in pred.lower():
                        correctly_classified_objects += 1
                    
                    # Calculate position and dimension errors
                    pred_position = np.array([obj["x"], obj["y"], obj["z"]])
                    pred_dimensions = np.array([obj["width"], obj["height"], obj["depth"]])
                    
                    position_error = np.linalg.norm(pred_position - gt_position)
                    dimension_error = np.linalg.norm(pred_dimensions - gt_dimensions)
                    
                    position_errors.append(position_error)
                    dimension_errors.append(dimension_error)

                    # Calculate IoU with GT bbox
                    gt_bbox = instance_bboxes[object_id, :6]
                    pred_bbox = np.array([obj["x"], obj["y"], obj["z"], obj["width"], obj["height"], obj["depth"]])
                    iou = box3d_iou_orthogonal(pred_bbox, gt_bbox)
                    ious.append(iou)
                    
                    break
        
        # Calculate metrics
        #   handle numpy float to Python float to let visualization clean
        identification_accuracy = correctly_identified_objects / total_objects if total_objects > 0 else 0
        classification_accuracy = correctly_classified_objects / total_objects if total_objects > 0 else 0
        
        mean_position_error = np.mean(position_errors).tolist() if position_errors else float('inf')
        mean_dimension_error = np.mean(dimension_errors).tolist() if dimension_errors else float('inf')

        ious = np.array(ious)
        mean_iou = np.mean(ious).tolist() if len(ious) > 0 else 0
        iou_accuracy = ((ious > iou_threshold).sum() / total_objects).tolist() if total_objects > 0 else 0
        
        # Create evaluation message
        message = (
            f"Single Object Info Task:\n"
            f"ID Acc: {identification_accuracy:.4f}, "
            f"Classification Acc: {classification_accuracy:.4f}, "
            f"Mean Position Error: {mean_position_error:.4f}, "
            f"Mean Dimension Error: {mean_dimension_error:.4f}, "
            f"Mean IoU: {mean_iou:.4f}, "
            f"IoU Acc@{iou_threshold:.2f}: {iou_accuracy:.4f}, "
            f"Total Objects: {total_objects}"
        )
        
        metrics = {
            "identification_accuracy": identification_accuracy,
            "classification_accuracy": classification_accuracy,
            "mean_position_error": mean_position_error,
            "mean_dimension_error": mean_dimension_error,
            "mean_iou": mean_iou,
            "iou_accuracy": iou_accuracy,
            "total_objects": total_objects,
            "correctly_identified_objects": correctly_identified_objects,
            "correctly_classified_objects": correctly_classified_objects,
        }
        
        return message, metrics

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

class Synthetic3DRelationalDataset(Synthetic3DDataset):
    """
    Extension of Synthetic3DDataset that focuses specifically on spatial relation queries.
    For example: "Find the chair to the left of the table" type tasks.
    """
    
    def __init__(
        self,
        name: str = "synthetic3d_relational",
        adjust_scene_layouts: bool = False,
        relational_data_ratio: float = 0.1,
        add_full_thinking_trace_for_relational: bool = False,
        add_partial_full_thinking_trace_for_relational: bool = False,
        **kwargs
    ):
        self.adjust_scene_layouts = adjust_scene_layouts
        self.add_full_thinking_trace_for_relational = add_full_thinking_trace_for_relational
        self.add_partial_full_thinking_trace_for_relational = add_partial_full_thinking_trace_for_relational
        assert not (self.add_full_thinking_trace_for_relational and self.add_partial_full_thinking_trace_for_relational), "Cannot add both full and partial thinking trace"
        
        # Relation query templates
        kwargs["instruction_templates"] = kwargs.get("instruction_templates", None) or [
            "Find the {target_class} {relation} the {reference_class}. Provide its ID, position, and dimensions.\n",
            "Identify the {target_class} that is {relation} the {reference_class}. Give its ID, location, and size.\n",
            "Which {target_class} is {relation} the {reference_class}? Provide its ID, coordinates, and dimensions.\n",
            "Locate the {target_class} {relation} the {reference_class}. Specify its ID, position, and size.\n",
            "Find all {target_class}s that are {relation} the {reference_class}. For each one, provide its ID, position, and dimensions.\n"
        ]
        
        # Relation response templates
        kwargs["response_templates"] = kwargs.get("response_templates", None) or [
            "Apeiria found the {target_class} {relation} the {reference_class}:\n{object_details}",
            "Roger. The {target_class} {relation} the {reference_class} is:\n{object_details}",
            "Apeiria has located the {target_class} {relation} the {reference_class}:\n{object_details}",
            "The {target_class} with ID {object_id} is {relation} the {reference_class}:\n{object_details}",
            "Found {count} {target_class}(s) {relation} the {reference_class}:\n{object_details}"
        ]
        
        self.relational_thinking_trace_template = [
            "[APEIRIA THINKS]\n"
            "Apeiria need to find the {target_class} {relation} the {reference_class}.\n"
            "Looking at the scene, I can see {reference_class}(s):\n{reference_object_details}\n"
            "Then, I notice the {target_class}(s):\n{target_object_details}\n"
            "Now, Apeiria will analyze if any {target_class} is {relation} the {reference_class}:\n"
            "{spatial_analysis}\n"
            "From analysis above, the {target_class} (ID {object_id}) is {relation} a {reference_class} (ID {reference_id}).\n"
            "Now, Apeiria will formulate the response based on the identified objects.\n"
            "[APEIRIA SPEAKS]\n"
        ]

        # in full thinking trace, we will include all object's details
        self.relational_thinking_trace_template_full = [
            "[APEIRIA THINKS]\n"
            "Apeiria need to find the {target_class} {relation} the {reference_class}.\n"
            "First, let me list all objects and their details:\n"
            "{object_details_with_class}\n"
            "Among these objects, Apeiria can see the {target_class}(s):\n"
            "{target_object_details}\n"
            "And the {reference_class}(s):\n"
            "{reference_object_details}\n"
            "Now, Apeiria will analyze if any {target_class} is {relation} the {reference_class}:\n"
            "{spatial_analysis}\n"
            "From analysis above, the {target_class} (ID {object_id}) is {relation} a {reference_class} (ID {reference_id}).\n"
            "Now, Apeiria will formulate the response based on the identified objects.\n"
            "[APEIRIA SPEAKS]\n"
        ]

        # in full thinking trace (partial), the we only list the object IDs and names for all objects
        self.relational_thinking_trace_template_partial_full = [
            "[APEIRIA THINKS]\n"
            "Apeiria need to find the {target_class} {relation} the {reference_class}.\n"
            "First, let me list the object IDs and names:\n"
            "Object {object_ids_with_class}\n"
            "Among these objects, Apeiria can see the {target_class}(s):\n"
            "{target_object_details}\n"
            "And the {reference_class}(s):\n"
            "{reference_object_details}\n"
            "Now, Apeiria will analyze if any {target_class} is {relation} the {reference_class}:\n"
            "{spatial_analysis}\n"
            "From analysis above, the {target_class} (ID {object_id}) is {relation} a {reference_class} (ID {reference_id}).\n"
            "Now, Apeiria will formulate the response based on the identified objects.\n"
            "[APEIRIA SPEAKS]\n"
        ]


        self.spatial_analysis_template = [
            "The {target_class} (ID {target_id}) is {relations} the {reference_class} (ID {reference_id}).",
            "Object {target_id} ({target_class}) is {relations} Object {reference_id} ({reference_class}).",
        ]

        if kwargs.get("fix_template", False):
            # use the first template for all
            self.spatial_analysis_template = [self.spatial_analysis_template[0]]

        
        # Initialize with parent class
        super().__init__(name=name, **kwargs) 
        self._take_partial_data(relational_data_ratio * kwargs.get("ratio", 1.0))
        
    
    def _generate_scene_layouts(self):
        """
        Override scene layout generation to ensure objects have unique spatial relations.
        """
        logger.info(f"Generating {self.num_scenes} synthetic 3D scene layouts with spatial relations...")
        
        # First generate scenes using parent method
        super()._generate_scene_layouts()
        
        # Now enhance each scene to ensure unique spatial relations between objects
        for scene_id in self.scene_list:
            instance_bboxes = self.scene_data[scene_id]["instance_bboxes"]
            locations = instance_bboxes[:, :6].copy()
            object_classes = instance_bboxes[:, 6].astype(int)
            
            # Track relations that already exist between classes
            class_relations = {}  # (class_a, class_b) -> set of relations
            
            # For each pair of objects, ensure they have unique spatial relations
            # NOTE: this does not guarantee all pairs have unique relations
            #       since after moving one object, another pair may have original relations lost or have new relations
            # NOTE: this will almost ensure N^2 relations, maybe too many, and set adjust_scene_layouts to False to disable
            max_attempts = 50 if self.adjust_scene_layouts else 0
            for i in range(len(object_classes)):
                for j in range(len(object_classes)):
                    if i == j:
                        continue
                    
                    class_i = object_classes[i]
                    class_j = object_classes[j]
                    class_pair = (class_i, class_j)
                    
                    if class_pair not in class_relations:
                        class_relations[class_pair] = set()
                    
                    # Get current relations
                    current_relations = set(r.value for r in get_spatial_relation(locations[i], locations[j]))
                    
                    # Check if there are any new relations
                    new_relations = set(current_relations) - class_relations[class_pair]
                    
                    # If no new relations, try adjusting positions to create one
                    if not new_relations:
                        for attempt in range(max_attempts):
                            # Slightly adjust object i's position
                            temp_location = locations[i].copy()
                            temp_location[:3] += np.random.uniform(-0.5, 0.5, 3)
                            
                            # Ensure within room bounds
                            temp_location[:3] = np.clip(temp_location[:3], 0, self.room_size)
                            
                            # Check relations with new position
                            test_relations = set(r.value for r in get_spatial_relation(temp_location, locations[j]))
                            new_relations = test_relations - class_relations[class_pair]
                            
                            if new_relations:
                                locations[i] = temp_location
                                break
                    
                    # Update existing relation sets
                    class_relations[class_pair].update(current_relations)
            
            # Update instance_bboxes with new positions
            self.scene_data[scene_id]["instance_bboxes"][:, :6] = locations
            
            # Update features and input predicted bboxes, they are ndarray
            object_features = self.frozen_features[scene_id][0].copy()
            object_features[:, -6:] = locations
            
            # Update bbox corners
            bbox_corners = np.zeros((len(object_classes), 8, 3))
            for i in range(len(object_classes)):
                center = locations[i, :3]
                size = locations[i, 3:]
                
                # Generate 8 corners of the box
                for j, (x, y, z) in enumerate([(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1),
                                            (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)]):
                    bbox_corners[i, j] = center + (np.array([x, y, z]) - 0.5) * size
            
            # Update frozen_features
            self.frozen_features[scene_id] = [
                object_features,
                self.frozen_features[scene_id][1],  # keep object_mask unchanged
                torch.tensor(bbox_corners, dtype=torch.float32)
            ]
            
            # Update input_predicted_bboxes
            self.input_predicted_bboxes[scene_id] = locations
    
    def _generate_data_from_layouts(self):
        """
        Generate annotations for relational queries.
        """
        logger.info("Generating relational query annotations...")
        
        # Generate relational queries for each scene
        for scene_id in self.scene_list:
            instance_bboxes = self.scene_data[scene_id]["instance_bboxes"]
            locations = instance_bboxes[:, :6]
            object_classes = instance_bboxes[:, 6].astype(int)

            all_relations = defaultdict(set) # (class_of_relation) -> set of (i, j) pairs
            for target_object_index in range(len(object_classes)):
                for reference_object_index in range(len(object_classes)):
                    if target_object_index == reference_object_index:
                        continue
                    relations = get_spatial_relation(locations[target_object_index], locations[reference_object_index])
                    for relation in relations:
                        all_relations[relation].add((target_object_index, reference_object_index))

            class_to_objects = defaultdict(set) # record the objects of each class
            for i, obj_class in enumerate(object_classes):
                class_to_objects[obj_class].add(i)
            
            # Find valid relations between objects
            for target_object_index in range(len(object_classes)):
                target_class = self.object_classes[int(object_classes[target_object_index])]
                
                # Find objects that have unique relations with this object
                for reference_object_index in range(len(object_classes)):
                    if target_object_index == reference_object_index:
                        continue
                    
                    reference_class = self.object_classes[int(object_classes[reference_object_index])]
                    
                    # Get spatial relations
                    relations = get_spatial_relation(locations[target_object_index], locations[reference_object_index])
                    
                    for relation in relations:
                        # Check if this relation uniquely identifies the target object
                        # among objects of the same class
                        is_unique = True
                        for k in class_to_objects[object_classes[target_object_index]]:
                            for l in class_to_objects[object_classes[reference_object_index]]:
                                if k != target_object_index:
                                    # if any other object of the same class has the same relation with any reference object of same class, then it's not unique
                                    if (k, l) in all_relations[relation]:
                                        is_unique = False
                                        break
                        
                        if is_unique:
                            # This relation uniquely identifies object i
                            # Create relational query
                            description = random.choice(self.instruction_templates).format(
                                target_class=target_class,
                                relation=relation.value,
                                reference_class=reference_class
                            )
                            
                            # Create object details
                            obj_detail = random.choice(self.object_detail_templates).format(
                                id=target_object_index,
                                x=locations[target_object_index, 0],
                                y=locations[target_object_index, 1],
                                z=locations[target_object_index, 2],
                                width=locations[target_object_index, 3],
                                height=locations[target_object_index, 4],
                                depth=locations[target_object_index, 5]
                            )
                            
                            # Create expected response
                            expected_response = random.choice(self.response_templates).format(
                                target_class=target_class,
                                relation=relation.value,
                                reference_class=reference_class,
                                object_id=target_object_index,
                                object_details=obj_detail,
                                count=1
                            )
                            
                            # Add thinking trace if enabled
                            if self.add_thinking_trace:
                                # Create detailed object descriptions for target and reference objects
                                
                                # List all reference object details
                                reference_object_details = []
                                for idx in range(len(object_classes)):
                                    if object_classes[idx] == object_classes[reference_object_index]:
                                        reference_object_details.append(
                                            random.choice(self.object_detail_templates).format(
                                                id=idx,
                                                x=locations[idx, 0],
                                                y=locations[idx, 1],
                                                z=locations[idx, 2],
                                                width=locations[idx, 3],
                                                height=locations[idx, 4],
                                                depth=locations[idx, 5]
                                            )
                                        )
                                
                                # List all target object details
                                target_object_details = []
                                for idx in range(len(object_classes)):
                                    if object_classes[idx] == object_classes[target_object_index]:
                                        target_object_details.append(
                                            random.choice(self.object_detail_templates).format(
                                                id=idx,
                                                x=locations[idx, 0],
                                                y=locations[idx, 1],
                                                z=locations[idx, 2],
                                                width=locations[idx, 3],
                                                height=locations[idx, 4],
                                                depth=locations[idx, 5]
                                            )
                                        )
                                
                                # Create spatial analysis text
                                # TODO: here we can do this because we have ALL relations, but in real world we don't have this, we can use "not <relation>" to get the opposite relation
                                spatial_analysis = []
                                for target_idx in range(len(object_classes)):
                                    if object_classes[target_idx] == object_classes[target_object_index]:
                                        for ref_idx in range(len(object_classes)):
                                            if object_classes[ref_idx] == object_classes[reference_object_index]:
                                                # might have no relation, in that case, add SpatialRelation.NONE ("no relation")
                                                target_rels = get_spatial_relation(locations[target_idx], locations[ref_idx], add_none_relation=True)
                                                rel_str = format_multiple_predicates([r.value for r in target_rels])
                                                spatial_analysis.append(
                                                    random.choice(self.spatial_analysis_template).format(
                                                        target_id=target_idx,
                                                        target_class=target_class,
                                                        relations=rel_str,
                                                        reference_id=ref_idx,
                                                        reference_class=reference_class
                                                    )
                                                )

                                # Format the thinking trace
                                if self.add_full_thinking_trace_for_relational:
                                    # add all object's details
                                    object_details_with_class = []
                                    for idx in range(len(object_classes)):
                                        obj_name = self.object_classes[int(object_classes[idx])]
                                        obj_detail = random.choice(self.object_detail_with_class_templates).format(
                                            id=idx,
                                            object_name=obj_name,
                                            x=locations[idx, 0],
                                            y=locations[idx, 1],
                                            z=locations[idx, 2],
                                            width=locations[idx, 3],
                                            height=locations[idx, 4],
                                            depth=locations[idx, 5]
                                        )
                                        object_details_with_class.append(obj_detail)


                                    thinking_trace = random.choice(self.relational_thinking_trace_template_full).format(
                                        object_details_with_class="\n".join(object_details_with_class),
                                        target_class=target_class,
                                        relation=relation.value,
                                        reference_class=reference_class,
                                        reference_object_details="\n".join(reference_object_details),
                                        target_object_details="\n".join(target_object_details),
                                        spatial_analysis="\n".join(spatial_analysis),
                                        object_id=target_object_index,
                                        reference_id=reference_object_index
                                    )
                                elif self.add_partial_full_thinking_trace_for_relational:
                                    # add only object IDs and names
                                    object_ids_with_class = []
                                    for idx in range(len(object_classes)):
                                        obj_name = self.object_classes[int(object_classes[idx])]
                                        obj_detail = self.object_id_with_class_templates.format(
                                            id=idx,
                                            object_name=obj_name
                                        )
                                        object_ids_with_class.append(obj_detail)

                                    thinking_trace = random.choice(self.relational_thinking_trace_template_partial_full).format(
                                        object_ids_with_class=", ".join(object_ids_with_class),
                                        target_class=target_class,
                                        relation=relation.value,
                                        reference_class=reference_class,
                                        reference_object_details="\n".join(reference_object_details),
                                        target_object_details="\n".join(target_object_details),
                                        spatial_analysis="\n".join(spatial_analysis),
                                        object_id=target_object_index,
                                        reference_id=reference_object_index
                                    )
                                    
                                else:
                                    thinking_trace = random.choice(self.relational_thinking_trace_template).format(
                                        target_class=target_class,
                                        relation=relation.value,
                                        reference_class=reference_class,
                                        reference_object_details="\n".join(reference_object_details),
                                        target_object_details="\n".join(target_object_details),
                                        spatial_analysis="\n".join(spatial_analysis),
                                        object_id=target_object_index,
                                        reference_id=reference_object_index
                                    )
                                
                                # Add thinking trace to expected response
                                expected_response = thinking_trace + expected_response
                            
                            # Create a program that represents this relational query
                            program = f"relate(filter(scene(), {target_class}), filter(scene(), {reference_class}), {relation.value})"
                            
                            # Add annotation
                            ann_id = len(self.annotation)
                            scanrefer_id = f"{scene_id}|{ann_id}"
                            hash_id = f"synthetic_{scene_id}_{ann_id}"
                            
                            self.annotation.append({
                                "scene_id": scene_id,
                                "ann_id": str(ann_id),
                                "description": description,
                                "program": program,
                                "object_name": target_class,
                                "object_id": target_object_index,
                                "object_ids": [target_object_index],  # Just this one object is unique
                                "relation": relation.value,
                                "reference_class": reference_class,
                                "reference_id": reference_object_index,
                                "is_relational": True,
                                "scanrefer_id": scanrefer_id,
                                "hash_id": hash_id,
                                "expected_response": expected_response
                            })
        
        # Update scanrefer_id to idx mapping
        self.scanrefer_id_to_idx = {data["scanrefer_id"]: idx for idx, data in enumerate(self.annotation)}
        
        logger.info(f"Generated {len(self.annotation)} relational query annotations")
    
    def evaluate(self, preds, gt_indices, iou_threshold=0.25, hash_id_index: bool=False, use_closest_gt: bool=False) -> Tuple[str, Dict]:
        """
        Evaluate predictions against ground truth.
        Enhanced to support evaluation of relational queries.
        """
        # First call the parent evaluation method
        message, metrics = super().evaluate(preds, gt_indices, iou_threshold, hash_id_index, use_closest_gt)
        
        # since it is single-object output, remove the recall, precision, and f1_score
        # metrics.pop("precision", None)
        metrics.pop("recall", None)
        metrics.pop("f1_score", None)

        message = (
            f"Relation Query Task:\n"
            f"ID Acc: {metrics['id_accuracy']:.4f}, "
            f"Mean IoU: {metrics['mean_iou']:.4f}, "
            f"IoU Acc@{iou_threshold:.2f}: {metrics['precision']:.4f}, " # for single prediction, precision == accuracy
            f"Total: {metrics['total']}"
        )
        
        return message, metrics

class Sr3DReasoningDataset:
    """使用真实Sr3D数据的3D推理任务数据集。"""
    
    SYSTEM_PROMPT: str = (
        "Respond in the following format, potraying \"Apeiria\":\n"
        "[APEIRIA THINKS]\n"
        "<... thinking predure ...>\n"
        "[APEIRIA SPEAKS]\n"
        "Apeiria <... responses ...>"
    )
    
    def __init__(self, tokenizer, split="train", max_objects=100, seed=42):
        """初始化数据集"""
        self.tokenizer = tokenizer
        self.data_path = DATA_PATH
        self.split = split
        self.max_objects = max_objects
        self.seed = seed
        
        # 设置随机种子
        random.seed(seed)
        np.random.seed(seed)
        
        # 加载Sr3D注释数据
        self.annotation_file = self._get_annotation_file(split)
        self.annotations = self._load_annotations()
        
        # 加载场景数据
        self.scene_data = self._load_scene_data()
        
        # 生成样本
        self.samples = self._generate_samples_from_annotations()
        
        # 记录示例样本
        if self.samples:
            logger.info(f"Sr3D样本提示: {self.samples[0]['prompt']}")
            logger.info(f"Sr3D预期响应: {self.samples[0]['answer']}")
    
    def _get_annotation_file(self, split):
        """获取注释文件路径"""
        SR3D_ANNO = {
            "train": f"{self.data_path}/sr3d_with_programs_train.json",
            "val": f"{self.data_path}/sr3d_with_programs_val.json",
        }
        return SR3D_ANNO[split]
    
    def _load_annotations(self):
        """从Sr3D文件加载注释"""
        with open(self.annotation_file, 'r') as f:
            annotations = json.load(f)
        logger.info(f"从{self.annotation_file}加载了{len(annotations)}条注释")
        return annotations
    
    def _load_scene_data(self):
        """加载所有场景的场景数据"""
        scene_data = {}
        scene_ids = set(anno["scene_id"] for anno in self.annotations)
        
        for scene_id in scene_ids:
            scene_file = f"{self.data_path}/apeiria_scannet_w_caption_gpt4o_and_corners_and_nyu_names_fixed/{scene_id}.json"
            try:
                with open(scene_file, 'r') as f:
                    scene_info = json.load(f)
                    scene_data[scene_id] = scene_info
            except FileNotFoundError:
                logger.warning(f"场景文件未找到: {scene_file}")
        
        logger.info(f"加载了{len(scene_data)}个场景数据文件")
        return scene_data
    
    def _format_object_set(self, objects):
        """将对象集格式化为字符串"""
        object_strings = []
        for obj in objects:
            location = obj.get("location", [0, 0, 0])
            size = obj.get("size", [0, 0, 0])
            object_strings.append(
                f"Object {obj['id']}: Category: {obj['name']}, "
                f"Position: ({location[0]}, {location[1]}, {location[2]}), "
                f"Size: {size[0]} x {size[1]} x {size[2]}"
            )
        return "\n".join(object_strings)
    
    def _generate_samples_from_annotations(self):
        """从Sr3D注释生成样本"""
        samples = []
        
        for anno in self.annotations:
            scene_id = anno["scene_id"]
            
            if scene_id not in self.scene_data:
                continue
                
            description = anno["description"]
            program = anno["program"]
            object_id = int(anno["object_id"]) if "object_id" in anno else None
            
            # 根据程序确定任务类型
            task_type = "filter"  # 默认
            if "relate(" in program:
                task_type = "relate"
            
            # 获取此场景的对象
            objects = self.scene_data[scene_id]["objects"]
            
            # 格式化对象集
            object_set = self._format_object_set(objects)
            
            # 创建输入提示
            prompt = (
                f"These are all objects in the scene: \n{object_set}\n"
                f"Think about the scene first. Find the object described as: \"{description}\"\n"
                f"In final answer, respond with \"Apeiria found...\" or \"didn't find any...\", and a list of Object <ID>: At (..., ..., ...), size: ... x ... x ..."
            )
            
            # 根据object_id查找目标对象
            target_objects = []
            if object_id is not None:
                for obj in objects:
                    if obj["id"] == object_id:
                        target_objects.append(obj)
            
            # 生成思考痕迹
            thinking_trace = (
                f"I need to find the object described as: \"{description}\".\n"
                f"The program for this task is: {program}\n"
            )
            
            if task_type == "filter":
                thinking_trace += "This task involves filtering objects by certain properties."
            elif task_type == "relate":
                thinking_trace += "This task involves finding objects with specific spatial relationships."
            
            if target_objects:
                thinking_trace += "\nTarget objects found:"
                for obj in target_objects:
                    location = obj.get("location", [0, 0, 0])
                    size = obj.get("size", [0, 0, 0])
                    thinking_trace += f"\n- Object {obj['id']} is a {obj['name']} at position ({location[0]}, {location[1]}, {location[2]}) with size {size[0]} x {size[1]} x {size[2]}"
            else:
                thinking_trace += f"\nI couldn't find any objects matching the description."
            
            # 生成预期的响应
            if target_objects:
                response_body = f"Apeiria found {len(target_objects)} object(s) matching the description:"
                for obj in target_objects:
                    location = obj.get("location", [0, 0, 0])
                    size = obj.get("size", [0, 0, 0])
                    response_body += f"\nObject {obj['id']}: At ({location[0]}, {location[1]}, {location[2]}), size: {size[0]} x {size[1]} x {size[2]}"
            else:
                response_body = f"Apeiria didn't find any objects matching the description."
            
            expected_response = f"[APEIRIA THINKS]\n{thinking_trace}\n[APEIRIA SPEAKS]\n{response_body}" + "<|im_end|>" + self.tokenizer.eos_token

            # 应用tokenizer聊天模板
            prompt_messages = [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ]
            formatted_prompt = self.tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
            
            # 存储格式化的响应和原始对象数据
            samples.append({
                "prompt": formatted_prompt,
                "answer": expected_response,
                "description": description,
                "objects": target_objects,
                "task_type": task_type,
                "program": program,
                "scene_id": scene_id,
                "object_id": object_id
            })
        
        return samples


class Sr3DRelationalReasoningDataset(Dataset, Sr3DReasoningDataset):
    """
    专注于处理真实世界3D场景中的关系查询的数据集，
    从程序代码生成思考痕迹，逐步执行函数调用。
    """
    
    def __init__(
        self,
        tokenizer,
        data_path,
        split="train",
        max_objects=100,
        seed=42,
        add_full_thinking_trace=False,
        add_partial_thinking_trace=False
    ):
        # super().__init__(tokenizer, data_path, split, max_objects, seed)
        super(Dataset).__init__()
        
        self.tokenizer = tokenizer
        self.data_path = data_path
        self.split = split
        self.max_objects = max_objects
        self.seed = seed

        # 设置随机种子
        random.seed(seed)
        np.random.seed(seed)
        
        # 加载Sr3D注释数据
        self.annotation_file = self._get_annotation_file(split)
        self.annotations = self._load_annotations()
        
        # 加载场景数据
        self.scene_data = self._load_scene_data()
        
        # 生成样本
        self.samples = self._generate_samples_from_annotations()
        
        # 记录示例样本
        if self.samples:
            logger.info(f"Sr3D样本提示: {self.samples[0]['prompt']}")
            logger.info(f"Sr3D预期响应: {self.samples[0]['answer']}")
        
        self.add_full_thinking_trace = add_full_thinking_trace
        self.add_partial_thinking_trace = add_partial_thinking_trace
        
        # 基本关系思考模板
        self.relational_thinking_trace_template = (
            "[APEIRIA THINKS]\n"
            "I need to find the {target_class} {relation} the {reference_class}.\n"
            "First, let me find all {target_class}s in the scene:\n"
            "{target_object_details}\n\n"
            "Next, let me find all {reference_class}s in the scene:\n"
            "{reference_object_details}\n\n"
            "Now, I'll check which {target_class}(s) are {relation} the {reference_class}(s):\n"
            "{spatial_analysis}\n\n"
            "Therefore, the {target_class} with ID {object_id} is {relation} the {reference_class} with ID {reference_id}.\n"
            "[APEIRIA SPEAKS]\n"
        )
        
        # 完整思考模板（包含所有对象）
        self.relational_thinking_trace_template_full = (
            "[APEIRIA THINKS]\n"
            "I need to find the {target_class} {relation} the {reference_class}.\n"
            "First, let me list all objects in the scene:\n"
            "{all_object_details}\n\n"
            "Among these, I'll find all {target_class}s:\n"
            "{target_object_details}\n\n"
            "And all {reference_class}s:\n"
            "{reference_object_details}\n\n"
            "Now, I'll check which {target_class}(s) are {relation} the {reference_class}(s):\n"
            "{spatial_analysis}\n\n"
            "Therefore, the {target_class} with ID {object_id} is {relation} the {reference_class} with ID {reference_id}.\n"
            "[APEIRIA SPEAKS]\n"
        )
        
        # 部分完整思考模板（只包含对象ID和名称）
        self.relational_thinking_trace_template_partial = (
            "[APEIRIA THINKS]\n"
            "I need to find the {target_class} {relation} the {reference_class}.\n"
            "First, let me list the object IDs and names:\n"
            "Object IDs and Names: {object_ids_with_names}\n\n"
            "Among these, I'll find all {target_class}s:\n"
            "{target_object_details}\n\n"
            "And all {reference_class}s:\n"
            "{reference_object_details}\n\n"
            "Now, I'll check which {target_class}(s) are {relation} the {reference_class}(s):\n"
            "{spatial_analysis}\n\n"
            "Therefore, the {target_class} with ID {object_id} is {relation} the {reference_class} with ID {reference_id}.\n"
            "[APEIRIA SPEAKS]\n"
        )
        
        # 空间关系分析模板
        self.spatial_analysis_templates = [
            "Object {target_id} ({target_class}) is {relation_result} Object {reference_id} ({reference_class}).",
            "The {target_class} (ID {target_id}) is {relation_result} the {reference_class} (ID {reference_id})."
        ]
        
        # 重新生成只包含关系查询的样本
        self.samples = self._generate_relational_samples()

    def _get_annotation_file(self, split):
        """获取注释文件路径"""
        SR3D_ANNO = {
            "train": f"{self.data_path}/sr3d_with_programs_train.json",
            "val": f"{self.data_path}/sr3d_with_programs_val.json",
        }
        return SR3D_ANNO[split]
    
    def _load_annotations(self):
        """从Sr3D文件加载注释"""
        with open(self.annotation_file, 'r') as f:
            annotations = json.load(f)
        logger.info(f"从{self.annotation_file}加载了{len(annotations)}条注释")
        return annotations
    
    def _parse_program(self, program):
        """解析程序字符串，提取目标类别、参考类别和关系"""
        # 常见的关系查询程序模式
        pattern = r"relate\(filter\(scene\(\), ([^)]+)\), filter\(scene\(\), ([^)]+)\), ([^)]+)\)"
        match = re.match(pattern, program)
        
        if match:
            target_class = match.group(1).strip().replace("_", " ")
            reference_class = match.group(2).strip().replace("_", " ")
            relation = match.group(3).strip()
            return target_class, reference_class, relation
        
        # 如果无法解析标准格式，尝试更复杂的解析
        logger.warning(f"无法解析程序: {program}")
        return None, None, None
    
    def _check_spatial_relation(self, obj1, obj2, relation):
        """检查两个对象之间是否存在指定的空间关系"""
        # 获取对象的位置
        obj1_pos = np.array(obj1["location"])
        obj2_pos = np.array(obj2["location"])
        
        # 计算距离
        distance = np.linalg.norm(obj1_pos - obj2_pos)
        
        # 基于关系类型判断
        if relation == "near":
            return distance < 2.0  # 在2.0单位内视为"近"
        elif relation == "far":
            return distance > 5.0  # 超过5.0单位视为"远"
        
        # 位置关系（轴向）
        elif relation == "left":
            return obj1_pos[0] < obj2_pos[0]
        elif relation == "right":
            return obj1_pos[0] > obj2_pos[0]
        elif relation == "above" or relation == "over":
            return obj1_pos[1] > obj2_pos[1]
        elif relation == "below" or relation == "under" or relation == "beneath" or relation == "underneath":
            return obj1_pos[1] < obj2_pos[1]
        elif relation == "front" or relation == "in_front":
            return obj1_pos[2] < obj2_pos[2]
        elif relation == "behind" or relation == "back":
            return obj1_pos[2] > obj2_pos[2]
        elif relation == "beside" or relation == "next":
            # "旁边"意味着x或z轴接近，但y轴（高度）可能不同
            horizontal_distance = np.sqrt((obj1_pos[0] - obj2_pos[0])**2 + (obj1_pos[2] - obj2_pos[2])**2)
            return horizontal_distance < 1.5 and abs(obj1_pos[1] - obj2_pos[1]) < 2.0
        
        # 默认返回False
        logger.warning(f"未知的关系类型: {relation}")
        return False
    
    def _generate_object_detail(self, obj):
        """生成对象的详细文本描述"""
        location = obj.get("location", [0, 0, 0])
        size = obj.get("size", [0, 0, 0])
        
        return (
            f"Object {obj['id']}: A {obj['name']} at position ({location[0]:.2f}, {location[1]:.2f}, {location[2]:.2f}) "
            f"with size {size[0]:.2f} x {size[1]:.2f} x {size[2]:.2f}"
        )
    
    def _generate_spatial_analysis(self, target_objects, reference_objects, relation):
        """生成空间关系分析文本"""
        analysis_lines = []
        template = random.choice(self.spatial_analysis_templates)
        
        for target_obj in target_objects:
            for ref_obj in reference_objects:
                is_related = self._check_spatial_relation(target_obj, ref_obj, relation)
                relation_result = relation if is_related else f"not {relation}"
                
                analysis_lines.append(
                    template.format(
                        target_id=target_obj['id'],
                        target_class=target_obj['name'],
                        relation_result=relation_result,
                        reference_id=ref_obj['id'],
                        reference_class=ref_obj['name']
                    )
                )
        
        return "\n".join(analysis_lines)
    
    def _get_objects_by_class(self, scene_objects, class_name):
        """获取特定类别的所有对象"""
        matching_objects = []
        class_name = class_name.lower()
        
        for obj in scene_objects:
            obj_name = obj["name"].lower()
            if obj_name == class_name or obj_name.replace(" ", "_") == class_name:
                matching_objects.append(obj)
        
        return matching_objects
    
    def _generate_thinking_trace(self, scene_objects, program):
        """根据程序生成思考痕迹"""
        # 解析程序
        target_class, reference_class, relation = self._parse_program(program)
        
        if not target_class or not reference_class or not relation:
            return "[APEIRIA THINKS]\nI couldn't understand the program.\n[APEIRIA SPEAKS]\n", []
        
        # 找到所有目标类别对象
        target_objects = self._get_objects_by_class(scene_objects, target_class)
        
        # 找到所有参考类别对象
        reference_objects = self._get_objects_by_class(scene_objects, reference_class)
        
        # 生成目标对象描述
        target_object_details = []
        for obj in target_objects:
            target_object_details.append(self._generate_object_detail(obj))
        
        if not target_object_details:
            target_object_details.append(f"No {target_class}s found in the scene.")
        
        # 生成参考对象描述
        reference_object_details = []
        for obj in reference_objects:
            reference_object_details.append(self._generate_object_detail(obj))
        
        if not reference_object_details:
            reference_object_details.append(f"No {reference_class}s found in the scene.")
        
        # 生成空间关系分析
        spatial_analysis = self._generate_spatial_analysis(target_objects, reference_objects, relation)
        
        # 找到满足关系的对象对
        matching_pairs = []
        for target_obj in target_objects:
            for ref_obj in reference_objects:
                if self._check_spatial_relation(target_obj, ref_obj, relation):
                    matching_pairs.append((target_obj, ref_obj))
        
        # 选择一个匹配对象用于结论
        if matching_pairs:
            object_id = matching_pairs[0][0]["id"]
            reference_id = matching_pairs[0][1]["id"]
        else:
            # 没找到匹配对象
            object_id = -1
            reference_id = -1
        
        # 生成思考痕迹
        if self.add_full_thinking_trace:
            # 包含所有对象详情
            all_object_details = []
            for obj in scene_objects:
                all_object_details.append(self._generate_object_detail(obj))
            
            thinking_trace = self.relational_thinking_trace_template_full.format(
                target_class=target_class,
                reference_class=reference_class,
                relation=relation,
                all_object_details="\n".join(all_object_details),
                target_object_details="\n".join(target_object_details),
                reference_object_details="\n".join(reference_object_details),
                spatial_analysis=spatial_analysis,
                object_id=object_id,
                reference_id=reference_id
            )
        elif self.add_partial_thinking_trace:
            # 只包含对象ID和名称
            object_ids_with_names = []
            for obj in scene_objects:
                object_ids_with_names.append(f"{obj['id']}:{obj['name']}")
            
            thinking_trace = self.relational_thinking_trace_template_partial.format(
                target_class=target_class,
                reference_class=reference_class,
                relation=relation,
                object_ids_with_names=", ".join(object_ids_with_names),
                target_object_details="\n".join(target_object_details),
                reference_object_details="\n".join(reference_object_details),
                spatial_analysis=spatial_analysis,
                object_id=object_id,
                reference_id=reference_id
            )
        else:
            # 使用基础模板
            thinking_trace = self.relational_thinking_trace_template.format(
                target_class=target_class,
                reference_class=reference_class,
                relation=relation,
                target_object_details="\n".join(target_object_details),
                reference_object_details="\n".join(reference_object_details),
                spatial_analysis=spatial_analysis,
                object_id=object_id,
                reference_id=reference_id
            )
        
        return thinking_trace, matching_pairs
    
    def _generate_relational_samples(self):
        """生成所有关系型查询样本"""
        relational_samples = []
        
        for anno in self.annotation:
            program = anno.get("program", "")
            
            # 只处理包含关系的程序
            if "relate(" not in program:
                continue
                
            scene_id = anno["scene_id"]
            if scene_id not in self.scene_data:
                continue
                
            description = anno["description"]
            
            # 获取场景对象
            scene_objects = self.scene_data[scene_id]["objects"]
            
            # 格式化对象列表供提示使用
            object_set = self._format_object_set(scene_objects)
            
            # 解析程序
            target_class, reference_class, relation = self._parse_program(program)
            
            if not target_class or not reference_class or not relation:
                continue
            
            # 创建输入提示
            prompt = (
                f"These are all objects in the scene: \n{object_set}\n\n"
                f"Think about the scene first. Find the {target_class} that is {relation} the {reference_class}.\n"
                f"In your final answer, respond with \"Apeiria found...\" or \"didn't find any...\", "
                f"and provide details in the format Object <ID>: At (..., ..., ...), size: ... x ... x ..."
            )
            
            # 生成思考痕迹和匹配对象
            thinking_trace, matching_pairs = self._generate_thinking_trace(scene_objects, program)
            
            # 生成预期响应
            if matching_pairs:
                obj = matching_pairs[0][0]  # 使用第一个匹配对象
                location = obj.get("location", [0, 0, 0])
                size = obj.get("size", [0, 0, 0])
                
                response_body = (
                    f"Apeiria found the {target_class} that is {relation} the {reference_class}:\n"
                    f"Object {obj['id']}: At ({location[0]:.2f}, {location[1]:.2f}, {location[2]:.2f}), "
                    f"size: {size[0]:.2f} x {size[1]:.2f} x {size[2]:.2f}"
                )
            else:
                response_body = f"Apeiria didn't find any {target_class} that is {relation} the {reference_class}."
            
            # 组合完整响应（思考+回答）
            expected_response = thinking_trace + response_body + "<|im_end|>" + self.tokenizer.eos_token
            
            # 应用tokenizer聊天模板
            prompt_messages = [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ]
            formatted_prompt = self.tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
            
            # 保存样本
            relational_samples.append({
                "prompt": formatted_prompt,
                "answer": expected_response,
                "description": description,
                "objects": [pair[0] for pair in matching_pairs] if matching_pairs else [],
                "task_type": "relate",
                "program": program,
                "scene_id": scene_id,
                "target_class": target_class,
                "reference_class": reference_class,
                "relation": relation
            })
        
        logger.info(f"生成了{len(relational_samples)}个关系推理样本")
        return relational_samples


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
