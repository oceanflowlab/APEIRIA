import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import sigmoid_focal_loss
import torch.distributed as dist
import signal
from accelerate.utils import clear_environment, patch_environment
from transformers import AutoModel, AutoTokenizer, GenerationMixin
from enum import Enum, auto
from typing import List, Dict, Union, Optional, Tuple, Any, Callable
from dataclasses import dataclass
import logging
import lark
import torch.cuda as cuda
import psutil
import os
import time
import transformers
from types import SimpleNamespace
from argparse import Namespace
import numpy as np
import re
import pretty_errors
from icecream import ic
import socket
from contextlib import closing
import sys
import json
import psutil
import threading
from packaging import version
import open_clip
import nest_asyncio
import gc

nest_asyncio.apply()

pretty_errors.configure(
    separator_character = '*',
    filename_display    = pretty_errors.FILENAME_FULL,
    line_number_first   = True,
    display_link        = True,
    lines_before        = 5,
    lines_after         = 2,
    line_color          = pretty_errors.RED + '> ' + pretty_errors.default_config.line_color,
    code_color          = '  ' + pretty_errors.default_config.line_color,
    truncate_code       = True,
    display_locals      = True
)

try:
    from sparsemax import Sparsemax
except ImportError:
    print("Sparsemax not installed, install if needed")

SGLANG_VERSION = None
SGLANG_REQUIRED_VERSION = version.parse("0.5.0")
try:
    import sglang as sgl
    SGLANG_VERSION = version.parse(sgl.__version__)
except ImportError:
    import traceback
    traceback.print_exc()
    print("sglang not installed, install if needed SGLang accelerated inference")

from peft import PeftModel
import accelerate

from fourier import FourierFeatureMapping

NEW_SGLANG = SGLANG_VERSION is not None and SGLANG_VERSION >= SGLANG_REQUIRED_VERSION # enable radix attention cache and cuda graph for LoRA

logger = logging.getLogger(__name__)

class MyObjectDict(dict):
    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            raise AttributeError(f"Attribute {name} not found")
    
    def __setattr__(self, name, value):
        self[name] = value

def get_my_all_child_pids():
    """
    Uses psutil to find and return a list of PIDs of the children
    of the currently running process.
    """
    child_pids = []
    try:
        # Get the psutil object for the current process
        current_process = psutil.Process() # No argument needed for current process
        parent_pid = current_process.pid

        # Get a list of direct child process objects
        # recursive=False is key here!
        all_children = current_process.children(recursive=True)

        # Extract the PID from each child process object
        child_pids = [child.pid for child in all_children]

        logger.info(f"[PID:{parent_pid}] Found {len(child_pids)} children.")

    except psutil.NoSuchProcess:
        # Should not happen for the current process, but good practice
        logger.info("Error: Could not find current process information.")
        raise
    except psutil.Error as e:
        # Handle other potential psutil errors
        logger.info(f"An error occurred while querying children: {e}")
        raise

    return child_pids

# Adapted from SGLang, the actual code executed in engine.shutdown(), but we want to add skip_pids to avoid
# killing the subprocesses we created, like the dataloaders and wandb processes
def kill_process_tree(parent_pid, include_parent: bool = True, skip_pid: int = None, skip_pids: list[int] = []):
    """Kill the process and all its child processes."""
    # Remove sigchld handler to avoid spammy logs.
    if threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGCHLD, signal.SIG_DFL)

    if parent_pid is None:
        parent_pid = os.getpid()
        include_parent = False

    try:
        itself = psutil.Process(parent_pid)
    except psutil.NoSuchProcess:
        return

    children = itself.children(recursive=True)
    for child in children:
        if child.pid == skip_pid or child.pid in skip_pids:
            continue
        try:
            child.kill()
        except psutil.NoSuchProcess:
            pass

    if include_parent:
        try:
            if parent_pid == os.getpid():
                itself.kill()
                sys.exit(0)

            itself.kill()

            # Sometime processes cannot be killed with SIGKILL (e.g, PID=1 launched by kubernetes),
            # so we send an additional signal to kill them.
            itself.send_signal(signal.SIGQUIT)
        except psutil.NoSuchProcess:
            pass

def get_module_device(module: nn.Module) -> torch.device:
    return next(module.parameters()).device

def find_free_port():
    """动态寻找一个可用的空闲端口"""
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))  # 绑定到一个随机空闲端口
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]  # 返回分配的端口号


class ObjectEmbeddingSimple(nn.Module):
    def __init__(self, object_feature_dim: int, decoder_hidden_size: int, dropout: float = 0.1, use_layer_norm: bool = True):
        super().__init__()

        self.object_feature_dim = object_feature_dim
        self.decoder_hidden_size = decoder_hidden_size

        self.net = nn.Sequential(
            nn.Linear(object_feature_dim, decoder_hidden_size // 2),
            # will it make the model can't separate different embedding components?
            # such as it can't separate object ID/class embedding with object position embedding
            # further experiments show: it stablize the model to be slower to converge, but more stable
            nn.LayerNorm(decoder_hidden_size // 2) if use_layer_norm else nn.Identity(), 
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(decoder_hidden_size // 2, decoder_hidden_size),
            nn.LayerNorm(decoder_hidden_size) if use_layer_norm else nn.Identity(),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self._init_weights()

    def _init_weights(self):
        # Initialize the weights with kaiming normal
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, object_features: torch.Tensor) -> torch.Tensor:
        return self.net(object_features)
    

class ObjectFourierEmbedding(nn.Module):
    def __init__(self, object_feature_dim: int, decoder_hidden_size: int, num_fourier_features: int, dropout: float = 0.1, use_layer_norm: bool = True, fourier_scale: float = 1.0):
        """
        Assume the input last 6 dimensions are object position and size, apply fourier features to them.
        First, split the input tensor into object features and object position features, then apply fourier features to object position features.
        then, concatenate object features and fourier features, apply a linear layer to project them to decoder hidden size.
        """
        super().__init__()

        self.object_feature_dim = object_feature_dim
        self.decoder_hidden_size = decoder_hidden_size
        self.num_fourier_features = num_fourier_features

        self.object_proj = ObjectEmbeddingSimple(object_feature_dim, decoder_hidden_size, dropout, use_layer_norm)
        self.fourier_embedding = FourierFeatureMapping(6, scale=fourier_scale, mapping_size=num_fourier_features) #　=> 2 * num_fourier_features
        self.linear = nn.Linear(2 * num_fourier_features, decoder_hidden_size)

    def forward(self, object_features: torch.Tensor) -> torch.Tensor:
        object_position_features = object_features[..., -6:]
        object_position_features = self.fourier_embedding(object_position_features) # [..., 2 * num_fourier_features]
        object_position_features = self.linear(object_position_features)

        x = self.object_proj(object_features) + object_position_features
        return x
    
    
class ObjectDiscreteLocationEmbedding(nn.Module):
    def __init__(self, object_feature_dim: int, decoder_hidden_size: int, dropout: float = 0.1, use_layer_norm: bool = True, 
                 num_bins: int = 101, decay_kernel: str="exponential", 
                 bin_range: Tuple[float, float] | List[Tuple[float, float]] = (0.0, 1.0), kernel_size: float = 1.0,
                 separate_location_embedding: bool = False,
        ):
        r"""
        Discretize the object location into num_bins, then apply embedding to it.
        Args:
            object_feature_dim: the dimension of object features
            decoder_hidden_size: the dimension of decoder hidden size
            dropout: dropout rate
            use_layer_norm: whether to use layer norm
            num_bins: the number of bins to discretize the object location
            decay_kernel: the kernel to calculate the distance between object location and bin values
            bin_range: the range of the bin values
            kernel_size: the kernel size to calculate the distance between object location and bin values, in the unit of bin interval(s)

        Note on the kernel size: If we assume kernels shall bave compact support and fixed standard deviation,
        the kernel size should be
            - \sqrt{6} * \sigma = 2.45 * \sigma for Linear kernel (triagular distribution)
            - (1/\sqrt{2}) * \sigma = 0.891 * \sigma for Exponential kernel (Laplacian distribution)
            - \sigma for Gaussian kernel
        Further, laplacian kernel have less compact support
        """
        
        super().__init__()

        self.object_feature_dim = object_feature_dim
        self.decoder_hidden_size = decoder_hidden_size
        self.num_bins = num_bins
        self.bin_range = bin_range
        self.decay_kernel = decay_kernel
        self.separate_location_embedding = separate_location_embedding
        if separate_location_embedding:
            object_feature_dim -= 6

        self.object_proj = ObjectEmbeddingSimple(object_feature_dim, decoder_hidden_size, dropout, use_layer_norm)
        self.embedding = nn.Linear(num_bins * 6, decoder_hidden_size)

        self.bin_values = nn.Parameter(torch.linspace(bin_range[0], bin_range[1], num_bins), requires_grad=False)
        self.bin_interval = (bin_range[1] - bin_range[0]) / (num_bins - 1)
        self.kernel_size = kernel_size

    def calculate_bin_coefficients(self, numerals: torch.Tensor) -> torch.Tensor:
        """
        Calculate the bin coefficients for the given numerals. 
        [B, 6] => [B, 6, num_bins]
        """
        bin_values = torch.linspace(self.bin_range[0], self.bin_range[1], self.num_bins, device=numerals.device)
        bin_values = bin_values.view(1, 1, -1) # [1, 1, num_bins]
        numerals = numerals.unsqueeze(-1) # [B, 6, 1]

        # calculate the distance between numerals and bin values
        numerals = torch.abs(numerals - bin_values) / self.bin_interval / self.kernel_size # [B, 6, num_bins]

        # apply decay kernel
        if self.decay_kernel == "exponential":
            numerals = torch.exp(-numerals) # for exponential/laplacian kernel, kernel_size is the standard deviation (with a factor of 1/sqrt(2))
        elif self.decay_kernel == "gaussian":
            numerals = torch.exp(-numerals ** 2) # for Gaussian kernel, kernel_size is the standard deviation
        elif self.decay_kernel == "linear":
            numerals = torch.clamp(1 - numerals, min=0) # for linear kernel, kernel_size is the hard threshold of the distance
        else:
            raise ValueError(f"Unknown decay kernel: {self.decay_kernel}")
        
        # normalize the coefficients
        numerals = numerals / numerals.sum(dim=-1, keepdim=True) # [B, 6, num_bins]
        return numerals


    def forward(self, object_features: torch.Tensor, object_position_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        if object_position_features is None:
            object_position_features = object_features[..., -6:]

        if self.separate_location_embedding:
            object_features = object_features[..., :-6]

        bin_coefficients = self.calculate_bin_coefficients(object_position_features)
        bin_coefficients = bin_coefficients.view(bin_coefficients.shape[0], -1)
        object_position_features = self.embedding(bin_coefficients)

        x = self.object_proj(object_features) + object_position_features
        return x
        
# Helper function to create a standard modality encoding network
def _create_modality_encoder_net(input_dim: int,
                                 output_dim: int, # This will be decoder_hidden_size
                                 dropout: float,
                                 use_layer_norm: bool,
                                 hidden_dim_factor: int = 2): # Determines hidden_dim = output_dim // hidden_dim_factor
    """
    Creates a two-layer MLP encoder for a modality.
    """
    if output_dim % hidden_dim_factor != 0:
        # Adjust hidden_dim_factor or output_dim if a clean division is desired for //
        # Or simply use a calculated hidden_dim, e.g., roughly half.
        # For simplicity, let's proceed with integer division.
        # Consider raising a warning or error if not perfectly divisible, depending on strictness.
        logger.warning(
            f"Output dimension {output_dim} is not divisible by hidden_dim_factor {hidden_dim_factor}. "
        )
        pass

    hidden_dim = output_dim // hidden_dim_factor
    if hidden_dim == 0 and output_dim > 0 : # Ensure hidden_dim is at least 1 if output_dim is positive
        hidden_dim = 1


    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity(),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, output_dim),
        nn.LayerNorm(output_dim) if use_layer_norm else nn.Identity(),
        nn.GELU(),
        nn.Dropout(dropout),
    )

class ObjectDiscreteLocationEmbeddingSeparate(nn.Module):
    def __init__(self,
                modality_dims: Dict[str, int],
                modality_order: str, # e.g., "id|2d|3d|location"
                decoder_hidden_size: int,
                location_modality_name: str, # Name of the modality for discrete location embedding
                num_bins: int = 101,
                decay_kernel: str = "exponential", # "exponential", "gaussian", "linear"
                bin_range: Tuple[float, float] = (0.0, 1.0), # Assumed uniform for all dims of location_modality_name
                kernel_size: float = 1.0, # In units of bin intervals
                dropout: float = 0.1,
                use_layer_norm: bool = True
                ):
        super().__init__()

        self.modality_dims = modality_dims
        self.modality_order_list = modality_order.split('|')
        self.decoder_hidden_size = decoder_hidden_size
        self.location_modality_name = location_modality_name
        
        self.num_bins = num_bins
        self.bin_range_tuple = bin_range 
        self.decay_kernel = decay_kernel
        self.kernel_size = kernel_size # Kernel size in units of bin intervals
        self.dropout = dropout
        self.use_layer_norm = use_layer_norm

        if self.location_modality_name not in self.modality_dims:
            raise ValueError(f"Location modality '{self.location_modality_name}' not found in modality_dims.")
        if self.location_modality_name not in self.modality_order_list:
            raise ValueError(f"Location modality '{self.location_modality_name}' not in modality_order string.")
        
        location_dim = self.modality_dims[self.location_modality_name]
        
        # Linear layer for the binned location features
        # Input is [num_bins * D_location]
        self.location_features_projection = nn.Linear(self.num_bins * location_dim, self.decoder_hidden_size)

        # Bin values and interval are calculated on-the-fly in calculate_bin_coefficients
        # to ensure they are on the correct device.

        self.other_modality_encoders = nn.ModuleDict()
        for modality_name in self.modality_order_list:
            if modality_name == self.location_modality_name:
                continue

            if modality_name not in self.modality_dims:
                raise ValueError(f"Dimension for modality '{modality_name}' not found in modality_dims.")
            
            current_modality_dim = self.modality_dims[modality_name]
            self.other_modality_encoders[modality_name] = _create_modality_encoder_net(
                input_dim=current_modality_dim,
                output_dim=self.decoder_hidden_size,
                dropout=self.dropout,
                use_layer_norm=self.use_layer_norm
            )
        
        self._init_weights()

    def _init_weights(self):
        # Initialize weights for location_features_projection
        if hasattr(self, 'location_features_projection') and isinstance(self.location_features_projection, nn.Linear):
            nn.init.kaiming_normal_(self.location_features_projection.weight)
            if self.location_features_projection.bias is not None:
                nn.init.zeros_(self.location_features_projection.bias)

        # Initialize weights for other_modality_encoders
        for modality_name in self.other_modality_encoders:
            for layer in self.other_modality_encoders[modality_name]:
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_normal_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

    def _split_features(self, object_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Splits the flat object_features tensor into a dictionary of modality tensors."""
        split_features = {}
        current_idx = 0
        for modality_name in self.modality_order_list:
            dim = self.modality_dims[modality_name]
            if current_idx + dim > object_features.shape[-1]:
                raise ValueError(
                    f"Modality '{modality_name}' with dim {dim} exceeds input tensor dimension "
                    f"({object_features.shape[-1]}) at current_idx {current_idx}."
                )
            split_features[modality_name] = object_features[..., current_idx : current_idx + dim]
            current_idx += dim
        
        if current_idx != object_features.shape[-1]:
            raise ValueError(
                f"Total dimension of modalities ({current_idx}) does not match "
                f"input feature dimension ({object_features.shape[-1]}). "
                f"Modality order: {self.modality_order_list}, Dims: {self.modality_dims}"
            )
        return split_features

    def calculate_bin_coefficients(self, location_modality_features: torch.Tensor) -> torch.Tensor:
        """
        Calculates bin coefficients for the given location features.
        Args:
            location_modality_features (torch.Tensor): Tensor of shape [Batch, D_location].
        Returns:
            torch.Tensor: Bin coefficients of shape [Batch, D_location, num_bins].
        """
        # D_location is location_modality_features.shape[-1]
        
        # Ensure bin_values and bin_interval are on the correct device
        bin_values_tensor = torch.linspace(
            self.bin_range_tuple[0], self.bin_range_tuple[1], self.num_bins, 
            device=location_modality_features.device
        )
        # Reshape for broadcasting: [1, 1, num_bins]
        bin_values_tensor = bin_values_tensor.view(1, 1, -1) 

        current_bin_interval = (self.bin_range_tuple[1] - self.bin_range_tuple[0]) / (self.num_bins - 1)
        if self.num_bins == 1: # Avoid division by zero if num_bins is 1
            current_bin_interval = 1.0 # Or handle as a special case

        # location_modality_features: [Batch, D_location]
        # Unsqueeze to [Batch, D_location, 1] for broadcasting with bin_values_tensor
        numerals_expanded = location_modality_features.unsqueeze(-1)

        # Distance from numeral to each bin center, normalized by bin_interval and kernel_size
        # dist shape: [Batch, D_location, num_bins]
        dist = torch.abs(numerals_expanded - bin_values_tensor)
        
        # Normalize distance by bin interval and kernel size
        # self.kernel_size is in units of bin intervals.
        # A kernel_size of 0 would cause division by zero.
        if self.kernel_size <= 0:
            raise ValueError("kernel_size must be positive.")
        if current_bin_interval <= 0 and self.num_bins > 1: # Should not happen if bin_range[0] < bin_range[1]
            raise ValueError("bin_interval must be positive for num_bins > 1.")

        normalized_dist = dist / current_bin_interval / self.kernel_size
        
        if self.decay_kernel == "exponential":
            coeffs = torch.exp(-normalized_dist)
        elif self.decay_kernel == "gaussian":
            coeffs = torch.exp(-normalized_dist.pow(2))
        elif self.decay_kernel == "linear": # Triangular kernel
            coeffs = torch.clamp(1 - normalized_dist, min=0)
        else:
            raise ValueError(f"Unknown decay kernel: {self.decay_kernel}")
        
        # Normalize coefficients along the num_bins dimension
        coeffs_sum = coeffs.sum(dim=-1, keepdim=True)
        
        # Add a small epsilon to prevent division from zero if all coeffs are zero
        # (e.g., numeral is too far from all bins for the chosen kernel_size)
        coeffs = coeffs / (coeffs_sum + 1e-8) 
        
        return coeffs # Shape: [Batch, D_location, num_bins]

    def forward(self, object_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        modality_input_features = self._split_features(object_features)
        output_embeddings = {}

        for modality_name in self.modality_order_list:
            input_feat = modality_input_features[modality_name]
            if modality_name == self.location_modality_name:
                # input_feat has shape [Batch, D_location]
                bin_coeffs = self.calculate_bin_coefficients(input_feat) # [Batch, D_location, num_bins]
                
                # Flatten [D_location, num_bins] -> [D_location * num_bins] for the linear layer
                # Batch dimension is preserved: [Batch, D_location * num_bins]
                bin_coeffs_flat = bin_coeffs.reshape(bin_coeffs.shape[0], -1) 
                
                output_embeddings[modality_name] = self.location_features_projection(bin_coeffs_flat)
            else:
                output_embeddings[modality_name] = self.other_modality_encoders[modality_name](input_feat)
                
        return output_embeddings

class MultimodalLanguageModelDecoderOnly(nn.Module):
    # APEIRIA_OPEN_UNUSED: Legacy 2D image encoder metadata. The final public
    # model receives object/proposal features only, so image_embedding_dim is
    # always None in the public training/inference entrypoints.
    image_encoder_to_embedding_dim_map = {
        "ViT-H-14-378-quickgelu|dfn5b": 1024, # dummy
        "ViT-gopt-16-SigLIP2-384|webli": 1536,
    }

    def __init__(
        self, 
        language_model: transformers.PreTrainedModel,
        tokenizer: transformers.PreTrainedTokenizer,
        object_feature_dim: int,
        max_objects: int = 50,
        no_object_in_language_model: bool = False,
        object_embedding_type: str = "simple",
        discrete_location_bins: int = 101,
        discrete_location_decay_kernel: str = "exponential",
        discrete_location_bin_range: Tuple[float, float] | List[Tuple[float, float]] = (0.0, 1.0),
        discrete_location_decay_kernel_size: float = 1.0,
        verbose: bool = False,
        separate_location_embedding: bool = False,
        dtype: torch.dtype = torch.bfloat16,
        modality_dims: Optional[Dict[str, int]] = None,
        modality_order: str = "2d|3d|id|location",
        image_embedding_dim: Optional[int] = None,
        image_encoder_model_id: Optional[str] = "ViT-H-14-378-quickgelu",  # APEIRIA_OPEN_UNUSED
        image_encoder_pretrained: Optional[str] = "dfn5b",  # APEIRIA_OPEN_UNUSED
        image_encoder_trainable: bool = False,  # APEIRIA_OPEN_UNUSED
        use_sglang: bool = False,
        sglang_model_path: Optional[str] = None,
        sglang_lora_paths: Optional[List[str]] = None,
        sglang_port: Optional[int] = None,  # SGLang服务器端口
        sglang_log_level: str = "info",
        coeff_grounding_loss: float = 0.3,
        delete_model_from_cpu: bool = False,
    ):
        super().__init__()

        self.no_object_in_language_model = no_object_in_language_model
        
        # Language model components
        self.tokenizer = tokenizer
        self.language_encoder = language_model
        self.vocab_size = self.tokenizer.vocab_size
        try:
            self.feature_dim = self.language_encoder.config.hidden_size
            self.decoder_hidden_size = self.language_encoder.config.hidden_size
        except AttributeError:
            # for Qwen3VL, its text config is separate
            self.feature_dim = self.language_encoder.config.text_config.hidden_size
            self.decoder_hidden_size = self.language_encoder.config.text_config.hidden_size

        self.modality_dims = modality_dims or {}
        self.modality_order = modality_order
        self.modality_order_list = modality_order.split('|')

        self.max_objects = max_objects
        self.verbose = verbose

        # SGLang related attributes
        self.use_sglang = use_sglang
        self.keep_pids = []
        self.sglang_model_path = sglang_model_path or getattr(language_model, "name_or_path", None) #?
        self.sglang_lora_paths = sglang_lora_paths
        self.sglang_gpu_id = dist.get_rank() if dist.is_initialized() else 0 # int(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0])
        # logger.info(f"Distributed initialized: {dist.is_initialized()}")
        # if dist.is_initialized():
        #     logger.info(f"Using SGLang GPU ID: {self.sglang_gpu_id} (rank {dist.get_rank()})")
        # else:
        #     logger.info(f"Using SGLang GPU ID: {self.sglang_gpu_id}, CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', '<not set>')}")
        self.sglang_port = sglang_port
        self.sglang_log_level = sglang_log_level

        self.sglang_engine = None
        
        # Object to text projection
        if object_embedding_type == "simple":
            self.object_proj = ObjectEmbeddingSimple(object_feature_dim, self.decoder_hidden_size)
        elif object_embedding_type == "fourier":
            self.object_proj = ObjectFourierEmbedding(object_feature_dim, self.decoder_hidden_size, num_fourier_features=256)
        elif object_embedding_type == "discrete_location":
            self.object_proj = ObjectDiscreteLocationEmbedding(object_feature_dim, self.decoder_hidden_size,
                                                                num_bins=discrete_location_bins, decay_kernel=discrete_location_decay_kernel,
                                                                bin_range=discrete_location_bin_range, kernel_size=discrete_location_decay_kernel_size,
                                                                separate_location_embedding=separate_location_embedding)
        elif object_embedding_type == "discrete_location_separate":
            self.object_proj = ObjectDiscreteLocationEmbeddingSeparate(
                modality_dims=self.modality_dims,
                modality_order=self.modality_order,
                decoder_hidden_size=self.decoder_hidden_size,
                location_modality_name="location",
                num_bins=discrete_location_bins, 
                decay_kernel=discrete_location_decay_kernel,
                bin_range=discrete_location_bin_range, 
                kernel_size=discrete_location_decay_kernel_size,
            )
        else:
            raise ValueError(f"Unknown encoder type: {object_embedding_type}")
        
        # Regression head for object grounding
        self.reg_head = nn.Sequential(
            nn.Linear(self.decoder_hidden_size, 256),
            nn.Dropout(0.15),
            nn.GELU(),
            nn.Linear(256, 6) # 输出物体的位置信息 xyzhwl
        )
        self.coeff_grounding_loss = coeff_grounding_loss
        
        self.image_proj = None
        if image_embedding_dim is not None:
            # we pre-encode image into image embeddings externally
            # self.image_model, _, self.image_preprocess_fn = open_clip.create_model_and_transforms(
            #     image_encoder_model_id, pretrained=image_encoder_pretrained,
            #     precision="bf16"
            # )
            # if not image_encoder_trainable:
            #     for param in self.image_model.parameters():
            #         param.requires_grad = False
            self.image_proj = ObjectEmbeddingSimple(image_embedding_dim, self.decoder_hidden_size)
            logger.info(f"Initialized image projector with dim {image_embedding_dim}")
        
        # Initialize SGLang if needed
        if self.use_sglang:
            self._init_sglang_engine(delete_model_from_cpu=delete_model_from_cpu)
            
        # turn all parameters to given dtype (except for the language model), required for FSDP
        # for name, param in self.named_parameters():
        #     if "language_encoder" not in name:
        #         param.data = param.data.to(dtype)

    def activate_sglang(self):
        self.use_sglang = True
        self._init_sglang_engine(load_non_lm_parameters=False)

    def deactivate_sglang(self):
        self.use_sglang = False
        self.shutdown_sglang()

    def shutdown_sglang(self):
        """Shutdown SGLang engine."""
        if self.sglang_engine is not None:
            # self.sglang_engine.shutdown(skip_pids=self.keep_pids)
            kill_process_tree(os.getpid(), include_parent=False, skip_pids=self.keep_pids)
            self.sglang_engine = None

    def _init_sglang_engine(self, load_non_lm_parameters: bool = True, delete_model_from_cpu: bool = False):
        """
        Initialize or reinitialize the SGLang engine
        Calling this function will shutdown the existing engine and 
        create a new one, and reload LoRAs if provided.
        """
        import sglang as sgl
        # Offload HF LLM to CPU, but keep the word embedding layer on GPU
        # FIXME: if we want to lower the memory usage, we can completely `del` the model
        embedding_device = get_module_device(self.language_encoder.get_input_embeddings())
        # self.language_encoder = self.language_encoder.to("cpu")
        self.language_encoder.to("cpu")
        self.language_encoder.get_input_embeddings().to(embedding_device)

        if delete_model_from_cpu:
            # remove module on CPU to save memory
            logger.info("Deleting language encoder weights from CPU to save memory...")
            embedding_params = set(self.language_encoder.get_input_embeddings().parameters())
            for param in self.language_encoder.parameters():
                if param in embedding_params:
                    continue
                # Replace the data with a small tensor to free memory
                param.data = torch.empty(0, dtype=param.dtype, device="cpu")
            
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # FIXME: putting this after SGLang engine init makes it hang
        #   even the access to self.object_proj is hanging!!!
        if self.sglang_lora_paths and load_non_lm_parameters:
            self.load_non_lm_parameters(self.sglang_lora_paths[0])

        torch.cuda.empty_cache()

        # show device placement
        # for name, param in self.named_parameters():
        #     print(f"Parameter {name} is on device {param.device}")

        # Shutdown existing engine if it exists
        if self.sglang_engine is not None:
            # self.sglang_engine.shutdown()
            self.shutdown_sglang()

        logger.info(f"Creating SGLang engine with model {self.sglang_model_path} on GPU {self.sglang_gpu_id}")
        
        # Initialize new engine
        port = self.sglang_port or find_free_port()
        logger.info(f"Using SGLang port: {port}")
        # logger.info(os.environ)
        # with clear_environment():
        # with patch_environment(
        #     LOCAL_RANK="", RANK="", WORLD_SIZE="", LOCAL_WORLD_SIZE="",
        #     MASTER_ADDR="", MASTER_PORT="",
        # ):


        # Record subprocessed not to be killed by engine.shutdown()
        self.keep_pids = get_my_all_child_pids()
        logger.info(f"Found {len(self.keep_pids)} subprocesses: {self.keep_pids}")

        # NOTE: need to bypass mm processor for Qwen3-VL
        # in `sglang/srt/managers/tokenizer_manager.py`
        # ```
        # if server_args.skip_tokenizer_init:
        #     self.mm_processor = None
        # else:
        #     self.mm_processor = get_mm_processor(
        #         self.model_config.hf_config, server_args, _processor, transport_mode
        #     )
        # ```
        # NOTE: need to tweak SGL's MLLM methods to accept input_embeds
        # in `sglang/srt/models/qwen3_vl.py`
        # add `input_embeds`` to `forward method``
        # in `sglang/srt/managers/mm_utils.py`
        # add `input_embeds`` to `general_mm_embed_routine`` method
        self.sglang_engine = sgl.Engine(
            model_path=self.sglang_model_path,
            disable_radix_cache=True, # embedding input can't use radix attention
            skip_tokenizer_init=True,  # We already have a tokenizer
            disable_cuda_graph=True if not NEW_SGLANG else False,
            lora_paths=self.sglang_lora_paths if self.sglang_lora_paths else None,
            random_seed=42,
            base_gpu_id=self.sglang_gpu_id,
            log_level=self.sglang_log_level,
            port=port,
            mem_fraction_static=0.7,
            # max_running_requests=256,
            # A100-40G, 4B: 300?, 8B: 64
            max_running_requests=192, 
            cuda_graph_max_bs=192,
            schedule_conservativeness=0.3, 
                # 0.7 * this will be initial new_token_ratio, and will decrease to 0.14 over steps, 600 step and will be lowest, but will then trigger
                # KV memory pool full and retract request, and will lead to error
                # Fix: in sglang/srt/managers/schedule_batch.py, in ScheduleBatch.filter_batch
                #   also filter self.input_embeds if present
            # attention_backend="triton",
            # json_model_override_args=json.dumps({"max_position_embeddings": 131_072}), 
            chunked_prefill_size=65536,
            max_prefill_tokens=65536,
            max_total_tokens=65536,
            enable_torch_compile=not NEW_SGLANG,
            torch_compile_max_bs=256,
            # NOTE: seems small max_position_embeddings leads to position overflow in Qwen2 - they only init 32768 RoPE precomputed embeddings
            # dist_init_addr=f"tcp://127.0.0.1:{port + 42 if port <= 60000 else port - 43}",
            grammar_backend=None,
            disable_fast_image_processor=True,
        )

        # logger.info(self)

        # load non-lm parameters from lora path, too
        assert len(self.sglang_lora_paths) == 1, "Providing multiple LoRA paths but only the first one will be used"
        
        logger.info(f"Initialized SGLang engine with model {self.sglang_model_path}")
        if self.sglang_lora_paths:
            logger.info(f"Using LoRA paths: {self.sglang_lora_paths}")

    def _non_restart_reload_sglang_weights(self, load_non_lm_parameters: bool = True):
        assert self.sglang_engine is not None, "SGLang engine is not initialized"
        assert self.sglang_lora_paths is not None and len(self.sglang_lora_paths) > 0, "No LoRA paths provided"

        logger.info(f"Reloading SGLang engine weights from {self.sglang_lora_paths[0]}")

        # Offload HF LLM to CPU, but keep the word embedding layer on GPU
        # FIXME: if we want to lower the memory usage, we can completely `del` the model
        embedding_device = get_module_device(self.language_encoder.get_input_embeddings())
        # self.language_encoder = self.language_encoder.to("cpu")
        self.language_encoder.to("cpu")
        self.language_encoder.get_input_embeddings().to(embedding_device)

        # FIXME: putting this after SGLang engine init makes it hang
        #   even the access to self.object_proj is hanging!!!
        if self.sglang_lora_paths and load_non_lm_parameters:
            self.load_non_lm_parameters(self.sglang_lora_paths[0], device="cpu")

        torch.cuda.empty_cache()

        self.sglang_engine.unload_lora_adapter(lora_name=self.sglang_lora_paths[0])
        self.sglang_engine.load_lora_adapter(lora_path=self.sglang_lora_paths[0], lora_name=self.sglang_lora_paths[0])

        logger.info(f"Reloaded LoRA weights into SGLang engine from {self.sglang_lora_paths[0]}")
    def get_token_ranges_for_objects(
        self, 
        input_ids: torch.Tensor, 
        targets: torch.Tensor, 
        include_brackets: bool = True  # <--- New Option
    ) -> List[Tuple[int, int]]:
        """
        Identify the start and end indices of object descriptions enclosed in brackets.
        Robust to tokenizer merging (e.g., ' [', ']\n', '][', '[123]').
        
        Args:
            input_ids: The token ids of the response sequence.
            targets: The ground truth regression targets [N_objects, 6]. 
            include_brackets: If False, attempts to exclude the tokens containing the brackets 
                              themselves from the range, provided they don't contain other content.
        
        Returns:
            List of (start_idx, end_idx) tuples relative to the input_ids.
        """
        ranges = []
        current_start = -1
        
        # Move to CPU and convert to list once for speed
        tokens = input_ids.tolist()
        
        # Cache specific bracket strings if simple matching is desired, 
        # but full decoding is safest for things like LLaMA's byte fallback.
        token_strs = [self.tokenizer.decode([tid], skip_special_tokens=False) for tid in tokens]
        
        # Helper to check if a token is "pure structure" (only brackets/whitespace)
        # This prevents us from dropping '[100' if we only wanted to drop '['
        def is_pure_bracket(s: str) -> bool:
            # Remove brackets and whitespace, check if anything remains
            return len(s.replace('[', '').replace(']', '').strip()) == 0

        for i, token_id in enumerate(tokens):
            token_str = token_strs[i]
            
            # We iterate through the string characters to handle order correctly.
            # Case 1: "][" -> Close previous, then Open new
            # Case 2: "[]" -> Open new, then Close new
            for char in token_str:
                if char == '[':
                    current_start = i
                
                elif char == ']':
                    if current_start != -1:
                        # Found a closing bracket for an active opening
                        # Initial raw range inclusive of the current token 'i'
                        start_idx = current_start
                        end_idx = i + 1
                        
                        # --- Post-processing for exclude logic ---
                        if not include_brackets:
                            # 1. Try to shrink from the left (Start Token)
                            # Only advance start if the token at start_idx is JUST brackets/whitespace
                            if start_idx < end_idx and is_pure_bracket(token_strs[start_idx]):
                                start_idx += 1
                            
                            # 2. Try to shrink from the right (End Token)
                            # Note: end_idx is exclusive, so we check end_idx - 1
                            # Only retreat end if the token at end_idx-1 is JUST brackets/whitespace
                            if start_idx < end_idx and is_pure_bracket(token_strs[end_idx - 1]):
                                end_idx -= 1
                        
                        # Only append if we still have a valid range (avoid empty ranges if [] was empty)
                        if start_idx < end_idx:
                            ranges.append((start_idx, end_idx))
                        
                        current_start = -1
                        
                        # Early exit if we matched enough objects
                        if len(ranges) == len(targets):
                            return ranges
        
        return ranges

    def forward_prepare_image_features(self, image_features: torch.Tensor) -> torch.Tensor:
        # images features are of shape [B, N_views, N_patches, image_embedding_dim], 
        # N_patches can be 1 (global image embedding) or more (patch embeddings, full or adaptively pooled)

        # cast to image_proj dtype
        image_proj_dtype = next(self.image_proj.parameters()).dtype
        if image_features.dtype != image_proj_dtype:
            image_features = image_features.to(image_proj_dtype)
        
        # perform normalize first
        image_features = torch.nn.functional.normalize(image_features, dim=-1)
        image_features = self.image_proj(image_features)

        if image_features.dtype != self.language_encoder.get_input_embeddings().weight.dtype:
            image_features = image_features.to(self.language_encoder.get_input_embeddings().weight.dtype)

        return image_features

    def forward_prepare_object_features(self, object_features: torch.Tensor) -> torch.Tensor | Dict[str, torch.Tensor]:
        # logger.debug(f"Input device: {object_features.device}")
        # logger.debug(f"Weight device: {next(self.object_proj.parameters()).device}")

        object_features = self.object_proj(object_features)

        if isinstance(object_features, torch.Tensor):
            dtype = object_features.dtype
        elif isinstance(object_features, dict):
            dtype = next(iter(object_features.values())).dtype

        lm_dtype = self.language_encoder.get_input_embeddings().weight.dtype
        
        # cast dtype to lm's dtype, if needed
        if dtype != lm_dtype:
            if "separate" in self.object_proj.__class__.__name__.lower():
                # object_features is a dict, cast each modality
                for modality_name in object_features:
                    object_features[modality_name] = object_features[modality_name].to(lm_dtype)
            else:
                # object_features is a tensor, cast it
                object_features = object_features.to(lm_dtype)

        return object_features
    
    def forward_prepare_object_features_multi(self, object_set_embeds: List[torch.Tensor]) -> List[torch.Tensor] | Dict[str, List[torch.Tensor]]:
        object_features = [self.forward_prepare_object_features(obj) for obj in object_set_embeds]

        # if is dict (separate location embedding), convert to dict
        if isinstance(object_features[0], dict):
            # check if all dicts have the same keys
            keys = object_features[0].keys()
            for obj in object_features[1:]:
                if set(obj.keys()) != set(keys):
                    raise ValueError("Not all object features have the same keys.")
            
            # convert to dict of lists
            object_features_dict = {key: [obj[key] for obj in object_features] for key in keys}
            return object_features_dict
        
        # else, return list of tensors
        return object_features

    def gather_input_embeds_for_lm(self,
                                prompt: str,
                                object_set_embeds: Union[List[torch.Tensor], Dict[str, List[torch.Tensor]]],
                                image_embeds: Optional[torch.Tensor] = None,
                                add_special_tokens: bool = True,
                                ) -> Dict[str, torch.Tensor]:
        """
        Prepares inputs_embeds and attention_mask for a language model,
        interleaving tokenized text with object embeddings based on |object_set| placeholders.

        Args:
            prompt (str): The input prompt string with |object_set| placeholders.
                          e.g., "Here are |object_set| objects and |object_set| walls."
            object_set_embeds (Union[List[torch.Tensor], Dict[str, List[torch.Tensor]]]):
                - If List[torch.Tensor]: A list of tensors. Each tensor corresponds to one
                  |object_set| placeholder and has shape [N_objects, H_lm].
                  These N_objects embeddings are concatenated for the placeholder.
                - If Dict[str, List[torch.Tensor]]: Keys are modality names (e.g., "id", "location").
                  Values are lists of tensors. object_set_embeds[mod_name][j] is the tensor
                  [N_objects, H_lm] for the j-th placeholder and modality 'mod_name'.
                  For each object within a placeholder, its modality embeddings (ordered by
                  self.modality_order_list) are retrieved from the dict and concatenated
                  to form a sequence of M_modalities tokens (where M is len(self.modality_order_list)).
                  Thus, N_objects become N_objects * M_modalities tokens for that placeholder.
                  H_lm is the language model's embedding dimension.

        Returns:
            Dict[str, torch.Tensor]: A dictionary with keys "inputs_embeds" and "attention_mask"
                                     containing the prepared tensors for the language model.
        """
        
        # device = self.language_encoder.device
        lm_input_embedding_layer = self.language_encoder.get_input_embeddings()
        device = next(lm_input_embedding_layer.parameters()).device # main LLM maybe put to cpu with SGLang, only embedding layer on GPU
        # Assuming H_lm (language model hidden size) can be inferred from the embedding layer
        # For empty object sets, we might need this explicitly if no other embeddings exist.
        # Example: lm_hidden_size = lm_input_embedding_layer.weight.shape[-1]

        all_input_embeds_parts = []
        all_attention_mask_parts = []

        if image_embeds is not None and "|image_set|" not in prompt:
            # 只在没有占位符时才添加
            # FIXME: we put image placeholder here, but ideally it should be in the prompt -- in the dataset
            prompt = "Those are the images of the scene: |image_set|" + prompt

        # Handle cases where no object processing is needed or no placeholders exist
        if self.no_object_in_language_model or "|object_set|" not in prompt:
            tokenized_full_prompt = self.tokenizer(
                prompt, return_tensors="pt", padding=False, add_special_tokens=add_special_tokens,
            )
            # Ensure tokenized output is not empty (e.g. tokenizer for "" with add_special_tokens=True)
            if tokenized_full_prompt.input_ids.shape[1] > 0:
                all_input_embeds_parts.append(
                    lm_input_embedding_layer(tokenized_full_prompt.input_ids.to(device))
                )
                all_attention_mask_parts.append(tokenized_full_prompt.attention_mask.to(device))
            else: # Fallback for truly empty prompt after tokenization
                pass # Let cat handle empty list if this rare case occurs, or add explicit BOS/EOS later if needed.

        else: # Process prompt with |object_set| placeholders
            # Split prompt by placeholder, keeping placeholder as a delimiter
            # prompt_elements = re.split(r'(\|object\_set\|)', prompt)
            prompt_elements = re.split(r'(\|(?:object|image)_set\|)', prompt) # also split |image_set|
            
            num_placeholders_in_prompt = prompt.count("|object_set|")
            num_image_placeholders = prompt.count("|image_set|")
            num_actual_image_embeds = 1 if image_embeds is not None else 0

            # Validate counts of placeholders vs provided object sets
            if isinstance(object_set_embeds, list):
                num_actual_object_sets = len(object_set_embeds)
            elif isinstance(object_set_embeds, dict):
                if not object_set_embeds: # Empty dictionary
                    num_actual_object_sets = 0
                else:
                    # All lists in the dict should have the same length, matching num_placeholders
                    try:
                        first_modality_key = next(iter(object_set_embeds.keys()))
                        num_actual_object_sets = len(object_set_embeds[first_modality_key])
                        for mod_name in object_set_embeds:
                            if len(object_set_embeds[mod_name]) != num_actual_object_sets:
                                raise ValueError(
                                    "Mismatch in lengths of embedding lists within the object_set_embeds dictionary."
                                )
                    except StopIteration: # Should not happen if `not object_set_embeds` is false
                        num_actual_object_sets = 0
            else:
                raise TypeError("object_set_embeds must be a List or Dict.")

            if num_placeholders_in_prompt != num_actual_object_sets:
                raise ValueError(
                    f"Mismatch between |object_set| placeholders in prompt ({num_placeholders_in_prompt}) "
                    f"and number of object sets provided ({num_actual_object_sets}). Prompt: '{prompt}'"
                )
            
            if num_image_placeholders != num_actual_image_embeds:
                raise ValueError(
                    f"Mismatch between |image_set| placeholders ({num_image_placeholders}) "
                    f"and image embeds provided ({num_actual_image_embeds})"
                )

            current_object_set_idx = 0
            processed_first_substantive_element = False # only for first text, we need to add BOS tokens (add_special_tokens)

            for part_content in prompt_elements:
                if not part_content: # Skip empty strings that re.split might produce
                    continue

                if part_content == "|object_set|":
                    object_sequence_for_lm = None # Placeholder for this object set's final embeddings

                    if isinstance(object_set_embeds, dict):
                        if not self.modality_order_list:
                            raise ValueError("self.modality_order_list is empty, but object_set_embeds is a Dict. Cannot determine order.")
                        
                        modal_embeds_for_this_placeholder = []
                        num_objects_this_set = -1

                        for mod_name in self.modality_order_list:
                            if mod_name not in object_set_embeds:
                                raise ValueError(f"Modality '{mod_name}' from self.modality_order_list not found in object_set_embeds keys.")
                            if current_object_set_idx >= len(object_set_embeds[mod_name]):
                                raise ValueError(
                                    f"Not enough embedding sets for placeholder index {current_object_set_idx} "
                                    f"in modality '{mod_name}'. List length: {len(object_set_embeds[mod_name])}."
                                )
                            
                            mod_tensor = object_set_embeds[mod_name][current_object_set_idx].to(device)

                            if num_objects_this_set == -1:
                                num_objects_this_set = mod_tensor.shape[0]
                            elif mod_tensor.shape[0] != num_objects_this_set:
                                raise ValueError(
                                    f"Mismatch in N_objects for placeholder {current_object_set_idx} across modalities. "
                                    f"Expected {num_objects_this_set}, got {mod_tensor.shape[0]} for modality '{mod_name}'."
                                )
                            # Optional: Check H_lm consistency
                            # if mod_tensor.shape[-1] != lm_hidden_size: raise ValueError(...)
                            modal_embeds_for_this_placeholder.append(mod_tensor)
                        
                        if num_objects_this_set > 0:
                            # Stack: [N_objects, M_modalities, H_lm]
                            stacked_embeds = torch.stack(modal_embeds_for_this_placeholder, dim=1)
                            # Reshape to [N_objects * M_modalities, H_lm]
                            object_sequence_for_lm = stacked_embeds.reshape(num_objects_this_set * len(self.modality_order_list), -1)
                        elif num_objects_this_set == 0: # No objects in this specific set
                            lm_hidden_size = lm_input_embedding_layer.weight.shape[-1]
                            object_sequence_for_lm = torch.empty((0, lm_hidden_size), device=device, dtype=lm_input_embedding_layer.weight.dtype)
                        # else num_objects_this_set == -1 (e.g. modality_order_list was empty but object_set_embeds dict was not, caught earlier)

                    elif isinstance(object_set_embeds, list):
                        object_sequence_for_lm = object_set_embeds[current_object_set_idx].to(device)
                    
                    # Add batch dimension if needed, making it [1, L_obj_sequence, H_lm]
                    if object_sequence_for_lm is not None:
                        if len(object_sequence_for_lm.shape) == 2: # [L_obj_seq, H_lm]
                            object_sequence_for_lm = object_sequence_for_lm.unsqueeze(0)
                        # If object_sequence_for_lm is already [1, L_obj_seq, H_lm] or [1,0,H_lm], shape is fine.
                        
                        # Only add if there are actual embeddings (non-zero number of elements)
                        if object_sequence_for_lm.nelement() > 0:
                            all_input_embeds_parts.append(object_sequence_for_lm)
                            obj_attention_mask = torch.ones(object_sequence_for_lm.shape[:-1], device=device, dtype=torch.long) # [1, L_obj_seq]
                            all_attention_mask_parts.append(obj_attention_mask)
                            processed_first_substantive_element = True
                    
                    current_object_set_idx += 1
                
                elif part_content == "|image_set|":
                    assert image_embeds is not None, "|image_set| placeholder found but no image_embeds provided."
                    # image_embeds shall be [N_views, N_patches, h]
                    image_sequence_for_lm = image_embeds.to(device)
                    if len(image_sequence_for_lm.shape) == 3: # [N_views, N_patches, h]
                        image_sequence_for_lm = image_sequence_for_lm.reshape(-1, image_sequence_for_lm.shape[-1]).unsqueeze(0) # [1, N_views * N_patches, h]
                    # If image_sequence_for_lm is already [1, L_image_seq, h], shape is fine.
                    if image_sequence_for_lm.nelement() > 0:
                        all_input_embeds_parts.append(image_sequence_for_lm)
                        img_attention_mask = torch.ones(image_sequence_for_lm.shape[:-1], device=device, dtype=torch.long) # [1, L_image_seq]
                        all_attention_mask_parts.append(img_attention_mask)
                        processed_first_substantive_element = True


                else: # This part_content is text
                    apply_special_tokens = not processed_first_substantive_element
                    if not add_special_tokens:
                        apply_special_tokens = False # if add_special_tokens is False, we never apply special tokens
                    
                    # Skip empty or whitespace-only text parts unless they are the very first part (to get BOS)
                    if not part_content.strip() and not apply_special_tokens:
                        continue
                        
                    tokenized_text = self.tokenizer(
                        part_content, return_tensors="pt", padding=False, 
                        add_special_tokens=apply_special_tokens # BOS for first, no special for others
                    )
                    
                    if tokenized_text.input_ids.shape[1] > 0: # If tokenizer produced some tokens
                        all_input_embeds_parts.append(
                            lm_input_embedding_layer(tokenized_text.input_ids.to(device))
                        )
                        all_attention_mask_parts.append(tokenized_text.attention_mask.to(device))
                        processed_first_substantive_element = True
            
            # Fallback if all parts resulted in no embeddings (e.g., prompt=" ", or only empty object sets)
            if not all_input_embeds_parts:
                # Generate a minimal input, e.g., BOS token then EOS token, or just BOS.
                # Behavior depends on tokenizer and model expectations.
                # A common practice for an "empty but valid" sequence.
                tokenized_empty = self.tokenizer("", return_tensors="pt", add_special_tokens=True)
                if tokenized_empty.input_ids.shape[1] > 0:
                    all_input_embeds_parts.append(lm_input_embedding_layer(tokenized_empty.input_ids.to(device)))
                    all_attention_mask_parts.append(tokenized_empty.attention_mask.to(device))
                else:
                    # This case should be extremely rare (tokenizer yields nothing for "" with special tokens)
                    # If it occurs, model will receive truly empty input, which might error.
                    # Consider raising an error or manually creating a [1,0,H] tensor.
                    lm_hidden_size = lm_input_embedding_layer.weight.shape[-1]
                    dtype = lm_input_embedding_layer.weight.dtype
                    all_input_embeds_parts.append(torch.empty((1,0,lm_hidden_size), device=device, dtype=dtype))
                    all_attention_mask_parts.append(torch.empty((1,0), device=device, dtype=torch.long))


        # Concatenate all collected parts
        final_input_embeds = torch.cat(all_input_embeds_parts, dim=1) # Shape: [1, L_total, H_lm]
        final_attention_mask = torch.cat(all_attention_mask_parts, dim=1) # Shape: [1, L_total]

        return {
            "inputs_embeds": final_input_embeds,
            "attention_mask": final_attention_mask,
        }

    
    def encode_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Encode input ids to input embeddings.
        """
        # get the input embeddings from the language model
        input_embedding = self.language_encoder.get_input_embeddings()
        
        # encode the input ids
        input_embeds = input_embedding(input_ids)

        return input_embeds
    
    def build_prompt_default(self, instruction: str, response: str=None) -> str:
        prompt = f"Existing objects in 3D scene: |object_set| {instruction}"
        return prompt
    
    def pad_left_embeds_and_attention_mask(self, inputs_embeds: List[torch.Tensor], attention_masks: List[torch.Tensor], start_token_indexes: Optional[List[int]]=None):
        """
        Pads the inputs_embeds and attention_masks to the left, so that all inputs_embeds have the same length.
        Further, updates start_token_indexes to account for the padding.
        """

        max_length = max([input_embed.shape[-2] for input_embed in inputs_embeds])
        for i, (input_embed, attention_mask) in enumerate(zip(inputs_embeds, attention_masks)):
            # pad to left, so the input is at the end
            pad_length = max_length - input_embed.shape[-2]
            inputs_embeds[i] = F.pad(input_embed, (0, 0, pad_length, 0), value=0)
            attention_masks[i] = F.pad(attention_mask, (pad_length, 0), value=0)

            # also add to start_token_indexes
            if start_token_indexes is not None:
                start_token_indexes[i] += pad_length

        return inputs_embeds, attention_masks, start_token_indexes
    
    def forward(self, 
            instructions: Optional[List[str]]=None,
            object_set_embeds: Optional[List[List[torch.Tensor]]]=None,
            image_embeds: Optional[List[torch.Tensor]]=None,
            responses: Optional[List[str]]=None,
            inputs_embeds: Optional[torch.Tensor | List[torch.Tensor]]=None,
            attention_masks: Optional[torch.Tensor | List[torch.Tensor]]=None,
            response_ids: Optional[torch.Tensor | List[torch.Tensor]]=None,
            response_mask: Optional[torch.Tensor | List[torch.Tensor]]=None,
            loss_type: str = "sft", # 'sft', 'dft', 'psft'
            psft_epsilon: float = 0.2,
            grounding_targets: Optional[torch.Tensor]=None, # [B, N_objects, 6] 6: xyzhwl, N_objects: number of objects to predict
        ):
        """
        instruction: list of strings, each string is a prompt
        response: list of strings, each string is a response
        object_set_embeds: list of list of tensors, each tensor is [N_objects, h]

        --- for inferece:
        inputs_embeds: optional, if provided, will be used instead putting the instruction and object_set_embeds into the language model
        attention_masks: optional, if provided, will be used instead putting the instruction and object_set_embeds into the language model
        response_ids, response_mask: optional, if provided, will be used instead of the response
        """
        assert responses is None or response_ids is None, "Only one of responses and response_ids can be provided"

        if inputs_embeds is not None and attention_masks is not None:
            # if inputs_embeds and attention_masks are provided, use them directly
            input_embeds = [
                {
                    "inputs_embeds": inputs_embeds[i].unsqueeze(0), # [1, L, h]
                    "attention_mask": attention_masks[i].unsqueeze(0), # [1, L]
                } for i in range(len(inputs_embeds))
            ]

        else:
            # encode instructions and object set embeds
            # project input object features to LM space
            # object_set_embeds = [[self.forward_prepare_object_features(object_embed) for object_embed in object_set] for object_set in object_set_embeds]
            object_set_embeds = [self.forward_prepare_object_features_multi(object_set) for object_set in object_set_embeds]

            if image_embeds is not None:
                image_embeds = self.forward_prepare_image_features(torch.stack(image_embeds, dim=0)) # [B, N_views, N_patches, h]
            else:
                image_embeds = [None] * len(instructions)

            if self.verbose:
                logger.info(f"Instruction in forward: {instructions[0]}")
                logger.info(f"Object set embeds shape: {object_set_embeds[0][0].shape}")

            # gather input embeddings for LM
            # instructions = [self.build_prompt_default(instruction, response) for instruction, response in zip(instructions, responses)]
            input_embeds = [self.gather_input_embeds_for_lm(instruction, object_set_embed, image_embeds=image_embed) for instruction, object_set_embed, image_embed in zip(instructions, object_set_embeds, image_embeds)]

        start_token_indexes = [input_embed["inputs_embeds"].shape[-2] for input_embed in input_embeds]
        start_token_indexes = torch.tensor(start_token_indexes, device=self.language_encoder.device) # [B]

        # if response is provided, encode them and append to input_embeds
        if responses is not None or response_ids is not None:
            if response_ids is not None:
                # encode response ids
                response_embeds = self.encode_input_ids(response_ids)
                if response_mask is None:
                    response_mask = torch.ones_like(response_ids, dtype=torch.long)
                for i, (input_embed, response_embed) in enumerate(zip(input_embeds, response_embeds)):
                    input_embeds[i]["inputs_embeds"] = torch.cat([input_embed["inputs_embeds"], response_embed.unsqueeze(0)], dim=-2)
                    input_embeds[i]["attention_mask"] = torch.cat([input_embed["attention_mask"], response_mask[i].unsqueeze(0)], dim=-1)
            else:
                for i, (input_embed, response) in enumerate(zip(input_embeds, responses)):
                    response_embeds = self.gather_input_embeds_for_lm(response, []) 
                    # response_embeds = self.gather_input_embeds_for_lm(response, [], add_special_tokens=False) 
                    # NOTE: if change no BOS/EOS tokens for response, but this will introduce inference discrepancy for old models
                    #   change it to True if need to inference old models
                    # NOTE: Qwen2Tokenizer does not include any special tokens/BOS/EOS, so it is safe
                    # FIXME: will add BOS token, is this ok? 
                    # On the other hand, the response label in later code does not add BOS token, will it be ok?
                    input_embeds[i]["inputs_embeds"] = torch.cat([input_embed["inputs_embeds"], response_embeds["inputs_embeds"]], dim=-2)
                    input_embeds[i]["attention_mask"] = torch.cat([input_embed["attention_mask"], response_embeds["attention_mask"]], dim=-1)

        # logger.info(f"Input embeds shape: {[input_embed['inputs_embeds'].shape for input_embed in input_embeds]}")
        # logger.info(f"Attention masks shape: {[input_embed['attention_mask'].shape for input_embed in input_embeds]}")
        
        input_embeds, attention_masks, start_token_indexes = self.pad_left_embeds_and_attention_mask(
            [input_embed["inputs_embeds"] for input_embed in input_embeds],
            [input_embed["attention_mask"] for input_embed in input_embeds],
            start_token_indexes=start_token_indexes.tolist()
        )

        # logger.info(f"Input embeds shape: {[input_embed.shape for input_embed in input_embeds]}")
        # logger.info(f"Attention masks shape: {[attn_mask.shape for attn_mask in attention_masks]}")


        all_input_embeds = torch.cat(input_embeds, dim=0).detach() # this will lead to non-trainable input_embeds and embedding layer!
        all_attention_mask = torch.cat(attention_masks, dim=0)

        # create position ids: sum the attention mask
        # position_ids = None
        position_ids = torch.cumsum(all_attention_mask.long(), dim=-1) - 1
        position_ids.masked_fill_(all_attention_mask == 0, 1)

        # logger.info(f"all_input_embeds shape: {all_input_embeds.shape}, all_attention_mask shape: {all_attention_mask.shape}, position_ids shape: {position_ids.shape}")

        # forward pass through the language model
        # Calculate loss if responses are provided (training mode)
        if responses is not None:
            # Prepare labels for loss calculation - initialize with ignore index
            labels = torch.full_like(all_attention_mask, -100, dtype=torch.long)  # -100 is the ignore index
            
            # For each example in the batch
            for i, (start_idx, response) in enumerate(zip(start_token_indexes, responses)):
                # Tokenize the response
                response_tokens = self.tokenizer(response, return_tensors="pt", add_special_tokens=False).to(self.language_encoder.device)
                response_length = response_tokens.input_ids.shape[1]

                # Set the labels for the response part (for causal LM, we predict the next token)
                # So we set labels starting from start_idx to include the response tokens
                if start_idx + response_length <= labels.shape[1]:
                    # ic(start_idx, response_length, start_idx+response_length, labels.shape[1])
                    labels[i, start_idx:start_idx+response_length] = response_tokens.input_ids[0]
                else:
                    logger.warning(f"Response too long for the model: {response}, maybe a bug in the code?")
            
            all_input_embeds.requires_grad_(True) # ensure gradients flow back, so gradient_checkpointing can work
            # Forward pass through the language model with labels - therefore with loss

            # === [Modified Logic Starts Here] ===
            # 分支 1: 标准 SFT (使用原逻辑，最高效)
            if loss_type == 'sft':
                outputs = self.language_encoder(
                    inputs_embeds=all_input_embeds,
                    attention_mask=all_attention_mask,
                    position_ids=position_ids,
                    labels=labels, # 直接传入 labels
                    return_dict=True,
                )
                # outputs.loss 已经被模型内部计算好了

                # turn to MyObjectDict: the original outputs does not support assignment 
                # will be filtered out when passed out by damn transformers
                outputs = MyObjectDict(outputs.__dict__)

                # --- Auxiliary Regression Loss Calculation ---
                if self.coeff_grounding_loss > 0 and grounding_targets is not None:
                    last_hidden_state = outputs.hidden_states
                    
                    reg_losses = []
                    
                    # Iterate over batch dimension
                    for b_idx in range(len(responses)):
                        # 1. Get ranges relative to the RESPONSE string
                        # Note: response_tokens were generated inside the loop earlier. 
                        # We need to recover the response_input_ids for this specific batch item.
                        # Re-tokenization is expensive, let's grab it from labels if possible, 
                        # but labels have -100. Best to trust the `responses` string list passed in.
                        
                        curr_response_ids = self.tokenizer(responses[b_idx], return_tensors="pt", add_special_tokens=False)["input_ids"][0]
                        curr_targets = grounding_targets[b_idx] # [N_objs, 6]
                        
                        # Get relative ranges (e.g., [(5, 12), (20, 28)])
                        obj_ranges = self.get_token_ranges_for_objects(curr_response_ids, curr_targets, include_brackets=False)

                        if len(obj_ranges) == 0:
                            logger.warning(f"No object ranges found in response: {responses[b_idx]}")
                            continue
                        
                        # The start index of the response in the full sequence
                        # start_token_indexes was calculated before padding.
                        # We need the index in `all_input_embeds` (which is left-padded).
                        # pad_left_embeds_and_attention_mask updated start_token_indexes, so it's correct.
                        seq_start_idx = start_token_indexes[b_idx] 
                        
                        # Calculate loss for each matched object
                        # zip ensures we only calculate for objects that exist in both text and GT
                        for (r_start, r_end), gt_val in zip(obj_ranges, curr_targets):
                            # Map relative range to absolute range in the batch
                            abs_start = seq_start_idx + r_start
                            abs_end = seq_start_idx + r_end
                            
                            # Sanity check for bounds
                            if abs_end > last_hidden_state.shape[1]:
                                logger.warning(f"Regression range out of bounds: {abs_end} > {last_hidden_state.shape[1]}")
                                continue

                            # 2. Extract Hidden States for this object span [...]
                            # Shape: [Span_Len, Hidden_Dim]
                            obj_span_hidden = last_hidden_state[b_idx, abs_start:abs_end, :]
                            
                            # 3. Mean Pooling
                            # Shape: [Hidden_Dim]
                            obj_repr = obj_span_hidden.mean(dim=0)

                            # upcast to reg_head's dtype if needed
                            if obj_repr.dtype != next(self.reg_head.parameters()).dtype:
                                obj_repr = obj_repr.to(next(self.reg_head.parameters()).dtype)
                            
                            # 4. Regression Prediction
                            pred_coords = self.reg_head(obj_repr) # [6]
                            
                            # 5. MSE Loss (Grounding Loss)
                            # Ensure gt_val is on correct device/dtype
                            gt_val = gt_val.to(pred_coords.device).to(pred_coords.dtype)
                            
                            # We compute loss per object
                            reg_losses.append(F.mse_loss(pred_coords, gt_val))

                    # logger.warning(f"Number of regression losses computed: {len(reg_losses)}")
                    
                    if reg_losses:
                        # Stack and mean
                        grounding_loss = torch.stack(reg_losses).mean()
                        if self.verbose:
                            logger.warning(f"Grounding Loss: {grounding_loss.item()}")
                            
                        # Combine with LM loss
                        # outputs is a MyObjectDict wrapper or similar, or we just overwrite outputs.loss
                        # Since outputs is returned, we need to modify the loss inside it
                        outputs.loss = outputs.loss + self.coeff_grounding_loss * grounding_loss
                        
                        # Optional: Store the pure regression loss for logging
                        if isinstance(outputs, (dict, MyObjectDict)):
                            outputs["grounding_loss"] = grounding_loss
                        else:
                            # If it's a HF output object, we can monkey-patch an attribute
                            outputs.grounding_loss = grounding_loss
            
            # 分支 2: DFT / PSFT (需要 Logits 进行自定义计算)
            else:
                # 传入 labels=None 以获取 logits
                outputs = self.language_encoder(
                    inputs_embeds=all_input_embeds,
                    attention_mask=all_attention_mask,
                    position_ids=position_ids,
                    labels=None, 
                    return_dict=True,
                )
                
                # --- 手动计算基础 Loss ---
                logits = outputs.logits
                # Shift: Logits 预测下一个 token
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                
                # 计算每个 Token 的 log_prob (即 -CrossEntropy)
                # reduction='none' 拿到 [B, Seq_Len-1] 的 loss map
                loss_per_token = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)), 
                    shift_labels.view(-1), 
                    reduction='none'
                )
                loss_per_token = loss_per_token.view(shift_labels.shape)
                
                # 忽略 label 为 -100 的位置
                valid_mask = (shift_labels != -100).float()

                # turn to MyObjectDict: the original outputs does not support assignment 
                # will be filtered out when passed out by damn transformers
                outputs = MyObjectDict(outputs.__dict__)
                
                # --- 应用 DFT 或 PSFT ---
                if loss_type == 'dft':
                    # DFT: weight = stop_gradient(prob)
                    # loss_per_token 其实是 -log(p)
                    log_probs = -loss_per_token
                    probs = torch.exp(log_probs)
                    
                    # L_DFT = - sg(p) * log(p) = sg(p) * (-log(p))
                    # [cite: 817] Equation 9
                    dft_loss_map = probs.detach() * loss_per_token
                    
                    final_loss = (dft_loss_map * valid_mask).sum() / valid_mask.sum()
                    
                elif loss_type == 'psft':
                    # PSFT: Trust Region Clipping
                    current_log_probs = -loss_per_token
                    
                    # 计算 Reference Log Probs
                    # 假设 self.ref_model 已经准备好且冻结
                    if hasattr(self, 'ref_model') and self.ref_model is not None:
                        with torch.no_grad():
                            ref_outputs = self.ref_model(
                                inputs_embeds=all_input_embeds,
                                attention_mask=all_attention_mask,
                                position_ids=position_ids,
                                return_dict=True
                            )
                            ref_shift_logits = ref_outputs.logits[..., :-1, :].contiguous()
                            
                            # 提取对应正确 token 的 log_prob
                            ref_log_probs_all = F.log_softmax(ref_shift_logits, dim=-1)
                            gather_labels = shift_labels.clone()
                            gather_labels[gather_labels == -100] = 0 # 防止 gather越界
                            ref_log_probs = torch.gather(ref_log_probs_all, -1, gather_labels.unsqueeze(-1)).squeeze(-1)
                            ref_log_probs = ref_log_probs * valid_mask # 保持数值纯净
                    else:
                        # Fallback 到 SFT
                        ref_log_probs = current_log_probs.detach()

                    # 计算 Ratio 和 Clip
                    # [cite: 79] Equation 5: min(ratio, clip(ratio))
                    ratio = torch.exp(current_log_probs - ref_log_probs)
                    clipped_ratio = torch.clamp(ratio, 1.0 - psft_epsilon, 1.0 + psft_epsilon)
                    
                    # 目标是最大化 objective，即最小化 -objective
                    psft_loss_map = -torch.min(ratio, clipped_ratio)
                    
                    final_loss = (psft_loss_map * valid_mask).sum() / valid_mask.sum()
                
                # 将手动计算的 Loss 赋值回 output
                outputs.loss = final_loss
            
        else:
            # Forward pass without labels (inference mode)
            all_input_embeds.requires_grad_(True)
            outputs = self.language_encoder(
                inputs_embeds=all_input_embeds,
                attention_mask=all_attention_mask,
                position_ids=position_ids,
                return_dict=True,
            )

        return outputs
    

    def generate(
        self,
        instructions: List[str],
        object_set_embeds: List[List[torch.Tensor]],
        image_embeds: Optional[List[torch.Tensor]] = None,
        max_length: int = 100,
        num_beams: int = 5,
        do_sample: bool = False,
        top_k: int = -1,
        top_p: float = 1,
        temperature: float = 0.1,
        use_static_cache: bool = False,
        return_dict: bool = False,
        lora_paths: Optional[List[str]] = None, # if None and sglang_engine is not None, use the engine's lora paths by default
        return_logprob: bool = False,
        prep_chunk_size: int = 32, # chunk size for forward preparation
        **generate_kwargs
    ):
        """
        Generate responses for the given instructions using the language model.
        
        Args:
            instructions: List of instruction strings
            object_set_embeds: List of lists of object embeddings
            max_length: Maximum length of generated text

            --- parameters for generation ---
            num_beams: Number of beams for beam search
            do_sample: Whether to use sampling for generation
            top_k: Top-k sampling parameter
            top_p: Top-p sampling parameter
            temperature: Sampling temperature
            prep_chunk_size: Processing chunk size during input embedding preparation to avoid OOM
            **generate_kwargs: Additional arguments for the generate method
            
        Returns:
            List of generated text responses
        """
        # project input object features to LM space
        # object_set_embeds = [[self.forward_prepare_object_features(object_embed) for object_embed in object_set] for object_set in object_set_embeds]
        object_set_embeds = [self.forward_prepare_object_features_multi(object_set) for object_set in object_set_embeds]

        if image_embeds is not None:
            image_embeds = self.forward_prepare_image_features(torch.stack(image_embeds, dim=0)) # [B, N_views, N_patches, h]
        else:
            image_embeds = [None] * len(instructions)

        # Prepare inputs for generation
        # instructions = [self.build_prompt_default(instruction) for instruction in instructions]
        # ic(instructions[0])
        if self.verbose:
            logger.info(f"Instruction in generation: {instructions[0]}")
            logger.info(f"Object set embeds shape: {object_set_embeds[0][0].shape}")

        inputs_embeds = None
        attention_masks = None
        inputs_embeds_list = None # For SGLang

        if self.use_sglang and self.sglang_engine is not None:
            # Chunked processing for SGLang to avoid OOM with large batches
            inputs_embeds_list = []
            
            # Temporary storage for return_dict if needed
            input_embeds_cpu = [] 
            
            total_size = len(instructions)
            for i in range(0, total_size, prep_chunk_size):
                chunk_instr = instructions[i:i+prep_chunk_size]
                chunk_objs = object_set_embeds[i:i+prep_chunk_size]
                chunk_imgs = image_embeds[i:i+prep_chunk_size]

                for instr, obj, img in zip(chunk_instr, chunk_objs, chunk_imgs):
                    res = self.gather_input_embeds_for_lm(instr, obj, image_embeds=img)
                    
                    # Convert to list for SGLang immediately and free GPU memory
                    inputs_embeds_list.append(res["inputs_embeds"].squeeze(0).detach().cpu().tolist())
                    
                    if return_dict:
                        input_embeds_cpu.append({
                            "inputs_embeds": res["inputs_embeds"].detach().cpu(), 
                            "attention_mask": res["attention_mask"].detach().cpu()
                        })
                
                torch.cuda.empty_cache()

            if return_dict:
                # Reconstruct padded batch on CPU if requested
                inputs_embeds_padded, attention_masks_padded, _ = self.pad_left_embeds_and_attention_mask(
                [x["inputs_embeds"] for x in input_embeds_cpu],
                [x["attention_mask"] for x in input_embeds_cpu]
                )
                inputs_embeds = torch.cat(inputs_embeds_padded, dim=0)
                attention_masks = torch.cat(attention_masks_padded, dim=0).long()
            
            batch_size = len(instructions)
            
        else:
            input_embeds = [self.gather_input_embeds_for_lm(instruction, object_set_embed, image_embeds=image_embed) 
                            for instruction, object_set_embed, image_embed in zip(instructions, object_set_embeds, image_embeds)]
            
            # used for return value -> provide full input info
            inputs_embeds, attention_masks, _ = self.pad_left_embeds_and_attention_mask(
                [input_embed["inputs_embeds"] for input_embed in input_embeds],
                [input_embed["attention_mask"] for input_embed in input_embeds],
            )
            inputs_embeds = torch.cat(inputs_embeds, dim=0) # [B, L, h]
            attention_masks = torch.cat(attention_masks, dim=0).long() # [B, L]
            
            batch_size = len(input_embeds)
            
        # logger.info(inputs_embeds.size()[0], attention_masks[:, -1].sum().item())
        # logger.warning(f"inputs_embeds bs: {inputs_embeds.size()[0]}, attention_masks with 1s at end: {attention_masks[:, -1].sum().item()}")
        # FIXME: test that here, the attention mask shall be all 1s.
        

        if self.use_sglang and self.sglang_engine is not None:
            # logger.info("Starting SGLang generation")
            # Use SGLang for generation
            sampling_params = {
                "temperature": temperature if do_sample else 0.0,
                "top_p": top_p if do_sample else 0.01,
                "top_k": top_k,
                "max_new_tokens": max_length,
                # "do_sample": do_sample,
                # "num_beams": num_beams,
            }
            
            # Add any additional parameters from generate_kwargs
            sampling_params.update(generate_kwargs)

            logger.critical(f"SGLang sampling parameters: {sampling_params}")
            
            
            # Convert inputs_embeds to list for SGLang (IF NOT ALREADY DONE)
            # inputs_embeds_list = [embeds.detach().cpu().tolist() for embeds in inputs_embeds]
            if inputs_embeds_list is None:
                 # Should not happen in new logic, but kept for compatibility flow if logic changes
                 assert input_embeds is not None
                 inputs_embeds_list = [embeds["inputs_embeds"].squeeze(0).detach().cpu().tolist() for embeds in input_embeds]

            # del inputs_embeds  # free memory

            # logger.info(f"Done making inputs_embeds list")

            # show the shapes
            # logger.info(f"Input embeds shape: {[embeds["inputs_embeds"].shape for embeds in input_embeds]}")
            
            # Generate with SGLang
            generate_time_begin = time.time()
            with torch.no_grad():
                # logger.info(f"Generating with SGLang engine, batch size: {batch_size}")

                outputs = self.sglang_engine.generate(
                    input_embeds=inputs_embeds_list,
                    sampling_params=sampling_params,
                    lora_path=[self.sglang_lora_paths[0]] * batch_size if self.sglang_lora_paths else None,
                    return_logprob=return_logprob,
                    logprob_start_len=-1,
                    top_logprobs_num=None,
                )


            
            
            generate_time = time.time() - generate_time_begin
            logger.info(f"SGLang generation time: {generate_time:.2f} seconds")

            # Process outputs
            generated_responses = []
            generated_ids = []
            generated_logprobs = [] # 用于存储 logprobs

            # if not isinstance(response, list):
            #     response = [response]
            for output in outputs:
                # SGLang returns output_ids which we need to decode
                generated_text = self.tokenizer.decode(output["output_ids"], skip_special_tokens=True)
                generated_responses.append(generated_text)
                generated_ids.append(output["output_ids"])

                # === 解析 Logprobs ===
                if return_logprob:
                    # SGLang output format: 
                    # output["meta_info"]["output_token_logprobs"] 
                    # 是一个 List[Tuple[logprob, token_id, decoded_text]] (结构随版本略有不同)
                    meta = output.get("meta_info", {})
                    token_logprobs = meta.get("output_token_logprobs", [])
                    generated_logprobs.append(token_logprobs)
                
            if self.verbose:
                if len(generated_responses) > 0:
                    logger.info(f"Generated response with SGLang: {generated_responses[0]}")
        else:
            with torch.no_grad():
                # Ensure HF generation has inputs_embeds
                assert inputs_embeds is not None and attention_masks is not None, "Input embeds missing for HF generation"

                old_use_cache = self.language_encoder.generation_config.use_cache
                old_cache_implementation = self.language_encoder.generation_config.cache_implementation

                self.language_encoder.generation_config.use_cache = True
                self.language_encoder.generation_config.cache_implementation = "dynamic"
                if use_static_cache:
                    self.language_encoder.generation_config.cache_implementation = "static"

                # if inputs_embeds and attention_masks are not on GPU, move them to GPU
                # assert it must be NOT on CPU
                if inputs_embeds.device.type != "cuda":
                    logger.warning(f"inputs_embeds and attention_masks are not on GPU but on {inputs_embeds.device.type}, moving them to GPU if available")
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    inputs_embeds = inputs_embeds.to(device)
                    attention_masks = attention_masks.to(device)

                outputs = self.language_encoder.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_masks,
                    max_new_tokens=max_length,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    num_beams=num_beams,
                    do_sample=do_sample,
                    top_k=top_k,
                    top_p=top_p,
                    temperature=temperature,
                    **generate_kwargs
                )

                self.language_encoder.generation_config.use_cache = old_use_cache
                self.language_encoder.generation_config.cache_implementation = old_cache_implementation
        
            # Process outputs
            generated_responses = []
            generated_ids = outputs
            generated_logprobs = []

            for i, generated_tokens in enumerate(outputs):
                # Decode the generated tokens
                # NOTE: generate() already only returns the generated tokens
                generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                
                # generated_text_with_special_tokens = self.tokenizer.decode(generated_tokens, skip_special_tokens=False)
                # logger.info(generated_text_with_special_tokens)
                # logger.info(repr(generated_text_with_special_tokens))
                # logger.info(f"Generated response: {generated_text}")
                generated_responses.append(generated_text)

        if return_dict:
            # return a dictionary with the generated responses and the outputs
            return {
                "responses": generated_responses,
                "completion_ids": generated_ids,
                "inputs_embeds": inputs_embeds,
                "attention_masks": attention_masks,
                "logprobs": generated_logprobs,
            }
        
        return generated_responses


    def save_non_lm_parameters(self, path: str, accelerator=None):
        logger.info(f"Saving non-LM parameters to {path}")

        if accelerator is not None:
            state_dict = accelerator.get_state_dict(self)
        else:
            state_dict = self.state_dict()
        # remove language model parameters
        state_dict = {k: v for k, v in state_dict.items() if "language_encoder" not in k}

        torch.save(state_dict, os.path.join(path, "non_lm_parameters.pth"))

    def load_non_lm_parameters(self, path: str, device=None):
        logger.info(f"Loading non-LM parameters from {path}")
        # logger.debug(self.object_proj)
        # logger.debug(list(self.object_proj.named_parameters()))
        object_proj_device = get_module_device(self.object_proj)
        # logger.debug(f"got object_proj_device: {object_proj_device}")
        state_dict = torch.load(os.path.join(path, "non_lm_parameters.pth"), map_location=object_proj_device if device is None else device)
        # logger.debug("reached here, start load_state_dict")
        message = self.load_state_dict(state_dict, strict=False)
        logger.info(f"Loaded non-LM parameters from {path}: {message}")
        del state_dict

    def save_pretrained(self, path: str, accelerator=None):
        self.save_non_lm_parameters(path, accelerator)

        if not isinstance(self.language_encoder, PeftModel):
            logger.warning("Saving non-PEFT language model parameters, might be too large")

        if accelerator is not None:
            self.language_encoder.save_pretrained(path, state_dict=accelerator.get_state_dict(unwrapped_model))
        else:
            self.language_encoder.save_pretrained(path)

    def has_image_support(self) -> bool:
        """Check if model supports image inputs"""
        return self.image_proj is not None
