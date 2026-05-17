import contextlib
import torch
import torch.distributed as dist
from typing import Optional, Union, List



# distributed communication functions
def all_gather_vlen(tensor: torch.Tensor, group=None, gather_dtype=True) -> list[torch.Tensor]:
    """Gather tensors with the same number of dimensions but different lengths."""
    world_size = dist.get_world_size(group=group)
    
    # Gather dtype information if needed
    if gather_dtype:
        dtype_str = str(tensor.dtype).split('.')[-1]
        dtype_list = [None] * world_size
        dist.all_gather_object(dtype_list, dtype_str, group=group)
        dtype_list = [getattr(torch, dt_str) for dt_str in dtype_list]
    else:
        # Use tensor's dtype for all outputs if not gathering dtype info
        dtype_list = [tensor.dtype] * world_size
    
    
    # Gather lengths
    shape = torch.as_tensor(tensor.shape, device=tensor.device)
    shapes = [torch.empty_like(shape) for _ in range(world_size)]
    dist.all_gather(shapes, shape, group=group)
    
    # Gather data with correct dtypes
    outputs = [
        torch.empty(*_shape, dtype=dtype_list[i], device=tensor.device)
        for i, _shape in enumerate(shapes)
    ]
    dist.all_gather(outputs, tensor.contiguous(), group=group)
    return outputs

def all_gather_vdim(tensor: torch.Tensor, group=None) -> list[torch.Tensor]:
    """Gather tensors with different number of dimensions."""
    world_size = dist.get_world_size(group=group)
    
    # Gather dtype information
    dtype_str = str(tensor.dtype).split('.')[-1]
    dtype_list = [None] * world_size
    dist.all_gather_object(dtype_list, dtype_str, group=group)
    dtype_list = [getattr(torch, dt_str) for dt_str in dtype_list]

    # Gather shapes (don't gather dtype for shape tensor)
    shapes = all_gather_vlen(
        torch.as_tensor(tensor.shape, device=tensor.device), 
        group=group,
        gather_dtype=False  # Shape tensor has consistent dtype
    )
    
    # Gather data with correct dtypes
    outputs = [
        torch.empty(*_shape, dtype=dtype_list[i], device=tensor.device)
        for i, _shape in enumerate(shapes)
    ]
    dist.all_gather(outputs, tensor.contiguous(), group=group)
    return outputs

# Alternative implementation using contextlib.contextmanager
@contextlib.contextmanager
def model_to_device(model, device, empty_cache=False):
    """
    Context manager for temporarily moving a model to a specific device.
    
    Args:
        model: PyTorch model to move
        device: Target device (e.g., 'cuda:0', 'cpu', torch.device object)
    """
    target_device = device if isinstance(device, torch.device) else torch.device(device)
    original_device = next(model.parameters()).device
    moved = False
    
    # Move to target device if needed
    if original_device != target_device:
        model.to(target_device)
        moved = True
        
    try:
        yield model
    finally:
        # Move back to original device if we moved it
        if moved:
            model.to(original_device)
            if empty_cache:
                # Clear cache if moved to GPU
                if original_device.type == 'cpu':
                    torch.cuda.empty_cache()

def show_gpu_memory(device_id: Optional[Union[int]] = None):
    """
    Print GPU memory usage for each device.
    """
    print("\n\n")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            if device_id is not None and (i != device_id):
                continue
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  Allocated: {torch.cuda.memory_allocated(i) / (1024 ** 2):.2f} MB")
            print(f"  Cached: {torch.cuda.memory_reserved(i) / (1024 ** 2):.2f} MB")
    else:
        print("No GPUs available.")

    print("\n\n")