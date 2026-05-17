from typing import List, Optional, Union, Callable
from datetime import datetime
import numpy as np
import torch
import torch.nn.functional as F

import accelerate

import logging
logger = logging.getLogger(__name__)


def print_once(message):
    if not hasattr(print_once, "printed"):
        print_once.printed = set()
    if message not in print_once.printed:
        print_once.printed.add(message)
        print(message)

def softmax_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2,
    reduction: str = "none",
) -> torch.Tensor:
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.

    Args:
        inputs (Tensor): A float tensor of arbitrary shape.
                The predictions for each example.
        targets (Tensor): A float tensor with the same shape as inputs. Stores the binary
                classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha (float): Weighting factor in range (0,1) to balance
                positive vs negative examples or -1 for ignore. Default: ``0.25``.
        gamma (float): Exponent of the modulating factor (1 - p_t) to
                balance easy vs hard examples. Default: ``2``.
        reduction (string): ``'none'`` | ``'mean'`` | ``'sum'``
                ``'none'``: No reduction will be applied to the output.
                ``'mean'``: The output will be averaged.
                ``'sum'``: The output will be summed. Default: ``'none'``.
    Returns:
        Loss tensor with the reduction option applied.
    """
    # Original implementation from https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/focal_loss.py

    # if not torch.jit.is_scripting() and not torch.jit.is_tracing():
    #     _log_api_usage_once(sigmoid_focal_loss)

    # replace sigmoid with softmax on two classes
    inputs = inputs[..., 1] - inputs[..., 0] # [B, 2] => [B]
    p = torch.sigmoid(inputs) 
    # p = torch.softmax(inputs, dim=-1)[..., 1] # inputs: [B, 2] => [B] 

    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    # Check reduction option and return loss accordingly
    if reduction == "none":
        pass
    elif reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
    else:
        raise ValueError(
            f"Invalid Value for arg 'reduction': '{reduction} \n Supported reduction modes: 'none', 'mean', 'sum'"
        )
    return loss


class Singleton(type):
    _instances = {}
    # we are going to redefine (override) what it means to "call" a class
    # as in ....  x = MyClass(1,2,3)
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            # we have not every built an instance before.  Build one now.
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        else:
            instance = cls._instances[cls]
            # here we are going to call the __init__ and maybe reinitialize.
            # if hasattr(cls, '__allow_reinitialization') and cls.__allow_reinitialization:
            if getattr(cls, '__allow_reinitialization', False):
                # if the class allows reinitialization, then do it
                instance.__init__(*args, **kwargs)  # call the init again
        return instance

class AverageMeter:
    def __init__(self, 
                 accelerator: Optional[accelerate.Accelerator] = None,
                 report_period: int = 10, 
                 print_fn: Optional[Callable] = print):
        """
        Initialize the AverageMeter with a configurable report period and print function
        
        Args:
            report_period (int): Number of updates before printing average values
            print_fn (Callable, optional): Custom print function, defaults to standard print
        """
        self.values = {}  # Dictionary to store lists of recent values for each name
        self.report_period = report_period
        self.accelerator = accelerator

        self.value_type_cache = {}

        def print_fn_new(*args, **kwargs):
            if self._need_print:
                print_fn(*args, **kwargs)

        self.print_fn = print_fn_new

    def _convert_to_float(self, value: Union[float, int, np.ndarray, torch.Tensor]) -> float:
        """
        Convert input to float, handling various input types
        
        Args:
            value: Input value to convert
        
        Returns:
            float: Converted value
        """
        if isinstance(value, (np.ndarray, torch.Tensor)):
            return float(value.item())
        return float(value)

    def _format_typed_float(self, value: float, type: str) -> str:
        if type == "amount":
            # .2f
            return f"{value:.2f}"

        elif type == "percent":
            return f"{100 * value:.2f}%"
        
        elif type == "integer":
            return f"{int(value)}"


    def _sync_scalar(self, accelerator, scalar):
        if self.accelerator is None:
            logger.warning("No accelerator found, returning scalar as is")
            return scalar
        
        # Convert to tensor and move to device, then gather and return mean
        scalar = torch.tensor(scalar).to(accelerator.device)
        return accelerator.gather(scalar).mean().item()

    def update(self, name: str, value: Union[float, int, np.ndarray, torch.Tensor], type: str="amount"):
        """
        Update a value for a specific name, tracking only recent values
        
        Args:
            name (str): Name of the value to track
            value: Value to add (supports float, int, numpy array, torch tensor)
        """
        # Convert to float
        float_value = self._convert_to_float(value)
        
        # Initialize if name doesn't exist
        if name not in self.values:
            self.values[name] = []
        
        # Add to list of values
        self.values[name].append(float_value)

        self.value_type_cache[name] = type
        
        # Trim to some multiple of the report period
        # self.values[name] = self.values[name][-self.report_period * 10:]
        
        # Check if we have enough values to report
        # if len(self.values[name]) == self.report_period:
        if len(self.values[name]) % self.report_period == 0:
            avg = self.get_avg(name)
            avg = self._sync_scalar(self.accelerator, avg)
            avg = self._format_typed_float(avg, type)
            self.print_fn(f"{name} - Avg: {avg} (last {self.report_period} values)")
    
    def get_avg(self, name: str) -> float:
        """
        Get the current average for a specific name
        
        Args:
            name (str): Name of the value to retrieve
        
        Returns:
            float: Current average, or 0 if no values
        """
        if name not in self.values or not self.values[name]:
            return 0.0
        # return sum(self.values[name]) / len(self.values[name])
        return sum(self.values[name][-self.report_period:]) / self.report_period
    
    def reset(self, name: Optional[str] = None):
        """
        Reset values for a specific name or all names, reporting averages before reset
        
        Args:
            name (str, optional): Name to reset. If None, reset all.
        """
        # skip if no values
        if len(self.values) == 0:
            logger.info("No values to reset, skipping")
            return 
        
        if name is None:
            # print a report begin and end message
            report_begin_string = "-" * 20 + " Report before reset " + "-" * 20
            self.print_fn(report_begin_string)
            # Report averages for all recorded values before clearing
            for key in self.values.keys():
                avg = self.get_avg(key)
                avg = self._sync_scalar(self.accelerator, avg)
                avg = self._format_typed_float(avg, self.value_type_cache[key])
                self.print_fn(f"{key}: {avg}")
            self.values.clear()
            
            self.print_fn("-" * len(report_begin_string))
        elif name in self.values:
            avg = self.get_avg(name)
            avg = self._sync_scalar(self.accelerator, avg)
            avg = self._format_typed_float(avg, self.value_type_cache[name])

            self.print_fn(f"{key}: {avg}")
            self.values[name].clear()

    @property
    def _need_print(self):
        return self.accelerator is None or self.accelerator.is_main_process

    def get_recent_values(self, name: str) -> List[float]:
        """
        Get recent values for a specific name
        
        Args:
            name (str): Name of the values to retrieve
        
        Returns:
            List[float]: Recent values
        """
        return self.values.get(name, [])

class TimingMeter:
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.start_time = datetime.now()
        self.last_time = self.start_time
        self.validation_time = 0
        self.num_samples = 0
        
    def update(self, num_samples=1):
        self.num_samples += num_samples

    def set_num_samples(self, num_samples):
        self.num_samples = num_samples
        
    def get_timing_stats(self, total_samples, include_validation=False):
        current_time = datetime.now()
        elapsed = current_time - self.start_time
        elapsed_seconds = elapsed.total_seconds()
        
        # Remove validation time if specified
        if not include_validation:
            elapsed_seconds -= self.validation_time
            
        samples_per_sec = self.num_samples / elapsed_seconds if elapsed_seconds > 0 else 0
        
        remaining_samples = total_samples - self.num_samples
        eta_seconds = remaining_samples / samples_per_sec if samples_per_sec > 0 else 0
        
        def format_time(seconds):
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = int(seconds % 60)
            if hours > 0:
                return f"{hours}h {minutes}m {secs}s"
            elif minutes > 0:
                return f"{minutes}m {secs}s"
            else:
                return f"{secs}s"
                
        return {
            "elapsed": format_time(elapsed_seconds),
            "eta": format_time(eta_seconds),
            "samples_per_sec": f"{samples_per_sec:.2f}"
        }
        
    def start_validation(self):
        self.validation_start = datetime.now()
        
    def end_validation(self):
        validation_elapsed = (datetime.now() - self.validation_start).total_seconds()
        self.validation_time += validation_elapsed
        return validation_elapsed