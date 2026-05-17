import logging

logger = logging.getLogger(__name__)

LIGER_KERNEL_AVAILABLE = False
try:
    from liger_kernel.transformers import apply_liger_kernel_to_qwen3, apply_liger_kernel_to_qwen2, apply_liger_kernel_to_qwen3_vl
    LIGER_KERNEL_AVAILABLE = True # disable for now
except ImportError:
    apply_liger_kernel_to_qwen2 = None
    apply_liger_kernel_to_qwen3 = None
    apply_liger_kernel_to_qwen3_vl = None

if LIGER_KERNEL_AVAILABLE:
    logger.info("liger_kernel found, will apply liger kernel to qwen3 model if applicable")
else:
    logger.info("liger_kernel not found or disabled, skipping liger kernel application")