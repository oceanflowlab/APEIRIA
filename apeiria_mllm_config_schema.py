# config_schema.py
from dataclasses import dataclass, field
from typing import List, Optional, Any
from hydra.core.config_store import ConfigStore # Import ConfigStore
from datetime import datetime


# Helper function to generate default run name (similar to original logic)
def get_default_run_name():
    return f"mllm-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

@dataclass
class Config:
    # --- Defaults that might need post-processing ---
    # Set defaults to None or "" here, handle logic in main script
    wandb_run_name: str = field(default_factory=get_default_run_name) 
    output_dir: str = "" 
    discrete_location_bin_range: Optional[List[float]] = None # Use list directly

    # --- Logging Configuration ---
    log_format: str = "[%(asctime)s %(name)s %(levelname)s %(funcName)s] %(message)s"
    log_datefmt: str = "%I:%M:%S"
    log_level_main: str = "INFO"  # Log level for the main process (e.g., "INFO", "DEBUG", "WARNING")
    log_level_other: str = "WARNING" # Log level for other processes
    transformers_verbosity_main: str = "info" # Verbosity for transformers on main process ("info", "warning", "error", "debug")
    transformers_verbosity_other: str = "warning"

    # --- Model parameters ---
    model_name: Optional[str] = None # Make mandatory fields None initially
    attn_implementation: str = "sdpa"
    max_objects: int = 50
    no_object_in_language_model: bool = False
    sync_bn: bool = False
    checkpoint_path: str = ""
    only_load_adapter: bool = False
    object_embedding_type: str = "simple" # Add choices later if needed via OmegaConf Enum or validation
    use_clip_class_embedding: bool = True
    clip_model_name: str= "ViT-H-14-378-quickgelu|dfn5b"
    use_proposal_feature: bool = False
    proposal_type: str = "uni3d-mask3d-gt"
    normalize_proposal_feature: bool = False
    use_2d_proposal_feature: bool = False

    separate_location_embedding: bool = True
    discrete_location_bins: int = 101
    discrete_location_decay_kernel: str = "gaussian"
    discrete_location_decay_kernel_size: float = 1.0
    # discrete_location_bin_range handled above
    image_encoder: Optional[str] = None
    image_feature_type: str = "global" # "patch", "global", "adaptive_12x12"
    n_views_in_m_views: str = "32_8"  # e.g., "32_8" means resampling 8 views from 32 views

    # --- LoRA parameters ---
    lora_rank: int = -1
    lora_alpha: float = 32.0 # Use float for consistency
    lora_dropout: float = 0.1
    use_pissa: bool = False
    unfreeze_word_embedding: bool = False
    lora_word_embedding: bool = False
    use_rslora: bool = False
    use_dora: bool = False

    # --- Dataset parameters ---
    ratio: float = 1.0
    load_from_cache: bool = False
    num_scenes: int = 100
    objects_per_scene: int = 10
    room_size: float = 10.0
    min_objects_per_class: int = 1
    max_objects_per_class: int = 3
    shuffle_objects: bool = False
    fix_template: bool = False
    dataset_type: str = "synthetic3d"
    r"""
    curriculum_stages:
        - epoch: 0
            datasets: ["sr3d_filter", "sr3d_object_info"]
        - epoch: 3
            datasets: ["sr3d_filter", "sr3d_object_info", "sr3d"]
    """
    curriculum_stages: List[dict] = field(default_factory=lambda: [
        {"epoch": 0, "datasets": ["sr3d_filter", "sr3d_object_info"]},
        {"epoch": 3, "datasets": ["sr3d_filter", "sr3d_object_info", "sr3d"]}
    ])
    enable_curriculum: bool = False
    no_object_id_input: bool = False
    add_thinking_trace: bool = False
    add_thinking_trace_prompt: bool = False # to indicate thinking trace is needed even if add_thinking_trace is False
    adjust_scene_layouts: bool = False
    relational_data_ratio: float = 0.1
    add_full_thinking_trace_for_relational: bool = False
    add_full_thinking_trace_for_filter_in_relational: bool = True
    only_add_positive_relations: bool = False
    add_plans_first: bool = False
    add_partial_full_thinking_trace_for_relational: bool = False
    add_partial_full_thinking_trace_for_filter: bool = False
    pre_filter_objects: bool = False
    max_filter_objects: int = 30
    # for caption cot datasets
    caption_sources: List[str] = field(default_factory=lambda: ["scanrefer", "nr3d", "sr3d", "multi3drefer"])
    max_captions_per_object_in_cot: int = 10
    max_caption_cot_len: int = 5000  # Max length for CoT context
    location_precision: int = 2  # Decimal places for location

    # --- Training parameters ---
    batch_size: int = 4
    eval_batch_size: int = 4
    epochs: int = 5
    lr: float = 5e-5
    lr_non_lm: float = 1e-4
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.1
    seed: int = 42
    lr_scheduler_type: str = "linear"
    gradient_checkpointing: bool = False
    optimizer: str = "adamw"
    galore_rank: int = 128
    find_unused_parameters: bool = True
    loss_type: str = "sft"  # "sft", "dft", "psft"
    coeff_grounding_loss: float = 0.3
    add_bracket_in_object_detail: bool = False # needed for reg_head and grounding loss

    # --- Generation parameters ---
    max_new_tokens: int = 256
    num_beams: int = 5
    do_sample: bool = False
    top_k: Optional[int] = 50
    top_p: Optional[float] = 0.95
    temperature: Optional[float] = 0.3
    eval_use_sglang: bool = False
    sglang_log_level: str = "info"
    temp_lora_path_eval: str = "/dev/shm/apeiria-temp-eval-lora"
    max_retries: int = 5 # rejection sampling for single-output datasets like ScanRefer and Nr3D

    # --- Output and logging ---
    # output_dir handled above
    logging_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 500
    wandb_project: str = "apeiria-3d-synthetic"
    no_save: bool = False
    eval_logging_frequency: int = 10
    iou_threshold: float = 0.25

    # --- Other optimization parameters ---
    compile_model: bool = False

    # --- Distributed training parameters ---
    master_addr: str = "0.0.0.0"
    master_port: str = "12345"
    use_fsdp: bool = False
    
    # --- Checkpointing ---
    resume_from_checkpoint: Optional[str] = None
    resume_from_checkpoint_epoch: bool = True # resume from checkpoint epoch

    # --- Hydra configuration ---
    hydra: Optional[Any] = None  # 或者 Optional[DictConfig] = None

    # --- Inference Related ---
    mode: Optional[str] = None
    split: Optional[str] = None
    num_inference_passes: Optional[int] = None
    resume_from_partial_results_dir: Optional[str] = None # resume inference
    validation_only: bool = False

    # --- External Traces config ---
    max_traces_per_sample: Optional[int] = None # None for unlimited
    external_traces_path: Optional[str] = None # Used for plans from program
    shuffle_traces: bool = True
    use_sglang_for_generation: bool = False
    use_nr3d_plan_from_program: bool = False

    external_plan_path: Optional[str] = None # For direct plans (natural language)
    gt_scene_data_path: Optional[str] = None # For external scene data during inference
    previous_results_path: Optional[str] = None # For using previous results during inference (to extract internal plan)

    single_prediction_prompt: str = "" # For single prediction during inference

    # --- DPO/IPO Training Configuration ---
    use_ipo: bool = True
    use_cpo: bool = False
    # The beta hyperparameter for DPO, controlling the strength of the preference signal.
    beta: float = 0.1
    # The tau hyperparameter for IPO, providing stable regularization.
    tau: float = 0.1
    warmup_ratio: float = 0.1  # 10% of total steps
    max_length: int = 2000
    offload_reference_model: bool = False  # Offload reference model to CPU to save GPU memory
    offload_policy_model: bool = False # Offload policy model to CPU (when calculating reference logps)
    nll_loss_weight: float = 0 # 0 for disable.

# -- Hydra config store setup --
# Create a ConfigStore instance.
cs = ConfigStore.instance()
# Register the Config class under the name 'config'.
cs.store(name="apeiria_mllm", node=Config)
cs.store(name="apeiria_mllm", node=Config, package="_global_")


# You can also register individual groups if you want to compose them differently
# cs.store(group="model", name="base_model", node=ModelConfig)
# cs.store(group="dataset", name="base_dataset", node=DatasetConfig)
# etc.
