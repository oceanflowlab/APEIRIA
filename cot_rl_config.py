import datetime
from dataclasses import dataclass, field
from typing import List, Optional

from hydra.core.config_store import ConfigStore


@dataclass
class MultimodalGRPOConfig:
    # Distributed settings
    master_addr: str = "0.0.0.0"
    master_port: str = "12345"

    # Model and data settings
    model_name: str = "../Qwen2.5-3B-Instruct"
    output_dir: str = "grpo_mllm"
    compile_model: bool = False
    compile_train_model: bool = False
    attn_implementation: str = "flash_attention_2"

    # Load checkpoint
    load_checkpoint: str = ""
    
    # Multimodal model settings
    max_objects: int = 100
    no_object_in_language_model: bool = False
    object_embedding_type: str = "simple"
    use_clip_class_embedding: bool = True
    clip_model_name: str = "ViT-H-14-378-quickgelu|dfn5b"
    separate_location_embedding: bool = True
    discrete_location_bins: int = 101
    discrete_location_decay_kernel: str = "gaussian"
    discrete_location_decay_kernel_size: float = 1.0
    discrete_location_bin_range: List[float] = field(default_factory=lambda: [0.0, 10.0])

    # APEIRIA_OPEN_UNUSED: Legacy 2D image/view feature fields. Kept only for
    # backwards-compatible config parsing; Real3DDataset ignores them.
    image_encoder: str = "" # no extra image encoder "ViT-gopt-16-SigLIP2-384|webli" # 1536d 
    n_views_in_m_views: str = "32_8"
    image_feature_type: str = "global" # global, patch, adaptive_12x12
    
    # SGLang settings
    use_sglang_for_generation: bool = False
    lora_update_path: str = "/dev/shm/apeiria-lora-update"
    
    # LoRA settings
    use_lora: bool = True
    lora_r: int = 128
    lora_alpha: int = 256
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    
    # Process allocation
    num_inference_gpus: int = 4
    num_training_gpus: int = 4
    
    # Training hyperparameters
    num_iterations: int = 3
    num_steps: int = 500
    rollout_batch_size: int = 1
    num_generations: int = 16
    generation_micro_batch_size: int = 32
    train_micro_batch_size: int = 4
    ref_model_micro_batch_size: int = 4

    max_completion_length: int = 1000
    max_length: int = 2000
    beta: float = 0.004
    learning_rate: float = 1e-6
    weight_decay: float = 0
    max_grad_norm: float = 1.0
    warmup_steps: int = 0
    mu: int = 1
    epsilon: float = 0.1
    epsilon_min: float = 0.2
    epsilon_max: float = 0.28
    update_iters: int = 1
    temperature: float = 1.0
    normalize_advantages: bool = False
    token_level_pg_loss: bool = False

    use_gradient_checkpointing: bool = True
    optimizer_type: str = "adamw"

    # Dataset parameters
    ratio: float = 1
    load_from_cache: bool = False
    dataset_type: str = "sr3d"
    num_scenes: int = 100
    objects_per_scene: int = 10
    room_size: float = 10.0
    min_objects_per_class: int = 1
    max_objects_per_class: int = 3
    shuffle_objects: bool = True
    fix_template: bool = False
    add_thinking_trace: bool = False
    add_thinking_trace_prompt: bool = True # to indicate thinking trace is needed in prompt
    adjust_scene_layouts: bool = False
    relational_data_ratio: float = 0.1
    pre_filter_objects: bool = False
    max_filter_objects: int = 30
    
    use_proposal_feature: bool = True
    proposal_type: str = "uni3d-mask3d-gt"
    normalize_proposal_feature: bool = True
    no_object_id_input: bool = True
    use_2d_proposal_feature: bool = True
    add_plans_first: bool = True

    add_bracket_in_object_detail: bool = False
    
    # Offloading settings
    offload_reference_model: bool = True
    offload_input_embeds: bool = False

    # Length shaping
    L_min: int = 250
    L_min_cache: int = 450
    L_max: int = 1200
    L_max_cache: int = 1050

    # Entropy shaping
    logp_factor_correct: float = 0.0
    logp_factor_wrong: float = 0.0
    logp_length_normalize: bool = True
    logp_group_normalize: bool = False
    logp_clip_ratio: float = -1 # less than 0 to disable
    
    # Misc
    seed: int = 42
    gradient_accumulation_steps: int = 1
    mixed_precision: str = "bf16"
    eval_size: int = 20
    report_interval: int = 25
    num_workers: int = 8
    system_prompt: str = (
        "Respond in the following format, potraying \"Apeiria\". In final answer, respond with \"Apeiria found...\" or \"Apeiria didn't find any...\", and a list of Object <ID>: At (..., ..., ...), size: ... x ... x ...:\n"
        "[APEIRIA THINKS]\n"
        "<... thinking procedure ...>\n"
        "[APEIRIA SPEAKS]\n"
        "Apeiria <... responses ...>"
    )
    save_iters: int = 100  # save model every N iterations
    start_iters: int = 0
    
    # Logging
    log_format: str = "[%(asctime)s %(name)s %(levelname)s %(funcName)s] %(message)s"
    log_datefmt: str = "%I:%M:%S"
    log_level_main: str = "INFO"
    log_level_other: str = "WARNING"
    transformers_verbosity_main: str = "info"
    transformers_verbosity_other: str = "warning"
    wandb_project: str = "apeiria-3d-grpo"
    wandb_run_name: str = field(default_factory=lambda: f"mllm-grpo-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}")
    sglang_log_level: str = "info"

    # NOTE: dummy? we don't use thinking trace pre-defined in dataset
    only_add_positive_relations: bool = True
    add_full_thinking_trace_for_relational: bool = False
    add_full_thinking_trace_for_filter_in_relational: bool = True
    add_partial_full_thinking_trace_for_relational: bool = True
    add_partial_full_thinking_trace_for_filter: bool = True
    max_traces_per_sample: Optional[int] = None
    external_traces_path: Optional[str] = None
    shuffle_traces: bool = False


cs = ConfigStore.instance()
cs.store(name="multimodal_grpo", node=MultimodalGRPOConfig)
