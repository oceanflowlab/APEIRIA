# Check if mx-smi exists
if command -v mx-smi ; then
    # MACA specific variables for MX platform
    export MACA_PATH=/opt/maca
    export CUCC_PATH=${MACA_PATH}/tools/cu-bridge
    export CUDA_PATH=${CUCC_PATH}
    export MACA_CLANG_PATH=${MACA_PATH}/mxgpu_llvm/bin
    export PATH=${CUDA_PATH}/bin:${MACA_CLANG_PATH}:${PATH}
    export LD_LIBRARY_PATH=${MACA_PATH}/lib:${MACA_PATH}/mxgpu_llvm/lib:${LD_LIBRARY_PATH}
    export MACA_SMALL_PAGESIZE_ENABLE=1
    export PYTORCH_ENABLE_SAME_SAME_RAND_A100=1
    export SET_DEVICE_NUMA_PREFERRED=1
    export MCCL_P2P_LEVEL=SYS
    export MCCL_FAST_WRITE_BACK=1
    export MCCL_EARLY_WRITE_BACK=15
    export MCCL_NET_GDR_LEVEL=SYS
    export MCCL_CROSS_NIC=1
    export MHA_BWD_NO_ATOMIC_F64=1
    export MCCL_ENABLE_FC=0

    # Set CUDA_VISIBLE_DEVICES based on mx-smi
    [ -z "$CUDA_VISIBLE_DEVICES" ] && export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $(($(mx-smi -L | wc -l)-3)))
    SMI_CMD="mx-smi"
else
    # Original NVIDIA setup
    [ -z "$CUDA_VISIBLE_DEVICES" ] && export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $(($(nvidia-smi -L | wc -l)-1)))
    SMI_CMD="nvidia-smi"
fi

# Common variables
export LANG=en_US.UTF-8
export OMP_NUM_THREADS=8

export NCCL_DEBUG=INFO
export NCCL_TIMEOUT=1800
export NCCL_P2P_LEVEL=SYS

# Auto GPU NUMBER
export SLURM_GPUS=$(($(echo $CUDA_VISIBLE_DEVICES | tr -cd , | wc -c)+1))
# export TRANSFORMERS_CACHE=../.cache/huggingface/hub
# export HF_HOME=$TRANSFORMERS_CACHE

echo "Number of processes (GPUs): $SLURM_GPUS"

export PORT=$(shuf -i 29000-30000 -n 1)
export TOKENIZERS_PARALLELISM=false
export HF_ENDPOINT=https://hf-mirror.com

TIMESTAMP=$(date +'%Y-%m-%d-%H-%M-%S')
LOG_DIR="../apeiria-logs"
mkdir -p $LOG_DIR # Create log directory if it doesn't exist
LOG_FILE="${LOG_DIR}/log-mllm-${TIMESTAMP}.log"

# export CUDA_LAUNCH_BLOCKING=1

$SMI_CMD

# expandable memory
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# accelerate launch --config_file "apeiria_lm.yaml" --num_processes=$SLURM_GPUS --main_process_port=$PORT \
#     train_apeiria_mllm.py \
#     --model_name "../Qwen2.5-3B-Instruct" \
#     --dataset_type "[sr3d, sr3d_filter, sr3d_object_info]" \
#     --add_thinking_trace --add_partial_full_thinking_trace_for_relational --add_partial_full_thinking_trace_for_filter \
#     --relational_data_ratio 0.2 \
#     --lora_rank 128 --lora_alpha 256 --galore_rank 128 \
#     --object_embedding_type "discrete_location" --discrete_location_decay_kernel linear --discrete_location_decay_kernel_size 2.45 \
#     --num_scenes 1000 --objects_per_scene 15 --fix_template \
#     --batch_size 1 --eval_batch_size 2 --lr 3e-5 --lr_non_lm 1e-3 --weight_decay 0 --epochs 10 --max_grad_norm 1.0 \
#     --gradient_accumulation_steps 2 --lr_scheduler_type "cosine" --warmup_ratio 0.25 \
#     --num_beams 1 --max_new_tokens 1500  --eval_logging_frequency 20 \
#     --logging_steps 2 --eval_steps 50000 --save_steps 50000 --no_save \
#     --gradient_checkpointing \
#     "$@" \
#     2>&1 | tee ../apeiria-logs/log-mllm-$(date +'%Y-%m-%d-%H-%M-%S').log

if command -v local_accelerate > /dev/null 2>&1; then
    ACCELERATE_CMD="local_accelerate"
    echo "Info: Using local_accelerate"
else
    ACCELERATE_CMD="accelerate"
    echo "Info: local_accelerate not found, using accelerate"
fi

echo "Starting training..."
echo "Using Accelerate command: $ACCELERATE_CMD"
echo "Number of processes: $NUM_PROCESSES"
echo "Main process port: $PORT"
echo "Arguments passed: $@"
echo "Logging to: $LOG_FILE"
ulimit -n 1024000

# used hydra to manage configurations
# "$ACCELERATE_CMD" launch --config_file "apeiria_lm.yaml" --num_processes=$SLURM_GPUS --main_process_port=$PORT \
python train_apeiria_mllm.py --config-name=apeiria_mllm \
    master_port=$PORT \
    "$@" \
    2>&1 | tee "$LOG_FILE" 

    # eval_use_sglang=true \

    # --use_pissa
    # --lora_rank 128 --lora_alpha 256 \
    # --discrete_location_decay_kernel gaussian --discrete_location_decay_kernel_size 1 \
    # --discrete_location_decay_kernel linear --discrete_location_decay_kernel_size 2.45
    # --batch_size 4 --eval_batch_size 4 [When using LoRA]
    # --lora_rank 128 --lora_alpha 256 \
    # --compile_model \


# --- Examples ---
# no-CoT, 1000 scene (enough data), no single object alignment
# ./train_mllm.sh --model_name "../Qwen2.5-3B-Instruct" --batch_size 2 --eval_batch_size 8 --gradient_accumulation_steps 1  --num_scenes 1000 --dataset_type "[synthetic3d]" 
# no-CoT, 1000 scene (enough data), add single object alignment
# ./train_mllm.sh --model_name "../Qwen2.5-3B-Instruct" --batch_size 2 --eval_batch_size 8 --gradient_accumulation_steps 1  --num_scenes 1000 --dataset_type "[synthetic3d,synthetic3d_object_info]

# CoT, 1000 scene (enough data), no single object alignment
# ./train_mllm.sh --model_name "../Qwen2.5-3B-Instruct" --batch_size 2 --eval_batch_size 8 --gradient_accumulation_steps 1  --num_scenes 1000 --dataset_type "[synthetic3d]" --add_`thinking`_trace
# CoT, 1000 scene (enough data), add single object alignment
# ./train_mllm.sh --model_name "../Qwen2.5-3B-Instruct" --batch_size 2 --eval_batch_size 8 --gradient_accumulation_steps 1  --num_scenes 1000 --dataset_type "[synthetic3d,synthetic3d_object_info] --add_thinking_trace


