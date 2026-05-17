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

# Auto GPU NUMBER
export SLURM_GPUS=$(($(echo $CUDA_VISIBLE_DEVICES | tr -cd , | wc -c)+1))
# export TRANSFORMERS_CACHE=../.cache/huggingface/hub
# export HF_HOME=$TRANSFORMERS_CACHE

echo "Number of processes (GPUs): $SLURM_GPUS"

export PORT=$(shuf -i 20000-30000 -n 1)
export TOKENIZERS_PARALLELISM=false

export CUDA_LAUNCH_BLOCKING=1

$SMI_CMD

# expandable memory
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,garbage_collection_threshold:0.95"
export LIBRARY_PATH="/usr/local/cuda/lib64/stubs:$LIBRARY_PATH"
export LD_FLAGS="-L/usr/local/cuda/lib64/stubs"
export HF_ENDPOINT="https://hf-mirror.com"
export SUPPORT_CUTLASS_BLOCK_FP8=1
# export TORCH_DISTRIBUTED_DEBUG=DETAIL
ulimit -n 1024000

# export SGLANG_MIN_NEW_TOKEN_RATIO_FACTOR=0.33

# accelerate launch --config_file "apeiria_lm.yaml" --num_processes=$SLURM_GPUS --main_process_port=$PORT \
python train_apeiria_mllm_cot_rl_async.py master_port=$PORT \
    "$@" \
    2>&1 | tee ../apeiria-logs/mllm-grpo-$(date +'%Y-%m-%d-%H-%M-%S').log