#!/usr/bin/env bash
set -euo pipefail

# --- Configuration ---
MASTER_ADDR=${1:-"localhost"}
MASTER_NODE_PORT=${2:-"29500"}
NODE_RANK=${3:-"0"}
NUM_NODES=${4:-"1"}
NUM_GPU_PER_NODE=8
BATCH_SIZE_PER_GPU=2
#ACCUMULATION_BATCH_SIZE=4
ACCUMULATION_STEPS=4
NUM_CPU_CORES=8

# --- Paths ---
DATA_DIR="/gcs-dir/hf-data"
MODEL_PATH="/gcs-dir/llama-7b"
OUTPUT_DIR="./results/llama-70b_scrolls_gov_report_r16_"
DEEPSPEED_CONFIG="configs/deepspeed.json"
HOSTFILE="hostfile"

# --- Setup ---
echo "Running on node rank $NODE_RANK of $NUM_NODES nodes with $NUM_GPU_PER_NODE GPUs."

# Set up NCCL environment variables
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_HCA=mlx5_0
export NCCL_IB_GID_INDEX=3
export NCCL_IB_TIMEOUT=22
export NCCL_IB_RETRY_CNT=7
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_IB_FIFO_TC=84

# Write hostfile
echo "$MASTER_ADDR slots=$NUM_GPU_PER_NODE" > $HOSTFILE
if [ "$NUM_NODES" -gt 1 ]; then
    for ((i=1; i<NUM_NODES; i++)); do
        node_addr="$(echo $MASTER_ADDR | sed 's/0-0/0-/'"${i}"'/g')"
        echo "$node_addr slots=$NUM_GPU_PER_NODE" >> "$HOSTFILE"
    done
fi

echo "Deepspeed hostfile:"
cat $HOSTFILE
echo ""

# Calculate gradient accumulation steps
#ACCUMULATION_STEPS=$((ACCUMULATION_BATCH_SIZE / (NUM_NODES * NUM_GPU_PER_NODE * BATCH_SIZE_PER_GPU)))

# --- Training ---
echo "Starting training..."

OMP_NUM_THREADS=$NUM_CPU_CORES deepspeed \
    --hostfile=$HOSTFILE \
    --no_ssh \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    --node_rank=$NODE_RANK \
    scripts/train.py \
    --dataset_path $DATA_DIR \
    --model_path $MODEL_PATH \
    --max_seq_len 8192 \
    --bf16 True \
    --logging_steps 24 \
    --eval_steps 48 \
    --output_dir $OUTPUT_DIR \
    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
    --gradient_accumulation_steps $ACCUMULATION_STEPS \
    --lr_scheduler_type cosine \
    --learning_rate 4e-4 \
    --weight_decay 0.0001 \
    --warmup_ratio 0 \
    --max_grad_norm 0.3 \
    --use_gradient_checkpointing True \
    --target_eval_loss 0.925 \
    --use_peft_lora True \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --max_steps 1024 \
    --use_flash_attn True \
    --seed 1234 \
    --lora_target_modules qkv_proj,o_proj \
    --deepspeed $DEEPSPEED_CONFIG

echo "Training completed."

                