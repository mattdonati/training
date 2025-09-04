#!/usr/bin/env bash
set -euo pipefail

DATADIR="/gcs-dir/data/data/gov_report"  # set your </path/to/dataset>, needed to change for my locations because of the double data
MODEL="/gcs-dir/data/data/model"  # set your </path/to/dataset>, needed to change for my locations because of the double data
#LOGDIR="</path/to/output_logdir>"  # set the place where the output logs will be saved
#export CONT=<docker/registry>/mlperf-nvidia:llama2_70b_lora-pyt
source config_XE9680lx8H200-SXM-141GB_1x8x2xtp1pp1cp2.sh  # select config and source it

# Correct Vars that are inccorrectd in the sourced config file 
DGXNNODES=2
TP=4
CP=1
PP=1
MINIBS=1
MBS=1

# Add in envs from run.sub 


# --- GKE-specific environment variables ---
MASTER_ADDR="train-workers-0-0.train"
MASTER_PORT=3389
NODE_RANK="$NODE_RANK"
NUM_NODES="$NODE_COUNT"

# --- User-configurable parameters ---
# Vars without defaults
: "${DGXSYSTEM:="nemo"}"

# Vars with defaults
: "${CHECK_COMPLIANCE:=1}"
: "${MLPERF_RULESET:=5.0.0}"
: "${MLPERF_SYSTEM_NAME:='unknown'}"
: "${MLPERF_SCALE:='unknown'}"
: "${MLPERF_CLUSTER_NAME:='unknown'}"
: "${MILESTONE_YML:=unknown}"
: "${DGXNGPU:=8}"
: "${NEXP:=1}"
: "${SEED_BASE:=${SEED-$RANDOM}}"
: "${DATESTAMP:=$(date +'%y%m%d%H%M%S%N')}"
: "${CLEAR_CACHES:=0}"
: "${LOGDIR:=/gcs-dir/results}" # changed so logs would go in the bucket 
: "${DROPCACHE_CMD:="sudo /sbin/sysctl vm.drop_caches=3"}"
: "${POWERLOGDIR:=' '}" # Power traces output dir
: "${POWERCMDDIR:=' '}" # Path to power monitor
: "${SET_MAXQ_CLK:=0}"
: "${SET_MINEDP_CLK:=0}"
: "${NCCL_TEST:=0}"
: "${NCCL_TEST_WALLTIME:=10}"
: "${NVTX_FLAG:=0}"
: "${WORK_DIR:=/workspace/ft-llm}"
: "${EXTRA_ASSETS:=}"
: "${MODEL_NAME:='llama2_70b'}"

unset NVTE_FLASH_ATTN                                                                                                                                                  │
unset NVTE_FUSED_ATTN                                                                                                                                                  │
unset NVTE_UNFUSED_ATTN    
# --- Paths ---
#DATA_DIR="/gcs-dir/hf-data"
#MODEL_PATH="/gcs-dir/llama-7b"
OUTPUT_DIR="${LOGDIR}/${MODEL_NAME}_${DGXSYSTEM}_${DATESTAMP}"

# --- Setup ---
echo "Running on node rank $NODE_RANK of $NUM_NODES nodes with $DGXNGPU GPUs."

# Clear caches
if [ "${CLEAR_CACHES}" -eq 1 ]; then
    echo "Clearing caches..."
    sync
    eval "${DROPCACHE_CMD}"
fi

# --- NCCL Test ---
if [ "${NCCL_TEST}" -eq 1 ]; then
    echo "Running NCCL test..."
    torchrun --nproc_per_node=${DGXNGPU} \
        --nnodes=$NUM_NODES \
        --node_rank=$NODE_RANK \
        --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT \
        --rdzv_id="${JOB_IDENTIFIER}" \
        --rdzv_backend static \
        -m torch.distributed.run \
        /opt/nccl-tests/build/all_reduce_perf -b 8 -e 128M -f 2 -g 1
fi

# --- Training ---
echo "Starting training..."

# Set GPU clocks
if [ "${SET_MAXQ_CLK}" -ne 0 ]; then
    echo "Setting MAXQ clocks..."
    /usr/bin/nvidia-smi -lgc "${SET_MAXQ_CLK}"
fi
if [ "${SET_MINEDP_CLK}" -ne 0 ]; then
    echo "Setting MINEDP clocks..."
    /usr/bin/nvidia-smi -lgc "${SET_MINEDP_CLK}"
fi

# Nsys setup
NSYSCMD=""
if [ "${NVTX_FLAG:-0}" -eq 1 ]; then
    NSYS_OUT="${OUTPUT_DIR}/nsys_n${NODE_RANK}"
    NSYSCMD="nsys profile --sample=cpu --cuda-graph-trace=node --cpuctxsw=none --trace=cuda,nvtx -f true --stats true -o ${NSYS_OUT}"
fi

PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" OMP_NUM_THREADS=8 torchrun \
--nproc-per-node="$DGXNGPU" \
--nnodes="${NUM_NODES}" \
--node_rank="${NODE_RANK}" \
--rdzv_id="${JOB_IDENTIFIER}" \
--rdzv_backend static \
--master_addr="${MASTER_ADDR}" \
--master_port="${MASTER_PORT}" \
train.py 

#CMD=( ${NSYSCMD} 'torchrun' \
##    --nproc_per_node=$DGXNGPU \
#    --nnodes=$NUM_NODES \
#    --node_rank=$NODE_RANK \
#   --master_addr=$MASTER_ADDR \
#    --master_port=$MASTER_PORT \
#   --rdzv_id="${JOB_IDENTIFIER}" \
#   --rdzv_backend static \
#   train.py 
#)

#${LOGGER:-} ${CMD[@]}
