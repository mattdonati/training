#!/bin/bash
# Copyright (c) 2023-2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export DIFFUSERS_OFFLINE=1
export HF_HOME=/hf_home

# TODO: remove it once the root issue of process hang resolved
export TORCHINDUCTOR_COMPILE_THREADS=1

readonly node_rank="${SLURM_NODEID:-0}"
readonly local_rank="${LOCAL_RANK:=${SLURM_LOCALID:=${OMPI_COMM_WORLD_LOCAL_RANK:-0}}}"

# where to copy checkpoints after the run. If not set (default), will not copy them.
CHECKPOINT_STORE_PATH="${CHECKPOINT_STORE_PATH:-""}"
# whether to clean up all checkpoints (default) after the run.
CLEAR_CHECKPOINT_DIR="${CLEAR_CHECKPOINT_DIR:-1}"

USE_SYNTHETIC_DATA="${USE_SYNTHETIC_DATA:-0}"
CONFIG_DATA=""
# Synthetic data environment usually comes without checkpoints and dataset
if [[ "${USE_SYNTHETIC_DATA}" -gt 0 ]]; then
    CONFIG_DATA+="+model.data.synthetic_data=true"
    CONFIG_DATA+=" +model.data.synthetic_data_length=10000"
    CONFIG_DATA+=" model.data.train.dataset_path=null"
    CONFIG_DATA+=" model.data.webdataset.local_root_path=null"
    CONFIG_DATA+=" model.ckpt_path=null"
    CONFIG_DATA+=" model.cond_stage_config.cache_dir=/tmp/clip"
    unset TRANSFORMERS_OFFLINE
    unset HF_HUB_OFFLINE
fi

CAPTURE_CUDAGRAPH_ITERS="${CAPTURE_CUDAGRAPH_ITERS:-"15"}"

USE_DIST_OPTIMIZER="${USE_DIST_OPTIMIZER:-"False"}"
if [[ "${USE_DIST_OPTIMIZER}" = "True" ]]; then
    OPTIMIZER_CONF="optim@model.optim=distributed_fused_adam"
    if [[ -n "${DISTRIBUTED_SIZE:-}" ]]; then
        OPTIMIZER_CONF+=" model.optim.distribute_within_nodes=false"
        OPTIMIZER_CONF+=" model.optim.distributed_size=${DISTRIBUTED_SIZE}"
    fi
else
    OPTIMIZER_CONF="optim@model.optim=megatron_fused_adam"
fi

declare -a CMD
if [[ "${LOCAL_WORLD_SIZE}" -gt 1 ]]; then
    # Mode 1: Slurm launched a task for each GPU and set some envvars
    CMD=( "${NSYSCMD}" 'python' '-u')
else
    # interactive run on single node, no need to bind
    CMD=( "${NSYSCMD}" 'torchrun' "--nproc_per_node=${DGXNGPU}" )
fi

: "${LOGGER:=""}"
if [[ -n "${APILOG_DIR:-}" ]]; then
    if [ "$node_rank" -eq 0 ] && [ "$local_rank" -eq 0 ]; then
      LOGGER="apiLog.sh -p MLPerf/${MODEL_NAME} -v ${FRAMEWORK}/train/${DGXSYSTEM}"
    fi
fi

# Assert $RANDOM is usable
if [[ -z "${RANDOM}" ]]; then
    echo "RANDOM is not set!" >&2
    exit 1
fi

if [[ "${node_rank}" -eq 0 && "${local_rank}" -eq 0 ]]; then
    echo "RANDOM_SEED=${RANDOM_SEED}"
fi

mkdir -p "/tmp/nemologs"

${LOGGER:-} ${CMD[@]} main.py \
    "trainer.num_nodes=${DGXNNODES}" \
    "trainer.devices=${DGXNGPU}" \
    "trainer.max_steps=${CONFIG_MAX_STEPS}" \
    ${OPTIMIZER_CONF} \
    "model.optim.lr=${LEARNING_RATE}" \
    "model.optim.sched.warmup_steps=${WARMUP_STEPS}" \
    "model.micro_batch_size=${BATCHSIZE}" \
    "model.global_batch_size=$((DGXNGPU * DGXNNODES * BATCHSIZE))" \
    "model.use_cudnn_layer_norm=${USE_CUDNN_LAYER_NORM}" \
    "model.use_torch_sched=${USE_TORCH_SCHED}" \
    "model.unet_config.use_flash_attention=${FLASH_ATTENTION}" \
    "model.unet_config.use_te_dpa=${USE_TE_DPA}" \
    "model.capture_cudagraph_iters=${CAPTURE_CUDAGRAPH_ITERS}" \
    "exp_manager.exp_dir=/tmp/nemologs" \
    "exp_manager.checkpoint_callback_params.every_n_train_steps=${CHECKPOINT_STEPS}" \
    "name=${EXP_NAME}" \
    "model.seed=${RANDOM_SEED}" \
    ${CONFIG_DATA} \
    --config-path "${CONFIG_PATH}" \
    --config-name "${CONFIG_NAME}" || exit 1

SKIP_EVAL="${SKIP_EVAL:-"0"}"
if [[ "${USE_SYNTHETIC_DATA}" -eq 0 && "${SKIP_EVAL}" -eq 0 ]]; then
    # disable SHARP
    export NCCL_SHARP_DISABLE=1
    export NCCL_COLLNET_ENABLE=0

    CKPT_PATH="/tmp/nemologs/${EXP_NAME}/checkpoints/"
    if [[ "${SLURM_NODEID}" -eq 0 && "${local_rank}" -eq 0 ]]; then
        echo "CKPT_PATH=${CKPT_PATH}"
    fi

    python infer_and_eval.py \
    "trainer.num_nodes=${DGXNNODES}" \
    "trainer.devices=${DGXNGPU}" \
    "custom.sd_checkpoint_dir=${CKPT_PATH}" \
    "custom.num_prompts=${INFER_NUM_IMAGES}" \
    "custom.infer_start_step=${INFER_START_STEP}" \
    "infer.batch_size=${INFER_BATCH_SIZE}"
fi

# cleanup
if [[ "${local_rank}" -eq 0 ]]; then
    if [[ "${node_rank}" -eq 0 && -n "${CHECKPOINT_STORE_PATH}" ]]; then
        echo "Saving a copy of checkpoints to ${CHECKPOINT_STORE_PATH}"
        cp -r "${CKPT_PATH}" "${CHECKPOINT_STORE_PATH}"
    fi

    if [[ "${CLEAR_CHECKPOINT_DIR}" -gt 1 ]]; then
        if [[ "${node_rank}" -eq 0 ]]; then
            echo "Cleaning up ${CKPT_PATH}"
        fi
        rm -r "${CKPT_PATH}"
    fi
fi
