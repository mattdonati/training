#!/bin/bash

set -x

##SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive

source config_MI300X_1x8x1.sh
#source config_${SLURM_JOB_NUM_NODES}xXE9680_8MI300X.sh

#export NVTE_USE_CAST_TRANSPOSE_TRITON=1
unset NVTE_USE_RMSNORM_TRITON
#export OMP_NUM_THREADS=1
#export PYTORCH_NVML_BASED_CUDA_CHECK=1
#export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 
export ROOT_SEED=$RANDOM


export NEXP=10



#source /home/frank/MLPerf_training_5.0_AMD/AMD_llama2_70b_lora_MN/config_MI300X_1x8x1.sh
#export DATADIR=/home/frank/mlperf_llama2_ds_amd
export DATADIR=/dev/shm/mlperf_llama2
export LOGDIR=/home/frank/results/mlperf_llama2_amd/${SLURM_JOB_NUM_NODES}-MI300X-llama2
export CONT=rocm/amd-mlperf:llama2_70b_training_5.0


 
srun bash run_with_docker_slurm.sh
