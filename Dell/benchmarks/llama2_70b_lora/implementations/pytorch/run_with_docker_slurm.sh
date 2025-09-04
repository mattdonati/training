#!/bin/bash

# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
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

set -euxo pipefail

#source config_MI300X_1x8x1.sh
#source /home/frank/MLPerf_training_5.0_AMD/AMD_llama2_70b_lora_MN/config_MI300X_1x8x1.sh
#export DATADIR=/home/frank/mlperf_llama2_ds_amd
#export LOGDIR=/home/frank/results/mlperf_llama2_amd/${SLURM_JOB_NUM_NODES}-MI300X-llama2
#export CONT=rocm/amd-mlperf:llama2_70b_training_5.0
rsync -avP /mnt/data/mlperf_llama2 /dev/shm/

#source config_MI300X_1x8x1.sh
#source config_${SLURM_JOB_NUM_NODES}xXE9680_8MI300X.sh
#source /home/frank/MLPerf_training_5.0_AMD/AMD_llama2_70b_lora_MN/config_MI300X_1x8x1.sh
#export DATADIR=/home/frank/mlperf_llama2_ds_amd
#export DATADIR=/dev/shm/mlperf_llama2
#export LOGDIR=/home/frank/results/mlperf_llama2_amd/${SLURM_JOB_NUM_NODES}-MI300X-llama2
#export CONT=rocm/amd-mlperf:llama2_70b_training_5.0


# Change directory to the model directory
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
cd $SCRIPT_DIR

# Vars without defaults
: "${DGXSYSTEM:?DGXSYSTEM not set}"
: "${CONT:?CONT not set}"
: "${DATADIR:?DATADIR not set}"

# Vars with defaults
: "${NEXP:=1}"
: "${CLEAR_CACHES:=1}"
: "${CHECK_COMPLIANCE:=1}"
: "${MLPERF_RULESET:=5.0.0}"
: "${LOGDIR:=./results}"
: "${CONT_NAME:=mlperf_llama2sft}"
: "${LOG_FREQ:=0}"
: "${ROOT_SEED:=${ROOT_SEED}}"

echo "ROOT SEED IS $ROOT_SEED"
# Other vars
readonly _config_file="./config_${DGXSYSTEM}.sh"
readonly _cont_name="${CONT_NAME}"
_cont_mounts=("--volume=${DATADIR}/data:/data" "--volume=${DATADIR}/model:/ckpt" "--volume=${PWD}:/workspace/code")

# Stop all other containers
echo "Stopping containers"
docker ps -q | xargs -r docker stop

# Setup directories
mkdir -p "${LOGDIR}"

# Get list of envvars to pass to docker
mapfile -t _config_env < <(env -i bash -c ". ${_config_file} && compgen -e" | grep -E -v '^(PWD|SHLVL)')
_config_env+=(DGXSYSTEM)

_config_env+=(SLURM_NNODES)
_config_env+=(SLURM_NODEID)
_config_env+=(SLURM_SRUN_COMM_HOST)

echo ${_config_env[@]}
mapfile -t _config_env < <(for v in "${_config_env[@]}"; do echo "--env=$v"; done)


# Cleanup container
cleanup_docker() {
    docker container rm -f "${_cont_name}" || true
}
cleanup_docker
trap 'set -eux; cleanup_docker' EXIT
echo $(pwd)

docker run --rm --init --detach \
      --net=host --uts=host \
      --ipc=host --device /dev/dri --device /dev/kfd \
      --security-opt=seccomp=unconfined \
      --name="${_cont_name}" "${_cont_mounts[@]}" \
      -e IMAGE_NAME="${CONT}" \
      "${CONT}" sleep infinity

# Make sure container has time to finish initialization
sleep 5

bash runtime_tunables.sh

docker exec "${_cont_name}" true

DATETIME=$(date "+%Y%m%d%H%M%S")

for _experiment_index in $(seq 0 $((NEXP-1))); do
{
  echo "Beginning trial ${_experiment_index} of ${NEXP}"
  bash -c "echo -n 'Clearing cache on ' && hostname && sync && sudo /sbin/sysctl vm.drop_caches=3"
  _current_log_file="${LOGDIR}/${DATETIME}_result_${_experiment_index}.txt"

#_current_log_file="${LOGDIR}/result_${_experiment_index}.txt"

(
    export SEED=$(($ROOT_SEED - 1 + 10#$_experiment_index))
    _config_env+=(--env=SEED=$SEED) 
    echo 'launching experiment using:'  ${_config_env[@]} ${_cont_name} ./run_and_time_slurm.sh
    docker exec ${_config_env[@]} ${_cont_name} ./run_and_time_slurm.sh
  ) |& tee ${_current_log_file}
} 
done 
