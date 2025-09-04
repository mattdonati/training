#!/bin/bash

declare -a CMD
MASTER_PORT=29500
export DGXNNODES=$SLURM_NNODES
CMD="torchrun --nproc_per_node=8  --nnodes=$SLURM_NNODES --node_rank=$SLURM_NODEID --master_addr=$SLURM_SRUN_COMM_HOST --master_port=$MASTER_PORT"
echo "COMMAND: $CMD"
echo "Starting with seed $SEED on Rank: $SLURM_NODEID"

# start timing
start=$(date +%s)
start_fmt=$(date +%Y-%m-%d\ %r)
echo "STARTING TIMING RUN AT $start_fmt"
$CMD src/train.py
ret_code=$?

if [[ $ret_code != 0 ]]; then exit $ret_code; fi

# end timing
end=$(date +%s)
end_fmt=$(date +%Y-%m-%d\ %r)
echo "ENDING TIMING RUN AT $end_fmt"
# report result
result=$(( end - start ))
result_name="LLM_FINETUNING"
echo "RESULT,$result_name,,$result,AMD,$start_fmt"

exit 0