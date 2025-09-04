#!/bin/bash
export UCX_NET_DEVICES=mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_11:1,mlx5_12:1,mlx5_15:1,mlx5_17:1,mlx5_4:1
export DATADIR=/training_datasets_v5.0/training_datasets_v4.1/criteo_1tb_multihot_raw
export LOGDIR=/root/training_results_v5.0/dlrmv2/multinode_results
export BINDCMD="bindpcie --cpu=node"
export CONT="dockerd://mlperf-dell:dlrm"
