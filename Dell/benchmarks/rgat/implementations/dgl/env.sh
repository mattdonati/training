#!/bin/bash
export UCX_NET_DEVICES=mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_11:1,mlx5_12:1,mlx5_15:1,mlx5_17:1,mlx5_4:1
export DATA_DIR=/training_datasets_v5.0/training_datasets_v4.1/gnn/igbh_full/converted/full
export GRAPH_DIR=/training_datasets_v5.0/training_datasets_v4.1/gnn/igbh_full/graph/full
export LOGDIR=/root/training_results_v5.0/rgat/results
export FP8_EMBEDDING=1
export ENABLE_IB_BINDING=0
export NCCL_P2P_DISABLE=NVL
export NCCL_SOCKET_IFNAME=eno8303
ulimit -l unlimited
export CONT="dockerd://mlperf-dell:rgat"
