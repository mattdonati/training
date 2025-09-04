#!/bin/bash
export WARMUP=True
export DGXNGPU=8
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )

export DGXNNODES=1
export WALLTIME_MINUTES=50
export WALLTIME=$(( (${NEXP:-1} * WALLTIME_MINUTES) ))

export NCCL_MIN_P2P_NCHANNELS=32;
export NCCL_MIN_CTAS=32;
export NCCL_NCHANNELS_PER_NET_PEER=32;
export NCCL_NVLS_ENABLE=0
export TORCH_NCCL_AVOID_RECORD_STREAMS=1 

export TP_COMM_OVERLAP=False 
export MC_TP_OVERLAP_AG=False
export MC_TP_OVERLAP_RS=False
export MC_TP_OVERLAP_RS_DGRAD=False

export CUBLAS_FORCE_XMMA_KERNEL_INIT=DEVICE

export NVTE_RS_STRIDED_ATOMIC=2
export NVTE_FP8_DPA_BWD=1
export NVTE_FLASH_ATTN=0
export NVTE_FUSED_ATTN=1
export NVTE_FUSED_ATTN_CK=1
export NVTE_FUSED_ATTN_AOTRITON=1
export NVTE_DEBUG=0
export NVTE_DEBUG_LEVEL=0
export NVTE_USE_HIPBLASLT=1
export NVTE_USE_CAST_TRANSPOSE_TRITON=0
export NVTE_USE_OPTIMIZED_HIPIFIED_CAST_TRANSPOSE=1
export USE_TE_SWIGLU=1

# FAv3
export NVTE_CK_USES_BWD_V3=1        # enable dqdkdv bwd kernel
export NVTE_CK_V3_SPEC=1            # use specialized kernel
export NVTE_CK_V3_BF16_CVT=1        # Use Round to away from ZERO for numerical stability
export CK_FUSED_ATTN_LOG_CONFIG=0   # Diable logging for CK fused attn. Enabled for debugging only

export LORA_A2A=1
export POSSIBLE_USER_WARNINGS=0
export CUDNN_FRONTEND_ATTN_DP_WORKSPACE_LIMIT=0

export MAX_STEPS=1024
export TP=1
export PP=1
export CP=1
export SP=False
export VBOOST_VALUE=1
export MBS=1
export LR=0.0004
export MINIBS=1
export SKIP_EVALS=3
export VAL_CHECK_INTERVAL=384
export HYDRA_FULL_ERROR=1
export CUDA_DEVICE_MAX_CONNECTIONS=1

export FP8_DPA=0    
export FP8=True
export FP8_AMAX_ALGO=most_recent
export FP8_REDUCE_AMAX=False
export FP8_AMAX_HISTORY=4
export FP8_ACTIVATION=True

export ACG=full && export ACM=block && export ACL=21
export FUSED_SOFTMAX=0
export RMSNORM_CAST=0

export PT_TENSOR_VALIDATION=0
export PROFILE_RPD=0

export USE_HIPBLASLT=1
export TORCH_BLAS_PREFER_HIPBLASLT=1

export MLPERF_SUBMISSION_ORG="AMD"
export MLPERF_SUBMISSION_PLATFORM="MI300X"

export NVTE_USE_RMSNORM_TRITON=1
