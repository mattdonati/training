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

export DGXNNODES=18
export DGXNGPU=4
export BATCHSIZE=8

source $(dirname ${BASH_SOURCE[0]})/config_gbs_576.sh

export FLASH_ATTENTION=${FLASH_ATTENTION:-False}
export USE_TE_DPA=${USE_TE_DPA:-True}
export NVTE_FLASH_ATTN=0
export NVTE_FUSED_ATTN=1



export WALLTIME_RUNANDTIME=50
export WALLTIME=$((5 + ${NEXP:-1} * ($WALLTIME_RUNANDTIME + 5)))
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )

# Load default settings
source $(dirname ${BASH_SOURCE[0]})/config_common.sh

# Load GB200 specific settings
source $(dirname ${BASH_SOURCE[0]})/config_GB200_common.sh
