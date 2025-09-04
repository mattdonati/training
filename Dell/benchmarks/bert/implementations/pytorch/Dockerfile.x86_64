# Copyright (c) 2019-2025, NVIDIA CORPORATION. All rights reserved.
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

ARG FROM_IMAGE_NAME=nvcr.io/nvidia/pytorch:25.04-py3

FROM ${FROM_IMAGE_NAME}

ARG GIT_COMMIT_ID
ENV GIT_COMMIT_ID=$GIT_COMMIT_ID

RUN git config --global user.name "a" && \
    git config --global user.email "a"

ENV PIP_CONSTRAINT=""

WORKDIR /workspace/bert
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


# Preprocessing
WORKDIR /workspace
RUN cd /workspace && git clone https://github.com/attardi/wikiextractor.git
RUN cd /workspace/wikiextractor && git checkout e4abb4cbd019b0257824ee47c23dd163919b731b

## TransformerEngine mlperf branch
ARG TE_REVISION=te1_blackwell_ea
ENV CUSTOM_TE_REVISION ${TE_REVISION}

RUN if [ "${TE_REVISION}" != SKIP ]; then \
      git clone https://github.com/NVIDIA/TransformerEngine.git transformerengine && \
      cd transformerengine && \
      git checkout ${TE_REVISION} && \
      echo TE_COMMIT_HASH=$(git rev-parse HEAD) && \
      echo $(git rev-parse HEAD) > /TE_COMMIT_HASH.env && \
      git submodule init && git submodule update && \
      git -C 3rdparty/flashattn_internal checkout 9c638b57 && \
      git -C 3rdparty/cudnn-frontend checkout 91b7532f && \
      git -C 3rdparty/googletest checkout f8d7d77c && \
      NVTE_CUDA_ARCHS="100" MAX_JOBS=4 NVTE_FRAMEWORK=pytorch pip install --force-reinstall --no-deps . \
    ; fi

# Install BERT
ENV BERT_PREP_WORKING_DIR /workspace/bert/data
WORKDIR /workspace/bert
COPY . .

ENV PYTHONPATH "/workspace/bert"

RUN cd /workspace/bert/mhalib && python setup.py build && cp build/lib*/mhalib* ../
RUN cd /workspace/bert/fast_dropout_lib && pip install .

WORKDIR /workspace/bert

