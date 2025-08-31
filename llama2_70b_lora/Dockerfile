ARG FROM_IMAGE_NAME=nvcr.io/nvidia/pytorch:24.08-py3  
#24.01-py3
FROM ${FROM_IMAGE_NAME}
ARG DEBIAN_FRONTEND=noninteractive
ENV STAGE_DIR=/tmp
RUN mkdir -p ${STAGE_DIR}
RUN echo "ClientAliveInterval 30" >> /etc/ssh/sshd_config
RUN cp /etc/ssh/sshd_config ${STAGE_DIR}/sshd_config && \
        sed "0,/^#Port 22/s//Port 22/" ${STAGE_DIR}/sshd_config > /etc/ssh/sshd_config

#RUN apt-get update && apt-get install -qy --no-install-recommends \
#	wget build-essential libncursesw5-dev libssl-dev libsqlite3-dev git \
#	tk-dev libgdbm-dev libc6-dev libbz2-dev libffi-dev zlib1g-dev lzma \
#	liblzma-dev libbz2-dev neovim numactl pdsh ninja-build curl pciutils libopenmpi-dev && \
#	rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/ft-llm
ADD . /workspace/ft-llm

RUN pip install -r requirements.txt
#RUN pip install flash-attn==2.4.1 --no-build-isolation