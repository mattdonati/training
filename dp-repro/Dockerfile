FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -qy --no-install-recommends \
	wget build-essential libncursesw5-dev libssl-dev libsqlite3-dev git \
	tk-dev libgdbm-dev libc6-dev libbz2-dev libffi-dev zlib1g-dev lzma \
	liblzma-dev libbz2-dev neovim numactl pdsh ninja-build curl pciutils libopenmpi-dev && \
	rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y rsync \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update -y && apt-get install -y curl gnupg && \
    echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" \
      > /etc/apt/sources.list.d/google-cloud-sdk.list && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg && \
    apt-get update -y && apt-get install -y google-cloud-cli python3-crcmod && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN wget https://www.python.org/ftp/python/3.11.12/Python-3.11.12.tgz && \
	tar xvf Python-3.11.12.tgz && rm Python-3.11.12.tgz && \
	cd Python-3.11.12 && ./configure --enable-optimizations && make install && \
	rm -rf /app/Python-3.11.12 && ln -s "$(which python3)" /usr/local/bin/python

RUN curl -sSL https://install.python-poetry.org | python3 -
RUN echo 'export PATH="/root/.local/bin:$PATH"' >> ~/.bashrc
ENV PATH="/root/.local/bin:$PATH"
RUN ln -s /root/.local/bin/poetry /usr/local/bin/poetry

RUN poetry self add keyrings.google-artifactregistry-auth && \
    poetry config virtualenvs.create false

COPY app/pyproject.toml app/poetry.lock /app/
#RUN --mount=type=cache,target=/root/.cache/pypoetry \
#    --mount=type=secret,id=gcp-key \
#    GOOGLE_APPLICATION_CREDENTIALS="/run/secrets/gcp-key" \
RUN poetry lock
RUN  poetry install --no-root --only main
COPY app/* /app/
