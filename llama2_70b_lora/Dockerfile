ARG FROM_IMAGE_NAME=us-east1-docker.pkg.dev/gpu-launchpad-playground/gke-llm/dp-repro:latest 
#nvcr.io/nvidia/pytorch:24.08-py3 
#24.01
FROM ${FROM_IMAGE_NAME}

WORKDIR /workspace/ft-llm
ADD . /workspace/ft-llm

#RUN pip install -r requirements.txt
#RUN pip install flash-attn==2.4.1 --no-build-isolation
#RUN git clone https://github.com/mlperf/logging.git mlperf-logging
#RUN pip install -e mlperf-logging
