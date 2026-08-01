ARG BASE_IMAGE=nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04@sha256:5f0d2d827f6436b3cb7468fd8acbdc8c1d41261614e579ae49afe6141da51133
FROM ${BASE_IMAGE}
ARG FAIRSEQ_COMMIT=3d262bb25690e4eb2e7d3c1309b1e9c406ca4b99
LABEL org.opencontainers.image.fairseq-revision=${FAIRSEQ_COMMIT}
ENV DEBIAN_FRONTEND=noninteractive PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends \
    git python3 python3-pip libsndfile1 sox \
    && rm -rf /var/lib/apt/lists/*
RUN python3 -m pip install --no-cache-dir \
    torch==2.7.1 torchaudio==2.7.1 \
    --index-url https://download.pytorch.org/whl/cu126
WORKDIR /workspace
COPY pyproject.toml /workspace/pyproject.toml
COPY src /workspace/src
COPY requirements/preparation.txt /workspace/requirements/preparation.txt
RUN python3 -m pip install --no-cache-dir /workspace
RUN python3 -m pip install --no-cache-dir -r /workspace/requirements/preparation.txt
COPY third_party/fairseq /opt/fairseq
COPY patches/fairseq /opt/fairseq-patches
RUN find /opt/fairseq-patches -type f -name '*.patch' -exec git -C /opt/fairseq apply {} \; \
    && python3 -m pip install --no-cache-dir --editable /opt/fairseq
ENTRYPOINT ["s2st-benchmark"]
