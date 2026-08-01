ARG BASE_IMAGE=nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04@sha256:5f0d2d827f6436b3cb7468fd8acbdc8c1d41261614e579ae49afe6141da51133
FROM ${BASE_IMAGE} AS espeak-builder
ARG ESPEAK_NG_COMMIT=4870adfa25b1a32b4361592f1be8a40337c58d6c
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates cmake g++ gcc git make \
    && rm -rf /var/lib/apt/lists/*
RUN git init /tmp/espeak-ng \
    && git -C /tmp/espeak-ng remote add origin https://github.com/espeak-ng/espeak-ng.git \
    && git -C /tmp/espeak-ng fetch --depth 1 origin ${ESPEAK_NG_COMMIT} \
    && git -C /tmp/espeak-ng checkout --detach FETCH_HEAD \
    && cmake -S /tmp/espeak-ng -B /tmp/espeak-ng/build \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/opt/espeak-ng \
        -DUSE_LIBSONIC=OFF \
        -DUSE_MBROLA=OFF \
        -DUSE_SPEECHPLAYER=OFF \
    && cmake --build /tmp/espeak-ng/build --parallel \
    && cmake --install /tmp/espeak-ng/build

FROM ${BASE_IMAGE}
ARG FAIRSEQ_COMMIT=3d262bb25690e4eb2e7d3c1309b1e9c406ca4b99
ARG ESPEAK_NG_COMMIT=4870adfa25b1a32b4361592f1be8a40337c58d6c
LABEL org.opencontainers.image.fairseq-revision=${FAIRSEQ_COMMIT}
LABEL org.opencontainers.image.espeak-ng-revision=${ESPEAK_NG_COMMIT}
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH=/opt/espeak-ng/bin:${PATH} \
    LD_LIBRARY_PATH=/opt/espeak-ng/lib:${LD_LIBRARY_PATH} \
    ESPEAK_DATA_PATH=/opt/espeak-ng/share/espeak-ng-data
COPY --from=espeak-builder /opt/espeak-ng /opt/espeak-ng
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates git python3 python3-pip libsndfile1 sox \
    && rm -rf /var/lib/apt/lists/*
RUN espeak-ng --version 2>&1 | grep -F "1.52.0"
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
