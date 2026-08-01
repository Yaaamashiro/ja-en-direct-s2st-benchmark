# syntax=docker/dockerfile:1.7
ARG BASE_IMAGE=python:3.11.9-slim-bookworm@sha256:8fb099199b9f2d70342674bd9dbccd3ed03a258f26bbd1d556822c6dfc60c317
FROM ${BASE_IMAGE}
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /workspace
COPY pyproject.toml /workspace/pyproject.toml
COPY src /workspace/src
RUN python -m pip install --no-cache-dir .
ENTRYPOINT ["s2st-exp"]
