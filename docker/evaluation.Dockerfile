ARG FAIRSEQ_IMAGE=direct-s2st-fairseq:locked
FROM ${FAIRSEQ_IMAGE}
COPY requirements/evaluation.txt /tmp/evaluation.txt
RUN python -m pip install --no-cache-dir -r /tmp/evaluation.txt
