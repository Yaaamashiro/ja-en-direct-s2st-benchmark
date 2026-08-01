ARG FAIRSEQ_IMAGE=direct-s2st-fairseq:locked
FROM ${FAIRSEQ_IMAGE}
COPY requirements/cascade.txt /tmp/cascade.txt
RUN python -m pip install --no-cache-dir -r /tmp/cascade.txt
