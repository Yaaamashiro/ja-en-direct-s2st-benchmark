ARG FAIRSEQ_IMAGE=direct-s2st-fairseq:locked
FROM ${FAIRSEQ_IMAGE}
ENTRYPOINT ["s2st-benchmark", "vocoder", "unit"]
