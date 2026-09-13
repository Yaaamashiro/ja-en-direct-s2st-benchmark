ARG FAIRSEQ_IMAGE=ja-en-direct-s2st-benchmark-fairseq:locked
FROM ${FAIRSEQ_IMAGE}
ENTRYPOINT ["s2st-benchmark", "vocoder", "mel"]
