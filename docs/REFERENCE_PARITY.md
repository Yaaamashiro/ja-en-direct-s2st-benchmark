# TT2 reference alignment and verification boundary

This is an independent implementation, not the original authors' released model
or a claim of reproduced translation quality. Voice preservation is explicitly
disabled for canonical-speaker output, as permitted by the project scope.

References: [Translatotron 2, Appendix A](https://arxiv.org/html/2107.08661v5),
[Conformer](https://arxiv.org/abs/2005.08100), and
[Non-Attentive Tacotron](https://research.google/pubs/non-attentive-tacotron-robust-and-controllable-neural-tts-synthesis-including-unsupervised-duration-modeling/).

## Architecture revision 2

The implementation now includes 2-D factor-four subsampling, relative-position
attention, kernel-32 convolution, padding-aware batch normalization, separate
shared-attention hidden/output dimensions, and normalized postnet convolutions.
Linguistic hidden state plus shared context conditions duration/range prediction
and autoregressive acoustic synthesis. No aligned phone-duration labels are
required. Optional beam search is a decoding choice, not a paper-parity claim;
neither greedy nor beam decoding forces EOS to manufacture successful output.

Fisher, CoVoST2 and conversational presets are available. Their peak learning
rates are 0.0042, 0.0022 and 0.0033; warmups are 10000, 20000 and 10000 updates.
Recipe Adam uses coupled L2 regularization of 1e-6. The reference global batch
sizes are 1024, 768 and 768. Checked-in commands deliberately remain bounded at
two updates with small batches; they do not implicitly launch paper-scale work.

Effective batch size = world size × per-rank batch size × update frequency.
Loss accumulation uses global valid frame/phone/utterance counts. Batch
normalization statistics remain microbatch- and rank-local, not SyncBatchNorm;
accumulation does not make these statistics equivalent to a single large batch.
Samples cycle deterministically. Initialization, asymmetric SAME padding,
Gaussian range parameterization and preprocessing have not been numerically
audited against original-author weights. These are documented implementation
choices, not evidence of numerical equivalence.

## Frontends and suite selection

The default benchmark frontend remains 16 kHz/80-bin target mel. Explicit
`--tt2-recipe fisher`, `covost2` or `conversational` selects independent source
and target specifications, the corresponding trainer, and a compatible 24 kHz
mel vocoder with hop 300. Target settings are 128 bins, 50 ms window and 12.5 ms
hop. Japanese source input uses 16 kHz/80 bins, 25 ms window and 10 ms hop;
this is an adaptation, not the original Fisher 8 kHz or CoVoST2 48 kHz corpus.
Resampling a 16 kHz target to 24 kHz cannot recover missing bandwidth.
Use a new derived-data root when switching feature specifications.

```sh
# Plan only; inspect before explicitly adding --execute.
python scripts/smoke/suite.py --env-file .env --name paper-smoke --tt2-recipe fisher

# Example on a separately provisioned Linux GPU host, prepared train/dev only.
torchrun --standalone --nproc-per-node=2 -m direct_s2st.translatotron2.train \
  --data-root /benchmark/translatotron2/fairseq --run-root /runs/tt2-distributed \
  --model-size fisher --device cuda --max-updates 2 --batch-size 1 \
  --update-freq 2 --learning-rate 0.0042 --warmup-updates 10000 \
  --l2-regularization 0.000001 --validate-interval 1
```

World size, accumulation and optimization settings are locked on resume.
Checkpoints retain each rank's RNG and normalization buffers. Legacy v1
checkpoints load through the old architecture; new training does not silently
reinterpret their weights as v2. Dev validation is teacher-forced loss only,
not reference-free translation evaluation.

## Evidence

CPU tests cover actual optimizer updates, padded batches, feature dimensions,
gradient paths, legacy loading, beam interface, dev immutability and checkpoint
resume. A real two-process Gloo test verifies bitwise-identical resumed updates.
Its inputs are synthetic feature tensors, not speech benchmark results.
Linux Docker/frontend integration, CUDA/NCCL training, trained-vocoder speech,
unforced real-data decoding and four-system WAV/ASR evaluation remain NOT_RUN
until corpus, compatible runtime and learned artifacts are supplied.
