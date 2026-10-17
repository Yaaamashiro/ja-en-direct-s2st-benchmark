# Translatotron 2: experimental native core

Implemented 2026-09-13 after explicit user approval to implement the core before
the real S2UT E2E gate. This changes work order, not completion criteria.
No production corpus, learned speech checkpoint, intelligible generated speech,
or translation metric has been verified. Voice preservation remains disabled.

## Components actually implemented

- Two stride-2 2-D source convolutions and a native relative-position Conformer
  encoder, kernel 32, with padding-aware batch normalization (architecture v2).
- Autoregressive stacked zoneout linguistic LSTM, with multihead shared acoustic
  attention driven by its hidden state. Predicts phonemes and EOS.
- Concatenated pre-softmax linguistic hidden state and the same attention context
  condition both the duration predictor and the acoustic synthesizer.
- Two-layer bidirectional LSTM duration predictor with positive outputs.
  Utterance-level squared duration-sum error needs no per-phone timing labels.
- A second two-layer bidirectional LSTM predicts positive Gaussian ranges.
  Differentiable normalized Gaussian upsampling preserves mel-loss gradients
  through predicted durations. Training rescales durations to target frame count;
  the duration loss uses the original, unscaled duration sum. EOS has no duration.
- Autoregressive zoneout acoustic LSTM with previous-mel prenet, linear mel head,
  and five-convolution residual postnet. Teacher forcing is used only in training.
- Masked pre/post mel L1, smoothed phoneme cross-entropy (weight 10), duration
  sum MSE (weight 1). Padding excluded from objectives and Gaussian normalization.
- Finite-loss and finite-gradient gates before an Adam update; clipping at 1.
  Atomic model/optimizer/RNG checkpoints, data hashes, per-update loss records.
  Resume rejects data, vocabulary, architecture and learning-rate changes.
- Greedy or optional beam reference-free phone decoding followed by duration/mel generation.
  Missing EOS or excessive duration fails explicitly, never silently truncates.
  Mels and their provenance feed the existing compatible mel-vocoder adapter.

## Implementation choices, not claims of exact paper reproduction

Primary references: [Translatotron 2, §3 and Appendix](https://arxiv.org/html/2107.08661v5)
and [Non-Attentive Tacotron](https://research.google/pubs/non-attentive-tacotron-robust-and-controllable-neural-tts-synthesis-including-unsupervised-duration-modeling/).

This is an independently implemented native PyTorch backend, not Google's
original code, not a fairseq-registered model, and not s2spect2_conformer.
The fixed fairseq revision is unchanged and supplies only the existing data
preparation/frontend. The native trainer does not read config_multitask.yaml
or fairseq's delta/specaugment transform settings. It uses its own linguistic
LSTM and loss, not the preparation file's historical transformer task.

Explicit Fisher, CoVoST2 and conversational presets specify the corresponding
Appendix dimensions, attention heads/context width, SpecAugment and optimizer
settings. `reference` aliases Fisher. Postnet and Conformer use masked batch
normalization. Gradient accumulation, torchrun/DDP, rank-specific RNG/buffer
checkpoint resume and optional teacher-forced dev validation are implemented.
Two-process CPU Gloo training/resume is tested; CUDA/NCCL is not yet verified.
See [reference alignment and remaining numerical uncertainty](REFERENCE_PARITY.md).

The corpus-compatible target mel is 16 kHz, 80 bins, 25 ms window, 10 ms hop;
the optional paper recipe uses 24 kHz/128-bin/50 ms/12.5 ms target settings.
Source and target specifications are independently fingerprinted.
The source uses the pinned fairseq log-mel frontend with source-only CMVN,
without deltas. Speaker conditioning and voice preservation remain intentionally
disabled for canonical single-speaker output. Reference-size training enables
SpecAugment; smoke disables it. Deterministic mini-batch cycling supports
--batch-size and exact resume. --warmup-updates defaults to zero for smoke.

## Running with a chosen environment

### Optional stabilized practical recipe

`--tt2-recipe stabilized_seconds_v1 --reproduction-mode paper_practical`
selects a **fresh** Fisher run with a seconds-valued duration predictor and
utterance MSE against `target_frames * target_hop_seconds`. The positive head
uses output scale 0.1 and a zero-weight/inverse-softplus-bias initialization at
0.08 seconds per acoustic token (including word separators, excluding EOS).
Gaussian upsampling and range prediction remain in frame coordinates; the
inference length limit is still measured in frames. No aligned durations,
forced EOS, new normalization layer, or different model dimensions are added.

Peak LR is 0.0003, with 400 warmup updates fixed from update zero, followed by
the existing inverse-square-root schedule. This is an explicit practical
deviation, not a correction claimed to match Google's unpublished code.
Weights remain mel/phone/duration=1/10/1, Adam/L2 and clipping are unchanged.
The effective global batch remains 1024, including after startup calibration.
Early stopping, full-dev validation, best-total-loss selection, and the absolute
1000-update cap are unchanged. See [Colab configuration](COLAB.md).

The default `legacy_frames` preserves the previous frame-valued head/objective,
initialization and Fisher recipe. Missing new ModelConfig fields in old native
v1/v2 checkpoints default to those legacy semantics. The new recipe is included
in the training fingerprint, and its unit/hop/initialization are checkpointed.
It does not migrate old model weights, Adam moments, monitor anchors or history
into the new objective. Prepared features/labels are reusable without alteration.
New checkpoints restore the model/Adam/RNG and schedule exactly on resume;
mid-run shortening is refused for this recipe. Old schedule overlays retain
their previous behavior and are not applied to stabilized runs.

Use the existing pinned fairseq container for real source feature extraction;
torch and torchaudio remain pinned at 2.7.1. Select CORPUS_ROOT,
EXPERIMENT_DATA_ROOT, RUNS_ROOT and CACHE_ROOT using the existing .env interface.
First run phonemize and prepare with the pinned eSpeak/fairseq tools.

```sh
python -m direct_s2st.translatotron2.train \
  --data-root /benchmark/translatotron2/fairseq \
  --run-root /runs/translatotron2-smoke \
  --model-size smoke --device cpu --max-updates 2 --save-interval-updates 1

python -m direct_s2st.translatotron2.train \
  --data-root /benchmark/translatotron2/fairseq \
  --run-root /runs/translatotron2-smoke \
  --model-size smoke --device cpu --max-updates 3 \
  --restore-file /runs/translatotron2-smoke/checkpoints/checkpoint_last.pt

python -m direct_s2st.translatotron2.infer \
  --data-root /benchmark/translatotron2/fairseq \
  --common-root /benchmark/common --run-root /runs/translatotron2-smoke \
  --predictions /runs/translatotron2-smoke/predictions/mels.jsonl --device cpu
```

A two-update model need not emit EOS or meaningful speech: decoding failure is
reported, not concealed with forced durations/phones or dummy outputs. Use
`--device cuda` only on a provisioned compatible host. `--model-size reference`
is available explicitly; default training never starts that large model.
CLI YAML commands expose both options and the run-root determines checkpoint
location. Do not change model size or learning rate during resume.

## Verification and remaining work

### Optional evaluation-statistics recipe

`stabilized_pooled_v2` is a separate fresh `paper_practical` run. It retains the
seconds-v1 training objective/optimizer/schedule and fits postnet evaluation
statistics to a deterministic, SHA256-ID-ordered train subset (up to128, batch4).
Raw teacher-forced eval Mels are cached on CPU once. Five postnet layers are
calibrated sequentially with FP64 pooled valid-frame moments and unbiased stored
variance, not an average of per-batch variances. Dev/test never fit statistics.

Validation uses these statistics; all training buffers, RNG and mixed module
modes are restored even on exceptions. Native model/Adam states remain training
states. Evaluation buffers, train IDs, update and parameter SHA are embedded
separately in each evaluated/final checkpoint. Native loading for resume never
applies them. Inference applies only the selected checkpoint's matching embedded
statistics and fails closed for missing/stale/foreign metadata. Best remains the
minimum teacher-forced Dev **total** loss, not phone loss or source-only quality.
Monitoring interval/policy remain unchanged; the absolute hard cap is1000.
Single-process FP32 only; no automatic migration of prior runs. Normal cell5
handles new/resumed runs. Paper-exact, legacy, seconds-v1 and S2UT are unchanged.

Local small-model evidence does not establish full-corpus convergence, A100
performance, the adequacy of128 calibration examples, intelligibility, BLEU/MOS,
or equivalence to Google's unpublished implementation. Phone prediction is
unchanged by this evaluation-only operation. See COLAB.md for opt-in controls.

CPU tests use explicitly synthetic feature tensors, not fake production data.
They execute actual forward/backward and Adam updates, verify mel gradients
reach linguistic/shared-attention/duration/range modules, perturb phone states
to change acoustic output, test padded losses and Gaussian masks, save/load
real updated model tensors, compare resumed versus uninterrupted updates
bit-for-bit, and execute the trainer through update 2 and resumed update 3.
The EOS-interface test deliberately controls the phone output head; it is not
evidence of learned speech generation.

Still required: real frontend/container integration, real-data optimization and
checkpoint generation, unforced decoding, compatible trained mel vocoder and
WAV/ASR evaluation, runtime/memory/quality review, and numerical comparison with
the original results. Vocoder fitting and the four-system orchestration are now
implemented; their real-data execution, S2UT training/E2E and Cascade E2E remain
unverified. See PIPELINE_COMPLETION.md. The implemented workflow is not evidence
that the four-system real-data benchmark has already succeeded.
