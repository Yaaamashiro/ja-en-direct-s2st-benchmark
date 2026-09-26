# Audit and implementation report

Date: 2026-09-13 (Asia/Tokyo). Scope: benchmark repository only.
No corpus files were edited or regenerated. No trained weights, audio, features,
predictions or large results were added to Git. No commit/push was performed.

**Not complete as a four-system real-data benchmark.** The user clarified during
implementation that the execution environment is undecided and should be
configurable. The `.env`/Docker-context smoke interface supports that request.
The real-model execution criteria remain unverified, not waived or claimed PASS.

## Initial Audit (before source changes)

### Common Dataset

- Schema: pair_id, split, corpus, source_id, ja/en audio, raw/normalized/TTS texts,
  durations and SHA-256. All six requested text fields already existed.
- Audio behavior: relative paths tried corpus_root then manifest parent; all
  paths became absolute. Missing Colab absolute paths were not rebased.
- Problems: no portable common path representation, unsafe manifest-relative
  fallback for the requested policy, no output-root check in import API.

### S2UT

- Main task: speech_to_speech, target-is-code, speech_to_unit criterion,
  s2ut_transformer architecture (base dimensions, not Fisher dimensions).
- Extraction: pinned Transformers HuBERT Base, layer 6, fairseq KM100 artifact
  with fixed checksum. Consecutive-unit reduction exists.
- Auxiliary tasks: none; no linguistic TSV/dictionaries or multitask CLI flag.
- Inference config referenced missing s2ut.fairseq_infer.
- Unit vocoder config referenced a missing infer module. Its hydra_train command
  was not an actual configured Code HiFi-GAN training workflow.

### Translatotron 2

- Existing pinned eSpeak phonemization, phoneme dictionary, mel ZIP and
  target_phoneme transformer/is_first_pass_decoder config were present.
- Contrary to the possible absence anticipated in the request, **the pinned
  submodule DOES register s2spect2_conformer** in
  fairseq/models/speech_to_speech/s2s_conformer_translatotron2.py.
- It passes translation-decoder hidden states to the second pass, but uses
  TTSTransformerDecoder; it has no required duration predictor/objective or
  duration-based synthesizer. It is not accepted as the requested core.
- Existing speech_to_spectrogram criterion did not supply the model's required
  prev_output_tokens_mt. Switching to the 2-pass criterion alone would not repair
  the scientific architecture mismatch.
- Both the custom inference adapter and mel vocoder infer module were absent.

### Cascade / Evaluation

- Cascade code existed with immutable Whisper/NLLB/Qwen3-TTS revisions.
- ASR-BLEU, common normalization, runtime, ECAPA similarity and optional BLASER
  existed. Actual model runs were not demonstrated.

### fairseq and environment

- Submodule initially unpopulated; initialized at the already pinned commit
  `3d262bb25690e4eb2e7d3c1309b1e9c406ca4b99`. No revision upgrade or source patch
  was made to fairseq. The existing librosa keyword patch remains unchanged.
- Local Docker executable/standard installation not found; no corpus accepted
  manifest/audio/model checkpoint was found in the scoped workspace.
- NVIDIA GTX 1650, 4 GB, driver 555.97 reporting CUDA 12.5 observed; this does not
  establish support for the pinned CUDA 12.6.3 container or training capacity.
- Isolated `.venv` created for development tests with repository-pinned direct
  dependencies. It is excluded from Git and is not a production training runtime.

### Implementation references

- [S2UT paper](https://arxiv.org/abs/2107.05604v2).
- [Pinned S2UT workflow and multitask values](https://github.com/facebookresearch/fairseq/blob/3d262bb25690e4eb2e7d3c1309b1e9c406ca4b99/examples/speech_to_speech/docs/direct_s2st_discrete_units.md).
- [Pinned model actually named s2spect2_conformer](https://github.com/facebookresearch/fairseq/blob/3d262bb25690e4eb2e7d3c1309b1e9c406ca4b99/fairseq/models/speech_to_speech/s2s_conformer_translatotron2.py).
- [Translatotron 2 paper, §3](https://arxiv.org/html/2107.08661v5).
- Pinned local SingleTaskConfig, speech_to_speech dataset/criterion, CodeGenerator
  and HiFi-GAN Generator were inspected for interfaces and duration behavior.

## Changes implemented

- Safe path resolution, unique audio/16k suffix rebasing, portable relative fields
  and runtime common-reader integration. Corpus remains read-only.
- All three S2UT auxiliary datasets and train-only dictionaries, deterministic
  character tokenization, explicit attachment/weight settings and data-lock
  metadata. Reject missing/extra IDs, unknown/empty tokens, disabled losses,
  inconsistent target-vs-CTC labels and text-vs-TTS content differences.
- Main train config requires multitask. Training preflight checks files, units,
  dictionaries, architecture resolution, layer bounds and CTC feasibility.
  Training checkpoint validation requires actual optimizer update/state, finite
  parameter tensors and all three auxiliary parameter groups.
- Actual fairseq generation adapter with complete numeric-ID mapping to the
  common test split; units remain intermediate predictions until vocoding.
- Checkpoint-backed unit and mel generator adapters, duration prediction for
  reduced units, full mel compatibility gate, finite/non-silent WAV readback and
  checkpoint/config/input hashes. These neural adapters are **not runtime verified**.
- Native experimental Translatotron 2 Conformer/linguistic LSTM/shared attention/
  duration-based autoregressive synthesizer, training and reference-free mel
  inference are implemented. See TRANSLATOTRON2.md for exact differences.
- Configurable Docker drivers: original S2UT smoke and new 25-stage four-system
  suite with train-only vocoder fitting, resume and real-artifact acceptance.
  Both are plan-only by default; neither has run on a provisioned real-data host.
- Cascade and evaluation model logic preserved; Cascade common input reader
  changed only to support portable corpus paths.

## S2UT Status

PASS here means only the explicitly named CPU fixture/check actually ran.
NOT_RUN is a **FAIL of the original completion criterion**, not a model failure
measurement. The fixture extractor used by CPU preparation tests is not HuBERT
and is not offered as production inference.

| Check | Status | Evidence / limitation |
| --- | --- | --- |
| Main speech→unit real data | NOT_RUN | Real speech/checkpoints/execution environment not supplied |
| Source linguistic auxiliary data | PASS (CPU fixture) | IDs, deterministic labels, dictionary checks executed |
| Target linguistic auxiliary data | PASS (CPU fixture) | Same |
| Decoder target CTC data | PASS (CPU fixture) | Same, plus equality with target labels |
| Multitask config | PASS (CPU fixture) | Three reference task weights/layers tested |
| Training forward/backward | NOT_RUN | No actual fairseq model built |
| Finite main/aux losses and gradients | NOT_RUN | Must be checked on real tiny training |
| Tiny training | NOT_RUN | Driver configured for 2 updates, never executed |
| Real checkpoint | NOT_RUN | Validator added, no fabricated weights generated |
| Inference | NOT_RUN | Parser/ID mapping tested, model generation not run |
| Unit vocoder | NOT_RUN | Contract checks tested, real-checkpoint test opt-in |
| Valid model-generated WAV | NOT_RUN | CPU sine serialization test is not neural vocoding |
| Real evaluation / E2E | NOT_RUN | CPU mocked-ASR evaluation tests are not E2E |

## Translatotron 2 Status

The user authorized core implementation before real S2UT E2E on 2026-09-13.
Voice preservation remains disabled. See [core report](TRANSLATOTRON2.md).

| Check | Status |
| --- | --- |
| Native required-component core | IMPLEMENTED — experimental, not fairseq-registered |
| Speech encoder, linguistic LSTM, shared attention | PASS (synthetic CPU forward/backward) |
| Linguistic/acoustic-context → duration synthesizer | PASS (mel gradients and conditioning perturbation) |
| Duration prediction / utterance duration objective | PASS (synthetic CPU, no per-phone timing labels) |
| Pre/post mel and phoneme losses | PASS (finite, masked synthetic CPU) |
| Adam update, checkpoint save/load, exact resume | PASS (synthetic CPU) |
| Trainer CLI update 2 → resumed update 3 | PASS (synthetic dataset adapter) |
| Reference-free inference interface / EOS limits | PASS (controlled-head CPU fixture only) |
| Real speech frontend / trained unforced decoding | NOT_RUN |
| Mel vocoder / WAV / ASR / evaluation | NOT_RUN |
| Paper-exact reproduction | NOT ESTABLISHED — deviations explicitly documented |

The native backend does not invoke the upstream duration-free TTSTransformerDecoder.
No real-speech learning or generation result has been fabricated.

## Final Report

| Completion area | Status |
| --- | --- |
| Common manifest path portability | PASS (CPU file fixtures, foreign-path and relocation tests) |
| S2UT multitask preparation | PASS (CPU fixtures only) |
| S2UT training / inference / vocoder / E2E | FAIL of completion criteria — NOT_RUN |
| Translatotron 2 core / optimizer / checkpoint | PASS for synthetic CPU mechanics; experimental implementation |
| Translatotron 2 real-data training / inference / E2E | NOT_RUN |
| Mel neural vocoder | NOT_RUN |
| Cascade E2E | NOT_RUN |
| Evaluation implementation regression tests | PASS (CPU fixtures) |
| Evaluation using actual generated speech and ASR | NOT_RUN |

### Tests and commands

- Initialized pinned submodule with `git submodule update --init --recursive`.
- Inspected GPU using `nvidia-smi`; checked Docker availability and scoped files.
- Installed pytest 8.4.2, PyYAML 6.0.2, NumPy 1.26.4, sacrebleu 2.5.1 and soundfile
  0.13.1 into ignored development `.venv`; all are existing pinned project values.
- Initial pytest attempt hit sandbox temporary-directory permissions. Authorized
  execution with the normal user's temporary directory resolved that environment
  problem. An initial evaluation regression run failed because sacrebleu was not
  installed; installing the existing required version resolved it.
- Final code test: `.venv/Scripts/python.exe -m pytest -q -m 'not gpu' --tb=short -p no:cacheprovider`:
  **92 passed, 3 deselected** (latest follow-up). No model download or GPU required.
- Added pinned torch 2.7.1+cpu and torchaudio 2.7.1+cpu to the ignored development
  environment. Eight new CPU tests exercise the TT2 core/trainer with synthetic
  tensors. Without torch/torchaudio these optional tests skip explicitly.
- `python scripts/smoke/run.py --env-file .env.example --limit 3 --max-updates 2`:
  **PLAN_ONLY**, ten stages **NOT_RUN**; no Docker/model execution.
- Normal `git diff --check`: no whitespace errors (Windows line-ending warnings
  only). A check with an inappropriate temporary core.autocrlf=false override
  reported existing CRLF as trailing whitespace; no source fault was inferred.

### Scientific deviations and remaining limitations

S2UT: retained HuBERT layer6/KM100/reduction and base 512-dimension architecture;
Fisher reference uses the smaller Fisher variant. Japanese/English codepoint
labels preserve case and punctuation. Unknown validation/test characters fail
instead of entering the training dictionary. Text/TTS checks are deliberately
strict and may require reviewed normalization changes for real corpus numbers,
symbols or Japanese readings. CTC zero_infinity=false exposes impossible
alignments instead of hiding their losses; real training stability is untested.
These decisions can affect accuracy, optimization and tiny-subset feasibility.

Translatotron 2: the native experimental core/objectives are implemented and CPU
mechanics tested. This is not paper-exact reproduction. Dimension, frontend,
attention, subsampler, training and inference differences are enumerated in
TRANSLATOTRON2.md. Real-speech convergence and generation remain untested.

Both neural vocoders require supplied, compatible real checkpoints. Metadata
checks cannot prove that declared mel metadata truthfully describes a checkpoint;
provenance must be verified. Train-only unit/mel vocoder fitting is now implemented
with pinned fairseq generators and official HiFi-GAN MPD/MSD losses. A CPU test
performed real optimizer updates with the pinned generator and one official
period discriminator; the complete MPD/MSD and CodeGenerator training remain
unexecuted on real data. Per-sample failure journals, artifact hashes and safe
resume now cover TT2, S2UT generation, vocoding, Cascade and evaluation.
Cross-run latency parity remains a documented measurement limitation.

Real-data per-stage loss/gradient checks and checkpoint resume, Docker builds,
HuBERT parity, neural outputs, speaker-model downloads and all four real E2E
paths remain unverified. The latest user instruction makes the environment
selectable; it does not turn these unexecuted checks into success.

### Modified files

```text
.gitignore
README.md
DESIGN.md
pyproject.toml
configs/s2ut/{train,infer}.yaml
configs/translatotron2/{prepare,train,infer}.yaml
configs/vocoder/{unit,mel}.yaml
docker/{unit-vocoder,mel-vocoder}.Dockerfile
scripts/smoke/README.md
src/direct_s2st/{cli,config,inference,training}.py
src/direct_s2st/manifests/{schema,import_corpus,validate}.py
src/direct_s2st/s2ut/{extract_units,prepare_fairseq,reduce_units}.py
src/direct_s2st/translatotron2/{phonemize,prepare_fairseq,train,infer}.py
src/direct_s2st/cascade/pipeline.py
tests/unit/test_experiment_config.py
```

### Added files

```text
.env.example
docs/REPRODUCTION.md
docs/IMPLEMENTATION_STATUS.md
scripts/smoke/run.py
src/direct_s2st/manifests/{paths,reader}.py
src/direct_s2st/s2ut/{multitask,preflight,checkpoint,fairseq_infer}.py
src/direct_s2st/s2ut/multitask.yaml
src/direct_s2st/translatotron2/status.py
src/direct_s2st/vocoders/inference.py
src/direct_s2st/vocoders/{unit,mel}/infer.py
tests/unit/{test_completion_validation,test_smoke_plan}.py
tests/integration/test_vocoder_checkpoints.py
```

### Native TT2 follow-up files

Added `translatotron2/{model,data,engine}.py`, `tests/unit/test_translatotron2_core.py`
and `docs/TRANSLATOTRON2.md`; connected train/infer/config/CLI wrappers and updated
README, DESIGN and reproduction instructions. No corpus modifications or push.

### Completion-implementation follow-up

- Implemented native vocoder fitting/checkpoint resume, shared per-sample journals,
  S2UT main/aux loss and gradient audit, TT2 attention width512/SpecAugment/batching/
  selectable learning-rate schedule, then-23-stage suite and real-artifact acceptance.
- Vendored only official HiFi-GAN discriminators/losses at commit
  4769534d45265d52a904b850da5a622601885777, including MIT license.
- New modules: vocoders/{train,discriminators}.py, journal.py,
  s2ut/fairseq_train.py, translatotron2/batching.py, evaluation/acceptance.py.
  New configs: vocoder/{unit,mel}-generator.json.
  New driver: scripts/smoke/suite.py. Tests: test_remaining_implementation.py.
- Full CPU result: 82 passed, 3 GPU tests deselected. One upstream weight_norm
  deprecation warning retained for pinned-state compatibility. No actual-speech
  result is inferred from synthetic tensor tests.
- Historical suite.py plan: PLAN_ONLY, 23 stages, ending in acceptance then comparison.
  Vocoder trainer module --help succeeded; fairseq submodule remains pristine;
  git diff --check passed (only CRLF conversion warnings).
- During verification, automatic command approval briefly hit a service usage
  limit; after the user requested resume, authorized execution succeeded.
- All four real-data E2E criteria still remain NOT_RUN. See
  [complete workflow and limitations](PIPELINE_COMPLETION.md).

### Reference-component completion (2026-09-13)

- Architecture v2: relative-position Conformer, kernel32, 2-D subsampling,
  masked normalization, explicit attention context/head dimensions.
- Fisher/CoVoST2/conversational presets and separate source/target frontend
  fingerprints; optional 24 kHz/128-bin target and matching vocoder recipe.
- Beam decoding, weighted gradient accumulation, torchrun/DDP, rank-local
  checkpoint state, and deterministic teacher-forced dev validation.
- Legacy v1 checkpoint loading remains explicit; v2 training does not silently
  reinterpret old weights. Fatal Cascade accelerator errors are journaled.
- Full CPU regression: 92 passed, 3 GPU tests deselected, one pinned upstream
  weight_norm deprecation warning. This includes real two-process Gloo optimizer
  updates with bitwise-equivalent resumed state, using synthetic feature inputs.
- Historical paper suite plan: PLAN_ONLY, 23 stages. Git diff check passed; fixed fairseq
  submodule pristine. No corpus edits, commit or push.
- Real-data four-system E2E remains NOT_RUN. CUDA/NCCL and numerical parity
  with original-author results remain unverified, not inferred from CPU tests.
  See [reference mapping](REFERENCE_PARITY.md).

## S2T→TTS / four-system extension

The required final systems are `s2ut`, `translatotron2`, `cascade`, and `s2t_tts`.
The new baseline uses pinned Whisper large-v3 translation and the existing QwenTTS,
without changing the original turbo ASR → NLLB → TTS model settings.
See [S2T→TTS implementation and verification](S2T_TTS.md) for commands, Colab,
new files, dependency limitations, and unverified GPU/E2E items.
Historical test counts below/above describe their original runs, not current validation.
