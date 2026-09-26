# Configurable execution environment

The execution environment has not been chosen. CPU development checks have run;
real-model E2E has not. No Docker installation, remote machine, or cloud job is
created automatically by this repository.

## Environment selection

On the selected Linux GPU host, clone this repository at the reviewed revision
and initialize the pinned submodule:

```sh
git submodule update --init --recursive
```

Create a local `.env` using `.env.example` as a guide. Set these paths on the
Docker daemon's host, not paths on a different laptop:

| Variable | Contents |
| --- | --- |
| CORPUS_ROOT | Existing corpus, mounted read-only; includes production/manifests/releases/accepted.jsonl |
| EXPERIMENT_DATA_ROOT | Dedicated benchmark tiny-data output; never inside CORPUS_ROOT |
| RUNS_ROOT | Checkpoints, predictions, logs and metrics |
| CACHE_ROOT | Pinned model caches, k-means and vocoder artifacts |

Do not reuse a full-data prepared directory for smoke. Input rows retain original
train/dev/test membership. `--limit 5` selects up to five from **each** split.
Use a tiny accepted manifest/subset whose dev/test characters are covered by
train. The tokenizer deliberately errors on unknown labels instead of borrowing
vocabulary from test. All three splits must be nonempty. Content mismatches
between `*_text` and `*_tts_text` require reviewing the normalization; ASCII
whitespace-only differences are accepted and recorded.

The `.env` file is excluded from Git. Docker context can be passed using
`--docker-context NAME` to the smoke driver. Compose bind mounts still refer to
the daemon host: the repository/configs must exist at that location. Prefer
running the driver on the GPU host itself, including for Runpod.

## Images and checkpoints

Build the existing pinned images before execution:

```sh
docker compose --env-file .env build common fairseq
docker compose --env-file .env build cascade evaluation
```

The CUDA image is 12.6.3; provision a compatible NVIDIA host driver and runtime.
The local audited Windows GPU is 4 GB with a driver reporting CUDA 12.5. It was
not used as evidence that the pinned containers or training fit/run successfully.
Host Python is used for the driver and CPU tests, not production model execution.

The S2UT extractor uses the existing locked HuBERT revision and KM100 hash in
`configs/s2ut/prepare.yaml`. The driver fetches k-means through the checksum-locked
artifact helper. HuBERT/ASR model libraries use their configured immutable model
revisions. Prepare cache/network access on the execution host.

For unit vocoding, supply the matching HuBERT Base layer-6/KM100, LJSpeech
Code HiFi-GAN checkpoint and JSON config under:

```text
CACHE_ROOT/models/vocoder/unit/g_00500000
CACHE_ROOT/models/vocoder/unit/config.json
```

Source links and attribution are centralized in `configs/vocoder/unit.yaml`.
No automatic vocoder download or unverified alternate checkpoint is selected.
The code is fairseq MIT; verify upstream checkpoint/data terms independently.
`vocoder-lock.json` records the exact SHA-256 of supplied weights/config and input.
The config must declare 100 embeddings, sampling_rate=16000 and
dur_predictor_params; F0/multispeaker variants are rejected by this adapter.

There is no verified compatible Mel checkpoint yet. `configs/vocoder/mel.yaml`
is an explicit file contract, not a claim that those files exist. Supply a
trained compatible generator plus its true complete `mel` specification:
sample_rate, n_fft, win_length, hop_length, n_mels, f_min, f_max, log_transform,
normalization, eps, normalize_volume. These must exactly match generated
`mel-spec.json`. Do not add matching metadata to incompatible weights.
Generic `fairseq_cli.hydra_train` placeholders have been replaced by
`direct_s2st.vocoders.train`: fixed fairseq generators, upstream HiFi-GAN MPD/MSD
and GAN/feature/mel losses, train-only fitting, and original-unit run-length
duration supervision. See [complete workflow](PIPELINE_COMPLETION.md).

## S2UT smoke

Install the driver dependencies (the repository already pins PyYAML). Plan first:

```sh
python scripts/smoke/run.py --env-file .env --run-id s2ut-smoke --limit 5 --max-updates 2
```

This prints `PLAN_ONLY`, all stages `NOT_RUN`, without loading models or writing
derived data. On the prepared execution host:

```sh
python scripts/smoke/run.py --env-file .env --run-id s2ut-smoke --limit 5 --max-updates 2 --execute
```

The driver writes resolved local configs to `configs/local/<run-id>/` (ignored
by Git), runs common import/validation → unit extraction → linguistic multitask
preparation/validation → training → actual checkpoint validation → unit
inference → duration-predicting vocoder → common evaluation. It records stage
exit status in `execution.json` and stops at the first failure. A completed set
of commands is `STAGES_COMPLETED_REVIEW_METRICS`, **not automatically E2E PASS**:
review waveform success count, ASR success, finite individual training losses,
speaker similarity and actual metrics before accepting the run.

The driver is a fresh-run orchestrator, not automatic recovery of partially
completed stages. For recovery, inspect execution.json and invoke the existing
CLI command at that stage with the appropriate `--resume`/`--overwrite` choice.
Never mix previous full-data outputs with a new smoke. The max-update bound is
1..10 and sample limit 1..100 per split; neither selects the full profile.

Training preserves its resolved config/environment in the run root. Inference
records execution metadata separately in `inference-state/`. Generation timing
includes model load and is amortized across samples; per-sample model-only
latency is not currently instrumented.

The main baseline uses `--multitask-config-yaml config_multitask.yaml`.
All auxiliary TSVs have `id\ttgt_text`, with equal IDs for every split, and
dictionary counts use train only. Before model loading, malformed artifacts
are rejected; runtime training validation also resolves the actual fairseq
architecture, layer bounds and CTC length feasibility. Do not shrink encoder
depth below the configured auxiliary attachment layers for a tiny run.

Unit inference produces `RUNS_ROOT/<run-id>/predictions/units.jsonl`, not a WAV
prediction prematurely labeled successful. Unit vocoding creates `audio/*.wav`
and the existing `predictions.jsonl`, which evaluation consumes.

## Cascade and evaluation

After S2UT succeeds, use the same imported common test split:

```sh
docker compose --env-file .env run --rm cascade cascade run --profile smoke --split test --limit 5 --resume
```

The default Cascade run is `cascade-smoke`. Create an evaluation YAML from
`configs/evaluation/default.yaml` with `run_id: cascade-smoke`; keep the same
ASR revision, normalization, reference text and speaker encoder for both systems.
Place this local config below `configs/local/` so Compose can read it.

```sh
docker compose --env-file .env run --rm evaluation evaluate run --config /workspace/configs/local/cascade-evaluation.yaml --resume
```

ASR-BLEU, RTF/runtime, target-reference and source speaker similarity remain
available. BLASER stays optional. A failed ASR or speaker encoder must be visible
in metrics/per-sample records; command completion alone is not successful
evaluation. Compare identical pair IDs; do not compare runs with different limits.

## Translatotron 2

The native experimental core now supports training, validation and reference-free
mel inference. Defaults are CPU, a reduced smoke model and two updates.
The user explicitly authorized core implementation before real S2UT E2E on
2026-09-13; real-data acceptance criteria remain unchanged.
See [core implementation and commands](TRANSLATOTRON2.md). Voice preservation is
disabled. No upstream duration-free substitute is used. The existing Docker
`run.py` driver remains S2UT-only; `suite.py` orchestrates all four systems,
vocoder fitting, evaluation and fail-closed real-artifact acceptance.

## Tests

```sh
python -m pip install -e '.[dev]' sacrebleu==2.5.1 soundfile==0.13.1
python -m pytest -q -m 'not gpu'
```

Optional `tests/integration/test_vocoder_checkpoints.py` uses supplied real
weights, never random/fake checkpoint files. Set `S2ST_UNIT_VOCODER_CHECKPOINT`
and `S2ST_UNIT_VOCODER_CONFIG`, or the four `S2ST_MEL_*` paths documented in that
test. Invoke `pytest -m gpu` explicitly in the provisioned model environment.
Pass these variables via Compose `-e` if using a container; set
`S2ST_TEST_DEVICE=cpu` only if intentionally testing pretrained vocoders on CPU.
Missing artifacts skip those tests and do not count as PASS.

## S2T→TTS / four-system extension

The required final systems are `s2ut`, `translatotron2`, `cascade`, and `s2t_tts`.
The new baseline uses pinned Whisper large-v3 translation and the existing QwenTTS,
without changing the original turbo ASR → NLLB → TTS model settings.
See [S2T→TTS implementation and verification](S2T_TTS.md) for commands, Colab,
new files, dependency limitations, and unverified GPU/E2E items.
Historical test counts below/above describe their original runs, not current validation.
