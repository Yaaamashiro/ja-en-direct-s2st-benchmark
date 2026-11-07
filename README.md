# Japanese → English Direct S2ST Benchmark

An experimental framework for comparing Japanese-to-English speech translation systems
using the same corpus, test split, and evaluation conditions. It covers corpus import,
preprocessing, training, inference, speech synthesis, and evaluation.

## Systems compared

| System | Pipeline |
| --- | --- |
| S2UT | Japanese speech → English discrete units → unit vocoder |
| Translatotron 2 | Japanese speech → phonemes and internal states → mel spectrogram → mel vocoder |
| ASR→MT→TTS cascade | Whisper large-v3-turbo → NLLB-200 distilled 600M → Qwen3-TTS |
| S2T→TTS cascade | Whisper large-v3 in translation mode → Qwen3-TTS |

Translatotron 2 is an independent implementation in native PyTorch, rather than the
original authors' implementation or pretrained weights. It targets canonical-speaker
output and does not preserve the source speaker's identity. End-to-end success and
translation quality across all four systems on real audio, as well as reproduction
of the original papers' numerical results, have not been verified.
See the [model documentation](docs/TRANSLATOTRON2.md) for its architecture and scope.

## Requirements

- An `accepted.jsonl` manifest and 16 kHz audio produced by the external
  `ja-en-direct-s2st-corpus` repository.
- Docker Compose, plus Linux, an NVIDIA GPU, a compatible driver, and the NVIDIA
  Container Toolkit for GPU workloads.
- Python 3.10 or later for execution planning and development tests.
- Network access to download the models and artifacts pinned in the configuration,
  or a populated cache.

The corpus and trained checkpoints are not included. Corpus generation is outside
this repository's scope. The input corpus is read-only; derived artifacts such as
features are written to a separate location. Regular data processing runs inside
Docker. Colab uses a dedicated Python environment with pinned dependencies.

## Google Colab

Upload the notebook for your use case to Colab. A CPU runtime is sufficient through
Cell 4A (data preparation). Cell 4B (HuBERT unit extraction), training, and system
comparison use a GPU. After switching runtimes, rerun Cell 1 with the same settings
and continue to the desired cell; the local environment is rebuilt automatically.

- [Smoke notebook](notebooks/colab_smoke.ipynb): five examples per split and two
  updates. Use this first to check the pipeline.
- [Training notebook](notebooks/colab_training.ipynb): training on the full dataset
  and comparison of all four systems. Long training requires explicit confirmation.

Edit the initial configuration form, then run the numbered cells in order. Smoke
tests and regular experiments use separate output locations. Environment setup,
preprocessing, and TT2, S2UT, and vocoder training can run without Docker.

Colab training uses Python 3.10.18. When migrating from an older version, start with
a new runtime and experiment output directory. Training runs in segments, with
verified checkpoints saved to Google Drive. After disconnection, training resumes
from the latest saved checkpoint. Select a training target from `tt2`, `s2ut`,
`unit`, and `mel`, then enable the four-system comparison after training each target.
See the [Colab guide](docs/COLAB.md) for time budgets, save frequency, recovery,
and operational notes.

The notebook's `PERFORMANCE='gpu80'` preset provides starting settings for an 80 GB
GPU. Batch sizes, parallel CPU data loading, prefetching, and save intervals are
configurable; execution speed and peak memory use on actual hardware have not been
verified.

## Setup

Run the following on the execution host:

```sh
git clone --recurse-submodules https://github.com/Yaaamashiro/ja-en-direct-s2st-benchmark.git
cd ja-en-direct-s2st-benchmark
python -m pip install -e .
```

For an existing checkout, run `git submodule update --init --recursive` to prepare
the pinned fairseq revision. Run subsequent commands from the repository root.

### Configure storage paths

Copy [.env.example](.env.example) to `.env` and configure the paths below.
The `.env` file is excluded from Git.

| Variable | Location |
| --- | --- |
| `CORPUS_ROOT` | Existing corpus containing `production/manifests/releases/accepted.jsonl` and audio |
| `EXPERIMENT_DATA_ROOT` | Dedicated directory for derived benchmark data, outside the corpus |
| `RUNS_ROOT` | Training logs, checkpoints, and inference results |
| `CACHE_ROOT` | Model and artifact cache |

Paths refer to the host running the Docker daemon. Keep smoke-test data separate
from full-training data. The execution driver accepts `--docker-context NAME` for
a remote Docker connection, but does not transfer files. See the
[environment and reproduction guide](docs/REPRODUCTION.md) for details.

### Build Docker images

```sh
docker compose --env-file .env build common fairseq
docker compose --env-file .env build cascade evaluation
```

Build in this order because `cascade` and `evaluation` use the fairseq image as
their base.

## Run a smoke test

First, inspect the execution plan for all four systems. This command alone does
not start model processing:

```sh
python scripts/smoke/suite.py --env-file .env --name benchmark-smoke --limit 5 --max-updates 2
```

After reviewing the plan, add `--execute` to run it. To resume an interrupted run,
use the same settings and add `--resume`:

```sh
python scripts/smoke/suite.py --env-file .env --name benchmark-smoke --limit 5 --max-updates 2 --execute
python scripts/smoke/suite.py --env-file .env --name benchmark-smoke --limit 5 --max-updates 2 --execute --resume
```

`--limit 5` selects up to five examples from each of the train, dev, and test splits.
All three splits are required. Character and phoneme vocabularies are built from
the training split only. See the [smoke-test instructions](docs/REPRODUCTION.md)
for input requirements.

The suite runs preprocessing, training and inference for each system, unit and
mel vocoder training, evaluation, and comparison in sequence. Two updates are
intended to exercise the pipeline; they do not guarantee intelligible speech or
end-to-end success. See the [pipeline guide](docs/PIPELINE_COMPLETION.md) for
pretrained vocoder options, individual stages, and failure handling.

## Configuration and outputs

- `configs/`: system, vocoder, and evaluation settings, with pinned model IDs and
  immutable revisions.
- `scripts/smoke/`: drivers for planning, executing, and resuming small runs.
- `src/direct_s2st/`: CLI, preprocessing, training, inference, and evaluation code.
- `tests/`: unit and integration tests.
- `docs/`: environment setup, model specifications, reproduction conditions, and
  verification records.

The suite writes generated configurations to `configs/local/<name>` and
system-specific artifacts to `RUNS_ROOT/<name>-<system>`. Completion status is
recorded in `EXPERIMENT_DATA_ROOT/results/e2e-acceptance.json`; comparison results
are written under the same `results` directory. Evaluation uses a shared, pinned
ASR model and normalization procedure, and retains records of failed samples.

The default profile is `smoke`. Full training requires explicit configuration and
confirmation. Existing outputs are not replaced without an explicit `--overwrite`;
resuming checks configurations, inputs, and artifacts for consistency. Do not add
audio, features, checkpoints, caches, or inference results to Git.

## Development and testing

```sh
python -m pip install -e ".[dev]"
python -m pytest -q -m "not gpu"
```

CPU model tests additionally require torch and torchaudio 2.7.1. Run GPU tests
explicitly after preparing compatible hardware and the required artifacts. See
the [reproduction guide](docs/REPRODUCTION.md) for instructions, including checks
that HuBERT unit sequences match the fairseq pipeline.

## Documentation

- [Design and data invariants](DESIGN.md)
- [Execution environments and artifact preparation](docs/REPRODUCTION.md)
- [Running, resuming, and checking completion for all four systems](docs/PIPELINE_COMPLETION.md)
- [Translatotron 2 architecture](docs/TRANSLATOTRON2.md)
- [Reference presets, distributed training, and correspondence with the papers](docs/REFERENCE_PARITY.md)
- [Colab paper_exact / paper_practical recipes and Griffin–Lim evaluation](docs/PAPER_RECIPES.md)
- [Implementation status and verification records](docs/IMPLEMENTATION_STATUS.md)

See the [S2T→TTS guide](docs/S2T_TTS.md) for execution, recovery, and pinned
revisions. Colab also supports all four systems under the same model conditions.
