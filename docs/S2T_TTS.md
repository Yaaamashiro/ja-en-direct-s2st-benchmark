# S2T→TTS baseline

## Models and outputs

The fourth system has ID `s2t_tts` and CLI command `s2st-benchmark s2t-tts run`.
It uses the same common test split as S2UT, Translatotron 2, and ASR→MT→TTS Cascade.

- S2T uses `openai/whisper-large-v3` to translate Japanese speech to English text
  with `task=translate`. Its revision is `06f233fe06e710322aca913c1bc4249a0d71fce1`,
  verified at implementation time against the [Hugging Face model API](https://huggingface.co/api/models/openai/whisper-large-v3).
- TTS uses `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` at revision
  `f00cf133b78d3c2c35857faba3b1be9b98c4f971`, with `Ono_Anna` and English.
- Settings come from `configs/s2t_tts/default.yaml`. A `main` revision is rejected.
- Whisper uses `do_sample=False, condition_on_prev_tokens=False`. The shared
  `QwenTTS` uses the pinned model's official sampling settings (both main and
  subtalker enabled) and a fixed seed per sentence. This creates a separate run
  from the old greedy setting. The decoding settings do not guarantee bitwise
  equality across GPUs or runtimes. See the [audit](CASCADE_TTS_AUDIT.md) for
  the rationale, EOS monitoring, and output limits.

The existing Cascade continues to use Whisper large-v3-turbo in `transcribe` mode,
followed by NLLB-200 distilled 600M and QwenTTS. The systems share the Whisper loader
and journal code. Existing Cascade journal identities and prediction formats are preserved.

`RUNS_ROOT/s2t_tts-smoke/predictions/` stores `predictions.jsonl`, `audio/*.wav`,
`predictions.lock.json`, and `run-metadata.json`. Change the output run via `run_id`
in the config. Translations are stored in `s2t_en_text`, with timings in `s2t_seconds`
and `tts_seconds`. The remaining prediction fields, evaluator, normalization,
and references are shared.

## Execution

Set the four root environment variables and generate and validate the common
manifests first. Rebuild the shared Cascade Docker image after updating the code.

```bash
docker compose run --rm cascade s2t-tts run --profile smoke --split test --dry-run
docker compose run --rm cascade s2t-tts run --profile smoke --split test --limit 5
docker compose run --rm cascade s2t-tts run --profile smoke --split test --limit 5 --resume
```

On Colab or a native runtime, pass the same arguments to `python -m direct_s2st.cli`.
`--dry-run` displays the config, manifest path, run ID, output location, and shard
plan without loading models. `--limit` applies per shard. Omit it to run the full
split. `--shard-index 0 --num-shards 2` uses the same stable sharding as the existing
systems. Shard outputs are stored under paths such as
`predictions/shard-00000-of-00002`. Merge all shards before evaluation. Like the
existing Cascade, the standalone CLI leaves the merge to the caller.

`--resume` requires identical inputs, model identity, run ID, split, limit, and
sharding. It verifies the SHA of successful audio, reuses it, and retries failed
samples. Missing or modified successful audio stops the run. After checking the
files, regenerate with an explicit `--overwrite` or use a new run ID. The overwrite
flag recreates the selected run's journal. Fatal errors such as CUDA failures or
out-of-memory errors are recorded and stop the process.

For shared evaluation, copy `configs/evaluation/default.yaml` and set
`run_id: s2t_tts-smoke`. Use the same evaluation settings across all systems.
The commands below plan, run, and resume all four systems respectively.

```bash
python scripts/smoke/suite.py --env-file .env --name benchmark-smoke
python scripts/smoke/suite.py --env-file .env --name benchmark-smoke --execute
python scripts/smoke/suite.py --env-file .env --name benchmark-smoke --execute --resume
```

The workflow has 25 stages when vocoder training is enabled, including the added
`s2t-tts` and `s2t-tts-evaluate` stages. Final acceptance requires all common test
IDs, real WAVs, matching evaluation conditions, and all four systems. Individual
results from earlier three-system runs remain readable.

## Colab

The [training notebook](../notebooks/colab_training.ipynb) and
[smoke notebook](../notebooks/colab_smoke.ipynb) provide inference, shared evaluation,
and comparison for all four systems. The workflow mounts Drive, sets up pinned
Python 3.10, checks the corpus and common manifests, trains or restores snapshots,
runs direct inference and vocoding, runs Cascade and S2T→TTS, then evaluates,
verifies, and aggregates. See the [Colab guide](COLAB.md) for configuration, storage,
and resume instructions.

`setup.py --inference` installs `colab-inference.txt` under the constraints of
`colab310.txt`, retains the pinned torch version, and installs ffmpeg. Each heavy
stage runs in its own subprocess to release GPU memory on exit. S2T needs no
training checkpoint.

**BLASER compatibility.** SONAR and fairseq2 require separate compatibility checks
in the restored Python 3.10 environment. Colab defaults to BLASER disabled and
uses ASR-BLEU, ECAPA, duration and silence metrics, and RTF. The
`evaluation-core.txt` file isolates these shared evaluation dependencies. Enabling
BLASER does not silently skip it. To use BLASER, evaluate all four systems with a
compatible Docker evaluation runtime. Docker's existing `evaluation.txt` still
includes SONAR. An actual Docker build and BLASER run remain unverified.

## Implementation and verification (2026-09-26)

### Main files added

- `configs/s2t_tts/default.yaml`
- `src/direct_s2st/s2t_tts/{__init__,s2t,pipeline}.py`
- `src/direct_s2st/cascade/whisper.py`
- `src/direct_s2st/colab_compare.py`
- `requirements/{colab-inference,evaluation-core}.txt`
- `tests/unit/test_s2t_tts_pipeline.py`

The change updated the CLI, SYSTEMS, acceptance, smoke suite, Colab notebooks,
setup and restore, Docker dependencies, existing tests, README, DESIGN, and
related documentation. It retained the Qwen implementation and existing Cascade
model settings.

Run the CPU tests from the checkout.

```powershell
.venv\Scripts\python.exe -m pytest -m "not gpu" -q -p no:cacheprovider
```

After restoring Python 3.10 support, a local CPU test run on Python 3.12.14 reported
126 passed and 3 GPU tests deselected. Tests covered four-system acceptance with
mocked checkpoint verification, rejection of only three systems, S2T/TTS call
order, rejection of SHA and identity mismatches, resume, overwrite, sharding,
shared evaluator acceptance, dry-run without model loading, notebook Python
syntax, and restore-only behavior. The 25-stage smoke plan reached `PLAN_ONLY`,
and `git diff --check` passed.

Dependency resolution for restored Python 3.10 was verified with
`uv pip compile requirements/colab310.txt requirements/colab-inference.txt
--python-version 3.10
--python-platform x86_64-unknown-linux-gnu`. This check covered dependency
resolution only.

Unverified items include English WAV generation on the real corpus, A100 timing
and VRAM use, Colab installation, imports, GPU behavior and recovery after VM
disconnection, Docker build and execution, four-system evaluation with real
models, BLASER, intelligibility, translation quality, and reproduction of the
papers' numerical results.
