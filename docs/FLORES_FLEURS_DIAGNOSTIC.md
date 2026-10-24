# Cell 8C reproduction audit

8C adds an external diagnostic, not another formal benchmark. Official data,
model revisions, inference adapters, English normalization and scoring stay
fixed. No training/optimizer/scheduler/checkpoint settings change.

## Reviewed sources and deviations

The corpus repository was inspected read-only at
[`efe987934c48ee78b02132e97144815589ebeea7`](https://github.com/Yaaamashiro/ja-en-direct-s2st-corpus/tree/efe987934c48ee78b02132e97144815589ebeea7).
Relevant files are `configs/colab-pro.yaml`, `src/s2st_corpus/{config,runner,seeding,audio}.py`,
and `notebooks/colab_pro.ipynb`. The production preset is a different file;
choosing it merely because it is named production would be unjustified.

| Setting | Corpus checked-in value | Historical effective value | 8C action |
|---|---|---|---|
| Model | Qwen3-TTS-12Hz-1.7B-CustomVoice @ `f00cf133b78d3c2c35857faba3b1be9b98c4f971` | Model SHA present in logs; unchanged notebook confirmed by user | Read fixed YAML and check SHA |
| Language / speaker | Japanese / Ono_Anna | User-confirmed unchanged notebook | Original pinned implementation |
| Instruct | empty string in colab YAML | User-confirmed unchanged notebook | Read unchanged fixed YAML |
| Sampling | do_sample/subtalker_do_sample True, remove_invalid_values False, renormalize_logits True | User-confirmed unchanged notebook | Read unchanged fixed YAML |
| Max tokens | 2048 in colab YAML | User-confirmed unchanged notebook | Read unchanged fixed YAML |
| Dtype / attention | auto / sdpa | Logs show BF16; notebook unchanged per user | Resolve auto and record actual precision |
| Batch size | notebook Cell 5: 72 on >=70GiB, otherwise 1 | Logs have multiple batch values; user confirms unchanged notebook for this reproduction | Apply fixed notebook hardware rule, record actual batch and seed |
| Output | mono PCM16, 16kHz, soxr HQ | Manifest audio SHA/rate | Reuse corpus helper and validate WAV |
| Content retry/QC | Up to one retry; acceptance based on source CER | Applied to original accepted corpus | Diagnostic uses attempt 0 without CER-based filtering; all failures stay in denominator |

The attempt-zero/QC difference is intentional: applying a CER acceptance filter
to the external sentences would hide source synthesis/ASR failures. It changes
selection, not the Qwen generation configuration. It is recorded in identity
and audit. This is not a claim to reconstruct byte-identical historical audio.
The original sample IDs, batch mates and RNG draw positions are different.

Target English Qwen, NLLB, JA Whisper and evaluation English Whisper are read
from the validated saved Cascade context. No beam, precision, language,
sampling, repetition, length, or language-token tuning is introduced. Existing
target Qwen EOS/time guards remain in effect. Source Qwen uses the pinned
corpus helper, including its nested code predictor configuration and seed
routine. CUDA failures abort; they are never converted into hundreds of empty
successful samples. Runtime/model-load/access errors abort and preserve stages.

## Source configuration basis

The user explicitly confirmed that the corpus was generated without changing
the notebook's settings. This confirmation is accepted as the configuration
basis; a generation-time receipt is no longer mandatory for this experiment.
Normal Cell 8C v2 defaults `FLORES_FLEURS_SOURCE_CONFIG_CONFIRMED=True` and sends
`--confirm-corpus-notebook-defaults`. Preflight is stdlib-only and accepts this
explicit mode without allocating GPU models or fabricating a historical file.

After dependency setup, the source checkout is fetched at the fixed reviewed
commit, or verified read-only if explicitly provided. SHA256 checks cover:

- `configs/colab-pro.yaml`: `450fc2b17a27cc8ce3d38c17c943aa1593664454f6b6219a4a9bb8ccb6af0007`.
- `notebooks/colab_pro.ipynb`: `3e6107581c8911f9a893889a83b03f64ab7212bd53c56c3659b440f3b65bec22`.
- `requirements/colab.txt`: `aff1db9ed92c233ee43c174f6a1e309ae2d1590ac45e748a0180d22323bae70a`.

Only checkout CRLF/LF conversion is normalized for these fixed-source checks;
the actual byte checksums are recorded separately and rechecked by workers.
The batch constants and GPU threshold are read from Cell 5's AST; the notebook
is never executed. All other TTS values come from the fixed YAML. `auto` dtype
uses the original BF16-support rule. Ambient batch environment overrides are
ignored. Requested and effective values, GPU VRAM, code/config/model SHA,
batch rule, seed/group membership, and current reproduction package versions
are recorded. Packages are checked against the fixed corpus requirements and
notebook Torch version. No TTS inference implementation is duplicated.

Audit records explicitly use `basis=user_confirmation` and
`historical_execution_revision_verified=False`. They do not claim that a
missing original execution receipt was recovered, or that the historical
runtime/Git commit is independently verified. The existing Python 3.10
diagnostic runtime remains a recorded execution environment; the corpus
notebook's Python 3.13 environment is not retroactively invented.

The optional documentary route remains supported. An explicitly supplied
`FLORES_FLEURS_SOURCE_PROVENANCE` always wins and is checked strictly; a bad
receipt never silently falls back to user confirmation. With neither evidence
nor confirmation, preflight still reports
`Cannot prove exact source-corpus Japanese TTS configuration` before GPU setup.

The supported saved receipt contract is `schema: source-corpus-tts-v1`:

- `repository`: `Yaaamashiro/ja-en-direct-s2st-corpus`.
- `code_revision`: original full 40-character source commit SHA.
- `config_relative_path`, `config_sha256`: original tracked YAML and byte SHA.
- `language`: Japanese; `attention`: recorded SDPA/eager implementation.
- `effective_tts`: **all** `TTSConfig` fields, including resolved dtype,
  device, model_id/revision, speaker, instruct, sampling, predictor options,
  max_new_tokens, output rate, batch size and content retry setting.
- `packages`: original qwen-tts, transformers, torch, numpy, soxr versions
  (additional dependencies may be included and are checked too).
- `evidence`: original saved files with `path` and byte `sha256` binding the
  receipt to generation-time evidence. Merely appending the current YAML as
  evidence cannot establish historical provenance. This gate validates a
  supplied receipt's integrity; it cannot authenticate a fabricated receipt.

The user-confirmed route constructs a reproducibility record from the checked
fixed files and runtime, labelled as confirmation rather than an original
receipt. It uses a distinct identity, so prior documentary and confirmed runs
are never silently mixed. Saved receipt metadata is never fabricated.

Code verification checks commit, tracked source/config cleanliness, config SHA,
and source-file SHA. Read-only source code can be supplied explicitly; otherwise
a separate pinned local checkout is fetched. No external corpus files are edited.
Source package versions must match the selected record; unsupported environments
fail instead of quietly selecting a different generation implementation. Cell
8C fetches v2 when an older training/code pin still contains v1, without
resetting the training checkout or changing saved training revisions.

## Dataset/access contract

Official repos only:

- [FLEURS](https://huggingface.co/datasets/google/fleurs/tree/70bb2e84b976b7e960aa89f1c648e09c59f894dd), Japanese validation/test,
  main's official Parquet files. Verified footer/schema/text-column range-read.
- [FLORES+](https://huggingface.co/datasets/openlanguagedata/flores_plus/tree/e707e62e762a9e2faca3933aa315f4665cc3cd8f), Japanese/English dev/devtest,
  gated JSONL. Accept official terms and log in; no mirrors or automatic acceptance.

Text, IDs and metadata use immutable dataset commits. Matching is exact after
NFKC and Unicode whitespace collapse. No case/punctuation stripping, edit
distance, fuzzy/semantic/MT matching, or row alignment. Japanese candidates
must be unique and the corresponding English `(split,id)` must also be unique.
Multiple FLEURS recordings are reduced with a deterministic SHA of their split,
ID, path and raw text. Selection is SHA(canonical sample ID), never input order.
Canonical shared Japanese is the safely normalized common text, not ASR text.

Dataset audit records selected IDs, both text hashes, unmatched/ambiguous counts,
source metadata SHA and local download checksums. Only selected decoded WAVs are
persisted. Range access can transfer an entire audio row group even when only
a subset is selected; it is not a promise of one network download per sentence.

## Test scope

CPU tests exercise matching, deterministic selection, reference alignment,
ambiguous/unmatched exclusion, strict split/revision/provenance checks, WAV
validation and SHA invalidation, failure-inclusive corpus BLEU/CER, sequential
stage recovery, immutable official files, notebook startup/order and runtime
dependency/import contracts. Unit tests use generated tiny WAVs and text fixtures,
not remote weights or large datasets. Real GPU synthesis and a complete gated
FLORES/FLEURS run remain separate operational validation.

## Changed files

- `notebooks/colab_training.ipynb`: one new launcher immediately after 8B; existing cells preserved.
- `src/direct_s2st/cascade/flores_fleurs_data.py`: fixed official datasets, exact joining, selection and natural audio.
- `src/direct_s2st/cascade/flores_fleurs_diagnostic.py`: three conditions, guarded stage journals and reports.
- `src/direct_s2st/cascade/source_corpus_tts.py`: historical evidence gate and original corpus runner bridge.
- `scripts/colab/flores_fleurs_diagnostic.py`: preflight and sequential worker dispatch.
- `scripts/colab/flores_fleurs_runtime.py`: dedicated environment, CUDA/BF16/import checks, local setup.
- `requirements/diagnostic-flores.in` and `requirements/diagnostic-flores.txt`: pinned dependency roots and resolved closure.
- `tests/unit/test_flores_fleurs_diagnostic.py`, `tests/unit/test_flores_fleurs_runtime.py`, and `tests/unit/test_corpus_source_confirmation.py`: tests.
- `tests/unit/test_cascade_corpus_diagnostic.py` and `tests/unit/test_cascade_mt_diagnostic.py`: allow 8C in notebook adjacency checks.
- `docs/COLAB.md` and this document: usage, limitations and source/setting audit.

## Initial release validation (2026-10-24)

- Full CPU suite: `pytest -m 'not gpu'`: **928 passed, 2 skipped, 8 GPU tests deselected**.
- New 8C tests on the isolated Python **3.10.18** with the installed diagnostic dependency closure: **55 passed**.
- The closure was actually installed; `pip check` reported no broken requirements.
  Qwen/Whisper/NLLB class imports, Parquet imports, WAV round trip and soxr resampling were exercised on CPU.
- Official PyPI distribution metadata was checked for all 101 Linux/Python 3.10 dependencies.
  SoX's Python wrapper is sdist-only; its pure-Python wheel build was also exercised.
- The fixed FLEURS Parquet footer/schema and one text-row range read were checked against the official repository.
- No full model weights or full external evaluation set were downloaded for these unit tests.
  Real CUDA/BF16 synthesis, a complete authenticated FLORES+ join and the eventual matched count
  remain operational checks. Source settings are now accepted on the user's explicit confirmation;
  this changes their recorded basis, not the scope of the completed GPU tests.

## Confirmed-notebook fix validation (2026-10-24)

- Full CPU suite: **944 passed, 2 skipped, 8 GPU tests deselected**.
- Related 8C tests on Python **3.10.18** with the installed diagnostic dependency closure: **71 passed**.
- `pip check`: no broken requirements.
- Added coverage for explicit user confirmation, independent receipt validation, hardware-dependent batch selection,
  automatic dtype resolution, fixed-source hashes, dependency mismatch rejection and normal-cell v1-to-v2 code recovery.
- The real pinned corpus checkout was checked on CPU; simulated GPU metadata resolved batch sizes 1 and 72
  from its original notebook. This was a configuration check, not a GPU inference test.
- Real CUDA synthesis and the complete authenticated external-dataset evaluation have not been run locally.
