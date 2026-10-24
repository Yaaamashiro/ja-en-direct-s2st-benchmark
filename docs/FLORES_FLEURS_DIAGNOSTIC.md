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
| Model | Qwen3-TTS-12Hz-1.7B-CustomVoice @ `f00cf133b78d3c2c35857faba3b1be9b98c4f971` | Model SHA present in logs | Require receipt, never infer other settings from model SHA |
| Language / speaker | Japanese / Ono_Anna | Language hard-coded, speaker in manifest | Verify receipt against actual pinned implementation |
| Instruct | empty string in colab YAML | Not persisted in inspected manifest/log | Stop if not proven |
| Sampling | do_sample/subtalker_do_sample True, remove_invalid_values False, renormalize_logits True | Full effective kwargs not persisted | Stop if not proven |
| Max tokens | 2048 in colab YAML | Not persisted in inspected manifest/log | Stop if not proven |
| Dtype / attention | auto / sdpa | Logs show resolved BF16; not a complete receipt | Require resolved dtype and attention |
| Batch size | config default 1, notebook/env override | Logs include 8 and 130 on different runs | No universal batch can be assumed; preserve proven setting/RNG grouping |
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

## Historical source evidence gate

The inspected repository does not emit a full effective-generation receipt.
It is unsafe to claim its current YAML proves which historical overrides ran.
8C reports `Cannot prove exact source-corpus Japanese TTS configuration` before
GPU setup in that situation. Finding a current repository is not sufficient.

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

No receipt is synthesized from the checked-in defaults by 8C. No unsafe example
with guessed historical values is provided. Incomplete evidence must be found,
or the experiment must explicitly be redesigned as a new source-TTS condition.

Code verification checks commit, tracked source/config cleanliness, config SHA,
and source-file SHA. Read-only source code can be supplied explicitly; otherwise
a separate pinned local checkout is fetched. No external corpus files are edited.
Source package versions must match the proven receipt; unsupported environments
fail instead of quietly selecting a different generation implementation.

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
- `tests/unit/test_flores_fleurs_diagnostic.py` and `tests/unit/test_flores_fleurs_runtime.py`: new tests.
- `tests/unit/test_cascade_corpus_diagnostic.py` and `tests/unit/test_cascade_mt_diagnostic.py`: allow 8C in notebook adjacency checks.
- `docs/COLAB.md` and this document: usage, limitations and source/setting audit.

## Validation record (2026-10-24)

- Full CPU suite: `pytest -m 'not gpu'`: **928 passed, 2 skipped, 8 GPU tests deselected**.
- New 8C tests on the isolated Python **3.10.18** with the installed diagnostic dependency closure: **55 passed**.
- The closure was actually installed; `pip check` reported no broken requirements.
  Qwen/Whisper/NLLB class imports, Parquet imports, WAV round trip and soxr resampling were exercised on CPU.
- Official PyPI distribution metadata was checked for all 101 Linux/Python 3.10 dependencies.
  SoX's Python wrapper is sdist-only; its pure-Python wheel build was also exercised.
- The fixed FLEURS Parquet footer/schema and one text-row range read were checked against the official repository.
- No full model weights or full external evaluation set were downloaded for these unit tests.
  Real CUDA/BF16 synthesis, a complete authenticated FLORES+ join, the eventual matched count,
  and historical Japanese TTS provenance remain operational checks. The provenance gate currently
  prevents claiming a completed real three-condition experiment from the inspected corpus artifacts.
