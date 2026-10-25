# Cell 8F: saved Synthetic source-condition evaluation

Use **Cell 1 → 8F**, with the same `OUTPUT_BASE / MODE / EXPERIMENT` as completed
8E and 8C. Do not rerun training, 8C Japanese synthesis or Natural 8E inference.
This is an external diagnostic, separate from the original formal benchmark.

## Exact comparison and model

8F selects an already-completed Natural 8E and freezes its source 8C identity,
canonical IDs, references, training snapshot identity, feature/dictionary
metadata, checkpoint SHA and inference/evaluation settings. Only the Japanese
source waveform and waveform-dependent duration/frame count differ.

The requested experiment is checked explicitly:

- Run: `s2ut-training-experiment-001-paper-fisher-v1`.
- Dev-best update: 91,000; training reached: 400,000.
- Checkpoint SHA256:
  `d2ee19bec9ce5f12c19f6b7ac64cf68516ea7010b373aa878a6d9e1f34fc0c7b`.
- Beam 10, the same reference-free fairseq adapter, byte-identical feature and
  vocabulary metadata, original HuBERT L6/KM100 configuration.
- Official Unit HiFi-GAN `g_00500000`, the same config/checkpoint SHA and predicted
  duration policy. No self-trained vocoder substitution.
- Same English Whisper model/revision, actual dtype/decoding audit, normalization,
  numeric/audio dependency versions and failure-inclusive corpus BLEU implementation.

No official model/recipe setting is changed. The inference functions and ASR
implementation are checked against the reviewed `32cc967` source fingerprint;
shared model/vocoder/scoring/dependency files are checked against the saved
Natural release. New dispatch affects plan validation/reporting only.

## Verified saved evidence

The inspected real metadata (read-only through the Google Drive skill) identified:

- 8C: `a3824b2b4256907424503e22c55518a90d7d16814224ea31572c7a4c557672a5`.
- Natural 8E:
  `4b6246b07e4f49c4cfd64c39a7b3fb3f6533f7e83e71644b154c40566c5d27ac`.
- Saved Natural S2UT BLEU: `0.51861564816`.
- Saved Cascade Natural BLEU: `21.153432814483`.
- Saved Cascade Synthetic BLEU: `20.975587799372`.

These scores are observations from inspected artifacts, not constants used in
scoring code. The real Natural documents, 300 per-sample records, completion
checksums, actual ASR audit and dependency/inference compatibility are verified
without opening a GPU model. Synthetic source bytes are verified by 8F on the
user's runtime; this local implementation does not claim to have evaluated new
Synthetic S2UT predictions or downloaded all 300 source WAVs.

Natural evidence is read from `plan.json`, its completion receipt, checksum-bound
reports/indices and evaluation runtime/audit. Unsupported release fingerprints,
checkpoint differences, changed references or missing evidence stop execution.
Several completed Natural runs for different 8C identities require an explicit
`FLEURS_SYNTHETIC_NATURAL_ID`; 8F never draws a replacement sample set.

8C's `identity.json`, `dataset-audit.json`, embedded report rows and
`per_sample.jsonl` must agree with Natural 8E. Synthetic group receipts are
reconstructed from the **same recorded 8C group implementation**, source proof,
batch size and canonical samples. Receipt/request/item sets must match exactly.
Every source WAV must match the per-sample Synthetic path/SHA and generation
receipt, remain below `synthetic-source/`, and be finite/non-silent mono PCM16 at
16 kHz. Missing/failed source synthesis is fatal; there is no TTS or data fallback.
WAV headers/content hashes are checked, then the common input preparation copies
bytes and verifies them before source-only generation.

## Reuse, isolation and recovery

8F uses the existing 8E functions for external preparation, source-only Unit
generation, sharded ZIP persistence, official vocoding, waveform restoration and
ASR journals. No separate S2UT generation algorithm is copied. English references
are attached after generation for scoring; no gold text, MT output or reference
Units enter fairseq inputs. Dummy targets/auxiliary labels are removed by the
existing adapter before inference.

New plan purpose and namespace:

`diagnostics/flores-fleurs-s2ut-synthetic/<identity>/`

Natural 8E/8C and original benchmark files are read-only. All existing notebook
cells, including the pinned Natural 8E execution path, stay unchanged. Synthetic
plan and archive identities are distinct, so Natural/Synthetic results cannot
share Unit/WAV/ASR caches accidentally. Repeating Cell 1 → 8F with identical
settings/code restores successful stages. The normal cell separates current display
code from the scientific execution checkout. An existing version-1 Synthetic plan
restores its recorded execution SHA and Natural identity to the same canonical
local code path, including after cache/VM loss or a display-only main update.
Invalid plan identities stop before execution; several Natural anchors require
an explicit NATURAL_ID. No code revision needs to be entered manually.
Complete results return after checksum verification without restoring model
checkpoints, rescanning the 300 source WAVs/group receipts, rebuilding inference
runtimes or rerunning stages. The completed report receipt, input audit and small
8C/Natural comparison evidence are still checked. Common
functions retain some `8E` progress labels; their inputs are the verified 8F
Synthetic plan, and long-running stages still log counts/remaining work.

There are no new package requirements. As in 8E, fairseq uses the saved training
runtime and English ASR uses the isolated pinned 8C runtime. They must not mix
NumPy 1.x/2.x environments. The existing runtime bootstrap requires CUDA/BF16
support. Source Qwen/NLLB models are never loaded or run; the reused 8C package
lock may contain those packages without executing their models.

## Output and interpretation

- `metrics.json`: Synthetic corpus BLEU, all failure counts, model/code/source
  identities, actual ASR settings, per-ID source hashes, Natural/Synthetic S2UT
  and Cascade scores/deltas and Synthetic/Natural S2UT ratio.
- `per_sample.jsonl/csv`: canonical/FLORES/FLEURS IDs, original texts, English ASR,
  sentence BLEU, source/generated audio hashes, status/error; JSONL also keeps
  Unit sequences and durable WAV ZIP/member information.
- `report.md`: comparison table, deltas, ratio, failure counts, source/model IDs.
- `audio/`, `audio-index.json`, generation/progress ZIPs: existing resumable storage.

The ratio is `null`/undefined when Natural BLEU is zero. All 300 examples remain
in corpus BLEU; generation/vocoder/ASR failures contribute empty hypotheses.
Sentence BLEU is diagnostic only and is not averaged to obtain corpus BLEU.

Original KFTT/JESC scores are optional. Both latest formal plans, common
ID/reference sets, checkpoint/ASR/failure policy, prediction binding and score
journals must verify. Their saved hypotheses are scored on CPU solely to check
agreement with saved formal BLEU; no ASR/translation/audio is rerun or written.
If safe verification is unavailable, the row is omitted with a reason. No
hard-coded historical BLEU value substitutes for missing artifacts.

Reports state only observations and explicitly say that the source-condition gap
is a diagnostic clue, not proof of a sole cause.

## Changed files

- `src/direct_s2st/s2ut/flores_fleurs_synthetic.py`: validation, plan ownership and comparison report.
- `scripts/colab/s2ut_flores_fleurs_synthetic.py`: normal-cell/worker orchestration.
- `src/direct_s2st/s2ut/flores_fleurs_diagnostic.py`: Synthetic-only dispatch and shared row rendering.
- `src/direct_s2st/s2ut/flores_fleurs_resume.py`: reviewed report-refactor fingerprint; Natural inference remains unchanged.
- `scripts/colab/cascade_diagnostic_runtime.py`: additive explicit Synthetic result contract.
- `notebooks/colab_training.ipynb`: one new 8F cell; previous cells unchanged.
- `docs/COLAB.md`, this document.
- `tests/unit/test_s2ut_flores_fleurs_synthetic.py`: new 8F tests.
- `tests/unit/test_s2ut_flores_fleurs_diagnostic.py`: reusable Synthetic fixtures,
  shared inference fingerprint and notebook insertion compatibility.
- `tests/unit/test_cascade_diagnostic_runtime.py`: explicit Synthetic JSON result
  contract and rejection of incomplete/inconsistent results.
- Existing notebook-insertion assertions only:
  `test_cascade_corpus_diagnostic.py`, `test_cascade_mt_diagnostic.py`,
  `test_flores_fleurs_diagnostic.py`, `test_text_pair_audit.py`.

Tests use local waveforms and model doubles. The normal-cell result receiver and
JSON validation execute with real subprocess stdout/stderr rather than a mocked
successful return. Real GPU model execution is a separate verification boundary.

## Verification performed

- Full Python 3.13 CPU suite (`pytest -m "not gpu"`): **1,129 passed, 2
  skipped, 8 GPU tests deselected**. The skips are the unconfigured real eSpeak
  executable and Windows symlink privilege requirement; neither is an 8F failure.
- Python 3.10.18: all 170 tests covering 8F, Natural 8E and the shared JSON
  receiver passed. `pip check` reported no broken requirements.
- The exact normal notebook entry point was exercised with real subprocess
  JSON transport for a warm runtime, cold code cache, inspect-only mode, saved
  execution revision restoration, cache loss and invalid saved plan rejection.
- Completed-resume tests forbid checkpoint restoration, source WAV/group scans,
  dependency setup and model/stage calls, while comparing saved result bytes.
- Interrupted tests cover durable Unit shards, waveform archives, local/durable
  ASR tails, failure-inclusive scoring and preservation of all original artifacts.
- The actual saved 300-row Natural metadata was also accepted by the new
  validator with its recorded checkpoint and ASR runtime. This is a metadata
  compatibility check, not a new GPU result.
