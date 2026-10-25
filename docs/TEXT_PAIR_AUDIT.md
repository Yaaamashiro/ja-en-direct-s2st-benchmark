# Cell 8D: blinded human text-pair audit

Cell 8D is a blinded human audit of JA–EN text-pair quality.
It performs no model inference and does not alter official benchmark results.
Do not inspect corpus identity before completing annotation, because doing so
can bias the manual quality judgment.

## Start and resume

Use a **CPU** Colab runtime. Open the updated notebook and run **Cell 1 → Cell 8D**.
For the existing experiment use:

- `OUTPUT_BASE=/content/drive/MyDrive/ja-en-direct-s2st-benchmark-data`
- `EXPERIMENT=experiment-001`
- leave `MODE=training` (the notebook default).

Cell 8D derives the context from these configuration values itself. It does not
require `REPO`, `PERSISTENT` or `DATA` to have been created by Cell 2. It mounts
Drive if necessary and obtains only the diagnostic code when the local checkout
does not include 8D. It does not run `ensure_workspace`, write session logs or
change saved training code revisions. The training checkout is preserved.

No Cell 2/3/4/5/6/7/8/8A/8B/8C, GPU, Hugging Face account/token, dataset terms
acceptance, online translation or model download is needed. A live notebook
kernel and a mounted Drive containing the saved experiment are needed.

Settings in Cell 8D:

| Setting | Default | Meaning |
|---|---|---|
| `TEXT_AUDIT_PER_STRATUM` | 100 | Independent upper limit in each of the four strata |
| `TEXT_AUDIT_SEED` | 20261025 | Fixed deterministic sampling and blind-order seed |
| `TEXT_AUDIT_SOURCE_MANIFEST` | empty | Optional path to the **locked accepted** manifest if common metadata is missing |
| `TEXT_AUDIT_FETCH_CODE` | True | Permit diagnostic-only local Git code retrieval |

The core uses Python standard-library modules plus the repository's lightweight
atomic I/O/progress helpers. The UI uses the **kernel's existing ipywidgets 7/8
and IPython**; local widget tests use ipywidgets 8.1.7 on Python 3.10 and 3.13.
The kernel dependency check runs before diagnostic output files are written.
A standard Colab Python runtime provides widgets; if missing, 8D stops with an
explicit dependency error and does not install packages into a training runtime.
The Colab custom widget manager is enabled by the launcher. The UI must be
executed in Colab/Jupyter; GitHub's static notebook view cannot accept annotations.
See the [Jupyter widget installation documentation](https://ipywidgets.readthedocs.io/en/stable/user_install.html)
and [button callback documentation](https://ipywidgets.readthedocs.io/en/stable/examples/Widget%20Events.html).

## What to annotate

Read the Japanese and English texts directly. Choose:

| Label | Meaning |
|---|---|
| `correct` | Same meaning; natural rewording is allowed |
| `acceptable` | Core meaning corresponds, with minor omission/addition/nuance differences |
| `partial_mismatch` | Some correspondence but consequential missing/extra material or boundary mismatch |
| `wrong_alignment` | Different content, reversed meaning or a substantially unrelated pair |
| `uncertain` | Insufficient context or expertise; do not force a definitive label |

Optional flags, confidence and note are collected independently of the primary
label. Select flags/confidence and write any note **before** clicking a label
button: each label button saves immediately and advances. Alternatively choose
the label dropdown and click **Save & Next**. A missing label never advances.
`uncertain` is saved with low confidence.

The UI contains only texts, text-only raw/TTS disclosure, human controls and
aggregate progress. No corpus, split, pair ID, source line, predictions, BLEU,
embeddings or automatic judgments are displayed. Startup population sizes are
shown as requested; they do not identify individual samples. Text itself can
suggest domain/style; this is metadata blinding, not a guarantee that annotators
cannot guess the source from content.

**Previous** enables explicit correction of an existing answer. Otherwise existing
answers are preserved. Use one annotating runtime per audit. Each answer is flushed,
fsynced and atomically replaced in `annotations.jsonl` before advancing. Derived
report views can be rebuilt after interruption. An OS advisory save lock releases
on process termination; no stale lock removal is needed after a normal runtime
reset. A mount that cannot support locking/atomic replacement fails closed.
No software can guarantee a write survives a Drive disconnection before Drive
acknowledges it; the UI stays on the sample if saving fails.

After a disconnect, run **Cell 1 → Cell 8D** with the same output base, experiment,
seed and per-stratum limit. The first unannotated sample is selected, previous
answers retain their original timestamp and contents, and old in-kernel UIs are
disabled before opening a replacement UI.

## Source of truth and corpus provenance

The source is exclusively:

```
<experiment>/data/common/dataset-lock.json
<experiment>/data/common/train.jsonl
<experiment>/data/common/test.jsonl
```

The lock must be schema 1, identify the accepted source manifest and SHA256, and
have split counts matching the actual experiment manifests. Existing
`data/{s2ut,translatotron2}/fairseq/data-lock.json` files, when present, must bind
the same common dataset lock. Their hashes are included in audit identity.
The diagnostic records the current experiment manifests; it does not claim to
recover a different historical manifest whose contents have since been replaced.

`corpus` is taken from the explicit common-manifest field, case-normalized to
`kftt`/`jesc`. **No pair-ID prefix inference is permitted.** If metadata is missing,
8D can stream the locked production accepted manifest, verify its SHA256 and join
only the missing experiment keys by `pair_id`, split, canonical texts and source ID.
Duplicate/missing/mismatched joins are excluded with saved reasons. Raw extra
rows never become audit candidates. If a required population has no provable
corpus provenance, processing stops and a diagnostic-only failure receipt is saved.
Other malformed/inconsistent input evidence stops with an explicit error.

Main texts are the experiment's exact `ja_text`/`en_text` (no new normalization).
When present, `ja_text_raw`/`en_text_raw` and `ja_tts_text`/`en_tts_text` are available
in **Show provenance (text only)** and the final export. Audio paths are not used.
Missing canonical texts are explicitly excluded and reported; eligibility and
exclusion coverage limit generalization of the final rates.

## Sampling, identity and storage

For each `train/test × kftt/jesc` stratum, take up to N pairs by ascending:

```
sha256(AUDIT_VERSION + NUL + seed + NUL + population + NUL + corpus + NUL + pair_id)
```

Mix all selected pairs by ascending:

```
sha256(AUDIT_VERSION + NUL + 'blind' + NUL + seed + NUL + pair_id)
```

Both use UTF-8 and pair-ID tie breaking. Version, exact experiment root, input
paths and byte SHA256s, N, seed, selected IDs/strata/original splits, exact canonical
and optional text hashes and blind-order algorithm form the identity SHA256.
Changing any input/configuration creates a separate audit with no copied labels.
The queue is validated against identity on resume; unknown/duplicate annotations,
queue changes and loss of saved annotation progress are errors, not a reset.

```
<experiment>/diagnostics/text-pair-audit/<identity-sha256>/
  identity.json           # INTERNAL mapping/provenance, do not inspect while annotating
  sampling-audit.json     # INTERNAL eligibility/exclusion reasons and stratum sizes
  audit_queue.jsonl       # INTERNAL blind ID → original ID/text/stratum mapping
  annotations.jsonl       # blind audit_id, human label/flags/confidence/note/timestamp
  summary.json           # counts only until completion
  report.md              # progress only until completion, final report afterwards
  per_stratum.csv        # created only after all answers are saved
  per_sample.csv         # created only after all answers are saved
```

The 16-hex-character `audit_id` hashes identity and pair ID; collisions are rejected.
Internal files are not encrypted or access-controlled against the annotator.
Do not open them manually before completion. No derived files are written under
the production corpus, common manifests, official benchmark results or checkpoints.

## Statistics and unblinding

Before completion, only `annotated`, `remaining` and `total` counts are published.
All samples must have a primary human label before corpus-level results are shown.
Each stratum reports all five counts; decided = total − uncertain. Rates:

- strict correct = correct / decided
- usable = (correct + acceptable) / decided
- partial = partial_mismatch / decided
- wrong alignment = wrong_alignment / decided
- serious error = (partial_mismatch + wrong_alignment) / decided
- uncertain = uncertain / total

Wilson 95% intervals are computed without external statistics libraries. Empty
denominators produce N/A, not zero. The report includes JESC−KFTT serious-error
differences separately for train/test, flag counts by corpus, sampling coverage,
exclusion reasons and source hashes. It makes no automatic causal claim, no
significance claim and no unweighted pooled estimate from the balanced strata.

## Real-manifest verification (2026-10-25)

The current Drive `experiment-001/data/common/` manifests were downloaded read-only
through the connected Google Drive plugin; no audio/model was downloaded. The
190,348,211-byte train manifest matched the previously saved local copy:
SHA256 `81fd9a06452c101286758a9df3b4030179be57243516215c6f16850c1303db1f`.
The dataset lock identifies the original accepted source SHA256
`cb450635c91e78f16ba16d2e774687e0bbf403a4a76ee8adddbb7c3ebd863670`.

| Population | Corpus | Manifest pairs | Default selected |
|---|---|---:|---:|
| train | kftt | 12,669 | 100 |
| train | jesc | 128,709 | 100 |
| test | kftt | 225 | 100 |
| test | jesc | 971 | 100 |

No canonical/TTS-text differences were found in these manifests. The actual
Colab audit identity and complete save path are printed when Cell 8D validates
the manifests and optional prepared-data locks at its mounted paths. A local
verification run has a different identity because absolute experiment/input
paths are intentionally part of identity. No real human labels were generated
or fabricated during implementation testing.

## Implementation validation (2026-10-25)

- Full CPU suite: **989 passed, 2 skipped, 8 GPU tests deselected** (266.09 seconds).
  The skipped tests require a separately configured real eSpeak executable or
  Windows symlink privileges; GPU tests remain opt-in. Existing third-party SWIG and
  PyTorch weight-norm deprecation warnings are not test failures.
- New 8D suite: **45 passed** on Python 3.13.7 and **45 passed** on Python 3.10.18,
  with actual ipywidgets 8.1.7 button callbacks (not a fake widget implementation).
- Tests include deterministic independent-stratum selection/blind order; metadata
  rather than misleading ID prefixes; strict accepted-source joins and rejection
  reasons; undersized strata; identity changes; unknown/missing saved answers;
  per-answer atomic save; failed Drive-write simulation; explicit correction;
  cold resume; forced process termination during a save lock; completion-only
  unblinding and CSV mapping; uncertainty denominators and Wilson intervals.
- Normal Cell 1 → 8D is exercised with no Cell 2 variables, no GPU/HF/model
  dependencies, an existing code checkout and a blank-code-runtime local Git
  fetch/worktree fixture. Training checkout/session/revision files stay unchanged.
- A fresh `python -S` process runs the core while refusing model/audio/scoring
  imports. Python 3.10 syntax is checked for the backend, launcher and notebook cell.
- `pip check` in the diagnostic Python 3.10 environment: no broken requirements.
- The real train/test manifest copies produced **400 selected**, **0 excluded**,
  **0 human annotations** and an identical queue on rerun. Only local diagnostic
  output was written. All common-manifest byte hashes stayed unchanged.
- Local real-manifest verification identity:
  `ff57e0b505086ddff2d5b8d7a45c664cc3748da7f7f5a2399a3b50ea523e7ed8`.
  This is **not** the eventual Colab identity: absolute input/root paths and any
  available prepared-data lock receipts are part of identity.
- Source test manifest SHA256:
  `2bff9a7d23c669f486ebc7a32efaef8f252f62c17c9b44a9aaa6265895761a3a`.
- Existing Cells 1–8C and the remaining notebook cells are byte-for-byte unchanged
  as cell objects relative to commit `837f379f2a6f37e4c28f64f685d0a677090de9b4`.
- A real Colab browser widget session and Google Drive mount failure/reconnect have
  not been exercised from this PC. Local tests cannot prove remote Drive durability
  or frontend availability; the user-facing save errors do not advance the sample.

## Changed files

- `notebooks/colab_training.ipynb`: add only Cell 8D immediately after 8C.
- `scripts/colab/text_pair_audit.py`: kernel dependency check, isolated code namespace and UI launch.
- `src/direct_s2st/diagnostics/__init__.py`: lightweight diagnostics package.
- `src/direct_s2st/diagnostics/text_pair_audit.py`: accepted-text provenance, deterministic queue, atomic human annotations, resume and reports.
- `src/direct_s2st/diagnostics/text_pair_audit_ui.py`: blinded ipywidgets controls and completion-only report display.
- `tests/unit/test_text_pair_audit.py`: 45 new cases including real widgets, real local Git bootstrap and interruption recovery.
- `tests/unit/test_cascade_corpus_diagnostic.py`, `tests/unit/test_cascade_mt_diagnostic.py`, `tests/unit/test_flores_fleurs_diagnostic.py`: allow the additional notebook cell while preserving existing-cell checks.
- `docs/COLAB.md` and this document: operation, evidence, limitations and validation.
