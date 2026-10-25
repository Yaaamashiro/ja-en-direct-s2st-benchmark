# Cell 8E: saved S2UT on the identical 8C Natural sample set

Cell **1 → 8E** on a GPU runtime starts or resumes the external evaluation.
Use the same `OUTPUT_BASE`, `MODE` and `EXPERIMENT` as the completed 8C experiment.
Cells 2–8D do not need to be rerun. Do not enable training, replace the training
test set, or regenerate 8C. L4 is the intended sequential inference runtime;
real GPU validation is still pending. A100 is not a recipe requirement. The reused 8C evaluation bootstrap currently
requires a CUDA GPU with BF16 support (L4/A100); it explicitly rejects T4.

## Controls

| Cell 8E field | Default | Meaning |
| --- | --- | --- |
| `FLEURS_S2UT_RUN_NAME` | empty | Existing Cell 6 selector: saved S2UT pointer, otherwise a unique durable run. Multiple ambiguous runs require an explicit name. |
| `FLEURS_S2UT_CASCADE_ID` | empty | Prefer a completed saved 300-item 8C evaluation; choose most recent within that group. An explicit 8C identity or its absolute folder path selects another saved set. |
| `FLEURS_S2UT_RESUME` | true | Restore successful stages and reports under exactly the same identity. |
| `FLEURS_S2UT_RUN` | true | Run inference/evaluation. False only inspects saved 8C metadata, without models or runtime installation. |

There is no new LIMIT, seed, random selection or tuning control. The selected
8C identity and count are printed **before** dependency restoration. Review this
line if several 8C evaluations exist. `TRAIN_TARGET`, `BENCHMARK_TARGET`,
`TRAIN_CODE_REVISION` and the training run name setting do not select the 8E run.
8E uses its own selectors and a separate, immutable diagnostic code checkout.

## Source of truth and checks

The completed 8C folder is the sole source of selected samples:

1. `identity.json`: canonical IDs, FLORES ID/split, FLEURS recording ID/split,
   Japanese text, English reference, SHA256s, locator, revisions and ASR identity.
2. `dataset-audit.json`: must match the dataset inside the identity, including
   exactly the same selected sample set.
3. `per_sample.jsonl`: joined by canonical ID, never by row position. It must
   match the embedded per-sample rows in `metrics.json`.
4. `stage-results/<natural request digest>.json`: checksummed 8C natural-audio
   receipt, bound to recording locator, source text SHA, source dataset revisions
   and exact WAV SHA. The saved `natural/` WAV is copied byte-for-byte and checked
   as finite, nonsilent mono 16 kHz. No resampling, re-selection or dataset download.
5. `execution-audit/evaluation_asr.json`: actual English Whisper dtype and
   generation config, checked again after loading the 8E evaluation model.

Missing/duplicate/extra IDs, changed text, changed WAV bytes, incompatible ASR
or unsupported source audio fail closed. A failed upstream Cascade TTS does not
exclude its reference from the S2UT evaluation. A missing original natural WAV
requires restoring that 8C artifact, not selecting a replacement recording.

`metrics.json` supplies `natural_final_ASR_BLEU` for comparison. 8E reads the
saved number; it does **not** recompute or overwrite Cascade scores. For the
inspected experiment, the saved 8C identity was
`a3824b2b4256907424503e22c55518a90d7d16814224ea31572c7a4c557672a5`,
with 300 samples and Natural final ASR-BLEU **21.153432814483**.

## Existing model and external fairseq input

The original Cell 6 `saved_run` restoration verifies the durable snapshot and
training configuration. `select_checkpoint(..., 'best')` retains the existing
Dev unit-loss selection; FLEURS is never involved in selecting a checkpoint.
The best update, reached training updates and checkpoint SHA256 are recorded.
Saved prepared metadata, dictionaries, `data-lock.json` and the exact HuBERT
L6/KM100 permutation required by the official vocoder are verified read-only.

Only local `<local>/<identity>/prepared/` is written:

- byte-identical saved feature `config.yaml` and unit dictionary;
- verified auxiliary dictionaries and side assets;
- `config_multitask.yaml` with only `data`/`dict` locations relocated;
- external-only `test.tsv` with the same source frame convention (16 kHz / 160
  samples = 10 ms), copied Japanese WAV paths and fixed dummy target Unit `0`;
- dummy auxiliary `<unk>` labels required by the pinned fairseq data loader.

No FLORES English reference, gold English Unit, or source transcript is written
into fairseq inference inputs. The 8E-only entry point removes dummy targets,
auxiliary labels and `prev_output_tokens` from the sample passed to generation;
target prefixes and constraints are forbidden. English references are attached
only after the fairseq subprocess exits. Training `common/test.jsonl`, prepared
`test.tsv`, `target_letter/test.tsv`, dictionaries and dataset locks stay intact.

## Scientific conditions and authoritative code review

No official model/recipe setting is changed for 8E. The affected path was checked
against the fixed fairseq submodule
`3d262bb25690e4eb2e7d3c1309b1e9c406ca4b99`, specifically
`speech_to_speech.py`, `speech_to_speech_dataset.py`, `fairseq_cli/generate.py`
and `sequence_generator.py`, and the existing benchmark adapters.

| Setting | Existing full-test / 8C value | 8E |
| --- | --- | --- |
| S2UT model, optimizer, scheduler, vocabulary | Saved trained run | Unchanged; no training |
| Checkpoint selection | Dev-best Unit loss | Unchanged |
| Beam / max-len-a / batch / batch multiple | 10 / 1 / 1 / 1 | Same `generation_command` |
| Target code size | 100 | Same |
| Feature extraction and validation transforms | Saved `config.yaml` | Byte-identical; train-only transforms not used |
| Unit vocoder | Official HuBERT100 LJSpeech `g_00500000` with predicted duration | Same `benchmark_worker.audio` and SHA pins |
| English evaluation ASR | Whisper large-v3-turbo, revision `60be3615a4d667e1258e8ad29130467587c489aa` | Same adapter, actual dtype/config checked |
| ASR task | English / transcribe / do_sample=false / condition_on_prev_tokens=false | Same |
| English normalization and BLEU | `nfkc_lower_english_v1`, sacreBLEU 2.5.1 | Same functions, all selected failures as empty |

Dummy labels and source-only adapter are data-loader accommodations, not a new
decoder strategy: pinned `SequenceGenerator` initializes from BOS and source
features when prefix size is zero. The copied dictionaries preserve the model
shape and auxiliary module loading. No unknown new feature/speaker conditioning
is silently approximated. The effect of this adapter on real GPU outputs has
not yet been measured; CPU tests verify its input boundary and pinned command.

## Dependencies and recovery

S2UT/fairseq and Unit vocoding use the **exact saved training environment**,
restored through the same environment recovery as Cell 6. English evaluation
uses the separate **8C environment/bootstrap** and compares the recorded Python
and relevant package versions with 8C (including numpy, torch, transformers,
tokenizers, audio libraries and sacreBLEU). This avoids installing 8C numpy 2.2.6
into a fairseq environment using numpy 1.26.4. Processes execute sequentially,
releasing S2UT and vocoder GPU memory before evaluation Whisper loads.

No new dependency set is introduced. If the 8C runtime was lost, its existing
fully pinned `diagnostic-flores.txt` is installed in its local dedicated venv.
That shared lock includes Qwen/pyarrow packages, but **8E never loads Qwen/NLLB
weights, synthesizes Japanese audio or accesses gated FLORES datasets**. Public
S2UT vocoder/Whisper weights may be downloaded. Training environment packages
are not replaced with 8C packages. If the notebook's launcher Python lacks
PyYAML, only PyYAML 6.0.2 is installed into a dedicated local launcher folder;
the notebook kernel's installed packages are not modified.

Each successful Unit shard is atomically ZIP-published with SHA receipts. WAV
generation directly reuses existing benchmark groups/archives; successful WAVs
are restored without synthesis. Once the full waveform stage has been published,
that manifest (including reported sample failures) stays immutable during ASR
resume; failed WAV rows do not change timing/identity or invalidate successful ASR.
English ASR uses the existing journal and ZIP
receipts (every 16 processed samples, and at completion). Same-VM local ASR rows
newer than the last ZIP are merged with the verified archive, preserving successful
results. Conflicting successful rows, references or journal identities fail closed.
Copied prepared files/source WAVs are verified again before decoding; locally
completed Unit shards whose upload failed are checked against their generation log.
A disconnect can repeat
at most the unpublished ASR tail; no unsynced Drive write is guaranteed durable.
Runtime/dependency/model-load errors abort before per-sample scoring. Genuine
generation/vocoder/ASR sample failures remain as empty hypotheses. Unit generation
failures saved in a completed shard are retained as failures, not silently retried
under a different decoding condition.

Same settings: **Cell 1 → 8E** resumes, including after runtime replacement.

The package-name repair (launcher version 2) normalizes distribution names using
case-insensitive `[-_.]+` aliases, while requiring identical exact versions and
Python/CUDA/cuDNN/TF32 settings. The saved 8C lock is checked before S2UT inference;
the actual restored ASR environment is checked again before scoring.

An interrupted pre-ASR `4e18c8f872561c57a6cb32f115164f5c331db972` plan can retain its
identity, Unit shards and WAV ZIPs through an explicitly recorded execution overlay.
Adoption requires identical scientific inputs/settings and reviewed code fingerprints:
unchanged generation/vocoder/scoring/dependency code and unchanged scientific function
bodies in the diagnostic. Original plan/archives remain byte-identical. The new report
records both execution revisions. Different settings/checkpoints use separate identities.
Already scored legacy results without an overlay are not adopted. Rerunning an adopted
result preserves its ASR progress and completed report checksums.
Completed reports are SHA-verified and displayed without rebuilding inference
environments. Their saved checkpoint still must be restorable. Run only one
8E execution per experiment/identity at a time. This does not provide a
cross-runtime distributed Drive locking service.

## Outputs

`<PERSISTENT>/diagnostics/flores-fleurs-s2ut/<identity>/` contains:

- `plan.json`: all pinned identities, exact source mapping and implementation SHAs;
- `external-input-audit.json`: exact copied prepared files and input WAV SHAs;
- `units.jsonl`, `generation/`: Unit sequences/SHAs and resumable ZIPs;
- `predictions.jsonl`, `audio/`, `audio-index.json/csv`: generated English PCM16
  WAV ZIPs, member paths, output SHAs and errors, using the existing archive format;
- `evaluation-runtime.json`, `evaluation-asr-audit.json`, `progress/`: dependency
  identity, actual ASR settings and recoverable English ASR journals;
- `metrics.json`, `per_sample.jsonl/csv`, `report.md`, `completion.json`:
  all-sample corpus ASR-BLEU, separate generation/vocoder/ASR failures, checkpoint,
  best/reached updates, revisions, source identity, saved Cascade value and
  **S2UT minus Cascade** difference. Per-sample JSONL includes IDs, texts, Units,
  generated WAV identity, ASR text, sentence BLEU and status/error.
  Both per-sample formats also name the durable ZIP/member and archive SHA256;
  the local WAV path alone is not needed after a runtime reset.

Existing Cell 9 continues to use the formal benchmark namespace; it is not
redirected to 8E. 8E WAVs can be extracted using their own `audio-index.json`
and the existing SHA-verified ZIP reader.

## Validation boundary

CPU tests cover exact joins/references, changed audio rejection, changed checkpoint
identities, reordered rows, duplicates/missing rows, no reference leakage,
original-artifact byte preservation, source-only generation, ZIP/journal recovery,
failure-inclusive BLEU, dependency mismatch rejection, model-free completion resume
and unchanged existing notebook cells. Real L4 inference/model download/Drive
remount has not been executed as part of local implementation testing.

Initial release local verification (`4e18c8f`, 2026-10-25):

- Entire CPU suite: **1032 passed, 2 skipped, 8 GPU tests deselected**, 312.16 s.
- New 8E tests: **43 passed** under Python 3.10.18 and Python 3.13.7.
- Additional push review covered interruption with 16 archived/17 local ASR rows,
  same-VM and VM-loss recovery, conflicting journal rejection, modified external
  input rejection, interrupted Unit publication and immutable completed waveform
  stages containing sample failures.
- Python 3.10 diagnostic environment: `pip check` reports no broken requirements.
- Python 3.10 syntax checks and `git diff --check`: passed.
- All **20 existing notebook cell objects** remain identical; only 8E was added.
- Skips: no explicitly configured real eSpeak executable, and unavailable Windows
  symlink privilege. GPU inference tests were deliberately not executed.
- Windows validation used a writable TEMP/TMP and short pytest `--basetemp`
  (`../e8w`), avoiding sandbox temporary-file permissions and the legacy Windows
  path-length limit in existing environment-recovery tests. No unrelated pipeline
  implementation was changed to accommodate these local test constraints.

Changed files:

- `notebooks/colab_training.ipynb` (one new 8E cell only)
- `scripts/colab/s2ut_flores_fleurs_diagnostic.py`
- `src/direct_s2st/s2ut/flores_fleurs_diagnostic.py`
- `src/direct_s2st/s2ut/external_generate.py`
- `docs/COLAB.md`, `docs/S2UT_FLORES_FLEURS_DIAGNOSTIC.md`
- `tests/unit/test_s2ut_flores_fleurs_diagnostic.py`
- Notebook compatibility assertions only in `test_cascade_corpus_diagnostic.py`,
  `test_cascade_mt_diagnostic.py`, `test_flores_fleurs_diagnostic.py`, and
  `test_text_pair_audit.py`.

## Package-name repair verification

The real saved 8C `identity.json` uses `huggingface_hub: 0.36.2`. The pinned
requirements and installed distribution version also require `0.36.2`; this was
a name-alias comparison error, not permission to change package versions.

Regression coverage includes actual installed distribution metadata, underscore /
hyphen / dot / case aliases, missing and conflicting names, genuine version
mismatches, Python/CUDA/cuDNN/TF32 mismatches and rejection before checkpoint or
model restoration. Recovery tests execute the original released Python source
from Git, create real recovery ZIPs with lightweight inference doubles, then
verify unchanged plan/Unit/WAV bytes with or without the previous VM/code cache.
They cover ASR interruption after adoption, completed model-free resume through
the normal launcher, altered model/reference/code rejection, and repaired Cell 8E
bootstrap from Cell 1. Neural inference remains outside these CPU tests.

Local repair verification (2026-10-25):

- Full CPU regression suite: **1064 passed, 2 skipped, 8 GPU tests deselected**
  (305.99 s; collected before the last normal-launcher integration test was added).
- Final 8E suite including that integration test: **76 passed** on Python 3.13.
- Python 3.10.18: **75 passed**, then **4 passed** for final normal-launcher /
  Cell 1-bootstrap / notebook compatibility checks, including the added test.
- Actual saved 8C Python/package preflight and local Python 3.10 `pip check`: passed.
- Skips remain the unconfigured real eSpeak executable and Windows symlink privilege.
- GPU/Colab model execution was not performed for this repair. Model settings,
  package pins, original 8C artifacts and the 20 earlier notebook cells are unchanged.

## Completed-result display repair

The shared `run_json` receiver previously required the 8A-only
`all_score_matches_saved=True` field for 8E as well. 8E produces a new external
score, so that assertion is inappropriate even when its worker, report and
completion receipt all succeeded. The result receiver now uses explicit
contracts: `saved_formal_all` (unchanged default for 8A/B/C), `external_s2ut`, and
`external_s2ut_inspect`. External reports must contain their expected purpose,
Natural condition, positive sample count, finite BLEU/baseline/difference and
display/output fields; no fabricated formal-score agreement flag is inserted.

Normal Cell 8E loads the repaired display code separately while keeping scientific
execution pinned to `32cc967f01808d5abc82df2d709b5c8262a9e961` at its existing
`/content/s2st-flores-fleurs-s2ut-code/<SHA>` path. The actual code-byte fingerprints
and absolute execution paths therefore retain the prior evaluation identity,
including a `4e18c8f` plan already adopted through the package-name repair. Cache
loss fetches that exact scientific SHA; stale display code fetches the current
wrapper. No new dependencies or scientific implementation changes are required.

With the same settings and resume enabled, the original launcher verifies the
completed report checksums and returns it without loading inference models or
rerunning Unit/vocoder/ASR stages. Existing result files are not rewritten. The
20 earlier notebook cells are unchanged.

Regression tests exercise real subprocess stdout/stderr JSON reception, an actual
completed launcher result, full notebook execution with the real receiver (warm
cache, cold cache, stale display wrapper, inspect-only), wrong report rejection,
continued strict formal-score checking and byte equivalence of every scientific
implementation file to the pinned release. Model work is replaced with local test
fixtures; Colab GPU inference is not part of these tests.

Local display-repair verification (2026-10-25):

- Full CPU regression suite: **1086 passed, 2 skipped, 8 GPU tests deselected**
  (283.74 s; before the last perfect-BLEU roundoff case was added).
- Final related runtime/8E tests: **128 passed** on both Python 3.10.18 and 3.13.
  This includes sacreBLEU's actual exact-match value `100.00000000000004`;
  validation accepts numerical roundoff without rounding or rewriting saved scores.
- A `python -S` parent and child verify stdlib-only result transport; local
  Python 3.10 `pip check` reports no broken requirements.
- All scientific implementation files compare byte-for-byte (with checkout LF
  normalization) against `32cc967`. The 20 earlier notebook cells are unchanged.
- Saved-result/worker logic and dependency pins have no implementation changes.
