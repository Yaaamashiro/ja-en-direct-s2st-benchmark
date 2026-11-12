# Google Colab smoke tests and standard experiments

## New S2UT runs with the official Fisher recipe and training-time BLEU

`S2UT_RECIPE='paper_fisher_v1'` starts a fresh run separate from legacy shortened runs. It uses **10,000 warmup updates**, an absolute maximum of **400,000 optimizer updates**, and no custom early stopping. Model, auxiliary tasks, loss weights, Adam, inverse-square-root schedule, LR 0.0005, FP16, seed 1, max_tokens 20,000, and update_freq 4 follow the official recipe. This official single-GPU setup accumulates four GPUs' worth of gradients. Numerical equivalence to a four-GPU run is unverified.

Metadata records the Japanese–English corpus, public LJSpeech vocoder weights, and pinned Whisper evaluation ASR. The original study used Fisher, a vocoder fitted on Fisher, and wav2vec2 ASR. Translation quality needs empirical evaluation. See the [paper](https://arxiv.org/abs/2107.05604) and [fairseq recipe](https://github.com/facebookresearch/fairseq/blob/main/examples/speech_to_speech/docs/direct_s2st_discrete_units.md).

Set the following in standard Cell 1, then run Cell 5. Reuse the completed 4A and 4B artifacts.

```python
TRAIN_TARGET = 's2ut'
S2UT_RECIPE = 'paper_fisher_v1'
S2UT_WARMUP_UPDATES = 10000
S2UT_TOTAL_UPDATES = 400000
S2UT_VOCODER_MODE = 'paper'
REPRODUCTION_MODE = 'paper_exact'
TRAIN_RUN_NAME = ''
RUN_TRAINING = True
CONFIRM_TRAINING = True
SESSION_MINUTES = 0
```

For the first use of the recipe, set `TRAIN_CODE_REVISION` to the full published SHA. An incompatible old launcher stops before execution. The new run is named `s2ut-training-<experiment>-paper-fisher-v1` and starts with fresh optimizer state. Resume through **Cells 1 → 5** with the same settings to restore the saved environment, optimizer, and evaluation results. To continue an old run, use `legacy_short` and its saved warmup, mode, and run name. Explicitly selecting an old run with the new recipe stops on mismatch.

### Training-time measurements on dev

| Measurement | Update schedule | Procedure |
|---|---|---|
| Full dev loss | Every 1,000 | Teacher-forced evaluation on all dev samples, recording all five losses |
| Auxiliary text BLEU | Every 1,000 | Up to 500 fixed dev samples, from Japanese speech through free target-letter generation to English text |
| Audio ASR-BLEU | 1k / 5k / 10k / 25k / 50k / 100k / 200k / 400k | Up to 100 fixed dev samples, from Japanese speech through beam-10 unit generation, official unit vocoder, and English ASR |

Auxiliary text BLEU is diagnostic and stored separately from audio ASR-BLEU. Fixed IDs use seed 1 and pair-ID SHA-256 ordering. The audio subset is drawn from the 500 text samples. Generation receives no target units, reference text, or previous-output targets. English references are used only for scoring. Failed samples are empty hypotheses and remain in the denominator, with failure counts and errors saved. Use `openai/whisper-large-v3-turbo` at `60be3615a4d667e1258e8ad29130467587c489aa` for English transcription. Verify the unit vocoder weights and config with SHA-256.

Evaluation alternates with training on the same GPU and pauses optimizer updates. The first model fetch adds network, VRAM, and disk costs. Inference and ASR add time. Check the 30-minute budget between items and mark remaining items as failed after expiry. The budget does not interrupt an individual item or download. Store a null metric if the evaluation fails as a whole and save completed updates in the last checkpoint. CUDA OOM and device-side assertions stop the process for recovery from the previous durable snapshot. Restore model modes, buffers, and Python, NumPy, and Torch RNG after evaluation. The optimizer remains untouched.

Choose the best full checkpoint by minimum full-dev `unit_loss`. BLEU does not control stopping or model selection. Resume from the last checkpoint. Retain train loss every 100 updates and dev loss every 1000 to bound history replay. A checkpoint at 400k performs no further optimizer updates.

### Charts, storage, and compute

Publish `training-monitor.svg/.csv/.json` for loss and `training-bleu.svg/.csv/.json` for both BLEU series in `PERSISTENT/checkpoints/<saved-run-name>/`. Include the small per-update identity, fixed IDs, successful and failed hypotheses, and scoring signature in verified snapshots. Reuse successes only when evaluation weight SHAs and policies match. Retry failures. Model weights for evaluation ASR and vocoding and generated WAVs stay outside recurring training backups.

`S2UT_PAPER_SAVE_INTERVAL=100` saves the last checkpoint. Asynchronous backup normally publishes every 600 seconds and is forced at dev evaluation and completion. Losing a VM returns training to the last durable snapshot. Local-only evaluation journals can be lost and then replayed. Keep snapshots for every full-dev and BLEU measurement, with bounded retention for other updates. If the default 1024 GB Drive budget is exceeded, stop and preserve files. Check storage and actual update speed for the long 400k run. A shorter trial records its actual training budget.

## Experimental TT2 stabilization recipe (new runs only)

Select `TT2_RECIPE='stabilized_seconds_v1'` explicitly in Cell 1 and start a new run in standard Cell 5. The default `legacy_frames` retains old checkpoint semantics. The new recipe is recorded as `paper_practical`.

- The duration head and utterance-length MSE use seconds, initialized at 80 ms per acoustic token in the phoneme sequence. Upsampling, Gaussian width, and inference max_frames use frames converted with the actual target hop.
- Use peak LR 0.0003 and warmup 400 from the start with `TT2_WARMUP_UPDATES=400`. Resume rejects changes to LR, warmup, or objective. The legacy mid-run shortening overlay does not apply.
- Keep Fisher dimensions, Adam/L2, loss weights 1/10/1, effective batch 1024, full dev evaluation every 10 updates, policy v3, best by dev total loss, and cap 1000.
- Probe VRAM with this recipe and keep physical batch × accumulation at 1024. Speed and maximum occupancy require hardware measurement.

Set `TRAIN_TARGET='tt2'`, `TT2_RECIPE='stabilized_seconds_v1'`, `TT2_WARMUP_UPDATES=400`, `TRAIN_RUN_NAME=''`, `RUN_TRAINING=True`, and `CONFIRM_TRAINING=True` in Cell 1, then run Cell 5. With an existing launcher pin, set `TRAIN_CODE_REVISION` to the compatible published 40-character SHA.

The run name is `tt2-training-<experiment>-paper_practical-stabilized-seconds-v1`. Automatic selection stays within the recipe. An explicitly selected legacy run stops on mismatch. After disconnection, use Cells 1 → 5 with the same settings and reuse 4A/4B, audio, mel, and phonemes. Duration and total losses use different scales from legacy runs. Resume old runs with `legacy_frames` and their original settings and names. S2UT and vocoders retain their training conditions.

The recipe came from local diagnostics. The TT2 paper does not establish the exact duration units and initialization. A short two-item CPU check on real data leaves full-corpus convergence, translation quality, and A100 performance unverified.

## Optional TT2 recipe with evaluation-only postnet calibration

`TT2_RECIPE='stabilized_pooled_v2'` in Cell 1 keeps v1's objective, initialization, LR, warmup, and effective batch of 1024. It calibrates postnet statistics only for evaluation and inference. This is a separate recipe with the same notebook default as before. Calibration changes statistics without updating model weights or optimizing phone loss.

- Select up to 128 train IDs in SHA-256 order and compute teacher-forced raw mel with batch size four. Calibrate five layers in order with means and unbiased variances over all valid frames. Use train data only. Save raw mel temporarily on CPU and replay the postnet.
- After validation, restore training buffers, RNG, and module modes. Optimizer and gradients are preserved. Embed calibrated values with the update count, parameter SHA, and fixed train IDs. Standard Cell 5 snapshots carry them, and training resume uses native buffers.
- Use single-process FP32 with `TT2_WARMUP_UPDATES=400` and `TRAIN_PRECISION='default'`. Best is minimum calibrated dev total loss. Evaluate every 10 updates under v3 with cap 1000. Save final-update calibration without adding an off-schedule dev evaluation.
- Inference checks and applies the calibration embedded in the selected checkpoint. It rejects uncalibrated intermediate last checkpoints and calibration for different weights. Normally select verified `checkpoint_best.pt`.

In Cell 1 set `TRAIN_TARGET='tt2'`, `TT2_RECIPE='stabilized_pooled_v2'`, and `TRAIN_RUN_NAME=''`, then run Cell 5. The name is `tt2-training-<experiment>-paper_practical-stabilized-pooled-v2`. After disconnection, repeat Cells 1 → 5 with the same settings. Set `TRAIN_CODE_REVISION` to the compatible 40-character SHA after publication. Reuse 4A/4B and keep old v1 and legacy runs with their original history. S2UT keeps its own recipe.

Small CPU trials on real examples observed improved mel and no interference with training updates. Calibration on 128 items, Fisher size, the full corpus, A100, and translation quality remain unverified. Calibrated dev loss uses different measurement conditions from older runs. Preserve prior history, anchors, and best checkpoints.

## Integrated resume in Cell 5

After disconnection, run Cell 1 with the same experiment, then Cell 5. Cell 5 connects Drive, fetches code, and rebuilds the runtime as needed. An empty `TRAIN_RUN_NAME` selects the only saved run for the chosen model. For example, `experiment-001` TT2 selects `tt2-training-experiment-001-paper_exact-recipe-v2-probe-fix-v1` when it has a snapshot and the recipe-v2 run is empty. If several trained runs exist, set the saved name in Cell 1. Unreadable markers or incorrect selection stop resume. Verify snapshot SHA, updates, and configuration, falling back to an older verified snapshot if the newest copy is corrupt.

Resume uses saved config, batch, LR schedule, and monitor state. The current request supplies the total update cap and session time. TT2 based on `48fcdb7` uses the reviewed `d70516a` execution code with the original dependency and snapshot identity. New training such as S2UT records its pinned launcher SHA and uses prepared 4A/4B data.

On the first launcher fetch, resolve origin/main to a full SHA and save it to `training-launcher-revision.txt`. Reuse it on subsequent runs. `<RUN_NAME>.launch.json` records the actual code and environment. Set `TRAIN_CODE_REVISION` to a compatible full SHA only when explicitly updating the launcher. A launcher update preserves saved execution code and optimizer conditions.

TT2's `TT2_WARMUP_UPDATES` defaults to 400, with choices 400/500/1000/10000. Standard Cell 5 can explicitly shorten the schedule and record a `paper_practical` transition in the checkpoint. Runs already shortened to 1000 or 500 can adopt 400 before their saved warmup ends. A run at update 400 with warmup 500 can apply the new schedule on the next update. Dev-loss stopping begins after warmup. Defaults are interval 10, patience 5, and at least 1% improvement. With no improvement at 410/420/430/440/450, it stops at 450. Anchors start at each warmup loss minimum. Best values still record improvements under 1%. At least 1% cumulative improvement from any anchor resets the shared counter and updates that anchor. Preserve weights, Adam, absolute update count, past losses and LR, and best checkpoint. The new LR applies prospectively. Selecting 10000 when resuming inherits the saved schedule. The verified local execution copy changes only schedule, monitor classes, and history restoration. Verify source/config SHAs and audit records and reject unreviewed changes or modified copies. Standard legacy S2UT also uses 400 warmup. Paper configuration files retain their original values.

Cell 1 explicitly requests monitoring policy v3 for TT2 and legacy S2UT. Verify earlier history, counter, and best SHA before migrating. Record the adoption point and both policies in `policy_history`. Preserve past losses, LR, best, model, and optimizer. Initialize each anchor from its historical dev minimum and reset the shared counter. Start counting evaluations after max(warmup end, adoption update). With no improvement, adoption at 400 stops at 450 at the earliest, and adoption at 430 stops at 480. Same-policy resume inherits the counter and stop state. Reject interval changes and downgrades. Explicit migration from stopped v1/v2 can clear its stop for future v3 evaluations.

### Early stopping v3 for all dev losses (TT2 cap 1000, legacy S2UT cap 3000)

TT2 monitors `loss / mel_loss / phone_loss / duration_loss`. Legacy S2UT monitors `loss / unit_loss / source_letter / target_letter / decoder_target_ctc`. For a metric with anchor `A_i` and current value `L_i`, `L_i <= A_i * (1 - 0.01)` counts as significant improvement. Update only improved anchors and reset shared `bad_evaluations` if any metric improves. Five consecutive evaluations with no significant improvement cause a stop. Zero anchors cannot improve. All dev samples and all losses, including total loss, are evaluated every 10 updates.

Every monitored validation loss has an independent anchor. Training continues while at least one
monitored validation loss improves by at least 1%. Early stopping is triggered after five consecutive
validation evaluations in which none of the monitored losses achieves a 1% improvement.

During warmup, record each metric's minimum as its anchor and defer stopping. When migrating, use the per-metric minimum from the saved dev history and preserve prior decisions. `policy_history` records adoption update and old and new policies. Resume restores anchors and counter. Open the updated standard notebook with the same experiment, saved run, and warmup 400. Set the compatible full launcher SHA in `TRAIN_CODE_REVISION` if needed and run Cells 1 → 5. Reuse 4A/4B data.

`training-monitor.json` and the last checkpoint store `stopping_metrics / anchors / bad_evaluations / best_values / best_updates / policy_history`. `best_value / best_update / best_sha256` still identify the inference checkpoint. CSV and SVG retain train/dev history for all losses. Evaluation logs rates of improvement, marks improved metrics with `*`, and reports shared counter and stopping state on stderr. TT2 still selects the best by dev total loss, and S2UT by dev unit loss, independently of stopping.

Standard Cell 5 and v3 trainers enforce absolute optimizer-update caps of `TOTAL_UPDATES=1000` for TT2 and `S2UT_TOTAL_UPDATES=3000` for legacy S2UT. Restoring S2UT at 610 permits at most 2390 more updates. At 3000 it performs none. Older S2UT v3 checkpoints recording cap 1000 restore anchors, counters, best state, and history. Raising the cap alone preserves an earlier stop. Patience resets cannot extend the cap. Vocoders retain the separate `VOCODER_TOTAL_UPDATES=10000` default. Scheduler, optimizer, batch, loss weights, validation split, and validation interval are retained.

The complete dependency environment is atomically saved and read back at `.runtime-environments/<SHA256>.json`. A new runtime uses all saved versions as constraints. When reusing the local venv across runs, restore both versions and the installed package set. If a complete verified manifest exists, verify that the venv is the dedicated runtime, record the previous pip freeze and removal list locally, and remove extra ordinary packages only there. The notebook's host Python, corpus, saved environment JSON, and checkpoints are preserved. Old `session.log` records alone may be incomplete and cannot authorize package removal. If the restored SHA differs, stop and record missing, extra, version, and metadata differences in `/content/s2st-runtime/dependency-recovery-difference-*.json`.

Restore constraints reference the original requirements by absolute path. Nested files such as `preparation.txt` resolve relative to their original directory. Cell 5 regenerates old restore files. When an older TT2 run lacks its environment JSON, infer candidate versions from successful `session.log` entries, install them, and use the environment only if its measured SHA matches the snapshot exactly. Otherwise stop before training. Keep saved hashes and environment JSON intact. Model logs include `tt2-session.log` and `s2ut-session.log`. After first launcher retrieval, integrated recovery handles the work of the former manual migration cell and standard Cell 3. CPU tests cover selection, restoration, and interruption. The full real Colab workflow remains unverified.

For legacy S2UT, `S2UT_WARMUP_UPDATES` defaults to 400, with choices 400/500/1000/10000. Shortening records `paper_practical` and adds a suffix such as `-warmup400`. If `REPRODUCTION_MODE=paper_exact` was selected, keep the 20,000-token budget, accumulation 4, and FP16 and disable extra batch-splitting calibration. Selecting 10000 uses the original name and conditions. Reuse 4A/4B. Mid-run warmup changes are unsupported for S2UT. Cell 5 rejects the mismatch before setup or download. Resume under the original warmup and start a separate experiment for another schedule.

`S2UT_MONITOR=True` evaluates all dev data every 10 updates and selects the best checkpoint by teacher-forced unit loss. Under v3, five consecutive post-warmup evaluations with no 1% improvement on any of the five metrics stop training. An improvement resets the shared counter and updates only that anchor. At warmup 400, the earliest stop is after 410/420/430/440/450. Each anchor starts at its warmup minimum. Smaller improvements still update best values, and only cumulative improvement of at least 1% from the anchor resets the counter. Store the best value and update for each loss. Unit loss remains the checkpoint selection metric. Set `S2UT_MONITOR_INTERVAL=10`, `S2UT_MONITOR_PATIENCE=5`, and `S2UT_MONITOR_MIN_RELATIVE_IMPROVEMENT=0.01`.

Load old v1 with unit-only stopping and v2 with separate unit and total counters. Explicit v3 adoption verifies history and best, records the adoption point, and resets the shared counter. Same-policy resume preserves the counter. The saved execution pin is retained. Write total, unit, source_letter, target_letter, and decoder_target_ctc train/dev curves to `training-monitor.json`, `.csv`, and `.svg`. Weight each loss by task sample_size and report bits. The total includes auxiliary weights. Dev loss drives stopping. The last checkpoint contains history, counters, and best SHA-256, and Drive snapshots include charts and the best checkpoint. Resume from last and prefer verified best for inference and comparison. Monitoring requires a resident single process, fixed batch, full dev, and synchronous checkpoint saves. It is disabled during VRAM probes. Same-policy resume of a stopped run returns `EARLY_STOPPED` with no updates. Explicit v1/v2→v3 adoption can continue future evaluation within the same TT2 and S2UT caps. Drive also gets the chart and CSV. History before initial adoption is unavailable.

### Initial S2UT FP16 overflow

When pinned fairseq logs `gradient overflow detected` and reduces the scale, for example from 128 to 64, that attempt returns `None` without an optimizer update. Only in this case does resident S2UT retry the logical batch and log `[fp16-retry] update=... retry=... loss_scale=128->64`. Verify that optimizer.step did not run, the update count did not change, and the scale decreased normally. Restore RNG, model and criterion buffers, and gradients. Discard monitoring loss and gradient evidence from the failed attempt. Retain the reduced scale and advance the count, loss history, and probe completions only on success. OOM, failure after optimization, minimum scale, and the 32-retry cap stop execution. Preserve FP16, token budget, accumulation, warmup, and Adam settings.

For runs pinned to `2486164` or `e56ca1c`, the new launcher in Cell 5 creates an execution copy with the reviewed FP16 runtime, TrainingAudit, and checkpoint validation, saving, and abnormal-exit recovery. Keep original checkout, config, base dependencies, recipe fingerprint, and snapshot. Record all source/config SHAs in `execution-overlays/`. Reject unreviewed old code and altered copies. If initial overflow left zero updates and no snapshot, rerun Cells 1 → 5 with the same experiment and settings. Reuse 4A/4B. Set `TRAIN_CODE_REVISION` in Cell 1 to a published fix SHA. CPU control-flow and Adam tests cover this path. A100 FP16 retry is unverified.

### First dev evaluation and unpublished checkpoint recovery

S2UT CTC needs `editdistance` for dev evaluation. New Colab and Docker runtimes pin `editdistance==0.8.1`. Before retrieving data, standard Cell 5 runs real CTC training forward/backward and dev forward, including character and word edit distance, on a small CPU input. This runtime probe leaves experiment weights, optimizer, and splits untouched.

For an old environment, verify and preserve saved pip freeze and `environment.json`. Verify SHA-256 for the official PyPI CPython 3.10 Linux x86_64 wheel and install separately. Keep the wheel and manifest and add only the import path to PYTHONPATH. Record the added version, wheel/hash, and base environment SHA in `.runtime-extras/`. CER/WER and original labels are retained.

If a newer unpublished local `checkpoint_last.pt` still exists on the VM, Cell 5 first verifies input hash, finite model and Adam state, recipe fingerprint, monitor history and counters, and agreement of best SHA with the report. On success, publish a separate immutable copy to Drive and resume there. Preserve the local file and old snapshots. If the VM was deleted, resume from the last verified Drive snapshot. Reuse 4A/4B.

In addition to the normal ten-minute interval, S2UT forces a backup after dev and after early stop or completion. After trainer failure, verify and back up the last local checkpoint. If publishing fails, keep local recovery such as `.pending-recovery-*` and report the failure. Forced VM deletion can prevent the recovery code from running. `learned_at_least` includes local saves, and `durable_updates` counts only Drive-published updates.

Local intermediate retention is bounded to the last two interval checkpoints and one epoch. Keep last, best, and historical Drive snapshots. Fairseq prunes local intermediate files only. Training batch, precision, LR, and warmup remain tied to the run.

## Choose a notebook

| Use | Notebook | Data and training | Drive location |
| --- | --- | --- | --- |
| Initial smoke test | [colab_smoke.ipynb](../notebooks/colab_smoke.ipynb) | Five samples per split, two updates, small TT2 | `OUTPUT_BASE/smoke/<experiment>` |
| Standard training and comparison | [colab_training.ipynb](../notebooks/colab_training.ipynb) | Full corpus, explicit update count, Fisher paper_exact | `OUTPUT_BASE/training/<experiment>` |

Edit the first configuration form and run cells in order. The corpus code and data are read-only. Smoke results exercise wiring. Standard experiments follow the published Fisher shape, optimizer, and effective batch while retaining 16 kHz target mel, eSpeak, and the Japanese–English corpus. See [paper recipes](PAPER_RECIPES.md) for settings and migration.

## Workflow

`OUTPUT_BASE` defaults to `/content/drive/MyDrive/ja-en-direct-s2st-benchmark-data`. For resume, use the experiment's original destination. A default change leaves old data at its original path.

1. Select a CPU runtime in Colab. A GPU runtime also works for preparation.
2. Set the corpus location, output location, and experiment name in Cell 1.
3. Run CPU preparation in Cell 4A. Drive connection, code retrieval, and setup equivalent to Cells 2 and 3 are automatic. In the standard notebook, continue on CPU with Cell 4A.5 to stage audio ZIPs. Verify publication to Drive, switch to GPU, and run Cells 1 → 4B with the same settings for HuBERT extraction and S2UT preparation.
4. Choose `TRAIN_TARGET` from `tt2`, `s2ut`, `unit`, or `mel` in Cell 1 and train in Cell 5. The standard trained-vocoder workflow uses TT2, S2UT, and unit runs. The unit vocoder fits on the corpus's English train audio. `S2UT_VOCODER_MODE=paper` uses public LJSpeech weights, which differ from the paper's Fisher-trained weights. Train mel for TT2 HiFi-GAN. Smoke uses four runs. The first standard run per target probes VRAM, saves the result, and starts from initial weights. Defaults are `TRAIN_CALIBRATE_BATCH=True` and 10% safety margin. TT2 keeps effective batch 1024, and resume fixes the selected batch.
5. After training, enable `RUN_BENCHMARK` and run Cells 6–8 to restore, infer with four systems, evaluate, and compare.

Standard `CONFIRM_FULL_DATA` defaults on and prepares the full corpus in 4A/4B. Turning it off stops before preparation. `RUN_TRAINING` defaults off. Enable it only for training and set cumulative `TOTAL_UPDATES` and `CONFIRM_TRAINING`. The shortened caps are `TOTAL_UPDATES=1000` for TT2 and `S2UT_TOTAL_UPDATES=3000` for legacy S2UT. Vocoders use `VOCODER_TOTAL_UPDATES=10000`. Approval to use all data and to train for a long run are separate. Migrate old 4B auxiliary labels to `fairseq-unigram-allchars-v1`. S2UT uses a new recipe-v3 run, and TT2 remains recipe-v2. Retain existing mel, units, and checkpoints.

`SESSION_MINUTES=0` by default sets no local time cap. Cell 5 continues toward its update target without a periodic 60-minute stop. Colab's own limits still apply. Resume from the same run's verified Drive snapshot and recompute only unsaved updates. A positive time budget applies to the training subprocess, including dev checks and its saves. Preparation, calibration, input preflight, and Drive backup wait are outside it. CLI `--session-seconds` defaults to the same zero behavior. Training requires explicit confirmation.

### Unknown character errors in 4B

`unknown source_letter tokens ... ['汎']` is an example of a legacy dictionary rejecting a dev/test character missing from train. The updated Japanese and English character dictionaries register all train/dev/test characters, while optimization uses only train. The CTC Unigram model still fits on English train text and encodes unknown pieces as `<unk>`. The lock records this implementation choice. The original paper does not specify registration from all splits.

After explicitly updating the saved revision, use this additional CPU cell with the same destination and experiment. It requires stored units and Japanese audio ZIPs and stops if either is missing. It skips unit extraction, model download, 4A/mel generation, and training. Incomplete preparation and validation can take time. A completed rerun reuses the receipt.

```python
ensure_runtime(require_gpu=False)
configure_preparation()
run(PYTHON, '-m', 'direct_s2st.preparation_workflow',
    '--stage', '4b-finish', '--profile', PROFILE)
```

S2UT checkpoints from the previous vocabulary cannot continue on the new model. Before training, run Cell 1 in the latest notebook with the same settings and use the S2UT recipe-v3 run name.

### Sample ID alignment failure in 4B validation

Old validation treated an opening `"` in an auxiliary label as a CSV quote and swallowed later samples into one row. The reader now uses fairseq's QUOTE_NONE, leaving generated TSVs intact. Genuine alignment failures still stop execution. Check IDs for all splits before audio existence. Errors include counts and examples of missing and extra IDs, and the file path.

If validation stopped after generation and the prepared `data-lock.json` exists, explicitly update the code revision and run this read-only check of the saved TSVs. It verifies IDs, dictionaries, units, and CTC while skipping audio existence. Training preflight still checks audio. If an unknown character stopped generation before output, use the previous recovery procedure.

```python
ensure_runtime(require_gpu=False)
run(PYTHON, '-c', '''
import json, sys
from pathlib import Path
from direct_s2st.s2ut.migrate_labels import paper_data_root
from direct_s2st.s2ut.multitask import validate_prepared
root = paper_data_root(Path(sys.argv[1]))
assert (root / 'data-lock.json').is_file(), 'Prepared data-lock is missing. Resume 4b-finish.'
print(json.dumps(validate_prepared(root, check_audio=False), ensure_ascii=False))
''', DATA)
```

After validation succeeds, continue to Cell 5 with existing 4A, HuBERT, and auxiliary labels. To also write the 4B completion receipt, rerun standard `4b-finish`. If alignment still fails, inspect missing/extra/path before training.

Smoke fixes five examples per split and two updates. The standard notebook has no sample limit. Its `PROFILE=pilot` selects CLI settings without trimming the corpus to ten hours. Preparation generates S2UT auxiliary labels and TT2 phonemes and mel.

### CPU preparation, GPU unit extraction, and A100 training

| Stage | Cells | Work |
| --- | --- | --- |
| CPU | 1 → 4A → 4A.5 (standard notebook) | Full import, WAV/SHA verification, phonemes, TT2 mel and validation, and durable English/Japanese WAV ZIPs |
| GPU | 1 → 4B with the same settings | Download and extract completed ZIPs, run HuBERT, and prepare and verify S2UT data |
| Training GPU such as A100 | 1 → 5 with the same settings | Train or resume the selected model |

Switch GPU manually in Colab's runtime settings. Switching clears Python variables, so **rerun Cell 1** with the same `CORPUS_PATH`, `OUTPUT_BASE`, and `EXPERIMENT`. Later cells mount Drive, possibly requiring authentication, fetch the pinned code, rebuild Python/fairseq/eSpeak, and restore environment variables. A usable existing environment is checked and reused. Cells 2 and 3 remain manual entry points.

4B and 5 use the existing 4A Drive data. 4B requires a GPU and checks that all English and Japanese ZIPs are published before fetching the model. If they are incomplete, return to 4A.5. ZIP transfer and extraction, k-means, S2UT setup, and validation can leave GPU utilization low. The `gpu80` preset applies to training. Speed and VRAM on other GPUs remain unverified. TT2/mel-only training can skip 4B. Four-system comparison requires the units.

CPU staging avoids GPU allocation during CPU and Drive work. Runtime rebuilds can offset overall time savings. CPU preparation installs CUDA-enabled PyTorch for compatibility. Billing and quota savings depend on plan and timings.

## Storage and resume after disconnection

Run Cell 1 with the same notebook, experiment, and settings, then continue to the needed stage. A saved training snapshot triggers resume. A corrupt snapshot stops the run. Before the first durable save, no recovery point exists. Run names and local work paths distinguish experiment, use case, and target.

- `data/` holds common manifests and prepared data.
- `checkpoints/` holds SHA-verified training snapshots.
- `runs/` holds each system's audio predictions, JSONL, and metrics.
- `data/results/` holds comparison and acceptance reports.
- `session.log` contains stdout and errors, also streamed to the cell.

Training repeats updates after its latest durable save. Inference verifies and reuses successful pairs by SHA. The comparison cell restores missing local checkpoints from Drive. Use `OVERWRITE_COMPARISON=True` in Cell 6 only to update an existing report.

Resume checks revision, training settings, data, and environment lock. Use a new experiment for changed conditions and preserve past snapshots. Old smoke-data and manual run names stay where they were saved. Continue an old experiment with its matching notebook revision and original lock.

## Runtime and performance

The dedicated environment uses Python 3.10.18 and torch/torchaudio 2.7.1 independently of Colab host Python. Start a fresh GPU runtime if setup was interrupted on 3.13. `requirements/colab310.txt` pins dependencies. `setup.py --inference` installs Cascade, S2T→TTS, and shared evaluation dependencies. CPU preparation also passes `--allow-cpu`.

The standard `PERFORMANCE=gpu80` preset sets TT2 batch 8, vocoder batch 16, workers 4, and S2UT max-tokens 20000. Actual VRAM depends on the run. TT2 exact uses update_freq 128 at batch 8 for effective 1024, and S2UT uses update_freq 4. Smoke uses small batches and workers 0. Additional controls are arguments to `direct_s2st.colab.make_config`. Saved runs keep their original conditions.

Subprocess exit releases model GPU memory. BLASER defaults off. Verify SONAR dependencies separately before enabling it. An explicit request to enable it is honored or fails. Evaluate all systems under the same conditions.

## Verification scope

### Progress logs for long operations

Corpus import, WAV/SHA-256 validation, phonemes, mel/units, model loading, training, inference, evaluation, and checkpoint save, validation, and restore log `[progress]` at start and end and about every ten seconds. Colab shows the messages and appends them to `session.log`. Training and inference subprocess output also streams to the cell and remains in `runs/.../logs/process.log`.

This example shows WAV validation at item 121.

```text
[progress] corpus: validate WAV and SHA256 status=running checked=120/5000 no_advance=2.1s elapsed=30.0s current=pair-121 activity=SHA256: pair-121.wav bytes=1048576
```

- `checked` counts processed items, including reuse, skips, and failures. An unknown total is `?`.
- `elapsed` is stage time in seconds. `no_advance` is seconds since the count last increased.
- `current` identifies the current item, file, or update. `activity` can show hashing, downloading, and bytes read.
- Stages such as model loading without countable items show the stage and elapsed time. Outer and inner stages can log together.
- `running` reports the monitor thread's liveness. Check changing counts or bytes to see advancement. A disconnected or stopped process cannot emit logs.

Progress is flushed to stderr, keeping CLI JSON on stdout and artifact formats intact. See the following section for 4A/4B verification reuse. Already-running processes keep their current code. Restart under the new code to use the logging.

### Parallel preparation and resume in 4A/4B

New mel extraction uses local VM storage. Pack up to 128 items or about 256 MiB per uncompressed ZIP, verify SHA-256 by readback, then publish to Drive. This replaces individual NPY writes for the roughly 140,000 items. ZIPs and immutable receipts under `packs-v1/<first-two-characters>/` are verified and reused per archive. Retain old `features/`, `features-v2/`, and checkpoints. Existing legacy training ZIPs remain compatible, including partially built sets. See [mel ZIP recovery](RESUME.md#drive-inputoutput-errors-with-many-mel-npys).

Set preparation limits and reuse behavior in Cell 1. These settings are independent of Cell 5 training batches.

- `PREP_MAX_WORKERS=32` caps CPU/I/O workers from 1 to 32 and never exceeds CPU count. Start at two or fewer and adjust about every 10 seconds based on throughput, CPU utilization, and free RAM. Reduce workers when Drive becomes the bottleneck.
- `HUBERT_MAX_BATCH=32` caps the adaptive HuBERT batch from 1 to 32. Adjust from free VRAM and measured speed. Retry OOM with a smaller batch. A single-item OOM stops and retains saved progress.
- `RECHECK_AUDIO=False` reuses a successful SHA-256/WAV check if expected SHA, path, size, mtime, and row content match. **Set True to verify all audio bytes again.** Fast reuse cannot detect edits that preserve size and mtime. Treat the corpus as immutable.

4A parallelizes path resolution, WAV/SHA checks, eSpeak phonemes, mel extraction, and WAV header reads. 4B parallelizes audio reads and S2UT conversion and uses one HuBERT model. Padding affects HuBERT convolution normalization, so group only **equal-length waveforms** in a read window of `max(16, HUBERT_MAX_BATCH)`, capped at 32. Different lengths may run singly. GPU utilization, maximum speed, and speedup depend on the data. Precision and full audio content are retained.

Drive stores preparation artifacts within the experiment.

- WAV validation under `data/.prep-checkpoints/corpus/`
- Phonemes under `data/translatotron2/phonemes/.checkpoints/`
- Mel under `data/translatotron2/.prep-checkpoints/mel/`, with new ZIPs in `packs-v1/`. Training ZIPs need separate space. Keep old NPYs.
- Units under `data/s2ut/units/.checkpoints/`, with existing `.units` files retained.

For non-mel stages, atomically publish small completion JSON after 64 items or roughly 10 seconds at the next result. Normal exceptions and interruptions save collected results. Forced VM termination can lose the unpublished chunk and small in-flight window. If final TSV/ZIP aggregation is interrupted, rebuild it from saved extracts. Stop on corrupt published data. Model or input changes require verification before reuse.

Mel publishes each ZIP at the 128-item/256 MiB limit, or after about 60 seconds at the next result when at least 32 items are ready. Graceful interruption also saves a small remainder. Forced termination repeats unpublished ZIPs and in-flight work. `persisted` counts Drive items, and `pending_local` counts local-only items. `S2ST_MEL_LOCAL_WORK` overrides the local temp parent. Use local VM storage outside Drive and the corpus.

After interruption, use the **same experiment, revision, and settings in Cells 1 → 4A or 4B**. The cell passes `--resume` and rebuilds the environment on a new VM or GPU. Already-running old code is unaffected and unsaved work cannot be recovered. Use one preparation process per output.

Logs show `checked`, `remaining` when the total is known, CPU workers, speed, free RAM, actual HuBERT batch, next cap, free VRAM, and final save/reuse counts. CLI equivalents are `S2ST_PREP_WORKERS`, `S2ST_HUBERT_BATCH_MAX`, and `S2ST_PREP_RECHECK=1`. Set `S2ST_PREP_ADAPTIVE=0` to fix CPU workers at the cap.

### Early stopping for unit and mel vocoders

Standard Cells 1 → 5 default to `VOCODER_MONITOR=True`, `VOCODER_MONITOR_INTERVAL=10`, `VOCODER_MONITOR_PATIENCE=5`, and `VOCODER_MONITOR_MIN_RELATIVE_IMPROVEMENT=0.01`. Mel monitors dev mel reconstruction L1. Unit also monitors duration MSE on log1p run lengths. At least 1% improvement against any metric's anchor resets patience. Stop after five evaluations without significant improvement. GAN discriminator, adversarial, feature, and total losses can fluctuate and are retained in train logs only.

Through `VOCODER_MONITOR_START_UPDATES=400`, collect minimum dev values as anchors and defer stopping. This guard controls stopping only. Evaluate every dev utterance on a fixed center segment, 32 frames by default, and average over utterances. Unit duration uses original units for the full utterance. Evaluation performs no optimizer updates and excludes test. Restore RNG, buffers, and train/eval modes afterward. These losses do not directly measure naturalness, translation accuracy, or MOS.

Resume uses `generator.pt` with both models and optimizers and RNG. It includes monitoring state and dev input identity so anchors, patience, and history survive disconnection. Save a separate full best checkpoint by dev mel L1 under `checkpoints/checkpoint_best.pt` and verify its SHA for comparison synthesis. Old runs without a best use last. Include JSON, CSV, and SVG in verified snapshots and publish them to the Drive dashboard.

For an existing unmonitored vocoder run, a compatible launcher applies only the reviewed monitoring overlay and keeps saved config and optimizer. History begins at resume. Reject unknown old implementations or changes to the policy, guard, or dev inputs after adoption. Saved monitoring persists even if the setting is off. Set `TRAIN_CODE_REVISION` to a compatible full SHA if needed for initial launcher selection. Resume through standard Cells 1 → 5. `VOCODER_TOTAL_UPDATES=10000` is an independent absolute cap.

#### ExistingOutputError after a VRAM probe

Through `93b04bf`, a vocoder could treat the launcher's `batch-calibration.json` as existing training output. The fix checks the destination, allowed prep filenames, and SHA-256 using a private standard Cell 5 preflight receipt. Generator files, checkpoints, monitoring history, and unknown files remain protected. Use the updated launcher and rerun Cells 1 → 5 with the same experiment. Reuse calibration and input ZIP/validation caches. Auto-resume preserves incomplete local output under `.interrupted-...`. Keep Drive data and checkpoints.

### Resident training and VRAM calibration in Cell 5

Standard `TRAIN_OPTIMIZE=True` keeps one training process resident on one GPU for the session. `CHUNK_UPDATES` and the 100-update restarts apply to legacy execution. The `make_config` API defaults to `optimize=False` for compatibility.

`TRAIN_CALIBRATE_BATCH=True` runs separate-process trials before training TT2, S2UT, and mel or unit vocoders. Record selection, peak VRAM, timing, and GPU on Drive. Discard probe weights and start training at update 0. TT2 chooses physical batch with effective 1024, and GAN chooses physical batch. S2UT exact only checks fit at max_tokens 20000 and update_freq 4. If it fails, use a larger GPU or explicitly start a separate practical run. Only S2UT practical selects a split cap. Resume uses the recorded settings. Defaults are `TRAIN_CALIBRATION_MAX_BATCH=1024`, `TRAIN_VRAM_RESERVE_RATIO=0.10`, and `TRAIN_CALIBRATION_STEPS=3`. MAX_BATCH does not affect S2UT exact. `TRAIN_CALIBRATION_OBJECTIVE=throughput` selects by measured work per second among safe candidates. `capacity` selects the largest fit. Both preserve S2UT exact batching. Logs include utilization samples, VRAM, throughput, and TT2 streaming wait. See [paper recipes](PAPER_RECIPES.md) for GPU changes and margins.

Probes use `python -m direct_s2st.batch_probe`. Every trainer uses the same completion signal and requires the requested count of finite positive measurements. Errors and insufficient data stop separately from OOM. Incomplete measurements are not marked reusable. Existing probe results and GPU changes use the same validation. S2UT throughput divides all accumulated samples by the complete optimizer update time. Microbatch reports the largest number processed simultaneously. Repeat interrupted candidates and reuse earlier published ones.

To upgrade a legacy probe that stopped at `ProbeComplete`, use `scripts/colab/use_paper_recipe.py --separate-code-fix-runs --overwrite` to back up and update the revision pin and add a run suffix such as `-probe-fix-v1`. Keep the experiment name and prepared data. Old configs and checkpoints remain, and the fresh run uses new optimizer state. If training never began, start at calibration. Set the same suffixed name after switching runtimes before running Cell 5. Calibration retains the selected batch and discards trial weights. TT2/vocoder probe logs count the final update. CPU tests exercise the actual `-m` entry point with CUDA/trainer mocks. A real small CPU TT2 model also verifies three backward/Adam updates and no trial checkpoint publication. GPU fit needs a separate test.

- `TRAIN_MAX_WORKERS=8` caps parallel loading. TT2/vocoder adjust about every 10 seconds from CPU load, RAM, and throughput. S2UT changes only at fairseq epoch boundaries. Zero is sequential.
- `TRAIN_CACHE_GB=8` limits the legacy local LRU cache. Drive-safe staging of complete inputs uses separate disk capacity. Insufficient local space stops before training, preserving verified local reads.
- `TRAIN_SAVE_INTERVAL=1` sets the standard optimizer-update save interval. Save also at the final update or time cap. Freeze the local copy synchronously, then copy to Drive and verify SHA-256 in a thread, one at a time. At most two copies may remain unverified. Slow Drive can block.
- `TRAIN_BACKUP_MIN_SECONDS=600` is the minimum Drive transfer interval for resident training. Local saves every 50 updates are retained. Initial, final, and time-budget saves transfer regardless of interval. Forced disconnection resumes from `durable_updates` and recomputes unverified updates.
- `TRAIN_BACKUP_MAX_GB=1024` caps Drive backups per run at 1024 GiB (1 TiB), including old snapshots and partial copies. At the limit, stop and retain recovery points. The Drive account must also have enough free space. Explicit settings are retained.
- `TT2_MONITOR=True` uses v3 for four metrics. Evaluate full dev every `TT2_MONITOR_INTERVAL=10` updates. At least 1% improvement on any anchor resets the shared counter. Stop after `TT2_MONITOR_PATIENCE=5` insufficient evaluations. Save best by minimum dev total loss. The default 400-update warmup gathers anchors before stopping. V3 caps at 1000 updates and retains LR, batch, and loss definitions and weights.
- TT2 writes `training-monitor.json` with full history, policy, per-metric anchors and best values, shared counter, and transitions, plus CSV/SVG. Verified snapshots contain them. `checkpoints/checkpoint_best.pt` includes model and optimizer at minimum dev total loss. `checkpoint_last.pt` is for resume. Initial adoption on a legacy checkpoint starts its history at that update. Explicit v1/v2→v3 uses saved dev minima and applies to future evaluations. See integrated Cell 5 resume above.
- After publishing a verified snapshot, copy monitor SVG, CSV, and JSON to `PERSISTENT/checkpoints/<saved-run-name>/training-monitor.svg` and matching extensions. The JSON covers the latest durable save. The chart and CSV cover the latest evaluation. A dashboard I/O failure preserves the verified snapshot, which can restore monitor data.
- TT2 inference defaults to verified dev-best, using last for older runs or before the first evaluation. Override with `--checkpoint-selection last/best`. Missing or corrupt recorded best stops. This implementation selects by loss. Free-generated audio and ASR-BLEU need separate evaluation. Including best increases backup size. Old recovery points are preserved.
- `TRAIN_ADAPTIVE_BATCH=False` can be enabled for TT2/S2UT. It adjusts splitting while keeping logical batch, order, and update count. TT2 BatchNorm and dropout differ, so treat it as a separate condition. Vocoder uses both optimizers together and does not split or retry OOM.
- `TRAIN_PRECISION='default'` keeps TT2/vocoder FP32 and recipe FP16 for S2UT. FP32 and BF16 can be selected explicitly. A GPU without BF16 support stops when BF16 is required.

Only adaptive mode catches CUDA OOM before an optimizer update, restores gradients, RNG, and model buffers, and retries the same samples in smaller groups. Single-item OOM, optimizer OOM, and device assertions stop. Online adaptation is practical-only and mutually exclusive with pre-training probing. Explicit adaptation fixes calibration after a short trial, shrinks on later OOM, and recalibrates on GPU change. Resume GAN OOM from the saved checkpoint.

`[training-performance]` reports cumulative timings, frames or units per second, CPU, RAM, and VRAM. CUDA interval timing includes waits and does not directly measure utilization. `[recovery] learned_at_least` includes local saves, while `durable_updates` reports copies verified through the Drive mount. Drive server synchronization is independent. Recompute unsaved or unverified updates after disconnection. Check time at update boundaries. Waiting for saves can exceed the budget. After 30 seconds of grace, request termination and use the last verified save.

Use a separate run for changed settings and preserve older configs and pins. Same-settings resume uses Cells 1 → 5. Real GPU speed, BF16 quality, and Drive interruption recovery remain unverified.

S2UT Cell 5 verifies and restores snapshots before fetching all audio ZIPs, checking recipe and checkpoint identities. The early check omits only source WAV existence. Verify all inputs and SHA after retrieval before use. Mismatches report fields such as `recipe.warmup_updates` or `command[index]`. A reviewed execution copy that changes only the Python file location can retain recipe-lock and checkpoint fingerprint. Reuse the same experiment and 4A/4B.

Auto-resume requires a **verified snapshot**. If the initial copy stopped before any publication, preserve the partial copy, report `starting at update 0`, and archive unpublished local files under `.interrupted-<ID>`. If published snapshots are all corrupt, stop. Explicit CLI `--resume` also stops when no recovery point exists.

Verify all vocoder train audio and units before starting the training time budget, using 4A's adaptive workers. Save per-sample evidence to `data/.prep-checkpoints/vocoder-inputs/`. Resume checks expected SHA, path, size, and mtime. `RECHECK_AUDIO=True` or CLI `S2ST_PREP_RECHECK=1` rehashes audio. Fast reuse assumes immutable source files. The training proof is a temporary session file and excluded from snapshots. Recheck size and mtime on sample reads and stop on post-validation changes.

See [phoneme migration](RESUME.md#migrate-phoneme-generation-and-dictionaries) for dictionaries and existing labels.

### Drive file counts and training I/O

The standard notebook sets `S2ST_DRIVE_SAFE=1`. The `direct_s2st.colab` CLI also defaults on. The Python API retains compatibility defaults. Preserve corpus originals and old mel and unit outputs.

- New 4B outputs store original and reduced units inline in JSONL, with progress in existing unit checkpoints. Old loose `.units` files remain readable. Migration writes separate `manifest.inline.shard-...jsonl` files and reuses matching model/input checkpoints. S2UT and unit vocoding accept both formats.
- Safe-mode checkpoint JSON uses 256 buckets and can read old top-level chunks.
- 4A packs needed mel input WAVs into immutable 128-item ZIPs. Standard CPU 4A.5 completes English and Japanese ZIPs across splits under `data/.drive-audio-packs/`. It needs disk for ZIPs in progress and does not rebuild phonemes, mel, or units or extract all audio. Initial staging reads and hashes individual WAVs. Fresh VMs download and extract completed ZIPs. 4B consumes published ZIPs. The format keeps names and identities from `55a5a90` and reuses ZIPs partially built in legacy 4B. If originals change, create a new common manifest and verify inputs.
- Publish ZIPs in plan, copy SHA-256 verification, then completion receipt order. Forced VM deletion repeats only unpublished work. A copied ZIP without a receipt is recoverable if plan and SHA match.
- TT2 Cell 5 defaults to `TRAIN_STREAM_INPUTS=True`, starting with metadata. Fetch WAV and mel ZIPs in training order and prefetch during training, with two transfers and 16 queued jobs at most. Keep sample order and wait for required ZIPs. Retain effective batch 1024, optimizer, and calibrated batch. The initial mel identity uses prepared `data-lock.json` SHA-256. Verify ZIP contents and individual WAV SHAs on retrieval before training. Reuse verified local copies on the same VM. A fresh VM fetches from the resume position. Calibration fetches only required train probe ZIPs.
- An old monolithic mel ZIP must download completely before first use and stays intact. For old data without saved mel SHA-256, use `TRAIN_STREAM_INPUTS=False`. S2UT and vocoders still stage all data upfront for input validation. TT2 with streaming off also retrieves all required WAVs and whole mel ZIPs before training. Local paths are used for reads while original paths, SHA values, and identities remain. Reuse local partial and complete copies on the same VM, and restore from Drive after VM loss. Local staging capacity is separate from `TRAIN_CACHE_GB`. Keep disk space for one pass because used ZIPs remain local. Disk or network failures stop before unverified input is used.
- `[stream-download] ready=...` counts retrieved and verified ZIPs. Streaming mode is outside config/optimizer identity, so the same code and training settings can resume with it on or off. Saved revision and environment migration are separate.
- Save each Drive snapshot as one ZIP with SHA-256 readback. Older snapshots remain restorable. Storage errors or caps preserve the previous recovery point.

Drive API, bandwidth, and storage limits still apply during original reads, ZIP transfer, and checkpoint publication. Free space uses mount reporting and may differ from server storage or sync status. After quota or capacity errors, keep existing data and resume with the same settings once resolved. Performance and interruption recovery on real GPU and Drive remain unverified.

### Fairseq copy failures

Copy symlinks as links so dangling `kaldi_self_train/st/utils` and `steps` targets do not fail. Publish only after copying into a temporary directory. Preserve copies without completion markers at `/content/s2st-runtime/fairseq.incomplete-<ID>`. Keep interrupted temporary directories for diagnosis and make a new copy on retry.

With the fixed setup, rerun Cell 3 on the same VM and retain corpus, checkpoints, and installed Python. Run setup while training is stopped. Retrieval checks out the saved Git revision. For new code, use a new experiment and VM with the updated checkout, preserving old revision and environment locks.

### Listen to three examples after training (standard Cells 6 and 7)

After training the required models, reopen the notebook. In Cell 1 set `RUN_TRAINING=False`, `RUN_PREVIEW=True`, `RUN_BENCHMARK=False`, and `PREVIEW_SAMPLES=3` (allowed range 1–10). Run **Cells 1 → 6 → 7**. For an experiment pinned to an old launcher, set `TRAIN_CODE_REVISION` to a full SHA that supports preview. Keep the saved training revision and experiment name.

Cells 6 and 7 apply the `TT2_VOCODER_MODE` and `S2UT_VOCODER_MODE` from Cell 1.

| Setting | Preview synthesis | Trained vocoder snapshot |
|---|---|---|
| `TT2_VOCODER_MODE='griffin_lim'` | Griffin–Lim with saved TT2 mel specification, 32 iterations, seed 1 | None |
| `TT2_VOCODER_MODE='hifigan'` | Verified best trained mel HiFi-GAN | Mel |
| `S2UT_VOCODER_MODE='paper'` | Official SHA-256-pinned LJSpeech unit HiFi-GAN | None |
| `S2UT_VOCODER_MODE='trained'` | Verified best trained unit HiFi-GAN | Unit |

Use `griffin_lim` and `paper` to diagnose synthesis with only trained TT2 and S2UT runs. On first use, fetch the official unit weights and config, verify SHA-256, and cache them. Also verify saved HuBERT revision, layer, cluster count, and k-means SHA. The public LJSpeech vocoder uses a different training source from Fisher or the target corpus. Its voice can differ and should be considered when assessing speaker preservation.

Cell 6 prefers `last-training-run-{tt2,s2ut,unit,mel}.json` for required runs. Without pointers, automatically select only a unique saved trained run. Resolve multiple choices with `PREVIEW_RUN_NAMES`. For example, restore the actual TT2 `-probe-fix-v1` run. Verify snapshot ZIPs, files, and config before restoring to a dedicated local area. Copy only selected Japanese and English dev WAVs with verification. Checkpoint ZIPs still need downloading and can be several GB. Drive limits apply.

Cell 7 uses a GPU. It restores each run's saved code location and dependencies in turn and runs the current pinned inference code on the selected dev samples. Same-VM reruns restore the full package manifest as well, including removing extra Unit/Mel dependencies that cause a TT2 environment SHA mismatch. Environment identity excludes GPU model. Preview uses saved weights and prepared data and skips training, optimizer updates, 4A/4B, ASR, and scoring.

Compare Japanese and English originals, reconstruction from reference mel or reduced units, and TT2/S2UT translations. **Unit reconstruction uses predicted duration.** Display labels and `listening.json` record the actual synthesis method and official weight identity. A different method or weights produce a separate preview identity. If settings change after Cell 6, rerun it before Cell 7. Legacy preview plans retain their trained HiFi-GAN interpretation. Set a compatible `TRAIN_CODE_REVISION` for an old launcher. Formal evaluation, model config, and checkpoints are preserved. The outputs allow separate inspection of reconstruction and translation. Generation failures are shown as errors.

The launcher and inference subprocesses use normal INFO logging and suppress fairseq-triggered Numba DEBUG dumps. Progress, INFO, warnings, errors, exceptions, and result JSON still appear. JIT and model computation remain enabled.

Preview artifacts go to `<experiment>/preview/<identity>/listening.json` and `audio/`. Checkpoints, corpus, and formal test predictions and results are read-only. After disconnection, repeat Cells 1 → 6 → 7 with the same settings. Local copies are temporary and Drive snapshots and previews are durable. For formal evaluation, set `RUN_PREVIEW=False` and `RUN_BENCHMARK=True` and start at Cell 6. Cell 8 evaluates formal test predictions.

CPU tests cover four vocoder combinations, skipping irrelevant snapshots, separate identities for settings changes, legacy plan compatibility, SHA and unit-ID mismatch rejection, failure retention, and runtime dispatch. Verified actual official unit weight/config SHAs and synthesized three reference and three predicted unit sequences on Windows/Python 3.13 using the isolated pinned fairseq generator and duration predictor. Full fairseq import and GPU E2E on Colab/Python 3.10 and audio quality remain unverified. Preview uses existing saved TT2/S2UT dependencies.

S2UT inference uses one item at a time and explicitly sets `--required-batch-size-multiple 1`. The pinned fairseq default of 8 can otherwise drop the last item.

### Formal S2UT-only test evaluation and audio storage (standard Cells 6–9)

To evaluate S2UT independently, set `RUN_TRAINING=False`, `RUN_PREVIEW=False`, `RUN_BENCHMARK=True`, `BENCHMARK_TARGET='s2ut'`, `S2UT_VOCODER_MODE='paper'`, `DATA_LIMIT=None`, and `CONFIRM_FULL_DATA=True`. An empty `BENCHMARK_RUN_NAME` prefers `last-training-run-s2ut.json` and then a unique saved run. Specify the exact name if several exist. `BENCHMARK_TARGET` selects evaluation independently of `TRAIN_TARGET`. If needed, set `TRAIN_CODE_REVISION` to a compatible full launcher SHA. Training revision, dependency lock, and snapshots are preserved.

1. **Cell 6** verifies the saved S2UT snapshot, original environment identity, and prepared metadata. Check `checkpoint_best.pt` against the minimum dev unit loss and SHA and display its update. The best update can precede final update 400000. Match every test ID and English reference against checkpoint-bound target letter labels. Fetch only test WAVs in both languages, reusing audio ZIPs and reading originals only to create missing test ZIPs in a separate location.
2. **Cell 7** generates all test units with pinned fairseq and beam 10. Use official SHA-pinned HuBERT L6/KM100 LJSpeech unit HiFi-GAN with duration prediction. This path uses only trained S2UT and the official vocoder, with no optimizer updates.
3. **Cell 8** restores the saved dependencies and uses pinned `openai/whisper-large-v3-turbo` for English ASR. Apply existing English normalization and sacreBLEU. **Every test example is included**, with generation or ASR failures scored as empty hypotheses. Save failure counts and ASR success rate. This standalone report excludes BLASER and speaker similarity and is separate from four-system acceptance.

Persist verified immutable ZIPs of completed unit shards, 32-item WAV groups, and ASR journals in groups of 32. After disconnection, run **Cells 1 → 6 → 7 → 8** with the same settings. Verify and restore receipts and ZIPs on a fresh VM. A group interrupted before publication can be repeated. Drive limits and disconnects still apply. Keep published ZIPs and atomically update the completion index as new groups finish.

Evaluation outputs go to `<experiment>/benchmarks/s2ut/<evaluation-id>/`.

- `plan.json` records the run, checkpoint/environment/metadata SHAs, ASR revision, and test IDs.
- `audio/part-....zip` stores groups of 32 WAVs, with ZIP and WAV SHAs in adjacent JSON receipts.
- `audio-index.json` and `audio-index.csv` map pair IDs to ZIP names and members, English references, and status.
- `predictions.jsonl` stores all generations and the original local paths. Playback uses ZIPs and the index.
- `results/metrics.json` and `report.md` report standalone ASR-BLEU.
- `results/per_sample.jsonl` and `.csv` store references, ASR text, sentence BLEU, and failure reasons.

For playback, run **Cells 1 → 9 on CPU**. Set a pair ID in `BENCHMARK_LISTEN_ID`, or leave it empty and set a zero-based `BENCHMARK_LISTEN_INDEX`. The cell verifies and extracts the chosen WAV and shows Japanese original, English reference audio, S2UT output, and ASR text. Playback uses saved output without a model or dedicated venv. Originals are read from the corpus. A missing original is reported and saved translation audio still plays. You can also download and extract WAV ZIPs locally. Playback preserves evaluation and preview artifacts.

CPU tests cover persistence, SHA rejection, resume, the denominator, and standard-cell branching. Full-test real Colab GPU E2E, quality, and runtime remain unverified. Use `BENCHMARK_TARGET='all'` for the four-system comparison. The S2UT single-item batch setting applies to preview and full inference and leaves training batches unchanged. Incomplete old shards are kept and new generation uses a new cache identity. If an old version reports `preview generated unexpected or missing sample IDs`, update `TRAIN_CODE_REVISION` and run standard Cells 1 → 6 → 7 using existing checkpoints and prepared data.

### Diagnose saved Cascade results by KFTT and JESC (Cell 8A)

Cell 8A is a diagnostic-only analysis that recomputes Cascade metrics separately
for KFTT/JESC from saved Cell 8 results. It performs no inference and does not
modify the official benchmark result.

The Cell 8A KFTT-only diagnostic shows Japanese ASR CER, MT corpus BLEU, and final ASR-BLEU for All, KFTT, and JESC. Group by manifest `corpus` and join by `pair_id`. Verify that All ASR-BLEU matches the official result and keep its failure handling. Missing or duplicate rows and mismatched references are integrity errors. CER divides total character edit distance by total reference characters after NFKC and whitespace removal, retaining punctuation. Saved MT text contributes to MT BLEU even when TTS failed.

`CASCADE_DIAGNOSTIC_TARGET='auto'` selects Cascade from the four-system result when `BENCHMARK_TARGET='all'`. Otherwise, including default `s2ut`, it uses standalone Cascade `latest.json`. Set `CASCADE_DIAGNOSTIC_EVALUATION_ROOT` to inspect another saved result. Use CPU and prefer existing `sacrebleu==2.5.1`. Try `PYTHON`, local venv, and host Python with a real import and short CPU BLEU calculation. Check PyYAML for `all`. Report missing dependencies and Python paths if unavailable. With `CASCADE_DIAGNOSTIC_INSTALL_CPU_DEPS=True`, install only pinned wheels from `requirements/diagnostic-cpu.txt` with `pip --isolated --no-deps --target` into `/content/s2st-diagnostic-deps/`, then verify and reuse. NumPy 2.1.3, with a Python 3.13 wheel, is used only in that directory. SacreBLEU, normalization, and tokenizer match formal evaluation. The All score check is required. The training venv, host packages, and Drive stay read-only. If installation is off and dependencies are missing, stop.

The cell reads stored text and score records. It does not load models or open saved WAVs and evaluation ZIPs. `CASCADE_DIAGNOSTIC_FETCH_CODE=True` permits retrieving missing diagnostic code into a local worktree outside the experiment and corpus. Training checkout and pin are preserved. Show low-scoring KFTT examples in a table and keep all rows in `cascade_corpus_diagnostic['kftt_samples']`. Differences between subsets alone do not establish a corpus quality cause. Relay subprocess progress and error details to the cell. Failure leaves saved results untouched.

### Compare translation from gold and ASR Japanese (Cell 8B, GPU)

With a saved formal Cascade test evaluation, run **Cells 1 → 8B** on GPU with the same experiment settings. It independently validates saved All ASR-BLEU, IDs, references, and SHAs. Reuse existing ASR and TTS results and skip Cells 3 and 6–8. Keep `TRAIN_CODE_REVISION`. If needed, fetch diagnostic code to a separate local worktree. Earlier cells and formal results keep their existing behavior and contents.

Use the same NLLB from the saved plan to translate `ja_text` (gold Japanese) and `asr_ja_text` (saved Japanese ASR) for each test item. English references are scoring-only. For All, KFTT, and JESC, display saved MT BLEU, gold-source MT BLEU, ASR-source retranslating BLEU, their difference, matches to saved MT, and failure counts. These are corpus BLEU scores over the selected set, with failed translations as empty hypotheses. Group by manifest `corpus` and verify joins and resume by `pair_id`.

`MT_DIAGNOSTIC_LIMIT=0` uses all test items. A positive value selects a deterministic pair-ID hash subset. `MT_DIAGNOSTIC_TARGET='auto'` selects as in 8A. Change `MT_DIAGNOSTIC_RUN_LABEL` to start a separate run. Write `metrics.json`, `per_sample.csv`, and `report.md` under `<experiment>/diagnostics/cascade-gold-mt/<ID>/`. Record NLLB language tokens, dtype, actual generation settings, dependency versions, and GPU. Resume requires matching model, input, environment, and code. Publish JSON every eight translation branches and flush pending items on graceful interruption. Forced termination can repeat up to seven branches. Completed diagnostics can aggregate without loading NLLB. CUDA/OOM stops execution.

Check imports for pinned NLLB dependencies and a real CUDA operation before loading. Prefer an existing runtime. With `MT_DIAGNOSTIC_INSTALL_DEPS=True`, install missing dependencies to `/content/s2st-mt-diagnostic-runtime/`, using Python 3.10.18, torch 2.7.1+cu126, Transformers 4.57.3, and the other pins in `requirements/diagnostic-mt.txt`. Preserve training and host packages. New NLLB weights and cache use `/content/s2st-mt-diagnostic-cache/`. A missing GPU stops before installation.

Use the original adapter's `src_lang=jpn_Jpan` and English `forced_bos_token_id`, model-default maximum generation length 200 and beam, sampling, and repetition settings, FP32, and one sentence per call. Record EOS and reaching the length limit as diagnostics. If outputs differ from saved MT, inspect environment and generation records. A low score on gold Japanese alone cannot establish broken bitext. Treat gold-source scores as diagnostic. See the [pinned official NLLB config](https://huggingface.co/facebook/nllb-200-distilled-600M/blob/f8d333a098d19b4fd9a8b18f94170487ad3f821d/generation_config.json) and [Transformers 4.57.3 NLLB docs](https://huggingface.co/docs/transformers/v4.57.3/en/model_doc/nllb).

### Formal Cascade-only test evaluation (standard Cells 6–9)

Set `RUN_TRAINING=False`, `RUN_PREVIEW=False`, `RUN_BENCHMARK=True`, `BENCHMARK_TARGET='cascade'`, `CONFIRM_FULL_DATA=True`, and `DATA_LIMIT=None` in Cell 1. Set `TRAIN_CODE_REVISION` to the supported full SHA and run Cells 1 → 6 → 7 → 8. Cascade uses its own pinned model config, independent of training-target and direct-model vocoder settings.

- **Cell 6** verifies all common test items and the dataset lock, then retrieves and checks Japanese and English test WAVs from existing ZIPs. Uses no training checkpoints and skips 4A/4B.
- **Cell 7** runs Whisper Japanese ASR, NLLB JA→EN MT, and Qwen English TTS from `configs/cascade/default.yaml`. Run stages separately in sequence. Pass ASR output to MT and MT output to TTS. References never enter inference. The Qwen adapter requires a BF16-capable CUDA GPU.
- **Cell 8** uses the same pinned English Whisper, normalization, and sacreBLEU as standalone S2UT. Include every test item and use empty hypotheses for failures. BLASER and speaker similarity are excluded.
- **Cell 9** plays saved WAVs on CPU with `BENCHMARK_TARGET='cascade'` and the chosen ID or index.

Save the separate evaluation receipt `runtime.json` and actual dependency manifest `environment.json`. After disconnection, use Cells 1 → 6 → 7 → 8 to restore dependencies and verify and reuse 32-item ASR/MT journal ZIPs, TTS WAV ZIPs, and evaluation ASR journals. Unpublished in-flight groups may repeat. A model, reference, or code change uses a different evaluation ID. CUDA OOM or device assertions stop the process. Sample-specific failures retain cause and intermediate text. Fully reusable stages skip model loading. References are used only for evaluation.

Store results in `<experiment>/benchmarks/cascade/<evaluation-id>/`. The standalone report and metrics go to `results/report.md` and `results/metrics.json`. `results/per_sample.csv` includes English reference, generated English ASR, sentence BLEU, `asr_ja_text`, and `mt_en_text`. `predictions.jsonl` records stage timings and failure reasons. `audio/` holds WAV ZIPs and `audio-index.csv/json` maps IDs to members. S2UT results and four-system acceptance are independent. Use `BENCHMARK_TARGET='all'` for four systems.

Use the pinned dependencies in `requirements/colab-inference.txt`, including `qwen-tts==0.1.1` and `transformers==4.57.3`. Check imports before fetching models. CPU and mock tests cover notebook branching, leakage prevention, ZIP restore, stage reuse, failure denominators, receipt checks, and preservation of S2UT output. Real Colab/GPU and full Qwen generation remain unverified.

#### Qwen tokenizer retrieval and resume with saved ASR/MT

The loader-only overlay below documents the legacy recovery path. The current official sampling and EOS policy starts a separate evaluation sharing only completed ASR/MT. Use standard Cells 1 → 6 → 7 → 8 and retain old plans and receipts. The policy explicitly sets both sampling flags True, a fixed seed per sentence, a 1501-token cap corresponding to 120 seconds, and a 600-second cooperative guard. Unverified EOS fails the sample. Real Qwen GPU behavior remains to be verified. See the [Qwen TTS audit](CASCADE_TTS_AUDIT.md).

`qwen-tts==0.1.1` can fail to forward the requested revision to internal tokenizer `cached_file` or processor calls. The main model can come from `f00cf13...` while `speech_tokenizer` resolves to another snapshot missing `preprocessor_config.json`. This retrieval error is independent of the FlashAttention warning.

Fetch and verify the **complete snapshot at its pinned SHA**, including root processor, tokenizer, weights, and required speech_tokenizer files and JSON. Pass the local directory to the SDK and reuse existing Hub blobs. Cell 7 checks the GPU, verifies saved ASR/MT read-only, resolves the pinned snapshot, loads Qwen, synthesizes `This is a synthesis test.`, then generates pending TTS. Unsupported CUDA/BF16 stops before loading. Log GPU, capability, bfloat16, and SDPA. Preflight saves and reads back WAV and verifies mono, finite values, non-silence, and duration. On failure, no formal test audio is generated. Store preflight records and WAV under `preflight/` outside formal predictions, index, and BLEU denominator. Restore RNG. The loader-only fix retained dependency versions, model SHA, voice, language, BF16/SDPA, and formal decoding. The new sampling settings use a separate evaluation.

For an evaluation that stopped **before the first TTS load** at `f9f1520047bca2c0e977b581cab70168892091e2`, set `TRAIN_CODE_REVISION` to the compatible fix and run standard Cells 1 → 6 → 7 → 8. Cell 6 can reuse the plan if model config, all test IDs, and manifest SHA match. Verify reviewed Git differences for loader, preflight, launcher, docs, and tests, plus unchanged worker ASR/MT/identity ASTs. Initial loader-fix adoption requires no existing TTS WAVs and records `execution-overlays/<fix-revision>.json`. Keep the original plan, journals, completion pointers, and runtime SHA. Restore dependencies from the original editable build path and use the fix via separate PYTHONPATH.

Completed text stages report `ASR: reuse 1196/1196` and `MT: reuse 1196/1196` and skip Whisper/NLLB loading. Verify pointers, identities, journals, ZIPs, SHAs, IDs, successful status, and input text and timing associations. Stop on missing or corrupt artifacts. A new run can generate ASR/MT normally. An incomplete stored text stage requires investigation before rerunning. Resume with the same fix reuses WAVs generated after its adoption.

If adoption of `c741fca0c8408d67750a480cb989370fbcab9071` is already recorded, verify original plan SHA, adoption receipt, Git ancestry, complete-snapshot processing, and unchanged ASTs for model load and synthesis arguments. Then migrate preflight and resume while retaining generated WAVs. Record parent receipt SHA in the new overlay and retain old receipts and WAVs. Final metrics and reports record both original and execution revisions.

Stop if legacy WAVs lack a verified adoption receipt, immutable ZIPs are unpublished, plan or dependency receipts mismatch, or unreviewed ASR/MT/model/dependency changes exist. Keep tokenizer files, `environment.json`, plan revisions, corpus, and saved text results intact. CPU tests include local config loading with the real Transformers Encodec feature extractor. Real Qwen synthesis remains unverified.

TTS first checks all saved group journals, identities, WAV SHAs, and formats. Progress includes `TTS: saved=... remaining=...` and `[cascade-tts] reused=... computed=... remaining=...`. Use `mt_en_text` as input and reuse successes. The same VM can reuse successful unpublished WAVs. A lost VM resumes from published 32-item groups and can repeat at most 31 items in the last group. For example, 480 of 500 processed items are durable if 480 were published. When all successes are saved, skip Qwen load and preflight. Cell 8 retains its evaluation definition. Batching and FlashAttention remain outside this change.

The opt-in GPU test is `tests/gpu/test_cascade_qwen_preflight.py`. `S2ST_RUN_QWEN_GPU_TEST=1` on the pinned Colab L4 setup loads actual Qwen and its audio tokenizer and generates a WAV. CPU success verifies only the CPU scope. To check operations, use standard Cells 1 → 6 if needed → 7 on the same experiment, inspect formal WAVs after preflight, then interrupt and rerun Cell 7 and verify saved counts and reuse logs.

### Three-condition external diagnostic (Cell 8C, GPU, resumable)

**Cell 8C is an external diagnostic and is not part of the formal benchmark.
It does not alter official KFTT/JESC results.** Run Cell 8C with the experiment settings from Cell 1. It is independent of 8A/8B variables and the training environment. Recheck the saved Cascade All score, manifest, and plan and model SHAs, then reuse its ASR, MT, TTS, and English evaluation ASR settings.

`FLORES_FLEURS_LIMIT=300` selects 300 sentences. Only 0 selects all matches. Uniquely match official FLEURS `ja_jp` validation/test text to official FLORES+ `jpn_Jpan` dev/devtest text with NFKC and Unicode whitespace normalization only. Find English by the same FLORES+ `(split,id)`. Use exact text matching and exclude ambiguous cases. Stop if there are fewer than 300 unique matches. Select one duplicate recording deterministically by content and ID/path SHA. Select sentences by canonical ID SHA. Record unmatched, ambiguous, and duplicate counts.

The dataset revisions and files were verified against official APIs and Parquet schemas on 2026-10-24.

| Dataset | Revision | Files used |
|---|---|---|
| `google/fleurs` | `70bb2e84b976b7e960aa89f1c648e09c59f894dd` | `parquet-data/ja_jp/{validation,test}-00000-of-00001.parquet` |
| `openlanguagedata/flores_plus` | `e707e62e762a9e2faca3933aa315f4665cc3cd8f` | `{dev,devtest}/{jpn_Jpan,eng_Latn}.jsonl` |

Accept the FLORES+ terms and authenticate using Hugging Face login in Colab or `HF_TOKEN`. Tokens stay out of output and artifacts. Failed access stops the run. Read text columns of official FLEURS Parquet first using range reads. Save only selected WAVs to Drive. Parquet audio is read by row group, so network blocks can include unselected recordings. Keep full Parquet off Drive and exclude external data from training. See the [official FLORES+ page](https://huggingface.co/datasets/openlanguagedata/flores_plus).

Compare gold Japanese text, natural FLEURS speech, and Japanese synthesis under the original corpus conditions with identical texts and English references. Load and release models sequentially in separate processes. For 300 items, run 300 Japanese TTS, 600 source ASR, 900 MT, 900 English TTS, and 900 English ASR operations. This is longer than 8A/8B. Use a BF16-capable CUDA GPU with sufficient VRAM for the saved Japanese TTS batch. The run preserves the source batch setting.

**This experiment uses the user's confirmation that corpus generation used unchanged notebook settings.** `FLORES_FLEURS_SOURCE_CONFIG_CONFIRMED=True` is the default. Leave `FLORES_FLEURS_SOURCE_PROVENANCE` and `FLORES_FLEURS_SOURCE_CODE` empty and run **Cells 1 → 8C** on GPU. Keep `TRAIN_CODE_REVISION`. When only old 8C v1 code is present, fetch v2 diagnostic code separately. HF terms and authentication still apply.

The confirmation is bound to corpus revision `efe987934c48ee78b02132e97144815589ebeea7` and verified SHAs of `configs/colab-pro.yaml`, `notebooks/colab_pro.ipynb`, and dependencies. Original Cell 5 selects TTS batch 72 at VRAM 70 GiB or above, otherwise 1. Resolve dtype `auto` with the original rule, bfloat16 on a supported 8C GPU. Pass these settings to the original runner and preserve instruct, sampling, token limits, and format. Identity and model audit record actual batch, dtype, GPU, model and config SHAs, and the basis of confirmation.

`basis=user_confirmation` identifies a user statement. Historical execution logs, commit, random state, and byte-identical synthesis are unverified. For a customized corpus run, turn confirmation off and supply saved settings. An explicit `FLORES_FLEURS_SOURCE_PROVENANCE` takes priority and must validate. Corruption or mismatch stops without falling back to confirmation. Absence of both forms of evidence also stops. See [8C reproducibility notes](FLORES_FLEURS_DIAGNOSTIC.md).

Resume verifies input-text SHA, model, environment, data identities, and WAV SHA per successful stage. Rerun Cell 8C to retry failures. Verify individual successful Japanese WAVs and reuse them even if another member of their batch failed. Pass only pending or failed items to the original runner. Keep the configured batch cap and original seed function, and record actual retry batch members and seed. Use attempt 0 for source Japanese synthesis. The original corpus's CER-based generation selection is omitted as an explicit diagnostic difference. Failures remain empty hypotheses in the denominator.

Save `metrics.json`, `report.md`, `per_sample.jsonl/csv`, audits, selected WAVs, and stage checkpoints to `<PERSISTENT>/diagnostics/flores-fleurs-cascade/<identity>/`. The standard cell passes `--overwrite` to update derived reports within this identity after retry. Preserve successful WAVs, immutable checkpoints, and formal results. The cell retains `flores_fleurs_diagnostic["per_sample"]`. Report **corpus BLEU** with benchmark English normalization and sacreBLEU. Source CER divides total edit distance by total reference characters after NFKC and whitespace removal, retaining punctuation.

| Observation | Possible contributor to investigate |
|---|---|
| High Gold MT and Natural, low Synthetic | Synthetic Japanese source audio |
| High Gold MT, low Natural and Synthetic | Source ASR, Whisper, or recording conditions |
| Low Gold MT, Natural, and Synthetic | NLLB, BLEU, reference, or domain compatibility |
| High MT, low final ASR-BLEU only | English TTS or English evaluation ASR |

Treat these patterns as hypotheses. Cells 8/8A/8B, formal predictions and metrics, manifests, and training checkpoints retain their existing artifacts. Pin dependencies in a separate 8C environment. CPU fixtures cover resume, failure, and SHA validation. Real GPU results for all three conditions require separate verification.

### Compare two systems on a custom Japanese sentence (Cell 10)

Run Cells 1 → 10. Qwen synthesizes shared Japanese audio for Cascade and the S2UT, TT2, or S2T→TTS system selected by `INTERACTIVE_MODEL`. Display audio, ASR text, diagnostic BLEU, timing, RAM, and VRAM. Cascade English supplies the automatic reference. Store the comparison under `interactive/` and run GPU inference sequentially. Timing describes the first inference on one sentence. Cached results retain original measurements. See the [custom comparison guide](INTERACTIVE_COMPARISON.md) for settings, resume, and timing scope.

### Test scope

See [interruption and resume](RESUME.md) for durable units of work, recovery, and legacy data limitations.

CPU and mock tests cover notebook syntax, separation of smoke and standard settings, the full-data guard, log display, existing snapshots and resume, CPU/GPU staging, automatic rebuild after VM loss, and environment reuse. Python 3.10/Linux dependency resolution for training and inference was checked. Real Colab setup, GPU training, Drive reconnection, and four-system real-audio E2E remain unverified.

## Cell 8D: Blinded human audit of Japanese–English text pairs (CPU)

Start or resume with **Cells 1 → 8D** using the existing `OUTPUT_BASE` and `EXPERIMENT=experiment-001` in Cell 1. The cell mounts Drive and retrieves diagnostic code if needed. It uses CPU and works without Cells 2/3/8C or Hugging Face authentication.

Cell 8D is a blinded human audit of JA–EN text-pair quality.
It performs no model inference and does not alter official benchmark results.
Do not inspect corpus identity before completing annotation, because doing so
can bias the manual quality judgment.

Defaults are `TEXT_AUDIT_PER_STRATUM=100` and `TEXT_AUDIT_SEED=20261025`. Draw only from the experiment's `data/common/{train,test}.jsonl` and `dataset-lock.json`. Mix 100 per train/test × KFTT/JESC stratum, normally 400 total. Verify corpus from explicit metadata. If an exact join cannot establish a row's origin, exclude it with a reason. Stop if the required population's origin cannot be verified.

Read both texts and choose one of five labels. Set optional flags, confidence, and note before pressing a label. A label button saves immediately and advances. The label selector and `Save & Next` also save. Use `Previous` to explicitly revise a saved judgment. Before completion, the UI hides corpus, split, ID, automatic score, and per-stratum quality rates. `Show provenance` displays raw and TTS text only.

Each annotation is atomically saved to Drive at `diagnostics/text-pair-audit/<identity>/annotations.jsonl`. After disconnection, run **Cells 1 → 8D** with the same settings to continue at the first unannotated item. Changes to manifest, text, count, seed, or version create a new audit. Use one runtime per audit.

On completion, `report.md`, `summary.json`, `per_stratum.csv`, and `per_sample.csv` report stratum counts, quality rates excluding uncertain, Wilson 95% intervals, and flag frequencies. Interim reports show only completed and remaining counts. Keep identity, queue, and sampling files closed until annotation is complete to preserve blinding. The audit does not establish what causes direct S2ST performance.

The UI uses `ipywidgets` 7/8 and IPython in the standard Colab kernel. Missing dependencies stop before creating Drive output. See [TEXT_PAIR_AUDIT.md](TEXT_PAIR_AUDIT.md) for labeling, inputs, statistics, and verification.

## Cell 8E: External FLEURS/FLORES S2UT evaluation (GPU, resumable)

Run **Cells 1 → 8E** on GPU with the `OUTPUT_BASE / MODE / EXPERIMENT` used in 8C. Use saved results and prepared data.

- Leave `FLEURS_S2UT_RUN_NAME` empty to select a saved S2UT run as in standard Cell 6. Specify a saved name if ambiguous.
- Leave `FLEURS_S2UT_CASCADE_ID` empty to select the latest saved 8C evaluation, with preference for 300 items. Set it explicitly to select a particular evaluation.
- Set `FLEURS_S2UT_RESUME=True` to restore successful unit, WAV, and ASR journals from ZIPs.
- Set `FLEURS_S2UT_RUN=True` to evaluate. False inspects saved 8C input metadata without loading models.

Match canonical IDs, FLORES and FLEURS IDs, Japanese and English texts, SHAs, and natural audio receipts. Use **the same natural audio, English references, and English evaluation Whisper** for ASR-BLEU. Keep best by dev unit loss, beam 10, official unit HiFi-GAN, English normalization, and the formal S2UT failure-inclusive denominator. Reuse the existing 300 selections and use references only to score.

Write `report.md`, `metrics.json`, `per_sample.jsonl/csv`, and WAV ZIPs under `diagnostics/flores-fleurs-s2ut/<identity>/`. Read the saved 8C Cascade Natural score and display the difference. Training, 8C, and formal evaluation are read-only.

Restore S2UT in its saved fairseq environment and evaluation Whisper in the separate 8C environment. NumPy 2 and NumPy 1 stay isolated. The 8C bootstrap supports L4/A100. T4 is unsupported. The cell reads saved dataset artifacts, with possible downloads of public vocoder and Whisper weights. A missing 8C runtime installs the existing pins, including Qwen dependencies. The diagnostic executes neither Qwen nor NLLB. See [8E implementation and verification](S2UT_FLORES_FLEURS_DIAGNOSTIC.md).

For legacy `4e18c8f` with `8E evaluation dependency differs ... huggingface-hub`, rerun **Cells 1 → 8E with resume enabled** and identical settings in the updated notebook. Canonicalize package name punctuation (`-` / `_` / `.`) and case while requiring matching versions. Reuse saved units and WAV ZIPs after verifying model, inputs, and inference conditions. Preserve original plans and 8C and record the fixed execution separately. Preserve completed ASR and evaluations from distinct conditions.

`8E helper did not confirm agreement with the saved formal All score` was a display check after completed evaluation. Rerun **Cells 1 → 8E with resume enabled** and the same settings in the updated notebook. If `completion.json` and report checksums match, display the saved result. Updated display code retains inference revision `32cc967` and the original cache path so evaluation identity stays stable. Fetch that SHA if absent. The formal All-score confirmation belongs to 8A. Training, inference conditions, and saved reports are preserved.

## Cell 8F: Synthetic FLEURS/FLORES S2UT evaluation

Run **Cells 1 → 8F** in the same experiment with completed 8E. Reuse the 300 saved 8C synthetic Japanese WAVs and Natural 8E's S2UT run, dev-best checkpoint, beam 10, official unit HiFi-GAN, and evaluation Whisper. The cell uses the existing selection, source synthesis, and trained weights.

- Leave `FLEURS_SYNTHETIC_NATURAL_ID` empty to select the corresponding completed 8E. If several 8C identities have natural evaluations, set the saved 8E directory name.
- Set `FLEURS_SYNTHETIC_RESUME=True` to resume saved synthetic unit, WAV, and ASR stages.
- Set `FLEURS_SYNTHETIC_RUN=True` to evaluate. False checks saved metadata without loading models.

Restore an existing 8F plan's code SHA and Natural 8E identity. Even after display-side main updates, use the same execution code and absolute path so identity stays stable. Set NATURAL_ID explicitly if 8F evaluations correspond to multiple natural results. Invalid saved plans stop.

Compare input WAV SHA with formal 8C results and generation records. Check all IDs, Japanese text, and English references against 8E. Missing data, mismatches, or a different checkpoint stop. The scientific settings are the same except for source WAVs. Some shared function logs show 8E. The plan and output belong to 8F.

Store `report.md`, `metrics.json`, `per_sample.csv/jsonl`, and WAV ZIPs under `diagnostics/flores-fleurs-s2ut-synthetic/<identity>/`. Display Natural and Synthetic Cascade and S2UT BLEU, S2UT Synthetic − Natural and its ratio, and Cascade Synthetic − Natural. Include original KFTT/JESC test values only when saved formal artifacts verify. The report leaves causal interpretation to the reader. See the [8F guide](S2UT_SYNTHETIC_FLORES_FLEURS_DIAGNOSTIC.md).
