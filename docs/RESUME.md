# Colab interruption and resume

Run Cell 1 with the same experiment name, output path, code revision, and settings, then rerun the interrupted stage. A missing local environment triggers the equivalent of Cell 3 automatically. Keep `RECHECK_AUDIO=False` and overwrite disabled. Resume replays work from the last durable unit of progress. In-memory state is lost.

## Recovery by stage

| Stage | Saved and reusable unit | Work repeated after interruption |
|---|---|---|
| Cells 1–3, runtime setup | Saved revision, environment lock, built components | A fresh VM rebuilds Python and dependencies. Interrupted build commands rerun. |
| 4A, corpus import | Per-record audio resolution and WAV/SHA-256 verification | Unsaved records, manifest loading, and metadata checks |
| 4A, phonemes | Per-sample phoneme sequence | Unsaved samples. Rebuild final TSVs and inventory from saved results. |
| 4A, mel preparation | Input WAV headers, distributed NPYs, split ZIPs | Unsaved NPYs and incomplete ZIPs. NPYs use 256 folders. New ZIPs normally split at 512 items or 256 MiB, with a large-single-NPY exception. |
| 4A, final verification | Per-sample successful verification | Unsaved samples and file hashing for the final fingerprint |
| 4A.5, CPU audio ZIP preparation (standard notebook) | Immutable 128-WAV ZIPs and receipts for both languages and all splits | Unpublished ZIPs. Rerun the same CPU cell. Originals and completed ZIPs are preserved, and full local extraction is unnecessary. |
| 4B, HuBERT | Extracted units and per-sample records | Unsaved samples or batches and model loading |
| 4B, training data preparation | Per-sample unit reads and WAV headers, final TSVs and dictionaries | Unsaved samples, text-label creation, and TSV consistency scans |
| 5, TT2/S2UT/vocoder training | Verified checkpoints and snapshots | Updates since the latest durable save. Before the first save, restart from the beginning. Rebuild local read cache on a fresh VM. |
| 5, concurrent TT2 input retrieval | Local verified receipt for each WAV/mel ZIP | Reuse completed ZIPs on the same VM. A fresh VM fetches inputs from the restored update and prefetches during training. Partial copies remain unpublished. |
| Training artifact restore and model retrieval | Complete artifacts and file hash checks | Interrupted copies or downloads may restart at the file boundary. |
| TT2/Cascade/S2T inference and synthesis | Successful samples and result journal | Unsaved and failed samples, model loading, and input hashing |
| S2UT inference | Successful fairseq shards of about 32 samples | Interrupted shard. Each shard reloads the model. |
| Evaluation | Per-sample evaluation journal | Unsaved and failed samples. A failure in an enabled additional metric also retries that sample, including ASR. |
| Acceptance and aggregation | Successful WAV verification records and identical CSV/JSON/Markdown output | Unverified WAVs, consistency and training checkpoint checks, and aggregation. Reuse partially published identical output. |

Sample checkpoints normally publish in groups of 64 or after about 10 seconds at the next record. Forced termination loses unpublished work. Graceful exception handling saves remaining records. ZIPs and S2UT generation save by successful shard. Training follows its configured interval. `durable_updates` is the recovery point. Drive mount writes are read back for verification. Server-side Drive synchronization is outside this guarantee.

## Concurrent retrieval and no time limit for TT2 started on 48fcdb7

`scripts/colab/resume_streaming_training.py` explicitly loads execution code from a separate clean Git worktree and retains the original pinned revision and dependencies. The saved config, snapshots, environment lock, and training conditions are preserved. `PERSISTENT/execution-overlays/<revision>.json` records the original and execution revisions and environment/config SHA-256 values. Trainer research metadata also records the execution revision. Training target, optimizer, model, and paper settings remain tied to the run.

The supported base is `48fcdb7eb023cf35860fe63716dd7dacdd277be5`. Stop on changed source outside approved execution/input code or changed configs, requirements, fairseq, or models. Run Cells 1 → 2 → 3 with the same experiment and destination to fetch and build the original revision. Then fetch this helper from the published fix and pass an explicit `--overlay-revision`. The helper uses the saved config directly. Repeat after disconnection, including retrieving the overlay on a fresh VM.

```python
# Supply the full SHA of the fix. Keep the original REPO checkout.
REVISION = '<full SHA of the published commit containing the fix>'
run('git', '-C', REPO, 'fetch', 'origin', REVISION)
helper = Path('/content/resume_streaming_training.py')
helper.write_text(subprocess.check_output(
    ['git', '-C', str(REPO), 'show', REVISION + ':scripts/colab/resume_streaming_training.py'],
    text=True), encoding='utf-8')
RUN_NAME = training_run_name('tt2')
assert TOTAL_UPDATES <= 10 or CONFIRM_TRAINING
options = ['--confirm-training'] if CONFIRM_TRAINING else []
run(sys.executable, helper, '--repo', REPO, '--persistent', PERSISTENT,
    '--config', PERSISTENT / (RUN_NAME + '.json'),
    '--work-root', '/content/s2st-work/' + RUN_NAME,
    '--backup-root', PERSISTENT / 'checkpoints' / RUN_NAME,
    '--overlay-revision', REVISION, '--total-updates', str(TOTAL_UPDATES), *options)
```

Execution uses `--resume`, `--session-seconds 0`, and concurrent retrieval. It verifies snapshots, inputs, and resume state and stops on a rebuilt environment SHA mismatch. Inspect the log and preserve the environment lock and checkpoint. Real Colab/GPU overlay execution and Drive speed remain unverified.

### Add monitoring explicitly to existing TT2 training

Use an overlay commit containing monitoring and the actual saved run name. For a run with the `-probe-fix-v1` suffix, set `RUN_NAME = training_run_name('tt2') + '-probe-fix-v1'`. Add `--monitor` to the helper. Its defaults are evaluation every 10 updates, patience 10, and relative improvement 0.001. The existing config/snapshot identity and optimizer settings are retained. Record adoption in the overlay audit and new checkpoint. The checkpoint stores the counter and best-checkpoint SHA-256. Immutable snapshots also include history, CSV, SVG, and the best checkpoint. Resume inherits the saved policy. An already stopped checkpoint returns `EARLY_STOPPED` without updates. History starts at adoption and stopping starts after warmup. If TOTAL_UPDATES is at or below warmup, the cap arrives first. This policy uses full-dev teacher-forced loss. The paper selects by dev BLEU.

### Shorten warmup to 1,000 absolute updates from a checkpoint such as update 85

For an existing TT2 checkpoint without monitoring, add `--monitor --monitor-interval 10 --practical-warmup-updates 1000` to the helper. Use the same config, work root, backup root, and actual run name including `-probe-fix-v1`. Restore the model, Adam, update count, and RNG from the latest verified snapshot. Starting with the next update, apply `base_lr * min(absolute_update / 1000, sqrt(1000 / absolute_update))`. LR rises linearly through 1000 and then decays by inverse square root. Restoring 85 makes 86 next, with 10 times the LR of update 86 under the old 10,000-step schedule. Reuse prepared data and learned state.

Keep the original config and fingerprint as provenance. Record the `paper_practical` change in the overlay audit, research metadata, and `schedule_transition` in last/best checkpoints. The change applies prospectively. Resume inherits the shortened schedule using this or a later compatible overlay. The helper rejects repeated changes, first adoption after 1000, and reset of existing monitor history. Early stopping starts after absolute update 1000. The 10,000 cap can be retained.

## Resume through standard Cell 5

The current notebook uses `training_session.py` in Cell 5 to find a saved run, restore compatible code and dependencies, and run `--resume`. This includes old TT2 probe-fix-v1 runs. All dependency versions are stored on Drive. For older runs without a manifest, check candidates from successful logs against measured environment hashes. Stop and retain data if the environment or run is ambiguous. See "Integrated resume in Cell 5" in COLAB.md. The manual overlay cells above document historical recovery. Standard Cell 5 handles the integrated flow.

## Rerun completed 4A and 4B

Enable S2UT monitoring explicitly via `S2ST_S2UT_MONITOR_POLICY`. The notebook normally sets it. Resume a run started with warmup 400 using the same name and config. The last checkpoint's `extra_state.s2st_monitor` restores history and counter, with the best model from the same snapshot. V2 retains both unit and total loss counters and anchors. A v2-capable execution revision can explicitly request v2 at the same evaluation interval. Verify earlier history and best checkpoint and audit the future-only policy change. Reject changes to the interval or legacy v1 policy, clearing of a stopped run, and missing or corrupt best checkpoints. Reject warmup changes before setup. Resume runs started with 1000 or 10000 under their original settings. Once enabled, saved monitoring applies even if the resume flag is omitted. `resume_streaming_training.py` is TT2-specific.

The notebook calls `direct_s2st.preparation_workflow` and records completed stages in `data/.prep-checkpoints/stages/`. If revision, settings, and input/output file lists, sizes, and modification times match, it skips expensive work and prints `[preparation-stage] ... status=reused`. Metadata checks can still take time on Drive. Interrupted stages continue from their saved sample checkpoints.

This fast reuse check assumes an immutable published corpus. Content changes that retain size and mtime can go undetected. Keep originals read-only. Set `RECHECK_AUDIO=True` to bypass stage receipts and revalidate WAVs. Compatible phoneme, HuBERT, and mel outputs can still be reused. Corruption and settings mismatches stop the run.

## Existing artifacts

- Legacy single mel ZIPs retain their paths and lock format. ZIP index validation also checkpoints progress.
- Older data without stage receipts needs one verification pass. Compatible checkpoints are reused.
- Progress never saved by earlier versions cannot be recovered.
- Running Colab sessions use their existing code. Cell 1 uses the saved revision even after a remote update. Use a supported migration and retain old artifacts and revision/lock metadata.

## When legacy 4B spends a long time building audio ZIPs

A line such as `stage WAV ZIPs: train/en checked=200/1105` means 200 completed ZIPs with up to 128 WAVs each. `audio=...` counts audio files, `built` counts new ZIPs, and `reused` counts existing ZIPs. `checked` restarts from zero on resume. Completed ZIPs skip reading and rebuilding source WAVs.

Stop and switch to a CPU runtime, then run Cells 1 → 2 with the same experiment and destination. Fetch and check out the fix and run the script below to update the pin and finish the ZIPs on CPU. Use it only before starting training. Completed phoneme/mel recovery from Cell ③ can be reused.

```python
import sys
REVISION = '<full SHA of the published commit containing the fix>'
run('git', '-C', REPO, 'fetch', 'origin', REVISION)
run('git', '-C', REPO, 'checkout', '--detach', REVISION)
configure_preparation()
run(sys.executable, REPO / 'scripts/colab/prepare_audio_packs.py',
    '--persistent', PERSISTENT, '--revision', REVISION,
    '--profile', PROFILE, '--overwrite')
```

`--overwrite` permits archiving and updating the old revision pin. Originals, phonemes, mel, units, and existing audio ZIPs are retained. The destination `data/.drive-audio-packs/` and ZIP identity match `55a5a90`, so completed ZIPs from legacy 4B are reused and only unpublished ZIPs read the originals. After another disconnection, rerun the same command or updated CPU Cell 4A.5.

After audio ZIP preparation completes, switch to GPU and run Cells 1 → 4B in the updated notebook. A fresh VM fetches and extracts ZIPs. 4B stops if the published English and Japanese ZIPs are incomplete. Performance and interruption recovery on the full 140,000-item corpus with Drive and GPU remain unverified.

## When mel preparation stops on missing eSpeak phonemes

The sample-word inventory for eSpeak NG 1.52.0 lacked seven phonemes found in train: `a‍ɪ‍ə`, `a‍ɪ‍ɚ`, `o`, `r`, `ɐ`, `ɑ̃`, and `ɔ`. Add this fixed supplement to the training dictionary and record the supplement and dictionary hashes in a new data lock. Keep original phonemes/TSV, inventory, metadata, and checkpoints. Validate every split against the dictionary before mel extraction. Unknown dev/test phonemes are not added automatically. A run that already started training cannot adopt this dictionary on resume.

After disconnection, set the same experiment in Cell 1, fetch the fix, and run `scripts/colab/resume_preparation.py --persistent <experiment-directory> --revision <new-full-SHA> --profile pilot --overwrite`. It verifies saved phoneme hash/count/IDs across splits, archives the old repository revision pin, and updates that pin. `--overwrite` applies only to the revision pin. Phonemes and mel use `--resume`. The script builds the environment, continues mel preparation and final verification, and saves stage completion on success. It stops for experiments with a training config or checkpoint. After completion, switch to GPU and run Cells 1 → 4B.

Runtime rebuild installs uv into host Python without inherited constraints such as old `PIP_CONSTRAINT`. Apply Python 3.10 dependency constraints after creating the venv.

## EOFError at the start of mel extraction

Older versions can report `EOFError: No data left in file` immediately after `mel: extract/reuse features checked=0`. They created an empty temporary NPY before calling pinned fairseq, whose default skips existing files. The fixed official extraction call uses `overwrite=True` only for a new unpublished temporary file. Published outputs and cache are preserved. Migrate to the fix and rerun the same recovery helper. Compatible phoneme and input WAV header checkpoints are retained. `saved=0` means no extracted mel outputs were available for reuse.

## Migrate phoneme generation and dictionaries

New generation builds a split-independent IPA dictionary from pinned eSpeak NG 1.52.0 definitions `base1 → en → en-us`, in addition to sample words. It uses this G2P and fixed dictionary across splits. Contractions, possessives, decimals, abbreviations, in-word hyphens, and times reach eSpeak intact, including `don't`, `prefecture's`, and `3.14`. Metadata records generator and dictionary versions and TSV SHA. An out-of-vocabulary phoneme stops immediately with the sample ID and text before saving an invalid label. Before mel extraction, validate the dictionary coverage, IDs, and metadata of saved labels across all splits.

The validation command checks phonemes without reading WAVs.

```bash
python -m direct_s2st.cli translatotron2 validate-phonemes --profile pilot
```

Standard `--resume` reuses completed legacy metadata and its original generation method. Incomplete legacy results with a mismatching dictionary method require explicit migration. Adding the seven supplemental phonemes alone keeps the old labels. To also correct contraction and number handling, fetch the fix before training and use the explicit regeneration flag.

```bash
python scripts/colab/resume_preparation.py \
  --persistent <experiment-directory> --revision <full-fix-SHA> \
  --profile pilot --overwrite --regenerate-phonemes
```

The old phonemes, dictionary, and data lock are archived under `data/translatotron2/phoneme-migrations/<ID>/`. Regeneration takes time. The corpus, saved mel, partial mel checkpoints, and published ZIPs are preserved and compatible features are reused. Migration checkpoints progress, so rerun the same command after disconnection. Experiments with training configs or checkpoints reject automatic migration.

## Verification scope

CPU tests cover interruption and resume, completed-result reuse, change and corruption detection, notebook entry points, training snapshot restoration, inference and evaluation journals, and repeat aggregation. Real Colab 24-hour termination, Drive synchronization failures, and full 140,000-item GPU end-to-end execution remain unverified.

## Drive input/output errors with many mel NPYs

Older code put all NPYs in one `features/` directory. The [Colab FAQ](https://research.google.com/colaboratory/intl/en-GB/faq.html#drive-timeout) describes I/O errors in large folders and recommends keeping folders below about 10,000 files. Operation and bandwidth quotas can produce the same error. The log alone cannot establish corruption or loss.

The new path works on local disk and saves ZIPs with up to 128 items and a target of 256 MiB under `<Mel checkpoint root>/packs-v1/<first-two-characters>/pack-<ID>.zip`. Packing reduces individual Drive writes. On resume, reuse ZIPs with published receipts. Extraction settings, checkpoint identity/key, and old checkpoints are retained. Existing copies in `features-v2/` are verified by original SHA-256 before packing. The old `features/`, `features-v2/`, and chunks stay in place.

Check each NPY against its saved SHA-256 and read back the ZIP to verify its hash. A matching immutable path/size/mtime receipt skips reading ZIP contents on the next run. Content changes with the same size and mtime can go undetected. Set `S2ST_PREP_RECHECK=1` to verify contents again while retaining extracted mel.

### Recover from a legacy folder that the mount cannot read

`scripts/colab/repair_mel_cache.py` can list the old folder read-only through the Drive API and download by file ID without opening the old mounted NPY paths. Download and build ZIPs locally, then write the saved ZIPs through the mount. Retaining the originals and checkpoints requires space for both recovery ZIPs and subsequent training ZIPs. Resolve API permissions, file sync, quotas, and destination write errors separately. Recovery time and success depend on those conditions.

### Recover from downloaded folder ZIPs

If Drive's web UI can download old `features` as ZIPs, place them on Colab's local disk. Repeat `--source-zip` for each part. The script reads the ZIPs directly. Large web downloads can take time or fail.

```python
MEL_CACHE = PERSISTENT / 'data/translatotron2/.prep-checkpoints/mel/2b2a74082e7462215d32ceca'
repair_args = [sys.executable, REPO / 'scripts/colab/repair_mel_cache.py',
               '--cache-root', MEL_CACHE, '--storage', 'packed',
               '--source-zip', '/content/features-part-1.zip',
               '--source-zip', '/content/features-part-2.zip']
run(*repair_args, '--limit', '3')
run(*repair_args)
```

Only NPYs matching checkpoint SHA-256 are saved. For entries missing from the ZIPs, the helper looks for existing distributed copies, then readable legacy files. Add API arguments below for missing files when the old mount is unreadable. Supply source ZIPs again on a fresh VM if pending files need them. Published packs are reused. The helper leaves source ZIPs in place.

### One-time API migration without source ZIPs

The API reads legacy media one file at a time. Listing and download requests default to one-second spacing. Only 403 rateLimitExceeded/userRateLimitExceeded, 429, and transient 5xx responses are retried, up to eight times with backoff. Rate limits increase the spacing for the current process. Permission and daily quota failures are not retried unconditionally. The initial migration can be long and one-second spacing can still exceed some quotas.

Run Cell 1 for the same experiment, then fetch the published fix while no stage is running. Leave saved pins, phonemes, mel, and checkpoints in place.

```python
import sys
ensure_workspace()
configure_preparation()
REVISION = '<full SHA of the fix>'
run('git', '-C', REPO, 'fetch', 'origin', REVISION)
run('git', '-C', REPO, 'checkout', '--detach', REVISION)
```

Open the old `features` folder inside the checkpoint named in the error log in Drive. Copy the ID after `/folders/` in its URL. Use the matching cache root. Authenticate in Colab as the same account as the mount.

```python
from google.colab import auth
auth.authenticate_user()
MEL_CACHE = DATA / 'translatotron2/.prep-checkpoints/mel/2b2a74082e7462215d32ceca'
FEATURES_FOLDER_ID = '<ID of the old features folder>'
# Use the host Python executable.
repair_args = [sys.executable, REPO / 'scripts/colab/repair_mel_cache.py',
               '--cache-root', MEL_CACHE, '--storage', 'packed',
               '--drive-folder-id', FEATURES_FOLDER_ID, '--api-interval', '1']
run(*repair_args, '--limit', '3')  # Check SHA and destination writes on three items first.
```

After the trial succeeds, copy with no limit in the same or next cell.

```python
run(*repair_args)
```

Track `[mel-recovery] packed=... reused=... persisted=... pending_local=... remaining=...`. `packed` counts newly packed entries, `persisted` counts ZIP-saved entries, and `pending_local` counts unpublished local entries. Forced exit only loses unpublished progress. A three-item trial checks the workflow and may miss quota or write issues at full scale. Resume with the same authentication and arguments. Existing packs and distributed copies avoid redownloads. The complete file-ID listing is saved under the same checkpoint root. An interrupted listing starts again. Use `--refresh-index` only for a stale listing. The old listing is archived, and original NPYs and chunks are retained. Missing or duplicate files and SHA mismatches stop the migration.

`authenticate_user()` may not resolve permission issues. If quota errors outlast retries, check request spacing (for example `--api-interval 2`), the FAQ, permissions, and available space before retrying. Backoff does not raise quota.

After copying, update the saved pin with the recovery helper and resume mel preparation and final verification. Here `--overwrite` authorizes the pin update. Omit `--regenerate-phonemes` if phoneme migration is already complete.

```python
run(sys.executable, REPO / 'scripts/colab/resume_preparation.py',
    '--persistent', PERSISTENT, '--revision', REVISION,
    '--profile', PROFILE, '--overwrite')
```

The helper reuses compatible saved mel, extracts remaining samples, and proceeds to ZIP/TSV creation and final verification. Source audio, phonemes, and the training dictionary are preserved.
