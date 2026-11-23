# S2UT Baseline reporting to Weights & Biases

The optional CPU reporter sends **existing measurements**, not new evaluations,
to the explicitly selected existing W&B project used by the transfer runs. Its run
display name is `baseline`; every chart uses the custom `update` axis. It does
not modify the scientific configuration, checkpoint bytes, training RNG,
optimizer, scheduler, checkpoint selection, or evaluation policy.

## Import the completed Baseline

Open [the independent import notebook](../notebooks/baseline_wandb_import.ipynb)
in Colab, using a **CPU runtime**. After this PR is merged, the normal entry is
[Open in Colab](https://colab.research.google.com/github/Yaaamashiro/ja-en-direct-s2st-benchmark/blob/main/notebooks/baseline_wandb_import.ipynb).
For pre-merge review, replace `main` in that URL with the PR branch or commit.

1. Keep `SOURCE` set to
   `/content/drive/MyDrive/alignment-transfer-s2ut-data/experiment-001/baseline`.
   Set `WANDB_ENTITY` to the **same account/team** as the transfer runs.
   Set `WANDB_PROJECT` to their actual existing project; the generic default is
   `alignment-transfer-s2ut`. No personal entity or live project is preset.
2. Add a W&B API key with project write access to the Colab secret named
   `WANDB_API_KEY`, and grant this notebook access. Never put credentials in code.
3. Run cells 1–3 to inspect the confirmed update, metric counts, summary, and
   source hashes. This part needs only Python's standard library.
4. Set `SYNC_TO_WANDB=True`, rerun cell 1, then cell 4. The notebook installs
   `wandb==0.31.0` and pinned dependencies into a separate CPU environment and
   prints the resulting W&B run URL. It does not build the training runtime.
5. Rerun cell 4 to retry or resynchronize. Reconnecting requires cells 1–2 first.
   Keep the same entity, project, and source experiment.

The notebook never loads Fairseq, Torch, a vocoder, an ASR model, checkpoint
payloads, or audio, and never recomputes BLEU. No changes to the original Drive
Baseline are needed. Reporter state and SDK logs are stored in the **sibling**
`experiment-001/wandb-baseline` directory. A state directory cannot be inside,
equal to, or an ancestor of the source directory.

The source must retain a published `snapshot.json` receipt (normally below a
`000000400000-...` snapshot folder). This small receipt proves the confirmed
optimizer update and contains report hashes. There is no checkpoint/ZIP download.
Reports must match the **latest** receipt exactly; a dashboard that is temporarily
newer/older than the receipt is deferred rather than treated as committed data.
An explicit `--snapshot` path supports a read-only copy of a selected receipt.

### Audited completed run

On 2026-11-24, the four source files matched the report hashes in the final
400,000-update snapshot. Importing them produces **4,000 combined update rows**:

| Measured series | Observations per series |
| --- | ---: |
| Train total, unit, source-letter, target-letter, decoder-CTC losses | 4,000 |
| Learning rate | 4,000 |
| Dev total, unit, source-letter, target-letter, decoder-CTC losses | 400 |
| Aux-BLEU, sample count, failure count | 400 |
| ASR-BLEU, sample count, failure count | 8 |
| Saved evaluation elapsed seconds | 400 |

The eight audio evaluations are at updates 1,000; 5,000; 10,000; 25,000;
50,000; 100,000; 200,000; and 400,000. Unscheduled ASR entries contain placeholder
`asr_samples=0` and `asr_failures=0`: **none of these are exported**. A measured
BLEU of zero or a measured failure count of zero **is** exported.

The saved best checkpoint is update **400,000**, dev unit loss
**2.7741764784057237** bits, SHA-256
`be2b62f1fd88cbef3620cf2592cfd1651084fe81a6d686de8a2b8a1f2c626806`.
Only this saved best metadata is reported; historical best checkpoints are not
invented. The JSON/CSV duplicates are reconciled, not counted twice.

| Source | SHA-256 |
| --- | --- |
| `training-monitor.json` | `d515db7916a900d981591f5b621f1e79551f2f0f95b931213cc22aa5535f04e9` |
| `training-monitor.csv` | `6def99a0f28f291089c1604baa775005a2132208724153cf2ae18cdece970a92` |
| `training-bleu.json` | `4183bfb50d19f3ae6695c3ed32248d951ec8d083d9a2fda233fc7eac8df9f708` |
| `training-bleu.csv` | `23b6e722267a1e9926b8bc8a091c24ee8cfc1570d24e4eab27243d57dd7de608` |

These counts are an audit of saved files, not a claim that an authenticated cloud
upload has already occurred. No W&B credentials are committed or requested in chat.

## Future Baseline training: normal Cell 5

Use the updated `colab_training.ipynb`. In Cell 1 set `TRAIN_TARGET='s2ut'`,
`BASELINE_WANDB=True`, and the same `WANDB_ENTITY` as the transfer runs (the public default is empty). Set
`WANDB_PROJECT` to the existing transfer project. The generic initial value
`alignment-transfer-s2ut` matches the transfer repository's default, not any
individual account's live project. Add the Colab secret as above, then use
normal Cell 5 for new training or resume. If a saved launcher predates this
feature, set `TRAIN_CODE_REVISION` to an immutable integration commit after
review. The scientific execution revision and saved experiment remain pinned.

W&B settings are launcher-only arguments; they are not added to `make_config`,
the scientific command, or the experiment fingerprint. Cell 5 starts a separate
CPU interpreter that polls the published report mirror every 60 seconds. Only
measurements at or below a published checkpoint update are eligible. SDK network
work never runs inside the trainer. Missing credentials/dependencies, mismatched
mirrors, network failures, or reporter crashes leave training running. Shutdown
waits at most 25 seconds plus bounded termination; a later sync can finish delivery.

Live state is stored at `PERSISTENT/wandb-baseline/<actual-training-run-name>`.
The helper deliberately supports only S2UT Baseline reporting. It does not turn
W&B on for TT2/vocoders, and does not change the other repository's training
workflow. That repository's completed Baseline can use the independent importer.

For a manual CPU retry against a future run's published backup:

```sh
/content/s2st-wandb-runtime/bin/python scripts/colab/baseline_wandb.py sync \
  --source "$PERSISTENT/checkpoints/$RUN_NAME" \
  --state "$PERSISTENT/wandb-baseline/$RUN_NAME" \
  --entity YOUR_EXISTING_ENTITY
```

The API key must already be supplied through the environment/Colab secret; do
not add it to command arguments. The same helper offers `inspect` without W&B
dependencies and `watch` for an external wrapper. `inspect` reads sources only.

## Metric definitions and transfer comparison

The mapping was reviewed against transfer repository commit
`572b874fbdc36ebfed418c5bf88de2ba37ffa613` (`tools/wandb_transfer.py` and
`s2ut/benchmark_extension.py`), and this repository's `s2ut/monitor.py`.
The SDK's fixed cloud API host is `https://api.forge.coreweave.com`, consistent
with the [W&B 0.31.0 release](https://github.com/wandb/wandb/releases/tag/v0.31.0).
An inherited `WANDB_BASE_URL` cannot silently redirect this reporter to another server.

| Saved Baseline field | W&B metric | Definition / comparison |
| --- | --- | --- |
| `train.unit_loss` | `baseline/train/unit_loss_bits` | Main label-smoothed criterion sum / sample size / ln(2). Transfer's native `train/main_loss` is in **nats**, so it is not aliased. |
| `train.loss` | `baseline/train/loss_bits` | Weighted multitask total / main sample size / ln(2). Transfer's total includes an alignment objective and uses nats; do not compare them under one name. |
| Train auxiliary losses | `baseline/train/<name>_bits` | Each auxiliary task uses its **own** denominator, divided by ln(2). |
| `dev.unit_loss` | `dev/main_loss` | Same label-smoothed main criterion, sample-size aggregation, and bits as transfer's native dev main loss. **Not** raw NLL. |
| Dev total / auxiliary losses | `baseline/dev/<name>_bits` | Keep explicit Baseline definitions rather than silently aliasing Fairseq aggregates. |
| Learning rate | `train/learning_rate` | Recorded optimizer LR; no inferred schedule. |
| Aux / ASR BLEU and counts | `eval/aux_bleu`, `eval/asr_bleu`, `eval/<aux-or-asr>_samples`, `eval/<aux-or-asr>_failures` | The existing measured score/count. Conditions/signatures are retained. Same name does not prove identical subsets, ASR, or vocoders; compare recorded conditions. |
| Max update / durable progress | `train/max_update`, `train/durable_update` | Saved maximum and confirmed absolute optimizer update; summary values. |
| Best checkpoint | `baseline/best_value`, `baseline/best_update`, `baseline/best_sha256` | Saved checkpoint-selection metadata; summary only. |

Other numeric train/dev measurements are retained under
`baseline/<train-or-dev>/observed/<original-name>` without claiming loss units.
Scoring details, measured elapsed seconds, status, scoring signatures,
normalization, model-weight hash, and a condition hash use `baseline/eval/*`.
Exact evaluation conditions, source file names/hashes/sizes, source identity,
and receipt path are retained as config/summary metadata. The private reference
sentences and sample text are not uploaded.
Every event also retains its own source file names, SHA-256 hashes, sizes,
snapshot receipt path, and confirmed update. Later source versions do not erase
the provenance of previously sent rows.

No conversion from bits to nats, interpolation, missing-value substitution,
extra validation, or checkpoint-selection changes are made. Loss comparability
does not imply the Baseline and transfer runs have identical scientific contracts;
check the recorded datasets, label smoothing, batching, and evaluation conditions.

## Identity, retries, and duplicate protection

The W&B ID is `baseline-<24 hex digits>` derived from the immutable snapshot
identity (scientific config, input hashes, original work path), explicit entity,
project, and mapping version. Copying artifacts to a new local path does not
create another run. Different experiments get different IDs even with the same
display name. Existing transfer runs are never searched by display name,
overwritten, deleted, or merged into the Baseline.

A separate state binding prevents accidentally reusing another experiment's
outbox. The remote run's binding is also checked before writes. A destination that differs from an existing state/run binding,
conflicting historical metric, changed report bytes, or stale source
fails closed. Only an existing destination project is accepted.

Each update's available train/dev/evaluation metrics are coalesced. Durable
outbox events have content hashes and assigned monotonic W&B internal steps.
Before replay the reporter scans **unsampled, uncached remote history**, verifies
event hashes, and sends only missing metric cells. A late BLEU result may add a
second row at the same custom `update`; it does not repeat the train/dev cells.
Uncertain acknowledgements retain the frozen step instead of allocating a new
one. If the server step has advanced before history is visible, delivery waits
for a later retry rather than appending duplicate data. Deleting local receipts
is unnecessary; even a new state directory reconciles against remote events.

Use **one writer per Baseline run**, including between the live reporter and
import notebook. A local file lock prevents concurrent writers sharing a state
directory; it is not a distributed lock between unrelated machines. Remote
history reconciliation is conservative, but W&B availability/consistency is
external: do not run simultaneous import/live writers on different runtimes.

## Validation

CPU unit tests cover JSON/CSV deduplication and conflicts, absent/nonfinite
values, real zero vs unscheduled ASR placeholders, publication gating and hashes,
read-only artifacts, idempotence with lost local state, failures before/after
network acceptance, late BLEU/resume, foreign runs/destinations, notebook syntax,
and reporter failure preserving the trainer outcome. Existing Cell-5 recovery
tests continue to exercise scientific code/environment/config preservation.
The affected-path CPU regression suite passed **336 tests**, with **one skip**
because this Windows host lacks symlink creation privileges. This includes
**22 new W&B tests**. It is not a claim that GPU/model-inference tests were run.

The pinned W&B SDK was additionally exercised offline with all 4,000 real update
rows (30,864 numeric/metadata cells). Linux/Colab Python 3.13 binary dependencies
were resolved; transitive versions are constrained separately. An authenticated
online delivery and actual Colab execution still require the user's W&B secret.
