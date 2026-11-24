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
   `alignment-transfer-s2ut`. No personal entity or live project is preset. Set `WANDB_GROUP` to the comparison group, for example
   `experiment-001/transfer-fixed-v1`. If the Baseline already has a verified
   W&B run, enter its exact ID in `WANDB_EXISTING_RUN_ID`.
2. Add a W&B API key with project write access to the Colab secret named
   `WANDB_API_KEY`, and grant this notebook access. Never put credentials in code.
3. Run cells 1–3 to inspect the confirmed update, metric counts, summary, and
   source hashes. This part needs only Python's standard library.
4. Set `SYNC_TO_WANDB=True`, rerun cell 1, then cell 4. The notebook installs
   `wandb==0.31.0` and pinned dependencies into a separate CPU environment and
   prints the resulting W&B run URL. It does not build the training runtime.
5. Rerun cell 4 to retry or resynchronize. Reconnecting requires cells 1–2 first.
   Keep the same entity, project, and source experiment.
   An `ACK_PENDING` result means the W&B history API has yet to confirm every
   submitted row. Keep the same `STATE` and rerun cell 4 to check again. If the
   SDK has finished uploading, retries only read the pending rows.

The notebook never loads Fairseq, Torch, a vocoder, an ASR model, checkpoint
payloads, or audio, and never recomputes BLEU. No changes to the original Drive
Baseline are needed. Reporter state and SDK logs are stored in the **sibling**
`experiment-001/wandb-baseline` directory. A state directory cannot be inside,
equal to, or an ancestor of the source directory.

The source must retain a published `snapshot.json` receipt (normally below a
`000000400000-...` snapshot folder). This small receipt proves the confirmed
optimizer update and contains report hashes. There is no checkpoint/ZIP download.
Reports must match the **latest** receipt exactly. If the dashboard differs, the
reporter reads only the archived report members and checks their hashes.
An explicit `--snapshot` path supports a read-only copy of a selected receipt.

### Audited completed run

On 2026-11-24, the four source files matched the report hashes in the final
400,000-update snapshot. Importing them produces **4,000 combined update rows**:

| Measured series | Observations per series |
| --- | ---: |
| Train total, unit, source-letter, target-letter, decoder-CTC losses | 4,000 |
| Learning rate | 4,000 |
| Dev total, unit, source-letter, target-letter, decoder-CTC losses | 400 |
| Aux-BLEU, sample count, failure count (`*_legacy_unclassified`) | 400 |
| ASR-BLEU, sample count, failure count | 8 |
| Saved evaluation elapsed seconds | 400 |

The eight audio evaluations are at updates 1,000; 5,000; 10,000; 25,000;
50,000; 100,000; 200,000; and 400,000. Unscheduled ASR entries contain placeholder
`asr_samples=0` and `asr_failures=0`: **none of these are exported**. A measured
BLEU of zero or a measured failure count of zero **is** exported. The 400 saved
Aux points have failures that were not classified by the original evaluator.
They enter `eval/aux_bleu_legacy_unclassified` and its companion count series.
The eight zero-failure ASR points enter `eval/asr_bleu`.

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
individual account's live project. Set `WANDB_GROUP` and
`WANDB_EXISTING_RUN_ID` as needed, then add the Colab secret as above and use
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

For a manual CPU retry against a future run's published backup, use the
interpreter path printed as `[baseline-wandb] CPU Python:`:

```sh
"$REPORTER_PYTHON" scripts/colab/baseline_wandb.py sync \
  --source "$PERSISTENT/checkpoints/$RUN_NAME" \
  --state "$PERSISTENT/wandb-baseline/$RUN_NAME" \
  --entity YOUR_EXISTING_ENTITY
```

The API key must already be supplied through the environment/Colab secret; do
not add it to command arguments. The same helper offers `inspect` without W&B
dependencies and `watch` for an external wrapper. `inspect` reads sources only.

## Metric definitions and transfer comparison

The comparison mapping follows [alignment-transfer-s2ut Metric Schema v2](https://github.com/Yaaamashiro/alignment-transfer-s2ut/blob/ffbb7a9dc1a290899321e80c2408f8fe5236bc05/docs/WANDB_BASELINE_COMPARISON.md).
The SDK's fixed cloud API host is `https://api.forge.coreweave.com`.

| Saved Baseline field | W&B metric | Definition |
| --- | --- | --- |
| Train Unit | `train/unit_ce_nats` | One logical update, label-smoothed Unit CE sum divided by main target-token count. Convert saved bits by multiplying by ln(2). |
| Train weighted task loss | `alignment/update/task_loss` | One logical update, Unit CE plus the three weighted auxiliary sums (8, 8, 1.6) divided by main target-token count. Convert saved bits by multiplying by ln(2). |
| Train auxiliaries | `train/aux/<task>_loss_bits` | Unweighted bits per auxiliary sample unit from one logical update. |
| Dev Unit | `dev/main_loss` | Full-Dev label-smoothed Unit loss in bits. |
| Dev total / auxiliaries | `dev/full/loss`, `dev/full/multitask_<task>_loss` | Full Dev. Weighted total in bits per main token and unweighted auxiliary bits per auxiliary sample unit. |
| Learning rate | `train/update/learning_rate` | Saved learning rate after the optimizer update. |
| Comparable BLEU and counts | `eval/<mode>_bleu`, `eval/<mode>_samples`, `eval/<mode>_failures` | The saved score and counts, gated separately for Aux and ASR. |
| Published progress | `train/max_update`, `train/durable_update` | Maximum verified progress on the run and verified update for this import, respectively. |
| Training target | `train/target_update` | Saved planned limit, e.g. 400,000. |
| Best checkpoint | `baseline/best_value`, `baseline/best_update`, `baseline/best_sha256` | Saved checkpoint-selection metadata. |

The shared BLEU keys require the matched protocol, scoring signature, counts, and
evaluation identity. Check the saved Dev scope and reference hashes when
comparing runs. Missing protocol evidence uses `*_unmatched`. Unclassified
failures use `*_legacy_unclassified`. To audit archived failures, enable
`AUDIT_FAILURE_JOURNALS` in the importer or `WANDB_AUDIT_JOURNALS` in Cell 1
(`--audit-journals` on the CLI). The reporter checks the published journal and
identity lock hashes, exact sample IDs, and every failed row against the pinned
decoder errors. Missing, corrupt or unclassified rows stay in the legacy series.
An affected Aux point does not gate a clean ASR point for the same update.

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

The unit conversion uses existing measurements. There is no loss or BLEU
reevaluation, interpolation, or missing-value substitution.

### Migrate the existing Baseline to Schema v2

Open the updated import notebook on a CPU runtime. Keep `SOURCE`, `STATE` and
the W&B entity, project and group from the original import. Preserve the
original `WANDB_EXISTING_RUN_ID` setting. For the registered Baseline it was
blank because the verified lineage derives Run `54aa109cb906644038c08f91`.
Set `SCHEMA_V2_ONLY=True` and inspect cell 3. For a pre-merge run use the
full PR commit SHA in `CODE_REVISION`. After inspection, enable
`SYNC_TO_WANDB` and run cells 1, 2 and 4.

Migration requires the existing idle Run and matching group. It checks
existing keys and stops on a conflict before opening a writer. It appends only
missing Schema v2 metric cells, preserving the old keys and the verified
`train/max_update`. A repeated migration with complete history appends zero
rows. Normal training reporting includes both new and legacy keys through
the same conversion. `dev/main_loss` and BLEU eligibility retain their
existing definitions.

### Audit of the 400k Baseline history

| Series | Measurements | Optimizer updates |
| --- | ---: | --- |
| `train/unit_ce_nats`, `alignment/update/task_loss`, `train/update/learning_rate` | 4,000 each | 100–400,000, every 100 |
| `dev/full/loss` and the three `dev/full/multitask_*_loss` series | 400 each | 1,000–400,000, every 1,000 |
| Existing `dev/main_loss` | 400 | 1,000–400,000, every 1,000 |
| Existing qualified `eval/asr_bleu` | 8 | 1,000; 5,000; 10,000; 25,000; 50,000; 100,000; 200,000; 400,000 |
| `train/window/*` | 0 recovered | No verified matching Fairseq execution log located |

The Baseline Drive folder listing had 926 snapshot folders and six loose
monitor/BLEU files. The receipts at 100, 201,000 and 400,000 updates point to
the monitor, checkpoint, metadata and evaluation artifacts. They contain no
Fairseq process log. The recorded benchmark revision
`4361e55a99c7f4ea3a00c5c138f6820c34a07d06` uses a snapshot file allowlist
without stdout or stderr. The transfer Colab sets the execution log directory
to `/content/alignment-transfer-work/experiment-001/execution` on the runtime
filesystem. Its Drive execution-provenance folder contained release receipts.
No matching saved process log was found in the inspected Drive and local
copies. The stored recipe requests JSON logging at interval 1. This proves
the producer configuration and says nothing about retention of the log.

The checkpointed monitor stores independent logical-update samples at every
100 updates. It has no intervening per-update numerator/sample-size tuples
or native window output. The saved gradient audit keeps a bounded tail of raw
microbatch losses and gradient maxima without the sample sizes and window
coordinates required here. Consequently `train/window/main_loss_fairseq`,
`train/window/loss_fairseq`, all `train/window/multitask_*_loss`, NLL,
perplexity and other window diagnostics are left absent. Full-Dev NLL and
perplexity are also absent. Unit CE uses label smoothing and cannot recover
NLL. There is no verified Baseline logical point before update 100 or Dev
point before update 1,000.

The common loss and LR series are comparable across Baseline, distillation,
annealed distillation and mask distillation where each has a point with
verified provenance at the same update. Old transfer `train/main_loss` may
contain NLL, and the new key cannot recover exact missing transfer CE.
Qualified BLEU is comparable at shared milestones with matching Dev inputs
and evaluator conditions. Baseline Aux-BLEU remains in its legacy series
pending journal verification.

## Identity, retries, and duplicate protection

PR #8 derives the Run ID from the first 24 digits of SHA-256 of the compact
UTF-8 JSON array `["baseline", baseline_run_key, training_fingerprint]`.
The key is the original publication directory relative to MyDrive. For a
relocated copy, supply the original key via `BASELINE_RUN_KEY` (importer) or
`WANDB_BASELINE_RUN_KEY` (training). Group, destination, and launcher revision
are independent of the derived ID.

For the existing transfer-repository Baseline, the import notebook discovers its
published registration under the sibling `stages/register-baseline` directory.
An explicit `REGISTRATION_SNAPSHOT` can select the exact registration receipt
when several candidates exist. The reporter checks the small registration archive
and binds its checkpoint, metadata and recipe hashes to the training receipt.
New native publications include lineage after checkpoint, metadata and recipe
verification during the existing checkpoint validation. For a pinned older
trainer without this evidence, supply a matching verified registration or wait
for a publication by a compatible run. Restore keeps its existing execution pin.

With an explicit existing Run ID the reporter verifies its condition and lineage
then appends with `resume="must"`. Pre-review imports can be adopted by exact
source identity. When adopting such a run, use a new reporter state path if the
old directory still has a v1 binding. Any old points already in W&B stay in
history. New deliveries use the corrected metric names. A requested Group must
match the group returned by W&B before metrics are sent.

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
Review regression tests also cover the two PR #8 Run ID fixtures, registration
and checkpoint lineage, Group-independent routing, adoption of an existing run,
journal classification and durable progress versus the training target.
Authenticated online delivery and actual Colab execution require a W&B secret.
