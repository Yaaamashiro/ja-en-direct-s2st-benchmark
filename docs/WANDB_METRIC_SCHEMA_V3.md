# S2UT comparison metric schema v3

This schema applies to the registered 400,000-update Baseline and the three new
transfer conditions (`distillation`, `annealed_distillation`, and
`mask_distillation`). The Baseline import republishes measured values from its
existing run. Its training fingerprint, checkpoint, recipe, and data are the
same. The historical transfer histories without an exact per-update Unit CE
cannot supply the v3 training series.

## Loss definitions

For each criterion call \(i\), let \(C_i\) be the sum of label-smoothed Unit CE,
\(N_i\) the main criterion sample size, \(A_{i,k}\) the unweighted auxiliary
loss sum for task \(k\), and \(M_{i,k}\) that task's sample size. Compute:

```text
Main       = sum_i C_i / sum_i N_i
Aux(k)     = sum_i A_i,k / sum_i M_i,k
Task       = sum_i (C_i + 8 A_i,source + 8 A_i,target + 1.6 A_i,ctc) / sum_i N_i
Total_base = Task
```

Unit CE uses label smoothing 0.2. Use the raw criterion numerators and sample
sizes with the registered recipe (`sentence_avg=false`, `rdrop_alpha=0`). Sum
them over **all** microbatches and workers for one logical optimizer update
(Train), or all **1,453** Dev samples for one full validation pass (Dev).
Use each auxiliary task's own denominator for its separate loss. The weighted
Task numerator is formed before main-denominator normalization. An arithmetic
mean of per-microbatch losses, a sum of normalized auxiliary losses, and Unit
NLL each represent a different quantity.

For Transfer, capture `train/main_loss`, `train/task_loss` and
`train/aux/*` from the task criterion before adding the alignment term.
`train/total_loss` comes from the actual combined optimization objective at
that update, with the alignment term and its weights at their applied scales.
Its unit is the objective value of that recipe. Dev uses the same shared
criterion and the same full split for all four conditions. Evaluate the task
loss there without the training-only alignment term, in teacher-forced mode.
Record the scheduler LR after the successful optimizer update. Skipped steps
and missing or unqualified measurements produce no metric value.

The archived Baseline [monitor reducer][monitor], [Fairseq criterion][criterion]
and [training recipe][recipe] were checked against the registered experiment.
The monitor reads the label-smoothed main numerator separately from the NLL,
the weighted objective numerator from the criterion, and each auxiliary
numerator and sample size from the multitask results. Its stored losses are
`sum / size / ln(2)`. The importer therefore multiplies the saved normalized
bits values by `ln(2)` once, using their full stored precision. It requires no
original microbatch numerators. The [shared fixture][fixture] uses uneven
microbatch sizes and different CE and NLL values.

## Metric names and Baseline sources

The source field below is in `training-monitor.json`. The same values are
cross-checked in `training-monitor.csv` when present, with the split prefixed
to the CSV field. LR is shared between both formats.

| v3 metric | Baseline source field | Definition and unit | Cadence and verified points |
| --- | --- | --- | --- |
| `train/main_loss` | `train.unit_loss` | Main, nats | 100 updates, 4,000 |
| `train/task_loss` | `train.loss` | Task, nats | 100 updates, 4,000 |
| `train/total_loss` | `train.loss` | Optimized objective, nats for Baseline | 100 updates, 4,000 |
| `train/aux/source_letter_loss` | `train.source_letter` | Aux(source), nats | 100 updates, 4,000 |
| `train/aux/target_letter_loss` | `train.target_letter` | Aux(target), nats | 100 updates, 4,000 |
| `train/aux/decoder_target_ctc_loss` | `train.decoder_target_ctc` | Aux(ctc), nats | 100 updates, 4,000 |
| `train/learning_rate` | `learning_rate` | Post-update scheduler LR | 100 updates, 4,000 |
| `dev/main_loss` | `dev.unit_loss` | Main over full Dev, nats | 1,000 updates, 400 |
| `dev/total_loss` | `dev.loss` | Task over full Dev, nats | 1,000 updates, 400 |
| `dev/aux/source_letter_loss` | `dev.source_letter` | Aux(source) over full Dev, nats | 1,000 updates, 400 |
| `dev/aux/target_letter_loss` | `dev.target_letter` | Aux(target) over full Dev, nats | 1,000 updates, 400 |
| `dev/aux/decoder_target_ctc_loss` | `dev.decoder_target_ctc` | Aux(ctc) over full Dev, nats | 1,000 updates, 400 |

BLEU sources are the actual saved `training-bleu.json` and `training-bleu.csv`,
plus the receipt-matched metadata and optional journals. The common milestones
are **1k, 5k, 10k, 25k, 50k, 100k, 200k, 400k**. Score, failure count and
sample count are included together only for a qualified measurement.

| v3 metric | Saved field | Definition and unit | Verified points |
| --- | --- | --- | --- |
| `eval/aux_bleu` | `aux_bleu` | Corpus SacreBLEU, 0–100 points, fixed 500 Dev samples | 0 |
| `eval/aux_failures` | `aux_failures` | Verified auxiliary generation failures, count | 0 |
| `eval/aux_samples` | `aux_samples` | Scoring population, count | 0 |
| `eval/asr_bleu` | `asr_bleu` | Audio-to-ASR corpus SacreBLEU, 0–100 points, fixed 100 Dev samples | 8 |
| `eval/asr_failures` | `asr_failures` | Verified audio generation failures, count | 8 |
| `eval/asr_samples` | `asr_samples` | Scoring population, count | 8 |

Aux decodes the encoder-attached target-letter decoder greedily, with a
256-token cap. ASR-BLEU uses Unit beam search (beam 10, max length
\(1 \times\) source length + 200), official LJSpeech Unit HiFi-GAN and pinned
Whisper large-v3-turbo (revision `60be3615a4d667e1258e8ad29130467587c489aa`).
Both use source audio only, the seed-1 fixed hash order of Dev IDs, the
registered references and benchmark English normalization (lowercase, NFKC
and punctuation removal except apostrophes). The ASR subset is the first
100 of the 500. Use the existing SacreBLEU 2.5.1 signature
`nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.5.1`.
Compare only points with matching subset IDs, references, decode settings and
evaluator. The [existing BLEU validator][bleu] verifies the stored contract.
With nonzero failures it also requires matching journal records for each ID.
Only recognized decoder failures with empty hypotheses may remain in the fixed
scoring population. Unclassified, vocoder and evaluation ASR errors do not
qualify. The importer does not run the evaluator.

The certified local copies yield **30,024 metric values across 4,000 updates**.
Train spans updates 100–400,000 and Dev spans 1,000–400,000. All eight ASR
points had zero failures. The 400 historical Aux observations had unclassified
failures in the available evidence. V3 omits them. If exact matching journals
are available, enable the optional audit to qualify individual milestones.

For the registered 400k Baseline, each Train metric must have measurements at
exactly `100, 200, …, 400000`, and each Dev metric at exactly
`1000, 2000, …, 400000`. Each of the eight ASR milestones must have the
qualified score, failure count and sample count. The importer compares each
metric's actual update set with this fixed schedule. An equal point count
with a shifted update fails validation. Invalid required values and
missing or unexpected coordinates are reported with the metric and update.
Cell 3, both history audits and sync validate the source first, so an
incomplete source cannot produce `VERIFIED` or open a W&B writer. Aux-BLEU
is optional and contributes only complete, qualified triplets.

Section [Audit of the 400k Baseline history](WANDB.md#audit-of-the-400k-baseline-history)
records the prior search for native Fairseq logs and snapshot evidence. The
receipts and archived snapshot rules do not preserve the complete native
process logs. Sampled updates cannot reconstruct a Fairseq weighted window.
Unit NLL, perplexity, gradient norm/clipping, FP16 overflow and attention
diagnostics are absent from the verified series. The saved gradient audit is
a bounded microbatch diagnostic without the matching denominators or window
coordinates. Missing values are left absent without zero-fill, interpolation,
estimated numerators or a rerun of the original experiment.

## Publication, identity and audit

| Field | Value |
| --- | --- |
| W&B entity | `yaaamashiro-national-institute-of-technology-okinawa-college` |
| Project | `direct-s2st-v3` |
| Group | `experiment-001/transfer-fixed-v2` |
| Display name and schema | `baseline`, 3 |
| Baseline publication key | `alignment-transfer-s2ut-data/experiment-001/baseline` |
| Baseline training fingerprint | `fe8b63b2f57c5c15f24f03ae9af4ec083ccb1eff6f1c897ac707b83e40b69e24` |
| Registered v3 Run ID | `5db658f8baac28cbe211f3b8` |
| Original Run ID in `direct-s2st` | `54aa109cb906644038c08f91` |
| Reporter state | Separate `wandb-baseline-v3` sibling directory |

V3 derives its ID from the first 24 hex digits of SHA-256 of compact UTF-8
JSON `["baseline",3,"direct-s2st-v3",baseline_run_key,training_fingerprint]`.
It requires an existing destination project and checks the group and complete
v3 binding. The binding records the original Run ID, fingerprint and hashes
of the original checkpoint/metadata/recipe snapshot members. Report bytes are
checked against the 400k snapshot receipt. The registration receipt verifies
its record and ties its checkpoint, recipe and metadata hashes to that
publication. Raw reference text and sample IDs are not uploaded. Use
[the mapping code][mapping] and [the delivery code][delivery] for exact keys.

Schema v1/v2 `dev/main_loss` is in bits. Its name is reused in v3 with nats,
so all v3 metrics belong in the new project and new runs. Existing legacy
bindings cannot be adopted for v3. The unique v3 run, lock, binding and
outbox permit interrupted and repeat syncs; run a single writer per run.
Preflight compares remote metric cells by `(update, key)`; it stops on
conflicting values, unidentified rows or duplicates. It appends missing cells
with a monotonic W&B `_step` and uses logical `update` for every chart.
An uncertain write is reconciled from unsampled remote history. A finished
upload awaiting W&B indexing produces `ACK_PENDING` and is polled on retry
without resubmitting it. The read-only audit also reports incomplete or
duplicate history by metric.

`train/target_update`, `train/max_update`, and `train/durable_update` are
Summary fields. The recipe target, latest confirmed optimizer update and
latest verified published update are all 400,000 for this Baseline. None is
written as a history point. Source hashes, receipt and scoring conditions are
retained as provenance.

## CPU Colab import

Open [Baseline W&B Import](../notebooks/baseline_wandb_import.ipynb) on a CPU
runtime. Create or confirm the new W&B project before the import. Use the
following Cell 1 settings:

| Setting | Value |
| --- | --- |
| `SOURCE` | `/content/drive/MyDrive/alignment-transfer-s2ut-data/experiment-001/baseline` |
| `STATE` | `/content/drive/MyDrive/alignment-transfer-s2ut-data/experiment-001/wandb-baseline-v3` |
| `WANDB_ENTITY` | `yaaamashiro-national-institute-of-technology-okinawa-college` |
| `WANDB_PROJECT` | `direct-s2st-v3` |
| `WANDB_GROUP` | `experiment-001/transfer-fixed-v2` |
| `METRIC_SCHEMA` | `3` |
| `WANDB_EXISTING_RUN_ID`, `CODE_REVISION` | Blank (derive the v3 ID and use the pinned reporter) |
| `SCHEMA_V2_ONLY`, `SYNC_TO_WANDB` | `False` initially |

Leave `REGISTRATION_SNAPSHOT` blank if there is a single published receipt in
the sibling `stages/register-baseline` tree. Otherwise point it at the matching
published `snapshot.json`. For a relocated local copy specify the original
`BASELINE_RUN_KEY`. Leave `AUDIT_FAILURE_JOURNALS=False` to use the certified
measurements above. If complete receipts and journals are available, enable
the option before the first sync and keep the same evidence settings when
resuming or auditing.

1. Run Cells 1–3. Check update 400,000, the fingerprint and the measured
   counts in the local source audit.
2. Grant the notebook access to the Colab secret `WANDB_API_KEY`. Run Cell 3A
   for the read-only remote audit. `NOT_IMPORTED` with the expected missing
   series is normal on the first run.
3. Set `SYNC_TO_WANDB=True`, rerun Cell 1, then run Cell 4 to publish. For
   `ACK_PENDING`, repeat Cell 4 with the same settings until it says
   `SYNCED`. After a disconnected runtime, run the preceding cells again.
4. Run read-only Cell 5. Check `VERIFIED`, empty missing and duplicate maps,
   and all three Summary values at 400,000. Inspect the `baseline` run in
   the new W&B Workspace using `update` as the X axis.

The CPU notebook mounts Drive and reads the published reports. Its SDK and
command logs are isolated from training. Sync writes the new sibling state and
the selected v3 W&B Run. The audit uses temporary local SDK state.

## Reporter release

The Schema v3 Colab importer pins the CPU reporter to Git revision
`6d242da09902bc15c0cff85f6eb331e5274d0731`. The revision contains the
metric mapping, completeness validator and fixture for the transfer
implementation. The notebook still defaults to Schema v2. Select v3
explicitly to use this release.

[monitor]: https://github.com/Yaaamashiro/ja-en-direct-s2st-benchmark/blob/4361e55a99c7f4ea3a00c5c138f6820c34a07d06/src/direct_s2st/s2ut/monitor.py
[criterion]: https://github.com/facebookresearch/fairseq/blob/3d262bb25690e4eb2e7d3c1309b1e9c406ca4b99/fairseq/criterions/speech_to_speech_criterion.py
[recipe]: https://github.com/Yaaamashiro/ja-en-direct-s2st-benchmark/blob/4361e55a99c7f4ea3a00c5c138f6820c34a07d06/configs/s2ut/train.yaml
[fixture]: ../tests/fixtures/wandb_metric_schema_v3.json
[mapping]: ../src/direct_s2st/baseline_wandb_v3.py
[delivery]: ../src/direct_s2st/baseline_wandb_delivery.py
[bleu]: ../src/direct_s2st/baseline_wandb_evidence.py
