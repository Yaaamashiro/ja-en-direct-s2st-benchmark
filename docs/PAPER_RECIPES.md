# Paper-aligned training on the 16 kHz corpus

The standard Colab notebook defaults to `paper_exact`, TT2 Fisher, Griffin–Lim for BLEU, and S2UT Code HiFi-GAN fitted on the target corpus's English train audio. **Exact mode fixes the main published recipe settings.** The independent implementation and different corpus still require separate verification of numerical results. TT2 keeps the published effective batch size, and S2UT keeps the per-GPU token budget. Reuse the existing corpus, 16 kHz/80-bin target mel, and extracted units. The small two-update TT2 smoke run checks pipeline wiring.

Select `TT2_RECIPE=stabilized_seconds_v1` for an explicit stabilization experiment. A fresh practical run uses duration MSE in seconds, initialization at 80 ms per token, peak LR 0.0003, warmup 400, and effective batch 1024. Duration units and initialization in the paper are unknown. Standard Cells 1 → 5 save this recipe under a separate run name. Legacy frame recipes and checkpoints retain their original meaning. See [setup and resume](COLAB.md#experimental-tt2-stabilization-recipe-new-runs-only).

## Settings

| Setting | Standard value | How changes are handled |
| --- | --- | --- |
| REPRODUCTION_MODE | paper_exact | Explicitly choose paper_practical for a separate run. |
| TT2_BATCH_SIZE | 8 | Physical batch with calibration off. Exact mode uses update_freq=1024/batch. |
| TRAIN_CALIBRATE_BATCH | True | Shared pre-training probe. S2UT exact checks whether the official batch fits. Actual training starts from initial weights. |
| TRAIN_CALIBRATION_MAX_BATCH | 1024 | Search ceiling, except in S2UT exact. Actual capacity depends on the probe. |
| TRAIN_VRAM_RESERVE_RATIO | 0.10 | Safety margin against measured peak VRAM, at least 5%. |
| TRAIN_CALIBRATION_STEPS | 3 | At least three real forward/backward/optimizer steps per candidate. |
| PRACTICAL_UPDATE_FREQ | 1 | Practical only. Effective batch = physical batch × update_freq × world_size. |
| TT2_VOCODER_MODE | griffin_lim | hifigan is a separate comparison run that requires mel vocoder training. |
| S2UT_VOCODER_MODE | paper | New paper_fisher_v1 uses the official public LJSpeech weights. trained selects separately fitted weights. |
| S2UT_RECIPE | paper_fisher_v1 | New run with 10000 warmup, a 400000-update cap, and no early stopping. Use legacy_short to resume an old run. |
| TOTAL_UPDATES | Researcher-specified | TT2 Appendix A does not specify the total updates. The S2UT recommendation of 400000 does not launch automatically. |
| S2UT_TOTAL_UPDATES | 400000 | S2UT only. RUN_TRAINING and CONFIRM_TRAINING default to off. Shorter runs are recorded as below the paper's training budget. |
| TRAIN_CALIBRATION_OBJECTIVE | throughput | Select by probe speed among safe candidates. capacity selects the largest fit. S2UT exact only checks fit. |
| TRAIN_SAVE_INTERVAL | 1 | Save locally every update because exact updates can take a long time. Drive publication has a separate time interval and capacity limit. |
| TT2_MONITOR | True | Custom dev-loss monitoring, best-checkpoint retention, and early stopping. The paper selects by dev BLEU. Optimizer, batch, and warmup are retained. |

`paper_exact` forbids adaptive batch splitting and precision changes during training. TT2 explicitly measures VRAM before training to select and fix the physical batch and accumulation. S2UT exact only checks whether the official token-budget batch fits, with no splitting or expansion. OOM stops execution. Physical batch 8 has no guaranteed fit. A new TT2 run can reduce physical batch while retaining effective batch 1024, with different microbatch-local BatchNorm statistics.

Practical runs normally keep Fisher, LR, warmup, and L2. An explicit `S2UT_WARMUP_UPDATES=400/500/1000` records the deviation from the paper's 10,000 updates. Short S2UT settings remain available as `legacy_short`, with `paper_fisher_v1` the new-run default. See `RESUME.md` for shortening a saved TT2 run.

TT2 and legacy S2UT short runs use 400 warmup updates, dev evaluation every 10 updates, patience 5, and a 1% minimum improvement. V3 tracks independent anchors for all four TT2 or five S2UT dev losses. Five consecutive evaluations without a 1% improvement on any metric trigger a stop. These custom recipes use a shorter budget and loss-based selection. New S2UT records full dev losses and auxiliary text BLEU on 500 fixed samples every 1000 updates, and audio ASR-BLEU on 100 fixed samples at selected updates. Best selection uses dev `unit_loss`. BLEU is diagnostic. See [COLAB.md](COLAB.md) for startup, resume, evaluation cost, and ASR and vocoder differences.

## Shared VRAM calibration (Colab Cell 5)

TT2, S2UT, and trainable mel and unit HiFi-GAN share the probe and persistence mechanism. Griffin–Lim, the official pretrained vocoder, and inference-only Cascade are outside calibration.

Except for S2UT exact, probe 1, 2, 4, 8, and so on until the cap, OOM, or insufficient margin. Capacity selects the largest passing physical batch. Throughput, the default, selects by examples per second among passing candidates. Exclude the first optimizer allocation step from speed scoring and include it in safety checks. Prefer the larger batch within 3% of the fastest. A brief worst-padding probe cannot guarantee end-to-end throughput or occupancy. S2UT practical and GAN probe between the last passing and failing candidates by binary search. TT2 tests only divisors of the effective batch, which are powers of two for 1024.

- Test each candidate from initial weights in a fresh process and include optimizer allocation. Actual training starts from update 0 in a separate process. Probe checkpoints are discarded.
- TT2 probes a conservative padding workload combining the longest train source, target, and phoneme sequences. Synthetic probe combinations are never training pairs. Exact keeps physical batch × accumulation at 1024. Practical keeps its configured effective batch.
- S2UT probes regular train token-budget batches with high attention load, long audio, or many examples. Exact performs one unsplit trial with max_tokens=20000 and update_freq=4. It ignores MAX_BATCH and stops if the batch does not fit. A recorded `selected=0` means official batching with no extra cap. Practical fixes a physical split cap while preserving the original token budget, update frequency, and sample order. Splitting is a deviation from the official execution mode. Variable-length token-budget batches vary in sample count.
- GAN probes both generator and discriminator optimizers on long train audio. The selected physical batch is the number of utterances per update, with no new accumulation.
- Atomically save GPU/PyTorch/CUDA, peak allocated, reserved and total memory, timings, work counts, and selection to `checkpoints/<RUN_NAME>/batch-calibration.json`. Resume reuses completed candidates and repeats pending ones. Training snapshots include the record.
- Reuse the selection on resume. On another GPU, check only whether the saved choice fits and stop if it does not.
- Calibration applies to new runs. Settings or data changes require a new run name. Prepared mel and units can be shared.

Calibration uses GPU time. A VM disconnection can lose an unpublished trial. A short probe cannot prevent every later OOM. Dictionary, input, and numerical errors stop the run with their actual cause. Training OOM preserves the last complete snapshot. CPU tests cover selection, splitting, persistence, and resume.

TT2 loads one physical batch at a time and weights accumulated losses by valid frame, phone, or utterance counts. Partial gradients are not checkpointed. Resume from the last complete Drive snapshot and recompute unsaved updates. `SESSION_MINUTES=0` normally allows training to the update target. With a time cap, a long exact update can exceed the grace period and be lost. Colab's own disconnect and execution limits apply independently.

## Training conditions

The generated TT2 command includes the following when physical batch is eight.

```text
--model-size fisher --reproduction-mode paper_exact
--learning-rate 0.0042 --warmup-updates 10000 --l2-regularization 0.000001
--batch-size 8 --update-freq 128 --vocoder-mode griffin_lim
```

S2UT uses pinned fairseq's `s2ut_transformer_fisher`. Verify the encoder's 12 layers, width 256, and four heads and the decoder's six layers, width 256, and eight heads against the code.

```text
--arch s2ut_transformer_fisher --lr 0.0005 --warmup-updates 10000
--warmup-init-lr 1e-7 --adam-betas (0.9,0.98) --clip-norm 10.0
--max-tokens 20000 --update-freq 4 --fp16
```

S2UT Appendix A, Table 4 specifies **20k max tokens per GPU and four GPUs**. Official single-process `update_freq=4` accumulates the equivalent budget, with numerical equality to four-GPU execution unverified. Exact mode rejects additional utterance caps, fixed splitting, online adaptation, and precision changes. Previous split S2UT exact runs require a fresh run under the new policy. Batch size is a variable token budget.

Keep HuBERT layer 6, k=100, the pinned model revision and k-means SHA, reduced units, source and target letter CE, and decoder CTC. CTC uses SentencePiece Unigram with a requested vocabulary of 1000 as in paper §4.2. Fit on English train text only. Small smoke corpora may produce fewer pieces, so record actual vocabulary size. Store model, train-text hash, SentencePiece version, and SHA-256 in the prepared lock. Letter and CTC dictionaries are distinct. Re-encode the same sentence to verify CTC pieces. Register train, dev, and test code points for source and target letters. Model optimization uses train only. Retain train frequencies and use serialized count 1 for held-out-only characters, listing them in the lock. The paper does not specify the splits used to build the letter dictionary, so this registration is documented as an implementation choice. Encode unknown CTC pieces as `<unk>` and record counts by split.

Migrate older completed 4B data to `fairseq-unigram-allchars-v1` with resume support. Reuse unit TSVs and configuration and generate letter and CTC labels. Preserve old fairseq data, mel, HuBERT units, checkpoints, and `fairseq-unigram-v1`. S2UT training uses a new recipe-v3 run name. An older checkpoint cannot resume with a different CTC model.

## Features, synthesis, and evaluation

- TT2 source features use 16 kHz/80 bins, 25 ms window, 10 ms hop, 125–7600 Hz, and utterance CMVN at training and inference time. Prepared audio and target mel are preserved.
- Target metadata records 16 kHz/80 bins, 25 ms/10 ms, 0–8000 Hz, and natural-log magnitude mel. The paper uses 24 kHz/128 bins, 50 ms/12.5 ms, and 20–12000 Hz.
- G2P uses eSpeak NG 1.52.0. The paper uses Google's G2P and different data and dictionary. Voice preservation is disabled.
- Griffin–Lim follows the BLEU path in TT2 §5.1, with a Slaney mel pseudoinverse, magnitude power 1, Hann window, center enabled, momentum 0, 32 iterations, and a fixed seed per utterance. Google's iteration count and phase initialization are unknown.
- HiFi-GAN remains an alternative under a comparison ID that includes vocoder mode. Evaluate the same TT2 checkpoint with either vocoder. WAVs, RTF, and results are specific to the mode.
- The S2UT paper vocoder fetches official LJSpeech HuBERT-100 checkpoints and config over HTTPS, verifies pinned SHA-256, and requires duration prediction. Initial retrieval requires network access and review of the weight provider's usage conditions.
- The trained alternative fits on the target corpus's English train audio and original units, with KM100 embeddings, duration predictor (128/kernel 3/dropout 0.5), log-duration MSE weight 1, and MPD/MSD, GAN, feature matching, and mel losses. Mel frontend, crop, physical batch, and LR schedule differ from the original Fisher vocoder. Document these deviations and the provenance of distributed paper-mode weights in metadata.
- The shared primary metric is Whisper ASR-sacreBLEU. Numerical equivalence to the paper's Google ASR BLEU is unverified. MOS and WaveRNN are outside this implementation. Speaker similarity measures similarity to the target reference.

The independent PyTorch model also leaves numerical parity of initialization, microbatch BatchNorm, and dropout unverified. Matching learning rates and tensor shapes alone does not establish paper-quality results.

## GPU implementation and measurements

TT2 draws uniform integer SpecAugment samples from checkpointed CPU RNG and batches teacher-forced prenet frames. It retains autoregressive LSTM and BatchNorm. Its RNG sequence differs from the old CUDA RNG and sequential dropout, so engine-v2 is part of the run fingerprint. Gradient and parameter checks remain and synchronize together. S2UT aggregates gradient auditing on GPU and retains the last 128 losses in memory. Fairseq logs retain all losses.

HuBERT stops at L6 without changing earlier hidden states. Small CPU models verify exact equality before and after the change. Prefetch up to 256 items or an estimated 128 MiB and group identical audio lengths without padding or unit reordering. Distinct lengths stay at batch 1.

Performance logs report VRAM, examples per second, data wait, and nvidia-smi GPU utilization samples when available. Unavailable is null. Utilization samples are instantaneous. A100 timing, long-run OOM behavior, and quality need separate Colab verification.

## Persistence and existing experiments

`research-metadata.json` stores actual model dimensions, optimizer, precision, batch, feature and dictionary/G2P settings, data fingerprint, repository revision, unit artifacts, and updates. After inference, `vocoder-lock.json` records actual weight, config, and input SHAs. Griffin–Lim has null weight SHA. Snapshots include metadata.

Typical run names are `tt2-training-<experiment>-paper_exact-recipe-v2` and `s2ut-training-<experiment>-paper_exact-recipe-v3`. Preserve prepared data and keep older batch and LR checkpoints with their recipes. Resume new-recipe runs using the same experiment, output path, and settings. Practical mode starts a separate run.

An existing Colab experiment may pin old code in `repository-revision.txt`. Updating the notebook leaves that pin in place. Explicitly update the revision while preserving existing snapshots and run settings. The original release required separate user approval before pushing.

After publication, obtain the complete commit SHA and run this migration. Existing snapshots are retained. The old model does not resume training under the new recipe.

```python
import sys
NEW_REVISION = 'REPLACE_WITH_THE_PUBLISHED_40_CHARACTER_SHA'
ensure_workspace()
run('git', '-C', REPO, 'fetch', 'origin', NEW_REVISION)
run('git', '-C', REPO, 'checkout', '--detach', NEW_REVISION)
run(sys.executable, REPO / 'scripts/colab/use_paper_recipe.py',
    '--persistent', PERSISTENT, '--revision', NEW_REVISION, '--overwrite',
    '--separate-recipe-v3-runs')
# Run Cells 1 → 5 in the updated notebook. The initial setup rebuilds automatically.
# Resume any unfinished 4A or 4B stage first.
```

`--separate-recipe-v3-runs` preserves old paper runs and is required when migrating them. Experiments that already use recipe-v3 need a separate code/run migration. Data, old configs, and checkpoints are retained.

## Sources and verification scope

`stabilized_pooled_v2` is another explicitly experimental fresh practical TT2
recipe, not a correction claimed by the paper. It inherits seconds-v1 training
and substitutes train-only pooled postnet running statistics for evaluation and
inference. Native optimizer/model buffers are retained for exact training resume;
the checkpoint embeds separately bound evaluation statistics. Train subset128,
batch4, five layerwise valid-frame pooled moments, single-process FP32, hard
cap1000. No old run's history, best selection metric or paper-exact defaults are
reinterpreted. Small local evidence is not a full-corpus translation-quality result.

[TT2 Appendix A and §5](https://arxiv.org/html/2107.08661v5), [S2UT Appendix A and Table 4](https://aclanthology.org/2022.acl-long.235.pdf#page=12), and the [pinned fairseq Fisher recipe](https://github.com/facebookresearch/fairseq/blob/3d262bb25690e4eb2e7d3c1309b1e9c406ca4b99/examples/speech_to_speech/docs/direct_s2st_discrete_units.md).

CPU tests cover generated commands, pinned fairseq architecture functions, real Adam updates and matching resume under sequential and standard accumulation, real Griffin–Lim synthesis and reuse, and SHA pinning. Drive and Colab real-data metadata, full GPU training, translation quality, and MOS remain unverified.
