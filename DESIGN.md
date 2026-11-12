# ja-en-direct-s2st-benchmark design

## Scope

The benchmark compares S2UT, Translatotron 2, ASR→MT→TTS Cascade, and S2T→TTS Cascade for Japanese speech → English speech on the same completed corpus manifest. Corpus generation, reverse translation, UnitY, streaming, multiple speakers, architecture improvements, and automatic launch of 300-hour training are outside the scope.

## Data ownership and roots

Treat `CORPUS_ROOT` as read-only. Write derived units, phonemes, mel features, and fairseq data under `EXPERIMENT_DATA_ROOT`, run artifacts such as checkpoints and predictions under `RUNS_ROOT`, and downloads and caches under `CACHE_ROOT`. Inputs are `production/manifests/releases/accepted.jsonl` and `production/audio/16k/{ja,en}`.

Common manifests preserve the corpus train/dev/test split and exclude QC ASR text as supervision. Verify pair IDs, audio existence, SHA-256, 16 kHz mono PCM16, duration, and required text. Record the input manifest hash in the dataset lock.

## Reproducibility and recovery

The standalone formal S2UT test evaluation is separate from four-system acceptance. Standard Cells 6/7/8 verify the actual saved run, best checkpoint by dev unit loss, and original dependency lock. Use beam 10, the official SHA-pinned LJSpeech unit vocoder, pinned English Whisper, and sacreBLEU. Keep every test ID and score generation and ASR failures as empty hypotheses in corpus BLEU. Select the checkpoint using train/dev only. Atomically save unit shards, 32-item WAV ZIPs, and the ASR journal to Drive for verified reuse after disconnection. Store the ID/ZIP/member/SHA index and ASR text to avoid a large number of loose WAVs. Standard Cell 9 plays saved audio without loading a model or using a GPU. This workflow needs only the S2UT evaluation artifacts and excludes TT2, trained vocoders, train audio, BLASER, and speaker similarity.

Use standard Cells 6/7/8/9 for the standalone Cascade test evaluation. Run the pinned Japanese ASR, JA→EN MT, and English TTS sequentially in separate processes, with no training restore and no reference text in inference. Store stage journals, WAV ZIPs, and the actual dependency manifest under `benchmarks/cascade` without changing existing S2UT evaluation. Use the same ASR-BLEU normalization, denominator, and evaluation ASR as standalone S2UT. Keep Qwen BF16 and fail on incompatible GPUs.

To avoid losing the revision inside the Qwen SDK, fetch and verify a complete snapshot at the pinned revision and give the SDK only its local path. On resume, verify stored ASR/MT read-only before generation. Stop on missing or corrupt artifacts and preserve the journal, pointers, ZIPs, and timing. Check CUDA and BF16 before loading. Synthesize a fixed non-reference preflight sentence with real weights and verify that the WAV is mono, finite, non-silent, and within duration limits before running the test set. Preflight audio is excluded from evaluation IDs and counts. When the initial Cascade stops before TTS load, a reviewed loader overlay can retain the original plan, dependency identity, and ASR/MT journal after verifying the same model and test split. Verify unchanged ASTs for the worker's ASR, MT, and identity functions, and limit changes to preflight synthesis and TTS persistence checks. Do not automatically reuse pre-overlay WAVs or unpublished ZIPs. Record the new execution revision in a separate receipt and report. When the known complete-snapshot fix `c741fca` was already adopted, verify the plan, receipt, Git ancestry, and unchanged TTS loading and decoding arguments before migrating generated WAVs. Record the parent receipt SHA.

Colab training defaults to no time limit (`SESSION_MINUTES=0` / `--session-seconds=0`) and continues to the explicitly confirmed update target. Colab itself can still disconnect. Use atomic snapshots, verification, and resume. Apply the time limit only when the user supplies a positive budget.

Long-running stages write atomically by sample or shard. `--resume` verifies and reuses artifacts. `--overwrite` is required to replace existing outputs. Pin models, fairseq, Python, PyTorch, CUDA, Docker, G2P, and evaluators by immutable revision, version, or digest. GPU OOM and device-side assertions usually stop the process. Sample-specific errors are recorded as failures.

With explicitly enabled adaptive microbatching on Colab, a CUDA OOM before an optimizer update can restore RNG, buffers, and gradients and retry the same samples in smaller batches. An OOM during optimization or a device-side assertion stops the run. Splitting changes statistics such as BatchNorm, so it defaults off and is recorded like mixed precision. GAN steps with two optimizers are excluded.

Resident S2UT FP16 retries the same logical batch only when pinned fairseq detects gradient overflow before the optimizer runs and lowers its loss scale. Restore RNG, model and criterion buffers, gradients, and loss and gradient evidence without advancing the update count. Keep the reduced dynamic scale and the original batch, learning rate, and precision. Stop after the optimizer was called, on an unexplained skipped update, at the minimum scale, or after 32 retries.

Before downloading input data, S2UT checks the real CTC train, backward, and dev paths so evaluation-only dependency failures surface early. Restore missing legacy editdistance from a separate SHA-pinned wheel. Keep the original venv and environment lock intact and record the added version, SHA, and base environment in a separate manifest. Standard Cell 5 verifies input, recipe, optimizer, and monitor in newer unpublished local S2UT state and backs it up to Drive before restoration. Back up after dev and abnormal exit. A forced VM deletion can lose unpublished work. Keep bounded intermediate local checkpoints, last/best, and immutable Drive snapshots.

S2UT recipe checks normalize the launcher module's `argv[0]` to a fixed name. Verify the old lock contents and fingerprint. Inherit the original fingerprint only when the arguments except `argv[0]`, training conditions, and input SHA match exactly. Preserve existing locks and checkpoints. Cell 5 restores and checks saved state, the recipe, and the checkpoint fingerprint before fetching all audio ZIPs. That preliminary check skips WAV presence only. The usual input and payload SHA checks follow retrieval.

Early stopping v3 gives independent anchors to all four TT2 dev losses and five legacy S2UT (`legacy_short`) dev losses. At least 1% improvement on any metric resets shared patience and updates only that metric's anchor. Five consecutive evaluations with insufficient improvement on every metric cause a stop. Evaluate all dev data every 10 updates. During warmup, collect each metric's minimum as its anchor. The shortened recipes have absolute caps of 1000 for TT2 and 3000 for legacy S2UT. Patience resets cannot extend the cap, and a restored run already at its cap performs no updates. Verify and load older v3 S2UT state with cap 1000, changing only cap metadata. Verify v1/v2 history, optimizer, and best checkpoint before adopting v3. Use historical dev minima for anchors, reset the shared counter, and apply v3 to future evaluations. Only explicit adoption can clear an earlier policy's stop. Same-policy resume restores the entire state. Save `policy_history`. Best checkpoint selection remains independent of stopping: minimum dev total loss for TT2 and dev `unit_loss` for S2UT. Model, loss definitions and weights, optimizer, scheduler, batch, validation split, and interval are retained.

The new S2UT `paper_fisher_v1` run uses the original 10000-step warmup and maximum of 400000 updates with no custom early stopping. It starts a new optimizer and history. Record all five dev losses and auxiliary text BLEU on a fixed 500-item dev subset every 1000 updates. During training, also measure audio ASR-BLEU on 100 fixed dev samples at 1k/5k/10k/25k/50k/100k/200k/400k. Generation receives no targets, failures stay in the denominator, and the two BLEU series are saved separately. Best checkpoint selection uses dev `unit_loss`. Model, loss, optimizer, and batch are retained. Document deviations for the Japanese–English corpus, public LJSpeech vocoder, and pinned Whisper ASR. See `docs/COLAB.md`.

All standard Colab training targets share a separate-process VRAM calibration before training. Probe train inputs only and discard probe weights and optimizer state. Training starts at update 0. TT2 selects a physical batch while keeping effective batch 1024. GANs select a physical batch. S2UT `paper_exact` retains the paper's Table 4 budget of 20k tokens per GPU and the official single-GPU accumulation of four. Check that it fits and stop if it does not. S2UT physical splitting is limited to `paper_practical`, with its deviation from the official batching recorded. Save every probe and keep the selected batch fixed during training and resume. On a different GPU, recheck only the saved choice. Numerical equality across hardware is unverified.

Standard preprocessing, training, inference, and evaluation run through Docker Compose. Keep `common`, `fairseq`, `cascade`, and `evaluation` separate and reserve GPU devices explicitly. Docker full training requires the Docker execution marker and actual image digest. The native Colab path uses pinned Python 3.10 and explicit training confirmation.

## Phases

1. Common manifests, HuBERT layer 6 and fairseq joblib k=100 units, reduced units, S2UT fairseq data, and unit vocoder.
2. English phonemes using pinned espeak-ng, a fixed split-independent phoneme dictionary, 80-bin mel features, derived Translatotron 2 data, mel vocoder, and Cascade. The existing fairseq `s2spect2_conformer` lacks the required duration-based core.
3. S2UT and Translatotron 2 training and inference, common test outputs, shared evaluation, and the comparison report.

Run smoke (5–100 sentences), pilot (about 5–10 hours), and full (about 300 hours) in that order. Start full only after smoke and pilot verification and explicit configuration approval.

## Standard prediction and evaluation

Each prediction contains pair/system/run IDs, source and reference, output audio and duration, inference time, RTF, status, and error. Failed samples remain as `status=failed`.

Use the same test split, ASR, text normalization, and references across systems. Record ASR-sacreBLEU, BLASER, ECAPA similarity to source and reference, duration, silence, clipping, ASR success rate, failure rate, RTF, and stage timings. ASR-BLEU must be able to finish independently when BLASER is unavailable.

## Completion criteria

All four systems accept the same Japanese test audio and produce English audio and shared prediction records. Both direct systems can train on the train split and resume from checkpoints. Cascade processes the same test set. BLEU, BLASER, speaker similarity, and RTF are comparable under the same conditions. Record configurations, revisions, and the dataset lock, and prevent test leakage.

## S2UT main baseline

The main task maps Japanese speech to English reduced units from HuBERT Base layer 6 and KM100. Record the extractor revision, layer, k-means hash, and cluster count in the unit and data locks. Recompute consecutive deduplication from the original units for verification. Follow Fisher reduced settings in `examples/speech_to_speech/docs/direct_s2st_discrete_units.md` at fairseq commit `3d262bb25690e4eb2e7d3c1309b1e9c406ca4b99`. Require `source_letter` (encoder layer 6, weight 8), `target_letter` (encoder layer 8, weight 8), and `decoder_target_ctc` (decoder layer 3, weight 1.6). Follow fairseq SingleTaskConfig's one-based layer numbering and pinned decoder CTC state indexing.

The source and target letter CE format, `unicode-codepoint-v1`, uses one code point per label in both languages. Train the SentencePiece Unigram decoder CTC tokenizer on English train text only, with requested vocabulary 1000. Record its model SHA, version, train-text hash, and actual vocabulary size. Validate CTC pieces by re-encoding the same sentence. Reject implicit use of legacy character CTC labels. Collapse ASCII whitespace and preserve Unicode normalization and case. Character acceptance uses a fixed code-point set. The letter dictionary registers characters from all splits and counts frequencies from train only. Dev/test-only characters have zero frequency and keep their labels. Optimizer updates use train only. Require review when TTS input and reference token sequences differ. This check compares text and does not establish spoken content by forced alignment or regenerate audio.

Standard S2UT uses the official Fisher width of 256. Verify the real registry's 12-layer, 4-head encoder and 6-layer, 8-head decoder. The Japanese–English corpus and letter labels, and retention of case and punctuation, differ from Fisher. Use `zero_infinity=false` so impossible alignments do not silently zero the CTC auxiliary loss. Immediately before training, verify IDs, dictionaries, unit ranges, minimum CTC length, runtime architecture, and layer connections.

## Translatotron 2 implementation boundary

The implementation follows the Conformer encoder, LSTM linguistic decoder, shared attention, duration-based NAT synthesizer driven by decoder state and attention context, utterance duration L2 objective, mel reconstruction, and residual refinement in [paper v5 §3](https://arxiv.org/html/2107.08661v5). The paper includes the Conformer. Per-phoneme duration supervision is optional.

The pinned fairseq `s2s_conformer_translatotron2.py` is registered with a Transformer linguistic decoder and TTSTransformerDecoder. It lacks a duration predictor and objective. The previous speech_to_spectrogram criterion also omitted `prev_output_tokens_mt`. Reuse data preparation and use an independent native PyTorch core for training and inference. See `docs/TRANSLATOTRON2.md` for differences. Voice preservation is disabled. Speaker similarity compares to the target reference speaker.

Standard Colab defaults to `paper_exact`, with effective TT2 batch 1024 and S2UT `max_tokens=20000/update_freq=4`. An explicitly selected `paper_practical` creates a separate run. Reuse target 16 kHz/80-bin mel features. Adapt only source features at runtime to 16 kHz/80 bins/125–7600 Hz. Preserve older checkpoint settings. Use Griffin–Lim for TT2 BLEU, and a distinct comparison ID for HiFi-GAN. WaveRNN and MOS are unimplemented. Fetch the official S2UT Code HiFi-GAN by pinned SHA and require duration prediction. Record differences from the paper, unknowns, actual feature settings, optimizer, and unit artifacts in research metadata.

The user approved implementing the TT2 core first on 2026-09-13. The TT2 core and CPU verification preceded real S2UT end-to-end verification. Real-data end-to-end acceptance still applies.

## Portable data and predictions

Colab TT2 training uses prepared mel SHA-256 identities and fetches WAV and mel ZIPs on demand, with up to two transfers and 16 pending prefetch items in training order. Verify ZIPs and individual WAV SHAs before exposing local paths. Preserve effective batch, sampler order, optimizer, and RNG restoration. A legacy monolithic ZIP blocks until its first retrieval completes. S2UT and vocoder paths retain upfront verification of all inputs. Preserve originals and prepared artifacts. Retrieval mode is excluded from training identity.

Keep an existing corpus absolute path. Otherwise resolve its `audio/16k/{ja,en}/...` suffix under `corpus_root` and `corpus_root/production`. Stop on ambiguity and avoid basename search. Save corpus-relative paths in common manifests so they can be resolved via `CORPUS_ROOT` after relocation. Preserve the original corpus manifest. Reprepare environment-dependent fairseq and unit paths after moving.

Intermediate S2UT and mel outputs use `units.jsonl` and `mels.jsonl` with shared IDs and reference information. Convert to the shared `predictions.jsonl` schema after vocoder success. The unit adapter uses the pinned fairseq CodeGenerator's trained duration predictor. The mel adapter requires exact parameter, log, and normalization agreement. Record checkpoint, config, and input SHA-256 values in the vocoder lock. S2UT timing records batch wall time averaged over samples. Check `timing_scope` before comparing with Cascade.

See [IMPLEMENTATION_STATUS.md](docs/IMPLEMENTATION_STATUS.md) for verification coverage and remaining work.

## Experimental TT2 stabilization recipe (2026-10-15)

Following the user's approval, `stabilized_seconds_v1` is available for new `paper_practical` TT2 runs. The duration head and utterance-length MSE use seconds, converted to frame coordinates using the target hop. Initialize at 80 ms per token. Start with peak learning rate 0.0003 and warmup 400. Retain Fisher dimensions, loss weights 1/10/1, Adam/L2, effective batch 1024, monitoring and best selection, and cap 1000. The original paper's unpublished implementation details and translation quality remain unverified. Legacy frame mode stays the default and retains its checkpoint semantics. Reuse prepared artifacts and standard Cell 5 run selection, environment/optimizer/RNG restore, and VRAM calibration. Results report the actual verification scope and measured values.

### Evaluation-only postnet calibration (2026-10-17)

`stabilized_pooled_v2` uses a separate new `paper_practical` run with the v1 objective, optimizer, and logical batch. Before validation, select 128 train IDs in SHA-256 order, or all train IDs if fewer. Run teacher-forced eval in batches of four. Temporarily store raw mel on CPU and replay only the postnet. Calibrate its five layers in order with pooled means and unbiased variances over valid frames. Fitting uses train data only. Restore all training buffers, RNG, and module modes after calibration and validation, including on exceptions. Save statistics in the native checkpoint with the update count, parameter SHA, and train IDs. Resume optimization using native buffers. Inference verifies and applies matching calibration statistics. Reject uncalibrated intermediate last checkpoints and use the verified best. Calibrate at the final update and retain the monitor interval, normally 10. Old recipes and dev history keep their original semantics. Use standard Cell 5 to start or resume. The first version supports single-process FP32. The small local trial leaves full 128-sample and full-corpus translation quality, generalization, and paper parity unverified.

## S2T→TTS baseline

`system_id=s2t_tts` uses Whisper large-v3 Japanese→English translation and shares QwenTTS. ASR→MT→TTS retains turbo/transcribe/NLLB. Record immutable revision, language, and task in the config and shared journal identity. Store S2T text in `s2t_en_text` and timing in `s2t_seconds`. Verify all four systems against the same common test split and evaluation config. Colab uses native subprocesses in pinned Python 3.10.

## Cascade TTS generation policy audit (2026-10-24)

Review rationale, suitability, evidence, and verification scope for changes from official settings. Greedy Qwen decoding lacked suitability checks. The user approved a separate evaluation using the pinned model's official sampling. The per-input seed, token budget equivalent to the existing 120-second limit, and cooperative time guard are documented deviations. Verify audio EOS in the actual main talker before decode. Preserve old TTS results and reuse SHA-pinned ASR/MT read-only. See the [audit](docs/CASCADE_TTS_AUDIT.md) for details and pending GPU checks.

## Two-system comparison on custom text (2026-10-24)

Standard Cell 10 generates shared Japanese audio with Qwen and sends it to Cascade and the selected S2UT, TT2, or S2T→TTS model. Cascade MT provides a machine-generated reference used only for scoring. Reuse the source-only frontend, dev-best checkpoint, and existing synthesis path without creating new training labels or manifests. Save results under `interactive/` and verify SHA per stage on resume. GPU stages run sequentially. Display diagnostic translation time excluding shared input synthesis, model retrieval/loading, and evaluation ASR, with process RSS and Torch allocator peaks. Treat saved measurements from different sessions or hardware as distinct. See the [comparison guide](docs/INTERACTIVE_COMPARISON.md).
