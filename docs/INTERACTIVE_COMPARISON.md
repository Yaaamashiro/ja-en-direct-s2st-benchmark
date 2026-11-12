# Compare two S2ST systems on a custom sentence (Cell 10)

## Workflow

1. Connect to a BF16-capable GPU and open the standard notebook with compatible code. Configure it in Cell 1.
2. Use the same experiment name and run Cell 1 → **Cell 10**. This workflow uses saved artifacts and skips Cells 4–8 and retraining.
3. Enter one Japanese sentence in `INTERACTIVE_JA_TEXT` in Cell 10. Use 1–500 characters with no line breaks. Start with a short sentence.
4. Set `INTERACTIVE_MODEL` to `s2ut`, `tt2`, or `s2t_tts`. S2UT and TT2 require saved runs. An empty `INTERACTIVE_RUN_NAME` uses the existing model-specific selection pointer. If selection is ambiguous, the cell lists candidates and stops. You can also name a run explicitly.
5. Review the Japanese audio from Qwen, the English audio and ASR transcripts from Cascade and the selected system, diagnostic BLEU, and the performance table.

Both systems receive the same Japanese WAV. Qwen CustomVoice synthesizes it using `language=Japanese`, Ono_Anna, the pinned SHA, official sampling, and a per-sentence seed. `scipy.signal.resample_poly(2/3)` converts the 24 kHz audio to 16 kHz to match training inputs. The cell saves both WAVs and the conversion method. This comparison uses one synthetic utterance. Performance on natural speech or the original test distribution needs a separate evaluation.

## Comparison and automatic reference

```
Japanese input text → Qwen Japanese TTS → shared 16 kHz WAV
                                          ├─ Cascade: Whisper Japanese ASR → NLLB English translation → Qwen English TTS
                                          └─ Selected system: S2UT / TT2 / Whisper S2T → Qwen English TTS
Each English output → pinned English ASR → sentence BLEU using Cascade's NLLB translation as the automatic reference
```

**Cascade provides a machine-generated reference.** BLEU here measures agreement with Cascade. A model that repeats a Cascade mistranslation can score well. Cascade's own ASR-BLEU mainly measures agreement across TTS and evaluation ASR. Sentence BLEU is sensitive to wording and sentence length. Keep it separate from formal test BLEU.

The automatic reference is used only for scoring. The selected system receives Japanese audio and its saved model. It has no access to Cascade ASR or MT text, gold Japanese text, or English targets. Evaluation ASR does not receive the reference as a prompt.

## Model settings

| Component | Settings | Rationale and scope |
|---|---|---|
| S2UT | Saved dev-best, existing fairseq frontend and non-training transforms, beam 10, max_len_a=1, FP32 inference | Uses the evaluation settings from `generation_command` and the official dataset's source-only path. No dummy targets, new dictionaries, or train/dev/test TSVs. |
| Unit synthesis | SHA-verified official LJSpeech Code HiFi-GAN with trained duration predictor | Uses the existing paper vocoder. The voice resembles LJSpeech, so this setup does not measure source-speaker preservation. |
| TT2 | Saved dev-best and calibration metadata, checkpoint-bound source frontend, max_phones=400, max_frames=3000, beam=1 | Uses the existing native inference conditions. Missing EOS counts as a failure. |
| Mel synthesis | Griffin–Lim, 32 iterations, phase seed 1 | Explicit settings for a small number of new inputs. The paper's iteration count is unpublished. Uses the existing mel specification and inverse transform. |
| Cascade and S2T→TTS | Models pinned by SHA in existing configs and the same English Qwen synthesis | Reuses model settings and weights for the downstream English pipeline. |
| Qwen | Shared `official_sampling_eos_v1` | Sampling, EOS monitoring, limits, and seed decisions are documented in the [audit](CASCADE_TTS_AUDIT.md). |

The cell runs inference and restores only the selected TT2 or S2UT run as needed. It restores the original dependency environment exactly and checks imports for pinned components such as Qwen. Existing dependency versions and saved SHAs govern restoration.

## Timing and memory

GPU inference runs **sequentially in separate processes** to avoid concurrent-load OOMs and interference. This is the user's chosen execution mode. Peak measurements describe sequential loading in this cell. The existing CLI can hold all models at once and has different VRAM requirements. Displayed timings cover a **first inference** for one utterance with batch size 1. Order, kernel initialization, GPU clocks, caches, and other Colab workload can affect them. This run uses no warmup, repeated measurements, or statistical tests.

- Translation and synthesis time sums ASR, MT, and English TTS for Cascade, and model inference and synthesis for the candidate. It includes the source frontend and local audio and feature serialization. Timing synchronizes CUDA.
- This interval excludes shared Japanese synthesis, process startup, model retrieval and loading, Drive writes, and English ASR for evaluation.
- Worker retrieval and load time is displayed separately. It can include Hugging Face retrieval but excludes Drive snapshot restoration by the parent process, so it cannot fairly compare cold download speeds.
- Input RTF divides translation and synthesis seconds by the shared Japanese duration. JSON also records RTF based on output audio duration.
- RAM is peak worker RSS sampled every 50 ms. This can miss short peaks and excludes the rest of Colab and other processes. RAM is unavailable if psutil or permissions are missing.
- VRAM uses `torch.cuda.max_memory_allocated/reserved`, includes loading and inference, and reports allocated memory as a fraction of total GPU capacity. It excludes the CUDA context, other processes, and non-Torch allocations. Device-wide nvidia-smi memory and GPU compute utilization are different measurements.
- Memory for sequential stages is aggregated using the maximum. Failures and unavailable measurements are shown as such.

Each measurement records its session ID and GPU. When resumed measurements come from different sessions or GPUs, the original values are kept and the UI and report mark the conditions as different. A formal performance study requires the same GPU and inputs, warmup, and repeated runs.

## Storage and resume

`<experiment>/interactive/<ID>/` stores the plan, stage receipts, input and output WAVs, intermediate mel and unit outputs, ASR text, `comparison.json`, and `report.md`. Training, test evaluation, and the corpus are read-only here. Verify SHA before saving each stage's audio and reuse successful stages. Failed attempts go to `history/`. Corrupt files stop the run.

After disconnection, use Cells 1 → 10 with the same settings. Completed comparisons display stored audio and measurements without fetching checkpoints or running GPU inference. They stay pinned to their saved snapshot as training under the same run name advances. Set `INTERACTIVE_REMEASURE=True` for a new snapshot or timing run. It creates a new ID and retains previous results. Set it back to False to resume an interrupted measurement. If a candidate such as TT2 fails, Cascade results remain available and the UI reports the error and unavailable BLEU and performance.

The cell uses existing SciPy, SoundFile, Torch, SacreBLEU, Transformers, fairseq, and Qwen dependencies. CPU tests and mocks cover source-only inputs and outputs, Japanese synthesis and resampling, exclusion of targets, save and resume, independent failures, memory aggregation, and the standard notebook flow. Real-weight Colab/GPU end-to-end inference remains unverified.

The hardware test is `tests/gpu/test_interactive_comparison_gpu.py`. Set `S2ST_INTERACTIVE_GPU_PLAN` to a verified S2UT plan to test real Qwen Japanese synthesis, both translations, English synthesis, and ASR and diagnostic BLEU in a separate temporary location. It fetches models and consumes compute units, so it is excluded from standard CPU runs. TT2 translation quality and EOS are independently tracked.

Verification reported 26 feature tests and **775 passed, 2 skipped, and 8 GPU tests deselected** for the full CPU run. Python 3.10 syntax compatibility and `git diff --check` passed. These checks ran locally on Python 3.13/CPU. Real weights and VRAM measurements on Colab Python 3.10/GPU remain to be verified.
