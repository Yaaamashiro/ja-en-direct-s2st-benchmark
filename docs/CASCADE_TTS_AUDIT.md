# Cascade Qwen TTS configuration audit and EOS handling (2026-10-24)

## Decision criteria

Review the official and previous values, rationale, suitability, and verification scope before adopting a deviation. Treat unknown official values as unknown. Changes that can affect quality get a separate evaluation. Evaluate every generation attempt under the configured policy, with no automatic fallback or selection of favorable outputs. This change concerns the Qwen adapter shared by Cascade and S2T→TTS.

In Colab, the short translation `The victim"s name is Sebastian Jones.` advanced the main talker's generation step from 7566 to 7621, approaching its 8192-token cap. The process was still generating an unusually long audio code sequence. The observation did not capture a final EOS exception or establish why EOS had not appeared. The inner `max_new_tokens=15` applies to the subtalker that generates the remaining 15 codebooks.

## Comparison with official settings

The reference sources are the pinned [generation config](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice/blob/f00cf133b78d3c2c35857faba3b1be9b98c4f971/generation_config.json), [model config](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice/blob/f00cf133b78d3c2c35857faba3b1be9b98c4f971/config.json), [codec config](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice/blob/f00cf133b78d3c2c35857faba3b1be9b98c4f971/speech_tokenizer/config.json), and the source inside the pinned `qwen-tts==0.1.1` wheel. Compatibility checks use immutable sources.

| Setting | Official and previous values | Adopted value | Rationale and verification scope |
|---|---|---|---|
| Main and subtalker sampling | Official: both True. Previous: both False. | Both True | Greedy decoding was chosen for deterministic evaluation without sufficient verification for this speech model. Restoring official sampling requires a separate evaluation from old WAVs. Whether it resolves this EOS issue remains unverified. |
| Other sampling parameters | top_k=50, top_p=1, temperature=0.9, repetition_penalty=1.05. Subtalker uses 50/1/0.9. | Explicit official values | Avoids a separate temperature or penalty search that could change quality. Checks both the pinned artifacts and actual talker arguments. |
| RNG | The official generation config has no per-sentence seed. | Derive from base seed 17 and the SHA-256 of the exact MT input text. | Makes resume, ordering, and preflight synthesis independent of random number consumption. References are excluded. Restores surrounding Python, NumPy, Torch, and CUDA RNG state. Bitwise equality across GPUs or libraries is unverified. |
| max_new_tokens | Official and previous: 8192. | 1500 audio frames plus EOS, for 1501 tokens, to match the existing 120-second acceptance limit. | Codec rate is 24000/1920 = 12.5 Hz. Previously, WAVs over 120 seconds already failed acceptance. Applying the limit before generation avoids excess work. Forced EOS and truncated WAVs do not count as successful. |
| Main generation time | No official time limit for this benchmark. | 600 seconds, checked cooperatively at token boundaries. | The observed rate of 55 main tokens in 12 seconds would take about 330 seconds for 1500 tokens. The 600-second guard includes slack and may reject valid generations on slow hardware. The plan records it. |
| Dtype | Official example uses BF16. | BF16 | Stops before model load if the GPU cannot support BF16. |
| Attention | Official example recommends FlashAttention 2. Previous code uses SDPA. | SDPA | Avoids an additional build dependency and retains compatibility with the pinned environment. Bitwise equality with FA2 is unverified. |
| Voice and language | CustomVoice's Ono_Anna is a Japanese speaker, and English is supported. | Ono_Anna / English | Holds voice constant when evaluating the generation change. English accent and audio quality are unverified. The observation does not establish a link between voice and the EOS issue. |
| Model and SDK | Pinned CustomVoice revision and qwen-tts 0.1.1. | Same SHA and dependencies | Loads the complete local snapshot, including the model root and speech_tokenizer. |

Sampling follows the official configuration. The time and token limits and fixed seeds are benchmark-specific controls. Actual GPU tests are needed to confirm the EOS behavior. [Upstream issue #341](https://github.com/QwenLM/Qwen3-TTS/issues/341) reports greedy degeneration for a different model type and environment.

## Implementation checks

`qwen-tts==0.1.1` does not forward arbitrary public SDK kwargs to the main talker. Passing `stopping_criteria` to the public API therefore does not install the observer. The adapter attaches the observer to the instance's main `talker.generate` only for the duration of the call.

- The pinned model's audio EOS is 2150. Its text EOS is 151673.
- Verify the actual sampling arguments, token limit, and audio EOS.
- Retain Hugging Face EOS stopping and log the actual token count and `no_advance` to stderr. In `checked=N/1501`, 1501 is a safety cap. Valid EOS can arrive sooner. The denominator does not provide an ETA or remaining token estimate.
- Check audio EOS at the end of the returned sequence before codec decode. Decode and save a successful WAV only when verified.
- Hitting the cap or missing EOS is a sample-level failure. CUDA, dependency, or API contract failures stop the run.
- Record main token count, seed, policy, EOS status, stop reason, and duration in `tts_generation`.
- Before reusing successful WAVs, verify SHA, format, policy, input seed, and the EOS record.
- Restore the instance's generate method, RNG state, and adapter lock on exceptions and KeyboardInterrupt.

The time guard is cooperative. A blocked CUDA kernel, work before the first token, and codec decode cannot be interrupted immediately. The check runs at main token boundaries after the kernel returns. If generation returns a valid EOS, that success takes precedence over the time cap. The guard does not kill or recover OS child processes.

## Standard cells for a separate evaluation

The user approved creating a separate evaluation with official sampling settings.

1. Run standard Cells 1 → 6 with the updated code to create a new evaluation ID and plan that include the policy.
2. Pin the completed ASR/MT plan, pointers, and runtime SHA in the new plan's `text_source`. Check matching test IDs, manifest, and ASR/MT model settings.
3. In Cell 7, verify old ZIPs, journals, input associations, and dependencies in read-only mode. Stop on missing, corrupted, or incompatible inputs. Whisper and NLLB are not loaded to regenerate the old text.
4. Run preflight synthesis on a fixed non-reference sentence, then generate the full TTS set using the new policy.
5. Evaluate the new WAVs in Cell 8 and listen in Cell 9. The existing BLEU definition counts failed samples as empty hypotheses across the full test set.

The original folders retain the old plan, ASR/MT data, runtime, WAVs, and report. `latest.json` points to the new evaluation and its previous bytes are saved to the new evaluation's `previous-latest.json`. To resume, use Cells 1 → 6 if needed → 7 → 8 with the same new policy. Only successful WAVs from that evaluation are reused. A standard rerun retries failed rows under the same sampling settings.

The old loader-only overlay must still reject importing greedy WAVs into the new policy. Retain the original editable dependency build location. The explicit allowlist for restoring the old build contains `f9f1520047bca2c0e977b581cab70168892091e2`. Reusing an unrecognized dependency revision requires review. Do not edit dependency receipts or plans to bypass validation.

## Resume and display fixes

- Group progress displays part IDs and omits the large tuple that contained ASR text. Repeated ASR text alone does not establish that ASR reran or that the process stalled.
- Restore by reconciling published Drive parts with the local retry journal. Preserve old successful rows and WAVs, plus newer results for previously failed rows and new local successes.
- Resuming the first part retains later published parts in the audio index.
- ZIPs still contain 32 items per part. Losing the VM can require regenerating up to 31 unpublished items in the last part.

## Verification and limitations

CPU coverage includes EOS and StoppingCriteria with a small real Transformers model, pinned SDK keyword forwarding, missing EOS, EOS on the last token, the time guard, invalid arguments, RNG and exception recovery, validation before PCM conversion, official artifact contracts, old evaluation bytes and modification times, stopping on corrupt ASR/MT, ZIP recovery, interrupted retries, index preservation, and standard notebook branches. These tests use a small model and SDK boundary doubles.

Real GPU tests are in `tests/gpu/test_cascade_qwen_preflight.py`. Use `S2ST_RUN_QWEN_GPU_TEST=1` for preflight synthesis with EOS using the pinned L4 and weights. Use `S2ST_RUN_QWEN_AB_TEST=1` for bounded greedy and sampling on three fixed non-reference sentences. Both conditions use the same seed, a 20-second-equivalent token budget, and a 180-second cooperative time budget. Both successes and failures are recorded. A short-budget failure can still succeed with the formal 120-second audio budget. The old greedy diagnostic is test-only.

Colab, compute units, and GPU inference with real Qwen weights were not used. Long generations and audio quality need hardware verification. Dependencies, pip, CUDA, and the attention backend were retained. The pinned SDK source and actual Transformers 4.57.3 generation API were checked. The worker continues to require the pinned dependency receipt.

The final CPU run used `pytest -q -m 'not gpu'` on Python 3.13 and reported **749 passed, 2 skipped, and 7 deselected**. There were six existing Swig and weight_norm deprecation warnings and zero failures. `git diff --check` passed. Real-weight end-to-end verification on Colab Python 3.10 and L4 remains pending.
