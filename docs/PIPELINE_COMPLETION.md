# Implementation scope and four-system workflow

## Implemented features

- Train unit and mel vocoders on the training split using the pinned fairseq CodeGenerator
  and Generator. Optimizer updates use MPD, MSD, least-squares GAN losses, feature matching,
  and mel L1. The unit duration predictor learns from run lengths in the original HuBERT
  sequence using MSE on log(1 + duration). Inference uses the trained duration predictor.
- Save the generator, discriminators, both optimizers, RNG state, and input and audio hashes
  in an atomic checkpoint. Resume verifies the configuration, source audio, and units.
  Vocoder fitting rejects dev and test data.
- TT2 implements a relative-position Conformer, 2D subsampling, kernel size 32, masked
  BatchNorm, reference presets, independent source and target mel settings, beam search,
  gradient accumulation, DDP, per-rank checkpoint recovery, and dev validation.
  The default is a small CPU model with two updates.
- The S2UT training wrapper audits finite, nonzero gradients for the main loss, three
  auxiliary losses, and five parameter groups using the pinned criterion and model.
- TT2, S2UT, vocoders, and Cascade record failures and support recovery. Evaluation verifies
  input content, and reuse of successful samples verifies output hashes.
- A 25-stage workflow runs four systems on the same test split. It checks WAV files,
  ASR results, evaluation conditions, and checkpoints before producing the comparison report.
  A zero exit code alone does not satisfy end-to-end acceptance.

## Run after configuring the environment

Use `.env.example` to set `CORPUS_ROOT`, `EXPERIMENT_DATA_ROOT`, `RUNS_ROOT`,
and `CACHE_ROOT`. Mount the corpus read-only and write derived data on the benchmark side.
Build the pinned Docker images first. Run the following from the repository root.
The first command prints the plan.

```sh
python scripts/smoke/suite.py --env-file .env --name benchmark-smoke --limit 5 --max-updates 2
python scripts/smoke/suite.py --env-file .env --name benchmark-smoke --limit 5 --max-updates 2 --execute
python scripts/smoke/suite.py --env-file .env --name benchmark-smoke --limit 5 --max-updates 2 --execute --resume
```

Use `--docker-context` to choose the Docker host. For remote Docker, the corpus, output,
and repository bind-mount sources must already exist on the daemon host. The driver
does not transfer files. To change the container config mount, set
`S2ST_CONTAINER_CONFIG_ROOT` in `.env` to the same value as `--container-config-root`.

The default fits both vocoders on the train split. To use verified compatible pretrained
weights, pass `--vocoder-mode pretrained` and set the checkpoint and config in
`configs/vocoder/{unit,mel}.yaml`. When fitting vocoders, generated configs reference
`vocoder-unit` and `vocoder-mel` within the run. Stages download the configured
models as needed, and a dedicated stage retrieves unit artifacts. The workflow reuses
the existing corpus and requires explicitly configured models.

Generated configs go to `configs/local/<name>` and system outputs go to
`RUNS_ROOT/<name>-<system>`. Completion is recorded in
`EXPERIMENT_DATA_ROOT/results/e2e-acceptance.json`, with comparison outputs in the same
`results` directory. Resume rejects changed plan, configuration, or environment file
hashes. It skips successful stages and continues from the failed stage. If training
stops before the first checkpoint, the existing run is preserved. Inspect its logs and
use a new name or derived-data root, or explicitly pass `--overwrite` to the individual command.

`--device cpu` applies to direct models and vocoders. Cascade and evaluation use their
own model settings, so some stages can still use a GPU. Acceptance here excludes
large-scale or multi-GPU performance and automated cloud provisioning.

## Verification and limitations

CPU regression tests reported **92 passed, 3 deselected**. They covered TT2 forward
and backward passes, optimizer updates, checkpoint saving and exact resume, optimizer
updates with the pinned fairseq HiFi-GAN generator and official period discriminator,
duration targets, recovery after failure, hash mismatch rejection, rejection of silent
WAVs during acceptance, and the 25-stage plan. A real two-process CPU Gloo run also
verified exact recovery with per-rank checkpoints.

The CPU GAN test uses one period discriminator to limit memory use. The full production
MPD/MSD, CodeGenerator training on real audio, the pinned fairseq frontend, and Linux
container integration were not run. Synthetic tensors and controlled EOS outputs are
unit-test inputs.

Docker, the target corpus, and trained weights were unavailable together in this
workspace. Real-audio end-to-end success across all four systems, convergence,
intelligibility, and BLEU results remain unverified. Two updates exercise the execution
path and provide no guarantee of TT2 reaching EOS or of vocoder quality. The pipeline
records missing EOS and overlong outputs as failures. It does not substitute forced
phonemes, silence, or random output. The independent implementation's correspondence
with the papers and unverified numerical results are documented in [REFERENCE_PARITY.md](REFERENCE_PARITY.md).

## External code and licenses

The generators use the unmodified pinned fairseq commit
`3d262bb25690e4eb2e7d3c1309b1e9c406ca4b99`.
The discriminators and losses incorporate the necessary parts of the
[pinned official HiFi-GAN source](https://github.com/jik876/hifi-gan/blob/4769534d45265d52a904b850da5a622601885777/models.py),
with only import adjustments. The full MIT license is in
`src/direct_s2st/vocoders/HIFIGAN_LICENSE`.
The benchmark recipe uses single-sample segments, AdamW with a fixed learning rate,
a mel weight of 45, and a duration weight of 1. It does not claim to match the entire
original training schedule. Torch and torchaudio remain pinned at 2.7.1. The
`weight_norm` deprecation warning remains for compatibility.

## S2T→TTS / four-system extension

The required final systems are `s2ut`, `translatotron2`, `cascade`, and `s2t_tts`.
The new baseline uses pinned Whisper large-v3 translation and the existing QwenTTS,
without changing the original turbo ASR → NLLB → TTS model settings.
See [S2T→TTS implementation and verification](S2T_TTS.md) for commands, Colab,
new files, dependency limitations, and unverified GPU/E2E items.
Historical test counts below/above describe their original runs, not current validation.
