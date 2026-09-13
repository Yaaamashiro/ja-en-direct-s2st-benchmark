"""Checkpoint-backed fairseq HiFi-GAN adapters; no synthetic fallback."""
import argparse
import json
import math
import os
from pathlib import Path
import tempfile
import time

from ..hashing import sha256_file
from ..io import ExistingOutputError, atomic_write_json, atomic_write_jsonl, read_jsonl
from ..s2ut.reduce_units import validate_units

MEL_KEYS = ("sample_rate", "n_fft", "win_length", "hop_length", "n_mels", "f_min", "f_max",
            "log_transform", "normalization", "eps", "normalize_volume")


def validate_config(kind: str, config: dict, *, sample_rate: int, mel: dict | None = None) -> None:
    if config.get("sampling_rate") != sample_rate or sample_rate <= 0:
        raise ValueError("vocoder sampling_rate differs from requested sample rate")
    if kind == "unit":
        if config.get("num_embeddings") != 100 or not config.get("dur_predictor_params"):
            raise ValueError("KM100 reduced units require 100 embeddings and a trained duration predictor")
        if config.get("f0") or config.get("multispkr") or config.get("embedder_params"):
            raise ValueError("this adapter requires a single-speaker vocoder without external F0")
    else:
        if mel is None or any(key not in mel or key not in config.get("mel", {}) or mel[key] != config["mel"][key] for key in MEL_KEYS):
            raise ValueError("complete matching model/vocoder mel specification is required")
        if mel["sample_rate"] != sample_rate or config.get("model_in_dim", 80) != mel["n_mels"]:
            raise ValueError("mel specification disagrees with generator input/sample rate")
        if not config.get("upsample_rates") or math.prod(config["upsample_rates"]) != mel["hop_length"]:
            raise ValueError("mel specification hop_length disagrees with generator upsampling")


def write_waveform(path: Path, waveform, sample_rate: int, *, overwrite: bool = False) -> float:
    import numpy as np
    import soundfile as sf
    values = np.asarray(waveform, dtype=np.float32).squeeze()
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all() or not np.any(values):
        raise ValueError("vocoder output must be a finite, nonempty, nonsilent mono waveform")
    pcm = np.clip(values * 32767, -32768, 32767).astype(np.int16)
    if not np.any(pcm):
        raise ValueError("vocoder output becomes silent at PCM16 precision")
    if path.exists() and not overwrite:
        raise ExistingOutputError(f"output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, suffix=".wav")
    os.close(fd)
    temporary = Path(name)
    try:
        sf.write(temporary, pcm, sample_rate, subtype="PCM_16")
        reread, rate = sf.read(temporary)
        if rate != sample_rate or reread.size != values.size or not np.any(reread):
            raise ValueError("WAV readback validation failed")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return values.size / sample_rate


def vocode(kind: str, input_path: Path, output_root: Path, checkpoint: Path, config_path: Path,
           *, sample_rate: int, device: str = "cuda", mel: dict | None = None,
           duration_prediction: bool = True, overwrite: bool = False, resume: bool = False) -> dict:
    from ..journal import Journal
    if os.environ.get('CORPUS_ROOT') and output_root.resolve().is_relative_to(Path(os.environ['CORPUS_ROOT']).resolve()):
        raise ValueError('vocoder output must not be in CORPUS_ROOT')
    if not checkpoint.is_file():
        raise FileNotFoundError(f"vocoder checkpoint not found: {checkpoint}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_config(kind, config, sample_rate=sample_rate, mel=mel)
    if kind == "unit" and not duration_prediction:
        raise ValueError("reduced units require duration prediction")
    records = list(read_jsonl(input_path))
    if not records or len({r['pair_id'] for r in records}) != len(records):
        raise ValueError('empty or duplicate vocoder inputs')
    for row in records:
        if row.get('system_id') != ('s2ut' if kind == 'unit' else 'translatotron2'):
            raise ValueError('vocoder input system mismatch')
    identity = dict(backend=f'fairseq_{kind}_hifigan', checkpoint_sha256=sha256_file(checkpoint),
                    config_sha256=sha256_file(config_path), input_sha256=sha256_file(input_path),
                    sample_rate=sample_rate, duration_prediction=kind == 'unit', mel=mel)
    # Bind successful mel inputs too: their paths alone are not content identities.
    if kind == 'mel':
        identity['mel_hashes'] = {r['pair_id']: sha256_file(Path(r['mel_path'])) for r in records
                                  if r.get('status') != 'failed'}
    if not (resume or overwrite) and (output_root / 'audio').exists():
        raise ExistingOutputError(str(output_root / 'audio'))
    journal = Journal(output_root / 'predictions.jsonl', identity, resume=resume, overwrite=overwrite)
    import numpy as np
    import torch
    if kind == "unit":
        from fairseq.models.text_to_speech.codehifigan import CodeGenerator
        model = CodeGenerator(config)
    else:
        from fairseq.models.text_to_speech.hifigan import Generator
        model = Generator(config)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if state.get('format') == 'direct-s2st-vocoder-v1' and state.get('updates', 0) < 1:
        raise ValueError('vocoder checkpoint has no optimizer updates')
    model.load_state_dict(state["generator"], strict=True)
    if any(not torch.isfinite(p).all() for p in model.parameters()):
        raise ValueError('nonfinite vocoder checkpoint')
    model.eval().to(device)
    for index, row in enumerate(records):
        previous = journal.rows.get(row['pair_id'])
        if previous and previous.get('status') == 'success':
            path = Path(previous['output_audio'])
            if path.is_file() and sha256_file(path) == previous.get('output_sha256'):
                continue
        started = time.perf_counter()
        path = output_root / "audio" / f"{index:08d}.wav"
        try:
            if row.get('status') == 'failed':
                raise ValueError('upstream generation failed: ' + str(row.get('error')))
            with torch.inference_mode():
                if kind == "unit":
                    units = validate_units(row["units"], clusters=100)
                    code = torch.tensor([units], dtype=torch.long, device=device)
                    lengths = (model.dur_predictor(model.dict(code)).exp() - 1).round().clamp_min(1)
                    if not torch.isfinite(lengths).all() or lengths.sum() * math.prod(config['upsample_rates']) > sample_rate * 120:
                        raise ValueError('predicted vocoder duration is nonfinite or exceeds 120 seconds')
                    result = model(code=code, dur_prediction=True)
                else:
                    feature = np.load(row["mel_path"], allow_pickle=False)
                    if feature.ndim != 2 or feature.shape[1] != mel["n_mels"] or feature.shape[0] == 0 or not np.isfinite(feature).all():
                        raise ValueError(f"invalid predicted mel: {row['pair_id']}")
                    if feature.shape[0] * mel['hop_length'] > sample_rate * 120:
                        raise ValueError('mel vocoder input exceeds 120 seconds')
                    result = model(torch.from_numpy(feature).float().to(device).T.unsqueeze(0))
                waveform = result.detach().float().cpu().numpy()
            duration = write_waveform(path, waveform, sample_rate, overwrite=overwrite or resume)
            elapsed = time.perf_counter() - started
            seconds = float(row.get("inference_seconds", 0)) + elapsed
            output = {**row, "output_audio": str(path.resolve()), "output_duration": duration,
                      "output_sha256": sha256_file(path), "inference_seconds": seconds,
                      "vocoder_seconds": elapsed, "real_time_factor": seconds / duration,
                      "status": "success", "error": None}
        except Exception as error:
            output = {**row, 'output_audio': None, 'output_duration': None, 'real_time_factor': None,
                      'inference_seconds': float(row.get('inference_seconds', 0)) + time.perf_counter() - started,
                      'status': 'failed', 'error': f'{type(error).__name__}: {error}'}
        journal.record(output)
    metadata = {**identity, 'samples': len(records),
                'failures': sum(r['status'] != 'success' for r in journal.rows.values())}
    atomic_write_json(output_root / "vocoder-lock.json", metadata, overwrite=overwrite or resume)
    return metadata


def main(kind: str) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--sample-rate", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--duration-prediction", action="store_true")
    parser.add_argument("--mel-spec", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    mel = json.loads(args.mel_spec.read_text(encoding="utf-8")) if args.mel_spec else None
    print(json.dumps(vocode(kind, args.input, args.output_root, args.checkpoint, args.config,
                            sample_rate=args.sample_rate, device=args.device, mel=mel,
                            duration_prediction=args.duration_prediction, overwrite=args.overwrite, resume=args.resume)))
