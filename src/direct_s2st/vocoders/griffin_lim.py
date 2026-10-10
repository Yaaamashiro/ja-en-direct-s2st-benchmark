"""Resumable TT2 BLEU waveform reconstruction; no weights or neural fallback.

Inverse of the pinned fairseq frontend: exp(log magnitude Mel), Slaney Mel
pseudoinverse, classical Griffin-Lim (power=1, momentum=0, centered Hann STFT).
Phase seed and iteration count are explicit implementation choices, not known
Google settings. Each utterance seed is independent of order/resume.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time

import numpy as np

from ..hashing import sha256_file
from ..io import atomic_write_json, read_jsonl, ExistingOutputError
from ..journal import Journal
from ..progress import operation, track
from .inference import MEL_KEYS, write_waveform


def validate_spec(spec):
    from ..translatotron2.prepare_fairseq import _mel_settings
    _mel_settings(spec)
    if any(key not in spec for key in MEL_KEYS) or spec['log_transform'] != 'natural_log_clamp_eps' or spec['normalization'] != 'none' or spec['normalize_volume']:
        raise ValueError('Griffin-Lim requires complete unnormalized natural-log magnitude Mel specification')


@operation('griffin-lim: load inverse Mel frontend')
def inverse_basis(spec):
    import librosa
    filters = librosa.filters.mel(sr=spec['sample_rate'], n_fft=spec['n_fft'],
                                  n_mels=spec['n_mels'], fmin=spec['f_min'], fmax=spec['f_max'],
                                  htk=False, norm='slaney', dtype=np.float32)
    return np.linalg.pinv(filters).astype(np.float32)


def reconstruct(feature, spec, basis, *, iterations=32, seed=1):
    import librosa
    if iterations < 1 or seed < 0:
        raise ValueError('iterations must be positive and seed nonnegative')
    if feature.ndim != 2 or feature.shape[1] != spec['n_mels'] or feature.shape[0] < 2 or not np.isfinite(feature).all():
        raise ValueError('invalid predicted Mel')
    if feature.shape[0] * spec['hop_length'] > spec['sample_rate'] * 120:
        raise ValueError('Griffin-Lim input exceeds 120 seconds')
    with np.errstate(over='raise', invalid='raise'):
        magnitude = np.maximum(basis @ np.exp(feature).T, 0)
    if not np.isfinite(magnitude).all():
        raise ValueError('nonfinite inverse Mel magnitude')
    return librosa.griffinlim(magnitude, n_iter=iterations, n_fft=spec['n_fft'],
        hop_length=spec['hop_length'], win_length=spec['win_length'], window='hann',
        center=True, pad_mode='reflect', momentum=0.0, init='random', random_state=seed,
        length=(feature.shape[0] - 1) * spec['hop_length'], dtype=np.float32)


@operation('griffin-lim: BLEU audio (generate/reuse)')
def vocode(input_path, output_root, mel_spec, *, iterations=32, seed=1, resume=False, overwrite=False):
    input_path, output_root, mel_spec = Path(input_path), Path(output_root), Path(mel_spec)
    if os.environ.get('CORPUS_ROOT') and output_root.resolve().is_relative_to(Path(os.environ['CORPUS_ROOT']).resolve()):
        raise ValueError('vocoder output must not be inside CORPUS_ROOT')
    if iterations < 1 or seed < 0:
        raise ValueError('invalid Griffin-Lim iterations/seed')
    spec = json.loads(mel_spec.read_text(encoding='utf-8'))
    validate_spec(spec)
    rows = list(read_jsonl(input_path))
    if not rows or len({r['pair_id'] for r in rows}) != len(rows) or any(r['system_id'] != 'translatotron2' for r in rows):
        raise ValueError('empty/duplicate/wrong-system Griffin-Lim inputs')
    identity = dict(backend='griffin_lim', input_sha256=sha256_file(input_path), mel=spec,
                    mel_spec_sha256=sha256_file(mel_spec), checkpoint_sha256=None,
                    iterations=iterations, seed=seed, momentum=0.0, magnitude_power=1,
                    mel_filter='librosa Slaney / norm=slaney / htk=False',
                    librosa_version=importlib.metadata.version('librosa'),
                    numpy_version=np.__version__,
                    paper_phase_iterations_status='unknown; explicit implementation choices',
                    mel_hashes={r['pair_id']: sha256_file(Path(r['mel_path']))
                                for r in track(rows, 'griffin-lim: verify Mel inputs') if r.get('status') == 'success'})
    if not (resume or overwrite) and (output_root / 'audio').exists():
        raise ExistingOutputError(str(output_root / 'audio'))
    journal = Journal(output_root / 'predictions.jsonl', identity, resume=resume, overwrite=overwrite)
    basis = inverse_basis(spec)
    for index, row in enumerate(track(rows, 'griffin-lim: reconstruct/reuse')):
        previous = journal.rows.get(row['pair_id'])
        if previous and previous.get('status') == 'success':
            path = Path(previous['output_audio'])
            if path.is_file() and sha256_file(path) == previous.get('output_sha256'):
                continue
        started = time.perf_counter()
        try:
            if row.get('status') != 'success':
                raise ValueError('upstream Mel generation failed: ' + str(row.get('error')))
            feature = np.load(row['mel_path'], allow_pickle=False)
            utterance_seed = int(hashlib.sha256(f"{seed}:{row['pair_id']}".encode()).hexdigest()[:8], 16)
            waveform = reconstruct(feature, spec, basis, iterations=iterations, seed=utterance_seed)
            path = output_root / 'audio' / f'{index:08d}.wav'
            duration = write_waveform(path, waveform, spec['sample_rate'], overwrite=resume or overwrite)
            elapsed = time.perf_counter() - started
            seconds = float(row.get('inference_seconds', 0)) + elapsed
            output = dict(row, status='success', error=None, output_audio=str(path.resolve()),
                output_duration=duration, output_sha256=sha256_file(path), inference_seconds=seconds,
                vocoder_seconds=elapsed, real_time_factor=seconds/duration, vocoder_type='griffin_lim')
        except Exception as error:
            output = dict(row, status='failed', error=f'{type(error).__name__}: {error}', output_audio=None,
                          output_duration=None, real_time_factor=None,
                          inference_seconds=float(row.get('inference_seconds', 0)) + time.perf_counter() - started,
                          vocoder_type='griffin_lim')
        journal.record(output)
    metadata = dict(identity, samples=len(rows), failures=sum(r['status'] != 'success' for r in journal.rows.values()))
    atomic_write_json(output_root / 'vocoder-lock.json', metadata, resume=resume, overwrite=resume or overwrite)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--mel-spec', type=Path, required=True)
    parser.add_argument('--iterations', type=int, default=32)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    print(json.dumps(vocode(args.input, args.output_root, args.mel_spec, iterations=args.iterations,
                           seed=args.seed, resume=args.resume, overwrite=args.overwrite)))


if __name__ == '__main__':
    main()
