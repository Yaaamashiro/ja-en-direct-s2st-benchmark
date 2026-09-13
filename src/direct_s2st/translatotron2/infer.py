"""Reference-free native TT2 inference; outputs mels, never fabricated WAVs."""
import argparse
import json
import os
from pathlib import Path
import tempfile
import time
import numpy as np
import torch
from ..hashing import sha256_file
from ..io import ExistingOutputError, atomic_write_json, atomic_write_jsonl
from ..manifests.reader import read_common_manifest
from .data import read_table, source_features
from .engine import load_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--common-root', type=Path, required=True)
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--split', choices=['dev', 'test'], default='test')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--max-phones', type=int, default=400)
    parser.add_argument('--max-frames', type=int, default=3000)
    parser.add_argument('--beam-size', type=int, default=1)
    parser.add_argument('--length-penalty', type=float, default=0.0)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if os.environ.get('CORPUS_ROOT') and args.predictions.resolve().is_relative_to(Path(os.environ['CORPUS_ROOT']).resolve()):
        raise ValueError('predictions must not be inside CORPUS_ROOT')
    feature_root = args.predictions.parent / 'mels'
    if not (args.overwrite or args.resume) and (args.predictions.exists() or feature_root.exists()):
        raise ExistingOutputError(str(args.predictions))
    checkpoint = args.run_root / 'checkpoints/checkpoint_last.pt'
    started = time.perf_counter()
    model, state = load_checkpoint(checkpoint, device=args.device)
    model.eval()
    spec_path = args.data_root / 'mel-spec.json'
    if state['data_fingerprint'].get('mel-spec.json') != sha256_file(spec_path):
        raise ValueError('checkpoint mel specification mismatch')
    spec = json.loads(spec_path.read_text(encoding='utf-8'))
    source_spec_path = args.data_root / 'source-mel-spec.json'
    source_spec = spec
    if source_spec_path.is_file():
        if state['data_fingerprint'].get('source-mel-spec.json') != sha256_file(source_spec_path):
            raise ValueError('checkpoint source mel specification mismatch')
        source_spec = json.loads(source_spec_path.read_text(encoding='utf-8'))
    prepared = read_table(args.data_root / f'{args.split}.tsv')
    common = list(read_common_manifest(args.common_root / f'{args.split}.jsonl'))
    if {r['id'] for r in prepared} != {r['pair_id'] for r in common} or len(common) != len(prepared):
        raise ValueError('common/prepared IDs mismatch')
    if any(row['split'] != args.split for row in common):
        raise ValueError('common split mismatch')
    loading_seconds = time.perf_counter() - started
    from ..journal import Journal
    identity = dict(checkpoint=sha256_file(checkpoint),
                    common=sha256_file(args.common_root / f'{args.split}.jsonl'),
                    source={r['pair_id']: sha256_file(Path(r['ja_audio'])) for r in common},
                    max_phones=args.max_phones, max_frames=args.max_frames, split=args.split,
                    beam_size=args.beam_size, length_penalty=args.length_penalty)
    journal = Journal(args.predictions, identity, resume=args.resume, overwrite=args.overwrite)
    feature_root.mkdir(parents=True, exist_ok=True)
    for index, row in enumerate(common):
        previous = journal.rows.get(row['pair_id'])
        if previous and previous.get('status') == 'success':
            path = Path(previous['mel_path'])
            if path.is_file() and sha256_file(path) == previous.get('mel_sha256'):
                continue
        started = time.perf_counter()
        output = dict(pair_id=row['pair_id'], system_id='translatotron2', run_id=args.run_root.name,
                      split=args.split, source_audio=row['ja_audio'], reference_audio=row['en_audio'],
                      reference_text=row['en_text'],
                      timing_scope='per-sample source frontend and generation; excludes model load')
        try:
            source = source_features(Path(row['ja_audio']), source_spec).to(args.device)
            result = model.generate(source[None], torch.tensor([len(source)], device=args.device),
                                    max_phones=args.max_phones, max_frames=args.max_frames,
                                    beam_size=args.beam_size, length_penalty=args.length_penalty)
            values = result['post_mel'][0, :int(result['frame_lengths'][0])].cpu().numpy()
            if not np.isfinite(values).all() or values.size == 0:
                raise ValueError('invalid generated mel')
            path = feature_root / f'{index:08d}.npy'
            descriptor, temporary = tempfile.mkstemp(dir=feature_root, suffix='.npy.tmp')
            try:
                with os.fdopen(descriptor, 'wb') as stream:
                    np.save(stream, values, allow_pickle=False)
                os.replace(temporary, path)
            finally:
                Path(temporary).unlink(missing_ok=True)
            output.update(status='success', error=None, mel_path=str(path.resolve()), mel_sha256=sha256_file(path))
        except Exception as error:
            output.update(status='failed', error=f'{type(error).__name__}: {error}', mel_path=None)
        output['inference_seconds'] = time.perf_counter() - started
        journal.record(output)
    atomic_write_json(args.predictions.parent / 'tt2-inference-lock.json',
                      dict(checkpoint_sha256=sha256_file(checkpoint), mel_spec_sha256=sha256_file(spec_path),
                           model_loading_seconds=loading_seconds, samples=len(journal.rows)),
                      overwrite=args.overwrite or args.resume)


if __name__ == '__main__':
    main()
