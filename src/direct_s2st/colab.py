"""Bounded native training sessions with immutable, verified Drive snapshots.

Only load checkpoints created by your own trusted training commands. A Drive
mount is not a transactional filesystem: completion markers and hashes detect
partial copies, but do not promise that unsynced writes survive VM deletion.
"""
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid

from .hashing import sha256_file
from .io import atomic_write_json


def make_config(repository, data, environment_lock, kind='tt2', model_size='smoke', *,
                performance='smoke', batch_size=None, num_workers=None, prefetch_factor=2,
                max_tokens=None, update_freq=1, save_interval=None):
    """Explicit throughput knobs; gpu80 is a starting point, not a VRAM guarantee."""
    from .config import load_config
    if performance not in ('smoke', 'gpu80'):
        raise ValueError('unknown performance preset')
    large = performance == 'gpu80'
    batch_size = batch_size if batch_size is not None else ((16 if kind in ('unit', 'mel') else 8) if large else 1)
    num_workers = num_workers if num_workers is not None else (4 if large else 0)
    max_tokens = max_tokens if max_tokens is not None else (20000 if large else 2000)
    save_interval = save_interval if save_interval is not None else (50 if large else 1)
    if min(batch_size, prefetch_factor, max_tokens, update_freq, save_interval) < 1 or num_workers < 0:
        raise ValueError('invalid throughput options')
    if kind in ('unit', 'mel') and update_freq != 1:
        raise ValueError('vocoder uses real batches, not gradient accumulation')
    repository, data = Path(repository), Path(data)
    if kind == 'tt2':
        root = data / 'translatotron2/fairseq'
        command = ['{python}', '-m', 'direct_s2st.translatotron2.train',
            '--data-root', str(root), '--run-root', '{run_root}', '--device', 'cuda',
            '--model-size', model_size, '--batch-size', str(batch_size), '--max-updates', '{updates}',
            '--save-interval-updates', str(save_interval), '--num-workers', str(num_workers),
            '--prefetch-factor', str(prefetch_factor), '--update-freq', str(update_freq)]
        checkpoint = 'checkpoints/checkpoint_last.pt'
    elif kind == 's2ut':
        root = data / 's2ut/fairseq'
        cfg = load_config(repository / 'configs/s2ut/train.yaml')
        command = cfg['training']['command'][:]
        values = dict(data_root=str(root), run_root='{run_root}', max_updates='{updates}',
                      save_interval_updates=str(save_interval), seed='1')
        command = [part.format(**values) for part in command]
        command[0] = '{python}'
        command[command.index('--max-tokens')+1] = str(max_tokens)
        command[command.index('--num-workers')+1] = str(num_workers)
        command[command.index('--update-freq')+1] = str(update_freq)
        from .s2ut.preflight import validate_training
        validate_training(root, command)
        checkpoint = 'checkpoints/checkpoint_last.pt'
    elif kind in ('unit', 'mel'):
        root = data / 'common'
        generator = repository / f'configs/vocoder/{kind}-generator.json'
        command = ['{python}', '-m', 'direct_s2st.vocoders.train', '--kind', kind,
            '--common-root', str(root), '--config', str(generator), '--output-root', '{run_root}',
            '--device', 'cuda', '--max-updates', '{updates}', '--batch-size', str(batch_size),
            '--num-workers', str(num_workers), '--prefetch-factor', str(prefetch_factor),
            '--save-interval-updates', str(save_interval)]
        if kind == 'unit':
            command += ['--units-root', str(data / 's2ut/units')]
        checkpoint = 'generator.pt'
    else:
        raise ValueError('unknown recipe')
    identity_files = [str(Path(environment_lock))]
    # Prepared archives/configs/labels are immutable for a run. Hashing large
    # archives costs I/O once per session but prevents accidental data changes.
    identity_files += [str(p) for p in sorted(root.rglob('*')) if p.is_file()]
    if len(identity_files) == 1:
        raise FileNotFoundError(f'prepare data first: {root}')
    if kind in ('unit', 'mel'):
        identity_files.append(str(generator))
    if kind == 'unit':
        identity_files += [str(p) for p in sorted((data/'s2ut/units').rglob('*')) if p.is_file()]
    return dict(kind='vocoder' if kind in ('unit', 'mel') else kind, command=command,
        checkpoint=checkpoint, identity_files=identity_files,
        resume_args=['--resume'] if kind in ('unit', 'mel') else ['--restore-file', '{checkpoint}'])


def safe_relative(value):
    path = Path(value)
    if path.is_absolute() or value.startswith('/') or not path.parts or '..' in path.parts or '\\' in value or ':' in value:
        raise ValueError('unsafe snapshot path')
    return path


def publish(work, destination, identity, updates):
    """Never replace an older recovery point; marker is written last."""
    work, destination = Path(work), Path(destination)
    target = destination / f'{updates:012d}-{uuid.uuid4().hex}'
    target.mkdir(parents=True, exist_ok=False)
    files = {}
    for source in sorted(work.rglob('*')):
        if source.is_symlink():
            raise ValueError('snapshot cannot contain symlinks')
        if not source.is_file():
            continue
        name = source.relative_to(work).as_posix()
        output = target / 'files' / name
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, output)
        files[name] = sha256_file(source)
        if sha256_file(output) != files[name]:
            raise OSError('snapshot copy checksum mismatch')
    atomic_write_json(target / 'snapshot.json', dict(identity=identity, updates=updates, files=files))
    return target


def latest(destination, identity):
    for marker in sorted(Path(destination).glob('*/snapshot.json'), reverse=True):
        try:
            state = json.loads(marker.read_text(encoding='utf-8'))
        except (ValueError, OSError):
            continue
        if state['identity'] != identity:
            raise ValueError('session identity changed; use a different backup directory')
        valid = bool(state['files'])
        for name, expected in state['files'].items():
            path = marker.parent / 'files' / safe_relative(name)
            if path.is_symlink() or not path.resolve().is_relative_to((marker.parent / 'files').resolve()) or not path.is_file() or sha256_file(path) != expected:
                valid = False
                break
        if valid:
            return marker.parent, state
    return None


def restore(snapshot, work):
    directory, state = snapshot
    work = Path(work)
    # Keep incomplete local progress for inspection; never overwrite it silently.
    if work.exists():
        work.rename(work.with_name(work.name + '.interrupted-' + uuid.uuid4().hex))
    work.mkdir(parents=True)
    for name in state['files']:
        relative = safe_relative(name)
        output = work / relative
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(directory / 'files' / relative, output)
        if sha256_file(output) != state['files'][name]:
            raise OSError('restored file checksum mismatch')


def checkpoint_updates(path, kind):
    import torch
    if kind == 's2ut':
        from .s2ut.checkpoint import validate_checkpoint
        return validate_checkpoint(path)['num_updates']
    state = torch.load(path, map_location='cpu', weights_only=True)
    if kind == 'tt2':
        if state.get('format') != 'direct-s2st-tt2-v2' or not state.get('optimizer', {}).get('state'):
            raise ValueError('missing TT2 optimizer checkpoint')
    elif state.get('format') != 'direct-s2st-vocoder-v1' or not state.get('optimizer_g') or not state.get('optimizer_d'):
        raise ValueError('missing vocoder optimizer checkpoint')
    return int(state['updates'])


def run_session(config, *, work, backup, total=2, chunk=1, seconds=3600,
                resume=False, confirm_training=False, runner=subprocess.run, clock=time.monotonic,
                inspect_checkpoint=checkpoint_updates):
    work, backup = Path(work).resolve(), Path(backup).resolve()
    if min(total, chunk, seconds) <= 0 or not math.isfinite(seconds):
        raise ValueError('total, chunk and seconds must be positive')
    if total > 10 and not confirm_training:
        raise ValueError('more than 10 updates requires --confirm-training')
    if work == Path(work.anchor) or work.is_relative_to(backup) or backup.is_relative_to(work):
        raise ValueError('work and backup must be separate dedicated directories')
    corpus = os.environ.get('CORPUS_ROOT')
    if corpus and any(path.is_relative_to(Path(corpus).resolve()) or Path(corpus).resolve().is_relative_to(path) for path in (work, backup)):
        raise ValueError('outputs must be outside CORPUS_ROOT')
    if config['kind'] not in ('tt2', 's2ut', 'vocoder'):
        raise ValueError('unknown trainer kind')
    checkpoint = work / safe_relative(config['checkpoint'])
    command = config['command']
    if not isinstance(command, list) or not all(isinstance(v, str) for v in command) or '{updates}' not in command:
        raise ValueError('command must be an argument list with {updates}')
    if not config.get('identity_files'):
        raise ValueError('identity_files must include data lock and environment lock')
    identity = dict(config=config, work=str(work), inputs={p: sha256_file(Path(p)) for p in config['identity_files']})
    previous = latest(backup, identity)
    if previous and not resume:
        raise FileExistsError('backup exists; use --resume')
    if resume and not previous:
        raise FileNotFoundError('no verified recovery point; do not silently restart')
    if not resume and work.exists() and any(work.iterdir()):
        raise FileExistsError('work directory is not empty')
    completed = 0
    if previous:
        restore(previous, work)
        completed = inspect_checkpoint(checkpoint, config['kind'])
        if completed != previous[1]['updates']:
            raise ValueError('snapshot update count mismatch')
    started = clock()
    while completed < total:
        remaining = seconds - (clock() - started)
        if remaining <= 0:
            break
        target = min(total, completed + chunk)
        mapping = dict(python=sys.executable, run_root=str(work), checkpoint=str(checkpoint), updates=str(target))
        args = [part.format(**mapping) for part in command]
        if completed:
            args += [part.format(**mapping) for part in config['resume_args']]
        work.mkdir(parents=True, exist_ok=True)
        print(f'Training through update {target}; last durable update {completed}', flush=True)
        try:
            result = runner(args, check=False, timeout=remaining)
        except subprocess.TimeoutExpired:
            print('Session budget reached. Incomplete local chunk is not published.', flush=True)
            break
        if result.returncode:
            raise RuntimeError(f'training failed ({result.returncode}); previous backup retained')
        actual = inspect_checkpoint(checkpoint, config['kind'])
        if actual != target:
            raise ValueError(f'expected update {target}, checkpoint contains {actual}')
        publish(work, backup, identity, actual)
        completed = actual
        print(f'Verified backup: update {completed}', flush=True)
    return dict(status='COMPLETE' if completed >= total else 'PAUSED', durable_updates=completed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--work-root', type=Path, required=True)
    parser.add_argument('--backup-root', type=Path, required=True)
    parser.add_argument('--total-updates', type=int, default=2)
    parser.add_argument('--chunk-updates', type=int, default=1)
    parser.add_argument('--session-seconds', type=float, default=3600)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--confirm-training', action='store_true')
    args = parser.parse_args()
    result = run_session(json.loads(args.config.read_text(encoding='utf-8')), work=args.work_root,
        backup=args.backup_root, total=args.total_updates, chunk=args.chunk_updates,
        seconds=args.session_seconds, resume=args.resume, confirm_training=args.confirm_training)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
