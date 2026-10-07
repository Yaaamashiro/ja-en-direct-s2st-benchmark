"""Bounded native training sessions with immutable, verified Drive snapshots.

Only load checkpoints created by your own trusted training commands. A Drive
mount is not a transactional filesystem: completion markers and hashes detect
partial copies, but do not promise that unsynced writes survive VM deletion.
"""
import argparse
from .progress import operation, track
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
import zipfile

from .hashing import sha256_file
from .io import atomic_write_json


@operation('direct_s2st/colab: make_config')
def make_config(repository, data, environment_lock, kind='tt2', model_size='smoke', *,
                performance='smoke', batch_size=None, num_workers=None, prefetch_factor=2,
                max_tokens=None, update_freq=1, save_interval=None, optimize=False,
                cache_gb=8, adaptive_batch=False, precision='default'):
    """Explicit throughput knobs; gpu80 is a starting point, not a VRAM guarantee."""
    from .config import load_config
    if performance not in ('smoke', 'gpu80'):
        raise ValueError('unknown performance preset')
    large = performance == 'gpu80'
    if not math.isfinite(cache_gb) or cache_gb < 0 or precision not in ('default', 'fp32', 'bf16'):
        raise ValueError('invalid training cache/precision setting')
    if not optimize and (adaptive_batch or precision != 'default'):
        raise ValueError('batch/precision tuning requires optimize=True')
    if kind in ('unit', 'mel') and adaptive_batch:
        raise ValueError('GAN batch adaptation is unsupported; keep its two-optimizer update fixed')
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
        if precision != 'default':
            command.remove('--fp16')
            if precision == 'bf16':
                command.append('--bf16')
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
        from .s2ut.unit_storage import manifest_paths
        identity_files += [str(data / 's2ut/units/unit-lock.json')]
        identity_files += [str(p) for p in manifest_paths(data / 's2ut/units')]
    result = dict(kind='vocoder' if kind in ('unit', 'mel') else kind, command=command,
        checkpoint=checkpoint, identity_files=identity_files,
        resume_args=['--resume'] if kind in ('unit', 'mel') else ['--restore-file', '{checkpoint}'])
    if optimize:
        result['runtime'] = dict(cache_gb=cache_gb, readers=max(1, num_workers) if kind == 's2ut' else 1,
                                 adaptive_batch=adaptive_batch, precision=precision)
    return result


@operation('training: stage all inputs before starting GPU work')
def stage_training(config, work):
    from .drive_staging import (stage_audio, ensure_local, copy_verified, register, raw_sha)
    from .preparation import file_stamp
    from .journal import digest
    files = [Path(p) for p in config['identity_files']]
    roots = [p.parent for p in files if p.name == 'dataset-lock.json']
    if not roots:
        roots = [p.parent.parent.parent / 'common' for p in files if p.name == 'data-lock.json']
    if not roots or len({str(p) for p in roots}) != 1:
        raise ValueError('cannot locate common data for local training staging')
    common = roots[0]
    local = ensure_local(Path(work).parent / '.inputs' / digest([str(common), config['kind']])[:16])
    vocoder = config['kind'] == 'vocoder'
    command = config['command']
    if '--units-root' in command:
        from .s2ut.unit_storage import records
        unit_root = Path(command[command.index('--units-root') + 1])
        unit_rows = records(unit_root)
        if unit_rows:
            files += [Path(row['units_original_path']) for row in unit_rows.values()
                      if row['split'] == 'train' and row.get('units_storage') != 'inline-v1']
        else:
            from .io import read_jsonl
            files += [unit_root / 'train/original' / (row['pair_id'] + '.units')
                      for row in read_jsonl(common / 'train.jsonl')]
    rows = stage_audio(common, common.parent / '.drive-audio-packs', local / 'audio',
                       splits=('train',) if vocoder else ('train', 'dev', 'test'),
                       languages=('en',) if vocoder else ('ja',))
    for source in track(files, 'training: stage metadata and whole Mel ZIPs'):
        if source.is_symlink():
            raise ValueError('training inputs must not be symlinks')
        stamp = file_stamp(source)
        target = local / 'prepared' / digest(str(source))[:24] / source.name
        receipt = target.with_suffix(target.suffix + '.receipt.json')
        reuse = False
        if receipt.is_file() and target.is_file():
            saved = json.loads(receipt.read_text(encoding='utf-8'))
            reuse = saved.get('stamp') == stamp and raw_sha(target) == saved['sha256']
        if not reuse:
            checksum = copy_verified(source, target)
            if file_stamp(source) != stamp:
                raise ValueError('training input changed during staging')
            atomic_write_json(receipt, dict(stamp=stamp, sha256=checksum), overwrite=True)
        register(rows, source, target, stamp)
    return rows, local


def safe_relative(value):
    path = Path(value)
    if path.is_absolute() or value.startswith('/') or not path.parts or '..' in path.parts or '\\' in value or ':' in value:
        raise ValueError('unsafe snapshot path')
    return path


@operation('direct_s2st/colab: publish')
def publish(work, destination, identity, updates):
    """Never replace an older recovery point; marker is written last."""
    work, destination = Path(work), Path(destination)
    if os.environ.get('S2ST_DRIVE_SAFE') == '1':
        return publish_packed(work, destination, identity, updates)
    target = destination / f'{updates:012d}-{uuid.uuid4().hex}'
    target.mkdir(parents=True, exist_ok=False)
    files = {}
    for source in track(sorted(work.rglob('*')), 'backup: copy and verify'):
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


@operation('backup: one verified ZIP with capacity budget')
def publish_packed(work, destination, identity, updates):
    from .drive_staging import raw_sha, capacity
    maximum = float(os.environ.get('S2ST_BACKUP_MAX_GB', '20')) * 1024**3
    if not math.isfinite(maximum) or maximum <= 0:
        raise ValueError('backup budget must be finite and positive')
    target = destination / f'{updates:012d}-{uuid.uuid4().hex}'
    destination.mkdir(parents=True, exist_ok=True)
    used = sum(p.stat().st_size for p in destination.rglob('*') if p.is_file())
    with tempfile.TemporaryDirectory(prefix='snapshot-pack-', dir=work.parent) as local:
        local_zip = Path(local) / 'snapshot.zip'
        files = {}
        with zipfile.ZipFile(local_zip, 'w', zipfile.ZIP_STORED) as pack:
            for source in track(sorted(work.rglob('*')), 'backup: pack local files'):
                if source.is_symlink():
                    raise ValueError('snapshot cannot contain symlinks')
                if source.is_file():
                    name = source.relative_to(work).as_posix()
                    safe_relative(name)
                    files[name] = raw_sha(source)
                    pack.write(source, name)
        size = local_zip.stat().st_size
        if used + size > maximum:
            raise RuntimeError(f'Backup budget exceeded: {used + size} > {int(maximum)} bytes. '
                               'Existing recovery points are preserved; increase TRAIN_BACKUP_MAX_GB explicitly.')
        capacity(destination, size)
        target.mkdir()
        temporary = target / 'snapshot.zip.tmp'
        shutil.copyfile(local_zip, temporary)
        checksum = raw_sha(local_zip)
        if raw_sha(temporary) != checksum:
            raise OSError('snapshot ZIP readback checksum mismatch')
        os.replace(temporary, target / 'snapshot.zip')
        atomic_write_json(target / 'snapshot.json',
                          dict(identity=identity, updates=updates, files=files,
                               storage='zip-v1', zip_sha256=checksum))
    return target


@operation('direct_s2st/colab: latest')
def latest(destination, identity):
    for marker in sorted(Path(destination).glob('*/snapshot.json'), reverse=True):
        try:
            state = json.loads(marker.read_text(encoding='utf-8'))
        except (ValueError, OSError):
            continue
        if state['identity'] != identity:
            raise ValueError('session identity changed; use a different backup directory')
        valid = bool(state['files'])
        if state.get('storage') == 'zip-v1':
            archive = marker.parent / 'snapshot.zip'
            if any(not safe_relative(name) for name in state['files']):
                raise ValueError('unsafe snapshot paths')
            if archive.is_symlink():
                raise ValueError('snapshot archive cannot be a symlink')
            if valid and archive.is_file() and sha256_file(archive) == state['zip_sha256']:
                return marker.parent, state
            continue
        for name, expected in track(state['files'].items(), 'backup: verify snapshot'):
            path = marker.parent / 'files' / safe_relative(name)
            if path.is_symlink() or not path.resolve().is_relative_to((marker.parent / 'files').resolve()) or not path.is_file() or sha256_file(path) != expected:
                valid = False
                break
        if valid:
            return marker.parent, state
    return None


@operation('direct_s2st/colab: restore')
def restore(snapshot, work):
    directory, state = snapshot
    work = Path(work)
    # Keep incomplete local progress for inspection; never overwrite it silently.
    if work.exists():
        work.rename(work.with_name(work.name + '.interrupted-' + uuid.uuid4().hex))
    work.mkdir(parents=True)
    if state.get('storage') == 'zip-v1':
        from .drive_staging import copy_verified
        archive = work / '.restore.zip'
        copy_verified(directory / 'snapshot.zip', archive, state['zip_sha256'])
        with zipfile.ZipFile(archive) as pack:
            if set(pack.namelist()) != set(state['files']) or len(pack.namelist()) != len(state['files']):
                raise ValueError('snapshot ZIP member mismatch')
            for name in track(state['files'], 'backup: restore local ZIP'):
                output = work / safe_relative(name)
                output.parent.mkdir(parents=True, exist_ok=True)
                with pack.open(name) as source, output.open('wb') as target:
                    shutil.copyfileobj(source, target)
                if sha256_file(output) != state['files'][name]:
                    raise OSError('restored file checksum mismatch')
        archive.unlink()
        return
    for name in track(state['files'], 'backup: restore and verify'):
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


@operation('direct_s2st/colab: run_session')
def run_session(config, *, work, backup, total=2, chunk=1, seconds=3600,
                resume=False, auto_resume=False, confirm_training=False, restore_only=False, runner=subprocess.run, clock=time.monotonic,
                inspect_checkpoint=checkpoint_updates, _staged=False):
    work, backup = Path(work).resolve(), Path(backup).resolve()
    if restore_only and not resume:
        raise ValueError('restore-only requires resume')
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
    if os.environ.get('S2ST_DRIVE_SAFE') == '1' and not _staged:
        from .drive_staging import active_map
        rows, local = stage_training(config, work)
        with active_map(rows, local):
            return run_session(config, work=work, backup=backup, total=total, chunk=chunk,
                seconds=seconds, resume=resume, auto_resume=auto_resume,
                confirm_training=confirm_training, restore_only=restore_only,
                runner=runner, clock=clock, inspect_checkpoint=inspect_checkpoint, _staged=True)
    checkpoint = work / safe_relative(config['checkpoint'])
    command = config['command']
    if not isinstance(command, list) or not all(isinstance(v, str) for v in command) or '{updates}' not in command:
        raise ValueError('command must be an argument list with {updates}')
    if not config.get('identity_files'):
        raise ValueError('identity_files must include data lock and environment lock')
    identity = dict(config=config, work=str(work), inputs={p: sha256_file(Path(p)) for p in config['identity_files']})
    previous = latest(backup, identity)
    if auto_resume:
        if resume or restore_only:
            raise ValueError('auto-resume cannot be combined with explicit resume/restore-only')
        if not previous and any(backup.glob('*/snapshot.json')):
            raise ValueError('backups exist but none are valid; inspect them before restarting')
        resume = previous is not None
        if not resume:
            print('[recovery] no verified snapshot; starting at update 0. '
                  'Incomplete Drive copies are retained.', file=sys.stderr, flush=True)
            if work.exists() and any(work.iterdir()):
                retained = work.with_name(work.name + '.interrupted-' + uuid.uuid4().hex)
                work.rename(retained)
                print(f'[recovery] retained unpublished local files: {retained}',
                      file=sys.stderr, flush=True)
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
    if restore_only:
        return dict(status='RESTORED', durable_updates=completed)
    if completed >= total:
        return dict(status='COMPLETE', durable_updates=completed)
    if config.get('runtime'):
        from .session_runtime import run_continuous
        return run_continuous(config, work=work, backup=backup, identity=identity,
                              completed=completed, total=total, seconds=seconds,
                              inspect_checkpoint=inspect_checkpoint, publish=publish)
    work.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.s2st-preflight-', dir=work.parent) as temporary:
        from .vocoders.verification import preflight_command
        mapping = dict(python=sys.executable, run_root=str(work), checkpoint=str(checkpoint), updates=str(total))
        verified_command = preflight_command([part.format(**mapping) for part in command], temporary)
        return _run_chunks(config, work, backup, checkpoint, identity, completed, total, chunk,
                           seconds, runner, clock, inspect_checkpoint,
                           verified_command[len(command):])


def _run_chunks(config, work, backup, checkpoint, identity, completed, total, chunk,
                seconds, runner, clock, inspect_checkpoint, preflight_args):
    started = clock()
    while completed < total:
        remaining = seconds - (clock() - started)
        if remaining <= 0:
            break
        target = min(total, completed + chunk)
        mapping = dict(python=sys.executable, run_root=str(work), checkpoint=str(checkpoint), updates=str(target))
        args = [part.format(**mapping) for part in config['command']] + preflight_args
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
    os.environ.setdefault('S2ST_DRIVE_SAFE', '1')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--work-root', type=Path, required=True)
    parser.add_argument('--backup-root', type=Path, required=True)
    parser.add_argument('--total-updates', type=int, default=2)
    parser.add_argument('--chunk-updates', type=int, default=1)
    parser.add_argument('--session-seconds', type=float, default=3600)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--auto-resume', action='store_true',
                        help='Resume a verified snapshot, or explicitly report starting at zero; retain partial copies')
    parser.add_argument('--restore-only', action='store_true', help='Verify and restore a Drive snapshot without training')
    parser.add_argument('--confirm-training', action='store_true')
    args = parser.parse_args()
    result = run_session(json.loads(args.config.read_text(encoding='utf-8')), work=args.work_root,
        backup=args.backup_root, total=args.total_updates, chunk=args.chunk_updates,
        seconds=args.session_seconds, resume=args.resume, auto_resume=args.auto_resume,
        confirm_training=args.confirm_training,
        restore_only=args.restore_only)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
