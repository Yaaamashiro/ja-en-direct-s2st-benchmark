"""One resident trainer with one bounded, verified Drive upload worker."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

from .progress import operation


@operation('training: resident session and asynchronous backup')
def run_continuous(config, *, work, backup, identity, completed, total, seconds,
                   inspect_checkpoint, publish, popen=subprocess.Popen):
    from .translatotron2.monitor import absolute_update_limit
    report = Path(work)/'training-monitor.json'
    policy = json.loads(report.read_text(encoding='utf-8'))['policy'] if report.is_file() else None
    total = absolute_update_limit(config['kind'], total, policy)
    if completed >= total:
        return dict(status='COMPLETE', durable_updates=completed)
    work.mkdir(parents=True, exist_ok=True)
    settings = config['runtime']
    mapping = dict(python=sys.executable, run_root=str(work),
                   checkpoint=str(work / config['checkpoint']), updates=str(total))
    args = [part.format(**mapping) for part in config['command']]
    if completed:
        args += [part.format(**mapping) for part in config['resume_args']]
    with tempfile.TemporaryDirectory(prefix='.s2st-training-', dir=work.parent) as temporary:
        temporary = Path(temporary).resolve()
        staging = temporary / 'snapshots'
        staging.mkdir()
        from .vocoders.verification import preflight_command
        args = preflight_command(args, temporary)
        # Input verification is resumable preparation, not optimization time.
        deadline = time.monotonic() + seconds if seconds else None
        env = dict(os.environ, S2ST_TRAIN_OPTIMIZE='1',
                   S2ST_TRAIN_STAGING=str(staging),
                   S2ST_TRAIN_CACHE=str(temporary / 'cache'),
                   S2ST_TRAIN_CACHE_GB=str(settings['cache_gb']),
                   S2ST_TRAIN_READERS=str(settings['readers']),
                   S2ST_TRAIN_PRECISION=settings['precision'],
                   S2ST_TRAIN_FIXED_MICROBATCH=str(settings.get('fixed_microbatch', 0)),
                   S2ST_TRAIN_ADAPTIVE_BATCH='1' if settings['adaptive_batch'] else '0')
        # Do not inherit a deadline from a previous bounded session. This also
        # disables cooperative stopping in all three trainers when unbounded.
        env.pop('S2ST_TRAIN_DEADLINE', None)
        if deadline is not None:
            env['S2ST_TRAIN_DEADLINE'] = str(deadline)
        print('[session] ' + ('no time limit; training until requested updates complete'
              if deadline is None else f'training time limit={seconds}s (includes dev validation)'),
              file=sys.stderr, flush=True)
        if os.environ.get('S2ST_DRIVE_SAFE') == '1':
            env.update(S2ST_TRAIN_STRICT_LOCAL='1',
                       S2ST_BACKUP_MIN_SECONDS=os.environ.get('S2ST_BACKUP_MIN_SECONDS', '600'))
        process = popen(args, env=env)
        future, current = None, None
        forced = False
        terminated_at = None
        reported = time.monotonic()
        learned = completed
        def upload(directory):
            state = json.loads((directory / 'ready.json').read_text(encoding='utf-8'))
            update = int(state['updates'])
            actual = inspect_checkpoint(directory / 'files' / config['checkpoint'], config['kind'])
            if not completed < update <= total or actual != update:
                raise ValueError('invalid immutable checkpoint update count')
            publish(directory / 'files', backup, identity, update)
            if config['kind'] in ('tt2', 's2ut') and (directory/'files/training-monitor.json').is_file():
                from .translatotron2.monitor import mirror_reports
                mirror_reports(directory/'files', backup)
            return update
        failure = None
        try:
            with ThreadPoolExecutor(max_workers=1) as uploader:
                while True:
                    if future is not None and future.done():
                        completed = future.result()
                        # Only our own acknowledged staging copy is removed.
                        if current.parent != staging or not current.resolve().is_relative_to(temporary):
                            raise ValueError('unsafe staging cleanup')
                        shutil.rmtree(current)
                        future = None
                        print(f'[recovery] learned_at_least={learned} durable_updates={completed}', file=sys.stderr, flush=True)
                    ready = sorted(staging.glob('*/ready.json'))
                    local_progress = work/'training-local-progress.json'
                    if local_progress.is_file():
                        try:
                            value = json.loads(local_progress.read_text())
                            if type(value.get('updates')) is int and completed <= value['updates'] <= total:
                                learned = max(learned, value['updates'])
                        except (ValueError, OSError):
                            pass  # A progress display is not a trusted resume checkpoint.
                    if ready:
                        learned = max(learned, *(int(json.loads(p.read_text())['updates']) for p in ready))
                    if future is None and ready:
                        current = ready[0].parent
                        future = uploader.submit(upload, current)
                    now = time.monotonic()
                    if now-reported >= 10:
                        print(f'[recovery] learned_at_least={learned} durable_updates={completed} '
                              f'pending_snapshots={len(ready)}', file=sys.stderr, flush=True)
                        reported = now
                    if deadline is not None and process.poll() is None and now > deadline + 30 and terminated_at is None:
                        forced = True
                        terminated_at = now
                        process.terminate()
                    if process.poll() is None and terminated_at is not None and now > terminated_at + 5:
                        process.kill()
                    if process.poll() is not None and future is None and not ready:
                        break
                    time.sleep(.25)
        except BaseException as error:
            failure = error
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        if failure is not None:
            # Do not let TemporaryDirectory discard acknowledged immutable
            # copies when Drive/upload verification failed. The writer is stopped.
            ready = list(staging.glob('*/ready.json'))
            if ready:
                retained = Path(tempfile.mkdtemp(prefix='.pending-upload-', dir=work.parent))
                for marker in ready:
                    source = marker.parent.resolve()
                    if source.parent != staging.resolve() or not source.is_relative_to(temporary):
                        raise ValueError('unsafe pending backup retention')
                    source.rename(retained/source.name)
                print(f'[recovery] unpublished_immutable_copies_preserved={retained}', file=sys.stderr, flush=True)
            raise failure
        if process.returncode and not forced:
            # The writer has exited. Verify and freeze its last saved update,
            # even when the normal ten-minute Drive upload interval deferred it.
            checkpoint = work/config['checkpoint']
            if checkpoint.is_file():
                local = inspect_checkpoint(checkpoint, config['kind'])
                if completed < local <= total:
                    from .recovery import freeze_files
                    retained = Path(tempfile.mkdtemp(prefix='.pending-recovery-', dir=work.parent))
                    freeze_files(work, checkpoint, retained/'files')
                    print(f'[recovery] trainer_failed local_update={local} force_backup=true retained={retained}',
                          file=sys.stderr, flush=True)
                    publish(retained/'files', backup, identity, local)
                    completed = local
                    print(f'[recovery] error_exit durable_updates={completed} local_copy_preserved={retained}',
                          file=sys.stderr, flush=True)
            raise RuntimeError(f'training failed ({process.returncode}); durable update {completed} retained')
        if not forced:
            from .translatotron2.monitor import early_stop_result
            result = early_stop_result(work, config['kind'], completed)
            if result is not None:
                return result
        if not forced and completed < total and (deadline is None or time.monotonic() < deadline):
            raise RuntimeError('trainer exited without publishing the requested final checkpoint')
    return dict(status='COMPLETE' if completed >= total else 'PAUSED', durable_updates=completed)
