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
        deadline = time.monotonic() + seconds
        env = dict(os.environ, S2ST_TRAIN_OPTIMIZE='1',
                   S2ST_TRAIN_STAGING=str(staging), S2ST_TRAIN_DEADLINE=str(deadline),
                   S2ST_TRAIN_CACHE=str(temporary / 'cache'),
                   S2ST_TRAIN_CACHE_GB=str(settings['cache_gb']),
                   S2ST_TRAIN_READERS=str(settings['readers']),
                   S2ST_TRAIN_PRECISION=settings['precision'],
                   S2ST_TRAIN_ADAPTIVE_BATCH='1' if settings['adaptive_batch'] else '0')
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
            return update
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
                    if process.poll() is None and now > deadline + 30 and terminated_at is None:
                        forced = True
                        terminated_at = now
                        process.terminate()
                    if process.poll() is None and terminated_at is not None and now > terminated_at + 5:
                        process.kill()
                    if process.poll() is not None and future is None and not ready:
                        break
                    time.sleep(.25)
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        if process.returncode and not forced:
            raise RuntimeError(f'training failed ({process.returncode}); durable update {completed} retained')
        if not forced and time.monotonic() < deadline and completed < total:
            raise RuntimeError('trainer exited without publishing the requested final checkpoint')
    return dict(status='COMPLETE' if completed >= total else 'PAUSED', durable_updates=completed)
