"""Optional CPU reporter environment and fail-soft process supervision."""
from contextlib import contextmanager
import os
from pathlib import Path
import runpy
import signal
import subprocess
import sys
import uuid


def _stop(process, stop):
    stop.touch()
    try:
        process.wait(timeout=25)
    except subprocess.TimeoutExpired:
        if os.name == 'posix':
            os.killpg(process.pid, signal.SIGTERM)
        else:
            process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            if os.name == 'posix':
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
            process.wait(timeout=5)


def prepare(repo, *, root=Path('/content/s2st-wandb-runtime')):
    """Never install W&B into the scientific runtime. Missing auth is optional."""
    try:
        progress = runpy.run_path(str(Path(repo)/'src/direct_s2st/progress.py'))['Progress']
        with progress('baseline W&B: isolated CPU reporter setup'):
            key = os.environ.get('WANDB_API_KEY')
            if not key:
                from google.colab import userdata
                key = userdata.get('WANDB_API_KEY')
            if not key:
                raise ValueError('missing W&B credential')
            python = Path(root) / 'bin/python'
            if not python.is_file():
                subprocess.run([sys.executable, '-m', 'venv', str(root)], check=True, timeout=30)
            subprocess.run([str(python), '-m', 'pip', '--isolated', 'install', '--no-input',
                            '--only-binary=:all:', '--disable-pip-version-check',
                            '-r', str(Path(repo) / 'requirements/wandb.txt')], check=True, timeout=120)
            return dict(python=str(python), key=key)
    except Exception as error:
        print('[baseline-wandb] disabled=' + type(error).__name__ +
              '; training remains enabled; set the WANDB_API_KEY Colab secret and retry sync',
              file=sys.stderr, flush=True)
        return None


@contextmanager
def reporter(repo, source, state, python, entity, project, *, key=None, group='',
             baseline_run_key='', existing_run_id='', registration='', audit_journals=False):
    """A separate interpreter owns all SDK/network work; trainer exit wins."""
    process = stop = None
    try:
        if python:
            state = Path(state)
            state.mkdir(parents=True, exist_ok=True)
            stop = state / ('stop-' + uuid.uuid4().hex)
            env = os.environ.copy()
            env.pop('PYTHONPATH', None)
            env.pop('S2ST_WANDB_API_KEY', None)
            if key:
                env['WANDB_API_KEY'] = key
            command = [str(python), str(Path(repo) / 'scripts/colab/baseline_wandb.py'), 'watch',
                       '--source', str(source), '--state', str(state), '--entity', entity,
                       '--project', project, '--stop-file', str(stop)]
            for flag, value in (('--group', group), ('--baseline-run-key', baseline_run_key),
                                ('--existing-run-id', existing_run_id), ('--registration', registration)):
                if value:
                    command.extend((flag, str(value)))
            if audit_journals:
                command.append('--audit-journals')
            process = subprocess.Popen(command, env=env, start_new_session=(os.name == 'posix'))
    except Exception as error:
        print('[baseline-wandb] reporter unavailable=' + type(error).__name__, file=sys.stderr, flush=True)
    try:
        yield
    finally:
        if process is not None:
            try:
                # Bounded shutdown; unsent reports/outbox remain resumable.
                _stop(process, stop)
            except Exception as error:
                print('[baseline-wandb] deferred shutdown=' + type(error).__name__, file=sys.stderr, flush=True)
            finally:
                try:
                    if stop and stop.exists():
                        stop.unlink()
                except OSError:
                    pass  # Reporter cleanup cannot replace the trainer's exit.
