"""Optional CPU reporter environment and fail-soft process supervision."""
from contextlib import contextmanager
from hashlib import sha256
import os
from pathlib import Path
import re
import runpy
import signal
import subprocess
import sys
import tempfile
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


def _setup_command(stage, command, env):
    try:
        result = subprocess.run(command, env=env, capture_output=True, text=True,
                                encoding='utf-8', errors='replace', timeout=120)
    except (OSError, subprocess.TimeoutExpired) as error:
        raise RuntimeError(f'{stage} failed ({type(error).__name__})') from None
    if result.returncode:
        output = (result.stderr or result.stdout).strip()
        # Package manager errors can contain credential-bearing URLs.
        output = re.sub(r'https?://[^\s\'"]+', '<URL>', output)
        raise RuntimeError(f'{stage} failed (exit {result.returncode})\n{output[-2000:]}')


def _reporter_python(repo, root):
    requirements = repo/'requirements/wandb.txt'
    constraints = requirements.with_name('wandb-constraints.txt')
    identity = sha256(requirements.read_bytes() + b'\0' + constraints.read_bytes()
                      + repr((sys.executable, sys.version)).encode()).hexdigest()[:20]
    cache = root/'environments'/identity
    cache.mkdir(parents=True, exist_ok=True)
    executable = Path('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(('PIP_', 'PYTHON', 'WANDB_', 'S2ST_WANDB_'))}
    env['PIP_CONFIG_FILE'] = os.devnull

    def probe(directory):
        python = directory/executable
        _setup_command('Verify isolated W&B imports', [str(python), '-I', '-c',
            'import sys, wandb, filelock; from pathlib import Path; '
            'assert Path(sys.prefix).resolve() == Path(sys.argv[1]).resolve() != Path(sys.base_prefix).resolve()',
            str(directory)], env)
        return str(python)

    for directory in sorted(cache.glob('venv-*')):
        if directory.is_symlink() or not (directory/'.ready').is_file():
            continue
        try:
            return probe(directory)
        except RuntimeError:
            continue

    # Seed with the host's pip. Colab's system Python may lack a working ensurepip.
    # An interrupted setup leaves no ready marker and is never selected on retry.
    directory = Path(tempfile.mkdtemp(prefix='venv-', dir=cache))
    _setup_command('Create isolated CPU environment',
                   [sys.executable, '-I', '-m', 'venv', '--without-pip', str(directory)], env)
    installer = [sys.executable, '-I', '-m', 'pip', '--isolated', '--python', str(directory)]
    _setup_command('Install pinned W&B wheels', installer + [
        'install', '--no-input', '--only-binary=:all:', '--disable-pip-version-check',
        '--index-url', 'https://pypi.org/simple', '-r', str(requirements)], env)
    _setup_command('Check W&B dependencies', installer + ['check'], env)
    python = probe(directory)
    (directory/'.ready').touch()
    return python


def prepare(repo, *, root=Path('/content/s2st-wandb-runtime'), required=False):
    """Build a separate CPU environment. Optional training reporting fails softly."""
    try:
        progress = runpy.run_path(str(Path(repo)/'src/direct_s2st/progress.py'))['Progress']
        with progress('baseline W&B: isolated CPU reporter setup'):
            key = os.environ.get('WANDB_API_KEY')
            if not key:
                try:
                    from google.colab import userdata
                    key = userdata.get('WANDB_API_KEY')
                except Exception:
                    raise RuntimeError('Cannot read Colab secret WANDB_API_KEY. Check notebook access.') from None
            if not key:
                raise RuntimeError('Colab secret WANDB_API_KEY is empty.')
            python = _reporter_python(Path(repo), Path(root))
            print(f'[baseline-wandb] CPU Python: {python}', file=sys.stderr, flush=True)
            return dict(python=python, key=key)
    except Exception as error:
        if required:
            raise
        print(f'[baseline-wandb] reporting unavailable; training continues: {error}',
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
