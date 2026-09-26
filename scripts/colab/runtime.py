"""Restore the disposable Colab runtime, never prepared data or experiment locks."""
import json
import os
from pathlib import Path
import subprocess
import sys


def configured_environment(repo, persistent, prefix):
    def prepend(variable, values):
        parts = values + os.environ.get(variable, '').split(os.pathsep)
        return os.pathsep.join(dict.fromkeys(part for part in parts if part))
    return {
        'EXPERIMENT_DATA_ROOT': str(persistent / 'data'),
        'RUNS_ROOT': str(persistent / 'runs'),
        'CACHE_ROOT': str(persistent / 'cache'),
        'S2ST_CONFIG_ROOT': str(repo / 'configs'),
        'PYTHONPATH': str(prefix / 'fairseq'),
        'PIP_CONSTRAINT': str(repo / 'requirements/colab310.txt'),
        'ESPEAK_DATA_PATH': str(prefix / 'espeak/share/espeak-ng-data'),
        'PATH': prepend('PATH', [str(prefix / 'venv/bin'), str(prefix / 'espeak/bin')]),
        'LD_LIBRARY_PATH': prepend('LD_LIBRARY_PATH', [str(prefix / 'espeak/lib')]),
        'PYTHONUNBUFFERED': '1',
    }


def ready(repo, prefix):
    required = ('venv/bin/python', 'fairseq/setup.py', 'espeak/bin/espeak-ng',
                'espeak/share/espeak-ng-data/en_dict', 'environment.json')
    if not all((prefix / path).is_file() for path in required):
        return False
    try:
        lock = json.loads((prefix / 'environment.json').read_text(encoding='utf-8'))
        revision = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    except (ValueError, OSError, subprocess.CalledProcessError):
        return False
    return isinstance(lock, dict) and lock.get('repository') == revision and lock.get('inference_dependencies') is True


def ensure_runtime(repo, persistent, run, *, require_gpu=False, prefix=Path('/content/s2st-runtime')):
    repo, persistent, prefix = Path(repo), Path(persistent), Path(prefix)
    # setup needs the corpus guard even before the virtual environment exists.
    env = configured_environment(repo, persistent, prefix)
    if not ready(repo, prefix):
        print('ローカル環境が未構築です。セル3相当の構築を自動実行します。', flush=True)
        args = [sys.executable, repo / 'scripts/colab/setup.py', '--inference']
        if not require_gpu:
            args.append('--allow-cpu')
        run(*args)
        if not ready(repo, prefix):
            raise RuntimeError('Runtime setup did not produce a complete environment')
    os.environ.update(env)
    python = prefix / 'venv/bin/python'
    run(python, '-c',
        'import torch, torchaudio, librosa, fairseq; '
        f'assert {not require_gpu!r} or torch.cuda.is_available(), '
        '"この工程はGPUが必要です。ColabでGPUへ切り替え、セル1とこのセルを再実行してください"; '
        'print("Runtime checked:", torch.cuda.get_device_name() if torch.cuda.is_available() else "CPU")')
    run(prefix / 'espeak/bin/espeak-ng', '--version')
    return python
