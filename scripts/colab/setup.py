"""Run once per Colab VM. Isolated Python 3.13, no Docker or global torch edits."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
PREFIX = Path('/content/s2st-runtime')
FAIRSEQ = '3d262bb25690e4eb2e7d3c1309b1e9c406ca4b99'
ESPEAK = '4870adfa25b1a32b4361592f1be8a40337c58d6c'
PYTHON_VERSION = '3.13.7'


def run(*args, **kwargs):
    subprocess.run([str(a) for a in args], check=True, **kwargs)


def main():
    if not Path('/content').is_dir() or sys.platform != 'linux':
        raise RuntimeError('This bootstrap is for a Linux Colab runtime')
    run('apt-get', 'update', '-qq')
    run('apt-get', 'install', '-y', 'build-essential', 'cmake', 'git', 'patch', 'libsndfile1', 'sox')
    run(sys.executable, '-m', 'pip', 'install', 'uv==0.8.22')
    run(sys.executable, '-m', 'uv', 'python', 'install', PYTHON_VERSION)
    python = PREFIX / 'venv/bin/python'
    if not python.exists():
        run(sys.executable, '-m', 'uv', 'venv', '--seed', '--python', PYTHON_VERSION, PREFIX / 'venv')
    actual = subprocess.check_output([str(python), '-c', 'import platform; print(platform.python_version())'], text=True).strip()
    if actual != PYTHON_VERSION:
        raise RuntimeError('Old runtime found; start a fresh Colab VM, do not reuse the 3.10 environment')
    os.environ['PIP_CONSTRAINT'] = str(ROOT / 'requirements/colab313.txt')
    run(python, '-m', 'pip', 'install', 'pip==25.2', 'setuptools==80.9.0', 'wheel==0.45.1')
    run(python, '-m', 'pip', 'install', 'torch==2.7.1', 'torchaudio==2.7.1',
        '--index-url', 'https://download.pytorch.org/whl/cu126')
    run(python, '-m', 'pip', 'install', '-e', ROOT, '-r', ROOT / 'requirements/colab313.txt')
    run('git', '-C', ROOT, 'submodule', 'update', '--init', '--recursive')
    source = ROOT / 'third_party/fairseq'
    revision = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain'], text=True).strip()
    if revision != FAIRSEQ or dirty:
        raise RuntimeError('fairseq must be pristine at the pinned commit')
    copied = PREFIX / 'fairseq'
    if not copied.exists():
        shutil.copytree(source, copied, ignore=shutil.ignore_patterns('.git'))
    patch = ROOT / 'patches/fairseq/0001-librosa-mel-keywords.patch'
    probe = subprocess.run(['patch', '--dry-run', '--directory='+str(copied), '-p1', '--forward', '--input='+str(patch)], capture_output=True)
    if probe.returncode == 0:
        run('patch', '--directory='+str(copied), '-p1', '--forward', '--input='+str(patch))
    else:
        run('patch', '--dry-run', '--directory='+str(copied), '-p1', '--reverse', '--input='+str(patch))
    run(python, ROOT / 'scripts/colab/compat313.py', copied)
    build_env = dict(os.environ, MAX_JOBS='2')
    # These recipes use PyTorch CUDA, not fairseq's optional NAT CUDA kernels.
    build_env.pop('CUDA_HOME', None)
    run(python, '-m', 'pip', 'install', '--no-build-isolation', '-e', copied, env=build_env)
    run(python, '-m', 'pip', 'check')
    espeak = PREFIX / 'espeak-src'
    if not espeak.exists():
        run('git', 'init', espeak)
        run('git', '-C', espeak, 'remote', 'add', 'origin', 'https://github.com/espeak-ng/espeak-ng.git')
    run('git', '-C', espeak, 'fetch', '--depth', '1', 'origin', ESPEAK)
    run('git', '-C', espeak, 'checkout', '--detach', ESPEAK)
    run('cmake', '-S', espeak, '-B', espeak / 'build', '-DCMAKE_BUILD_TYPE=Release',
        '-DCMAKE_INSTALL_PREFIX='+str(PREFIX / 'espeak'), '-DUSE_LIBSONIC=OFF',
        '-DUSE_MBROLA=OFF', '-DUSE_SPEECHPLAYER=OFF')
    run('cmake', '--build', espeak / 'build', '--parallel', '2')
    run('cmake', '--install', espeak / 'build')
    env = dict(os.environ, PYTHONPATH=str(copied))
    run(python, ROOT / 'scripts/colab/check313.py', '--require-extensions', env=env)
    run(python, '-c', 'import torch, torchaudio; import examples.speech_synthesis.data_utils; '
        'from fairseq.data.audio.audio_utils import get_mel_filters; '
        'assert get_mel_filters(16000,512,80,0,8000).shape == (80,257); '
        'assert torch.cuda.is_available(), "Select a GPU runtime"; print(torch.cuda.get_device_name())', env=env)
    freeze = subprocess.check_output([str(python), '-m', 'pip', 'freeze'], text=True)
    lock = dict(python=actual, fairseq=FAIRSEQ, espeak=ESPEAK, packages=freeze,
        compatibility_sha256=hashlib.sha256((ROOT/'scripts/colab/compat313.py').read_bytes()).hexdigest(),
        requirements_sha256=hashlib.sha256((ROOT/'requirements/colab313.txt').read_bytes()).hexdigest(),
        repository=subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip())
    (PREFIX / 'environment.json').write_text(json.dumps(lock, indent=2), encoding='utf-8')
    print('Runtime ready:', python)


if __name__ == '__main__':
    main()
