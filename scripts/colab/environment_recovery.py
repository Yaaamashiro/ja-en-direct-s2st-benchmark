"""Restore actual dependency versions; never substitute an old environment hash."""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))
from direct_s2st.io import atomic_write_bytes, atomic_write_text
from direct_s2st.progress import operation, track

EDITABLE = {'fairseq', 'ja-en-direct-s2st-benchmark'}


def canonical(name):
    return re.sub(r'[-_.]+', '-', name).lower()


def freeze_pins(freeze):
    result = {}
    for line in freeze.splitlines():
        if not line or line.startswith('#'):
            continue
        if line.startswith('-e '):
            continue  # Editable paths are restored by the verified repository setup.
        match = re.fullmatch(r'([A-Za-z0-9_.-]+)==([A-Za-z0-9_.+!-]+)', line)
        if not match:
            raise ValueError('unsupported saved dependency; cannot silently approximate: '+line)
        name, version = match.groups()
        key = canonical(name)
        if key in result:
            raise ValueError('duplicate saved dependency: '+key)
        if key not in EDITABLE:
            result[key] = version
    return result


@operation('training recovery: read legacy dependency evidence')
def legacy_pins(log):
    """Latest successful execution overlay is a candidate, validated by SHA later."""
    installed, witness = {}, None
    if not Path(log).is_file():
        raise FileNotFoundError('saved environment manifest and legacy session.log are both missing')
    with Path(log).open(encoding='utf-8', errors='replace') as stream:
        for line in track(stream, 'recovery: scan dependency history'):
            if 'Creating virtual environment with seed packages at: /content/s2st-runtime/venv' in line:
                installed = {}
            if line.startswith('Successfully installed '):
                for token in line.split()[2:]:
                    match = re.fullmatch(r'(.+)-([0-9][A-Za-z0-9_.+!-]*)', token)
                    if match:
                        name, version = match.groups()
                        installed[canonical(name)] = version
            match = re.match(r'Requirement already satisfied: ([A-Za-z0-9_.-]+)(?:[^ ]*) in /content/s2st-runtime/venv/.* \(([^()]+)\)\s*$', line.strip())
            if match:
                installed[canonical(match[1])] = match[2]
            if line.startswith('[recovery-overlay] ') and installed:
                witness = dict(installed)
    if witness is None:
        raise ValueError('no successful legacy training environment evidence; refusing to guess')
    return {k: v for k, v in witness.items() if k not in EDITABLE}


def manifest_path(persistent, expected):
    if not re.fullmatch('[0-9a-f]{64}', expected):
        raise ValueError('invalid saved environment SHA256')
    return Path(persistent)/'.runtime-environments'/f'{expected}.json'


def verified_manifest(persistent, expected):
    path = manifest_path(persistent, expected)
    if not path.is_file():
        return None
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError('saved dependency environment manifest is damaged')
    return path


@operation('training recovery: restore exact dependency environment')
def ensure_environment(repo, persistent, run, *, expected=None, prefix=Path('/content/s2st-runtime')):
    import runtime
    repo, persistent, prefix = Path(repo), Path(persistent), Path(prefix)
    path, python = prefix/'environment.json', prefix/'venv/bin/python'
    manifest = verified_manifest(persistent, expected) if expected else None

    def setup(lock=None):
        command = [sys.executable, ROOT/'scripts/colab/setup.py', '--repository', repo, '--inference']
        if lock:
            command += ['--environment-lock', lock]
        run(*command)

    if not runtime.ready(repo, prefix):
        setup(manifest)
    observed = subprocess.check_output([str(python), '-m', 'pip', 'freeze'], text=True)
    if json.loads(path.read_text()).get('packages') != observed:
        setup(manifest)  # Never trust a stale lock after manual package changes.
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if expected and actual != expected:
        wanted = freeze_pins(json.loads(manifest.read_text())['packages']) if manifest else legacy_pins(persistent/'session.log')
        current = freeze_pins(subprocess.check_output([str(python), '-m', 'pip', 'freeze'], text=True))
        changed = [f'{name}=={version}' for name, version in wanted.items() if current.get(name) != version]
        if changed:
            print('[environment-recovery] restore_versions='+', '.join(changed), flush=True)
            run(python, '-m', 'pip', 'install', '--no-deps', '--extra-index-url',
                'https://download.pytorch.org/whl/cu126', *changed)
            run(python, '-m', 'pip', 'check')
        setup(manifest)  # Re-measure real packages and repository using the actual bootstrap.
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError('dependency restoration still differs from saved SHA256; checkpoint retained, no training started. '
                             'Do not change environment.json or snapshot identities')
    if json.loads(path.read_text()).get('packages') != subprocess.check_output([str(python), '-m', 'pip', 'freeze'], text=True):
        raise ValueError('actual installed packages differ from the measured environment; no training started')
    target = manifest_path(persistent, actual)
    atomic_write_bytes(target, path.read_bytes(), resume=True)
    if target.read_bytes() != path.read_bytes():
        raise OSError('environment manifest readback mismatch')
    print(f'[environment-recovery] status=verified sha256={actual} manifest={target}', flush=True)
    # This ready check no longer rebuilds: setup above matched the actual code.
    return runtime.ensure_runtime(repo, persistent, run, require_gpu=True, prefix=prefix)
