"""Restore actual dependency versions; never substitute an old environment hash."""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))
from direct_s2st.io import atomic_write_bytes, atomic_write_json
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


def verify_managed_venv(python, prefix):
    """Never uninstall from the notebook kernel or another Python installation."""
    prefix, python = Path(prefix).absolute(), Path(python).absolute()
    venv = prefix/'venv'
    if (prefix == Path(prefix.anchor) or python != venv/'bin/python'
            or any(p.is_symlink() or getattr(p, 'is_junction', lambda: False)()
                   for p in (prefix, venv))
            or not (venv/'pyvenv.cfg').is_file()):
        raise ValueError('dependency cleanup requires the dedicated runtime virtual environment')
    if not re.search(r'^include-system-site-packages\s*=\s*false\s*$',
                     (venv/'pyvenv.cfg').read_text(encoding='utf-8'), re.I | re.M):
        raise ValueError('dependency cleanup requires an isolated virtual environment without system packages')
    corpus = os.environ.get('CORPUS_ROOT')
    if corpus and prefix.resolve().is_relative_to(Path(corpus).resolve()):
        raise ValueError('runtime dependency cleanup must stay outside CORPUS_ROOT')
    actual = subprocess.check_output([str(python), '-c', 'import sys; print(sys.prefix)'], text=True).strip()
    if Path(actual).resolve() != venv.resolve():
        raise ValueError('runtime Python is not executing inside the expected virtual environment')


def dependency_difference(wanted, observed):
    expected_pins, actual_pins = freeze_pins(wanted['packages']), freeze_pins(observed['packages'])
    return dict(
        missing={k:v for k,v in expected_pins.items() if k not in actual_pins},
        extra={k:v for k,v in actual_pins.items() if k not in expected_pins},
        changed={k:dict(expected=v, actual=actual_pins[k]) for k,v in expected_pins.items()
                 if k in actual_pins and actual_pins[k] != v},
        metadata={k:dict(expected=wanted.get(k), actual=observed.get(k))
                  for k in wanted.keys() | observed.keys() if k != 'packages' and wanted.get(k) != observed.get(k)},
        freeze_text_differs=wanted['packages'] != observed['packages'])


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
        saved = json.loads(manifest.read_text()) if manifest else None
        wanted = freeze_pins(saved['packages']) if saved else legacy_pins(persistent/'session.log')
        observed = subprocess.check_output([str(python), '-m', 'pip', 'freeze'], text=True)
        current = freeze_pins(observed)
        changed = [f'{name}=={version}' for name, version in wanted.items() if current.get(name) != version]
        # Installing constraints does not remove packages left by another run.
        # Only a complete SHA-verified manifest can authorize exact membership:
        # legacy session-log evidence may be partial and must never drive removals.
        extra = sorted(set(current)-set(wanted)) if saved is not None else []
        if extra:
            verify_managed_venv(python, prefix)
            receipt = prefix/('dependency-recovery-before-'+expected+'-'+hashlib.sha256(observed.encode()).hexdigest()+'.json')
            atomic_write_json(receipt, dict(expected_environment=expected, packages=observed,
                              removed={name:current[name] for name in extra}), resume=True)
            print('[environment-recovery] remove_unsaved_packages='+', '.join(extra)+
                  f' scope=runtime_venv_only recovery_receipt={receipt}', file=sys.stderr, flush=True)
            run(python, '-m', 'pip', 'uninstall', '--yes', *extra)
        if changed:
            print('[environment-recovery] restore_versions='+', '.join(changed), file=sys.stderr, flush=True)
            run(python, '-m', 'pip', 'install', '--no-deps', '--extra-index-url',
                'https://download.pytorch.org/whl/cu126', *changed)
        if changed or extra:
            run(python, '-m', 'pip', 'check')
        setup(manifest)  # Re-measure real packages and repository using the actual bootstrap.
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            measured = json.loads(path.read_text())
            difference = dependency_difference(saved, measured) if saved else dict(
                missing={k:v for k,v in wanted.items() if k not in freeze_pins(measured['packages'])},
                legacy_manifest_missing=True)
            diagnostic = prefix/f'dependency-recovery-difference-{expected[:12]}-{actual[:12]}.json'
            atomic_write_json(diagnostic, dict(expected_sha256=expected, actual_sha256=actual,
                              difference=difference), resume=True)
            print('[environment-recovery] unresolved_difference='+json.dumps(difference, sort_keys=True)+
                  f' diagnostic={diagnostic}', file=sys.stderr, flush=True)
            raise ValueError('dependency restoration still differs from saved SHA256; checkpoint retained, no training started. '
                             f'See {diagnostic}. Do not change environment.json or snapshot identities')
    if json.loads(path.read_text()).get('packages') != subprocess.check_output([str(python), '-m', 'pip', 'freeze'], text=True):
        raise ValueError('actual installed packages differ from the measured environment; no training started')
    target = manifest_path(persistent, actual)
    atomic_write_bytes(target, path.read_bytes(), resume=True)
    if target.read_bytes() != path.read_bytes():
        raise OSError('environment manifest readback mismatch')
    print(f'[environment-recovery] status=verified sha256={actual} manifest={target}', file=sys.stderr, flush=True)
    # This ready check no longer rebuilds: setup above matched the actual code.
    return runtime.ensure_runtime(repo, persistent, run, require_gpu=True, prefix=prefix)
