"""Stdlib-only CPU bootstrap; optional isolated packages, no experiment writes."""
from collections import deque
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import runpy
import tempfile
import threading
import uuid

DIAGNOSTIC_RESULT_CONTRACT_VERSION = 2


def select_target(mode, benchmark_target):
    if mode == 'auto':
        # Cell 1 defaults to S2UT. That training/evaluation setting must not
        # redirect a Cascade-only analysis to a different model's artifacts.
        return benchmark_target if benchmark_target in ('cascade', 'all') else 'cascade'
    if mode not in ('cascade', 'all'):
        raise RuntimeError('8A target must be auto, cascade, or all')
    return mode


def cpu_environment(environment=None):
    return dict(os.environ if environment is None else environment,
                CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1', PYTHONIOENCODING='utf-8',
                HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')


def select_python(candidates, target, env, *, required_versions=None):
    """Probe existing interpreters, including actual CPU scoring imports.

    Do not resolve symlinks: a venv and its base Python can be the same binary
    while having different installed packages. No pip/setup/download fallback.
    """
    probe = '''import sys, importlib.metadata
assert sys.version_info >= (3, 10), 'Python >= 3.10 is required'
assert importlib.metadata.version('sacrebleu') == '2.5.1', 'requires sacrebleu==2.5.1'
import sacrebleu, numpy, portalocker, regex, tabulate, colorama
from lxml import etree
assert abs(sacrebleu.corpus_bleu(['one two three four five'], [['one two three four five']]).score - 100) < 1e-8
'''
    if target == 'all':
        probe += 'import yaml\nassert callable(yaml.safe_load)\n'
    if required_versions:
        probe += ('\nimport yaml\nfor name, version in '+repr(required_versions)+'.items():\n'
                  "    assert importlib.metadata.version(name) == version, name+' version mismatch'\n")
    checked, failures = set(), []
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(str(candidate)).absolute()
        key = os.path.normcase(str(path))
        if key in checked:
            continue
        checked.add(key)
        if not path.is_file():
            failures.append(str(path)+': interpreter not present')
            continue
        print('8A CPU dependency check: '+str(path), flush=True)
        try:
            result = subprocess.run([str(path), '-B', '-c', probe], capture_output=True,
                                    text=True, encoding='utf-8', errors='replace',
                                    timeout=30, env=env)
        except (OSError, subprocess.TimeoutExpired) as error:
            failures.append(str(path)+': '+str(error))
            continue
        if result.returncode == 0:
            return str(path)
        detail = (result.stderr or result.stdout or 'dependency probe failed').strip()
        failures.append(str(path)+': '+detail[-1800:])
    raise RuntimeError('8A has no usable existing CPU Python. Requires Python >=3.10 and '
                       'sacrebleu==2.5.1 (also PyYAML for all mode).\n'+'\n'.join(failures)+
                       '\nNo packages, runtime, models, or saved results were changed. '
                       'Select an already prepared interpreter with PYTHON; no automatic install is permitted.')


def ensure_python(candidates, target, env, *, dependency_root, protected=(), allow_install=False):
    """Prefer existing environments; optionally build a separate text-only cache."""
    candidates = list(candidates)
    try:
        return select_python(candidates, target, env), env
    except RuntimeError as missing:
        if not allow_install:
            raise
        print(str(missing).split('\nNo packages, runtime')[0], flush=True)
    root = Path(dependency_root).absolute()
    if any(root.resolve().is_relative_to(Path(path).resolve()) for path in protected):
        raise RuntimeError('8A CPU dependencies must be outside saved experiments and CORPUS_ROOT')
    source = Path(__file__).resolve().parents[2]
    requirements = source/'requirements/diagnostic-cpu.txt'
    content = requirements.read_text(encoding='utf-8')
    versions = dict(line.strip().split('==') for line in content.splitlines()
                    if line.strip() and not line.lstrip().startswith('#'))
    base, fingerprint = None, None
    for candidate in dict.fromkeys(str(p) for p in candidates if p and Path(str(p)).is_file()):
        try:
            check = subprocess.run([candidate, '-B', '-c',
                "import sys,sysconfig,json,pip; assert (3,10)<=sys.version_info[:2]<=(3,13), 'CPU wheels tested on Python 3.10..3.13'; print(json.dumps([sys.implementation.cache_tag,sysconfig.get_platform(),sys.executable]))"],
                capture_output=True, text=True, encoding='utf-8', errors='replace', env=env, timeout=30)
        except (OSError, subprocess.TimeoutExpired):
            continue
        if check.returncode == 0:
            base, fingerprint = candidate, check.stdout.strip()
            break
    if base is None:
        raise RuntimeError('8A isolated CPU dependencies need an existing Python 3.10..3.13 with pip; '
                           'no system packages or training environments were changed')
    key = hashlib.sha256((content+fingerprint).encode()).hexdigest()[:20]
    def overlay(path):
        return dict(env, PYTHONPATH=str(path)+(os.pathsep+env['PYTHONPATH'] if env.get('PYTHONPATH') else ''))
    for cache in sorted(root.glob(key+'*')) if root.is_dir() else ():
        if not cache.is_dir() or cache.is_symlink():
            continue
        try:
            selected = select_python([base], target, overlay(cache), required_versions=versions)
            return selected, overlay(cache)
        except RuntimeError:
            pass  # Preserve incomplete caches; never overwrite or delete one.
    root.mkdir(parents=True, exist_ok=True)
    # Progress is pure stdlib. Load its fixed local source without changing the
    # notebook's sys.path or importing an old training package into this kernel.
    Progress = runpy.run_path(str(source/'src/direct_s2st/progress.py'))['Progress']
    destination = root/(key if not (root/key).exists() else key+'-'+uuid.uuid4().hex)
    with tempfile.TemporaryDirectory(prefix='.install-', dir=root) as temporary:
        packages = Path(temporary)/'packages'
        if not packages.resolve().is_relative_to(root.resolve()):
            raise RuntimeError('invalid isolated CPU dependency directory')
        command = [base, '-B', '-m', 'pip', '--isolated', 'install', '--disable-pip-version-check',
                   '--no-input', '--no-cache-dir', '--only-binary=:all:', '--no-deps',
                   '--index-url', 'https://pypi.org/simple', '--target', str(packages), '-r', str(requirements)]
        with Progress('8A: install isolated CPU text-metric dependencies'):
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       stdin=subprocess.DEVNULL, text=True, encoding='utf-8',
                                       errors='replace', bufsize=1, env=env)
            try:
                with process.stdout:
                    for line in process.stdout:
                        print(line, end='', flush=True)
                status = process.wait()
            except BaseException:
                stop_child(process)
                raise
            if status:
                raise RuntimeError(f'8A isolated CPU dependency install failed ({status}); '
                                   'see pip output above. No training environment or saved results were changed.')
            select_python([base], target, overlay(packages), required_versions=versions)
            packages.rename(destination)
    return select_python([base], target, overlay(destination), required_versions=versions), overlay(destination)


def stop_child(process):
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def validate_result(report, result_kind, name):
    """8A rechecks a formal score; 8E produces an independent external score."""
    if not isinstance(report, dict):
        raise RuntimeError(name+' helper did not return a result object')
    if result_kind == 'saved_formal_all':
        if report.get('all_score_matches_saved') is not True:
            raise RuntimeError(name+' helper did not confirm agreement with the saved formal All score')
        return
    def text(key):
        return isinstance(report.get(key), str) and bool(report[key].strip())
    valid = text('display')
    if result_kind == 'external_s2ut_inspect':
        valid = valid and text('cascade_root') and type(report.get('samples')) is int and report['samples'] > 0
    else:
        metrics = report.get('metrics')
        valid = valid and text('output_root') and isinstance(metrics, dict)
        if valid:
            valid = (metrics.get('purpose') == 'external_flores_fleurs_s2ut_natural_not_formal_benchmark'
                     and metrics.get('condition') == 'natural'
                     and type(metrics.get('sample_count')) is int and metrics['sample_count'] > 0
                     and all(type(metrics.get(k)) in (int, float) and math.isfinite(metrics[k])
                             for k in ('bleu', 'cascade_natural_asr_bleu', 'difference')))
            if valid:
                # sacreBLEU can report 100.00000000000004 for an exact match.
                # Accept floating-point roundoff without changing saved scores.
                valid = (0 <= metrics['bleu'] <= 100+1e-8 and 0 <= metrics['cascade_natural_asr_bleu'] <= 100+1e-8
                         and math.isclose(metrics['difference'], metrics['bleu']-metrics['cascade_natural_asr_bleu'],
                                          rel_tol=1e-10, abs_tol=1e-8))
    if not valid:
        raise RuntimeError(name+' helper did not return a valid external S2UT '+
                           ('inspection' if result_kind == 'external_s2ut_inspect' else 'evaluation report'))


def run_json(command, env, *, name='8A', mode='CPU diagnostic', failure_note=None,
             result_kind='saved_formal_all'):
    """Relay stderr through Python so Colab actually displays logs/errors."""
    if result_kind not in ('saved_formal_all','external_s2ut','external_s2ut_inspect'):
        raise ValueError('unknown diagnostic result contract: '+str(result_kind))
    tail = deque(maxlen=60)
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               stdin=subprocess.DEVNULL, text=True, encoding='utf-8',
                               errors='replace', bufsize=1, env=env)
    def relay():
        with process.stderr:
            for line in process.stderr:
                tail.append(line)
                print(line, end='', flush=True)
    reader = threading.Thread(target=relay, name='cascade-diagnostic-stderr', daemon=True)
    reader.start()
    try:
        with process.stdout:
            output = process.stdout.read()
        status = process.wait()
    except BaseException:
        stop_child(process)
        raise
    finally:
        reader.join(timeout=5)
    if status:
        note = failure_note or 'Saved results were not repaired or overwritten; no inference was attempted.'
        raise RuntimeError(f'{name} {mode} exited with code {status}.\n'+''.join(tail)+'\n'+note)
    try:
        report = json.loads(output)
    except json.JSONDecodeError as error:
        raise RuntimeError(name+' helper did not return valid result JSON: '+output[-1200:]) from error
    validate_result(report, result_kind, name)
    return report
