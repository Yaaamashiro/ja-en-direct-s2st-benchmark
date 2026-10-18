"""Switching saved runs must restore membership, not only package versions."""
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'scripts/colab'))
import environment_recovery as recovery


@pytest.fixture
def runtime_fixture(tmp_path, monkeypatch):
    prefix, persistent = tmp_path/'runtime', tmp_path/'drive'
    venv = prefix/'venv'
    venv.mkdir(parents=True)
    (venv/'pyvenv.cfg').write_text('include-system-site-packages = false\n')
    persistent.mkdir()
    checkpoint = persistent/'checkpoint.pt'
    checkpoint.write_bytes(b'immutable optimizer and model fixture')
    installed = {'bitarray':'3.12.2', 'editdistance':'0.8.1'}
    editable = '# Editable install with no version control (fairseq==0.12.2)\n-e /content/s2st-runtime/fairseq\n'

    def freeze(pins):
        return ''.join(f'{k}=={v}\n' for k, v in sorted(pins.items()))+editable

    environment = prefix/'environment.json'
    metadata = {'repository':'saved-code', 'python':'3.10.18'}
    def measure():
        environment.write_bytes(json.dumps(dict(metadata, packages=freeze(installed))).encode())
    measure()

    def save(pins, **overrides):
        raw = json.dumps(dict(metadata, packages=freeze(pins), **overrides)).encode()
        sha = hashlib.sha256(raw).hexdigest()
        lock = recovery.manifest_path(persistent, sha)
        lock.parent.mkdir(exist_ok=True)
        lock.write_bytes(raw)
        return sha, raw

    calls = []
    def output(command, **kwargs):
        if command[-1] == 'import sys; print(sys.prefix)':
            return str(venv)+'\n'
        assert command[1:] == ['-m', 'pip', 'freeze']
        return freeze(installed)

    def run(*command):
        calls.append(command)
        if 'uninstall' in command:
            # A recoverable receipt must already exist before the mutation.
            receipts = list(prefix.glob('dependency-recovery-before-*.json'))
            assert receipts
            receipt = json.loads(receipts[-1].read_text())
            assert receipt['packages'] == freeze(installed)
            for name in command[command.index('--yes')+1:]:
                assert name in receipt['removed']
                del installed[name]
        elif 'install' in command:
            for pin in command[command.index('--no-deps')+1:]:
                if '==' in str(pin):
                    name, version = pin.split('==')
                    installed[name] = version
        elif '--repository' in command:
            measure()  # Measure actual state; never copy the requested lock.
        assert checkpoint.read_bytes() == b'immutable optimizer and model fixture'

    monkeypatch.setattr(recovery.subprocess, 'check_output', output)
    monkeypatch.setitem(sys.modules, 'runtime', SimpleNamespace(
        ready=lambda *a: True, ensure_runtime=lambda *a, **k: venv/'bin/python'))
    monkeypatch.delenv('CORPUS_ROOT', raising=False)
    return SimpleNamespace(prefix=prefix, persistent=persistent, venv=venv, environment=environment,
        installed=installed, save=save, run=run, calls=calls, checkpoint=checkpoint, measure=measure)


def restore(fixture, sha):
    return recovery.ensure_environment(ROOT, fixture.persistent, fixture.run,
                                       expected=sha, prefix=fixture.prefix)


def test_old_tt2_after_vocoder_removes_only_verified_extra_then_reinstalls_for_vocoder(runtime_fixture):
    f = runtime_fixture
    tt2, tt2_bytes = f.save({'bitarray':'3.12.0'})
    vocoder, vocoder_bytes = f.save(dict(f.installed))
    before = f.checkpoint.read_bytes()
    assert restore(f, tt2) == f.venv/'bin/python'
    assert f.installed == {'bitarray':'3.12.0'}
    assert f.environment.read_bytes() == tt2_bytes
    removal = next(c for c in f.calls if 'uninstall' in c)
    assert removal == (f.venv/'bin/python', '-m', 'pip', 'uninstall', '--yes', 'editdistance')
    assert f.calls.index(removal) < next(i for i, c in enumerate(f.calls) if 'install' in c)
    assert any('check' in c for c in f.calls)
    receipt = json.loads(next(f.prefix.glob('dependency-recovery-before-*.json')).read_text())
    assert receipt['removed'] == {'editdistance':'0.8.1'}
    assert receipt['expected_environment'] == tt2
    assert 'editdistance==0.8.1' in receipt['packages']
    # Reverse restoration installs the package again. Repeated normal cells
    # neither weaken the lock nor remove editable fairseq/project installs.
    restore(f, vocoder)
    assert f.environment.read_bytes() == vocoder_bytes
    assert f.installed == {'bitarray':'3.12.2', 'editdistance':'0.8.1'}
    restore(f, tt2)
    assert f.environment.read_bytes() == tt2_bytes
    assert recovery.verified_manifest(f.persistent, tt2).read_bytes() == tt2_bytes
    assert recovery.verified_manifest(f.persistent, vocoder).read_bytes() == vocoder_bytes
    assert f.checkpoint.read_bytes() == before


def test_legacy_partial_evidence_never_authorizes_package_removal(runtime_fixture):
    f = runtime_fixture
    expected, _ = f.save({'bitarray':'3.12.0'})
    recovery.manifest_path(f.persistent, expected).unlink()
    (f.persistent/'session.log').write_text(
        'Successfully installed bitarray-3.12.0\n[recovery-overlay] saved_at_least=580\n')
    with pytest.raises(ValueError, match='still differs'):
        restore(f, expected)
    assert not any('uninstall' in c for c in f.calls)
    assert f.installed['editdistance'] == '0.8.1'
    assert not recovery.manifest_path(f.persistent, expected).exists()
    report = json.loads(next(f.prefix.glob('dependency-recovery-difference-*.json')).read_text())
    assert report['difference']['legacy_manifest_missing'] is True


@pytest.mark.parametrize('unsafe', ['missing_cfg', 'kernel_python', 'wrong_sys_prefix', 'corpus', 'system_packages'])
def test_cleanup_rejects_nonmanaged_interpreters_and_corpus(runtime_fixture, monkeypatch, unsafe):
    f = runtime_fixture
    if unsafe == 'missing_cfg':
        (f.venv/'pyvenv.cfg').unlink()
    elif unsafe == 'wrong_sys_prefix':
        monkeypatch.setattr(recovery.subprocess, 'check_output', lambda *a, **k: str(f.prefix/'global'))
    elif unsafe == 'corpus':
        monkeypatch.setenv('CORPUS_ROOT', str(f.prefix.parent))
    elif unsafe == 'system_packages':
        (f.venv/'pyvenv.cfg').write_text('include-system-site-packages = true\n')
    python = f.prefix/'kernel/python' if unsafe == 'kernel_python' else f.venv/'bin/python'
    with pytest.raises(ValueError, match='virtual environment|CORPUS_ROOT'):
        recovery.verify_managed_venv(python, f.prefix)
    assert not f.calls


def test_corrupt_saved_manifest_never_authorizes_package_removal(runtime_fixture):
    f = runtime_fixture
    expected, _ = f.save({'bitarray':'3.12.0'})
    recovery.manifest_path(f.persistent, expected).write_text('corrupt')
    with pytest.raises(ValueError, match='manifest is damaged'):
        restore(f, expected)
    assert not f.calls
    assert f.installed['editdistance'] == '0.8.1'


def test_failed_uninstall_retains_recovery_receipt_and_checkpoint(runtime_fixture):
    f = runtime_fixture
    expected, raw = f.save({'bitarray':'3.12.0'})
    before = f.checkpoint.read_bytes()
    def failed(*command):
        assert 'uninstall' in command
        raise OSError('simulated runtime interruption')
    with pytest.raises(OSError, match='interruption'):
        recovery.ensure_environment(ROOT, f.persistent, failed, expected=expected, prefix=f.prefix)
    receipt = json.loads(next(f.prefix.glob('dependency-recovery-before-*.json')).read_text())
    assert receipt['removed'] == {'editdistance':'0.8.1'}
    assert recovery.verified_manifest(f.persistent, expected).read_bytes() == raw
    assert f.checkpoint.read_bytes() == before
    restore(f, expected)
    assert f.environment.read_bytes() == raw


def test_cleanup_guard_fails_before_any_uninstall(runtime_fixture):
    f = runtime_fixture
    expected, raw = f.save({'bitarray':'3.12.0'})
    (f.venv/'pyvenv.cfg').unlink()
    before = f.checkpoint.read_bytes()
    with pytest.raises(ValueError, match='dedicated runtime virtual environment'):
        restore(f, expected)
    assert not any('uninstall' in c for c in f.calls)
    assert recovery.verified_manifest(f.persistent, expected).read_bytes() == raw
    assert f.checkpoint.read_bytes() == before


def test_mismatched_metadata_keeps_strict_sha_and_reports_difference(runtime_fixture):
    f = runtime_fixture
    expected, raw = f.save(dict(f.installed), inference_dependencies=True)
    with pytest.raises(ValueError, match='still differs'):
        restore(f, expected)
    report = json.loads(next(f.prefix.glob('dependency-recovery-difference-*.json')).read_text())
    assert report['expected_sha256'] == expected
    assert report['actual_sha256'] == hashlib.sha256(f.environment.read_bytes()).hexdigest()
    assert report['difference']['metadata'] == {
        'inference_dependencies':{'expected':True, 'actual':None}}
    assert recovery.verified_manifest(f.persistent, expected).read_bytes() == raw
    assert f.environment.read_bytes() != raw
    assert not any('uninstall' in c for c in f.calls)


def test_difference_includes_missing_extra_changed_and_editable_text():
    wanted = {'packages':'Old==1.0\nShared==1.0\n-e /original\n', 'repository':'old'}
    observed = {'packages':'Extra==2.0\nShared==2.0\n-e /different\n', 'repository':'new'}
    assert recovery.dependency_difference(wanted, observed) == {
        'missing':{'old':'1.0'}, 'extra':{'extra':'2.0'},
        'changed':{'shared':{'expected':'1.0','actual':'2.0'}},
        'metadata':{'repository':{'expected':'old','actual':'new'}}, 'freeze_text_differs':True}
