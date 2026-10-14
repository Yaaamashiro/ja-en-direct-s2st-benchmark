import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'scripts/colab'))
import validation_dependencies as dependencies


def wheel_fixture(monkeypatch, files=None):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, 'w') as archive:
        for name, value in (files or {'editdistance/__init__.py': 'native package',
                                     'editdistance/native.so': 'fixture binary',
                                     'editdistance-0.8.1.dist-info/METADATA': 'Version: 0.8.1'}).items():
            archive.writestr(name, value)
    data = stream.getvalue()
    monkeypatch.setattr(dependencies, 'WHEEL_SHA', hashlib.sha256(data).hexdigest())
    monkeypatch.setattr(dependencies.urllib.request, 'urlopen', lambda *a, **k: io.BytesIO(data))
    return data


def test_isolated_wheel_verifies_bytes_and_preserves_distribution_without_installing_into_venv(tmp_path, monkeypatch):
    wheel_fixture(monkeypatch)
    package = dependencies.provision(tmp_path)
    assert not list(package.glob('*.dist-info'))
    assert dependencies.provision(tmp_path) == package
    (package/'editdistance/native.so').write_text('modified')
    with pytest.raises(FileExistsError):
        dependencies.provision(tmp_path)


def test_dependency_checksum_and_unsafe_zip_members_fail_closed(tmp_path, monkeypatch):
    wheel_fixture(monkeypatch, {'../escape': 'unsafe', 'editdistance/__init__.py':'fixture'})
    with pytest.raises(ValueError, match='unsafe'):
        dependencies.provision(tmp_path)
    wheel_fixture(monkeypatch)
    monkeypatch.setattr(dependencies, 'WHEEL_SHA', '0'*64)
    with pytest.raises(ValueError, match='checksum'):
        dependencies.provision(tmp_path)


def test_legacy_validation_sidecar_keeps_base_freeze_and_manifest_unchanged(tmp_path, monkeypatch):
    import os
    monkeypatch.setattr(os, 'environ', dict(os.environ))
    prefix = tmp_path/'runtime'
    prefix.mkdir()
    lock = prefix/'environment.json'
    lock.write_text(json.dumps(dict(packages='torch==2.7.1\n')))
    before = lock.read_bytes()
    monkeypatch.setattr(dependencies.subprocess, 'check_output', lambda *a, **k: 'torch==2.7.1\n')
    calls = []
    sidecar = dependencies.ensure(tmp_path/'python', prefix, tmp_path/'persistent', lambda *a: calls.append(a))
    assert str(sidecar) in os.environ['PYTHONPATH']
    assert lock.read_bytes() == before and len(calls) == 2
    assert str(calls[-1][1]).endswith('check_s2ut_runtime.py')
    assert list((tmp_path/'persistent/.runtime-extras').glob('*.json'))
    monkeypatch.setattr(dependencies.subprocess, 'check_output', lambda *a, **k: 'torch==2.7.1\neditdistance==0.8.1\n')
    with pytest.raises(ValueError, match='changed the base environment'):
        dependencies.ensure(tmp_path/'python', prefix, tmp_path/'persistent', lambda *a: None)


def test_runtime_failure_stops_before_any_staging_and_new_env_uses_pinned_package(tmp_path, monkeypatch):
    prefix = tmp_path/'runtime'
    prefix.mkdir()
    (prefix/'environment.json').write_text(json.dumps(dict(packages='editdistance==0.8.1\n')))
    calls = []
    def fail(*args):
        calls.append(args)
        raise ModuleNotFoundError('runtime dependency')
    with pytest.raises(ModuleNotFoundError):
        dependencies.ensure(tmp_path/'python', prefix, tmp_path, fail)
    assert len(calls) == 1 and str(calls[0][1]).endswith('check_s2ut_runtime.py')
    assert 'editdistance==0.8.1' in (ROOT/'requirements/colab310.txt').read_text()
    assert 'editdistance==0.8.1' in (ROOT/'docker/fairseq.Dockerfile').read_text()
