import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'scripts/colab'))
import environment_recovery as recovery
import training_session as session


def snapshot(persistent, name, *, updates=133, kind='tt2', base=None, expected='a'*64, warmup=400):
    config = dict(kind=kind, runtime={'cache_gb':8}, identity_files=['/content/s2st-runtime/environment.json'],
                  research_metadata=dict(repository_revision=base or session.legacy.BASE_REVISION))
    if kind == 's2ut':
        config['research_metadata']['warmup_updates'] = warmup
    path = persistent/(name+'.json')
    path.write_text(json.dumps(config))
    marker = persistent/'checkpoints'/name/f'{updates:012d}-fixture'/'snapshot.json'
    marker.parent.mkdir(parents=True)
    marker.write_text(json.dumps(dict(updates=updates, identity=dict(config=config,
        work=str(session.WORK_ROOT/name), inputs={'/content/s2st-runtime/environment.json': expected}))))
    return config, path, marker


def test_automatic_selection_finds_actual_probe_suffix_not_empty_default(tmp_path):
    original = 'tt2-training-experiment-001-paper_exact-recipe-v2'
    (tmp_path/(original+'.json')).write_text(json.dumps({'kind':'tt2'}))
    name = original+'-probe-fix-v1'
    snapshot(tmp_path, name)
    assert session.choose_run(tmp_path, 'tt2', original)[0] == name
    snapshot(tmp_path, 's2ut-other', kind='s2ut')
    assert session.choose_run(tmp_path, 'tt2', original)[0] == name
    snapshot(tmp_path, 'tt2-second')
    with pytest.raises(ValueError, match='multiple trained runs'):
        session.choose_run(tmp_path, 'tt2', original)
    assert session.choose_run(tmp_path, 'tt2', original, name)[0] == name
    with pytest.raises(ValueError, match='no readable snapshot'):
        session.choose_run(tmp_path, 'tt2', original, 'typo')


def test_bad_markers_never_start_zero_and_readable_older_marker_is_retained(tmp_path):
    _, _, marker = snapshot(tmp_path, 'tt2-original')
    newer = marker.parent.parent/'000000000134-broken'/'snapshot.json'
    newer.parent.mkdir()
    newer.write_text('interrupted')
    assert session.choose_run(tmp_path, 'tt2', 'tt2-new')[1]['updates'] == 133
    marker.write_text('also broken')
    with pytest.raises(ValueError, match='refusing a fresh restart'):
        session.choose_run(tmp_path, 'tt2', 'tt2-new')


def test_legacy_witness_ignores_newer_failed_setup_and_rejects_arbitrary_requirements(tmp_path):
    log = tmp_path/'session.log'
    log.write_text('Successfully installed pydantic-2.14.0 pydantic-core-2.50.0 bitarray-3.12.0 fairseq-0.12.2\n'
                   '[recovery-overlay] saved_at_least=85\n'
                   'Creating virtual environment with seed packages at: /content/s2st-runtime/venv\n'
                   'Successfully installed pydantic-2.14.1 bitarray-3.12.1\n'
                   'ValueError: environment differs\n')
    assert recovery.legacy_pins(log) == {'pydantic':'2.14.0', 'pydantic-core':'2.50.0', 'bitarray':'3.12.0'}
    assert recovery.freeze_pins('# editable package\n-e /content/benchmark\nPyYAML==6.0.2\ntorch==2.7.1+cu126\n') == {'pyyaml':'6.0.2','torch':'2.7.1+cu126'}
    with pytest.raises(ValueError, match='unsupported'):
        recovery.freeze_pins('package @ https://unexpected.example/requirements\n')


def test_actual_dependency_restoration_must_match_hash_before_manifest_publish(tmp_path, monkeypatch):
    repo, persistent, prefix = tmp_path/'repo', tmp_path/'drive', tmp_path/'runtime'
    persistent.mkdir()
    prefix.mkdir()
    environment = prefix/'environment.json'
    target = {'repository':'base', 'packages':'pydantic==2.14.0\nbitarray==3.12.0\n'}
    wanted = json.dumps(target).encode()
    expected = hashlib.sha256(wanted).hexdigest()
    installed = {'pydantic':'2.14.1', 'bitarray':'3.12.1'}
    environment.write_text(json.dumps(dict(target, packages='pydantic==2.14.1\nbitarray==3.12.1\n')))
    (persistent/'session.log').write_text('Successfully installed pydantic-2.14.0 bitarray-3.12.0\n[recovery-overlay] saved_at_least=133\n')
    calls = []
    monkeypatch.setattr(recovery.subprocess, 'check_output', lambda *a, **k: ''.join(f'{name}=={version}\n' for name,version in installed.items()))
    monkeypatch.setitem(sys.modules, 'runtime', NS(ready=lambda *a: True, ensure_runtime=lambda *a, **kw: prefix/'venv/bin/python'))
    def run(*command):
        calls.append(command)
        if 'install' in command:
            for part in command:
                if '==' in str(part):
                    name, version = str(part).split('==')
                    installed[name] = version
        elif '--repository' in command:
            environment.write_text(json.dumps(dict(target, packages=''.join(f'{name}=={version}\n' for name,version in installed.items()))))
    recovery.ensure_environment(repo, persistent, run, expected=expected, prefix=prefix)
    manifest = recovery.verified_manifest(persistent, expected)
    assert manifest.read_bytes() == wanted == environment.read_bytes()
    assert any('pydantic==2.14.0' in command for command in calls)
    monkeypatch.setattr(recovery, 'legacy_pins', lambda *a: pytest.fail('must use persisted full environment now'))
    installed['pydantic'] = '2.14.1'
    environment.write_text('{}')
    recovery.ensure_environment(repo, persistent, run, expected=expected, prefix=prefix)
    assert any('--environment-lock' in command for command in calls)
    manifest.write_text('corrupt')
    with pytest.raises(ValueError, match='manifest is damaged'):
        recovery.ensure_environment(repo, persistent, run, expected=expected, prefix=prefix)


def test_unrestorable_environment_does_not_publish_fake_manifest(tmp_path, monkeypatch):
    prefix = tmp_path/'runtime'
    prefix.mkdir()
    (prefix/'environment.json').write_text('{"packages":"pkg==2.0\\n"}')
    (tmp_path/'session.log').write_text('Successfully installed pkg-1.0\n[recovery-overlay] saved_at_least=5\n')
    monkeypatch.setitem(sys.modules, 'runtime', NS(ready=lambda *a: True))
    monkeypatch.setattr(recovery.subprocess, 'check_output', lambda *a, **k: 'pkg==2.0\n')
    with pytest.raises(ValueError, match='still differs'):
        recovery.ensure_environment(tmp_path, tmp_path, lambda *a: None, expected='a'*64, prefix=prefix)
    assert not recovery.manifest_path(tmp_path, 'a'*64).exists()


def test_cell5_resume_retains_config_work_snapshot_and_execution_pin(tmp_path, monkeypatch):
    import os
    monkeypatch.setattr(os, 'environ', dict(os.environ))
    name = 'tt2-training-experiment-001-paper_exact-recipe-v2-probe-fix-v1'
    monkeypatch.setattr(session, 'WORK_ROOT', tmp_path/'work')
    monkeypatch.setattr(session, 'RUNTIME_ROOT', tmp_path/'runtime')
    session.RUNTIME_ROOT.mkdir()
    environment = session.RUNTIME_ROOT/'environment.json'
    environment.write_text('{"packages":"fixture==1"}')
    expected = hashlib.sha256(environment.read_bytes()).hexdigest()
    config, path, marker = snapshot(tmp_path, name, expected=expected)
    before = path.read_bytes(), marker.read_bytes()
    monkeypatch.setattr(session, 'git', lambda repo, *args: session.legacy.BASE_REVISION)
    monkeypatch.setattr(session, 'checkout', lambda *a: tmp_path/'reviewed-execution')
    monkeypatch.setattr(session.legacy, 'validate_revision', lambda *a: None)
    checked = []
    monkeypatch.setattr(session.legacy, 'validate_snapshot', lambda cfg, *a: checked.append(cfg))
    monkeypatch.setattr(session, 'ensure_environment', lambda *a, **kw: tmp_path/'python')
    import tt2_schedule_overlay
    monkeypatch.setattr(tt2_schedule_overlay, 'prepare', lambda execution, *a: execution)
    monkeypatch.setenv('S2ST_TT2_MONITOR_POLICY', '{"interval":10,"patience":10,"min_relative_improvement":0.001}')
    calls = []
    for _ in range(2):
        assert session.launch(tmp_path/'repo', tmp_path, tmp_path/'data', 'tt2', 'tt2-empty',
            'fisher', {'batch_size':999, 'tt2_warmup_updates':400}, 10000, 0, confirm=True, run=lambda *a: calls.append(a)) == name
    assert before == (path.read_bytes(), marker.read_bytes())
    assert checked == [config, config]
    assert os.environ['S2ST_TT2_PRACTICAL_WARMUP'] == '400'
    assert all('--resume' in c and '--auto-resume' not in c for c in calls)
    assert all(str(path) in list(map(str,c)) for c in calls)
    route = json.loads((tmp_path/(name+'.launch.json')).read_text())
    assert route['execution_revision'] == session.LEGACY_EXECUTION
    assert route['environment_sha256'] == expected


def test_new_native_run_records_code_and_environment_then_reuses_them(tmp_path, monkeypatch):
    import os
    monkeypatch.setattr(os, 'environ', dict(os.environ))
    monkeypatch.setattr(session, 'WORK_ROOT', tmp_path/'work')
    monkeypatch.setattr(session, 'RUNTIME_ROOT', tmp_path/'runtime')
    session.RUNTIME_ROOT.mkdir()
    environment = session.RUNTIME_ROOT/'environment.json'
    environment.write_text('{"packages":"fixture==1"}')
    expected = hashlib.sha256(environment.read_bytes()).hexdigest()
    code = 'b'*40
    monkeypatch.setattr(session, 'git', lambda repo, *args: code)
    checked_out = []
    def checkout(repo, revision, run):
        checked_out.append(revision)
        return tmp_path/'execution'/revision
    monkeypatch.setattr(session, 'checkout', checkout)
    restored_environments = []
    def restore(*args, **kwargs):
        restored_environments.append(kwargs['expected'])
        return tmp_path/'python'
    monkeypatch.setattr(session, 'ensure_environment', restore)
    cfg = dict(kind='s2ut', runtime={'cache_gb':8}, identity_files=['/content/s2st-runtime/environment.json'],
               research_metadata=dict(repository_revision=code, warmup_updates=400))
    monkeypatch.setattr(session.subprocess, 'check_output', lambda *a, **kw: json.dumps(cfg))
    calls = []
    name = 's2ut-training-experiment-001-paper_practical-recipe-v3-warmup400'
    session.launch(tmp_path/'repo', tmp_path, tmp_path/'data', 's2ut', name, 'fisher',
                   {'s2ut_warmup_updates':400}, 1000, 0, confirm=True, run=lambda *a: calls.append(a))
    assert '--auto-resume' in calls[-1]
    _, path, marker = snapshot(tmp_path, name, kind='s2ut', base=code, expected=expected)
    before = path.read_bytes(), marker.read_bytes()
    session.launch(tmp_path/'repo', tmp_path, tmp_path/'data', 's2ut', name, 'fisher',
                   {'s2ut_warmup_updates':400, 'batch_size':999}, 1000, 0, confirm=True, run=lambda *a: calls.append(a))
    assert '--resume' in calls[-1]
    assert restored_environments == [None, expected]
    assert checked_out == [code, code]
    assert before == (path.read_bytes(), marker.read_bytes())
    with pytest.raises(ValueError, match='warmup differs from saved run'):
        session.launch(tmp_path/'repo', tmp_path, tmp_path/'data', 's2ut', name, 'fisher',
                       {'s2ut_warmup_updates':1000}, 1000, 0, confirm=True, run=lambda *a: calls.append(a))
    assert restored_environments == [None, expected]  # Reject before setup/download.
    assert before == (path.read_bytes(), marker.read_bytes())
