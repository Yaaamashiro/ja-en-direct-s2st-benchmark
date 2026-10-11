import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('resume_overlay', ROOT/'scripts/colab/resume_streaming_training.py')
overlay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(overlay)
REVISION = 'a'*40


def git_fixture(monkeypatch, *, changes=None, protected='', dirty='', base=None):
    def git(repo, *args):
        if args == ('rev-parse', 'HEAD'):
            return base or overlay.BASE_REVISION
        if args[0] == 'status':
            return dirty
        if args[0] == 'rev-parse':
            return REVISION
        if args[0] == 'merge-base':
            return overlay.BASE_REVISION
        if args[-1] == 'src':
            return '\n'.join(changes if changes is not None else overlay.EXECUTION_FILES)
        return protected
    monkeypatch.setattr(overlay, 'git', git)


def test_reviewed_changes_only_and_full_sha(monkeypatch):
    git_fixture(monkeypatch)
    overlay.validate_revision(Path('fixture'), REVISION)
    with pytest.raises(ValueError, match='full lowercase'):
        overlay.validate_revision(Path('fixture'), 'main')


@pytest.mark.parametrize('options,match', [
    ({'changes': ['src/direct_s2st/translatotron2/model.py']}, 'unreviewed'),
    ({'protected': 'configs/translatotron2/train-fisher.yaml'}, 'settings changed'),
    ({'dirty': ' M src/direct_s2st/translatotron2/model.py'}, 'local changes'),
    ({'base': 'b'*40}, 'base revision'),
])
def test_unreviewed_code_environment_or_base_cannot_migrate(monkeypatch, options, match):
    git_fixture(monkeypatch, **options)
    with pytest.raises(ValueError, match=match):
        overlay.validate_revision(Path('fixture'), REVISION)


def fixture(tmp_path):
    environment = tmp_path/'environment.json'
    environment.write_text('{"repository":"'+overlay.BASE_REVISION+'"}')
    backup, work = tmp_path/'backup', tmp_path/'work'
    directory = backup/'000000000005'
    directory.mkdir(parents=True)
    config = dict(kind='tt2', runtime={'fixture': True}, identity_files=[str(environment)],
                  research_metadata={'repository_revision': overlay.BASE_REVISION})
    identity = dict(config=config, work=str(work), inputs={str(environment): hashlib.sha256(environment.read_bytes()).hexdigest()})
    marker = directory/'snapshot.json'
    marker.write_text(json.dumps(dict(identity=identity, updates=5)))
    return config, backup, environment, work, marker


def test_original_environment_config_and_snapshot_are_preserved(tmp_path):
    config, backup, environment, work, marker = fixture(tmp_path)
    before = marker.read_bytes(), environment.read_bytes(), json.dumps(config)
    sha, updates = overlay.validate_snapshot(config, backup, environment, work)
    assert updates == 5 and sha == hashlib.sha256(environment.read_bytes()).hexdigest()
    assert before == (marker.read_bytes(), environment.read_bytes(), json.dumps(config))
    with pytest.raises(ValueError, match='different work path'):
        overlay.validate_snapshot(config, backup, environment, tmp_path/'other')
    environment.write_text('changed dependencies')
    with pytest.raises(ValueError, match='dependency environment differs'):
        overlay.validate_snapshot(config, backup, environment, work)


def test_no_recovery_point_and_changed_config_never_restart_zero(tmp_path):
    config, backup, environment, work, marker = fixture(tmp_path)
    marker.unlink()
    with pytest.raises(FileNotFoundError, match='update zero'):
        overlay.validate_snapshot(config, backup, environment, work)
    marker.write_text('corrupt JSON')
    with pytest.raises(FileNotFoundError, match='no readable'):
        overlay.validate_snapshot(config, backup, environment, work)
    marker.write_text(json.dumps(dict(identity=dict(config=config, work=str(work),
        inputs={str(environment): hashlib.sha256(environment.read_bytes()).hexdigest()}), updates=5)))
    config['runtime'] = {'changed': True}
    with pytest.raises(ValueError, match='config differs'):
        overlay.validate_snapshot(config, backup, environment, work)


def test_selected_calibration_config_is_checked_not_replaced(tmp_path):
    from direct_s2st.batch_calibration import selected_config
    config, backup, environment, work, marker = fixture(tmp_path)
    config.update(command=['fixture', '--batch-size', '8', '--update-freq', '128'],
                  batch_calibration={'max_batch': 1024})
    config['runtime']['adaptive_batch'] = False
    (backup/'batch-calibration.json').write_text('{"selected":16}')
    saved = json.loads(marker.read_text())
    saved['identity']['config'] = selected_config(config, 16)
    marker.write_text(json.dumps(saved))
    assert overlay.validate_snapshot(config, backup, environment, work)[1] == 5
    assert config['command'][config['command'].index('--batch-size')+1] == '8'
    assert json.loads((backup/'batch-calibration.json').read_text()) == {'selected': 16}


def test_child_prefers_separate_overlay_code_without_modifying_base(tmp_path):
    source = tmp_path/'execution-code/src'
    shutil.copytree(ROOT/'src/direct_s2st', source/'direct_s2st', ignore=shutil.ignore_patterns('__pycache__'))
    environment = dict(os.environ, PYTHONPATH=str(source))
    result = subprocess.run([sys.executable, '-c',
        'import direct_s2st, direct_s2st.stream_inputs; print(direct_s2st.__file__)'],
        capture_output=True, text=True, env=environment, timeout=30)
    assert result.returncode == 0, result.stderr
    assert Path(result.stdout.strip()).is_relative_to(source)
