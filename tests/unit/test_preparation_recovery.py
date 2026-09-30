import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def module(name):
    path = ROOT / 'scripts/colab' / f'{name}.py'
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def test_bootstrap_does_not_install_python310_inputs_into_host_python(monkeypatch):
    setup = module('setup')
    for name in ('PIP_CONSTRAINT', 'PIP_REQUIREMENT', 'PIP_BUILD_CONSTRAINT'):
        monkeypatch.setenv(name, 'old-python310-input.txt')
    monkeypatch.setenv('CORPUS_ROOT', '/readonly/corpus')
    env = setup.bootstrap_environment()
    assert env['CORPUS_ROOT'] == '/readonly/corpus'
    assert not any(name in env for name in ('PIP_CONSTRAINT', 'PIP_REQUIREMENT', 'PIP_BUILD_CONSTRAINT'))
    assert setup.os.environ['PIP_CONSTRAINT'] == 'old-python310-input.txt'


def test_recovery_checks_saved_all_split_labels_and_refuses_stale_input(tmp_path):
    from test_translatotron2_prepare import _common
    from direct_s2st.translatotron2.phonemize import phonemize_manifests
    recovery = module('resume_preparation')
    common, phones = tmp_path / 'common', tmp_path / 'translatotron2/phonemes'
    _common(common)
    phonemize_manifests(common, phones, phonemizer=lambda _: 'a', engine='espeak-ng',
                       version='1.52.0', fixed_vocabulary=['a'])
    assert recovery.check_saved_phonemes(tmp_path)['counts'] == dict(train=1, dev=1, test=1)
    original = (phones / 'dev.tsv').read_bytes()
    (phones / 'dev.tsv').write_text('pair-wrong\ta\n', encoding='utf-8')
    with pytest.raises(ValueError, match='incomplete'):
        recovery.check_saved_phonemes(tmp_path)
    (phones / 'dev.tsv').write_bytes(original)
    with (common / 'train.jsonl').open('a') as stream:
        stream.write('\n')
    with pytest.raises(ValueError, match='stale train'):
        recovery.check_saved_phonemes(tmp_path)


def test_revision_migration_requires_overwrite_and_retains_old_pin(tmp_path):
    recovery = module('resume_preparation')
    old, new = 'a' * 40, 'b' * 40
    pin = tmp_path / 'repository-revision.txt'
    pin.write_text(old + '\n')
    with pytest.raises(ValueError, match='explicit --overwrite'):
        recovery.update_revision(tmp_path, new)
    assert pin.read_text().strip() == old
    recovery.update_revision(tmp_path, new, overwrite=True)
    assert pin.read_text().strip() == new
    assert (tmp_path / f'repository-revision.before-{new[:12]}.txt').read_text().strip() == old
    recovery.update_revision(tmp_path, new, overwrite=True)
    (tmp_path / 'training-run.json').write_text(json.dumps({'updates': 1}))
    with pytest.raises(ValueError, match='training configuration'):
        recovery.update_revision(tmp_path, 'c' * 40, overwrite=True)
    assert pin.read_text().strip() == new


def test_recovery_continues_from_prepare_without_running_phonemization(tmp_path, monkeypatch):
    from test_translatotron2_prepare import _common
    from direct_s2st.translatotron2.phonemize import phonemize_manifests
    import runpy
    recovery = module('resume_preparation')
    persistent, corpus, repo = [tmp_path / name for name in ('experiment', 'corpus', 'repo')]
    common, phones = persistent / 'data/common', persistent / 'data/translatotron2/phonemes'
    _common(common)
    phonemize_manifests(common, phones, phonemizer=lambda _: 'a', engine='espeak-ng',
                       version='1.52.0', fixed_vocabulary=['a'])
    old, new = 'a' * 40, 'b' * 40
    (persistent / 'repository-revision.txt').write_text(old + '\n')
    monkeypatch.setenv('CORPUS_ROOT', str(corpus))
    monkeypatch.setattr(recovery, 'ROOT', repo)
    monkeypatch.setattr(recovery.subprocess, 'check_output',
                        lambda args, **kwargs: new if 'rev-parse' in args else '')
    monkeypatch.setattr(runpy, 'run_path', lambda _: {'ensure_runtime': lambda *a, **k: 'fixture-python'})
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        if command[4] == 'prepare':
            prepared = persistent / 'data/translatotron2/fairseq'
            prepared.mkdir(parents=True, exist_ok=True)
            (prepared / 'fixture.tsv').write_text('prepared')
    monkeypatch.setattr(recovery.subprocess, 'run', run)
    monkeypatch.setattr(recovery.sys, 'argv', ['resume', '--persistent', str(persistent),
                                           '--revision', new, '--profile', 'pilot', '--overwrite'])
    recovery.main()
    assert [call[4] for call in calls] == ['prepare', 'validate']
    assert all('--resume' in call and '--overwrite' not in call for call in calls)
    assert (persistent / 'data/.prep-checkpoints/stages/translatotron2-validate.json').is_file()
    assert (persistent / f'repository-revision.before-{new[:12]}.txt').read_text().strip() == old
