import importlib.util
import json
import os
from pathlib import Path

import pytest


@pytest.fixture
def setup_module_fixture():
    path = Path(__file__).resolve().parents[2] / 'scripts/colab/setup.py'
    spec = importlib.util.spec_from_file_location('colab_setup_copy', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def source_tree(tmp_path):
    source = tmp_path / 'source'
    source.mkdir()
    (source / 'setup.py').write_text('# fixture')
    (source / '.git').write_text('protected source metadata')
    return source


def test_dangling_kaldi_links_are_preserved(tmp_path, setup_module_fixture):
    source = source_tree(tmp_path)
    folder = source / 'examples/wav2vec/unsupervised/kaldi_self_train/st'
    folder.mkdir(parents=True)
    try:
        for name in ('utils', 'steps'):
            (folder / name).symlink_to('../../wsj/s5/' + name, target_is_directory=True)
    except OSError as error:
        pytest.skip(f'symlink privilege unavailable: {error}')
    copied = setup_module_fixture.prepare_fairseq_copy(source, tmp_path / 'runtime')
    for name in ('utils', 'steps'):
        link = copied / folder.relative_to(source) / name
        assert link.is_symlink() and not link.exists()
        assert os.readlink(link) == os.readlink(folder / name)
    assert not (copied / '.git').exists()
    assert (source / '.git').read_text() == 'protected source metadata'


def test_old_partial_copy_is_retained_and_complete_copy_reused(tmp_path, setup_module_fixture, monkeypatch):
    source = source_tree(tmp_path)
    runtime = tmp_path / 'runtime'
    prior = runtime / 'fairseq'
    prior.mkdir(parents=True)
    (prior / 'partial.txt').write_text('preserve me')
    copied = setup_module_fixture.prepare_fairseq_copy(source, runtime)
    backups = list(runtime.glob('fairseq.incomplete-*'))
    assert len(backups) == 1 and (backups[0] / 'partial.txt').read_text() == 'preserve me'
    assert (copied / 'setup.py').is_file()
    assert not list(runtime.glob('.fairseq-copy-*'))
    (copied / 'setup.py').write_text('# applied compatibility patch')
    monkeypatch.setattr(setup_module_fixture.shutil, 'copytree', lambda *a, **k: pytest.fail('repeated copy'))
    assert setup_module_fixture.prepare_fairseq_copy(source, runtime) == copied
    assert (copied / 'setup.py').read_text() == '# applied compatibility patch'


def test_copy_failure_never_publishes_or_moves_old_copy(tmp_path, setup_module_fixture, monkeypatch):
    source = source_tree(tmp_path)
    runtime = tmp_path / 'runtime'
    prior = runtime / 'fairseq'
    prior.mkdir(parents=True)
    (prior / 'partial.txt').write_text('original')
    real_copy = setup_module_fixture.shutil.copytree
    def fail(src, dst, **kwargs):
        assert kwargs['symlinks'] is True
        Path(dst).mkdir()
        (Path(dst) / 'partial.txt').write_text('interrupted')
        raise OSError('simulated interruption')
    monkeypatch.setattr(setup_module_fixture.shutil, 'copytree', fail)
    with pytest.raises(OSError, match='interruption'):
        setup_module_fixture.prepare_fairseq_copy(source, runtime)
    assert (prior / 'partial.txt').read_text() == 'original'
    assert not (prior / '.s2st-copy-complete.json').exists()
    assert len(list(runtime.glob('.fairseq-copy-*'))) == 1
    monkeypatch.setattr(setup_module_fixture.shutil, 'copytree', real_copy)
    assert (setup_module_fixture.prepare_fairseq_copy(source, runtime) / 'setup.py').is_file()


def test_copy_rejects_wrong_identity_and_unsafe_roots(tmp_path, setup_module_fixture, monkeypatch):
    source = source_tree(tmp_path)
    with pytest.raises(ValueError, match='separate'):
        setup_module_fixture.prepare_fairseq_copy(source, source / 'runtime')
    runtime = tmp_path / 'runtime'
    copied = setup_module_fixture.prepare_fairseq_copy(source, runtime)
    (copied / '.s2st-copy-complete.json').write_text(json.dumps({'revision': 'different'}))
    with pytest.raises(ValueError, match='identity'):
        setup_module_fixture.prepare_fairseq_copy(source, runtime)
    monkeypatch.setenv('CORPUS_ROOT', str(tmp_path))
    with pytest.raises(ValueError, match='CORPUS_ROOT'):
        setup_module_fixture.prepare_fairseq_copy(source, runtime)
