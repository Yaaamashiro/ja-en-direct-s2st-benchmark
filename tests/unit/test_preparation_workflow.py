import json
from pathlib import Path

import numpy as np
import pytest

from direct_s2st.preparation_workflow import run_stage
from direct_s2st.translatotron2.recovery import publish_archives, validate_prepared


def test_receipt_skips_completed_step_and_rejects_changed_inputs(tmp_path):
    source, output, marker = [tmp_path/p for p in ('input', 'output', 'receipt.json')]
    source.write_text('input')
    calls = []
    def run():
        calls.append(1)
        output.write_text('output')
    kwargs = dict(marker=marker, identity={'revision': 'fixed'}, inputs=[source], outputs=[output], run=run)
    run_stage(**kwargs)
    run_stage(**kwargs)
    assert len(calls) == 1
    source.write_text('changed input')
    run_stage(**kwargs)
    assert len(calls) == 2
    output.unlink()
    run_stage(**kwargs)
    assert len(calls) == 3
    run_stage(**kwargs, force=True)
    assert len(calls) == 4


def test_failed_stage_never_publishes_completion(tmp_path):
    source = tmp_path/'input'
    source.write_text('input')
    marker = tmp_path/'receipt.json'
    def fail():
        raise KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        run_stage(marker, {}, [source], [tmp_path/'output'], fail)
    assert not marker.exists()


def test_archive_shards_resume_and_detect_corruption(tmp_path, monkeypatch):
    from direct_s2st.translatotron2 import prepare_fairseq as module
    files = []
    for index in range(3):
        path = tmp_path/f'{index}.npy'
        np.save(path, np.ones((2, 80), dtype=np.float32))
        files.append((path.name, path))
    original = module._write_feature_zip
    calls = []
    def write(root, path, entries):
        calls.append(path.name)
        if len(calls) == 2:
            raise KeyboardInterrupt()
        original(root, path, entries)
    monkeypatch.setattr(module, '_write_feature_zip', write)
    args = dict(resume=True, max_files=1)
    with pytest.raises(KeyboardInterrupt):
        publish_archives(files, tmp_path/'mel.zip', **args)
    paths, lengths, hashes = publish_archives(files, tmp_path/'mel.zip', **args)
    assert calls.count('mel.zip') == 1
    assert len(paths) == len(hashes) == 3
    assert lengths == {'0': 2, '1': 2, '2': 2}
    from direct_s2st.translatotron2.data import load_mel
    assert load_mel(tmp_path, paths['2']).shape == (2, 80)
    (tmp_path/'mel-00001.zip').write_bytes(b'corrupt')
    with pytest.raises(Exception):
        publish_archives(files, tmp_path/'mel.zip', **args)


def test_final_validation_resumes_without_recomputing_success(tmp_path, monkeypatch):
    from direct_s2st.translatotron2 import data
    source = tmp_path/'audio.wav'
    source.write_bytes(b'fixture')
    root = tmp_path/'fairseq'
    root.mkdir()
    (root/'mel.zip').write_bytes(b'fixture')
    calls = []
    class Dataset:
        def __init__(self, root, split):
            self.corpus_root = None
            self.rows = [dict(id=split, src_audio=str(source), tgt_audio='mel.zip:0:7')]
        def __len__(self):
            return 1
        def __getitem__(self, index):
            name = self.rows[index]['id']
            calls.append(name)
            if len(calls) == 2:
                raise KeyboardInterrupt()
    monkeypatch.setattr(data, 'PreparedDataset', Dataset)
    monkeypatch.setattr(data, 'fingerprint', lambda root: {})
    with pytest.raises(KeyboardInterrupt):
        validate_prepared(root, resume=True)
    assert validate_prepared(root, resume=True)['splits'] == dict(train=1, dev=1, test=1)
    assert calls == ['train', 'dev', 'dev', 'test']
    validate_prepared(root, resume=True)
    assert len(calls) == 4
    monkeypatch.setenv('S2ST_PREP_RECHECK', '1')
    validate_prepared(root, resume=True)
    assert len(calls) == 7


def test_s2ut_generation_keeps_completed_shards(tmp_path):
    from direct_s2st.s2ut.fairseq_infer import generate_shards
    calls = []
    def runner(args, **kwargs):
        shard = int(args[args.index('--distributed-rank')+1])
        calls.append(shard)
        if len(calls) == 2:
            raise KeyboardInterrupt()
        directory = Path(args[args.index('--results-path')+1])
        directory.mkdir(parents=True)
        (directory/'generate-test.txt').write_text(f'D-{shard}\t0\t1 2\n')
    args = (['fixture', '--results-path', str(tmp_path)], tmp_path, 'test', ['a', 'b', 'c'], {'model': 'fixed'})
    with pytest.raises(KeyboardInterrupt):
        generate_shards(*args, resume=True, shard_size=1, runner=runner)
    records, timings = generate_shards(*args, resume=True, shard_size=1, runner=runner)
    assert calls == [0, 1, 1, 2]
    assert records == {key: [1, 2] for key in ('a', 'b', 'c')}
    assert set(timings) == set(records)


def test_legacy_zip_index_resumes_after_interruption(tmp_path, monkeypatch):
    from direct_s2st.translatotron2.prepare_fairseq import _write_feature_zip, _zip_manifest
    for index in range(3):
        np.save(tmp_path / f'{index}.npy', np.ones((index + 1, 80), dtype=np.float32))
    path = tmp_path / 'legacy.zip'
    _write_feature_zip(tmp_path, path)
    original = np.load
    calls = []
    def load(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise KeyboardInterrupt()
        return original(*args, **kwargs)
    monkeypatch.setattr(np, 'load', load)
    with pytest.raises(KeyboardInterrupt):
        _zip_manifest(path, resume=True)
    _, lengths = _zip_manifest(path, resume=True)
    assert lengths == {'0': 1, '1': 2, '2': 3}
    assert len(calls) == 4
    _zip_manifest(path, resume=True)
    assert len(calls) == 4


def test_corrupt_receipt_and_changed_input_never_skip_work(tmp_path):
    source, output, marker = [tmp_path / p for p in ('input', 'output', 'receipt.json')]
    source.write_text('input')
    def mutate():
        source.write_text('changed during execution')
        output.write_text('output')
    with pytest.raises(ValueError, match='inputs changed'):
        run_stage(marker, {}, [source], [output], mutate)
    assert not marker.exists()
    run_stage(marker, {}, [source], [output], lambda: None)
    document = json.loads(marker.read_text())
    document['sha256'] = 'invalid'
    marker.write_text(json.dumps(document))
    with pytest.raises(ValueError, match='corrupt'):
        run_stage(marker, {}, [source], [output], lambda: pytest.fail('must not execute'))


def test_s2ut_preparation_reuses_source_headers_and_units(tmp_path, monkeypatch):
    from test_completion_validation import prepared
    from direct_s2st.s2ut import prepare_fairseq as module
    target = prepared(tmp_path)
    monkeypatch.setattr(module, '_ten_ms_frames', lambda _: pytest.fail('header reread'))
    monkeypatch.setattr(module, 'load_unit_file', lambda *a, **k: pytest.fail('units reread'))
    module.prepare_fairseq(tmp_path / 'common', tmp_path / 'units', target, resume=True)


def test_acceptance_wav_checks_resume_and_detect_changed_content(tmp_path, monkeypatch):
    from test_evaluation import _wav, _prediction
    from direct_s2st.evaluation import acceptance
    from direct_s2st.hashing import sha256_file
    audio = tmp_path / 'output.wav'
    _wav(audio, sample=100)
    rows = [_prediction('ok', audio, 'success')]
    lock = dict(audio={str(audio): sha256_file(audio)})
    acceptance._verify_audio(rows, lock, tmp_path, resume=True)
    monkeypatch.setattr(acceptance.sf, 'read', lambda *a, **k: pytest.fail('WAV reread'))
    acceptance._verify_audio(rows, lock, tmp_path, resume=True)
    audio.write_bytes(audio.read_bytes() + b'changed')
    with pytest.raises(ValueError, match='content has changed'):
        acceptance._verify_audio(rows, lock, tmp_path, resume=True)


def test_cpu_audio_stage_never_fetches_models_and_gpu_requires_both_languages(tmp_path, monkeypatch):
    import sys
    from direct_s2st import preparation_workflow as workflow
    from direct_s2st import drive_staging as stage
    from test_drive_staging import audio_fixture
    data = tmp_path/'data'
    data.mkdir()
    common, rows = audio_fixture(data)
    rows = [row | dict(ja_audio=row['en_audio'], ja_sha256=row['en_sha256']) for row in rows]
    for split in ('train', 'dev', 'test'):
        (common/f'{split}.jsonl').write_text(''.join(json.dumps(row | dict(split=split)) + '\n' for row in rows))
    (common/'dataset-lock.json').write_text('{}')
    corpus = tmp_path/'corpus'
    corpus.mkdir()
    monkeypatch.setenv('CORPUS_ROOT', str(corpus))
    monkeypatch.setenv('EXPERIMENT_DATA_ROOT', str(data))
    monkeypatch.setenv('S2ST_DRIVE_SAFE', '1')
    monkeypatch.setattr(workflow.subprocess, 'check_output', lambda *a, **k: 'a'*40)
    monkeypatch.setattr(workflow.subprocess, 'run', lambda *a, **k: pytest.fail('unexpected model/Unit/Mel command'))
    import tempfile
    monkeypatch.setattr(tempfile, 'gettempdir', lambda: str(tmp_path/'cpu'))
    monkeypatch.setattr(sys, 'argv', ['preparation', '--stage', 'audio-packs'])
    workflow.main()
    assert len(list((data/'.drive-audio-packs').rglob('*.zip'))) == 6
    # All English packs exist, but a Japanese publication is missing. Stop
    # before fetch-artifacts, HuBERT or any English ZIP transfer on the GPU.
    archive = stage.audio_archive_path(data/'.drive-audio-packs', stage.raw_sha(common/'test.jsonl'), 'test', 'ja', 0, 128)
    archive.with_suffix('.json').unlink()
    monkeypatch.setattr(sys, 'argv', ['preparation', '--stage', '4b'])
    with pytest.raises(RuntimeError, match='CPU audio packing incomplete'):
        workflow.main()


def test_cpu_audio_migration_updates_only_revision_and_rejects_training(tmp_path, monkeypatch):
    import importlib.util
    import sys
    import runpy
    repo = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location('audio_pack_recovery', repo/'scripts/colab/prepare_audio_packs.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    update_revision = runpy.run_path(str(repo/'scripts/colab/resume_preparation.py'))['update_revision']
    persistent = tmp_path/'experiment'
    common = persistent/'data/common'
    common.mkdir(parents=True)
    (common/'dataset-lock.json').write_text('{}')
    for split in ('train', 'dev', 'test'):
        (common/f'{split}.jsonl').write_text('{}\n')
    old, new = 'a'*40, 'b'*40
    pin = persistent/'repository-revision.txt'
    pin.write_text(old + '\n')
    artifact = persistent/'data/translatotron2/mel.zip'
    artifact.parent.mkdir()
    artifact.write_bytes(b'unchanged Mel')
    monkeypatch.setenv('CORPUS_ROOT', str(tmp_path/'corpus'))
    monkeypatch.setattr(module.subprocess, 'check_output',
                        lambda command, **kw: new if command[-1] == 'HEAD' else '')
    monkeypatch.setattr(sys, 'argv', ['recovery', '--persistent', str(persistent), '--revision', new, '--overwrite'])
    calls = []
    def ensure_runtime(repo, persistent, run, **kwargs):
        assert kwargs == {'require_gpu': False}
        return 'fixture-python'
    monkeypatch.setattr(module.runpy, 'run_path', lambda path: (
        {'update_revision': update_revision} if str(path).endswith('resume_preparation.py')
        else {'ensure_runtime': ensure_runtime}))
    monkeypatch.setattr(module.subprocess, 'run', lambda args, **kwargs: calls.append(args))
    module.main()
    assert pin.read_text().strip() == new
    assert (persistent/f'repository-revision.before-{new[:12]}.txt').read_text().strip() == old
    assert calls == [['fixture-python', '-m', 'direct_s2st.preparation_workflow', '--stage', 'audio-packs', '--profile', 'smoke']]
    assert artifact.read_bytes() == b'unchanged Mel'
    (persistent/'runs/model').mkdir(parents=True)
    (persistent/'runs/model/checkpoint.pt').write_bytes(b'preserve')
    with pytest.raises(ValueError, match='training configuration/checkpoints'):
        module.main()
