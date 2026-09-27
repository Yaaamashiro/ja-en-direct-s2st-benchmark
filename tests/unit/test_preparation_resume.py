import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from direct_s2st.preparation import Checkpoints, WorkerController, adaptive_map, checkpoint_map


def test_checkpoint_flushes_on_interrupt_and_ignores_unpublished_files(tmp_path, monkeypatch):
    monkeypatch.setenv('S2ST_PREP_WORKERS', '1')
    calls = []
    def work(value):
        calls.append(value)
        if value == 2:
            raise KeyboardInterrupt()
        return value * 10
    with pytest.raises(KeyboardInterrupt):
        with Checkpoints(tmp_path, {'stage': 1}) as cache:
            list(checkpoint_map(work, range(4), cache, str, 'fixture'))
    (cache.root / '.incomplete.tmp').write_text('broken')
    calls.clear()
    with Checkpoints(tmp_path, {'stage': 1}, resume=True) as cache:
        assert list(checkpoint_map(lambda n: calls.append(n) or n * 10,
                                   range(4), cache, str, 'fixture')) == [0, 10, 20, 30]
    assert calls == [2, 3]
    chunk = next(cache.root.glob('chunk-*.json'))
    document = json.loads(chunk.read_text())
    document['sha256'] = 'bad'
    chunk.write_text(json.dumps(document))
    with pytest.raises(ValueError, match='corrupt'):
        Checkpoints(tmp_path, {'stage': 1}, resume=True)


def test_workers_are_bounded_ordered_and_back_off():
    assert list(adaptive_map(lambda n: n * 2, range(10), maximum=3)) == list(range(0, 20, 2))
    controller = WorkerController(3)
    assert controller.observe(10, cpu=10, free_ram=.8) == 3
    assert controller.observe(5, cpu=10, free_ram=.8) == 2
    assert controller.observe(10, cpu=90, free_ram=.1) == 1


def test_explicit_overwrite_starts_new_checkpoint_generation(tmp_path):
    with Checkpoints(tmp_path, {'stage': 1}) as cache:
        cache.record('sample', [1])
    old_root = cache.root
    with Checkpoints(tmp_path, {'stage': 1}, overwrite=True) as cache:
        cache.record('sample', [2])
    with Checkpoints(tmp_path, {'stage': 1}, resume=True) as cache:
        assert cache.get('sample') == [2]
    assert list(old_root.glob('chunk-*.json'))  # Old state is recoverable, not deleted.


def test_wav_resume_reuses_hashes_but_rechecks_changes(tmp_path, monkeypatch):
    from test_common_manifest import _accepted_row
    from direct_s2st.manifests.schema import CommonManifestRow
    from direct_s2st.manifests import validate
    corpus = tmp_path / 'corpus'
    row = CommonManifestRow.from_corpus_row(_accepted_row(corpus, 'pair', 'train'),
                                           corpus_root=corpus, manifest_parent=tmp_path)
    options = dict(checkpoint_root=tmp_path / 'cache', resume=True)
    original = validate.sha256_file
    calls = []
    monkeypatch.setattr(validate, 'sha256_file', lambda p: calls.append(p) or original(p))
    validate.validate_rows([row], **options)
    assert len(calls) == 2
    calls.clear()
    validate.validate_rows([row], **options)
    assert calls == []
    monkeypatch.setenv('S2ST_PREP_RECHECK', '1')
    validate.validate_rows([row], **options)
    assert len(calls) == 2
    monkeypatch.setenv('S2ST_PREP_RECHECK', '0')
    path = Path(row.en_audio)
    stamp = path.stat()
    path.write_bytes(path.read_bytes()[:-2] + b'\x01\x00')
    os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns + 1000000000))
    with pytest.raises(validate.ManifestValidationError, match='SHA-256'):
        validate.validate_rows([row], **options)


def test_mel_and_phonemes_resume_after_failure(tmp_path, monkeypatch):
    from test_translatotron2_prepare import _common, _mel_fixture
    from direct_s2st.translatotron2.phonemize import phonemize_manifests
    from direct_s2st.translatotron2.prepare_fairseq import prepare_fairseq
    monkeypatch.setenv('S2ST_PREP_WORKERS', '1')
    common, phones, output = [tmp_path / p for p in ('common', 'phones', 'fairseq')]
    _common(common)
    count = 0
    def phonemizer(_):
        nonlocal count
        count += 1
        if count == 2:
            raise RuntimeError('interrupted')
        return 'A'
    options = dict(phonemizer=phonemizer, engine='fixture', version='1', fixed_vocabulary=['A'], resume=True)
    with pytest.raises(RuntimeError, match='interrupted'):
        phonemize_manifests(common, phones, **options)
    phonemize_manifests(common, phones, **options)
    assert count == 4  # First train result is reused.
    calls = []
    def mel(path, destination, settings):
        calls.append(path.name)
        if len(calls) == 2:
            raise RuntimeError('interrupted')
        _mel_fixture(path, destination, settings)
    options = dict(mel_config={'n_mels': 80}, feature_extractor=mel, resume=True)
    with pytest.raises(RuntimeError, match='interrupted'):
        prepare_fairseq(common, phones, output, **options)
    prepare_fairseq(common, phones, output, **options)
    assert calls.count('dev-en.wav') == 1
    assert (output / 'logmelspec80.zip').is_file()
    from direct_s2st.io import ExistingOutputError
    with pytest.raises(ExistingOutputError):
        prepare_fairseq(common, phones, output, **(options | dict(mel_config={'n_mels': 80, 'hop_length': 160})))


def test_units_resume_reconstruct_missing_files_and_reject_wrong_identity(tmp_path):
    from test_s2ut_units import _common
    from direct_s2st.s2ut.extract_units import extract_units
    from direct_s2st.io import ExistingOutputError
    common, output = tmp_path / 'common', tmp_path / 'units'
    _common(common)
    calls = []
    def extractor(path):
        calls.append(path.name)
        if len(calls) == 2:
            raise RuntimeError('interrupted')
        return [1, 1, 2]
    options = dict(extractor=extractor, split=None, clusters=100, hubert_model='fixture',
                   hubert_revision='a' * 40, hubert_layer=6, kmeans_sha256='b' * 64, resume=True)
    with pytest.raises(RuntimeError, match='interrupted'):
        extract_units(common, output, **options)
    (output / 'train/original/pair-train.units').unlink()
    extract_units(common, output, **options)
    assert calls.count('train-en.wav') == 1
    with pytest.raises(ExistingOutputError):
        extract_units(common, output, **(options | dict(hubert_revision='c' * 40)))
    (output / 'train/original/pair-train.units').write_text('9\n')
    with pytest.raises(ExistingOutputError):
        extract_units(common, output, **options)


def test_hubert_batches_only_equal_lengths_and_retries_oom(monkeypatch):
    import numpy as np
    from direct_s2st.s2ut.extract_units import HubertKMeansExtractor
    class OOM(RuntimeError):
        pass
    extractor = object.__new__(HubertKMeansExtractor)
    extractor.device = SimpleNamespace(type='cuda')
    extractor.torch = SimpleNamespace(cuda=SimpleNamespace(
        OutOfMemoryError=OOM, empty_cache=lambda: None, mem_get_info=lambda _: (80, 100)))
    monkeypatch.setenv('S2ST_HUBERT_BATCH_MAX', '4')
    extractor._batch_size = 4
    extractor._read_audio = lambda n: np.zeros(n)
    batches = []
    def batch(waves):
        lengths = [len(wave) for wave in waves]
        assert len(set(lengths)) == 1
        batches.append(len(waves))
        if len(waves) > 2:
            raise OOM()
        return [[length] for length in lengths]
    extractor._batch = batch
    assert extractor.extract_many([3, 4, 3, 3, 3]) == [[3], [4], [3], [3], [3]]
    assert batches[:2] == [4, 2]
