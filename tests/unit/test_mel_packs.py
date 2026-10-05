import io
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

import numpy as np
import pytest

from direct_s2st.hashing import sha256_file
from direct_s2st.journal import digest
from direct_s2st.preparation import Checkpoints
from direct_s2st.translatotron2.mel_packs import MelPacks, export_packs, npy_frames
from test_preparation_recovery import module


def real_legacy(tmp_path, count=3):
    identity = dict(stage='mel-v1', fixture='packed')
    with Checkpoints(tmp_path / 'mel', identity) as cache:
        for i in range(count):
            key = digest(['packed-fixture', i])
            path = cache.root / 'features' / (key[:32] + '.npy')
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, np.full((11 + i, 80), i, dtype=np.float32))
            cache.record(key, dict(id=f'pair-{i}', path=str(path), sha256=sha256_file(path)))
    return cache.root, cache.rows


def add(store, key, row):
    local = store.local / (key[:32] + '.npy')
    local.write_bytes(Path(row['path']).read_bytes())
    store.add(key, row, local)


def test_pack_and_training_locators_preserve_exact_payloads(tmp_path):
    root, rows = real_legacy(tmp_path)
    originals = {Path(r['path']): Path(r['path']).read_bytes() for r in rows.values()}
    chunks = {p: p.read_bytes() for p in root.glob('chunk-*.json')}
    output = tmp_path / 'prepared'
    output.mkdir()
    with MelPacks(root, max_files=2) as store:
        for key, row in rows.items():
            add(store, key, row)
        store.flush()
        paths, lengths, hashes = export_packs(store, list(rows), output / 'logmelspec80.zip')
        assert len(hashes) == 2
        for row in rows.values():
            filename, offset, length = paths[row['id']].split(':')
            with (output / filename).open('rb') as raw:
                raw.seek(int(offset))
                payload = raw.read(int(length))
            assert payload == originals[Path(row['path'])]
            assert np.load(io.BytesIO(payload), allow_pickle=False).shape == (lengths[row['id']], 80)
        before = {p: p.read_bytes() for p in output.glob('*.zip')}
        assert export_packs(store, list(rows), output / 'logmelspec80.zip', resume=True)[0] == paths
        assert before == {p: p.read_bytes() for p in before}
    assert originals == {p: p.read_bytes() for p in originals}
    assert chunks == {p: p.read_bytes() for p in chunks}
    assert not (root / 'features-v2').exists()


def test_lost_vm_local_disk_reuses_only_durable_packs(tmp_path):
    root, rows = real_legacy(tmp_path)
    keys = list(rows)
    store = MelPacks(root, max_files=1)
    add(store, keys[0], rows[keys[0]])
    store.max_files = 128
    add(store, keys[1], rows[keys[1]])
    store.owner.cleanup()  # VM dies without a graceful __exit__/flush.
    with MelPacks(root) as restarted:
        assert set(restarted.rows) == {keys[0]}
        for key in keys[1:]:
            add(restarted, key, rows[key])
    with MelPacks(root) as restarted:
        assert set(restarted.rows) == set(rows)


def test_orphan_zip_adopted_without_overwriting_payload(tmp_path):
    root, rows = real_legacy(tmp_path, 1)
    key, row = next(iter(rows.items()))
    with MelPacks(root) as store:
        add(store, key, row)
    archive = next((root / 'packs-v1').glob('*/*.zip'))
    before = archive.read_bytes()
    archive.with_suffix('.json').unlink()  # Crash between archive and receipt.
    with MelPacks(root) as store:
        add(store, key, row)
    assert archive.read_bytes() == before
    with MelPacks(root) as store:
        assert key in store.rows


def test_corrupt_pack_or_receipt_stops_without_fallback(tmp_path):
    root, rows = real_legacy(tmp_path, 1)
    key, row = next(iter(rows.items()))
    with MelPacks(root) as store:
        add(store, key, row)
    receipt = next((root / 'packs-v1').glob('*/pack-*.json'))
    before = receipt.read_bytes()
    receipt.write_text('{}')
    with pytest.raises(ValueError, match='corrupt Mel pack receipt'):
        MelPacks(root)
    receipt.write_bytes(before)
    receipt.with_suffix('.zip').write_bytes(b'broken')
    with pytest.raises(ValueError, match='corrupt cached Mel'):
        MelPacks(root)
    assert Path(row['path']).is_file()


def test_zip_repair_and_distributed_copy_import_without_api(tmp_path):
    repair = module('repair_mel_cache')
    root, rows = real_legacy(tmp_path)
    keys = sorted(rows)
    row = rows[keys[0]]
    distributed = root / 'features-v2' / keys[0][:2] / (keys[0][:32] + '.npy')
    distributed.parent.mkdir(parents=True)
    distributed.write_bytes(Path(row['path']).read_bytes())
    source_zip = tmp_path / 'folder-download.zip'
    with zipfile.ZipFile(source_zip, 'w', zipfile.ZIP_DEFLATED) as archive:
        for key in keys[1:]:
            source = Path(rows[key]['path'])
            archive.write(source, 'Takeout/features/' + source.name)
    reader = SimpleNamespace(index=lambda *a: pytest.fail('all bytes available without API'))
    before = distributed.read_bytes()
    result = repair.repair_packed(root, source_zips=[source_zip], reader=reader)
    assert result['persisted'] == 3 and result['remaining'] == 0
    assert distributed.read_bytes() == before
    # Resuming the rescue no longer needs even the supplied archive or old files.
    assert repair.repair_packed(root, reader=reader)['reused'] == 3


def test_api_failure_publishes_completed_local_work_and_resume_skips_downloads(tmp_path):
    repair = module('repair_mel_cache')
    root, rows = real_legacy(tmp_path)
    payloads = {Path(r['path']).name: Path(r['path']).read_bytes() for r in rows.values()}
    calls = []
    def download(name, local):
        calls.append(name)
        assert local.is_relative_to(Path(__import__('tempfile').gettempdir()))
        assert not local.is_relative_to(root)
        if len(calls) == 2:
            raise RuntimeError('network interrupted')
        local.write_bytes(payloads[name])
    reader = SimpleNamespace(index=lambda *a: {name: name for name in payloads}, download=download)
    with pytest.raises(RuntimeError, match='network interrupted'):
        repair.repair_packed(root, reader=reader, folder_id='features-id')
    with MelPacks(root) as store:
        assert len(store.rows) == 1
    calls.clear()
    reader.download = lambda name, local: calls.append(name) or local.write_bytes(payloads[name])
    result = repair.repair_packed(root, reader=reader, folder_id='features-id')
    assert len(calls) == 2 and result['reused'] == 1 and result['persisted'] == 3
    assert not (root / 'features-v2').exists()


@pytest.mark.parametrize('name', ['../bad.npy', '/root.npy', 'C:/bad.npy'])
def test_unsafe_zip_paths_rejected(tmp_path, name):
    repair = module('repair_mel_cache')
    source = tmp_path / 'bad.zip'
    with zipfile.ZipFile(source, 'w') as archive:
        archive.writestr(name, b'bad')
    with pytest.raises(ValueError, match='unsafe source ZIP'):
        repair.ZipSources([source], set())


def test_wrong_source_checksum_and_invalid_npy_never_publish(tmp_path):
    root, rows = real_legacy(tmp_path, 1)
    key, row = next(iter(rows.items()))
    with MelPacks(root) as store:
        local = store.local / 'wrong.npy'
        local.write_bytes(b'bad')
        with pytest.raises(ValueError, match='SHA256 mismatch'):
            store.add(key, row, local)
        with pytest.raises(ValueError, match='NPY magic'):
            npy_frames(local)
    assert not list((root / 'packs-v1').glob('*/pack-*'))


def test_local_work_cannot_use_drive_or_corpus(tmp_path, monkeypatch):
    root, _ = real_legacy(tmp_path)
    with pytest.raises(ValueError, match='local'):
        MelPacks(root, local_parent=root / 'work')
    monkeypatch.setenv('CORPUS_ROOT', str(tmp_path / 'corpus'))
    with pytest.raises(ValueError, match='CORPUS_ROOT'):
        MelPacks(root, local_parent=tmp_path / 'corpus')


def test_rate_limiter_retries_only_rate_quota_and_remains_slow_after_retry(monkeypatch):
    repair = module('repair_mel_cache')
    sleeps = []
    monkeypatch.setattr(repair.time, 'sleep', lambda n: sleeps.append(n))
    monkeypatch.setattr(repair.random, 'random', lambda: 0)
    class APIError(Exception):
        def __init__(self, status, reason):
            self.resp = SimpleNamespace(status=status)
            self.content = json.dumps({'error': {'errors': [{'reason': reason}]}})
    reader = repair.DriveReader(None, None, interval=1, max_retries=2)
    calls = []
    def limited():
        calls.append(1)
        if len(calls) == 1:
            raise APIError(403, 'rateLimitExceeded')
        return 'okay'
    assert reader.call(limited) == 'okay'
    assert reader.interval == 2 and all(n <= 30 for n in sleeps)
    calls.clear()
    def denied():
        calls.append(1)
        raise APIError(403, 'insufficientPermissions')
    with pytest.raises(APIError):
        reader.call(denied)
    assert len(calls) == 1
    calls.clear()
    def persistent_quota():
        calls.append(1)
        raise APIError(403, 'userRateLimitExceeded')
    with pytest.raises(APIError):
        reader.call(persistent_quota)
    assert len(calls) == 3


def test_completed_old_archive_plan_resumes_without_accessing_npys(tmp_path, monkeypatch):
    from test_translatotron2_prepare import _common, _mel_fixture
    from direct_s2st.translatotron2.phonemize import phonemize_manifests
    from direct_s2st.translatotron2.prepare_fairseq import prepare_fairseq
    common, phones, output = [tmp_path / p for p in ('common', 'phones', 'fairseq')]
    _common(common)
    phonemize_manifests(common, phones, phonemizer=lambda _: 'a',
                       engine='fixture', version='1', fixed_vocabulary=['a'])
    monkeypatch.setenv('S2ST_MEL_STORAGE', 'files')
    options = dict(mel_config={'n_mels': 80}, feature_extractor=_mel_fixture, resume=True)
    prepare_fairseq(common, phones, output, **options)
    original_open = Path.open
    original_stat = Path.stat
    def no_npy_open(path, *args, **kwargs):
        if path.suffix == '.npy':
            pytest.fail('completed archive resume must not read per-NPY Drive files')
        return original_open(path, *args, **kwargs)
    def no_npy_stat(path, *args, **kwargs):
        if path.suffix == '.npy':
            pytest.fail('completed archive resume must not stat per-NPY Drive files')
        return original_stat(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', no_npy_open)
    monkeypatch.setattr(Path, 'stat', no_npy_stat)
    monkeypatch.setenv('S2ST_MEL_STORAGE', 'packed')
    before = (output / 'logmelspec80.zip').read_bytes()
    options['feature_extractor'] = lambda *a: pytest.fail('completed archive must not be regenerated')
    prepare_fairseq(common, phones, output, **options)
    assert (output / 'logmelspec80.zip').read_bytes() == before
