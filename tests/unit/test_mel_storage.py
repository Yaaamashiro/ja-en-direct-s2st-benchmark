import errno
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from direct_s2st.hashing import sha256_file
from direct_s2st.journal import digest
from direct_s2st.preparation import Checkpoints
from direct_s2st.translatotron2 import mel_storage as storage
from test_preparation_recovery import module


def legacy_cache(tmp_path, count=3):
    identity = {'stage': 'mel-v1', 'fixture': 1}
    payloads = {}
    with Checkpoints(tmp_path / 'mel', identity) as cache:
        for i in range(count):
            key = digest(['fixture', i])
            path = cache.root / 'features' / f'{key[:32]}.npy'
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = b'fixture-mel-' + str(i).encode()
            path.write_bytes(payload)
            row = dict(id=f'pair-{i}', path=str(path), sha256=sha256_file(path))
            cache.record(key, row)
            payloads[path.name] = payload
    return cache.root, cache.rows, payloads


def test_distributed_layout_preserves_old_checkpoint_and_source(tmp_path):
    root, rows, _ = legacy_cache(tmp_path)
    chunks = {p: p.read_bytes() for p in root.glob('chunk-*.json')}
    old = {Path(r['path']): Path(r['path']).read_bytes() for r in rows.values()}
    for row in rows.values():
        target = storage.resolve_feature(root, row)
        assert target.parent.parent == root / 'features-v2'
        assert target.parent.name == target.name[:2]
        assert target.read_bytes() == old[Path(row['path'])]
    assert chunks == {p: p.read_bytes() for p in root.glob('chunk-*.json')}
    assert old == {p: p.read_bytes() for p in old}
    assert storage.load_rows(root) == rows


def test_api_recovery_never_reads_legacy_mount_and_can_resume(tmp_path, monkeypatch):
    repair = module('repair_mel_cache')
    root, rows, payloads = legacy_cache(tmp_path)
    chunks = {p: p.read_bytes() for p in root.glob('chunk-*.json')}
    original_open = Path.open
    def forbidden_old_open(path, *args, **kwargs):
        if path.parent.name == 'features':
            raise OSError(errno.EIO, 'DriveFS flat folder unavailable')
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', forbidden_old_open)
    calls = []
    def download(name, temp):
        calls.append(name)
        if len(calls) == 2:
            raise KeyboardInterrupt()
        temp.write_bytes(payloads[name])
    reader = SimpleNamespace(index=lambda *a: {n: n for n in payloads}, download=download)
    with pytest.raises(KeyboardInterrupt):
        repair.repair(root, reader=reader, folder_id='features-id')
    reader.download = lambda name, temp: temp.write_bytes(payloads[name])
    result = repair.repair(root, reader=reader, folder_id='features-id')
    assert result['copied'] == 2 and result['reused'] == 1
    assert repair.repair(root, reader=reader, folder_id='features-id')['reused'] == 3
    for row in rows.values():
        assert storage.resolve_feature(root, row).read_bytes() == payloads[Path(row['path']).name]
    assert chunks == {p: p.read_bytes() for p in root.glob('chunk-*.json')}
    assert not list((root / 'features-v2').glob('*/.recover-*.tmp'))


def test_failed_download_or_checksum_never_publishes_or_reextracts(tmp_path):
    root, rows, _ = legacy_cache(tmp_path, 1)
    row = next(iter(rows.values()))
    source = Path(row['path'])
    before = source.read_bytes()
    with pytest.raises(ValueError, match='SHA256 mismatch'):
        storage.copy_feature(root, row, lambda temp: temp.write_bytes(b'wrong'))
    assert not storage.feature_path(root, source.name).exists()
    assert source.read_bytes() == before
    def denied(temp):
        raise PermissionError('Drive API unauthorized')
    with pytest.raises(PermissionError):
        storage.copy_feature(root, row, denied)
    assert source.read_bytes() == before


def test_receipt_reuses_hash_and_recheck_or_changed_stat_revalidates(tmp_path, monkeypatch):
    root, rows, _ = legacy_cache(tmp_path, 1)
    row = next(iter(rows.values()))
    target = storage.resolve_feature(root, row)
    original_hash = storage.sha256_file
    calls = []
    monkeypatch.setattr(storage, 'sha256_file', lambda p: calls.append(p) or original_hash(p))
    storage.resolve_feature(root, row)
    assert not calls
    monkeypatch.setenv('S2ST_PREP_RECHECK', '1')
    storage.resolve_feature(root, row)
    assert calls == [target]
    monkeypatch.setenv('S2ST_PREP_RECHECK', '0')
    stamp = target.stat()
    target.write_bytes(b'corrupt')
    os.utime(target, ns=(stamp.st_atime_ns, stamp.st_mtime_ns + 1000000000))
    with pytest.raises(ValueError, match='corrupt cached Mel'):
        storage.resolve_feature(root, row)
    assert Path(row['path']).is_file()  # No fallback or overwrite on corruption.


def test_corrupt_receipt_is_not_ignored(tmp_path):
    root, rows, _ = legacy_cache(tmp_path, 1)
    row = next(iter(rows.values()))
    target = storage.resolve_feature(root, row)
    receipt = next(target.parent.glob('*.verified.json'))
    receipt.write_text('{}')
    with pytest.raises(ValueError, match='corrupt Mel verification'):
        storage.resolve_feature(root, row)


def test_copy_published_before_receipt_is_verified_on_restart(tmp_path):
    root, rows, _ = legacy_cache(tmp_path, 1)
    row = next(iter(rows.values()))
    target = storage.feature_path(root, Path(row['path']).name)
    target.parent.mkdir(parents=True)
    target.write_bytes(Path(row['path']).read_bytes())
    assert storage.copy_feature(root, row) == (target, False)
    assert list(target.parent.glob('*.verified.json'))


@pytest.mark.parametrize('name', ['../bad.npy', 'a.npy', 'A' * 32 + '.npy'])
def test_invalid_feature_name_is_rejected(tmp_path, name):
    with pytest.raises(ValueError, match='filename'):
        storage.feature_path(tmp_path, name)


def test_corpus_and_outside_paths_are_rejected(tmp_path, monkeypatch):
    root, rows, _ = legacy_cache(tmp_path, 1)
    row = next(iter(rows.values()))
    with pytest.raises(ValueError, match='outside its cache'):
        storage.checked_row(root, row | {'path': str(tmp_path / Path(row['path']).name)})
    monkeypatch.setenv('CORPUS_ROOT', str(tmp_path))
    with pytest.raises(ValueError, match='CORPUS_ROOT'):
        storage.feature_path(root, Path(row['path']).name)


def test_invalid_checkpoint_or_wrong_generation_is_rejected(tmp_path):
    root, rows, _ = legacy_cache(tmp_path, 1)
    chunk = next(root.glob('chunk-*.json'))
    document = json.loads(chunk.read_text())
    document['sha256'] = 'bad'
    chunk.write_text(json.dumps(document))
    with pytest.raises(ValueError, match='corrupt Mel checkpoint'):
        storage.load_rows(root)
    document['sha256'] = digest(document['rows'])
    chunk.write_text(json.dumps(document))
    renamed = root.with_name('wrong-root')
    root.rename(renamed)
    # Old paths no longer belong to this root; reject before writing anything.
    with pytest.raises(ValueError):
        storage.load_rows(renamed)


def test_retry_is_bounded_and_does_not_retry_permanent_errors(monkeypatch):
    monkeypatch.setattr(storage.time, 'sleep', lambda _: None)
    calls = []
    def transient():
        calls.append(1)
        if len(calls) == 1:
            raise OSError(errno.EIO, 'temporary')
        return 'okay'
    assert storage.retry_io(transient) == 'okay' and len(calls) == 2
    for number, expected in [(errno.EIO, 2), (errno.ENOSPC, 1), (errno.ENOENT, 1), (errno.EACCES, 1)]:
        calls.clear()
        def fail():
            calls.append(1)
            raise OSError(number, 'failed')
        with pytest.raises(OSError):
            storage.retry_io(fail)
        assert len(calls) == expected


def test_unreadable_destination_is_not_treated_as_missing_or_overwritten(tmp_path, monkeypatch):
    root, rows, _ = legacy_cache(tmp_path, 1)
    row = next(iter(rows.values()))
    target = storage.resolve_feature(root, row)
    before = target.read_bytes()
    original_stat = Path.stat
    monkeypatch.setattr(storage.time, 'sleep', lambda _: None)
    def failing_stat(path, *args, **kwargs):
        if path == target and kwargs.get('follow_symlinks', True):
            raise OSError(errno.EIO, 'Drive quota')
        return original_stat(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'stat', failing_stat)
    def forbidden(temp):
        pytest.fail('an unreadable destination must not be replaced')
    with pytest.raises(OSError):
        storage.copy_feature(root, row, forbidden)
    assert target.read_bytes() == before


def test_prepare_resumes_actual_legacy_checkpoint_without_reextracting(tmp_path, monkeypatch):
    from test_translatotron2_prepare import _common, _mel_fixture
    from direct_s2st.translatotron2.phonemize import phonemize_manifests
    from direct_s2st.translatotron2.prepare_fairseq import prepare_fairseq
    monkeypatch.setenv('S2ST_PREP_WORKERS', '1')
    common, phones, output = [tmp_path / p for p in ('common', 'phones', 'fairseq')]
    _common(common)
    phonemize_manifests(common, phones, phonemizer=lambda _: 'a',
                       engine='fixture', version='1', fixed_vocabulary=['a'])
    calls = []
    def interrupted(audio, dest, settings):
        calls.append(audio.name)
        if len(calls) == 2:
            raise KeyboardInterrupt()
        _mel_fixture(audio, dest, settings)
    with pytest.raises(KeyboardInterrupt):
        prepare_fairseq(common, phones, output, mel_config={'n_mels': 80},
                        feature_extractor=interrupted, resume=True)
    root = next((tmp_path / '.prep-checkpoints/mel').iterdir())
    chunk = next(root.glob('chunk-*.json'))
    document = json.loads(chunk.read_text())
    row = next(iter(document['rows'].values()))
    new = Path(row['path'])
    old = root / 'features' / new.name
    old.parent.mkdir()
    old.write_bytes(new.read_bytes())
    # Simulate a saved legacy extraction chunk before the layout upgrade.
    row['path'] = str(old)
    document['sha256'] = digest(document['rows'])
    chunk.write_text(json.dumps(document))
    before = chunk.read_bytes()
    new.unlink()
    calls.clear()
    def remaining(audio, dest, settings):
        calls.append(audio.name)
        _mel_fixture(audio, dest, settings)
    prepare_fairseq(common, phones, output, mel_config={'n_mels': 80},
                    feature_extractor=remaining, resume=True)
    assert len(calls) == 2 and 'dev-en.wav' not in calls
    assert chunk.read_bytes() == before
    assert new.read_bytes() == old.read_bytes()
    assert (output / 'logmelspec80.zip').is_file()


class FakeDrive:
    def __init__(self, root_name, pages):
        self.root_name, self.pages, self.calls = root_name, pages, []

    def files(self):
        return self

    def get(self, **kwargs):
        self.calls.append(('get', kwargs))
        value = (dict(name='features', mimeType='application/vnd.google-apps.folder', parents=['parent'])
                 if kwargs['fileId'] == 'folder-id' else dict(name=self.root_name))
        return SimpleNamespace(execute=lambda **kw: value)

    def list(self, **kwargs):
        self.calls.append(('list', kwargs))
        page = self.pages[0 if kwargs['pageToken'] is None else 1]
        return SimpleNamespace(execute=lambda **kw: page)


def api_entry(name):
    return dict(id='id-' + name, name=name, mimeType='application/octet-stream',
                capabilities={'canDownload': True})


def test_drive_index_pagination_cache_missing_and_duplicate_safety(tmp_path):
    repair = module('repair_mel_cache')
    root, _, payloads = legacy_cache(tmp_path, 2)
    names = sorted(payloads)
    pages = [dict(files=[api_entry(names[0])], nextPageToken='page2'),
             dict(files=[api_entry(names[1])])]
    service = FakeDrive(root.name, pages)
    reader = repair.DriveReader(service, None)
    index = reader.index('folder-id', root, set(names))
    assert index == {n: 'id-' + n for n in names}
    assert len([c for c in service.calls if c[0] == 'list']) == 2
    assert reader.index('folder-id', root, set(names)) == index
    assert len([c for c in service.calls if c[0] == 'list']) == 2
    assert all(call[0] in ('get', 'list') for call in service.calls)
    other, _, _ = legacy_cache(tmp_path / 'other', 1)
    reader = repair.DriveReader(FakeDrive(other.name, [dict(files=[])]), None)
    with pytest.raises(ValueError, match='missing from Drive'):
        reader.index('folder-id', other, set(names))
    reader.service.pages = [dict(files=[api_entry(names[0]), api_entry(names[0])])]
    with pytest.raises(ValueError, match='ambiguous duplicate'):
        reader.index('folder-id', other, set(names))


def test_drive_read_only_download_and_folder_validation(tmp_path):
    repair = module('repair_mel_cache')
    downloads = []
    service = SimpleNamespace(files=lambda: SimpleNamespace(
        get_media=lambda **kw: downloads.append(kw) or 'request'))
    class Downloader:
        def __init__(self, handle, request, chunksize):
            assert request == 'request' and chunksize == 4 * 1024 * 1024
            self.handle = handle
        def next_chunk(self, num_retries):
            assert num_retries == 2
            self.handle.write(b'mel')
            return None, True
    repair.DriveReader(service, Downloader).download('file-id', tmp_path / 'temp')
    assert (tmp_path / 'temp').read_bytes() == b'mel'
    assert downloads == [dict(fileId='file-id', supportsAllDrives=True)]
    root, _, payloads = legacy_cache(tmp_path)
    reader = repair.DriveReader(FakeDrive('wrong-cache', []), None)
    with pytest.raises(ValueError, match='another Mel cache'):
        reader.index('folder-id', root, set(payloads))
    with pytest.raises(ValueError, match='folder ID'):
        reader.index("bad'id", root, set(payloads))


def test_new_extraction_uses_distributed_layout_and_retains_zip(tmp_path):
    from test_translatotron2_prepare import _common, _mel_fixture
    from direct_s2st.translatotron2.phonemize import phonemize_manifests
    from direct_s2st.translatotron2.prepare_fairseq import prepare_fairseq
    common, phones, output = [tmp_path / p for p in ('common', 'phones', 'fairseq')]
    _common(common)
    phonemize_manifests(common, phones, phonemizer=lambda _: 'a',
                       engine='fixture', version='1', fixed_vocabulary=['a'])
    settings = dict(mel_config={'n_mels': 80}, feature_extractor=_mel_fixture, resume=True)
    prepare_fairseq(common, phones, output, **settings)
    root = next((tmp_path / '.prep-checkpoints/mel').iterdir())
    assert not (root / 'features').exists()
    assert len(list((root / 'features-v2').glob('*/*.npy'))) == 3
    before = (output / 'logmelspec80.zip').read_bytes()
    def forbidden(*args):
        pytest.fail('completed Mel must not be regenerated')
    prepare_fairseq(common, phones, output, **(settings | dict(feature_extractor=forbidden)))
    assert (output / 'logmelspec80.zip').read_bytes() == before
