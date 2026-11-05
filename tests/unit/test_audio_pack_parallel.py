from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import threading
import zipfile

import pytest

from direct_s2st import drive_staging as stage
from direct_s2st import audio_pack_runtime as runtime
from direct_s2st.journal import digest
from test_drive_staging import audio_fixture


OPTIONS = dict(splits=('train',), languages=('en',), pack_files=2, materialize=False)


@pytest.mark.parametrize('workers', [1, 2, 4])
def test_parallel_packs_have_identical_layout_bytes_order_and_receipts(tmp_path, workers, capsys):
    common, rows = audio_fixture(tmp_path, 9)
    baseline, packs = tmp_path / 'serial', tmp_path / 'parallel'
    stage.stage_audio(common, baseline, tmp_path / 'vm-1', workers=1, **OPTIONS)
    stage.stage_audio(common, packs, tmp_path / 'vm-2', workers=workers, **OPTIONS)
    assert {p.relative_to(baseline): p.read_bytes() for p in baseline.rglob('*') if p.is_file()} == {
        p.relative_to(packs): p.read_bytes() for p in packs.rglob('*') if p.is_file()}
    for start in range(0, len(rows), 2):
        archive = stage.audio_archive_path(packs, stage.raw_sha(common/'train.jsonl'), 'train', 'en', start, 2)
        document = json.loads(archive.with_suffix('.json').read_text())
        body = document['body']
        assert document['sha256'] == digest(body)
        assert body['archive_sha256'] == stage.raw_sha(archive)
        with zipfile.ZipFile(archive) as zipped:
            selected = rows[start:start+2]
            assert zipped.namelist() == [digest([r['pair_id'], 'en'])[:32]+'.wav' for r in selected]
            for name, row in zip(zipped.namelist(), selected):
                assert zipped.getinfo(name).compress_type == zipfile.ZIP_STORED
                assert hashlib.sha256(zipped.read(name)).hexdigest() == row['en_sha256']
    log = capsys.readouterr().err
    assert f'workers={workers} completed=5/5' in log
    assert 'built=5 reused=0 failed=0' in log
    for key in ('zip_build_seconds=', 'originals_seconds=', 'drive_copy_seconds=', 'readback_seconds=',
                'packs_per_sec=', 'MiB_per_sec=', 'temporary_reserved_peak_bytes='):
        assert key in log


def test_serial_partial_publication_reuses_old_zip_with_four_workers(tmp_path, monkeypatch, capsys):
    common, rows = audio_fixture(tmp_path, 7)
    packs = tmp_path/'packs'
    real = stage.atomic_write_json
    def interrupt(path, *args, **kwargs):
        if Path(path).name == '00000002.json':
            raise KeyboardInterrupt()
        return real(path, *args, **kwargs)
    monkeypatch.setattr(stage, 'atomic_write_json', interrupt)
    with pytest.raises(KeyboardInterrupt):
        stage.stage_audio(common, packs, tmp_path/'old-vm', workers=1, **OPTIONS)
    old = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in packs.rglob('*.zip')}
    assert len(old) == 2  # Second ZIP has a plan but no completion receipt.
    for row in rows[:4]:
        Path(row['en_audio']).unlink()
    monkeypatch.setattr(stage, 'atomic_write_json', real)
    stage.stage_audio(common, packs, tmp_path/'new-vm', workers=4, **OPTIONS)
    assert all((p.read_bytes(), p.stat().st_mtime_ns) == state for p, state in old.items())
    assert len(list(packs.rglob('*.zip'))) == 4
    assert 'built=2 reused=2 failed=0' in capsys.readouterr().err


def test_parallelism_is_real_and_futures_remain_bounded(tmp_path, monkeypatch):
    common, _ = audio_fixture(tmp_path, 21)
    barrier = threading.Barrier(4)
    lock = threading.Lock()
    active = maximum = issued = 0
    original = stage._stage_audio_pack
    def process(*args, **kwargs):
        nonlocal active, maximum, issued
        with lock:
            active += 1
            issued += 1
            maximum = max(maximum, active)
            first = issued <= 4
        try:
            if first:
                barrier.wait(timeout=10)
            return original(*args, **kwargs)
        finally:
            with lock:
                active -= 1
    class BoundedPool(ThreadPoolExecutor):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.outstanding = 0
        def submit(self, *args, **kwargs):
            self.outstanding += 1
            assert self.outstanding <= 4
            f = super().submit(*args, **kwargs)
            def complete(_):
                self.outstanding -= 1
            f.add_done_callback(complete)
            return f
    monkeypatch.setattr(stage, '_stage_audio_pack', process)
    monkeypatch.setattr(runtime, 'ThreadPoolExecutor', BoundedPool)
    stage.stage_audio(common, tmp_path/'packs', tmp_path/'vm', workers=4, **OPTIONS)
    assert maximum == 4 and active == 0


@pytest.mark.parametrize('failure', ['source', 'readback', 'copy'])
def test_failed_workers_propagate_join_and_remove_temporary_files(tmp_path, monkeypatch, failure, capsys):
    common, rows = audio_fixture(tmp_path, 20)
    packs, local = tmp_path/'packs', tmp_path/'vm'
    if failure == 'source':
        Path(rows[0]['en_audio']).write_bytes(b'corrupt source')
        expected = ValueError
    else:
        original = stage.shutil.copyfile
        def copy(source, target, *args, **kwargs):
            if Path(target).name.startswith('.publish-'):
                if failure == 'copy':
                    raise OSError('fixture Drive copy failure')
                result = original(source, target, *args, **kwargs)
                Path(target).write_bytes(b'bad readback')
                return result
            return original(source, target, *args, **kwargs)
        monkeypatch.setattr(stage.shutil, 'copyfile', copy)
        expected = OSError if failure == 'copy' else ValueError
    with pytest.raises(expected):
        stage.stage_audio(common, packs, local, workers=4, **OPTIONS)
    assert not list(local.glob('audio-pack-*'))
    assert not list(packs.rglob('.publish-*'))
    assert not any(t.name.startswith('audio-pack') for t in threading.enumerate())
    for receipt in packs.rglob('*.json'):
        if receipt.name.endswith('.plan.json'):
            continue
        body = json.loads(receipt.read_text())['body']
        assert stage.raw_sha(receipt.with_suffix('.zip')) == body['archive_sha256']
    assert 'status=failed' in capsys.readouterr().err


def test_corrupt_or_uncommitted_archive_is_not_overwritten(tmp_path):
    common, _ = audio_fixture(tmp_path, 3)
    packs = tmp_path/'packs'
    stage.stage_audio(common, packs, tmp_path/'vm', workers=1, **OPTIONS)
    archive = next(packs.rglob('*.zip'))
    archive.write_bytes(b'corrupt')
    with pytest.raises(ValueError):
        stage.stage_audio(common, packs, tmp_path/'vm2', workers=4, **OPTIONS)
    assert archive.read_bytes() == b'corrupt'
    archive.with_suffix('.json').unlink()
    archive.with_suffix('.plan.json').unlink()
    with pytest.raises(RuntimeError, match='Uncommitted'):
        stage.stage_audio(common, packs, tmp_path/'vm3', workers=4, **OPTIONS)
    assert archive.read_bytes() == b'corrupt'


def test_gpu_restore_never_starts_a_pack_executor_or_reads_originals(tmp_path, monkeypatch):
    common, rows = audio_fixture(tmp_path, 5)
    packs = tmp_path/'packs'
    stage.stage_audio(common, packs, tmp_path/'cpu', workers=2, **OPTIONS)
    for row in rows:
        Path(row['en_audio']).unlink()
    monkeypatch.setattr(runtime, 'ThreadPoolExecutor', lambda **kw: pytest.fail('GPU pack executor'))
    monkeypatch.setattr(stage, 'read_common_manifest_row', lambda *a: pytest.fail('GPU original read'))
    options = dict(OPTIONS, materialize=True, create=False, workers=4)
    assert len(stage.stage_audio(common, packs, tmp_path/'gpu', **options)) == 5


@pytest.mark.parametrize('workers', [0, 5, -1, True, 1.5, None])
def test_invalid_workers_are_rejected(tmp_path, workers):
    common, _ = audio_fixture(tmp_path, 1)
    with pytest.raises(ValueError, match='workers'):
        stage.stage_audio(common, tmp_path/'packs', tmp_path/'vm', workers=workers, **OPTIONS)


def test_disk_reservations_include_other_workers_and_are_released(tmp_path):
    calls = []
    budget = runtime.DiskBudget(tmp_path, lambda root, amount: calls.append(amount))
    with budget.reserve(100):
        with budget.reserve(200):
            assert budget.reserved == 300
    assert calls == [100, 300] and budget.peak == 300 and budget.reserved == 0
    with pytest.raises(RuntimeError):
        with budget.reserve(50):
            raise RuntimeError('fixture')
    assert budget.reserved == 0


def test_archive_publisher_lock_prevents_double_writes(tmp_path):
    archive = tmp_path/'pack.zip'
    with runtime.archive_lock(archive):
        with pytest.raises(RuntimeError, match='already active'):
            with runtime.archive_lock(archive):
                pytest.fail('double writer')
    with runtime.archive_lock(archive):
        pass  # No stale lock after normal failure/exit.
