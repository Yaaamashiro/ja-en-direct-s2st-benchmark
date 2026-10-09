import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
import pytest

from direct_s2st import drive_staging as stage
from direct_s2st.hashing import sha256_file
from direct_s2st.preparation import file_stamp
from direct_s2st.io import atomic_write_json
from direct_s2st.journal import digest


def audio_fixture(tmp_path, count=3):
    common = tmp_path / 'common'
    common.mkdir()
    rows = []
    for index in range(count):
        source = tmp_path / f'{index}.wav'
        source.write_bytes(bytes([index]) * 100)
        rows.append(dict(pair_id=f'id-{index}', split='train', en_audio=str(source),
                         en_sha256=sha256_file(source)))
    (common / 'train.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
    return common, rows


def test_wav_packs_reuse_without_original_reads_and_preserve_identity(tmp_path, monkeypatch):
    common, rows = audio_fixture(tmp_path)
    expected = {row['en_audio']: file_stamp(row['en_audio']) for row in rows}
    options = dict(splits=('train',), languages=('en',), pack_files=2)
    persistent, local = tmp_path / 'packs', tmp_path / 'local'
    mapped = stage.stage_audio(common, persistent, local, **options)
    assert len(list(persistent.rglob('*.zip'))) == 2
    for row in rows:
        Path(row['en_audio']).rename(Path(row['en_audio']).with_suffix('.retained'))
    # Simulate a new VM: only persisted ZIPs remain accessible.
    mapped = stage.stage_audio(common, persistent, tmp_path / 'new-vm', **options)
    with stage.active_map(mapped, tmp_path / 'new-vm'):
        for row in rows:
            assert sha256_file(Path(row['en_audio'])) == row['en_sha256']
            assert file_stamp(row['en_audio']) == expected[row['en_audio']]
            from direct_s2st.train_runtime import cached_file
            with cached_file(row['en_audio']) as path:
                assert path == stage.local_path(row['en_audio'])
        bad = stage.local_path(rows[0]['en_audio'])
        bad.write_bytes(b'changed')
        with pytest.raises(ValueError, match='changed'):
            stage.local_path(rows[0]['en_audio'])


def test_audio_publication_interruption_adopts_verified_orphan(tmp_path, monkeypatch):
    common, rows = audio_fixture(tmp_path, 1)
    persistent, local = tmp_path / 'packs', tmp_path / 'local'
    real = stage.atomic_write_json
    def interrupt(path, *args, **kwargs):
        if Path(path).parent.is_relative_to(persistent) and Path(path).suffix == '.json' and not str(path).endswith('.plan.json'):
            raise KeyboardInterrupt()
        return real(path, *args, **kwargs)
    monkeypatch.setattr(stage, 'atomic_write_json', interrupt)
    options = dict(splits=('train',), languages=('en',))
    with pytest.raises(KeyboardInterrupt):
        stage.stage_audio(common, persistent, local, **options)
    original = next(persistent.rglob('*.zip'))
    before = original.read_bytes()
    Path(rows[0]['en_audio']).unlink()
    monkeypatch.setattr(stage, 'atomic_write_json', real)
    assert len(stage.stage_audio(common, persistent, local, **options)) == 1
    assert original.read_bytes() == before


def test_wrong_wav_hash_never_commits_and_disk_failure_is_explicit(tmp_path, monkeypatch):
    common, rows = audio_fixture(tmp_path, 1)
    row = rows[0] | dict(en_sha256='0' * 64)
    (common / 'train.jsonl').write_text(json.dumps(row) + '\n')
    with pytest.raises(ValueError, match='checksum'):
        stage.stage_audio(common, tmp_path/'packs', tmp_path/'local', splits=('train',), languages=('en',))
    assert not list((tmp_path/'packs').rglob('*.zip'))
    monkeypatch.setattr(stage.shutil, 'disk_usage', lambda _: type('Disk', (), {'free': 0})())
    with pytest.raises(RuntimeError, match='insufficient'):
        stage.copy_verified(Path(rows[0]['en_audio']), tmp_path/'local/copy')


def test_staged_mel_reads_correct_offset_without_remote_open(tmp_path):
    from direct_s2st.translatotron2.data import load_mel
    source, copied = tmp_path/'data/mel.zip', tmp_path/'local/mel.zip'
    source.parent.mkdir()
    copied.parent.mkdir()
    array = tmp_path/'mel.npy'
    np.save(array, np.ones((3, 80), dtype=np.float32))
    with zipfile.ZipFile(source, 'w', zipfile.ZIP_STORED) as pack:
        pack.write(array, 'id.npy')
        info = pack.getinfo('id.npy')
        offset, size = info.header_offset + 30 + len('id.npy'), info.file_size
    stage.copy_verified(source, copied)
    rows = {}
    stage.register(rows, source, copied, file_stamp(source))
    source.unlink()
    with stage.active_map(rows, copied.parent):
        assert load_mel(source.parent, f'mel.zip:{offset}:{size}').shape == (3, 80)


def test_stage_rejects_drive_or_corpus_destination(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match='VM'):
        stage.ensure_local('/content/drive/MyDrive/staged')
    monkeypatch.setenv('CORPUS_ROOT', str(tmp_path))
    with pytest.raises(ValueError, match='CORPUS_ROOT'):
        stage.stage_audio(tmp_path/'common', tmp_path/'packs', tmp_path/'local')


def test_cpu_publish_does_not_download_or_materialize_and_gpu_uses_old_layout(tmp_path, monkeypatch, capsys):
    common, rows = audio_fixture(tmp_path)
    persistent = tmp_path / 'packs'
    options = dict(splits=('train',), languages=('en',), pack_files=2)
    copy = stage.copy_verified
    monkeypatch.setattr(stage, 'copy_verified', lambda *a, **k: pytest.fail('CPU downloaded its own ZIP'))
    assert stage.stage_audio(common, persistent, tmp_path/'cpu', materialize=False, **options) == {}
    assert not list((tmp_path/'cpu').rglob('*.wav'))
    assert not list((tmp_path/'cpu').rglob('complete.json'))
    assert 'zip=2/2 audio=3/3 built=2 reused=0' in capsys.readouterr().err
    from direct_s2st.journal import digest
    namespace = digest([stage.raw_sha(common/'train.jsonl'), 'train', 'en', 2])[:24]
    assert stage.audio_archive_path(persistent, stage.raw_sha(common/'train.jsonl'), 'train', 'en', 0, 2) == (
        persistent / namespace[:2] / namespace / digest(0)[:2] / '00000000.zip')
    monkeypatch.setattr(stage, 'copy_verified', copy)
    for row in rows:
        Path(row['en_audio']).unlink()  # VM switch: corpus originals inaccessible.
    assert stage.require_audio_packs(common, persistent, **options) == 2
    assert len(stage.stage_audio(common, persistent, tmp_path/'gpu-vm', create=False, **options)) == 3


def test_interrupted_cpu_zip_publish_resumes_without_rereading_published_sources(tmp_path, monkeypatch):
    common, rows = audio_fixture(tmp_path)
    persistent = tmp_path/'packs'
    real = stage.atomic_write_json
    calls = []
    def interrupt(path, *args, **kwargs):
        if Path(path).parent.is_relative_to(persistent) and str(path).endswith('.json') and not str(path).endswith('.plan.json'):
            calls.append(path)
            if len(calls) == 2:
                raise KeyboardInterrupt()  # second ZIP published, receipt not yet.
        return real(path, *args, **kwargs)
    monkeypatch.setattr(stage, 'atomic_write_json', interrupt)
    options = dict(splits=('train',), languages=('en',), pack_files=1, materialize=False)
    with pytest.raises(KeyboardInterrupt):
        stage.stage_audio(common, persistent, tmp_path/'cpu-1', **options)
    before = {p: p.read_bytes() for p in persistent.rglob('*.zip')}
    assert len(before) == 2
    for row in rows[:2]:
        Path(row['en_audio']).unlink()
    monkeypatch.setattr(stage, 'atomic_write_json', real)
    stage.stage_audio(common, persistent, tmp_path/'cpu-2', **options)
    assert len(list(persistent.rglob('*.zip'))) == 3
    assert all(p.read_bytes() == content for p, content in before.items())
    assert stage.require_audio_packs(common, persistent, splits=('train',), languages=('en',), pack_files=1) == 3


def test_gpu_missing_packs_fail_before_any_copy_or_original_read(tmp_path, monkeypatch):
    common, rows = audio_fixture(tmp_path)
    persistent = tmp_path/'packs'
    options = dict(splits=('train',), languages=('en',), pack_files=2)
    stage.stage_audio(common, persistent, tmp_path/'cpu', materialize=False, **options)
    last = stage.audio_archive_path(persistent, stage.raw_sha(common/'train.jsonl'), 'train', 'en', 2, 2)
    last.with_suffix('.json').unlink()
    monkeypatch.setattr(stage, 'copy_verified', lambda *a, **k: pytest.fail('download before preflight'))
    monkeypatch.setattr(stage, 'read_common_manifest_row', lambda *a: pytest.fail('read originals on GPU'))
    with pytest.raises(RuntimeError, match='CPU audio packing incomplete'):
        stage.stage_audio(common, persistent, tmp_path/'gpu', create=False, **options)
    assert not list((tmp_path/'gpu').rglob('*.wav'))


def test_inline_units_resume_and_migrate_without_reextracting(tmp_path):
    from test_s2ut_units import _common
    from direct_s2st.s2ut.extract_units import extract_units
    from direct_s2st.s2ut.unit_storage import records, sequence
    common, root = tmp_path/'common', tmp_path/'units'
    _common(common)
    options = dict(split=None, clusters=100, hubert_model='fixture', hubert_revision='a'*40,
                   hubert_layer=6, kmeans_sha256='b'*64, resume=True)
    extract_units(common, root, extractor=lambda _: [1, 1, 2], storage='files', **options)
    from direct_s2st.s2ut.prepare_fairseq import prepare_fairseq
    prepared = tmp_path / 'prepared'
    prepare_fairseq(common, root, prepared)
    old_lock = (prepared / 'data-lock.json').read_bytes()
    legacy = {p: p.read_bytes() for p in root.rglob('*.units')}
    extract_units(common, root, extractor=lambda _: pytest.fail('reextracted'), **options)
    assert legacy == {p: p.read_bytes() for p in root.rglob('*.units')}
    record = records(root)['pair-train']
    assert sequence(record, 'original', 100) == [1, 1, 2]
    assert sequence(record, 'reduced', 100) == [1, 2]
    prepare_fairseq(common, root, prepared, resume=True)
    assert (prepared / 'data-lock.json').read_bytes() == old_lock
    assert (prepared / 'unit-storage-migration.json').is_file()
    record['units_original'] = [8]
    with pytest.raises(ValueError, match='corrupt'):
        sequence(record, 'original', 100)


def test_inline_vocoder_verification_same_hash_as_legacy(tmp_path):
    from test_s2ut_units import _common
    from direct_s2st.s2ut.extract_units import extract_units, serialize_units
    from direct_s2st.vocoders.verification import verify_inputs
    common, root = tmp_path/'common', tmp_path/'units'
    _common(common)
    path = common/'train.jsonl'
    row = json.loads(path.read_text())
    row['en_sha256'] = sha256_file(Path(row['en_audio']))
    path.write_text(json.dumps(row) + '\n')
    extract_units(common, root, extractor=lambda _: [1, 1, 2], split=None, clusters=100,
                  hubert_model='fixture', hubert_revision='a'*40, hubert_layer=6, kmeans_sha256='b'*64)
    result = verify_inputs(common, 'unit', root)
    assert result['sequences']['pair-train'] == [1, 1, 2]
    assert result['units']['pair-train'] == hashlib.sha256(serialize_units([1, 1, 2]).encode()).hexdigest()


def test_distributed_checkpoint_chunks_reuse_legacy_chunks(tmp_path, monkeypatch):
    from direct_s2st.preparation import Checkpoints
    with Checkpoints(tmp_path, {'stage': 'fixture'}) as cache:
        cache.record('old', [1])
    monkeypatch.setenv('S2ST_DRIVE_SAFE', '1')
    with Checkpoints(tmp_path, {'stage': 'fixture'}, resume=True) as cache:
        cache.record('new', [2])
    assert len(list(cache.root.glob('*/chunk-*.json'))) == 1
    with Checkpoints(tmp_path, {'stage': 'fixture'}, resume=True) as cache:
        assert cache.rows == {'old': [1], 'new': [2]}


def test_packed_snapshots_restore_and_fallback_and_capacity_budget(tmp_path, monkeypatch):
    from direct_s2st.colab import publish, latest, restore
    work, backup = tmp_path/'work', tmp_path/'backup'
    work.mkdir()
    (work/'checkpoint.pt').write_text('trusted fixture')
    monkeypatch.setenv('S2ST_DRIVE_SAFE', '1')
    identity = {'fixture': True}
    first = publish(work, backup, identity, 1)
    second = publish(work, backup, identity, 2)
    (second/'snapshot.zip').write_bytes(b'corrupt')
    chosen = latest(backup, identity)
    assert chosen[0] == first
    restore(chosen, tmp_path/'restored')
    assert (tmp_path/'restored/checkpoint.pt').read_text() == 'trusted fixture'
    monkeypatch.setenv('S2ST_BACKUP_MAX_GB', '0.000000001')
    with pytest.raises(RuntimeError, match='budget exceeded'):
        publish(work, backup, identity, 3)
    assert latest(backup, identity)[0] == first


def test_backup_interval_defers_upload_but_final_checkpoint_is_forced(tmp_path, monkeypatch):
    from direct_s2st import train_runtime as runtime
    work, staging = tmp_path/'work', tmp_path/'staging'
    work.mkdir()
    staging.mkdir()
    checkpoint = work/'checkpoint.pt'
    monkeypatch.setenv('S2ST_TRAIN_STAGING', str(staging))
    monkeypatch.setenv('S2ST_BACKUP_MIN_SECONDS', '600')
    now = [0]
    monkeypatch.setattr(runtime.time, 'monotonic', lambda: now[0])
    checkpoint.write_text('1')
    runtime.checkpoint_saved(work, checkpoint, 1)
    now[0] = 50
    checkpoint.write_text('2')
    runtime.checkpoint_saved(work, checkpoint, 2)
    assert len(list(staging.glob('*/ready.json'))) == 1
    runtime.checkpoint_saved(work, checkpoint, 2, force=True)
    assert len(list(staging.glob('*/ready.json'))) == 2


def test_training_stages_whole_inputs_before_runner_and_preserves_resume(tmp_path, monkeypatch):
    import subprocess
    from direct_s2st.colab import run_session
    from direct_s2st.train_runtime import cached_file
    from test_s2ut_units import _common
    common = tmp_path / 'data/common'
    prepared = tmp_path / 'data/translatotron2/fairseq'
    _common(common)
    for split in ('train', 'dev', 'test'):
        manifest = common / f'{split}.jsonl'
        row = json.loads(manifest.read_text())
        for language in ('ja', 'en'):
            row[f'{language}_sha256'] = sha256_file(Path(row[f'{language}_audio']))
        manifest.write_text(json.dumps(row) + '\n')
    prepared.mkdir(parents=True)
    lock = prepared / 'data-lock.json'
    lock.write_text('{"fixture":true}')
    mel = prepared / 'mel.zip'
    with zipfile.ZipFile(mel, 'w') as archive:
        archive.writestr('fixture', 'payload')
    config = dict(kind='tt2', checkpoint='checkpoint.pt', identity_files=[str(lock), str(mel)],
                  command=['fixture', '{run_root}', '{updates}'], resume_args=['--resume'])
    monkeypatch.setenv('S2ST_DRIVE_SAFE', '1')
    calls = []
    real_open = Path.open
    def no_training_remote_wavs(path, *args, **kwargs):
        if path.parent == common and path.suffix == '.wav' and stage.mapping():
            pytest.fail('original WAV opened during training')
        return real_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', no_training_remote_wavs)
    def runner(args, **kwargs):
        calls.append(args)
        assert stage.local_path(mel) != mel
        assert sha256_file(mel) == stage.raw_sha(stage.local_path(mel))
        with cached_file(common/'train-ja.wav') as local:
            assert local.parent != common and local.is_file()
        (Path(args[1]) / 'checkpoint.pt').write_text(args[2])
        return subprocess.CompletedProcess(args, 0)
    kwargs = dict(work=tmp_path/'work', backup=tmp_path/'backup', runner=runner,
                  inspect_checkpoint=lambda p, k: int(p.read_text()))
    assert run_session(config, total=1, **kwargs)['durable_updates'] == 1
    assert run_session(config, total=2, resume=True, **kwargs)['durable_updates'] == 2
    assert '--resume' in calls[-1]


def test_strict_training_never_silently_bypasses_unstaged_drive(monkeypatch):
    from direct_s2st.train_runtime import cached_file
    monkeypatch.setenv('S2ST_TRAIN_STRICT_LOCAL', '1')
    with pytest.raises(RuntimeError, match='not staged'):
        with cached_file('/content/drive/MyDrive/missing.wav'):
            pytest.fail('remote fallback')
