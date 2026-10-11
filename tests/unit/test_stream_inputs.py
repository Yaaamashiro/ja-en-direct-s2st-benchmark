"""CPU coverage of lazy verified reads; fixtures are not benchmark speech."""
from concurrent.futures import ThreadPoolExecutor
import io
import json
import os
import subprocess
import sys
import threading
from pathlib import Path
import zipfile

import numpy as np
import pytest

from direct_s2st import drive_staging as stage, stream_inputs as stream
from direct_s2st.hashing import sha256_file
from direct_s2st.journal import digest
from direct_s2st.preparation import file_stamp


def fixture(tmp_path, train_count=4):
    common = tmp_path/'data/common'
    prepared = tmp_path/'data/translatotron2/fairseq'
    common.mkdir(parents=True)
    prepared.mkdir(parents=True)
    (common/'dataset-lock.json').write_text('{"fixture":true}')
    sources, mel_hashes, tables = {}, {}, {}
    (prepared/'target_phoneme').mkdir()
    (prepared/'target_phoneme/dict.txt').write_text('a 10\nb 10\n')
    (prepared/'mel-spec.json').write_text('{"n_mels":80}')
    for split, count in [('train', train_count), ('dev', 1), ('test', 1)]:
        manifest, table, phones = [], [], []
        for index in range(count):
            name = f'{split}-{index}'
            source = common/f'{name}.wav'
            source.write_bytes(name.encode()*10)
            sources[name] = source
            manifest.append(dict(pair_id=name, split=split, ja_audio=str(source),
                                 ja_sha256=stage.raw_sha(source)))
            mel = prepared/f'{name}.zip'
            values = io.BytesIO()
            np.save(values, np.full((6, 80), index/100, dtype=np.float32))
            with zipfile.ZipFile(mel, 'w', zipfile.ZIP_STORED) as archive:
                archive.writestr(name+'.npy', values.getvalue())
                info = archive.getinfo(name+'.npy')
                offset = info.header_offset+30+len(name+'.npy')
            mel_hashes[mel.name] = stage.raw_sha(mel)
            table.append(dict(id=name, src_audio=str(source), tgt_audio=f'{mel.name}:{offset}:{info.file_size}',
                              tgt_n_frames=6, src_n_frames=12))
            phones.append(f'{name}\ta b\n')
        (common/f'{split}.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in manifest))
        fields = list(table[0])
        (prepared/f'{split}.tsv').write_text('\t'.join(fields)+'\n'+''.join(
            '\t'.join(str(row[f]) for f in fields)+'\n' for row in table))
        (prepared/f'target_phoneme/{split}.tsv').write_text('id\ttgt_text\n'+''.join(phones))
        tables[split] = table
    (prepared/'data-lock.json').write_text(json.dumps(dict(
        common_dataset_lock_sha256=stage.raw_sha(common/'dataset-lock.json'), mel_shards_sha256=mel_hashes)))
    stage.stage_audio(common, common.parent/'.drive-audio-packs', tmp_path/'cpu',
                      languages=('ja',), materialize=False)
    config = dict(kind='tt2', checkpoint='checkpoints/checkpoint_last.pt',
                  identity_files=[str(p) for p in sorted(prepared.rglob('*')) if p.is_file()],
                  command=['fixture', '{run_root}', '{updates}'], resume_args=['--restore-file', '{checkpoint}'])
    return config, prepared, sources, tables


def test_plan_and_fingerprint_never_download_payloads_then_verified_demand(tmp_path, monkeypatch):
    from direct_s2st.translatotron2.data import fingerprint, load_mel
    config, root, sources, tables = fixture(tmp_path)
    expected = fingerprint(root)
    original_stamp = file_stamp(sources['train-0'])
    copy = stage.copy_verified
    calls = []
    def record(source, target, *args, **kwargs):
        calls.append(Path(source))
        return copy(source, target, *args, **kwargs)
    monkeypatch.setattr(stage, 'copy_verified', record)
    with stream.streaming_map(config, tmp_path/'work'):
        assert fingerprint(root) == expected
        assert all(p.suffix != '.zip' for p in calls)
        assert not list(stream.downloader().root.rglob('*.zip'))
        for source in sources.values():
            source.unlink()  # Original corpus is not read on the new VM.
        assert stage.local_path(sources['train-0']).read_bytes() == b'train-0'*10
        assert file_stamp(sources['train-0']) == original_stamp
        assert load_mel(root, tables['train'][0]['tgt_audio']).shape == (6, 80)
        assert len([p for p in calls if p.suffix == '.zip']) == 2
        cached = stage.local_path(sources['train-0'])
        cached.write_bytes(b'changed')
        with pytest.raises(ValueError, match='changed'):
            stage.local_path(sources['train-0'])
    assert not stage.mapping()


def test_interrupted_download_not_published_and_retry_reuses_complete_local_shards(tmp_path, monkeypatch):
    config, root, _, _ = fixture(tmp_path)
    first, second = root/'train-0.zip', root/'train-1.zip'
    copy = stage.copy_verified
    def fail(source, target, *args, **kwargs):
        if Path(source) == second:
            raise OSError('simulated disconnect')
        return copy(source, target, *args, **kwargs)
    monkeypatch.setattr(stage, 'copy_verified', fail)
    with stream.streaming_map(config, tmp_path/'work'):
        stage.local_path(first)
        with pytest.raises(OSError, match='disconnect'):
            stage.local_path(second)
        assert len(list(stream.downloader().root.rglob('ready.json'))) == 1
    calls = []
    def retry(source, target, *args, **kwargs):
        calls.append(Path(source))
        return copy(source, target, *args, **kwargs)
    monkeypatch.setattr(stage, 'copy_verified', retry)
    with stream.streaming_map(config, tmp_path/'work'):
        stage.local_path(first)
        stage.local_path(second)
    assert first not in calls and second in calls


def test_wrong_mel_bytes_never_exposed_even_if_locked_identity_was_read(tmp_path):
    config, root, _, _ = fixture(tmp_path)
    path = root/'train-0.zip'
    expected = stage.raw_sha(path)
    path.write_bytes(b'corrupt payload')
    with stream.streaming_map(config, tmp_path/'work'):
        assert sha256_file(path) == expected  # Identity only, not a byte-verification result.
        with pytest.raises(ValueError, match='checksum'):
            stage.local_path(path)
        assert not list(stream.downloader().root.rglob('ready.json'))


def test_wrong_audio_receipt_fails_before_exposing_original_or_local_fallback(tmp_path):
    config, _, sources, _ = fixture(tmp_path)
    with stream.streaming_map(config, tmp_path/'work'):
        manager = stream.downloader()
        job = manager.jobs[stage.mapping()[stage.logical(sources['train-0'])]['job']]
        receipt = Path(job['source']).with_suffix('.json')
        saved = json.loads(receipt.read_text())
        saved['body']['expected']['train-0'] = '0'*64
        saved['sha256'] = digest(saved['body'])
        receipt.write_text(json.dumps(saved))
        with pytest.raises(ValueError, match='common manifest'):
            stage.local_path(sources['train-0'])


def test_background_download_overlaps_real_cpu_update_and_preserves_order(tmp_path, monkeypatch):
    import torch
    from direct_s2st.translatotron2.data import PreparedDataset
    from direct_s2st.translatotron2.engine import optimization_step
    from direct_s2st.translatotron2.model import ModelConfig, Translatotron2
    from direct_s2st.translatotron2.batching import collate
    config, root, _, _ = fixture(tmp_path)
    torch.set_num_threads(1)
    started, release = threading.Event(), threading.Event()
    copy = stage.copy_verified
    def delayed(source, target, *args, **kwargs):
        if Path(source) == root/'train-1.zip':
            started.set()
            assert release.wait(15), 'background test download did not release'
        return copy(source, target, *args, **kwargs)
    monkeypatch.setattr(stage, 'copy_verified', delayed)
    # Mock only the audio frontend. Real Mel loading, model, backward and Adam.
    monkeypatch.setattr('direct_s2st.translatotron2.data.source_features',
                        lambda *args: torch.ones(12, 80))
    with stream.streaming_map(config, tmp_path/'work'):
        dataset = PreparedDataset(root)
        model = Translatotron2(ModelConfig.smoke(), len(dataset.tokens))
        opt = torch.optim.Adam(model.parameters())
        stream.prefetch_dataset(dataset, 0, len(dataset))
        try:
            assert started.wait(5)
            before = model.embedding.weight.detach().clone()
            optimization_step(model, opt, collate([dataset[0]]))
            assert not torch.equal(before, model.embedding.weight)
            assert not release.is_set()  # Optimizer advanced during another shard's transfer.
        finally:
            release.set()
        assert [float(dataset[i]['target'][0, 0, 0]) for i in range(len(dataset))] == pytest.approx([0, .01, .02, .03])


def test_prefetch_has_bounded_queue_and_two_transfers(tmp_path, monkeypatch):
    config, _, _, _ = fixture(tmp_path, train_count=20)
    rows, local, document = stream.plan(config, tmp_path/'work')
    manager = stream.Downloader(document)
    release, started = threading.Event(), threading.Event()
    lock = threading.Lock()
    active, maximum = [0], [0]
    def fake_fetch(*args):
        with lock:
            active[0] += 1
            maximum[0] = max(maximum[0], active[0])
            if active[0] == 2:
                started.set()
        assert release.wait(5)
        with lock:
            active[0] -= 1
        return {}, False
    monkeypatch.setattr(manager, 'fetch', fake_fetch)
    try:
        manager.prefetch(list(manager.jobs))
        assert started.wait(3)
        assert len(manager.pending) == 16
        assert maximum[0] == 2
    finally:
        manager.close()
        assert sum(future.cancelled() for future in manager.pending.values()) >= 14
        release.set()
        manager.pool.shutdown(wait=True)


def test_concurrent_requests_share_manager_and_one_payload_transfer(tmp_path, monkeypatch):
    config, root, _, _ = fixture(tmp_path)
    copy, calls = stage.copy_verified, []
    with stream.streaming_map(config, tmp_path/'work'):
        def record(source, target, *args, **kwargs):
            calls.append(Path(source))
            return copy(source, target, *args, **kwargs)
        monkeypatch.setattr(stage, 'copy_verified', record)
        with ThreadPoolExecutor(8) as pool:
            targets = list(pool.map(stage.local_path, [root/'train-0.zip']*8))
        assert len(set(targets)) == 1
        assert calls == [root/'train-0.zip']


def test_session_can_resume_same_identity_between_full_and_lazy_staging(tmp_path, monkeypatch):
    from direct_s2st.colab import run_session
    config, root, _, _ = fixture(tmp_path)
    monkeypatch.setenv('S2ST_DRIVE_SAFE', '1')
    commands = []
    def runner(args, **kwargs):
        commands.append(args)
        # Only the required current sample; no other payload demand at startup.
        stage.local_path(root/'train-0.zip')
        (Path(args[1])/'checkpoints').mkdir(exist_ok=True)
        (Path(args[1])/config['checkpoint']).write_text(args[2])
        if len(commands) == 1:
            assert len(stream.downloader().ready) == 1
        return subprocess.CompletedProcess(args, 0)
    options = dict(work=tmp_path/'work', backup=tmp_path/'backup', runner=runner,
                   inspect_checkpoint=lambda p, k: int(p.read_text()))
    monkeypatch.setenv('S2ST_STREAM_INPUTS', '1')
    assert run_session(config, total=1, **options)['durable_updates'] == 1
    monkeypatch.setenv('S2ST_STREAM_INPUTS', '0')
    assert run_session(config, total=2, resume=True, **options)['durable_updates'] == 2
    monkeypatch.setenv('S2ST_STREAM_INPUTS', '1')
    assert run_session(config, total=3, resume=True, **options)['durable_updates'] == 3
    assert all('--restore-file' in command for command in commands[1:])


def test_corrupt_stream_index_and_unsafe_target_rejected(tmp_path):
    config, _, _, _ = fixture(tmp_path)
    rows, local, document = stream.plan(config, tmp_path/'work')
    bad = dict(document, sha256='0'*64)
    with pytest.raises(ValueError, match='corrupt'):
        stream.Downloader(bad)
    next(iter(document['body']['jobs'].values()))['target'] = str(tmp_path/'outside')
    document['sha256'] = digest(document['body'])
    with pytest.raises(ValueError, match='escapes'):
        stream.Downloader(document)


def test_missing_legacy_hash_and_changed_common_lock_are_explicit(tmp_path):
    config, root, _, _ = fixture(tmp_path)
    lock_path = root/'data-lock.json'
    lock = json.loads(lock_path.read_text())
    lock.pop('mel_shards_sha256')
    lock_path.write_text(json.dumps(lock))
    with pytest.raises(ValueError, match='missing'):
        stream.plan(config, tmp_path/'work')
    (tmp_path/'data/common/dataset-lock.json').write_text('changed')
    with pytest.raises(ValueError, match='common dataset lock'):
        stream.plan(config, tmp_path/'work')


def test_actual_trainer_resumes_lazy_with_identical_weights_optimizer_and_rng(tmp_path, monkeypatch):
    import torch
    from direct_s2st.translatotron2 import train
    config, root, _, _ = fixture(tmp_path)
    torch.set_num_threads(1)
    def frontend(path, spec):
        stage.local_path(path)  # Demand verified audio even with the synthetic frontend.
        return torch.ones(12, 80)
    monkeypatch.setattr('direct_s2st.translatotron2.data.source_features', frontend)
    def run(destination, total, restore=None):
        arguments = ['tt2-train', '--data-root', str(root), '--run-root', str(destination),
                     '--max-updates', str(total), '--num-workers', '2']
        if restore:
            arguments += ['--restore-file', str(restore)]
        monkeypatch.setattr('sys.argv', arguments)
        train.main()
    baseline, resumed = tmp_path/'baseline', tmp_path/'resumed'
    run(baseline, 3)
    run(resumed, 1)
    checkpoint = resumed/'checkpoints/checkpoint_last.pt'
    with stream.streaming_map(config, resumed):
        manager = stream.downloader()
        run(resumed, 3, checkpoint)
        assert len(list(manager.root.rglob('ready.json'))) < len(manager.jobs)
    expected = torch.load(baseline/'checkpoints/checkpoint_last.pt', weights_only=False)
    actual = torch.load(checkpoint, weights_only=False)
    assert expected['updates'] == actual['updates'] == 3
    assert expected['data_fingerprint'] == actual['data_fingerprint']
    for key, tensor in expected['model'].items():
        torch.testing.assert_close(tensor, actual['model'][key], atol=0, rtol=0)
    for parameter, values in expected['optimizer']['state'].items():
        for key, tensor in values.items():
            torch.testing.assert_close(tensor, actual['optimizer']['state'][parameter][key], atol=0, rtol=0)
    torch.testing.assert_close(expected['rng'], actual['rng'], atol=0, rtol=0)


def test_child_process_inherits_lazy_index_and_reuses_verified_local_payload(tmp_path):
    config, root, _, _ = fixture(tmp_path)
    path = root/'train-0.zip'
    with stream.streaming_map(config, tmp_path/'work'):
        target = stage.local_path(path)
        code = ('import sys; from direct_s2st.drive_staging import local_path, raw_sha; '
                'p=local_path(sys.argv[1]); print(raw_sha(p))')
        environment = dict(os.environ)
        environment['PYTHONPATH'] = str(Path(__file__).resolve().parents[2]/'src')
        result = subprocess.run([sys.executable, '-c', code, str(path)],
                                env=environment, capture_output=True, text=True, timeout=30)
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == stage.raw_sha(target)
        assert 'reused=True' in result.stderr


def test_changed_local_payload_is_detected_when_new_process_reuses_receipt(tmp_path):
    config, root, _, _ = fixture(tmp_path)
    path = root/'train-0.zip'
    with stream.streaming_map(config, tmp_path/'work'):
        target = stage.local_path(path)
        original = target.stat()
        content = bytearray(target.read_bytes())
        content[-1] ^= 1
        target.write_bytes(content)
        os.utime(target, ns=(original.st_atime_ns, original.st_mtime_ns))
    with stream.streaming_map(config, tmp_path/'work'):
        with pytest.raises(ValueError, match='changed'):
            stage.local_path(path)


def test_missing_audio_pack_fails_on_demand_without_original_fallback(tmp_path):
    config, _, sources, _ = fixture(tmp_path)
    with stream.streaming_map(config, tmp_path/'work'):
        manager = stream.downloader()
        job = manager.jobs[stage.mapping()[stage.logical(sources['train-0'])]['job']]
        Path(job['source']).with_suffix('.json').unlink()
        with pytest.raises(RuntimeError, match='CPU audio packing incomplete'):
            stage.local_path(sources['train-0'])


def test_single_legacy_mel_archive_uses_saved_hash_and_checks_entire_payload(tmp_path):
    config, root, _, _ = fixture(tmp_path)
    path = root/'train-0.zip'
    config['identity_files'] = [p for p in config['identity_files'] if not p.endswith('.zip') or Path(p) == path]
    lock_path = root/'data-lock.json'
    lock = json.loads(lock_path.read_text())
    lock.pop('mel_shards_sha256')
    lock['mel_zip_sha256'] = stage.raw_sha(path)
    lock_path.write_text(json.dumps(lock))
    with stream.streaming_map(config, tmp_path/'work'):
        assert sha256_file(path) == lock['mel_zip_sha256']
        assert stream.downloader().completed == 0
        assert stage.raw_sha(stage.local_path(path)) == lock['mel_zip_sha256']


def test_dev_validation_only_prefetches_dev_and_keeps_weights_and_rng(tmp_path, monkeypatch):
    import torch
    from direct_s2st.translatotron2.data import PreparedDataset
    from direct_s2st.translatotron2.validation import validate
    from direct_s2st.translatotron2.model import ModelConfig, Translatotron2
    config, root, _, _ = fixture(tmp_path)
    torch.set_num_threads(1)
    def frontend(path, spec):
        stage.local_path(path)
        return torch.ones(12, 80)
    monkeypatch.setattr('direct_s2st.translatotron2.data.source_features', frontend)
    with stream.streaming_map(config, tmp_path/'work'):
        model = Translatotron2(ModelConfig.smoke(), 5)
        before = {key: value.clone() for key, value in model.state_dict().items()}
        rng = torch.get_rng_state()
        result = validate(model, PreparedDataset(root, 'dev'), 'cpu')
        assert np.isfinite(result['loss']) and model.training
        torch.testing.assert_close(rng, torch.get_rng_state(), atol=0, rtol=0)
        for key, tensor in before.items():
            torch.testing.assert_close(tensor, model.state_dict()[key], atol=0, rtol=0)
        manager = stream.downloader()
        assert len(manager.ready) == 2  # Only dev's WAV pack and Mel archive.
        assert all('dev' in job['source'] or job['kind'] == 'audio'
                   for key, job in manager.jobs.items() if key in manager.ready)
