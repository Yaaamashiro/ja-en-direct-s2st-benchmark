import json
from pathlib import Path
import sys
import wave
import zipfile

import pytest

from direct_s2st.benchmark_storage import publish_tree, restore_tree, read
from direct_s2st.drive_staging import raw_sha
from direct_s2st.io import atomic_write_json, atomic_write_jsonl, read_jsonl
from direct_s2st.journal import digest
from direct_s2st.s2ut import benchmark_worker as worker
from direct_s2st.vocoders.reference import UNIT_CONFIGURATION

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'scripts/colab'))
import benchmark_session as session


def wav(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), 'wb') as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16000)
        stream.writeframes(b'\x01\x00'*1600)


def test_audio_archive_replays_one_member_and_rejects_changes(tmp_path):
    source, drive, restored = tmp_path/'s', tmp_path/'d', tmp_path/'r'
    wav(source/'audio/00000000.wav')
    wav(source/'audio/00000001.wav')
    receipt = publish_tree(source, drive, 'part-0', {'run':'x'})
    restore_tree(drive, 'part-0', restored, {'run':'x'}, members=['audio/00000001.wav'])
    assert not (restored/'audio/00000000.wav').exists()
    assert raw_sha(restored/'audio/00000001.wav') == raw_sha(source/'audio/00000001.wav')
    with pytest.raises(ValueError, match='identity'):
        restore_tree(drive, 'part-0', restored, {'run':'y'})
    (restored/'audio/00000001.wav').write_bytes(b'changed')
    with pytest.raises(ValueError, match='different local'):
        restore_tree(drive, 'part-0', restored, {'run':'x'})
    archive = drive/receipt['body']['archive']
    archive.write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='checksum'):
        restore_tree(drive, 'part-0', tmp_path/'another', {'run':'x'})


@pytest.mark.parametrize('name', ['../escape.wav', '/escape.wav', 'C:/escape.wav', 'audio\\escape.wav'])
def test_malicious_archive_never_extracts(tmp_path, name):
    drive = tmp_path/'d'
    drive.mkdir()
    archive = drive/'part.zip'
    with zipfile.ZipFile(archive, 'w') as stream:
        stream.writestr(name, b'a')
    body = dict(version=1, identity={}, archive=archive.name, sha256=raw_sha(archive),
                files={name:dict(sha256='a'*64, size=1)})
    atomic_write_json(drive/'part.json', dict(body=body, sha256=digest(body)))
    with pytest.raises(ValueError, match='unsafe'):
        restore_tree(drive, 'part', tmp_path/'r', {})
    assert not (tmp_path/'escape.wav').exists()


def benchmark_fixture(tmp_path, monkeypatch):
    from test_listening_preview import fixture_run
    persistent, data, local = tmp_path/'d', tmp_path/'d/data', tmp_path/'l'
    directory = data/'s2ut/fairseq'
    directory.mkdir(parents=True)
    rows = []
    for index in range(2):
        source = tmp_path/'corpus'/f'{index}.wav'
        wav(source)
        rows.append(dict(pair_id=f'test_{index}', split='test', ja_audio=str(source), en_audio=str(source),
                         ja_sha256=raw_sha(source), en_sha256=raw_sha(source),
                         ja_text='日本語', en_text='the book is new'))
    atomic_write_jsonl(data/'common/test.jsonl', rows)
    atomic_write_json(data/'common/dataset-lock.json', {'dataset':'original'})
    atomic_write_json(directory/'data-lock.json', dict(unit_configuration=UNIT_CONFIGURATION,
                      common_dataset_lock_sha256=raw_sha(data/'common/dataset-lock.json')))
    from direct_s2st.io import atomic_write_text
    atomic_write_text(directory/'test.tsv', 'id\tsrc_audio\tsrc_n_frames\ttgt_audio\ttgt_n_frames\n'+
                      ''.join(f"{r['pair_id']}\t{r['ja_audio']}\t10\t1 2\t2\n" for r in rows))
    from direct_s2st.s2ut.multitask import tokenize
    atomic_write_text(directory/'target_letter/test.tsv', 'id\ttgt_text\n'+
                      ''.join(r['pair_id']+'\t'+' '.join(tokenize(r['en_text']))+'\n' for r in rows))
    name, _, _ = fixture_run(persistent, 's2ut', data, update=400000)
    def selected(work, mode):
        assert mode == 'best'
        atomic_write_json(Path(work)/'training-monitor.json', dict(best_update=390000), overwrite=True)
        return Path(work)/'checkpoints/checkpoint_last.pt'  # Byte fixture, no torch.
    monkeypatch.setattr('direct_s2st.translatotron2.monitor.select_checkpoint', selected)
    # Tests must use real WAV/ZIP staging but never install dependencies or load models.
    plan = session.prepare(persistent, data, local, name, 'a'*40)
    return persistent, data, local, rows, plan


def test_prepare_only_s2ut_best_full_test_and_reuses_plan(tmp_path, monkeypatch):
    persistent, data, local, rows, plan = benchmark_fixture(tmp_path, monkeypatch)
    assert plan['run']['updates'] == 400000 and plan['best_update'] == 390000
    assert plan['ids'] == [r['pair_id'] for r in rows] and plan['split'] == 'test'
    assert plan['vocoder'] == 'official_hubert100_ljspeech'
    assert 'tt2' not in plan
    second = session.prepare(persistent, data, local, plan['run']['name'], 'a'*40)
    assert second == plan
    # Test references must agree with the immutable prepared target labels.
    changed = dict(rows[0], en_text='not the original')
    atomic_write_jsonl(data/'common/test.jsonl', [changed, rows[1]], overwrite=True)
    with pytest.raises(ValueError, match='reference differs'):
        session.validate_test(plan['run'], data)


def test_worker_audio_saved_and_reused_without_models_then_cpu_listening(tmp_path, monkeypatch):
    persistent, data, local, rows, plan = benchmark_fixture(tmp_path, monkeypatch)
    root = Path(plan['local'])
    units = [dict(pair_id=r['pair_id'], system_id='s2ut', run_id=plan['run']['name'],
                  source_audio=r['ja_audio'], reference_audio=r['en_audio'], reference_text=r['en_text'],
                  units=[1,2], status='success', error=None, inference_seconds=0.01) for r in rows]
    atomic_write_jsonl(root/'units/predictions.jsonl', units)
    calls = []
    def ensure(checkpoint, config):
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(b'mocked generator')
        atomic_write_json(config, dict(sampling_rate=16000), overwrite=True)
    def vocode(kind, inputs, output, *args, **kwargs):
        calls.append(kind)
        values = []
        for index, row in enumerate(read_jsonl(inputs)):
            path = output/'audio'/f'{index:08d}.wav'
            wav(path)
            values.append(dict(row, output_audio=str(path), output_sha256=raw_sha(path),
                               output_duration=0.1, real_time_factor=0.1))
        atomic_write_jsonl(output/'predictions.jsonl', values, overwrite=True)
        from direct_s2st.vocoders.reference import CHECKPOINT_SHA256, CONFIG_SHA256
        atomic_write_json(output/'predictions.lock.json', dict(backend='fairseq_unit_hifigan',
                          checkpoint_sha256=CHECKPOINT_SHA256, config_sha256=CONFIG_SHA256,
                          input_sha256=raw_sha(inputs), sample_rate=16000,
                          duration_prediction=True, mel=None), overwrite=True)
    monkeypatch.setattr('direct_s2st.vocoders.reference.ensure_reference', ensure)
    monkeypatch.setattr('direct_s2st.vocoders.inference.vocode', vocode)
    worker.audio(plan)
    worker.audio(plan)
    assert calls == ['unit']  # No repeated model loading or synthesis.
    index = read(Path(plan['output'])/'audio-index.json')
    assert len(index['body']['samples']) == 2
    worker.restore_audio(plan)
    asr_calls = []
    def load_asr(**kwargs):
        assert kwargs == dict(plan['asr'], device=0)
        def transcribe(path):
            asr_calls.append(path)
            return 'the book is new'
        return transcribe
    monkeypatch.setattr('direct_s2st.evaluation.transcribe.EvaluationASR', load_asr)
    worker.evaluate(plan)
    metrics = read(Path(plan['output'])/'results/metrics.json')
    assert metrics['samples'] == 2 and metrics['bleu'] == 100
    assert metrics['training_updates'] == 400000 and metrics['selected_best_update'] == 390000
    # Simulate losing only this test's local VM folders, retaining Drive ZIPs.
    (root/'audio').rename(tmp_path/'lost-audio')
    asr_directory = next(root.glob('asr-*'))
    asr_directory.rename(tmp_path/'lost-asr')
    worker.evaluate(plan)
    assert len(asr_calls) == 2  # Restored ASR journal is reused, not retranscribed.
    # A fresh listening root needs no environment recovery, torch, or inference.
    def forbidden(*args, **kwargs):
        raise AssertionError('CPU saved listening must not load models')
    monkeypatch.setattr(session, 'environment', forbidden)
    listened = session.listen(persistent, tmp_path/'fresh', pair_id='test_1')
    assert listened['pair_id'] == 'test_1'
    assert set(listened['audio']) == {'S2UT翻訳','日本語原音','英語正解'}
    assert all(Path(path).is_file() for path in listened['audio'].values())
    assert listened['evaluation']['hypothesis_raw'] == 'the book is new'
    (tmp_path/'corpus').rename(tmp_path/'lost-corpus')
    translation_only = session.listen(persistent, tmp_path/'another-vm', pair_id='test_1')
    assert set(translation_only['audio']) == {'S2UT翻訳'}
    assert set(translation_only['original_errors']) == {'日本語原音','英語正解'}
    with pytest.raises(ValueError, match='not present'):
        session.listen(persistent, tmp_path/'fresh', pair_id='absent')


def test_generation_shards_survive_vm_loss_and_only_missing_shard_runs(tmp_path):
    from direct_s2st.s2ut.fairseq_infer import generate_shards, generation_command
    generated, drive = tmp_path/'gen', tmp_path/'drive'
    command = generation_command('prepared', 'best.pt', generated, 'test', 10, 'cpu')
    calls = []
    fail = True
    def runner(args, **kwargs):
        shard = int(args[args.index('--distributed-rank')+1])
        calls.append(shard)
        if shard == 1 and fail:
            raise RuntimeError('VM interrupted')
        path = Path(args[args.index('--results-path')+1])/'generate-test.txt'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f'D-{shard}\t0\t1 2\n')
    def durable():
        publish_tree(generated, drive, 'units', {'model':'best'})
    with pytest.raises(RuntimeError, match='VM interrupted'):
        generate_shards(command, generated, 'test', ['a','b'], {}, resume=True,
                        shard_size=1, runner=runner, on_shard=durable)
    generated.rename(tmp_path/'lost')
    restore_tree(drive, 'units', generated, {'model':'best'})
    fail = False
    records, _ = generate_shards(command, generated, 'test', ['a','b'], {}, resume=True,
                                 shard_size=1, runner=runner, on_shard=durable)
    assert records == {'a':[1,2],'b':[1,2]} and calls == [0,1,1]


def test_execution_restores_original_env_without_tt2_and_rejects_changed_best(tmp_path, monkeypatch):
    persistent, _, _, _, plan = benchmark_fixture(tmp_path, monkeypatch)
    monkeypatch.setenv('S2ST_INPUT_MAP', '')  # Undo the launcher process environment at teardown.
    monkeypatch.setattr(session, 'environment', lambda *args: '/saved/python3.10')
    calls = []
    session.execute(tmp_path/'repo', persistent, 'run', lambda *args: calls.append(args), 'a'*40)
    workers = [c for c in calls if 'direct_s2st.s2ut.benchmark_worker' in c]
    assert [c[-1] for c in workers] == ['units','audio']
    assert all(c[0] == '/saved/python3.10' for c in calls)
    assert all('tt2' not in str(c) for c in calls)
    Path(plan['checkpoint']).write_bytes(b'changed best')
    with pytest.raises(ValueError, match='checkpoint changed'):
        session.execute(tmp_path/'repo', persistent, 'run', lambda *args: None, 'a'*40)
    with pytest.raises(ValueError, match='revision changed'):
        session.execute(tmp_path/'repo', persistent, 'run', lambda *args: None, 'b'*40)


def test_evaluation_full_denominator_and_resume_callback(tmp_path):
    from test_evaluation import _prediction
    from direct_s2st.evaluation.run import evaluate_predictions
    source = tmp_path/'a.wav'
    wav(source)
    path = tmp_path/'predictions.jsonl'
    atomic_write_jsonl(path, [_prediction('ok', source, 'success'), _prediction('bad', None, 'failed')])
    calls = []
    metrics = evaluate_predictions(path, tmp_path/'scores', transcribe=lambda _: 'the book is new',
                                   include_failed_in_bleu=True, on_record=lambda n,t: calls.append((n,t)))
    assert 0 < metrics['bleu'] < 100 and metrics['asr_success_rate'] == 0.5
    assert metrics['bleu_denominator'] == 'all_samples_failed_as_empty'
    assert calls == [(1,2),(2,2)]
    def no_retry(_):
        raise AssertionError('completed ASR must be reused')
    resumed = evaluate_predictions(path, tmp_path/'scores', transcribe=no_retry,
                                   include_failed_in_bleu=True, resume=True)
    assert resumed == metrics


def test_evaluation_cuda_failure_is_fatal(tmp_path):
    from test_evaluation import _prediction
    from direct_s2st.evaluation.run import evaluate_predictions
    source = tmp_path/'a.wav'
    wav(source)
    path = tmp_path/'predictions.jsonl'
    atomic_write_jsonl(path, [_prediction('ok', source, 'success')])
    def oom(_):
        raise RuntimeError('CUDA out of memory')
    with pytest.raises(RuntimeError, match='CUDA'):
        evaluate_predictions(path, tmp_path/'scores', transcribe=oom, include_failed_in_bleu=True)


def test_full_test_ids_and_notebook_normal_cells():
    with pytest.raises(ValueError, match='exactly all'):
        worker.ordered([{'pair_id':'a'}], ['a','b'])
    nb = read(ROOT/'notebooks/colab_training.ipynb')
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            compile(''.join(cell['source']), '<notebook>', 'exec')
    for action in ('prepare','run','evaluate'):
        assert f"run_s2ut_benchmark('{action}')" in ''.join(''.join(c['source']) for c in nb['cells'])
    assert "--action', 'listen'" in ''.join(''.join(c['source']) for c in nb['cells'])
