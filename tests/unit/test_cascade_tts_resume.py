"""Fail-closed text reuse, isolated synthesis preflight, and real ZIP recovery."""
from pathlib import Path
import json
import sys

import pytest

from direct_s2st.cascade import benchmark_worker as worker, tts
from direct_s2st.benchmark_storage import read
from direct_s2st.drive_staging import raw_sha
from direct_s2st.io import atomic_write_json
from direct_s2st.journal import digest
from test_cascade_benchmark import fixture, models, session, generation_record
from test_s2ut_benchmark import wav


def prepared(tmp_path, monkeypatch, count=3):
    persistent, _, _, _, plan = fixture(tmp_path, monkeypatch, count)
    models(monkeypatch)
    worker.text_phase(plan, 'asr')
    worker.text_phase(plan, 'mt')
    return persistent, plan


def files(root):
    return {str(p.relative_to(root)): (p.read_bytes(), p.stat().st_mtime_ns)
            for p in root.rglob('*') if p.is_file()}


def test_completed_text_is_read_only_and_never_loads_asr_mt(tmp_path, monkeypatch, capsys):
    persistent, plan = prepared(tmp_path, monkeypatch)
    output, local = Path(plan['output']), Path(plan['local'])
    before = files(output)
    def forbidden(*args, **kwargs):
        raise AssertionError('saved text must not load/recompute Whisper or NLLB')
    monkeypatch.setattr('direct_s2st.cascade.asr.WhisperASR', forbidden)
    monkeypatch.setattr('direct_s2st.cascade.mt.NllbTranslator', forbidden)
    monkeypatch.setattr(worker, 'loader', forbidden)
    (local/'text').rename(tmp_path/'lost-text')
    rows = worker.verify_text(plan)
    assert len(rows) == 3
    assert files(output) == before  # Even pointers/ZIP timestamps stay untouched.
    calls = []
    monkeypatch.setattr(session, 'cascade_environment', lambda *args: '/runtime/python')
    session.execute_cascade(persistent, 'run', lambda *args: calls.append(args), 'a'*40)
    assert [c[-1] for c in calls if 'direct_s2st.cascade.benchmark_worker' in c] == ['verify-text','tts']
    log = capsys.readouterr().err
    assert 'ASR: reuse 3/3' in log and 'MT: reuse 3/3' in log


@pytest.mark.parametrize('damage', ['missing_pointer', 'local_text', 'zip', 'identity', 'missing_rows'])
def test_damaged_text_fails_never_regenerates(tmp_path, monkeypatch, damage):
    _, plan = prepared(tmp_path, monkeypatch)
    output, local = Path(plan['output']), Path(plan['local'])
    pointer_path = output/'text/mt.json'
    pointer = read(pointer_path)
    body = pointer['body']
    directory = local/'text'/body['label']
    if damage == 'missing_pointer':
        pointer_path.unlink()
    elif damage == 'local_text':
        (directory/'predictions.jsonl').write_text('broken')
    elif damage == 'zip':
        archive = read(output/'progress'/(body['label']+'.json'))['body']['archive']
        (output/'progress'/archive).write_bytes(b'corrupt')
    elif damage == 'identity':
        body['identity']['inputs'] = 'c'*64
        atomic_write_json(pointer_path, dict(body=body, sha256=digest(body)), overwrite=True)
    else:
        # Hash-valid but incomplete journal/receipt must also fail.
        from direct_s2st.io import read_jsonl, atomic_write_jsonl
        from direct_s2st.benchmark_storage import publish_tree
        path = directory/'predictions.jsonl'
        atomic_write_jsonl(path, list(read_jsonl(path))[:-1], overwrite=True)
        publish_tree(directory, output/'progress', body['label'], body['identity'])
        body['predictions_sha256'] = raw_sha(path)
        atomic_write_json(pointer_path, dict(body=body, sha256=digest(body)), overwrite=True)
    monkeypatch.setattr(worker, 'loader', lambda *args: pytest.fail('must not regenerate'))
    with pytest.raises((ValueError, OSError)):
        worker.verify_text(plan)


@pytest.mark.parametrize('mode', ['exception', 'silent', 'stereo'])
def test_failed_preflight_produces_zero_formal_samples(tmp_path, monkeypatch, mode):
    _, plan = prepared(tmp_path, monkeypatch)
    calls = []
    def run(text, path):
        import numpy as np
        import soundfile as sf
        calls.append(text)
        assert text == 'This is a synthesis test.'
        if mode == 'exception':
            raise RuntimeError('model/tokenizer failure')
        sf.write(path, np.zeros(100) if mode == 'silent' else np.ones((100, 2))*.1, 24000)
    monkeypatch.setattr(worker, 'loader', lambda *args: run)
    with pytest.raises((RuntimeError, ValueError)):
        worker.synthesize(plan)
    assert calls == ['This is a synthesis test.']
    assert not list((Path(plan['local'])/'audio').rglob('*.wav'))
    assert not list((Path(plan['local'])/'audio').rglob('predictions.jsonl'))
    assert not (Path(plan['output'])/'audio-index.json').exists()


def test_partial_tts_resume_preserves_wavs_timings_and_preflight_order(tmp_path, monkeypatch, capsys):
    _, plan = prepared(tmp_path, monkeypatch)
    calls = []
    def run(text, path):
        calls.append((text, path))
        if len(calls) == 3:
            raise KeyboardInterrupt()
        wav(path)
        generation_record(run, plan, text)
    monkeypatch.setattr(worker, 'loader', lambda *args: run)
    with pytest.raises(KeyboardInterrupt):
        worker.synthesize(plan)
    assert calls[0][0] == 'This is a synthesis test.'
    path = calls[1][1]
    before = (path.read_bytes(), path.stat().st_mtime_ns)
    saved = read(next((Path(plan['local'])/'audio').glob('*/predictions.jsonl')))
    calls.clear()
    def resumed(text, path):
        calls.append((text, path))
        wav(path)
        generation_record(resumed, plan, text)
    monkeypatch.setattr(worker, 'loader', lambda *args: resumed)
    result = worker.synthesize(plan)
    assert len(calls) == 3 and calls[0][0] == 'This is a synthesis test.'
    assert path not in [p for _, p in calls]  # Saved sample is never sent to Qwen.
    assert (path.read_bytes(), path.stat().st_mtime_ns) == before
    assert result[0] == saved
    assert all(r['mt_en_text'] == 'actual translated english' and r['tts_seconds'] > 0 for r in result)
    assert {r['pair_id'] for r in result} == set(plan['ids'])
    index = read(Path(plan['output'])/'audio-index.json')['body']['samples']
    assert len(index) == 3 and all('preflight' not in r['member'] for r in index)
    assert 'TTS: saved=1 remaining=2' in capsys.readouterr().err
    # Whole VM loss after committed part: restore ZIPs; zero Qwen loads/calls.
    (Path(plan['local'])/'audio').rename(tmp_path/'lost-audio')
    monkeypatch.setattr(worker, 'loader', lambda *args: pytest.fail('all WAVs already saved'))
    assert worker.synthesize(plan) == result


@pytest.mark.parametrize('lose_vm', [False, True])
def test_complete_group_and_partial_group_survive_midrun_stop(tmp_path, monkeypatch, lose_vm):
    _, plan = prepared(tmp_path, monkeypatch, count=34)
    formal = []
    def run(text, path):
        if text != 'This is a synthesis test.':
            formal.append(path)
            if len(formal) == 34:
                raise KeyboardInterrupt()
        wav(path)
        generation_record(run, plan, text)
    monkeypatch.setattr(worker, 'loader', lambda *args: run)
    with pytest.raises(KeyboardInterrupt):
        worker.synthesize(plan)
    assert len(list((Path(plan['output'])/'audio').glob('*.json'))) == 1
    if lose_vm:
        (Path(plan['local'])/'audio').rename(tmp_path/'lost-audio')
    formal.clear()
    result = worker.synthesize(plan)
    assert len(result) == 34 and len(formal) == (2 if lose_vm else 1)


def test_corrupt_saved_wav_fails_before_preflight_or_regeneration(tmp_path, monkeypatch):
    _, plan = prepared(tmp_path, monkeypatch)
    result = worker.synthesize(plan)
    Path(result[0]['output_audio']).write_bytes(b'corrupt')
    monkeypatch.setattr(worker, 'loader', lambda *args: pytest.fail('must not regenerate'))
    with pytest.raises(ValueError, match='local TTS artifact changed'):
        worker.synthesize(plan)


def test_preflight_restores_rng_and_records_isolated_metadata(tmp_path, monkeypatch):
    import random
    import numpy as np
    import torch
    _, plan = prepared(tmp_path, monkeypatch)
    random.seed(17)
    np.random.seed(17)
    torch.manual_seed(17)
    expected = (random.random(), np.random.random(), torch.rand(1))
    random.seed(17)
    np.random.seed(17)
    torch.manual_seed(17)
    def synth(text, path):
        random.random()
        np.random.random()
        torch.rand(5)
        wav(path)
        generation_record(synth, plan, text)
    worker.synthesis_preflight(plan, synth)
    actual = (random.random(), np.random.random(), torch.rand(1))
    assert expected[:2] == actual[:2] and torch.equal(expected[2], actual[2])
    receipt = read(next((Path(plan['output'])/'preflight').glob('*.json')))
    assert receipt['sha256'] == digest(receipt['body'])
    assert receipt['body']['benchmark_sample'] is False
    assert receipt['body']['output_sha256'] == raw_sha(receipt['body']['output_audio'])
    assert receipt['body']['tts_generation']['eos_observed'] is True


def test_interrupted_retry_merges_new_local_progress_without_overwriting_durable_success(tmp_path, monkeypatch):
    from direct_s2st.io import atomic_write_jsonl
    _, plan = prepared(tmp_path, monkeypatch)
    def initial(text, path):
        if path.stem == 'test_000':
            raise ValueError('transient synthesis failure')
        wav(path)
        generation_record(initial, plan, text)
    monkeypatch.setattr(worker, 'loader', lambda *args: initial)
    first = worker.synthesize(plan)
    assert first[0]['status'] == 'failed'
    before = files(Path(plan['output'])/'audio')
    preserved = Path(first[1]['output_audio']).read_bytes()
    # Local retry committed successfully, but its newer ZIP was not published.
    journal = next((Path(plan['local'])/'audio').glob('*/predictions.jsonl'))
    repaired = dict(first[0])
    for field in ('status', 'error', 'output_sha256', 'output_duration', 'tts_seconds', 'tts_generation'):
        repaired[field] = first[1][field]
    repaired['output_audio'] = str(journal.parent/'audio/test_000.wav')
    wav(Path(repaired['output_audio']))
    atomic_write_jsonl(journal, [repaired, *first[1:]], overwrite=True)
    monkeypatch.setattr(worker, 'loader', lambda *args: pytest.fail('successful local retry must be reused'))
    result = worker.synthesize(plan)
    assert all(r['status'] == 'success' for r in result)
    assert Path(first[1]['output_audio']).read_bytes() == preserved
    after = files(Path(plan['output'])/'audio')
    assert all(after[k] == v for k, v in before.items() if k.endswith('.zip'))


def test_resumed_audio_index_never_hides_saved_later_parts(tmp_path, monkeypatch):
    _, plan = prepared(tmp_path, monkeypatch, count=33)
    worker.synthesize(plan)
    save = worker.save_index
    counts = []
    def observed(output, identity, samples):
        counts.append(len(samples))
        save(output, identity, samples)
    monkeypatch.setattr(worker, 'save_index', observed)
    monkeypatch.setattr(worker, 'loader', lambda *args: pytest.fail('all saved'))
    worker.synthesize(plan)
    assert counts == [33, 33]


@pytest.mark.parametrize('field,value', [('reference_text', 'wrong reference'), ('tts_generation', {})])
def test_new_policy_reuse_checks_provenance_and_eos_evidence(tmp_path, monkeypatch, field, value):
    from direct_s2st.io import atomic_write_jsonl
    from direct_s2st.benchmark_storage import publish_tree
    from direct_s2st.io import read_jsonl
    _, plan = prepared(tmp_path, monkeypatch)
    worker.synthesize(plan)
    journal = next((Path(plan['local'])/'audio').glob('*/predictions.jsonl'))
    rows = list(read_jsonl(journal))
    rows[0][field] = value
    atomic_write_jsonl(journal, rows, overwrite=True)
    publish_tree(journal.parent, Path(plan['output'])/'audio', journal.parent.name, read(journal.with_suffix('.lock.json')))
    monkeypatch.setattr(worker, 'loader', lambda *args: pytest.fail('must fail before model loading'))
    with pytest.raises((ValueError, RuntimeError), match='input changed|verified EOS/policy evidence'):
        worker.synthesize(plan)


@pytest.mark.parametrize('cuda,bf16', [(False, False), (True, False)])
def test_cuda_bf16_guard_precedes_any_download_or_load(monkeypatch, cuda, bf16):
    import torch
    from contextlib import nullcontext
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: cuda)
    monkeypatch.setattr(torch.cuda, 'device', lambda device: nullcontext())
    monkeypatch.setattr(torch.cuda, 'is_bf16_supported', lambda: bf16)
    monkeypatch.setattr(torch.cuda, 'get_device_name', lambda device: 'NVIDIA L4')
    monkeypatch.setattr(tts, 'pinned_snapshot', lambda *args: pytest.fail('must check GPU first'))
    with pytest.raises(RuntimeError, match='no FP16/CPU fallback'):
        tts.QwenTTS(model_id='Qwen/model', revision='a'*40, speaker='Ono_Anna', language='English')


def test_reviewed_overlay_rejects_changed_text_functions(tmp_path, monkeypatch):
    import cascade_loader_overlay as overlay
    name = 'src/direct_s2st/cascade/benchmark_worker.py'
    original = (session.ROOT/name).read_text(encoding='utf-8')
    monkeypatch.setattr(overlay, 'git', lambda *args: original)
    path = tmp_path/name
    path.parent.mkdir(parents=True)
    path.write_text(original, encoding='utf-8')
    overlay.verify_text_code(tmp_path, overlay.BASE_REVISION)
    path.write_text(original.replace("cfg = plan['pipeline'][phase]", "cfg = {}"), encoding='utf-8')
    with pytest.raises(ValueError, match='unreviewed worker ASR/MT'):
        overlay.verify_text_code(tmp_path, overlay.BASE_REVISION)


def test_only_verified_complete_loader_audio_can_migrate_without_touching_old_files(tmp_path, monkeypatch):
    import cascade_loader_overlay as overlay
    from test_qwen_tts_loading import overlay_plan
    path, plan = overlay_plan(tmp_path)
    old = dict(version=1, reason='complete_pinned_qwen_snapshot_before_local_sdk_load',
               plan=plan['identity'], original_plan_sha256=raw_sha(path),
               base_revision=overlay.BASE_REVISION, execution_revision=overlay.PINNED_LOADER_REVISION,
               changed_files=['src/direct_s2st/cascade/tts.py'])
    previous = Path(plan['output'])/'execution-overlays'/(overlay.PINNED_LOADER_REVISION+'.json')
    atomic_write_json(previous, dict(body=old, sha256=digest(old)))
    audio = Path(plan['local'])/'audio/existing.wav'
    wav(audio)
    name = 'src/direct_s2st/cascade/tts.py'
    target = tmp_path/name
    target.parent.mkdir(parents=True)
    original = (session.ROOT/name).read_text(encoding='utf-8')
    target.write_text(original, encoding='utf-8')
    def git(root, *args):
        if args[:1] == ('show',):
            return original
        if args[:1] == ('diff',):
            return name
        if args[:1] == ('merge-base',):
            return args[1]
        if args[:1] == ('status',):
            return ''
        if args == ('rev-parse', 'HEAD'):
            return 'c'*40
        raise AssertionError(args)
    monkeypatch.setattr(overlay, 'git', git)
    before = files(Path(plan['output'])), files(Path(plan['local']))
    receipt = overlay.approve(tmp_path, path, plan, 'c'*40, adopt=True)
    assert receipt['body']['parent_receipt_sha256'] == digest(old)
    assert all(files(Path(plan['output']))[key] == value for key, value in before[0].items())
    assert files(Path(plan['local'])) == before[1]
    assert overlay.approve(tmp_path, path, plan, 'c'*40) == receipt
    # A different decoding setting cannot inherit old audio under this migration.
    target.write_text(original.replace('attn_implementation="sdpa"', 'attn_implementation="eager"'), encoding='utf-8')
    with pytest.raises(ValueError, match='decoding contract changed'):
        overlay.approve(tmp_path, path, plan, 'c'*40)
    target.write_text(original, encoding='utf-8')
    old['original_plan_sha256'] = '0'*64
    atomic_write_json(previous, dict(body=old, sha256=digest(old)), overwrite=True)
    with pytest.raises(ValueError, match='receipt/lineage changed'):
        overlay.approve(tmp_path, path, plan, 'c'*40)
