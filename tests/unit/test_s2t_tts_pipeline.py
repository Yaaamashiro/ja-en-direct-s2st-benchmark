import json
import wave
from pathlib import Path
import pytest
from direct_s2st.io import read_jsonl
from direct_s2st.config import load_config
from direct_s2st.s2t_tts.pipeline import run_pipeline

ROOT = Path(__file__).resolve().parents[2]


def wav(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), 'wb') as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(16000)
        f.writeframes(b'\x10\x01' * 1600)


def manifest(root, count=4):
    wav(root/'source.wav')
    rows = [dict(pair_id=str(i), split='test', ja_audio=str(root/'source.wav'),
                 en_audio=str(root/'source.wav'), en_text='The book is new.') for i in range(count)]
    path = root/'test.jsonl'
    path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    return path


def test_pipeline_resume_identity_and_hash(tmp_path):
    source = manifest(tmp_path)
    calls = []
    def s2t(path):
        calls.append('s2t')
        return 'The book is new.'
    def tts(text, path):
        assert text == 'The book is new.'
        calls.append('tts')
        wav(path)
    kwargs = dict(run_id='trial', s2t=s2t, tts=tts, limit=1, model_identity={'revision': 'one'})
    out = tmp_path/'out'
    assert run_pipeline(source, out, **kwargs)['successes'] == 1
    row = next(read_jsonl(out/'predictions.jsonl'))
    assert row['s2t_en_text'] == 'The book is new.'
    assert row['system_id'] == 's2t_tts' and row['output_duration'] == .1
    assert len(row['output_sha256']) == 64 and 'asr_ja_text' not in row and 'mt_en_text' not in row
    run_pipeline(source, out, resume=True, **kwargs)
    assert calls == ['s2t', 'tts']
    with pytest.raises(ValueError, match='identity'):
        run_pipeline(source, out, resume=True, **(kwargs | {'run_id': 'different'}))
    Path(row['output_audio']).write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='SHA'):
        run_pipeline(source, out, resume=True, **kwargs)
    run_pipeline(source, out, overwrite=True, **kwargs)
    assert calls == ['s2t', 'tts', 's2t', 'tts']


@pytest.mark.parametrize('message,fatal', [('translation failed', False), ('CUDA out of memory', True)])
def test_failures_are_journaled(tmp_path, message, fatal):
    def fail(_):
        raise RuntimeError(message)
    def execute():
        return run_pipeline(manifest(tmp_path, 1), tmp_path/'out', run_id='trial',
                            s2t=fail, tts=lambda *_: pytest.fail('TTS called'))
    if fatal:
        with pytest.raises(RuntimeError, match='CUDA'):
            execute()
    else:
        assert execute()['failures'] == 1
    row = next(read_jsonl(tmp_path/'out/predictions.jsonl'))
    assert row['status'] == 'failed' and message in row['error']


def test_shards_partition_and_evaluation(tmp_path):
    from direct_s2st.evaluation.run import evaluate_predictions
    source = manifest(tmp_path)
    ids = []
    for shard in range(2):
        out = tmp_path/str(shard)
        run_pipeline(source, out, run_id='trial', s2t=lambda _: 'The book is new.',
                     tts=lambda text, path: wav(path), shard_index=shard, num_shards=2)
        rows = list(read_jsonl(out/'predictions.jsonl'))
        ids.extend(r['pair_id'] for r in rows)
        if rows:
            metrics = evaluate_predictions(out/'predictions.jsonl', out/'metrics',
                                           transcribe=lambda _: 'The book is new.')
            assert metrics['system_id'] == 's2t_tts' and metrics['bleu'] == 100
    assert sorted(ids) == ['0', '1', '2', '3']


def test_whisper_translation_config_and_tts_reuse(monkeypatch):
    from direct_s2st.s2t_tts import s2t, pipeline
    from direct_s2st.cascade.tts import QwenTTS
    calls = []
    monkeypatch.setattr(s2t, 'load_whisper_pipeline', lambda *args:
        lambda path, **kw: (calls.append((args, kw)) or {'text': ' English. '}))
    cfg = load_config(ROOT/'configs/s2t_tts/default.yaml')
    s2t.validate_s2t_config(cfg)
    speech = cfg['s2t']
    model = s2t.WhisperS2T(model_id=speech['model'], revision=speech['revision'],
                           language=speech['language'], task=speech['task'])
    assert model(Path('input.wav')) == 'English.'
    assert calls[0][1]['generate_kwargs'] == dict(language='japanese', task='translate',
        do_sample=False, condition_on_prev_tokens=False)
    assert calls[0][0][0] == 'openai/whisper-large-v3'
    assert pipeline.QwenTTS is QwenTTS
    assert cfg['tts'] == load_config(ROOT/'configs/cascade/default.yaml')['tts']
    cfg['s2t']['revision'] = 'main'
    with pytest.raises(ValueError, match='immutable'):
        s2t.validate_s2t_config(cfg)


def test_cli_dry_run_never_loads_models(tmp_path, monkeypatch, capsys):
    from direct_s2st.cli import main
    from direct_s2st.s2t_tts import pipeline
    for name in ('CORPUS_ROOT', 'EXPERIMENT_DATA_ROOT', 'RUNS_ROOT', 'CACHE_ROOT'):
        monkeypatch.setenv(name, str(tmp_path/name))
    monkeypatch.setattr(pipeline, 'components_from_config', lambda _: pytest.fail('models loaded'))
    assert main(['s2t-tts', 'run', '--profile', 'smoke', '--split', 'test', '--dry-run']) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan['system_id'] == 's2t_tts'
    assert plan['config']['s2t']['task'] == 'translate'
    assert plan['manifest'].endswith('test.jsonl')


def test_four_system_acceptance_rejects_three(tmp_path, monkeypatch):
    from direct_s2st.evaluation.acceptance import verify_suite
    from direct_s2st.evaluation.run import evaluate_predictions
    from direct_s2st.predictions import SYSTEMS
    from direct_s2st.s2ut import checkpoint
    from direct_s2st.translatotron2 import engine
    # Checkpoint verification is mocked here; this is not a real E2E claim.
    monkeypatch.setattr(checkpoint, 'validate_checkpoint', lambda _: None)
    monkeypatch.setattr(engine, 'load_checkpoint', lambda _: None)
    source = manifest(tmp_path, 1)
    roots = []
    for system in SYSTEMS:
        root = tmp_path/('trial-'+system)
        roots.append(root)
        run_pipeline(source, root/'predictions', run_id=root.name,
                     s2t=lambda _: 'The book is new.', tts=lambda text, path: wav(path))
        path = root/'predictions/predictions.jsonl'
        rows = list(read_jsonl(path))
        rows[0]['system_id'] = system
        path.write_text(json.dumps(rows[0])+'\n')
        (root/'gradient-audit-rank-0.json').write_text('{"status":"PASS"}')
        evaluate_predictions(path, root/'metrics', transcribe=lambda _: 'The book is new.',
                             evaluation_identity={'run_id': root.name, 'asr': 'mock-common'})
    assert verify_suite(tmp_path, roots, tmp_path/'report')['status'] == 'PASS'
    with pytest.raises(ValueError, match='exactly four'):
        verify_suite(tmp_path, roots[:3], tmp_path/'incomplete')


def test_original_asr_remains_transcription(monkeypatch):
    from direct_s2st.cascade import whisper
    from direct_s2st.cascade.asr import WhisperASR
    calls = []
    monkeypatch.setattr(whisper, 'load_whisper_pipeline', lambda *args:
        lambda path, **kwargs: (calls.append((args, kwargs)) or {'text': ' Japanese text '}))
    cfg = load_config(ROOT/'configs/cascade/default.yaml')['asr']
    model = WhisperASR(model_id=cfg['model'], revision=cfg['revision'])
    assert model(Path('input.wav')) == 'Japanese text'
    assert calls[0][0][0] == 'openai/whisper-large-v3-turbo'
    assert calls[0][1]['generate_kwargs']['task'] == 'transcribe'


def test_invalid_audio_is_not_success(tmp_path):
    def silent(text, path):
        wav(path)
        with wave.open(str(path), 'wb') as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(16000)
            f.writeframes(b'\0\0' * 1600)
    assert run_pipeline(manifest(tmp_path, 1), tmp_path/'out', run_id='trial',
        s2t=lambda _: 'English', tts=silent)['failures'] == 1
    assert 'silent' in next(read_jsonl(tmp_path/'out/predictions.jsonl'))['error']


def test_changed_input_cannot_resume(tmp_path):
    path = manifest(tmp_path, 1)
    kwargs = dict(run_id='trial', s2t=lambda _: 'English', tts=lambda text, path: wav(path))
    run_pipeline(path, tmp_path/'out', **kwargs)
    (tmp_path/'source.wav').write_bytes(b'changed input')
    with pytest.raises(ValueError, match='identity'):
        run_pipeline(path, tmp_path/'out', resume=True, **kwargs)
