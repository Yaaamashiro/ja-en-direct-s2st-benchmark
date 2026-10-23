import json
from pathlib import Path
import sys

import pytest

from direct_s2st.benchmark_storage import read
from direct_s2st.drive_staging import raw_sha
from direct_s2st.io import atomic_write_json, atomic_write_jsonl, read_jsonl
from direct_s2st.journal import digest
from direct_s2st.cascade import benchmark_worker as worker
from direct_s2st.s2ut import benchmark_worker as scoring

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'scripts/colab'))
import benchmark_session as session


def fixture(tmp_path, monkeypatch, count=3):
    from test_s2ut_benchmark import wav
    persistent, data, local = tmp_path/'d', tmp_path/'d/data', tmp_path/'l'
    rows = []
    for index in range(count):
        path = tmp_path/'corpus'/f'{index}.wav'
        wav(path)
        rows.append(dict(pair_id=f'test_{index:03d}', split='test', ja_audio=str(path), en_audio=str(path),
                         ja_sha256=raw_sha(path), en_sha256=raw_sha(path),
                         ja_text='これは教師の日本語', en_text='the book is new'))
    atomic_write_jsonl(data/'common/test.jsonl', rows)
    atomic_write_json(data/'common/dataset-lock.json', {'original_dataset':True})
    def no_training(*args, **kwargs):
        raise AssertionError('cascade must not discover/restore a trained run')
    monkeypatch.setattr(session, 'saved_run', no_training)
    monkeypatch.setattr(session, 'environment', no_training)
    plan = session.prepare_cascade(persistent, data, local, 'a'*40)
    body = dict(plan=plan['identity'], environment_sha256='b'*64)
    atomic_write_json(Path(plan['output'])/'runtime.json', dict(body=body, sha256=digest(body)))
    monkeypatch.setenv('S2ST_BENCHMARK_ENVIRONMENT_SHA256', 'b'*64)
    monkeypatch.setenv('S2ST_INPUT_MAP', str(Path(plan['local'])/'input-map.json'))
    return persistent, data, local, rows, plan


def models(monkeypatch):
    from test_s2ut_benchmark import wav
    loads, calls = [], []
    def load(plan, phase):
        loads.append(phase)
        def run(argument, output=None):
            calls.append((phase, argument))
            if phase == 'asr':
                assert isinstance(argument, Path) and argument.is_file()
                assert 'inputs' in argument.parts  # Reads local ZIPs, not Drive WAVs.
                return '実際に認識した日本語'
            if phase == 'mt':
                assert argument == '実際に認識した日本語'  # Never the JA/EN reference transcript.
                return 'actual translated english'
            assert argument == 'actual translated english'
            wav(output)
        return run
    monkeypatch.setattr(worker, 'loader', load)
    return loads, calls


def test_prepare_has_no_s2ut_tt2_or_vocoder_and_isolated_pointer(tmp_path, monkeypatch):
    persistent, data, local, rows, plan = fixture(tmp_path, monkeypatch)
    assert plan['system_id'] == 'cascade' and plan['split'] == 'test'
    assert 'checkpoint' not in plan and 'updates' not in plan['run']
    assert plan['ids'] == [r['pair_id'] for r in rows]
    assert plan['pipeline']['mt']['source_language'] == 'jpn_Jpan'
    assert set(plan['pipeline']) == {'asr','mt','tts'}
    assert plan['pipeline']['mt']['target_language'] == 'eng_Latn'
    atomic_write_json(persistent/'benchmarks/s2ut/latest.json', {'original':'s2ut result'})
    before = (persistent/'benchmarks/s2ut/latest.json').read_bytes()
    assert session.prepare_cascade(persistent, data, local, 'a'*40) == plan
    assert (persistent/'benchmarks/s2ut/latest.json').read_bytes() == before
    assert session.load_plan(persistent, 'cascade')[1] == plan
    with pytest.raises(ValueError, match='unsupported'):
        session.load_plan(persistent, 'unknown')
    rows[0]['split'] = 'dev'
    atomic_write_jsonl(data/'common/test.jsonl', rows, overwrite=True)
    with pytest.raises(ValueError, match='unique test IDs'):
        session.prepare_cascade(persistent, data, local, 'a'*40)


def test_all_stages_real_journals_zip_recovery_scoring_and_cpu_listening(tmp_path, monkeypatch):
    persistent, data, local, rows, plan = fixture(tmp_path, monkeypatch)
    loads, calls = models(monkeypatch)
    worker.text_phase(plan, 'asr')
    worker.text_phase(plan, 'mt')
    predictions = worker.synthesize(plan)
    assert loads == ['asr','mt','tts'] and len(calls) == 9
    assert all(p['system_id'] == 'cascade' and p['status'] == 'success' for p in predictions)
    asr_loads = []
    def load_asr(**kwargs):
        asr_loads.append(kwargs)
        return lambda path: 'the book is new'
    monkeypatch.setattr('direct_s2st.evaluation.transcribe.EvaluationASR', load_asr)
    scoring.evaluate(plan)
    output, owned = Path(plan['output']), Path(plan['local'])
    metrics = read(output/'results/metrics.json')
    assert metrics['bleu'] == 100 and metrics['system_id'] == 'cascade' and metrics['samples'] == 3
    assert 'selected_best_update' not in metrics and metrics['pipeline'] == plan['pipeline']
    csv = (output/'results/per_sample.csv').read_text(encoding='utf-8')
    assert 'actual translated english' in csv and '実際に認識した日本語' in csv
    assert 'Cascade full-test ASR-BLEU' in (output/'results/report.md').read_text()
    listened = session.listen(persistent, tmp_path/'first-listen', index=1, target='cascade')
    assert set(listened['audio']) == {'カスケード翻訳','日本語原音','英語正解'}
    # Lose this test's VM work folders only; immutable Drive archives survive.
    for name in ('text', 'audio'):
        (owned/name).rename(tmp_path/('lost-'+name))
    next(owned.glob('asr-*')).rename(tmp_path/'lost-scores')
    (tmp_path/'corpus').rename(tmp_path/'lost-originals')  # Verified input ZIPs are sufficient.
    worker.text_phase(plan, 'asr')
    worker.text_phase(plan, 'mt')
    worker.synthesize(plan)
    scoring.evaluate(plan)
    assert loads == ['asr','mt','tts'] and len(calls) == 9 and len(asr_loads) == 1
    listened = session.listen(persistent, tmp_path/'another-vm', index=1, target='cascade')
    assert listened['pair_id'] == 'test_001'
    assert set(listened['audio']) == {'カスケード翻訳'}
    assert listened['evaluation']['hypothesis_raw'] == 'the book is new'


def test_interrupted_asr_restores_first_32_and_only_computes_missing(tmp_path, monkeypatch):
    _, _, _, _, plan = fixture(tmp_path, monkeypatch, count=33)
    calls = []
    def load(plan, phase):
        def run(path):
            calls.append(path)
            if len(calls) == 33:
                raise RuntimeError('CUDA out of memory')
            return '認識文'
        return run
    monkeypatch.setattr(worker, 'loader', load)
    with pytest.raises(RuntimeError, match='CUDA'):
        worker.text_phase(plan, 'asr')
    (Path(plan['local'])/'text').rename(tmp_path/'lost-text')
    restored = worker.text_phase(plan, 'asr')
    assert len(restored) == 33 and len(calls) == 34


def test_failed_asr_is_not_sent_to_mt_or_tts_and_stays_in_bleu_denominator(tmp_path, monkeypatch):
    _, _, _, _, plan = fixture(tmp_path, monkeypatch, count=2)
    from test_s2ut_benchmark import wav
    calls = []
    def load(plan, phase):
        def run(value, output=None):
            calls.append(phase)
            if phase == 'asr':
                if calls.count('asr') == 1:
                    raise ValueError('invalid sample')
                return '日本語'
            if phase == 'mt':
                return 'the book is new'
            wav(output)
        return run
    monkeypatch.setattr(worker, 'loader', load)
    worker.text_phase(plan, 'asr')
    worker.text_phase(plan, 'mt')
    predictions = worker.synthesize(plan)
    assert calls == ['asr','asr','mt','tts']
    assert predictions[0]['status'] == 'failed' and predictions[0]['output_audio'] is None
    monkeypatch.setattr('direct_s2st.evaluation.transcribe.EvaluationASR', lambda **kwargs: lambda path:'the book is new')
    scoring.evaluate(plan)
    metrics = read(Path(plan['output'])/'results/metrics.json')
    assert metrics['generation_failures'] == 1 and metrics['asr_success_rate'] == 0.5
    assert 0 < metrics['bleu'] < 100 and metrics['bleu_denominator'] == 'all_samples_failed_as_empty'


def test_tts_cuda_is_fatal_and_runtime_mismatch_rejected(tmp_path, monkeypatch):
    _, _, _, _, plan = fixture(tmp_path, monkeypatch)
    models(monkeypatch)
    worker.text_phase(plan, 'asr')
    worker.text_phase(plan, 'mt')
    def load(plan, phase):
        def oom(*args):
            raise RuntimeError('CUDA out of memory')
        return oom
    monkeypatch.setattr(worker, 'loader', load)
    with pytest.raises(RuntimeError, match='CUDA'):
        worker.synthesize(plan)
    monkeypatch.setenv('S2ST_BENCHMARK_ENVIRONMENT_SHA256', 'c'*64)
    with pytest.raises(ValueError, match='dependency environment'):
        worker.text_phase(plan, 'asr')


def test_cascade_environment_is_pinned_separately_without_training(tmp_path, monkeypatch):
    persistent, _, _, _, plan = fixture(tmp_path, monkeypatch)
    # This is an independent first environment capture, not a trained-run manifest.
    (Path(plan['output'])/'runtime.json').unlink()
    prefix = tmp_path/'runtime'
    observed = []
    def ensure(repo, persistent, invoke, **kwargs):
        assert repo == session.ROOT
        observed.append(kwargs.get('expected'))
        atomic_write_json(prefix/'environment.json', {'packages':'qwen-tts==0.1.1\n'}, overwrite=True)
        return prefix/'venv/bin/python'
    monkeypatch.setattr('environment_recovery.ensure_environment', ensure)
    monkeypatch.setenv('PYTHONPATH', '')
    monkeypatch.setenv('S2ST_CONFIG_ROOT', '')
    python = session.cascade_environment(persistent, plan, lambda *args: None, prefix=prefix)
    checksum = raw_sha(prefix/'environment.json')
    assert python == prefix/'venv/bin/python' and observed == [None]
    session.cascade_environment(persistent, plan, lambda *args: None, prefix=prefix)
    assert observed == [None, checksum]
    assert read(Path(plan['output'])/'environment.json') == read(prefix/'environment.json')


def test_normal_cells_route_only_cascade_and_worker_processes_are_separate(tmp_path, monkeypatch):
    persistent, _, _, _, plan = fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(session, 'cascade_environment', lambda *args: '/cascade/python3.10')
    calls = []
    session.execute_cascade(persistent, 'run', lambda *args: calls.append(args), 'a'*40)
    phases = [c[-1] for c in calls if 'direct_s2st.cascade.benchmark_worker' in c]
    assert phases == ['asr','mt','tts']
    assert all(c[0] == '/cascade/python3.10' for c in calls)
    with pytest.raises(ValueError, match='revision changed'):
        session.execute_cascade(persistent, 'run', lambda *args: None, 'b'*40)
    nb = read(ROOT/'notebooks/colab_training.ipynb')
    namespace = {}
    exec(''.join(nb['cells'][1]['source']), namespace)
    namespace.update(BENCHMARK_TARGET='cascade', RUN_BENCHMARK=True,
                     REPO=tmp_path/'repo', PERSISTENT=persistent, DATA=persistent/'data',
                     S2UT_VOCODER_MODE='unused', BENCHMARK_RUN_NAME='unused', json=json)
    namespace['benchmark_launcher'] = lambda: ('python', session.ROOT/'scripts/colab/benchmark_session.py')
    commands = []
    namespace['run'] = lambda *args: commands.append(args)
    for title in ('6.', '7.', '8.'):
        cell = next(c for c in nb['cells'] if ''.join(c['source']).startswith('#@title '+title))
        exec(''.join(cell['source']), namespace)
    assert [c[c.index('--action')+1] for c in commands] == ['prepare','run','evaluate']
    assert all(c[c.index('--target')+1] == 'cascade' for c in commands)
