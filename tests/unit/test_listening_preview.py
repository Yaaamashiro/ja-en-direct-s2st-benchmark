import ast
import csv
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS

import numpy as np
import pytest

from direct_s2st.drive_staging import raw_sha
from direct_s2st.io import atomic_write_json, atomic_write_jsonl
from direct_s2st.preview_selection import common_rows, sample_ids
from direct_s2st.s2ut.preview_generate import selected_iterator

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'scripts/colab'))
import inference_session as session


def write(path, body):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding='utf-8')
    return path


def fixture_run(persistent, kind, data, *, suffix='', update=100):
    name = kind+'-training-original'+suffix
    directory = data/('translatotron2' if kind == 'tt2' else 's2ut')/'fairseq'
    command = ['{python}', '-m', 'module', '--data-root', str(directory)]
    if kind == 's2ut':
        command = ['{python}', '-m', 'module', str(directory)]
    if kind in ('unit', 'mel'):
        config = write(persistent/(kind+'-generator.json'), json.dumps({'sampling_rate':16000}))
        command = ['{python}', '-m', 'module', '--kind', kind, '--config', str(config)]
    cfg = dict(kind='vocoder' if kind in ('unit','mel') else kind, command=command)
    atomic_write_json(persistent/(name+'.json'), cfg)
    files = {'generator.pt' if kind in ('unit', 'mel') else 'checkpoints/checkpoint_last.pt': 'checkpoint bytes'}
    backup = persistent/'checkpoints'/name
    marker = backup/f'{update:012d}-fixture'
    checksums = {p: raw_sha(write(marker/'files'/p, body)) for p,body in files.items()}
    bound = {str(p): raw_sha(p) for p in data.rglob('*') if p.is_file() and p.suffix in ('.tsv','.jsonl')}
    if kind in ('unit','mel'):
        bound[str(config)] = raw_sha(config)
    identity = dict(config=cfg, work='/content/s2st-work/'+name,
                    inputs=bound | {'/content/s2st-runtime/environment.json': 'a'*64})
    state = dict(identity=identity, updates=update, files=checksums)
    atomic_write_json(marker/'snapshot.json', state)
    atomic_write_json(persistent/(name+'.launch.json'), dict(version=1, run_name=name, kind=kind,
                      work=identity['work'], build_revision='b'*40, editable_repo='primary', environment_sha256='a'*64))
    return name, marker, cfg


def test_actual_names_pointers_ambiguity_and_missing_snapshots(tmp_path):
    persistent, data = tmp_path/'drive', tmp_path/'data'
    persistent.mkdir()
    name, _, _ = fixture_run(persistent, 'tt2', data, suffix='-probe-fix-v1')
    atomic_write_json(persistent/'tt2-default.json', {'kind':'tt2'})
    assert session.choose(persistent, 'tt2') == name
    other, _, _ = fixture_run(persistent, 'tt2', data, suffix='-other')
    with pytest.raises(ValueError, match='select PREVIEW_RUN_NAMES'):
        session.choose(persistent, 'tt2')
    atomic_write_json(persistent/'last-training-run-tt2.json', dict(run_name=name))
    assert session.choose(persistent, 'tt2') == name
    assert session.choose(persistent, 'tt2', other) == other
    with pytest.raises(ValueError, match='no durable snapshot'):
        session.choose(persistent, 'tt2', 'tt2-default')
    with pytest.raises(ValueError, match='invalid preview run'):
        session.choose(persistent, 'tt2', '../escape')


def test_snapshot_restoration_is_verified_separate_and_reusable(tmp_path):
    persistent, data, local = tmp_path/'drive', tmp_path/'data', tmp_path/'preview'
    persistent.mkdir()
    name, marker, config = fixture_run(persistent, 'tt2', data, suffix='-probe-fix-v1')
    run = session.saved_run(persistent, 'tt2', '', local)
    restored = Path(run['work'])/'checkpoints/checkpoint_last.pt'
    assert restored.read_text() == 'checkpoint bytes'
    assert str(local) in str(restored) and Path(run['work']).name == name
    stamp = restored.stat().st_mtime_ns
    session.saved_run(persistent, 'tt2', '', local)
    assert restored.stat().st_mtime_ns == stamp
    assert (marker/'files/checkpoints/checkpoint_last.pt').read_text() == 'checkpoint bytes'
    config['command'].append('--changed')
    atomic_write_json(persistent/(name+'.json'), config, overwrite=True)
    with pytest.raises((ValueError, FileNotFoundError)):
        session.saved_run(persistent, 'tt2', '', local)


def test_calibrated_config_does_not_accept_arbitrary_saved_config(tmp_path):
    from direct_s2st.batch_calibration import selected_config
    cfg = dict(kind='vocoder', command=['python','--batch-size','1'], runtime={'adaptive_batch':False})
    atomic_write_json(tmp_path/'batch-calibration.json', dict(selected=8))
    session.verify_config(cfg, selected_config(cfg, 8), tmp_path)
    changed = selected_config(cfg, 8)
    changed['command'].append('--not-reviewed')
    with pytest.raises(ValueError, match='configuration'):
        session.verify_config(cfg, changed, tmp_path)


def test_prepare_copies_only_bounded_dev_audio_and_never_stages_training(tmp_path, monkeypatch):
    persistent, data, local = tmp_path/'drive', tmp_path/'data', tmp_path/'local'
    persistent.mkdir()
    rows = []
    for i in range(4):
        row = dict(pair_id=f'p{i}', split='dev', ja_text='日本語', en_text='English')
        for language in ('ja','en'):
            source = write(tmp_path/'corpus'/f'{i}-{language}.wav', f'fixture-{i}-{language}')
            row[language+'_audio'] = str(source)
            row[language+'_sha256'] = raw_sha(source)
        rows.append(row)
    atomic_write_jsonl(data/'common/dev.jsonl', rows)
    table = 'id\tsrc_audio\tsrc_n_frames\ttgt_audio\ttgt_n_frames\n'+''.join(
        f'{r["pair_id"]}\t{r["ja_audio"]}\t100\t1 2\t2\n' for r in rows)
    for system in ('translatotron2','s2ut'):
        write(data/system/'fairseq/dev.tsv', table)
    for kind in session.KINDS:
        fixture_run(persistent, kind, data, suffix='-actual')
    copied = []
    original = session.copy_verified
    def copy(source, target, expected=None):
        copied.append(str(source))
        assert '3-ja.wav' not in str(source) and '3-en.wav' not in str(source)
        return original(source, target, expected)
    monkeypatch.setattr(session, 'copy_verified', copy)
    plan = session.prepare(persistent, data, local, {}, 3, 'c'*40)
    assert plan['ids'] == ['p0','p1','p2'] and plan['split'] == 'dev'
    assert len(copied) == 6
    assert all(v['name'].endswith('-actual') for v in plan['runs'].values())
    assert len(json.loads((Path(plan['local'])/'input-map.json').read_text())['rows']) == 6
    copied.clear()
    assert session.prepare(persistent, data, local, {}, 3, 'c'*40) == plan
    assert copied == []
    # Tampered metadata is rejected before any fresh WAV acquisition.
    with (data/'common/dev.jsonl').open('a') as stream:
        stream.write('\n')
    with pytest.raises(ValueError, match='metadata changed'):
        session.prepare(persistent, data, local, {}, 3, 'c'*40)
    assert copied == []


def test_selection_resolves_only_requested_rows(tmp_path, monkeypatch):
    import direct_s2st.drive_staging as stage
    calls = []
    def resolve(row, language):
        calls.append((row['pair_id'], language))
        assert row['pair_id'] == 'b'
        return row
    monkeypatch.setattr(stage, 'read_common_manifest_row', resolve)
    path = tmp_path/'dev.jsonl'
    atomic_write_jsonl(path, [dict(pair_id='a'), dict(pair_id='b')])
    rows, full = common_rows(path, ['b'])
    assert full == {'a','b'} and rows == [dict(pair_id='b')]
    assert calls == [('b','ja'),('b','en')]
    selection = tmp_path/'ids.json'
    for value in ([], ['b']*2, list(map(str,range(11))), 'b'):
        atomic_write_json(selection, value, overwrite=True)
        with pytest.raises(ValueError, match='1..10'):
            sample_ids(selection)


def test_fairseq_preview_keeps_original_numeric_ids_and_restores_iterator():
    class Dataset:
        def __len__(self): return 20
        def ordered_indices(self): return np.arange(20)
    dataset = Dataset()
    task = NS(load_dataset=lambda name: None, dataset=lambda name: dataset)
    setup = lambda *a, **kw: task
    tasks = NS(setup_task=setup)
    with selected_iterator(tasks, [19, 2, 11], 20, 'dev'):
        configured = tasks.setup_task()
        configured.load_dataset('dev')
        assert list(configured.dataset('dev').ordered_indices()) == [19,2,11]
    assert tasks.setup_task is setup
    assert list(dataset.ordered_indices()) == list(range(20))
    with pytest.raises(ValueError, match='size differs'):
        with selected_iterator(tasks, [0], 19, 'dev'):
            tasks.setup_task().load_dataset('dev')
    assert tasks.setup_task is setup
    with pytest.raises(ValueError, match='invalid bounded'):
        with selected_iterator(tasks, [0], 20, 'train'):
            pass


def test_s2ut_preview_shard_restricts_generation_without_reindexing(tmp_path):
    from direct_s2st.s2ut.fairseq_infer import generate_shards
    ids = ['a','b','c','d']
    calls = []
    command = ['python','-m','fairseq_cli.generate','prepared','--results-path','unused']
    def runner(args, **kwargs):
        calls.append(args)
        assert args[2] == 'direct_s2st.s2ut.preview_generate'
        assert json.loads(args[args.index('--indices')+1]) == [3,1]
        assert args[args.index('--dataset-size')+1] == '4'
        directory = Path(args[args.index('--results-path')+1])
        write(directory/'generate-dev.txt', 'D-3\t-1\t1 2\nD-1\t-1\t3 4\n')
    values, _ = generate_shards(command, tmp_path/'gen', 'dev', ids, {'test':True},
                               selected_ids=['d','b'], runner=runner, resume=True)
    assert values == {'d':[1,2], 'b':[3,4]} and len(calls) == 1
    generate_shards(command, tmp_path/'gen', 'dev', ids, {'test':True},
                    selected_ids=['d','b'], runner=lambda *a, **kw: pytest.fail('should reuse'), resume=True)


@pytest.mark.parametrize('count', [1, 2, 3, 8, 10, 32, 33])
def test_pinned_fairseq_batching_keeps_all_single_example_inference_batches(count, monkeypatch):
    """Execute the actual pinned Python wrapper, including its drop-last branch.

    Only its Cython kernel is replaced with singleton batches, the exact result
    of max_sentences=1. Importing the entire old fairseq isn't Python-3.13 safe.
    """
    from direct_s2st.s2ut.fairseq_infer import generation_command
    source = ROOT/'third_party/fairseq/fairseq/data/data_utils.py'
    node = next(n for n in ast.parse(source.read_text(encoding='utf-8')).body
                if isinstance(n, ast.FunctionDef) and n.name == 'batch_by_size')
    def kernel(indices, fn, max_tokens, max_sentences, multiple):
        assert max_sentences == 1
        return [np.asarray([i]) for i in indices]
    monkeypatch.setitem(sys.modules, 'fairseq.data.data_utils_fast', NS(
        batch_by_size_fn=kernel, batch_by_size_vec=kernel, batch_fixed_shapes_fast=None))
    namespace = dict(np=np)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), namespace)
    batch = namespace['batch_by_size']
    indices = np.arange(count)
    # Reproduce the reported missing final sample, without touching corpus/GPU.
    old = batch(indices, lambda i: 100, max_sentences=1, required_batch_size_multiple=8)
    assert [int(v) for b in old for v in b] == list(range(count-1))
    for split in ('dev', 'test'):
        command = generation_command('prepared', 'best.pt', 'generated', split, 10, 'cuda')
        multiple = int(command[command.index('--required-batch-size-multiple')+1])
        assert command[command.index('--batch-size')+1] == '1' and multiple == 1
        fixed = batch(indices, lambda i: 100, max_sentences=1, required_batch_size_multiple=multiple)
        assert [int(v) for b in fixed for v in b] == list(range(count))
    assert '--cpu' in generation_command('prepared', 'best.pt', 'generated', 'dev', 10, 'cpu')


def test_preview_command_forwards_complete_batch_policy_and_failure_lists_missing_ids(tmp_path):
    from direct_s2st.s2ut.fairseq_infer import generate_shards, generation_command
    command = generation_command('prepared', 'best.pt', tmp_path/'unused', 'dev', 10, 'cpu')
    def runner(args, **kwargs):
        assert args[2] == 'direct_s2st.s2ut.preview_generate'
        assert args[args.index('--required-batch-size-multiple')+1] == '1'
        assert args[args.index('--distributed-world-size')+1] == '1'
        directory = Path(args[args.index('--results-path')+1])
        write(directory/'generate-dev.txt', 'D-0\t-1\t1 2\nD-2\t-1\t3 4\n')
    with pytest.raises(ValueError, match=r"expected=3 generated=2 missing=\['b'\]"):
        generate_shards(command, tmp_path/'gen', 'dev', ['a','b','c'], {},
                        selected_ids=['a','b','c'], runner=runner, resume=True)


def test_fixed_generation_does_not_reuse_partial_legacy_shard_cache(tmp_path):
    from direct_s2st.s2ut.fairseq_infer import generate_shards, generation_command
    from direct_s2st.preparation import Checkpoints
    generated, identity = tmp_path/'gen', {'checkpoint':'same'}
    with Checkpoints(generated/'.checkpoints', dict(stage='generation-v1', inputs=identity, num_shards=1), resume=True) as old:
        old.record('0', dict(records={'a':[1,2]}, seconds=1, log_sha256='old'))
        old.flush()
        old_path = old.root
    calls = []
    def runner(args, **kwargs):
        calls.append(args)
        directory = Path(args[args.index('--results-path')+1])
        write(directory/'generate-dev.txt', 'D-0\t-1\t1 2\nD-1\t-1\t3 4\n')
    command = generation_command('prepared', 'best.pt', 'unused', 'dev', 10, 'cpu')
    values, _ = generate_shards(command, generated, 'dev', ['a','b'], identity, runner=runner, resume=True)
    assert values == {'a':[1,2], 'b':[3,4]} and len(calls) == 1
    assert old_path.is_dir()  # retained for inspection, never overwritten/deleted


def test_publish_keeps_failures_and_separates_preview_from_test_results(tmp_path):
    from direct_s2st.preview import publish
    local, output = tmp_path/'local', tmp_path/'drive/preview/identity'
    row = dict(pair_id='pair', ja_text='日本語', en_text='English')
    for language in ('ja','en'):
        source = write(local/(language+'.wav'), language)
        row[language+'_local'], row[language+'_sha256'] = str(source), raw_sha(source)
    for label in ('mel-reference','unit-reference','mel-translated','unit-translated'):
        source = write(local/label/'audio.wav', label)
        prediction = dict(pair_id='pair', status='success', output_audio=str(source), output_sha256=raw_sha(source))
        if label == 'mel-translated':
            prediction = dict(pair_id='pair', status='failed', error='phone EOS not reached')
        atomic_write_jsonl(local/label/'predictions.jsonl', [prediction])
    plan = dict(local=str(local), output=str(output), purpose='dev_listening_only_not_benchmark',
                execution_revision='a'*40, rows=[row], ids=['pair'],
                runs={'tt2':dict(name='actual-probe-fix-v1',updates=580,snapshot='retained')})
    result = publish(plan)
    assert result['split'] == 'dev' and len(result['samples']) == 1
    assert result['samples'][0]['audio']['mel-translated'] == dict(status='failed', error='phone EOS not reached')
    assert len(list((output/'audio').rglob('*.wav'))) == 5
    assert not (tmp_path/'drive/results').exists()
    assert publish(plan) == result


def test_unit_reference_uses_saved_reduced_units_and_best_generator(tmp_path, monkeypatch):
    import direct_s2st.preview as preview
    import direct_s2st.vocoders.inference as inference
    import direct_s2st.vocoders.monitor as monitor
    root = tmp_path/'local'
    monkeypatch.setattr(session.os, 'environ', dict(session.os.environ))
    from direct_s2st.journal import digest
    atomic_write_json(root/'sample-ids.json', ['p'])
    atomic_write_json(root/'input-map.json', dict(rows={}, sha256=digest({})))
    config = tmp_path/'unit.json'
    atomic_write_json(config, dict(sampling_rate=16000))
    table = tmp_path/'prepared/dev.tsv'
    write(table, 'id\tsrc_audio\tsrc_n_frames\ttgt_audio\ttgt_n_frames\np\tmissing.wav\t100\t3 5 2\t3\n')
    plan = dict(purpose='dev_listening_only_not_benchmark', split='dev', ids=['p'], local=str(root),
                common_sha256='common-sha', rows=[dict(pair_id='p', en_text='reference')],
                runs=dict(unit=dict(work=str(tmp_path/'trained-unit'), name='unit-real',
                          config={'command':['python','--config',str(config)]}),
                          s2ut=dict(data_root=str(table.parent))))
    best = tmp_path/'generator_best.pt'
    monkeypatch.setattr(monitor, 'select_generator', lambda work: best)
    calls = []
    monkeypatch.setattr(inference, 'vocode', lambda *a, **kw: calls.append((a,kw)))
    preview.worker(plan, 'unit')
    assert len(calls) == 2
    assert all(a[3] == best and kw['duration_prediction'] is True for a,kw in calls)
    records = list(preview.read_jsonl(root/'unit-reference-input.jsonl'))
    assert records[0]['units'] == [3,5,2]
    assert records[0]['reference_reconstruction'] is True and records[0]['duration_mode'] == 'predicted'
    assert not (root/'mel-reference-input.jsonl').exists()


def test_mel_reference_extracts_only_selected_english_and_uses_saved_frontend(tmp_path, monkeypatch):
    import direct_s2st.preview as preview
    import direct_s2st.vocoders.inference as inference
    import direct_s2st.vocoders.monitor as monitor
    import direct_s2st.translatotron2.prepare_fairseq as preparation
    root = tmp_path/'local'
    monkeypatch.setattr(session.os, 'environ', dict(session.os.environ))
    from direct_s2st.journal import digest
    atomic_write_json(root/'sample-ids.json', ['p'])
    atomic_write_json(root/'input-map.json', dict(rows={}, sha256=digest({})))
    spec = json.loads((ROOT/'configs/vocoder/mel-generator.json').read_text())['mel']
    config = tmp_path/'mel.json'
    atomic_write_json(config, dict(sampling_rate=16000, mel=spec))
    prepared = tmp_path/'prepared'
    atomic_write_json(prepared/'mel-spec.json', spec)
    plan = dict(purpose='dev_listening_only_not_benchmark', split='dev', ids=['p'], local=str(root),
                common_sha256='common-sha', rows=[dict(pair_id='p', en_text='reference', en_local='only-selected-en.wav')],
                runs=dict(mel=dict(work=str(tmp_path/'trained-mel'), name='mel-real',
                          config={'command':['python','--config',str(config)]}), tt2=dict(data_root=str(prepared))))
    monkeypatch.setattr(monitor, 'select_generator', lambda work: tmp_path/'generator_best.pt')
    calls, extracted = [], []
    def extract(source, target, settings):
        extracted.append((source,settings))
        np.save(target, np.ones((20,spec['n_mels']), dtype=np.float32))
    monkeypatch.setattr(preparation, '_extract_logmel_official', extract)
    monkeypatch.setattr(inference, 'vocode', lambda *a, **kw: calls.append((a,kw)))
    preview.worker(plan, 'mel')
    assert extracted == [(Path('only-selected-en.wav'), spec)]
    assert len(calls) == 2 and all(kw['mel'] == spec for _,kw in calls)
    record = list(preview.read_jsonl(root/'mel-reference-input.jsonl'))[0]
    assert record['status'] == 'success' and record['mel_sha256'] == raw_sha(Path(record['mel_path']))
    preview.worker(plan, 'mel')
    assert len(extracted) == 1  # resume does not recompute valid reference Mel


def test_environment_restores_original_build_and_overlays_only_inference(tmp_path, monkeypatch):
    import environment_recovery
    import training_session
    repo, persistent = tmp_path/'repo', tmp_path/'drive'
    run = dict(route=dict(build_revision='b'*40, editable_repo='primary'),
               environment_sha256='a'*64, config={'kind':'tt2'})
    calls = []
    monkeypatch.setattr(training_session, 'git', lambda repo,*args: 'b'*40)
    def environment(build, drive, invoke, *, expected):
        assert build == repo and drive == persistent and expected == 'a'*64
        calls.append(expected)
        return Path('/runtime/python')
    monkeypatch.setattr(environment_recovery, 'ensure_environment', environment)
    monkeypatch.setattr(session.os, 'environ', dict(session.os.environ))
    python = session.environment(repo, persistent, run, lambda *a: pytest.fail('should not checkout same revision'))
    assert python == Path('/runtime/python') and calls == ['a'*64]
    assert str(ROOT/'src') in session.os.environ['PYTHONPATH']
    assert session.os.environ['S2ST_CONFIG_ROOT'] == str(ROOT/'configs')


def test_cell7_processes_all_kinds_sequentially_without_training_or_full_staging(tmp_path, monkeypatch):
    import direct_s2st.preview as preview
    persistent, local = tmp_path/'drive', tmp_path/'local'
    common = tmp_path/'common/dev.jsonl'
    write(common, '{}\n')
    write(local/'input-map.json', '{}')
    plan = dict(execution_revision='c'*40, common_root=str(common.parent), common_sha256=raw_sha(common),
                local=str(local), runs={})
    for kind in session.KINDS:
        checkpoint = write(local/'models'/kind/'model.pt', 'verified')
        plan['runs'][kind] = dict(work=str(checkpoint.parent), files={'model.pt':raw_sha(checkpoint)})
    path = persistent/'preview/id/plan.json'
    atomic_write_json(path, plan)
    atomic_write_json(persistent/'preview/latest.json', dict(plan=str(path), sha256=raw_sha(path)))
    monkeypatch.setattr(session.subprocess, 'check_output', lambda *a, **kw: 'c'*40)
    kinds = []
    monkeypatch.setattr(session, 'environment', lambda repo, drive, run, invoke: Path('runtime-python'))
    monkeypatch.setattr(session.os, 'environ', dict(session.os.environ))
    def invoke(*args):
        assert args[:3] == (Path('runtime-python'), '-m', 'direct_s2st.preview')
        kinds.append(args[-1])
    published = []
    monkeypatch.setattr(preview, 'publish', lambda plan: published.append(plan))
    session.execute(tmp_path/'repo', persistent, invoke)
    assert kinds == list(session.KINDS) and published == [plan]


def test_normal_cells_are_opt_in_bounded_and_full_test_guards_remain():
    notebook = json.loads((ROOT/'notebooks/colab_training.ipynb').read_text(encoding='utf-8'))
    codes = [''.join(c['source']) for c in notebook['cells'] if c['cell_type'] == 'code']
    for code in codes:
        ast.parse(code)
    namespace = {}
    exec(codes[0], namespace)
    assert namespace['RUN_PREVIEW'] is False and namespace['PREVIEW_SAMPLES'] == 3
    assert namespace['RUN_TRAINING'] is False and namespace['RUN_BENCHMARK'] is False
    cell6 = next(s for s in codes if s.startswith('#@title 6.'))
    cell7 = next(s for s in codes if s.startswith('#@title 7.'))
    cell8 = next(s for s in codes if s.startswith('#@title 8.'))
    assert "'--action', 'prepare'" in cell6 and 'PREVIEW_RUN_NAMES' in cell6
    assert "'--action', 'run'" in cell7 and 'display(Audio' in cell7
    assert 'all common/test IDs' in cell6 and "elif RUN_BENCHMARK:" in cell8
    assert '正式評価に使用しません' in cell8
