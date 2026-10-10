import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from direct_s2st.batch_calibration import calibrate, candidates, selected_config, stress_fairseq_iterator, trial
from direct_s2st.batch_probe import Probe, ProbeComplete
from direct_s2st.recipes import option


HW = dict(name='fixture GPU', total_memory=100, capability=[8, 0], torch='fixture', cuda='fixture')


def config(kind='tt2'):
    module = {'tt2':'direct_s2st.translatotron2.train', 's2ut':'direct_s2st.s2ut.fairseq_train',
              'vocoder':'direct_s2st.vocoders.train'}[kind]
    return dict(kind=kind, command=['{python}', '-m', module, '--batch-size', '8',
                                   '--update-freq', '128', '--max-updates', '{updates}', '--run-root', '{run_root}'],
                runtime=dict(adaptive_batch=False, precision='default', cache_gb=0, readers=1),
                research_metadata=dict(effective_batch_size=1024, physical_batch_size=8, update_freq=128),
                batch_calibration=dict(max_batch=64, steps=3, reserve_ratio=.1, trial_seconds=60))


def measured(size, *, used=None):
    return dict(status='ok', steps=[dict(seconds=.1, samples=size, allocated=size,
                                        reserved=size, used=size if used is None else used, total=100)]*3)


def exact_s2ut_config():
    from direct_s2st.config import load_config
    root = Path(__file__).resolve().parents[2]
    cfg = config('s2ut')
    cfg['command'] = [v.format(data_root='fixture-data', run_root='{run_root}',
                                max_updates='{updates}', save_interval_updates='1', seed='1')
                      for v in load_config(root/'configs/s2ut/train.yaml')['training']['command']]
    cfg['command'][0] = '{python}'
    cfg['research_metadata'] = dict(reproduction_mode='paper_exact', max_tokens=20000, update_freq=4)
    return cfg


def test_tt2_candidates_and_selection_preserve_all_1024_samples():
    cfg = config()
    assert candidates(cfg) == [1, 2, 4, 8, 16, 32, 64]
    for size in candidates(cfg):
        selected = selected_config(cfg, size)
        assert int(option(selected['command'], '--batch-size')) * int(option(selected['command'], '--update-freq')) == 1024
        assert selected['research_metadata']['effective_batch_size'] == 1024
    assert option(cfg['command'], '--batch-size') == '8'
    with pytest.raises(ValueError, match='divide'):
        selected_config(cfg, 3)


def test_s2ut_practical_keeps_logical_token_batches_but_adds_physical_cap():
    cfg = config('s2ut')
    cfg['command'] = ['{python}', '-m', 'fixture', '--max-tokens', '20000', '--update-freq', '4']
    selected = selected_config(cfg, 32)
    assert selected['command'] == cfg['command']
    assert selected['runtime']['fixed_microbatch'] == 32
    assert selected['research_metadata']['physical_batch_cap'] == 32


def test_s2ut_exact_only_tests_unmodified_official_batches(tmp_path):
    cfg, calls = exact_s2ut_config(), []
    cfg['batch_calibration']['max_batch'] = 1  # Must not cap exact batches at one.
    def measure(c, size, parent):
        calls.append(size)
        selected = selected_config(c, size)
        assert selected['command'] == cfg['command']
        assert selected['runtime']['fixed_microbatch'] == 0
        return measured(24, used=80)  # Actual batch size, not sentinel zero.
    path = tmp_path/'record'
    selected = calibrate(cfg, {}, path, tmp_path/'local', measure=measure, get_hardware=lambda: HW)
    assert candidates(cfg) == [0] and calls == [0]
    assert selected['command'] == cfg['command']
    assert selected['research_metadata']['physical_batch_cap'] is None
    assert 'fit check only' in selected['research_metadata']['batch_policy']
    record = json.loads(path.read_text())
    assert record['selected'] == 0 and record['policy'] == 'official_batch_fit_check'
    fail = lambda *a: pytest.fail('restore/completed reuse must not access GPU')
    assert calibrate(cfg, {}, path, tmp_path/'local', inspect_only=True,
                     measure=fail, get_hardware=fail) == selected
    assert calibrate(cfg, {}, path, tmp_path/'local', measure=measure, get_hardware=lambda: HW) == selected
    assert calls == [0]
    calibrate(cfg, {}, path, tmp_path/'local', measure=measure, get_hardware=lambda: dict(HW, name='new GPU'))
    assert calls == [0, 0]  # Recheck official batches, not a capacity search.


@pytest.mark.parametrize('result', [dict(status='oom'), measured(24, used=95)])
def test_s2ut_exact_insufficient_vram_stops_without_smaller_batch(tmp_path, result):
    calls = []
    def measure(c, size, parent):
        calls.append(size)
        return result
    with pytest.raises(RuntimeError, match='No batch was changed'):
        calibrate(exact_s2ut_config(), {}, tmp_path/'record', tmp_path/'local',
                  measure=measure, get_hardware=lambda: HW)
    assert calls == [0]
    record = json.loads((tmp_path/'record').read_text())
    assert record['selected'] is None and len(record['trials']) == 1


def test_s2ut_exact_rejects_caps_mode_mismatch_and_old_calibration(tmp_path):
    from direct_s2st.journal import digest
    cfg = exact_s2ut_config()
    with pytest.raises(ValueError, match='only checks official'):
        selected_config(cfg, 32)
    capped = copy.deepcopy(cfg)
    capped['runtime']['fixed_microbatch'] = 32
    with pytest.raises(ValueError, match='forbids'):
        selected_config(capped, 0)
    mismatch = copy.deepcopy(cfg)
    mismatch['research_metadata']['reproduction_mode'] = 'paper_practical'
    with pytest.raises(ValueError, match='differs'):
        candidates(mismatch)
    old = dict(request=digest(dict(config=cfg, inputs={}, algorithm='startup-batch-v1')), selected=32)
    path = tmp_path/'record'
    path.write_text(json.dumps(old))
    with pytest.raises(ValueError, match='new run name'):
        calibrate(cfg, {}, path, tmp_path/'local', inspect_only=True)
    assert json.loads(path.read_text()) == old


def test_s2ut_exact_trial_overrides_inherited_cap_with_zero(tmp_path, monkeypatch):
    import subprocess
    monkeypatch.setenv('S2ST_TRAIN_FIXED_MICROBATCH', '32')
    cfg = exact_s2ut_config()
    def run(args, *, env, **kw):
        assert env['S2ST_TRAIN_FIXED_MICROBATCH'] == '0'
        assert env['S2ST_TRAIN_ADAPTIVE_BATCH'] == '0'
        assert option(args, '--max-tokens') == '20000'
        assert option(args, '--update-freq') == '4' and '--fp16' in args
        Path(env['S2ST_BATCH_PROBE']).write_text(json.dumps(measured(24)))
        return subprocess.CompletedProcess(args, 0)
    monkeypatch.setattr(subprocess, 'run', run)
    assert trial(cfg, 0, tmp_path)['status'] == 'ok'


def test_uncapped_s2ut_passes_original_batch_objects_and_rejects_capped_resume(monkeypatch):
    from direct_s2st.train_runtime import Microbatches
    monkeypatch.setenv('S2ST_TRAIN_FIXED_MICROBATCH', '0')
    monkeypatch.setenv('S2ST_TRAIN_ADAPTIVE_BATCH', '0')
    batches = [dict(nsentences=24, ntokens=20000)]*4
    assert Microbatches().run(None, None, batches, lambda values: values) is batches
    with pytest.raises(ValueError, match='saved fixed microbatch'):
        Microbatches(dict(fixed=32, hardware=('other GPU', 0)))


@pytest.mark.parametrize('kind', ['tt2', 's2ut', 'vocoder'])
def test_shared_calibration_selects_largest_safe_batch_and_records_each_trial(tmp_path, kind):
    cfg, seen = config(kind), []
    def measure(cfg, size, parent):
        seen.append(size)
        return dict(status='oom') if size > 16 else measured(size)
    path = tmp_path/'drive/batch-calibration.json'
    selected = calibrate(cfg, {'fixture':'hash'}, path, tmp_path/'local', measure=measure, get_hardware=lambda: HW)
    assert seen == ([1, 2, 4, 8, 16, 32] if kind == 'tt2' else [1, 2, 4, 8, 16, 32, 24, 20, 18, 17])
    record = json.loads(path.read_text())
    assert record['selected'] == 16 and record['weights_reused'] is False
    assert len(record['trials']) == len(seen)
    assert selected['runtime']['adaptive_batch'] is False
    seen.clear()
    assert calibrate(cfg, {'fixture':'hash'}, path, tmp_path/'local', measure=measure, get_hardware=lambda: HW) == selected
    assert not seen


def test_vram_reserve_includes_allocator_peak(tmp_path):
    seen = []
    def measure(cfg, size, parent):
        seen.append(size)
        return measured(size, used=95 if size == 8 else 80)
    result = calibrate(config(), {}, tmp_path/'record', tmp_path/'local', measure=measure, get_hardware=lambda: HW)
    assert option(result['command'], '--batch-size') == '4'
    assert seen == [1, 2, 4, 8]


def test_interrupted_calibration_reuses_completed_trials(tmp_path):
    cfg, calls = config(), []
    path = tmp_path/'record'
    def interrupted(cfg, size, parent):
        calls.append(size)
        if size == 8:
            raise KeyboardInterrupt()
        return measured(size)
    with pytest.raises(KeyboardInterrupt):
        calibrate(cfg, {}, path, tmp_path/'local', measure=interrupted, get_hardware=lambda: HW)
    assert json.loads(path.read_text())['selected'] is None
    calls.clear()
    def measure(cfg, size, parent):
        calls.append(size)
        return measured(size) if size == 8 else dict(status='oom')
    calibrate(cfg, {}, path, tmp_path/'local', measure=measure, get_hardware=lambda: HW)
    assert calls == [8, 16]


def test_non_oom_errors_are_not_mistaken_for_capacity_limits(tmp_path):
    def broken(*args):
        raise ValueError('missing phoneme dictionary')
    with pytest.raises(ValueError, match='phoneme'):
        calibrate(config(), {}, tmp_path/'record', tmp_path/'local', measure=broken, get_hardware=lambda: HW)
    assert json.loads((tmp_path/'record').read_text())['selected'] is None


def test_changed_data_or_settings_refuse_reuse(tmp_path):
    cfg, path = config(), tmp_path/'record'
    calibrate(cfg, {'input':'one'}, path, tmp_path/'local', measure=lambda c,s,p: measured(s), get_hardware=lambda: HW)
    with pytest.raises(ValueError, match='changed'):
        calibrate(cfg, {'input':'two'}, path, tmp_path/'local', inspect_only=True)
    changed = copy.deepcopy(cfg)
    changed['batch_calibration']['reserve_ratio'] = .2
    with pytest.raises(ValueError, match='changed'):
        calibrate(changed, {'input':'one'}, path, tmp_path/'local', inspect_only=True)


def test_gpu_switch_tests_same_saved_batch_without_retuning(tmp_path):
    cfg, path = config(), tmp_path/'record'
    calibrate(cfg, {}, path, tmp_path/'local', measure=lambda c,s,p: measured(s), get_hardware=lambda: HW)
    calls = []
    changed_gpu = dict(HW, name='another GPU')
    def measure(c,s,p):
        calls.append(s)
        return measured(s)
    result = calibrate(cfg, {}, path, tmp_path/'local', measure=measure, get_hardware=lambda: changed_gpu)
    assert calls == [64] and option(result['command'], '--batch-size') == '64'
    with pytest.raises(ValueError, match='unsafe'):
        calibrate(cfg, {}, path, tmp_path/'local', measure=lambda c,s,p: dict(status='oom'),
                  get_hardware=lambda: dict(HW, name='small GPU'))
    assert json.loads(path.read_text())['selected'] == 64


def test_restore_only_never_requires_gpu_or_executes_probes(tmp_path):
    path = tmp_path/'record'
    calibrate(config(), {}, path, tmp_path/'local', measure=lambda c,s,p: measured(s), get_hardware=lambda: HW)
    fail = lambda *args: pytest.fail('no GPU access on restore-only')
    assert option(calibrate(config(), {}, path, tmp_path/'local', inspect_only=True,
                            measure=fail, get_hardware=fail)['command'], '--batch-size') == '64'


def test_fixed_s2ut_chunks_preserve_order_and_never_retry_oom(monkeypatch):
    from direct_s2st.train_runtime import Microbatches
    monkeypatch.setenv('S2ST_TRAIN_FIXED_MICROBATCH', '2')
    monkeypatch.setenv('S2ST_TRAIN_ADAPTIVE_BATCH', '0')
    batch = dict(id=torch.arange(5), nsentences=5, target_lengths=torch.tensor([2]*5), ntokens=10,
                 net_input=dict(src_tokens=torch.ones(5, 12)))
    chunks = Microbatches().run(None, None, [batch], lambda values: values)
    assert [c['nsentences'] for c in chunks] == [2, 2, 1]
    assert torch.cat([c['id'] for c in chunks]).tolist() == list(range(5))
    assert sum(c['ntokens'] for c in chunks) == 10
    calls = []
    def oom(values):
        calls.append(values)
        raise torch.cuda.OutOfMemoryError('synthetic')
    with pytest.raises(torch.cuda.OutOfMemoryError):
        Microbatches().run(None, None, [batch], oom)
    assert len(calls) == 1


def test_probe_measures_optimizer_allocations_and_stops_before_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setenv('S2ST_BATCH_PROBE', str(tmp_path/'measurement.json'))
    monkeypatch.setenv('S2ST_BATCH_PROBE_STEPS', '3')
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda: None)
    monkeypatch.setattr(torch.cuda, 'reset_peak_memory_stats', lambda: None)
    monkeypatch.setattr(torch.cuda, 'mem_get_info', lambda: (60, 100))
    monkeypatch.setattr(torch.cuda, 'memory_reserved', lambda: 20)
    monkeypatch.setattr(torch.cuda, 'max_memory_allocated', lambda: 50)
    monkeypatch.setattr(torch.cuda, 'max_memory_reserved', lambda: 60)
    observer = Probe()
    for _ in range(2):
        observer.begin()
        observer.finish(8)
    observer.begin()
    with pytest.raises(ProbeComplete):
        observer.finish(8)
    rows = json.loads((tmp_path/'measurement.json').read_text())['steps']
    assert len(rows) == 3 and all(r['used'] == 80 for r in rows)


def test_stress_iterator_only_reorders_disposable_train_batches():
    class Dataset:
        def __len__(self):
            return 4
        def size(self, i):
            value = [100, 10, 20, 30][i]
            return value, value
    dataset = Dataset()
    itr = SimpleNamespace(dataset=dataset, frozen_batches=[[0], [1,2,3]])
    stress_fairseq_iterator(itr, 3, 4)
    assert len(itr._frozen_batches) == 12 and itr.disable_shuffling
    assert itr._frozen_batches[:4] == ([0],)*4
    assert itr._frozen_batches[8:] == ([1,2,3],)*4


def test_trial_fresh_child_isolated_output_no_resume_or_publication(tmp_path, monkeypatch):
    import subprocess
    cfg = config()
    monkeypatch.setenv('S2ST_TRAIN_STAGING', 'never-use-real-backups')
    monkeypatch.setenv('S2ST_TRAIN_DEADLINE', '1')
    def run(args, *, env, check, timeout):
        assert args[1:3] == ['-m', 'direct_s2st.batch_probe']
        assert '--restore-file' not in args and '--resume' not in args
        assert 'S2ST_TRAIN_STAGING' not in env and 'S2ST_TRAIN_DEADLINE' not in env
        assert env['S2ST_TRAIN_ADAPTIVE_BATCH'] == '0'
        path = Path(env['S2ST_BATCH_PROBE'])
        path.write_text(json.dumps(measured(16)))
        return subprocess.CompletedProcess(args, 0)
    monkeypatch.setattr(subprocess, 'run', run)
    assert trial(cfg, 16, tmp_path)['status'] == 'ok'
    assert not list(tmp_path.glob('.batch-probe-*'))


def test_probe_cannot_write_into_real_training_run(tmp_path, monkeypatch):
    import subprocess
    cfg = config()
    cfg['command'][cfg['command'].index('--run-root')+1] = str(tmp_path/'real-training')
    monkeypatch.setattr(subprocess, 'run', lambda *a, **kw:pytest.fail('unsafe trainer must never launch'))
    with pytest.raises(ValueError, match='private output'):
        trial(cfg, 16, tmp_path)


def test_notebook_enables_shared_startup_calibration_and_compiles():
    root = Path(__file__).resolve().parents[2]
    notebook = json.loads((root/'notebooks/colab_training.ipynb').read_text(encoding='utf-8'))
    source = '\n'.join(''.join(c['source']) for c in notebook['cells'])
    assert 'TRAIN_CALIBRATE_BATCH = True' in source
    assert 'calibrate_batch=TRAIN_CALIBRATE_BATCH' in source
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            compile(''.join(cell['source']), 'notebook', 'exec')


@pytest.mark.parametrize('kind', ['tt2', 's2ut'])
def test_calibrated_session_starts_at_zero_and_restores_fixed_selection(tmp_path, monkeypatch, kind):
    from direct_s2st import batch_calibration as calibration
    from direct_s2st import session_runtime
    from direct_s2st.colab import run_session
    original = calibration.calibrate
    trials, launches = [], []
    def measure(cfg, size, parent):
        trials.append(size)
        return measured(size or 24) if size <= 16 else dict(status='oom')
    monkeypatch.setattr(calibration, 'hardware', lambda: HW)
    monkeypatch.setattr(calibration, 'calibrate', lambda *a, **kw:
                        original(*a, **kw, measure=measure, get_hardware=lambda: HW))
    cfg = config() if kind == 'tt2' else exact_s2ut_config()
    lock = tmp_path/'data-lock'
    lock.write_text('immutable inputs')
    cfg.update(identity_files=[str(lock)], checkpoint='checkpoint.pt', resume_args=['--restore-file', '{checkpoint}'])
    def continuous(config, *, work, backup, identity, completed, total, publish, **kwargs):
        physical = int(option(config['command'], '--batch-size')) if kind == 'tt2' else config['runtime']['fixed_microbatch']
        launches.append((completed, physical))
        assert (work/'batch-calibration.json').is_file()
        assert int(option(config['command'], '--update-freq')) == (64 if kind == 'tt2' else 4)
        if kind == 's2ut':
            assert option(config['command'], '--max-tokens') == '20000'
            assert '--fp16' in config['command'] and physical == 0
        (work/'checkpoint.pt').write_text(str(total))
        publish(work, backup, identity, total)
        return dict(status='COMPLETE', durable_updates=total)
    monkeypatch.setattr(session_runtime, 'run_continuous', continuous)
    kwargs = dict(work=tmp_path/'work', backup=tmp_path/'drive', auto_resume=True,
                  inspect_checkpoint=lambda p,k:int(p.read_text()))
    assert run_session(cfg, total=1, **kwargs)['durable_updates'] == 1
    physical = 16 if kind == 'tt2' else 0
    assert launches == [(0, physical)]  # No probe weights/updates reused.
    assert trials == ([1,2,4,8,16,32] if kind == 'tt2' else [0])
    trials.clear()
    assert run_session(cfg, total=2, **kwargs)['durable_updates'] == 2
    assert launches[-1] == (1, physical) and not trials
    monkeypatch.setattr(calibration, 'hardware', lambda:pytest.fail('finished runs require no GPU'))
    assert run_session(cfg, total=2, **kwargs)['durable_updates'] == 2
    assert len(launches) == 2


def test_config_startup_calibration_is_explicit_and_not_online_adaptation(tmp_path):
    from direct_s2st.colab import make_config
    root = Path(__file__).resolve().parents[2]
    data = tmp_path/'data/translatotron2/fairseq'
    data.mkdir(parents=True)
    (data/'fixture').write_text('prepared')
    env = tmp_path/'environment.json'
    env.write_text('{}')
    cfg = make_config(root, tmp_path/'data', env, optimize=True, calibrate_batch=True)
    assert cfg['batch_calibration']['max_batch'] == 1024
    assert not cfg['runtime']['adaptive_batch']
    with pytest.raises(ValueError, match='requires'):
        make_config(root, tmp_path/'data', env, calibrate_batch=True)
    with pytest.raises(ValueError, match='requires'):
        make_config(root, tmp_path/'data', env, optimize=True, calibrate_batch=True, adaptive_batch=True)
    for options in [dict(calibration_reserve_ratio=0), dict(calibration_steps=1), dict(calibration_max_batch=0)]:
        with pytest.raises(ValueError, match='invalid startup'):
            make_config(root, tmp_path/'data', env, optimize=True, calibrate_batch=True, **options)


def test_non_power_of_two_capacity_is_measured_not_rounded_down(tmp_path):
    cfg = config('vocoder')
    cfg['batch_calibration']['max_batch'] = 50
    assert candidates(cfg)[-1] == 50
    calls = []
    def measure(c,s,p):
        calls.append(s)
        return measured(s) if s <= 37 else dict(status='oom')
    selected = calibrate(cfg, {}, tmp_path/'record', tmp_path/'local', measure=measure, get_hardware=lambda: HW)
    assert option(selected['command'], '--batch-size') == '37'
    assert 37 in calls and 38 in calls


def test_existing_training_does_not_get_silently_restarted_for_calibration(tmp_path):
    from direct_s2st.colab import run_session
    cfg = config()
    lock = tmp_path/'data-lock'
    lock.write_text('inputs')
    cfg.update(checkpoint='checkpoint.pt', identity_files=[str(lock)])
    work = tmp_path/'work'
    work.mkdir()
    (work/'checkpoint.pt').write_text('legacy training')
    with pytest.raises(ValueError, match='new run name'):
        run_session(cfg, work=work, backup=tmp_path/'drive', auto_resume=True)
    assert (work/'checkpoint.pt').read_text() == 'legacy training'


def test_frozen_snapshot_contains_batch_selection_and_recipe_resume_locks(tmp_path, monkeypatch):
    from direct_s2st.train_runtime import checkpoint_saved
    work, staging = tmp_path/'work', tmp_path/'staging'
    work.mkdir()
    staging.mkdir()
    (work/'checkpoint.pt').write_text('optimizer update 1')
    names = ['batch-calibration.json', 'research-metadata.json',
             'session-research-metadata.json', 's2ut-recipe-lock.json']
    for name in names:
        (work/name).write_text('immutable ' + name)
    monkeypatch.setenv('S2ST_TRAIN_STAGING', str(staging))
    checkpoint_saved(work, work/'checkpoint.pt', 1, force=True)
    frozen = next(staging.glob('*/files'))
    for name in names:
        assert (frozen/name).read_text() == 'immutable ' + name
        (work/name).write_text('subsequent update')
        assert (frozen/name).read_text() == 'immutable ' + name
