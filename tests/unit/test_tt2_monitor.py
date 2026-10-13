import json
from pathlib import Path
import xml.etree.ElementTree as ET

import pytest

from direct_s2st.translatotron2.monitor import Monitor, early_stop_result, validate_policy
from direct_s2st.drive_staging import raw_sha


POLICY = dict(interval=1, patience=2, min_relative_improvement=.01)


def losses(value):
    return dict(loss=value, mel_loss=value, phone_loss=value, duration_loss=value)


def bind_best(monitor, work):
    monitor.save_best(work, lambda p: (p.parent.mkdir(parents=True, exist_ok=True), p.write_text('best')))


def test_warmup_guard_patience_counts_evaluations_not_updates(tmp_path):
    monitor = Monitor(dict(POLICY, interval=2), warmup=4)
    for update in range(1, 9):
        improved = monitor.observe(update, losses(1), losses(1) if update % 2 == 0 else None, .001)
        if improved:
            bind_best(monitor, tmp_path)
        assert monitor.stopped == (update == 8)
        if update <= 4:
            assert monitor.bad_evaluations == 0
    assert monitor.best_update == 2
    assert monitor.bad_evaluations == 2


def test_small_improvements_keep_best_but_do_not_reset_patience(tmp_path):
    monitor = Monitor(POLICY, 0)
    for update, value in enumerate([1, .999, .998], 1):
        assert monitor.observe(update, losses(1), losses(value), .001)
        bind_best(monitor, tmp_path)
    assert monitor.stopped and monitor.best_update == 3
    reset = Monitor(POLICY, 0)
    for update, value in enumerate([1, .99, .98], 1):
        if reset.observe(update, losses(1), losses(value), .001):
            bind_best(reset, tmp_path)
    assert not reset.stopped and reset.bad_evaluations == 0


def test_history_counters_and_best_bytes_restore_without_reset(tmp_path):
    monitor = Monitor(POLICY, 0)
    for update in (1, 2):
        if monitor.observe(update, losses(1), losses(1), .001):
            bind_best(monitor, tmp_path)
    restored = Monitor.restore(None, monitor.state(), 2, 0, tmp_path)
    assert restored.observe(3, losses(1), losses(1), .001) is False
    assert restored.stopped and restored.best_update == 1
    with pytest.raises(ValueError, match='policy changed'):
        Monitor.restore(dict(POLICY, patience=3), monitor.state(), 2, 0, tmp_path)
    invalid = monitor.state()
    invalid['bad_evaluations'] = 0
    with pytest.raises(ValueError, match='counters'):
        Monitor.restore(POLICY, invalid, 2, 0, tmp_path)
    (tmp_path/'checkpoints/checkpoint_best.pt').write_text('corrupt')
    with pytest.raises(ValueError, match='best dev checkpoint'):
        Monitor.restore(POLICY, monitor.state(), 2, 0, tmp_path)


def test_legacy_checkpoint_adoption_starts_at_actual_update_without_fake_history():
    monitor = Monitor.restore(POLICY, None, 80, 10000, Path('unused'))
    assert monitor.origin == 80 and not monitor.history
    monitor.observe(81, losses(2), losses(2), .001)
    assert monitor.update == 81 and not monitor.stopped
    assert Monitor.restore(None, None, 80, 10000, Path('unused')) is None


@pytest.mark.parametrize('policy', [dict(POLICY, interval=0), dict(POLICY, patience=True),
    dict(POLICY, min_relative_improvement=float('nan')), dict(POLICY, min_relative_improvement=1)])
def test_invalid_policies_fail_closed(policy):
    with pytest.raises(ValueError):
        validate_policy(policy)


def test_reports_are_machine_readable_and_terminal_state_is_checkpoint_aligned(tmp_path):
    from direct_s2st.translatotron2.monitor import select_checkpoint
    assert select_checkpoint(tmp_path) == tmp_path/'checkpoints/checkpoint_last.pt'
    with pytest.raises(FileNotFoundError):
        select_checkpoint(tmp_path, 'best')
    monitor = Monitor(POLICY, 0)
    for update in range(1, 4):
        if monitor.observe(update, losses(1), losses(1), .001):
            bind_best(monitor, tmp_path)
    monitor.write_reports(tmp_path)
    assert json.loads((tmp_path/'training-monitor.json').read_text()) == monitor.state()
    assert len((tmp_path/'training-monitor.csv').read_text().splitlines()) == 4
    ET.fromstring((tmp_path/'training-monitor.svg').read_text())
    assert early_stop_result(tmp_path, 'tt2', 3)['status'] == 'EARLY_STOPPED'
    with pytest.raises(ValueError, match='version/warmup'):
        early_stop_result(tmp_path, 's2ut', 3)
    assert select_checkpoint(tmp_path) == tmp_path/'checkpoints/checkpoint_best.pt'
    assert select_checkpoint(tmp_path, 'last') == tmp_path/'checkpoints/checkpoint_last.pt'
    with pytest.raises(ValueError, match='checkpoint'):
        early_stop_result(tmp_path, 'tt2', 2)
    (tmp_path/'checkpoints/checkpoint_best.pt').write_text('corrupt')
    with pytest.raises(ValueError, match='missing or changed'):
        select_checkpoint(tmp_path)  # Never silently fall back to last.


def test_trainer_early_stop_and_resume_keep_optimizer_and_history(tmp_path, monkeypatch):
    torch = pytest.importorskip('torch')
    from direct_s2st.translatotron2 import train
    from direct_s2st.translatotron2.engine import load_checkpoint
    torch.set_num_threads(1)
    torch.manual_seed(10)
    sample = dict(source=torch.randn(1, 12, 80), source_lengths=torch.tensor([12]),
                  phones=torch.tensor([[3, 4, 2]]), phone_lengths=torch.tensor([3]),
                  target=torch.randn(1, 8, 80), target_lengths=torch.tensor([8]))
    class Dataset:
        spec = dict(n_mels=80)
        tokens = list('abcdefg')
        def __init__(self, root, split): assert split in ('train', 'dev')
        def __len__(self): return 1
        def __getitem__(self, index): return sample
    monkeypatch.setattr(train, 'PreparedDataset', Dataset)
    monkeypatch.setattr(train, 'fingerprint', lambda root: {'fixture': 'synthetic'})
    monkeypatch.setattr('direct_s2st.translatotron2.validation.validate', lambda *a: losses(1))
    monkeypatch.setenv('S2ST_TT2_MONITOR_POLICY', json.dumps(POLICY))
    monkeypatch.delenv('S2ST_TRAIN_STAGING', raising=False)
    monkeypatch.delenv('S2ST_TRAIN_DEADLINE', raising=False)
    monkeypatch.delenv('S2ST_TRAIN_OPTIMIZE', raising=False)
    def run(work, total, restore=None):
        args = ['train', '--data-root', str(tmp_path/'data'), '--run-root', str(work),
                '--warmup-updates', '3', '--save-interval-updates', '2', '--max-updates', str(total)]
        if restore:
            args += ['--restore-file', str(restore)]
        monkeypatch.setattr('sys.argv', args)
        train.main()
        return load_checkpoint(work/'checkpoints/checkpoint_last.pt')[1]
    continuous, resumed = tmp_path/'continuous', tmp_path/'resumed'
    expected = run(continuous, 9)
    assert expected['updates'] == 5 and expected['monitor_state']['stopped']
    _, best = load_checkpoint(continuous/'checkpoints/checkpoint_best.pt')
    assert best['updates'] == 1 and best['optimizer']['state']
    assert expected['monitor_state']['best_sha256'] == raw_sha(continuous/'checkpoints/checkpoint_best.pt')
    run(resumed, 3)
    actual = run(resumed, 9, resumed/'checkpoints/checkpoint_last.pt')
    assert actual['updates'] == 5
    assert actual['monitor_state']['history'] == expected['monitor_state']['history']
    for name, value in expected['model'].items():
        torch.testing.assert_close(value, actual['model'][name], rtol=0, atol=0)
    for key, values in expected['optimizer']['state'].items():
        for name, value in values.items():
            torch.testing.assert_close(value, actual['optimizer']['state'][key][name], rtol=0, atol=0)
    before = (resumed/'checkpoints/checkpoint_last.pt').read_bytes()
    run(resumed, 9, resumed/'checkpoints/checkpoint_last.pt')
    assert (resumed/'checkpoints/checkpoint_last.pt').read_bytes() == before


def test_monitor_and_real_dev_validation_do_not_change_training_weights(tmp_path, monkeypatch):
    torch = pytest.importorskip('torch')
    from direct_s2st.translatotron2 import train
    from direct_s2st.translatotron2.engine import load_checkpoint
    torch.set_num_threads(1)
    torch.manual_seed(30)
    sample = dict(source=torch.randn(1, 12, 80), source_lengths=torch.tensor([12]),
                  phones=torch.tensor([[3, 4, 2]]), phone_lengths=torch.tensor([3]),
                  target=torch.randn(1, 8, 80), target_lengths=torch.tensor([8]))
    splits = []
    class Dataset:
        spec = dict(n_mels=80)
        tokens = list('abcdefg')
        def __init__(self, root, split):
            assert split in ('train', 'dev')  # Never consult test.
            splits.append(split)
        def __len__(self): return 1
        def __getitem__(self, index): return sample
    monkeypatch.setattr(train, 'PreparedDataset', Dataset)
    monkeypatch.setattr(train, 'fingerprint', lambda root: {'fixture': 'synthetic'})
    for name in ('S2ST_TRAIN_STAGING', 'S2ST_TRAIN_DEADLINE', 'S2ST_TRAIN_OPTIMIZE'):
        monkeypatch.delenv(name, raising=False)
    states = []
    for enabled in (False, True):
        if enabled:
            monkeypatch.setenv('S2ST_TT2_MONITOR_POLICY', json.dumps(dict(POLICY, interval=2)))
        else:
            monkeypatch.delenv('S2ST_TT2_MONITOR_POLICY', raising=False)
        work = tmp_path/str(enabled)
        monkeypatch.setattr('sys.argv', ['train', '--data-root', str(tmp_path/'data'), '--run-root', str(work),
            '--warmup-updates', '10000', '--max-updates', '4'])
        train.main()
        states.append(load_checkpoint(work/'checkpoints/checkpoint_last.pt')[1])
    assert splits == ['train', 'train', 'dev']
    for name, value in states[0]['model'].items():
        torch.testing.assert_close(value, states[1]['model'][name], rtol=0, atol=0)
    torch.testing.assert_close(states[0]['rng'], states[1]['rng'], rtol=0, atol=0)
    assert states[0]['data_fingerprint'] == states[1]['data_fingerprint']
    # Adopt a shorter schedule from a real legacy Adam checkpoint, then resume
    # again without the override: the new schedule must remain checkpointed.
    from direct_s2st.translatotron2.batching import learning_rate
    work = tmp_path/'False'
    monkeypatch.setenv('S2ST_TT2_PRACTICAL_WARMUP', '1000')
    for total in (6, 8):
        monkeypatch.setattr('sys.argv', ['train', '--data-root', str(tmp_path/'data'), '--run-root', str(work),
            '--warmup-updates', '10000', '--max-updates', str(total),
            '--restore-file', str(work/'checkpoints/checkpoint_last.pt')])
        train.main()
        continued = load_checkpoint(work/'checkpoints/checkpoint_last.pt')[1]
        assert continued['updates'] == total
        assert continued['data_fingerprint'] == states[0]['data_fingerprint']
        assert continued['schedule_transition']['adopted_at'] == 4
        assert continued['monitor_state']['warmup'] == 1000
        assert continued['optimizer']['param_groups'][0]['lr'] == learning_rate(1e-4, total, 1000)
        assert max(int(v['step']) for v in continued['optimizer']['state'].values()) == total
        best = load_checkpoint(work/'checkpoints/checkpoint_best.pt')[1]
        assert best['schedule_transition'] == continued['schedule_transition']
        monkeypatch.delenv('S2ST_TT2_PRACTICAL_WARMUP', raising=False)
    before_history = continued['monitor_state']['history']
    previous_best = continued['monitor_state']['best_update']
    monkeypatch.setenv('S2ST_TT2_PRACTICAL_WARMUP', '500')
    monkeypatch.setattr('sys.argv', ['train', '--data-root', str(tmp_path/'data'), '--run-root', str(work),
        '--warmup-updates', '10000', '--max-updates', '9',
        '--restore-file', str(work/'checkpoints/checkpoint_last.pt')])
    train.main()
    shortened = load_checkpoint(work/'checkpoints/checkpoint_last.pt')[1]
    assert shortened['data_fingerprint'] == continued['data_fingerprint']
    assert shortened['monitor_state']['warmup'] == 500
    assert shortened['monitor_state']['history'][:len(before_history)] == before_history
    assert shortened['monitor_state']['best_update'] in (previous_best, 9)
    assert max(int(v['step']) for v in shortened['optimizer']['state'].values()) == 9
    assert shortened['optimizer']['param_groups'][0]['lr'] == learning_rate(1e-4, 9, 500)
    assert shortened['schedule_transition']['changes'] == [
        dict(warmup=1000, adopted_at=4), dict(warmup=500, adopted_at=8)]
    monkeypatch.delenv('S2ST_TT2_PRACTICAL_WARMUP', raising=False)
    monkeypatch.setattr('sys.argv', ['train', '--data-root', str(tmp_path/'data'), '--run-root', str(work),
        '--warmup-updates', '10000', '--max-updates', '10',
        '--restore-file', str(work/'checkpoints/checkpoint_last.pt')])
    train.main()
    inherited = load_checkpoint(work/'checkpoints/checkpoint_last.pt')[1]
    assert inherited['schedule_transition'] == shortened['schedule_transition']
    assert inherited['monitor_state']['warmup'] == 500
    assert inherited['monitor_state']['history'][:-1] == shortened['monitor_state']['history']
    assert inherited['optimizer']['param_groups'][0]['lr'] == learning_rate(1e-4, 10, 500)


def test_warmup_transition_from_85_is_absolute_and_records_explicit_shortening():
    from direct_s2st.translatotron2.monitor import resume_schedule
    warmup, record = resume_schedule(10000, 85, {'updates': 85}, 1000)
    assert warmup == 1000 and record['adopted_at'] == 85
    assert resume_schedule(10000, 1100, {'schedule_transition': record}) == (1000, record)
    # An original v1 checkpoint (the real experiment) remains supported.
    old = dict(version=1, base_warmup=10000, warmup=1000, adopted_at=85, mode='paper_practical')
    warmup, revised = resume_schedule(10000, 133, {'schedule_transition': old}, 500)
    assert warmup == 500 and revised['changes'] == [dict(warmup=1000, adopted_at=85), dict(warmup=500, adopted_at=133)]
    assert resume_schedule(10000, 600, {'schedule_transition': revised}) == (500, revised)
    with pytest.raises(ValueError, match='existing warmup'):
        resume_schedule(10000, 1001, {'schedule_transition': old}, 500)
    with pytest.raises(ValueError, match='existing warmup'):
        resume_schedule(10000, 90, {'schedule_transition': old}, 2000)
    assert resume_schedule(10000, 0, None, 500)[0] == 500
    corrupt = dict(revised, warmup=250)
    with pytest.raises(ValueError, match='invalid saved'):
        resume_schedule(10000, 600, {'schedule_transition': corrupt})


def test_monitor_warmup_migration_preserves_history_best_and_patience(tmp_path):
    monitor = Monitor(dict(POLICY, interval=10, patience=10), 1000, origin=85)
    for update in range(86, 134):
        if monitor.observe(update, losses(1), losses(1) if update % 10 == 0 else None, .001):
            bind_best(monitor, tmp_path)
    saved = monitor.state()
    shortened = Monitor.restore(None, saved, 133, 500, tmp_path)
    assert shortened.history == saved['history']
    assert shortened.best_sha256 == saved['best_sha256']
    assert shortened.best_update == saved['best_update']
    assert shortened.state()['warmup_history'] == [{'from':1000, 'to':500, 'adopted_at':133}]
    for update in range(134, 601):
        shortened.observe(update, losses(1), losses(1) if update % 10 == 0 else None, .001)
        if update in (134, 500, 550):
            restored = Monitor.restore(None, shortened.state(), update, 500, tmp_path)
            assert restored.state() == shortened.state()
        assert shortened.stopped == (update == 600)
    with pytest.raises(ValueError, match='warmup changed'):
        Monitor.restore(None, shortened.state(), 600, 1000, tmp_path)


@pytest.mark.parametrize('kind', ['tt2', 's2ut'])
def test_frozen_best_reports_and_terminal_session_restore_without_training(tmp_path, monkeypatch, kind):
    from direct_s2st.train_runtime import checkpoint_saved
    from direct_s2st.session_runtime import run_continuous
    from direct_s2st.colab import publish, latest, run_session
    from direct_s2st.hashing import sha256_file
    work, backup = tmp_path/'work', tmp_path/'drive'
    work.mkdir()
    lock = tmp_path/'lock'
    lock.write_text('data')
    cfg = dict(kind=kind, checkpoint='checkpoint.pt', command=['fixture', '{updates}'],
               resume_args=[], identity_files=[str(lock)],
               runtime=dict(cache_gb=0, readers=1, precision='default', adaptive_batch=False))
    identity = dict(config=cfg, work=str(work), inputs={str(lock): sha256_file(lock)})
    class Process:
        returncode = 0
        def __init__(self, args, env):
            from direct_s2st.s2ut.monitor import S2UTMonitor
            monitor = (Monitor if kind == 'tt2' else S2UTMonitor)(POLICY, 0)
            for update in range(1, 4):
                values = {key: 1. for key in monitor.metrics}
                if monitor.observe(update, values, values, .001):
                    bind_best(monitor, work)
            monitor.write_reports(work)
            (work/'checkpoint.pt').write_text('3')
            with monkeypatch.context() as child:
                child.setenv('S2ST_TRAIN_STAGING', env['S2ST_TRAIN_STAGING'])
                checkpoint_saved(work, work/'checkpoint.pt', 3, force=True)
        def poll(self): return self.returncode
    monkeypatch.delenv('S2ST_TT2_MONITOR_POLICY', raising=False)
    monkeypatch.delenv('S2ST_S2UT_MONITOR_POLICY', raising=False)
    monkeypatch.delenv('S2ST_DRIVE_SAFE', raising=False)
    result = run_continuous(cfg, work=work, backup=backup, identity=identity, completed=0,
        total=20, seconds=0, inspect_checkpoint=lambda p,k: int(p.read_text()), publish=publish, popen=Process)
    assert result['status'] == 'EARLY_STOPPED' and result['durable_updates'] == 3
    saved = latest(backup, identity)[0]/'files'
    for name in ('training-monitor.json', 'training-monitor.csv', 'training-monitor.svg', 'checkpoints/checkpoint_best.pt'):
        assert (saved/name).read_bytes() == (work/name).read_bytes()
    for name in ('training-monitor.json', 'training-monitor.csv', 'training-monitor.svg'):
        assert (backup/name).read_bytes() == (saved/name).read_bytes()
    restored = run_session(cfg, work=work, backup=backup, total=20, seconds=0, resume=True,
                           confirm_training=True, inspect_checkpoint=lambda p,k: int(p.read_text()),
                           runner=lambda *a, **k: pytest.fail('terminal run must not train'))
    assert restored == result


def test_dashboard_copy_failure_never_invalidates_recovery_point(tmp_path, monkeypatch, capsys):
    from direct_s2st.translatotron2 import monitor as module
    frozen = tmp_path/'frozen'
    frozen.mkdir()
    (frozen/'training-monitor.json').write_text('{"fixture":true}')
    def fail(*a, **kw):
        raise OSError('Drive unavailable')
    monkeypatch.setattr(module, 'atomic_write_bytes', fail)
    module.mirror_reports(frozen, tmp_path/'backup')
    assert 'verified snapshot preserved' in capsys.readouterr().err
    assert (frozen/'training-monitor.json').read_text() == '{"fixture":true}'
