import json
import math
from pathlib import Path
from types import ModuleType, SimpleNamespace as NS
import xml.etree.ElementTree as ET

import pytest
import torch

from direct_s2st.s2ut.monitor import LossSums, S2UTMonitor, runtime_hooks
from direct_s2st.translatotron2.monitor import early_stop_result, select_checkpoint


POLICY = dict(interval=2, patience=2, min_relative_improvement=.001)
DUAL_POLICY = dict(version=2, interval=10, patience=5, min_relative_improvement=.01)


def observe(monitor, update, unit, total, work):
    values = {key: 1. for key in monitor.metrics}
    values.update(unit_loss=unit, loss=total)
    if monitor.observe(update, values, values if monitor.due(update) else None, .001):
        def save(path):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(str(update))
        monitor.save_best(work, save)


@pytest.mark.parametrize('improving', ['unit_loss', 'loss'])
def test_dual_stopping_waits_for_both_losses_after_warmup(tmp_path, improving):
    monitor = S2UTMonitor(DUAL_POLICY, 400, origin=380)
    for update in range(381, 401):
        # Worse at the warmup boundary: anchors must retain historical minima.
        observe(monitor, update, 100 if update < 400 else 110,
                100 if update < 400 else 120, tmp_path)
        assert not monitor.stopped and monitor.bad_evaluations == monitor.total_bad_evaluations == 0
    for update in range(401, 461):
        values = dict(unit_loss=100, loss=100)
        if update >= 410:
            values[improving] = 98
        observe(monitor, update, values['unit_loss'], values['loss'], tmp_path)
        assert monitor.stopped == (update == 460)
    assert monitor.best_update == (410 if improving == 'unit_loss' else 390)
    assert monitor.best_total_update == (410 if improving == 'loss' else 390)
    restored = S2UTMonitor.restore(DUAL_POLICY, monitor.state(), 460, 400, tmp_path)
    assert restored.state() == monitor.state()
    with pytest.raises(ValueError, match='monitor updates'):
        observe(restored, 461, 97, 97, tmp_path)


def test_dual_small_best_improvements_accumulate_and_resume_without_reset(tmp_path):
    monitor = S2UTMonitor(DUAL_POLICY, 400, origin=380)
    scores = {410:99.8, 420:99.5, 430:99.2, 440:99.01, 450:99.}
    for update in range(381, 501):
        unit = scores.get(update//10*10, 100 if update < 410 else 99)
        observe(monitor, update, unit, 100, tmp_path)
        if update == 440:
            assert monitor.bad_evaluations == monitor.total_bad_evaluations == 4
            original = monitor.state()
            monitor = S2UTMonitor.restore(DUAL_POLICY, original, 440, 400, tmp_path)
            assert monitor.state() == original
        if update == 450:
            assert monitor.best_update == 450 and monitor.anchor == 99
            assert monitor.bad_evaluations == 0 and monitor.total_bad_evaluations == 5
        assert monitor.stopped == (update == 500)
    invalid = monitor.state()
    invalid['total_bad_evaluations'] = 0
    with pytest.raises(ValueError, match='history/counters'):
        S2UTMonitor.restore(DUAL_POLICY, invalid, 500, 400, tmp_path)


def test_dual_policy_adoption_keeps_history_and_bests_without_retrospective_count(tmp_path):
    old = S2UTMonitor(dict(POLICY, interval=10), 400, origin=380)
    for update in range(381, 401):
        observe(old, update, 100 if update < 400 else 110,
                200 if update < 400 else 210, tmp_path)
    history, best_sha = old.state()['history'], old.best_sha256
    monitor = S2UTMonitor.restore(DUAL_POLICY, old.state(), 400, 400, tmp_path)
    assert monitor.history == history and monitor.best_sha256 == best_sha
    assert monitor.anchor == 100 and monitor.total_anchor == 200
    for update in range(401, 451):
        observe(monitor, update, 105, 202, tmp_path)
        if update == 430:
            original = monitor.state()
            monitor = S2UTMonitor.restore(DUAL_POLICY, original, 430, 400, tmp_path)
            assert monitor.state() == original
        assert monitor.stopped == (update == 450)
    assert monitor.best_update == monitor.best_total_update == 390
    with pytest.raises(ValueError, match='policy changed'):
        S2UTMonitor.restore(dict(DUAL_POLICY, patience=6), monitor.state(), 450, 400, tmp_path)
    with pytest.raises(ValueError, match='warmup changed'):
        S2UTMonitor.restore(DUAL_POLICY, monitor.state(), 450, 500, tmp_path)


def output(value, size=2):
    logs = {'multitask': {key: dict(loss=value*size, sample_size=size)
                          for key in S2UTMonitor.metrics[2:]}}
    return torch.tensor(value*size), size, logs


def test_loss_reduction_uses_per_task_token_denominators():
    sums = LossSums()
    sums.record(output(1, 2), 2)
    sums.record(output(3, 6), 18)
    assert sums.result()['unit_loss'] == pytest.approx(2.5/math.log(2))
    invalid = output(1)
    invalid[2]['multitask']['source_letter']['sample_size'] = 0
    with pytest.raises(ValueError, match='sample_size'):
        sums.record(invalid, 2)


def test_selection_uses_unit_loss_even_when_total_loss_improves(tmp_path):
    monitor = S2UTMonitor(dict(POLICY, interval=1), 0)
    for update in range(1, 4):
        values = {key: 1. for key in monitor.metrics}
        values['loss'] = 4-update
        if monitor.observe(update, values, values, .001):
            monitor.save_best(tmp_path, lambda path: (path.parent.mkdir(parents=True, exist_ok=True), path.write_text('best')))
    monitor.write_reports(tmp_path)
    assert monitor.stopped and monitor.best_update == 1
    assert early_stop_result(tmp_path, 's2ut', 3)['selection_metric'] == 'teacher_forced_dev_unit_loss'
    assert select_checkpoint(tmp_path).name == 'checkpoint_best.pt'
    assert 'dev_unit_loss' in (tmp_path/'training-monitor.csv').read_text()
    ET.fromstring((tmp_path/'training-monitor.svg').read_text())
    with pytest.raises(ValueError, match='policy changed'):
        S2UTMonitor.restore(dict(POLICY, interval=1, patience=3), monitor.state(), 3, 0, tmp_path)


def install_trainer(monkeypatch):
    """Run hooks with real CPU Adam and trusted files, without importing old Hydra."""
    fairseq, trainer_module, checkpoint, cli, loop = [ModuleType(n) for n in
        ('fairseq', 'fairseq.trainer', 'fairseq.checkpoint_utils', 'fairseq_cli', 'fairseq_cli.train')]
    class Trainer:
        def __init__(self, work):
            self.work, self.updates = work, 0
            self.model = torch.nn.Linear(1, 1)
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=.001)
            self.data_parallel_world_size = 1
            self.cfg = NS(optimization=NS(max_update=20, stop_time_hours=0),
                dataset=NS(disable_validation=False, max_valid_steps=None),
                checkpoint=NS(no_save=False, no_last_checkpoints=False, no_save_optimizer_state=False,
                    write_checkpoints_asynchronously=False, save_interval_updates=3))
        def get_num_updates(self): return self.updates
        def get_lr(self): return .001
        def cumulative_training_time(self): return 0
        def consolidate_optimizer(self): pass
        def train_step(self, audit):
            self.optimizer.zero_grad()
            loss = self.model(torch.ones(1, 1)).square().sum()
            audit.monitor.record(output(float(loss.detach())), loss.detach()*2)
            loss.backward()
            self.optimizer.step()
            self.updates += 1
            return {'fixture': True}
        def save_checkpoint(self, path, extra):
            torch.save(dict(model=self.model.state_dict(), optimizer=self.optimizer.state_dict(),
                            updates=self.updates, extra_state=extra), path)
            return str(path)
        def load_checkpoint(self, path):
            if path is None: return None
            state = torch.load(path, weights_only=True)
            self.model.load_state_dict(state['model'])
            self.optimizer.load_state_dict(state['optimizer'])
            self.updates = state['updates']
            return state['extra_state']
    trainer_module.Trainer = Trainer
    fairseq.checkpoint_utils = checkpoint
    cli.train = loop
    loop.validate_and_save = lambda *a: pytest.fail('unmonitored loop invoked')
    dev_updates = []
    def validate(cfg, trainer, task, iterator, splits):
        assert splits == ['dev']
        dev_updates.append(trainer.updates)
        # Unequal batch sizes test token-weighted aggregation over full dev.
        task.monitor.record(output(1, 2), 2)
        task.monitor.record(output(1, 6), 6)
        torch.rand(3)  # validation must preserve train RNG
        return [1.]
    loop.validate = validate
    def publish(cfg, trainer, iterator, score):
        assert score is None  # fairseq must not overwrite monitored best
        path = trainer.work/'checkpoints/checkpoint_last.pt'
        path.parent.mkdir(parents=True, exist_ok=True)
        return trainer.save_checkpoint(path, {'train_iterator': iterator.state_dict()})
    checkpoint.save_checkpoint = publish
    for module in (fairseq, trainer_module, checkpoint, cli, loop):
        monkeypatch.setitem(__import__('sys').modules, module.__name__, module)
    return Trainer, loop, dev_updates


def test_paper_reporting_online_evaluation_and_real_optimizer_resume(tmp_path, monkeypatch):
    from direct_s2st.s2ut.paper_recipe import report_policy
    from direct_s2st.s2ut import bleu_monitor
    Trainer, loop, dev_updates = install_trainer(monkeypatch)
    monkeypatch.setenv('S2ST_S2UT_PAPER_MONITOR', json.dumps(report_policy(interval=2,train_interval=1)))
    monkeypatch.setenv('S2ST_S2UT_BLEU_POLICY', '{}')  # Evaluator is isolated below.
    for name in ('S2ST_BATCH_PROBE','S2ST_TRAIN_ADAPTIVE_BATCH','S2ST_TRAIN_DEADLINE'):
        monkeypatch.delenv(name, raising=False)
    calls=[]
    def evaluate(trainer,task,work,update):
        calls.append(update)
        if update==4: raise RuntimeError('bounded simulated ASR loading failure')
    monkeypatch.setattr(bleu_monitor,'evaluate',evaluate)
    def run(work, count, restore=False):
        work.mkdir(exist_ok=True)
        trainer, audit = Trainer(work), NS()
        command=['wrapper','--save-dir',str(work/'checkpoints'),'--warmup-updates','10000']
        last=work/'checkpoints/checkpoint_last.pt'
        if restore: command+=['--restore-file',str(last)]
        iterator=NS(state_dict=lambda:{'iterations':trainer.updates})
        with runtime_hooks(audit,command):
            trainer.load_checkpoint(last if restore else None)
            for _ in range(count):
                trainer.train_step(audit)
                _,stop=loop.validate_and_save(trainer.cfg,trainer,audit,iterator,['dev'],False)
                assert not stop
        return torch.load(last,weights_only=True)
    torch.manual_seed(8)
    continuous=run(tmp_path/'continuous',6)
    torch.manual_seed(8)
    first=run(tmp_path/'resumed',3)
    resumed=run(tmp_path/'resumed',3,True)
    assert resumed['updates']==continuous['updates']==6 and first['updates']==3
    for name,tensor in continuous['model'].items():
        torch.testing.assert_close(tensor,resumed['model'][name],rtol=0,atol=0)
    for parameter,state in continuous['optimizer']['state'].items():
        for key,value in state.items():
            torch.testing.assert_close(value,resumed['optimizer']['state'][parameter][key],rtol=0,atol=0)
    assert resumed['extra_state']['s2st_monitor']['stopped'] is False
    assert select_checkpoint(tmp_path/'resumed').name=='checkpoint_best.pt'
    assert torch.load(select_checkpoint(tmp_path/'resumed'),weights_only=True)['updates']==2
    failed=json.loads((tmp_path/'resumed/training-bleu/000000004/result.json').read_text())
    assert failed['status']=='failed' and failed['asr_bleu'] is None
    assert calls==[2,4,6,2,4,6]


@pytest.mark.parametrize('policy, terminal', [(POLICY, 6), (dict(DUAL_POLICY, interval=2), 12)])
def test_real_adam_hooks_resume_counters_best_and_restore_methods(tmp_path, monkeypatch, policy, terminal):
    Trainer, loop, dev_updates = install_trainer(monkeypatch)
    monkeypatch.setenv('S2ST_S2UT_MONITOR_POLICY', json.dumps(policy))
    monkeypatch.delenv('S2ST_BATCH_PROBE', raising=False)
    monkeypatch.delenv('S2ST_TRAIN_ADAPTIVE_BATCH', raising=False)
    monkeypatch.delenv('S2ST_TRAIN_DEADLINE', raising=False)
    original_step = Trainer.train_step
    def run(work, limit, resume=False):
        work.mkdir(exist_ok=True)
        audit = NS()
        command = ['wrapper', '--save-dir', str(work/'checkpoints'), '--warmup-updates', '2']
        path = work/'checkpoints/checkpoint_last.pt'
        if resume: command += ['--restore-file', str(path)]
        trainer = Trainer(work)
        iterator = NS(state_dict=lambda: dict(iterations=trainer.updates))
        with runtime_hooks(audit, command):
            trainer.load_checkpoint(path if resume else None)
            for _ in range(limit):
                trainer.train_step(audit)
                rng = torch.get_rng_state().clone()
                _, stop = loop.validate_and_save(trainer.cfg, trainer, audit, iterator, ['dev'], False)
                torch.testing.assert_close(rng, torch.get_rng_state(), rtol=0, atol=0)
                if stop: break
        assert Trainer.train_step is original_step and not hasattr(audit, 'monitor')
        return torch.load(path, weights_only=True)
    torch.manual_seed(8)
    continuous = run(tmp_path/'continuous', 20)
    assert continuous['updates'] == terminal
    torch.manual_seed(8)
    initial = run(tmp_path/'resumed', 3)
    assert initial['updates'] == 3
    resumed = run(tmp_path/'resumed', 20, resume=True)
    assert resumed['updates'] == terminal
    for field, value in continuous['extra_state']['s2st_monitor'].items():
        if field != 'best_sha256':  # Serialized best file names can differ.
            assert resumed['extra_state']['s2st_monitor'][field] == value
    for name, tensor in continuous['model'].items():
        torch.testing.assert_close(tensor, resumed['model'][name], rtol=0, atol=0)
    for parameter, state in continuous['optimizer']['state'].items():
        for key, value in state.items():
            torch.testing.assert_close(value, resumed['optimizer']['state'][parameter][key], rtol=0, atol=0)
    result = early_stop_result(tmp_path/'resumed', 's2ut', terminal)
    assert result['best_update'] == 2
    if policy.get('version') == 2:
        assert result['stopping_metrics'] == ['unit_loss', 'loss'] and result['stopping_rule'] == 'all_plateau'
        assert result['best_total_update'] == 2
    best = torch.load(select_checkpoint(tmp_path/'resumed'), weights_only=True)
    assert best['updates'] == 2 and best['optimizer']['state']
    assert best['extra_state']['train_iterator']['iterations'] == 2
    assert dev_updates == list(range(2, terminal+1, 2))*2
    (tmp_path/'resumed/checkpoints/checkpoint_best.pt').write_text('corrupt')
    with pytest.raises(ValueError, match='best dev checkpoint'):
        run(tmp_path/'resumed', 1, resume=True)
    assert Trainer.train_step is original_step


def test_full_dev_only_policy_drift_and_probe_bypass(tmp_path, monkeypatch):
    Trainer, loop, _ = install_trainer(monkeypatch)
    monkeypatch.setenv('S2ST_S2UT_MONITOR_POLICY', json.dumps(POLICY))
    monkeypatch.delenv('S2ST_BATCH_PROBE', raising=False)
    monkeypatch.delenv('S2ST_TRAIN_ADAPTIVE_BATCH', raising=False)
    command = ['wrapper', '--save-dir', str(tmp_path/'checkpoints'), '--warmup-updates', '2']
    audit, trainer = NS(), Trainer(tmp_path)
    original = Trainer.train_step
    iterator = NS(state_dict=lambda: {})
    with runtime_hooks(audit, command):
        trainer.load_checkpoint(None)
        trainer.train_step(audit)
        with pytest.raises(ValueError, match='full dev'):
            loop.validate_and_save(trainer.cfg, trainer, audit, iterator, ['test'], False)
    assert Trainer.train_step is original
    monkeypatch.setenv('S2ST_BATCH_PROBE', 'disposable.json')
    with runtime_hooks(audit, command):
        assert Trainer.train_step is original and not hasattr(audit, 'monitor')
