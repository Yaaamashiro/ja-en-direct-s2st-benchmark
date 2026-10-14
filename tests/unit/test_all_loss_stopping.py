"""All-loss v3 decisions, immutable legacy adoption, and absolute safety caps."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace as NS
import xml.etree.ElementTree as ET

import pytest
import torch

from direct_s2st.translatotron2.monitor import Monitor, absolute_update_limit, early_stop_result
from direct_s2st.s2ut.monitor import S2UTMonitor, runtime_hooks
from direct_s2st.drive_staging import raw_sha

POLICY = dict(version=3, interval=1, patience=5, min_relative_improvement=.01)


def baseline(cls):
    return (dict(loss=100., mel_loss=10., phone_loss=10., duration_loss=80.) if cls is Monitor else
            dict(loss=100., unit_loss=10., source_letter=10., target_letter=10., decoder_target_ctc=80.))


def save_best(monitor, work):
    def save(path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f'{monitor.best_update}:{monitor.best_value}', encoding='utf-8')
    monitor.save_best(work, save)


def observe(monitor, update, dev, work):
    improved = monitor.observe(update, baseline(type(monitor)), dev, .001)
    if improved:
        save_best(monitor, work)
    return improved


def seeded(cls, work):
    monitor = cls(POLICY, 1)
    observe(monitor, 1, baseline(cls), work)
    return monitor


@pytest.mark.parametrize('cls', [Monitor, S2UTMonitor])
def test_one_loss_improves_and_only_its_anchor_changes(cls, tmp_path, capsys):
    monitor = seeded(cls, tmp_path)
    initial = dict(monitor.anchors)
    metric = 'mel_loss' if cls is Monitor else 'source_letter'
    dev = {k:v*.995 for k,v in initial.items()}
    dev[metric] = initial[metric]*.98
    monitor.bad_evaluations = 2
    observe(monitor, 2, dev, tmp_path)
    assert monitor.anchors == dict(initial, **{metric:dev[metric]})
    assert monitor.bad_evaluations == 0 and not monitor.stopped
    output = capsys.readouterr()
    assert f'meaningful_metrics=["{metric}"]' in output.err
    assert f'{metric}={dev[metric]:.6g} improvement=2.000% *' in output.err
    assert 'bad_evaluations=0/5' in output.err and not output.out


@pytest.mark.parametrize('cls', [Monitor, S2UTMonitor])
def test_five_consecutive_all_metric_misses_stop(cls, tmp_path):
    monitor = seeded(cls, tmp_path)
    initial = dict(monitor.anchors)
    for count in range(1, 6):
        observe(monitor, count+1, {k:v*(1-count*.001) for k,v in initial.items()}, tmp_path)
        assert monitor.anchors == initial
        assert monitor.bad_evaluations == count
        assert monitor.stopped == (count == 5)


@pytest.mark.parametrize('cls', [Monitor, S2UTMonitor])
def test_different_loss_resets_shared_counter_after_two_misses(cls, tmp_path):
    monitor = seeded(cls, tmp_path)
    initial = dict(monitor.anchors)
    for update in (2, 3):
        observe(monitor, update, baseline(cls), tmp_path)
    assert monitor.bad_evaluations == 2
    metric = 'duration_loss' if cls is Monitor else 'decoder_target_ctc'
    observe(monitor, 4, dict(initial, **{metric:initial[metric]*.98}), tmp_path)
    assert monitor.bad_evaluations == 0
    assert monitor.anchors == dict(initial, **{metric:initial[metric]*.98})
    observe(monitor, 5, dict(monitor.anchors), tmp_path)
    assert monitor.bad_evaluations == 1 and not monitor.stopped


@pytest.mark.parametrize('cls', [Monitor, S2UTMonitor])
def test_best_checkpoint_is_independent_of_auxiliary_stopping_improvement(cls, tmp_path):
    monitor = seeded(cls, tmp_path)
    before = (tmp_path/'checkpoints/checkpoint_best.pt').read_bytes()
    best = monitor.best_value, monitor.best_update, monitor.best_sha256
    dev = baseline(cls)
    dev[monitor.score_key] *= 1.01  # Selection metric got worse.
    dev['mel_loss' if cls is Monitor else 'target_letter'] *= .98
    monitor.bad_evaluations = 4
    assert not observe(monitor, 2, dev, tmp_path)
    assert monitor.bad_evaluations == 0 and not monitor.stopped
    assert (monitor.best_value, monitor.best_update, monitor.best_sha256) == best
    assert (tmp_path/'checkpoints/checkpoint_best.pt').read_bytes() == before


@pytest.mark.parametrize('cls', [Monitor, S2UTMonitor])
def test_resume_exactly_restores_anchors_counter_and_all_metric_bests(cls, tmp_path):
    monitor = seeded(cls, tmp_path)
    for update in (2, 3):
        observe(monitor, update, {k:v*.995 for k,v in baseline(cls).items()}, tmp_path)
    saved = monitor.state()
    # A checkpoint's JSON-compatible monitor payload survives serialization.
    restored = cls.restore(POLICY, json.loads(json.dumps(saved)), 3, 1, tmp_path)
    assert restored.state() == saved
    assert restored.anchors == monitor.anchors and restored.bad_evaluations == 2
    for update in (4, 5, 6):
        observe(restored, update, {k:v*.995 for k,v in baseline(cls).items()}, tmp_path)
    assert restored.stopped and restored.bad_evaluations == 5
    for field in ('anchors', 'best_values', 'best_updates'):
        invalid = deepcopy(saved)
        invalid[field][restored.score_key] = 999
        with pytest.raises(ValueError, match='history/counters'):
            cls.restore(None, invalid, 3, 1, tmp_path)


@pytest.mark.parametrize('cls', [Monitor, S2UTMonitor])
@pytest.mark.parametrize('legacy_version', [1, 2])
def test_legacy_terminal_adoption_is_future_only_and_keeps_original_bytes(cls, legacy_version, tmp_path, monkeypatch):
    old = cls(dict(POLICY, version=legacy_version), 1)
    for update in range(1, 7):
        observe(old, update, baseline(cls), tmp_path)
    assert old.stopped
    saved = old.state()
    best_bytes = (tmp_path/'checkpoints/checkpoint_best.pt').read_bytes()
    monitor = cls.restore(POLICY, saved, 6, 1, tmp_path)
    assert saved == old.state()
    assert monitor.history == saved['history']
    assert monitor.anchors == baseline(cls)
    assert monitor.best_values == baseline(cls)
    assert monitor.best_updates == {k:1 for k in monitor.metrics}
    assert monitor.bad_evaluations == 0 and not monitor.stopped
    assert monitor.best_sha256 == saved['best_sha256']
    assert (tmp_path/'checkpoints/checkpoint_best.pt').read_bytes() == best_bytes
    assert monitor.state()['policy_history'][-1] == dict(adopted_at=6, **{'from':saved['policy'], 'to':POLICY})
    assert cls.restore(None, monitor.state(), 6, 1, tmp_path).state() == monitor.state()
    old.write_reports(tmp_path)
    variable = 'S2ST_TT2_MONITOR_POLICY' if cls is Monitor else 'S2ST_S2UT_MONITOR_POLICY'
    monkeypatch.setenv(variable, json.dumps(POLICY))
    assert early_stop_result(tmp_path, 'tt2' if cls is Monitor else 's2ut', 6) is None
    assert json.loads((tmp_path/'training-monitor.json').read_text(encoding='utf-8')) == saved
    for update in range(7, 12):
        observe(monitor, update, baseline(cls), tmp_path)
        assert monitor.stopped == (update == 11)
    assert cls.restore(None, monitor.state(), 11, 1, tmp_path).state() == monitor.state()


@pytest.mark.parametrize('cls', [Monitor, S2UTMonitor])
def test_warmup_uses_per_metric_historical_minimum_not_last_observation(cls, tmp_path):
    monitor = cls(POLICY, 4)
    minima = {}
    for update in range(1, 5):
        dev = {k:v*(.8 if i == update-1 else 1.1) for i,(k,v) in enumerate(baseline(cls).items())}
        for k,v in dev.items(): minima[k] = min(minima.get(k,v),v)
        observe(monitor, update, dev, tmp_path)
        assert not monitor.stopped and monitor.bad_evaluations == 0
    assert monitor.anchors == minima
    assert any(monitor.anchors[k] != dev[k] for k in monitor.metrics)
    assert cls.restore(None, monitor.state(), 4, 4, tmp_path).state() == monitor.state()


@pytest.mark.parametrize('cls', [Monitor, S2UTMonitor])
def test_inclusive_threshold_and_zero_anchor_cannot_fake_improvement(cls, tmp_path):
    monitor = seeded(cls, tmp_path)
    observe(monitor, 2, {k:v*.99 for k,v in baseline(cls).items()}, tmp_path)
    assert monitor.bad_evaluations == 0
    assert monitor.anchors == {k:v*.99 for k,v in baseline(cls).items()}
    zero = cls(POLICY, 1)
    for update in range(1,7):
        observe(zero, update, {k:0. for k in zero.metrics}, tmp_path)
    assert zero.stopped and zero.bad_evaluations == 5


@pytest.mark.parametrize('cls', [Monitor, S2UTMonitor])
def test_csv_svg_preserve_all_train_dev_metrics(cls, tmp_path):
    monitor = seeded(cls, tmp_path)
    observe(monitor, 2, baseline(cls), tmp_path)
    monitor.write_reports(tmp_path)
    result = json.loads((tmp_path/'training-monitor.json').read_text(encoding='utf-8'))
    assert result['stopping_metrics'] == list(monitor.metrics)
    csv = (tmp_path/'training-monitor.csv').read_text(encoding='utf-8')
    svg = (tmp_path/'training-monitor.svg').read_text(encoding='utf-8')
    ET.fromstring(svg)
    for key in monitor.metrics:
        assert 'train_'+key in csv and 'dev_'+key in csv and key in svg


@pytest.mark.parametrize('kind', ['tt2', 's2ut'])
def test_cap_is_absolute_and_does_not_affect_vocoder_or_legacy_policy(kind, monkeypatch):
    variable = 'S2ST_TT2_MONITOR_POLICY' if kind == 'tt2' else 'S2ST_S2UT_MONITOR_POLICY'
    monkeypatch.setenv(variable, json.dumps(POLICY))
    assert absolute_update_limit(kind, 400000) == 1000
    assert absolute_update_limit(kind, 900) == 900
    assert absolute_update_limit('vocoder', 10000) == 10000
    monkeypatch.delenv(variable)
    assert absolute_update_limit(kind, 400000, POLICY) == 1000
    assert absolute_update_limit(kind, 400000, dict(POLICY, version=2)) == 400000


def test_s2ut_real_adam_policy_adoption_preserves_optimizer_and_best_then_cap_prevents_update_1001(tmp_path, monkeypatch):
    from test_s2ut_monitor import install_trainer
    Trainer, loop, _ = install_trainer(monkeypatch)
    for key in ('S2ST_BATCH_PROBE', 'S2ST_TRAIN_ADAPTIVE_BATCH', 'S2ST_TRAIN_DEADLINE'):
        monkeypatch.delenv(key, raising=False)
    policy = dict(POLICY, interval=2)
    work = tmp_path/'work'
    work.mkdir()
    path = work/'checkpoints/checkpoint_last.pt'
    command = ['wrapper', '--save-dir', str(work/'checkpoints'), '--warmup-updates', '2']
    audit, trainer = NS(), Trainer(work)
    iterator = NS(state_dict=lambda:dict(iterations=trainer.updates))
    monkeypatch.setenv('S2ST_S2UT_MONITOR_POLICY', json.dumps(dict(policy, version=2)))
    with runtime_hooks(audit, command):
        trainer.load_checkpoint(None)
        for _ in range(12):
            trainer.train_step(audit)
            _, stop = loop.validate_and_save(trainer.cfg, trainer, audit, iterator, ['dev'], False)
            if stop: break
    assert trainer.updates == 12
    old = torch.load(path, weights_only=True)
    best_bytes = (work/'checkpoints/checkpoint_best.pt').read_bytes()
    monkeypatch.setenv('S2ST_S2UT_MONITOR_POLICY', json.dumps(policy))
    trainer = Trainer(work)
    trainer.cfg.optimization.max_update = 5000
    with runtime_hooks(audit, command+['--restore-file',str(path)]):
        trainer.load_checkpoint(path)
        for key,fields in old['optimizer']['state'].items():
            for field,value in fields.items():
                torch.testing.assert_close(value, trainer.optimizer.state_dict()['state'][key][field],rtol=0,atol=0)

        assert audit.monitor.monitor.history == old['extra_state']['s2st_monitor']['history']
        assert audit.monitor.monitor.bad_evaluations == 0
        assert not audit.monitor.monitor.stopped
        assert trainer.cfg.optimization.max_update == 1000
        trainer.train_step(audit)
        loop.validate_and_save(trainer.cfg, trainer, audit, iterator, ['dev'], False)
    assert (work/'checkpoints/checkpoint_best.pt').read_bytes() == best_bytes
    # Synthetic late-position fixture with real Adam tensors, avoiding 1000 test steps.
    trainer = Trainer(work)
    trainer.updates = 999
    trainer.cfg.optimization.max_update = 5000
    with runtime_hooks(audit, command):
        trainer.load_checkpoint(None)
        trainer.train_step(audit)
        _, stop = loop.validate_and_save(trainer.cfg, trainer, audit, iterator, ['dev'], False)
        assert stop and trainer.updates == 1000
        state = deepcopy(trainer.optimizer.state_dict())
        assert trainer.train_step(audit) is None
        assert loop.validate_and_save(trainer.cfg, trainer, audit, iterator, ['dev'], False)[1]
        assert trainer.updates == 1000
        for key, fields in state['state'].items():
            for field,value in fields.items():
                torch.testing.assert_close(value, trainer.optimizer.state_dict()['state'][key][field],rtol=0,atol=0)


def test_tt2_real_checkpoint_migration_keeps_optimizer_best_and_absolute_limit(tmp_path, monkeypatch):
    from direct_s2st.translatotron2 import train
    from direct_s2st.translatotron2.engine import load_checkpoint, save_checkpoint
    torch.set_num_threads(1)
    torch.manual_seed(10)
    sample = dict(source=torch.randn(1,12,80), source_lengths=torch.tensor([12]),
                  phones=torch.tensor([[3,4,2]]), phone_lengths=torch.tensor([3]),
                  target=torch.randn(1,8,80), target_lengths=torch.tensor([8]))
    class Dataset:
        spec = dict(n_mels=80)
        tokens = list('abcdefg')
        def __init__(self, root, split): assert split in ('train','dev')
        def __len__(self): return 1
        def __getitem__(self,index): return sample
    monkeypatch.setattr(train, 'PreparedDataset', Dataset)
    monkeypatch.setattr(train, 'fingerprint', lambda root: {'fixture':'synthetic'})
    monkeypatch.setattr('direct_s2st.translatotron2.validation.validate', lambda *a: {k:1. for k in Monitor.metrics})
    for key in ('S2ST_TRAIN_STAGING','S2ST_TRAIN_DEADLINE','S2ST_TRAIN_OPTIMIZE','S2ST_TT2_PRACTICAL_WARMUP'):
        monkeypatch.delenv(key, raising=False)
    policy = dict(POLICY, interval=2)
    def run(work, total, policy, restore=False):
        monkeypatch.setenv('S2ST_TT2_MONITOR_POLICY', json.dumps(policy))
        args = ['train','--data-root',str(tmp_path/'data'),'--run-root',str(work),
                '--warmup-updates','4','--save-interval-updates','1','--max-updates',str(total)]
        if restore: args += ['--restore-file',str(work/'checkpoints/checkpoint_last.pt')]
        monkeypatch.setattr('sys.argv',args)
        train.main()
        return load_checkpoint(work/'checkpoints/checkpoint_last.pt')[1]
    work = tmp_path/'tt2'
    old = run(work, 40, dict(policy,version=2))
    assert old['updates'] == 14 and old['monitor_state']['stopped']
    original_best = (work/'checkpoints/checkpoint_best.pt').read_bytes()
    original_load = torch.optim.Adam.load_state_dict
    loaded = []
    def load(self,state):
        result = original_load(self,state)
        loaded.append(deepcopy(self.state_dict()))
        return result
    monkeypatch.setattr(torch.optim.Adam, 'load_state_dict', load)
    migrated = run(work, 15, policy, True)
    for key,fields in old['optimizer']['state'].items():
        for field,value in fields.items():
            torch.testing.assert_close(value,loaded[0]['state'][key][field],rtol=0,atol=0)
    assert migrated['monitor_state']['history'][:14] == old['monitor_state']['history']
    assert migrated['monitor_state']['policy_history'][-1]['adopted_at'] == 14
    assert migrated['monitor_state']['bad_evaluations'] == 0
    assert migrated['monitor_state']['anchors'] == {k:1. for k in Monitor.metrics}
    assert (work/'checkpoints/checkpoint_best.pt').read_bytes() == original_best
    assert migrated['data_fingerprint'] == old['data_fingerprint']
    # Synthetic late-position counter fixture; reuse real model/Adam tensors.
    model,state = load_checkpoint(work/'checkpoints/checkpoint_last.pt')
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    optimizer.load_state_dict(state['optimizer'])
    late = tmp_path/'cap'
    save_checkpoint(late/'checkpoints/checkpoint_last.pt',model,optimizer,999,Dataset.tokens,
                    state['data_fingerprint'],monitor_state=Monitor(policy,4,origin=999).state())
    capped = run(late, 5000, policy, True)
    assert capped['updates'] == 1000 and not capped['monitor_state']['stopped']
    before = (late/'checkpoints/checkpoint_last.pt').read_bytes()
    original_step = torch.optim.Adam.step
    monkeypatch.setattr(torch.optim.Adam, 'step', lambda *a,**kw: pytest.fail('optimizer update beyond absolute cap'))
    run(late, 5000, policy, True)
    assert (late/'checkpoints/checkpoint_last.pt').read_bytes() == before
    monkeypatch.setattr(torch.optim.Adam, 'step', original_step)


@pytest.mark.parametrize('kind,cls', [('tt2',Monitor),('s2ut',S2UTMonitor)])
def test_completed_cap_restores_without_launching_any_trainer(kind, cls, tmp_path, monkeypatch):
    from direct_s2st.colab import run_session, publish
    from direct_s2st.session_runtime import run_continuous
    from direct_s2st.hashing import sha256_file
    for key in ('S2ST_DRIVE_SAFE','S2ST_TT2_MONITOR_POLICY','S2ST_S2UT_MONITOR_POLICY'):
        monkeypatch.delenv(key,raising=False)
    work, backup = tmp_path/'work',tmp_path/'drive'
    work.mkdir()
    config = dict(kind=kind,command=['fixture','{updates}'],checkpoint='checkpoint.pt',resume_args=[],
                  identity_files=[str(tmp_path/'input')])
    (tmp_path/'input').write_text('same dataset')
    identity = dict(config=config,work=str(work),inputs={str(tmp_path/'input'):sha256_file(tmp_path/'input')})
    (work/'checkpoint.pt').write_text('1000')
    cls(dict(POLICY,interval=10),400,origin=1000).write_reports(work)
    publish(work,backup,identity,1000)
    assert run_session(config,work=work,backup=backup,total=5000,resume=True,confirm_training=True,
                       inspect_checkpoint=lambda p,k:int(p.read_text()),
                       runner=lambda *a,**kw:pytest.fail('trainer launched at cap')) == dict(status='COMPLETE',durable_updates=1000)
    assert run_continuous(config,work=work,backup=backup,identity=identity,completed=1000,total=5000,
                          seconds=0,inspect_checkpoint=None,publish=None,
                          popen=lambda *a,**kw:pytest.fail('resident trainer launched at cap')) == dict(status='COMPLETE',durable_updates=1000)


@pytest.mark.parametrize('cls', [Monitor,S2UTMonitor])
def test_production_interval_adoption_at_610_starts_future_checks_at_620(cls, tmp_path):
    policy = dict(POLICY,interval=10)
    monitor = cls(dict(policy,version=2),400,origin=550)
    for update in range(551,611):
        observe(monitor,update,baseline(cls) if monitor.due(update) else None,tmp_path)
    assert monitor.stopped
    old = monitor.state()
    monitor = cls.restore(policy,old,610,400,tmp_path)
    assert monitor.bad_evaluations == 0 and not monitor.stopped
    for update in range(611,661):
        observe(monitor,update,baseline(cls) if monitor.due(update) else None,tmp_path)
        assert monitor.stopped == (update == 660)
        if update == 619: assert monitor.bad_evaluations == 0
        if update == 620: assert monitor.bad_evaluations == 1
    assert monitor.history[:60] == old['history']


@pytest.mark.parametrize('cls', [Monitor,S2UTMonitor])
def test_repeated_auxiliary_improvement_never_extends_hard_limit(cls, tmp_path):
    policy = dict(POLICY,interval=10)
    monitor = cls(policy,400,origin=380)
    metric = 'phone_loss' if cls is Monitor else 'source_letter'
    current = baseline(cls)
    limit = absolute_update_limit('tt2' if cls is Monitor else 's2ut',5000,policy)
    assert limit == 1000
    for update in range(381,limit+1):
        if update > 400 and monitor.due(update): current[metric] *= .98
        observe(monitor,update,current if monitor.due(update) else None,tmp_path)
        assert not monitor.stopped
    assert monitor.update == 1000 and monitor.bad_evaluations == 0


def test_s2ut_native_cap_returns_before_model_optimizer_and_empty_gradient_audit(tmp_path, monkeypatch, capsys):
    from types import ModuleType
    import sys
    from direct_s2st.s2ut import fairseq_train
    dependency = ModuleType('editdistance')
    dependency.eval = lambda *a:1
    monkeypatch.setitem(sys.modules, 'editdistance', dependency)
    policy = dict(POLICY, interval=10)
    path = tmp_path/'checkpoints/checkpoint_last.pt'
    path.parent.mkdir()
    # Explicit synthetic fairseq-shaped tensors; the cap guard is real.
    state = dict(model={name+'.weight':torch.ones(1) for name in
        ('encoder','source_letter_decoder','target_letter_decoder','decoder_target_ctc_decoder')},
        optimizer_history=[dict(num_updates=1000)], last_optimizer_state={'state':{0:{'exp_avg':torch.zeros(1)}}},
        extra_state=dict(s2st_recipe_fingerprint='synthetic',s2st_monitor=S2UTMonitor(policy,400,origin=1000).state()))
    torch.save(state,path)
    before = path.read_bytes()
    command = ['wrapper','fixture','--max-update','5000','--save-dir',str(path.parent),'--restore-file',str(path)]
    monkeypatch.setenv('S2ST_S2UT_MONITOR_POLICY', json.dumps(policy))
    monkeypatch.setattr(fairseq_train, 'prepare_research_metadata',lambda *a:(command,(tmp_path,dict(training_fingerprint='synthetic'))))
    monkeypatch.setattr('sys.argv',command)
    fairseq_train.main()
    assert path.read_bytes() == before
    assert 'completed=1000 absolute_max_update=1000 optimizer_steps=0' in capsys.readouterr().err
