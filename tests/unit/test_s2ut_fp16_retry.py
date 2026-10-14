"""CPU Adam plus the actual pinned trainer/scaler control flow (no GPU claim)."""
import ast
import contextlib
import importlib.util
import logging
from pathlib import Path
import random
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

from direct_s2st.s2ut.fairseq_train import GROUPS, TrainingAudit
from direct_s2st.s2ut.monitor import LossSums
from direct_s2st.s2ut.performance import completed_step
from direct_s2st.train_runtime import Microbatches

ROOT = Path(__file__).resolve().parents[2]


def pinned_step():
    source = ROOT/'third_party/fairseq/fairseq/trainer.py'
    cls = next(n for n in ast.parse(source.read_text(encoding='utf-8')).body
               if isinstance(n, ast.ClassDef) and n.name == 'Trainer')
    node = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'train_step')
    node.decorator_list = []
    namespace = dict(torch=torch, contextlib=contextlib,
        logger=logging.getLogger('pinned-fixture'), utils=NS(has_parameters=lambda c: False),
        metrics=NS(log_start_time=lambda *a, **k: None, log_stop_time=lambda *a, **k: None,
                   log_scalar=lambda *a, **k: None))
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), namespace)
    return namespace['train_step']


def fixture(monkeypatch, *, scale=128., failure=None):
    monkeypatch.setenv('S2ST_TRAIN_FIXED_MICROBATCH', '0')
    monkeypatch.setenv('S2ST_TRAIN_ADAPTIVE_BATCH', '0')
    # Only the pinned overflow handler's dummy .cuda() scalar is redirected.
    monkeypatch.setattr(torch.Tensor, 'cuda', lambda self, *a, **k: self)
    path = ROOT/'third_party/fairseq/fairseq/optim/dynamic_loss_scaler.py'
    spec = importlib.util.spec_from_file_location('pinned_scaler_fixture', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = torch.nn.Module()
    for name in GROUPS:
        setattr(model, name, torch.nn.Linear(1, 1))
    model.register_buffer('seen', torch.zeros(1))
    criterion = torch.nn.Module()
    criterion.register_buffer('seen', torch.zeros(1))
    audit = TrainingAudit()
    audit.attach(model)
    audit.monitor = NS(collector=LossSums(), pending=None)
    attempts, calls = [], []

    class Optimizer:
        def __init__(self):
            self.inner = torch.optim.Adam(model.parameters(), lr=.001)
            self.scaler = module.DynamicLossScaler(init_scale=scale)
            self.step_calls = 0
        def zero_grad(self): self.inner.zero_grad()
        def all_reduce_grads(self, model): pass
        def multiply_grads(self, factor):
            for parameter in model.parameters():
                parameter.grad.mul_(factor)
        def clip_grad_norm(self, clip):
            if self.scaler.loss_scale > 64:
                # Emulate half-gradient saturation; actual scaler/handler below
                # must decrease scale, return None, and never enter Adam.
                audit.gradients = {key: torch.tensor(float('inf')) for key in GROUPS}
                self.scaler.check_overflow(float('inf'))
            return torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        def step(self):
            self.step_calls += 1
            if failure == 'optimizer_oom':
                raise RuntimeError('out of memory during optimizer')
            self.inner.step()
            self.scaler.update()
        def state_dict(self): return dict(adam=self.inner.state_dict(), scaler=vars(self.scaler).copy())
        def load_state_dict(self, saved):
            self.inner.load_state_dict(saved['adam'])
            vars(self.scaler).update(saved['scaler'])

    optimizer = Optimizer()
    trainer = NS(model=model, criterion=criterion, optimizer=optimizer, updates=0, cuda=False,
        tpu=False, is_fsdp=False, data_parallel_world_size=1,
        cfg=NS(common=NS(fp16=True, amp=False, empty_cache_freq=0),
               ema=NS(store_ema=False), optimization=NS(use_bmuf=False, clip_norm=10.),
               distributed_training=NS(ddp_backend='pytorch_ddp', distributed_world_size=1)))
    trainer._s2st_tuning = Microbatches()
    trainer.get_num_updates = lambda: trainer.updates
    trainer.set_num_updates = lambda count: setattr(trainer, 'updates', count)
    trainer.zero_grad = optimizer.zero_grad
    trainer.clip_grad_norm = optimizer.clip_grad_norm
    trainer._sync_stats = lambda: False
    trainer._prepare_sample = lambda sample: (sample, False)
    trainer._check_grad_norms = lambda norm: None
    trainer._log_oom = lambda error: None
    trainer._set_seed = lambda: torch.manual_seed(1+trainer.updates)
    trainer._reduce_and_log_stats = lambda outputs, *a: {'loss': 1., 'count': len(outputs)}

    def train_step(sample, model, criterion, optimizer, update_num, ignore_grad, **kwargs):
        calls.append(tuple(sample['id']))
        attempts.append((float(model.seen), float(criterion.seen), random.random(), float(np.random.rand())))
        model.seen.add_(1)
        criterion.seen.add_(1)
        values = [torch.nn.functional.dropout(getattr(model, name)(torch.ones(2, 1)), .1).square().sum()
                  for name in GROUPS]
        total, main = sum(values), sum(values[:2])
        logs = {'multitask': {key: dict(loss=value.detach(), sample_size=2) for key, value in
                             zip(('source_letter', 'target_letter', 'decoder_target_ctc'), values[2:])}}
        audit.record(total, main.detach(), logs)
        audit.monitor.collector.record((total, 2, logs), main.detach())
        total.backward()
        return total, 2, logs
    trainer.task = NS(train_step=train_step, optimizer_step=lambda opt, **k: opt.step())
    original = pinned_step()
    def step(pieces):
        if failure == 'no_update': return None
        if failure == 'backward_oom': raise RuntimeError('out of memory in backward')
        if failure == 'minimum': raise FloatingPointError('Minimum loss scale reached')
        if failure == 'changed_after_optimizer':
            optimizer.step()
            optimizer.scaler.loss_scale /= 2
            return None
        return original(trainer, pieces, raise_oom=True)
    samples = [dict(id=[1, 2], nsentences=2, ntokens=6), dict(id=[3, 4], nsentences=2, ntokens=6)]
    return trainer, audit, step, samples, attempts, calls


def test_pinned_overflow_retries_same_batch_and_discards_failed_evidence(monkeypatch, capsys):
    torch.manual_seed(7)
    random.seed(8)
    np.random.seed(9)
    trainer, audit, step, samples, attempts, calls = fixture(monkeypatch)
    optimizer = trainer.optimizer
    assert 'step' not in vars(optimizer)
    assert completed_step(trainer, samples, step, audit) == dict(loss=1., count=2)
    assert trainer.updates == optimizer.step_calls == 1 and optimizer.scaler.loss_scale == 64
    assert calls == [(1, 2), (3, 4)]*2 and attempts[:2] == attempts[2:]
    assert trainer.model.seen == trainer.criterion.seen == 2
    assert audit.loss_count == len(audit.losses) == 2 and audit.result()['status'] == 'PASS'
    assert audit.monitor.collector.sizes == {key:4. for key in LossSums().sizes}
    assert 'step' not in vars(optimizer)
    assert '[fp16-retry]' in capsys.readouterr().err
    state = trainer.model.state_dict()
    adam = optimizer.inner.state_dict()
    # Identical successful pass at scale64, including buffers and CPU Adam.
    torch.manual_seed(7)
    random.seed(8)
    np.random.seed(9)
    other, evidence, other_step, batch, _, _ = fixture(monkeypatch, scale=64)
    completed_step(other, batch, other_step, evidence)
    for key, value in state.items():
        torch.testing.assert_close(value, other.model.state_dict()[key], rtol=0, atol=0)
    for parameter, values in adam['state'].items():
        for key, value in values.items():
            torch.testing.assert_close(value, other.optimizer.inner.state_dict()['state'][parameter][key], rtol=0, atol=0)
    assert audit.losses == evidence.losses


@pytest.mark.parametrize('failure', ['no_update', 'backward_oom', 'optimizer_oom', 'minimum', 'changed_after_optimizer'])
def test_unknown_failures_and_optimizer_phase_never_retry(monkeypatch, failure, capsys):
    trainer, audit, step, samples, _, calls = fixture(monkeypatch, scale=64, failure=failure)
    with pytest.raises((RuntimeError, FloatingPointError)):
        completed_step(trainer, samples, step, audit)
    assert trainer.updates == 0 and 'step' not in vars(trainer.optimizer)
    assert '[fp16-retry]' not in capsys.readouterr().err
    assert len(calls) <= 2


def test_retry_budget_is_bounded_and_retains_lowered_scaler(monkeypatch, capsys):
    trainer, audit, step, samples, _, calls = fixture(monkeypatch, scale=256)
    with pytest.raises(RuntimeError, match='retry limit'):
        completed_step(trainer, samples, step, audit, max_retries=1)
    assert len(calls) == 4 and trainer.updates == trainer.optimizer.step_calls == 0
    assert trainer.optimizer.scaler.loss_scale == 64 and 'step' not in vars(trainer.optimizer)
    assert capsys.readouterr().err.count('[fp16-retry]') == 1


def test_lowered_scale_and_adam_resume_from_trusted_checkpoint(monkeypatch, tmp_path):
    trainer, audit, step, samples, _, _ = fixture(monkeypatch)
    completed_step(trainer, samples, step, audit)
    checkpoint = tmp_path/'last.pt'
    torch.save(dict(model=trainer.model.state_dict(), criterion=trainer.criterion.state_dict(),
                    optimizer=trainer.optimizer.state_dict(), updates=trainer.updates), checkpoint)
    completed_step(trainer, samples, step, audit)
    expected = trainer.model.state_dict()
    restored, evidence, resumed_step, batch, _, calls = fixture(monkeypatch)
    saved = torch.load(checkpoint, weights_only=True)
    restored.model.load_state_dict(saved['model'])
    restored.criterion.load_state_dict(saved['criterion'])
    restored.optimizer.load_state_dict(saved['optimizer'])
    restored.updates = saved['updates']
    assert restored.optimizer.scaler.loss_scale == 64
    completed_step(restored, batch, resumed_step, evidence)
    assert restored.updates == 2 and len(calls) == 2
    for key, value in expected.items():
        torch.testing.assert_close(value, restored.model.state_dict()[key], rtol=0, atol=0)
