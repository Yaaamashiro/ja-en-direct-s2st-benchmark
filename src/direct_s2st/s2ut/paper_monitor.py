"""Sparse loss reporting/dev-unit selection with NO early stopping or short cap."""
from copy import deepcopy
import math
from pathlib import Path

from ..translatotron2.monitor import Monitor
from .paper_recipe import validate_report_policy


class PaperMonitor:
    version = 's2ut-paper-report-v1'
    metrics = ('loss','unit_loss','source_letter','target_letter','decoder_target_ctc')
    metric, score_key = 'teacher_forced_dev_unit_loss', 'unit_loss'
    title = 'S2UT paper recipe: losses (report only; no early stopping)'
    absolute_max_update = 400000
    stopped = False
    save_best, write_reports, svg = Monitor.save_best, Monitor.write_reports, Monitor.svg

    def __init__(self, policy, warmup, *, origin=0):
        self.policy = deepcopy(validate_report_policy(policy))
        if warmup != 10000 or origin != 0:
            raise ValueError('paper S2UT requires new optimizer history and warmup 10000')
        self.warmup, self.origin, self.update = warmup, origin, 0
        self.history = []
        self.best_value = self.best_update = self.best_sha256 = None

    def due(self, update):
        return update % self.policy['interval'] == 0

    def observe(self, update, train, dev, lr):
        if (type(update) is not int or update != self.update+1 or update > self.absolute_max_update or not 0 < lr < float('inf')
                or (dev is not None) != self.due(update)):
            raise ValueError('paper monitor optimizer/evaluation schedule mismatch')
        if train is None:
            raise ValueError('missing paper monitor train losses')
        for values in (train, dev):
            if values is not None and (set(values) != set(self.metrics) or
                    any(not math.isfinite(float(values[k])) or values[k] < 0 for k in self.metrics)):
                raise ValueError('invalid paper monitor loss')
        self.update = update
        if update % self.policy['train_interval'] == 0 or dev is not None:
            self.history.append(dict(update=update, train=dict(train), dev=deepcopy(dev), learning_rate=lr))
        improved = dev is not None and (self.best_value is None or dev['unit_loss'] < self.best_value)
        if improved:
            self.best_value, self.best_update, self.best_sha256 = float(dev['unit_loss']), update, None
        return improved

    def state(self):
        return dict(version=self.version, metric=self.metric, policy=deepcopy(self.policy),
            warmup=self.warmup, origin=0, update=self.update, history=deepcopy(self.history),
            best_value=self.best_value, best_update=self.best_update, best_sha256=self.best_sha256,
            stopped=False, early_stopping=False, absolute_max_update=self.absolute_max_update)

    @classmethod
    def restore(cls, requested, saved, completed, warmup, work):
        if saved is None:
            if completed:
                raise ValueError('paper S2UT cannot adopt an old checkpoint; use a new run')
            return cls(requested, warmup)
        if saved.get('version') != cls.version or (requested is not None and requested != saved['policy']):
            raise ValueError('paper S2UT report policy/checkpoint changed; saved run retained')
        result = cls(saved['policy'], warmup)
        if type(completed) is not int or not 0 <= completed <= cls.absolute_max_update:
            raise ValueError('invalid paper monitor completed updates')
        if [r['update'] for r in saved['history']] != list(range(result.policy['train_interval'], completed+1, result.policy['train_interval'])):
            raise ValueError('incomplete paper monitor history')
        previous = 0
        for row in saved['history']:
            update = row['update']
            if type(update) is not int or not previous < update <= completed or update % result.policy['train_interval']:
                raise ValueError('invalid paper monitor history')
            result.update = update-1
            result.observe(update, row['train'], row['dev'], row['learning_rate'])
            previous = update
        result.update = completed
        result.best_sha256 = saved['best_sha256']
        if result.state() != saved:
            raise ValueError('paper monitor differs from optimizer checkpoint')
        if result.best_update is not None:
            from ..hashing import sha256_file
            best = Path(work)/'checkpoints/checkpoint_best.pt'
            if not result.best_sha256 or not best.is_file() or sha256_file(best) != result.best_sha256:
                raise ValueError('paper monitor best checkpoint missing or damaged')
        return result
