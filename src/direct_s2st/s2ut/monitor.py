"""Checkpointed unit-loss selection around the pinned fairseq training loop."""
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import random
import sys
import tempfile

from ..progress import operation
from ..translatotron2.monitor import Monitor, validate_policy


class S2UTMonitor(Monitor):
    metrics = ('loss', 'unit_loss', 'source_letter', 'target_letter', 'decoder_target_ctc')
    version = 's2ut-dev-unit-monitor-v1'
    metric = 'teacher_forced_dev_unit_loss'
    score_key = 'unit_loss'
    absolute_max_update = 3000
    title = 'S2UT: train / teacher-forced dev losses (bits; not BLEU)'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.best_total_value = self.best_total_update = self.total_anchor = None
        self.total_bad_evaluations = 0

    def reset_total_stopping(self):
        # Policy adoption changes future stopping only, retaining raw bests.
        self.total_anchor, self.total_bad_evaluations = self.best_total_value, 0

    def observe(self, update, train, dev, lr):
        improved = super().observe(update, train, dev, lr)
        if dev is not None:
            value = float(dev['loss'])
            if self.best_total_value is None or value < self.best_total_value:
                self.best_total_value, self.best_total_update = value, update
            if self.policy.get('version') == 2:
                meaningful = (self.total_anchor is None or
                    (value < self.total_anchor and value <= self.total_anchor *
                     (1-self.policy['min_relative_improvement'])))
                adoption = getattr(self, '_policy_history', [])
                start = max(self.warmup, adoption[-1]['adopted_at'] if adoption else self.origin)
                if meaningful or update <= start:
                    self.reset_total_stopping()
                else:
                    self.total_bad_evaluations += 1
                self.stopped = (update > start and self.bad_evaluations >= self.policy['patience'] and
                                self.total_bad_evaluations >= self.policy['patience'])
            elif self.policy.get('version') == 3:
                self.total_anchor = self.anchors['loss']
                self.total_bad_evaluations = self.bad_evaluations
        return improved  # Unit loss alone selects the best inference checkpoint.

    def state(self):
        result = super().state()
        if self.policy.get('version') in (2, 3):
            result.update(stopping_metrics=['unit_loss', 'loss'],
                best_total_value=self.best_total_value, best_total_update=self.best_total_update,
                total_anchor=self.total_anchor, total_bad_evaluations=self.total_bad_evaluations)
            if self.policy.get('version') == 3:
                result.update(stopping_metrics=list(self.metrics), total_anchor=self.anchors['loss'],
                              total_bad_evaluations=self.bad_evaluations)
        return result


def policy_from_env():
    value = os.environ.get('S2ST_S2UT_MONITOR_POLICY')
    return validate_policy(json.loads(value)) if value else None


class LossSums:
    """Sum losses and their own denominators, not averages of batch averages."""
    def __init__(self):
        self.sums = {key: 0. for key in S2UTMonitor.metrics}
        self.sizes = dict(self.sums)

    def record(self, output, main):
        total, size, logs = output
        values = {'loss': (total, size), 'unit_loss': (main, size)}
        for key in S2UTMonitor.metrics[2:]:
            item = logs['multitask'][key]
            values[key] = item['loss'], item['sample_size']
        for key, (loss, count) in values.items():
            loss, count = float(loss.detach()) if hasattr(loss, 'detach') else float(loss), float(count)
            if not math.isfinite(loss) or loss < 0 or not math.isfinite(count) or count <= 0:
                raise ValueError('invalid S2UT monitor loss/sample_size: '+key)
            self.sums[key] += loss
            self.sizes[key] += count

    def result(self):
        if any(size <= 0 for size in self.sizes.values()):
            raise ValueError('S2UT monitor did not observe all criterion losses')
        return {key: self.sums[key]/self.sizes[key]/math.log(2) for key in self.sums}


class Controller:
    def __init__(self, requested, work, warmup):
        self.requested, self.work, self.warmup = requested, Path(work), warmup
        self.monitor = None
        self.collector = None
        self.pending = None
        self.writing_best = False

    def record(self, output, main):
        if self.collector is not None:
            self.collector.record(output, main)

    def restore(self, trainer, extra):
        self.monitor = S2UTMonitor.restore(self.requested, (extra or {}).get('s2st_monitor'),
                                          trainer.get_num_updates(), self.warmup, self.work)
        from ..translatotron2.monitor import absolute_update_limit
        if self.monitor and self.monitor.policy.get('version') == 3:
            configured = trainer.cfg.optimization.max_update or self.monitor.absolute_max_update
            trainer.cfg.optimization.max_update = absolute_update_limit('s2ut', configured, self.monitor.policy)


@contextmanager
def runtime_hooks(audit, command):
    # Calibration is disposable train-only work; it must never evaluate dev.
    if os.environ.get('S2ST_BATCH_PROBE'):
        yield
        return
    from ..recipes import option
    requested = policy_from_env()
    work = Path(option(command, '--save-dir')).parent
    restore_file = option(command, '--restore-file')
    # The report is restored along with the last checkpoint by run_session.
    if requested is None and not (work/'training-monitor.json').is_file():
        yield
        return
    if os.environ.get('S2ST_TRAIN_ADAPTIVE_BATCH') == '1':
        raise ValueError('S2UT monitoring requires fixed batches (startup calibration is supported)')
    from fairseq.trainer import Trainer
    from fairseq_cli import train as loop
    from fairseq import checkpoint_utils
    import numpy as np
    import torch
    from ..train_runtime import stopping

    controller = Controller(requested, work, int(option(command, '--warmup-updates', '0')))
    original_step, original_load, original_save = Trainer.train_step, Trainer.load_checkpoint, Trainer.save_checkpoint
    original_validate_save = loop.validate_and_save

    def load(self, *args, **kwargs):
        extra = original_load(self, *args, **kwargs)
        controller.restore(self, extra)
        return extra

    def step(self, *args, **kwargs):
        if self.data_parallel_world_size != 1:
            raise ValueError('S2UT monitoring supports a single training process')
        if controller.monitor is None:
            if restore_file:
                raise ValueError('S2UT checkpoint monitor was not restored')
            controller.restore(self, None)
        before = self.get_num_updates()
        if controller.monitor.stopped or (controller.monitor.policy.get('version') == 3 and
                                          before >= self.cfg.optimization.max_update):
            # Upstream's epoch loop calls train_step once before its limit check.
            # A resumed checkpoint already at the cap must not optimize again.
            return None
        controller.collector = LossSums()
        try:
            result = original_step(self, *args, **kwargs)
            after = self.get_num_updates()
            if after == before + 1:
                controller.pending = controller.collector.result()
            elif after != before:
                raise ValueError('unexpected S2UT optimizer update count')
            return result
        finally:
            controller.collector = None

    def save(self, filename, extra_state):
        extra = dict(extra_state)
        if controller.monitor and not controller.writing_best:
            extra['s2st_monitor'] = controller.monitor.state()
        return original_save(self, filename, extra)

    @operation('s2ut: monitored validation and checkpoint')
    def validate_save(cfg, trainer, task, epoch_itr, valid_subsets, end_of_epoch):
        monitor = controller.monitor
        if monitor is None:
            return original_validate_save(cfg, trainer, task, epoch_itr, valid_subsets, end_of_epoch)
        if (list(valid_subsets) != ['dev'] or cfg.dataset.disable_validation or
                cfg.dataset.max_valid_steps is not None or cfg.checkpoint.no_save or
                cfg.checkpoint.no_last_checkpoints or cfg.checkpoint.no_save_optimizer_state or
                cfg.checkpoint.write_checkpoints_asynchronously):
            raise ValueError('S2UT monitor requires full dev and synchronous complete last checkpoints')
        update = trainer.get_num_updates()
        if update == monitor.update:
            return [None], monitor.stopped or (cfg.optimization.max_update > 0 and update >= cfg.optimization.max_update)
        dev = None
        if monitor.due(update):
            controller.collector = LossSums()
            py_rng, np_rng = random.getstate(), np.random.get_state()
            try:
                with torch.random.fork_rng():
                    loop.validate(cfg, trainer, task, epoch_itr, ['dev'])
                dev = controller.collector.result()
            finally:
                random.setstate(py_rng)
                np.random.set_state(np_rng)
                controller.collector = None
        if controller.pending is None:
            raise ValueError('missing completed S2UT train losses')
        improved = monitor.observe(update, controller.pending, dev, float(trainer.get_lr()))
        controller.pending = None
        should_stop = (monitor.stopped or stopping() or
                       (cfg.optimization.max_update > 0 and update >= cfg.optimization.max_update) or
                       (cfg.optimization.stop_time_hours > 0 and
                        trainer.cumulative_training_time()/3600 > cfg.optimization.stop_time_hours))
        if dev is not None and monitor.policy.get('version') != 3:
            print(f'[training-monitor] S2UT update={update} dev_unit_loss={dev["unit_loss"]:.6g} '
                  f'dev_loss={dev["loss"]:.6g} '
                  f'best_update={monitor.best_update} unit_bad_evaluations={monitor.bad_evaluations}/'
                  f'{monitor.policy["patience"]} total_bad_evaluations={monitor.total_bad_evaluations}/'
                  f'{monitor.policy["patience"]} stopping_metrics={(["unit_loss", "loss"] if monitor.policy.get("version") == 2 else ["unit_loss"])} '
                  f'warmup_guard={update <= monitor.warmup} '
                  f'early_stopped={monitor.stopped}', file=sys.stderr, flush=True)
        save_due = (should_stop or dev is not None or end_of_epoch or
                    (cfg.checkpoint.save_interval_updates > 0 and update % cfg.checkpoint.save_interval_updates == 0))
        if save_due:
            if improved:
                controller.writing_best = True
                try:
                    def save_best(path):
                        path.parent.mkdir(parents=True, exist_ok=True)
                        fd, name = tempfile.mkstemp(dir=path.parent, suffix='.pt.tmp')
                        os.close(fd)
                        try:
                            trainer.consolidate_optimizer()
                            trainer.save_checkpoint(name, dict(train_iterator=epoch_itr.state_dict(),
                                                               val_loss=dev['unit_loss']))
                            os.replace(name, path)
                        finally:
                            Path(name).unlink(missing_ok=True)
                    monitor.save_best(work, save_best)
                finally:
                    controller.writing_best = False
            monitor.write_reports(work, render=dev is not None or should_stop)
            if should_stop:
                # Existing resident snapshot hook must force the final backup.
                trainer.cfg.optimization.max_update = update
            # Best selection belongs to this monitor. Prevent upstream from
            # overwriting it with a different metric or a rounded/tied value.
            path = checkpoint_utils.save_checkpoint(cfg.checkpoint, trainer, epoch_itr, None)
            if path is not None and hasattr(task, 'post_save'):
                task.post_save(path, update)
        return [dev['unit_loss'] if dev else None], should_stop

    audit.monitor = controller
    Trainer.train_step, Trainer.load_checkpoint, Trainer.save_checkpoint = step, load, save
    loop.validate_and_save = validate_save
    try:
        yield
    finally:
        Trainer.train_step, Trainer.load_checkpoint, Trainer.save_checkpoint = original_step, original_load, original_save
        loop.validate_and_save = original_validate_save
        del audit.monitor
