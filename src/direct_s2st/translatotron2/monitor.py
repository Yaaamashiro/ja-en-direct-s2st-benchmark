"""Resumable dev-loss stopping policy; not free-running BLEU evaluation."""
from copy import deepcopy
import csv
import io
import json
import math
import os
import sys
from pathlib import Path

from ..io import atomic_write_json, atomic_write_text, atomic_write_bytes
from ..progress import operation


METRICS = ('loss', 'mel_loss', 'phone_loss', 'duration_loss')
VERSION = 'tt2-dev-loss-monitor-v1'


def resume_schedule(base_warmup, completed, checkpoint, requested=None):
    """An explicit practical continuation; retain the verified base fingerprint."""
    saved = checkpoint.get('schedule_transition') if checkpoint else None
    if requested is not None and (type(requested) is not int or requested < 1):
        raise ValueError('practical warmup must be a positive integer')
    if saved is not None and saved.get('version') == 1:
        if (set(saved) != {'version', 'base_warmup', 'warmup', 'adopted_at', 'mode'} or
                saved['version'] != 1 or saved['mode'] != 'paper_practical' or
                saved['base_warmup'] != base_warmup or
                type(saved['adopted_at']) is not int or
                type(saved['warmup']) is not int or
                not 0 < saved['adopted_at'] < saved['warmup'] < base_warmup or
                saved['adopted_at'] > completed):
            raise ValueError('invalid saved warmup transition')
        changes = [dict(warmup=saved['warmup'], adopted_at=saved['adopted_at'])]
        effective = saved['warmup']
    else:
        changes, effective = [], base_warmup
    # An old trainer rejects version 2 instead of silently changing the guard.
    revised = saved if saved is not None and saved.get('version') != 1 else None
    if revised is not None:
        if set(revised) != {'version', 'base_warmup', 'changes', 'warmup', 'adopted_at', 'mode'} or revised['version'] != 2 or revised['base_warmup'] != base_warmup or revised['mode'] != 'paper_practical':
            raise ValueError('invalid saved warmup transition')
        changes = deepcopy(revised['changes'])
        previous, adopted = base_warmup, -1
        if not isinstance(changes, list) or not changes:
            raise ValueError('invalid saved warmup transition')
        for change in changes:
            if (set(change) != {'warmup', 'adopted_at'} or type(change['warmup']) is not int or
                    type(change['adopted_at']) is not int or
                    not 0 < change['warmup'] < previous or
                    not adopted < change['adopted_at'] < previous or change['adopted_at'] > completed):
                raise ValueError('invalid saved warmup transition')
            previous, adopted = change['warmup'], change['adopted_at']
        if revised['warmup'] != previous or revised['adopted_at'] != adopted:
            raise ValueError('invalid saved warmup transition')
        effective = previous
    if requested is None or requested == effective:
        return effective, deepcopy(revised or saved)
    if not 0 <= completed < effective or not 0 < requested < effective:
        raise ValueError('warmup shortening requires an update before the existing warmup ends and a smaller endpoint')
    changes.append(dict(warmup=requested, adopted_at=completed))
    return requested, dict(version=2, base_warmup=base_warmup, changes=changes,
                           warmup=requested, adopted_at=completed, mode='paper_practical')


def policy_from_env():
    value = os.environ.get('S2ST_TT2_MONITOR_POLICY')
    return json.loads(value) if value else None


def validate_policy(policy):
    fields = {'interval', 'patience', 'min_relative_improvement'}
    if not isinstance(policy, dict) or set(policy) not in (fields, fields | {'version'}):
        raise ValueError('invalid TT2 monitor policy fields')
    if 'version' in policy and (type(policy['version']) is not int or policy['version'] != 2):
        raise ValueError('invalid TT2 monitor policy version')
    for key in ('interval', 'patience'):
        if type(policy[key]) is not int or policy[key] < 1:
            raise ValueError('monitor interval/patience must be positive integers')
    delta = policy['min_relative_improvement']
    if isinstance(delta, bool) or not isinstance(delta, (int, float)) or not math.isfinite(delta) or not 0 <= delta < 1:
        raise ValueError('monitor relative improvement must be finite in [0, 1)')
    return deepcopy(policy)


class Monitor:
    metrics = METRICS
    version = VERSION
    metric = 'teacher_forced_dev_loss'
    score_key = 'loss'
    title = 'TT2: train / teacher-forced dev loss (not BLEU)'

    def __init__(self, policy, warmup, *, origin=0):
        self.policy = validate_policy(policy)
        if type(warmup) is not int or warmup < 0 or type(origin) is not int or origin < 0:
            raise ValueError('invalid monitor warmup/origin')
        self.warmup, self.origin = warmup, origin
        self.history = []
        self.best_value = self.best_update = self.anchor = self.best_sha256 = None
        self.bad_evaluations = 0
        self.stopped = False

    @property
    def update(self):
        return self.history[-1]['update'] if self.history else self.origin

    def due(self, update):
        return update % self.policy['interval'] == 0

    def observe(self, update, train, dev, lr):
        if self.stopped or type(update) is not int or update != self.update + 1:
            raise ValueError('monitor updates must follow the restored optimizer checkpoint')
        for values in (train, dev):
            if values is not None and any(not math.isfinite(float(values[k])) or float(values[k]) < 0 for k in self.metrics):
                raise ValueError('monitor requires finite nonnegative losses')
        if not math.isfinite(lr) or lr <= 0 or (dev is not None) != self.due(update):
            raise ValueError('monitor learning rate/evaluation schedule mismatch')
        self.history.append(dict(update=update, train={k: float(train[k]) for k in self.metrics},
                                 dev={k: float(dev[k]) for k in self.metrics} if dev is not None else None,
                                 learning_rate=lr))
        improved = False
        if dev is not None:
            value = float(dev[self.score_key])
            improved = self.best_value is None or value < self.best_value
            if improved:
                self.best_value, self.best_update = value, update
                self.best_sha256 = None  # Caller must publish this exact checkpoint.
            modern = self.policy.get('version') == 2 and self.version in (VERSION, 's2ut-dev-unit-monitor-v1')
            meaningful = (self.anchor is None or
                          (value < self.anchor and value <= self.anchor * (1-self.policy['min_relative_improvement']))) if modern else (
                          self.anchor is None or value < self.anchor * (1-self.policy['min_relative_improvement']))
            adoption = getattr(self, '_policy_history', [])
            start = max(self.warmup, adoption[-1]['adopted_at'] if adoption else self.origin) if modern else self.warmup
            if meaningful or update <= start:
                # v2 starts after warmup/adoption, using the lowest observed
                # dev loss, not the last (possibly worse) warmup evaluation.
                self.anchor, self.bad_evaluations = self.best_value if modern else value, 0
            else:
                self.bad_evaluations += 1
            self.stopped = update > start and self.bad_evaluations >= self.policy['patience']
        return improved

    def state(self):
        result = dict(version=self.version, metric=self.metric, policy=deepcopy(self.policy),
                    warmup=self.warmup, origin=self.origin, history=deepcopy(self.history),
                    best_value=self.best_value, best_update=self.best_update, best_sha256=self.best_sha256,
                    anchor=self.anchor, bad_evaluations=self.bad_evaluations, stopped=self.stopped)
        if getattr(self, '_warmup_history', None):
            result['warmup_history'] = deepcopy(self._warmup_history)
        if getattr(self, '_policy_history', None):
            result['policy_history'] = deepcopy(self._policy_history)
        return result

    @classmethod
    def restore(cls, requested, saved, completed, warmup, work):
        if saved is None:
            return cls(requested, warmup, origin=completed) if requested is not None else None
        if saved.get('version') != cls.version:
            raise ValueError('monitor version/warmup changed')
        requested = validate_policy(requested) if requested is not None else None
        migrating = requested is not None and requested != saved['policy']
        if migrating and (cls.version not in (VERSION, 's2ut-dev-unit-monitor-v1') or requested.get('version') != 2 or
                          requested['interval'] != saved['policy']['interval'] or saved['stopped']):
            raise ValueError('monitor policy changed; retain the saved policy for this run')
        changes = deepcopy(saved.get('warmup_history', []))
        initial = changes[0]['from'] if changes else saved['warmup']
        policies = deepcopy(saved.get('policy_history', []))
        if not isinstance(policies, list):
            raise ValueError('invalid monitor policy history')
        original_policy = policies[0]['from'] if policies else saved['policy']
        result = cls(original_policy, initial, origin=saved['origin'])
        previous_at = result.origin - 1
        for change in changes:
            if (set(change) != {'from', 'to', 'adopted_at'} or
                    any(type(change[k]) is not int for k in change) or
                    not 0 < change['to'] < change['from'] or
                    not previous_at < change['adopted_at'] <= completed or
                    change['adopted_at'] >= change['from']):
                raise ValueError('invalid monitor warmup history')
            previous_at = change['adopted_at']
        previous_at, previous_policy = result.origin - 1, result.policy
        for policy_change in policies:
            if (cls.version not in (VERSION, 's2ut-dev-unit-monitor-v1') or set(policy_change) != {'from', 'to', 'adopted_at'} or
                    type(policy_change['adopted_at']) is not int or
                    not previous_at < policy_change['adopted_at'] <= completed or
                    validate_policy(policy_change['from']) != previous_policy or
                    validate_policy(policy_change['to']).get('version') != 2 or
                    policy_change['to']['interval'] != previous_policy['interval'] or
                    policy_change['from'] == policy_change['to']):
                raise ValueError('invalid monitor policy history')
            previous_at, previous_policy = policy_change['adopted_at'], policy_change['to']
        pending = iter(changes)
        change = next(pending, None)
        policy_pending = iter(policies)
        policy_change = next(policy_pending, None)
        def apply_changes(update):
            nonlocal change, policy_change
            while change is not None and change['adopted_at'] == update:
                if result.warmup != change['from'] or result.stopped or result.bad_evaluations:
                    raise ValueError('invalid monitor warmup history')
                result.warmup = change['to']
                result._warmup_history = getattr(result, '_warmup_history', []) + [change]
                change = next(pending, None)
            while policy_change is not None and policy_change['adopted_at'] == update:
                if result.policy != policy_change['from'] or result.stopped:
                    raise ValueError('invalid monitor policy history')
                result.policy = deepcopy(policy_change['to'])
                result.anchor, result.bad_evaluations = result.best_value, 0
                if hasattr(result, 'reset_total_stopping'):
                    result.reset_total_stopping()
                result._policy_history = getattr(result, '_policy_history', []) + [policy_change]
                policy_change = next(policy_pending, None)
        for row in saved['history']:
            apply_changes(result.update)
            result.observe(row['update'], row['train'], row['dev'], row['learning_rate'])
        apply_changes(result.update)
        result.best_sha256 = saved['best_sha256']
        if result.update != completed or result.state() != saved:
            raise ValueError('monitor history/counters do not match checkpoint')
        if result.best_update is not None:
            from ..drive_staging import raw_sha
            best = Path(work)/'checkpoints/checkpoint_best.pt'
            if not result.best_sha256 or not best.is_file() or raw_sha(best) != result.best_sha256:
                raise ValueError('best dev checkpoint missing or changed; restore its verified snapshot')
        if warmup != result.warmup:
            # Only TT2 explicitly shortened schedules may migrate the guard.
            # Validate all old history first; never reinterpret losses or reset
            # patience/best/optimizer state under the new schedule.
            if (cls.version != VERSION or type(warmup) is not int or
                    not 0 < warmup < result.warmup or completed >= result.warmup or result.stopped or result.bad_evaluations):
                raise ValueError('monitor version/warmup changed')
            result._warmup_history = changes + [
                {'from': result.warmup, 'to': warmup, 'adopted_at': completed}]
            result.warmup = warmup
        if migrating:
            # Explicit v2 request changes future decisions only. Keep and first
            # verify all old losses, best bytes, counters and policy timelines.
            result._policy_history = policies + [
                {'from': deepcopy(result.policy), 'to': deepcopy(requested), 'adopted_at': completed}]
            result.policy = deepcopy(requested)
            result.anchor, result.bad_evaluations = result.best_value, 0
            if hasattr(result, 'reset_total_stopping'):
                result.reset_total_stopping()
            print(f'[training-monitor-policy] adopted_at={completed} policy={json.dumps(requested, sort_keys=True)} '
                  'history_best_optimizer_preserved=True stopping_counter_restarted=True', file=sys.stderr, flush=True)
        return result

    def save_best(self, work, save):
        """Save a separate full optimizer checkpoint, then bind its bytes to state."""
        from ..drive_staging import raw_sha
        path = Path(work)/'checkpoints/checkpoint_best.pt'
        save(path)
        self.best_sha256 = raw_sha(path)

    def write_reports(self, work, *, render=True):
        work = Path(work)
        atomic_write_json(work/'training-monitor.json', self.state(), overwrite=True)
        if not render:
            return
        table = io.StringIO(newline='')
        fields = ['update', 'learning_rate'] + [f'{split}_{key}' for split in ('train', 'dev') for key in self.metrics]
        writer = csv.DictWriter(table, fieldnames=fields)
        writer.writeheader()
        for row in self.history:
            writer.writerow(dict(update=row['update'], learning_rate=row['learning_rate'],
                **{f'{split}_{key}': row[split][key] if row[split] is not None else ''
                   for split in ('train', 'dev') for key in self.metrics}))
        atomic_write_text(work/'training-monitor.csv', table.getvalue(), overwrite=True)
        atomic_write_text(work/'training-monitor.svg', self.svg(), overwrite=True)

    def svg(self):
        height = 100 + math.ceil(len(self.metrics)/2)*290
        parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="880" height="{height}" viewBox="0 0 880 {height}">',
                 f'<rect width="880" height="{height}" fill="white"/>',
                 f'<text x="25" y="25" font-size="18">{self.title}</text>',
                 '<text x="25" y="48" fill="#2563eb">train</text><text x="95" y="48" fill="#dc2626">dev</text>']
        maximum_update = max(1, self.update)
        for index, key in enumerate(self.metrics):
            left, top = 65 + (index % 2)*435, 100 + (index // 2)*290
            points = {split: [(r['update'], r[split][key]) for r in self.history if r[split] is not None]
                      for split in ('train', 'dev')}
            high = max([v for rows in points.values() for _, v in rows] or [1]) or 1
            parts.extend([f'<text x="{left}" y="{top-20}">{key} (max {high:.4g})</text>',
                          f'<path d="M {left} {top} V {top+210} H {left+340}" fill="none" stroke="#64748b"/>',
                          f'<text x="{left}" y="{top+235}">0</text>',
                          f'<text x="{left+260}" y="{top+235}">update {maximum_update}</text>'])
            for split, color in (('train', '#2563eb'), ('dev', '#dc2626')):
                rows = points[split]
                sampled = rows[::max(1, len(rows)//1500)]
                if rows and (not sampled or sampled[-1] != rows[-1]):
                    sampled.append(rows[-1])
                coords = ' '.join(f'{left+u/maximum_update*340:.2f},{top+210-v/high*210:.2f}' for u, v in sampled)
                parts.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="1.5"/>')
                if sampled:
                    u, v = sampled[-1]
                    parts.append(f'<circle cx="{left+u/maximum_update*340:.2f}" cy="{top+210-v/high*210:.2f}" r="2" fill="{color}"/>')
        parts.append('</svg>')
        return '\n'.join(parts)


def early_stop_result(work, kind, completed):
    """Accept termination only with a published checkpoint-aligned monitor record."""
    path = Path(work)/'training-monitor.json'
    if kind not in ('tt2', 's2ut') or not path.is_file():
        return None
    saved = json.loads(path.read_text(encoding='utf-8'))
    if not saved.get('stopped'):
        return None
    monitor_class, policy = Monitor, policy_from_env()
    if kind == 's2ut':
        from ..s2ut.monitor import S2UTMonitor, policy_from_env as s2ut_policy
        monitor_class, policy = S2UTMonitor, s2ut_policy()
    monitor = monitor_class.restore(policy, saved, completed, saved['warmup'], work)
    result = dict(status='EARLY_STOPPED', durable_updates=completed,
                  best_update=monitor.best_update, best_dev_loss=monitor.best_value,
                  selection_metric=monitor.metric)
    if kind == 's2ut' and monitor.policy.get('version') == 2:
        result.update(stopping_metrics=['unit_loss', 'loss'], stopping_rule='all_plateau',
                      best_total_dev_loss=monitor.best_total_value, best_total_update=monitor.best_total_update)
    return result


def select_checkpoint(work, selection='auto'):
    """Inference prefers verified dev-best; training resumes checkpoint_last only."""
    if selection not in ('auto', 'last', 'best'):
        raise ValueError('invalid TT2 checkpoint selection')
    work = Path(work)
    report = work/'training-monitor.json'
    if selection != 'last' and report.is_file():
        saved = json.loads(report.read_text(encoding='utf-8'))
        if saved.get('best_update') is not None:
            completed = saved['history'][-1]['update'] if saved['history'] else saved['origin']
            monitor_class = Monitor
            if saved.get('version') == 's2ut-dev-unit-monitor-v1':
                from ..s2ut.monitor import S2UTMonitor
                monitor_class = S2UTMonitor
            monitor_class.restore(None, saved, completed, saved['warmup'], work)
            return work/'checkpoints/checkpoint_best.pt'
    if selection == 'best':
        raise FileNotFoundError('no verified dev-best checkpoint yet')
    return work/'checkpoints/checkpoint_last.pt'


@operation('training monitor: publish durable dashboard')
def mirror_reports(frozen, backup):
    """Convenience preview only; resumability relies on the verified snapshot."""
    frozen, backup = Path(frozen), Path(backup)
    if not (frozen/'training-monitor.json').is_file():
        return
    try:
        for name in ('training-monitor.csv', 'training-monitor.svg', 'training-monitor.json'):
            source = frozen/name
            if source.is_file():
                value = source.read_bytes()
                target = backup/name
                atomic_write_bytes(target, value, overwrite=True)
                if target.read_bytes() != value:
                    raise OSError('dashboard readback mismatch: '+name)
    except OSError as error:
        # A completed, verified checkpoint must not be invalidated merely by a
        # failed convenience preview. The immutable snapshot retains reports.
        print(f'[training-monitor] dashboard_copy_failed={error}; verified snapshot preserved',
              file=sys.stderr, flush=True)
