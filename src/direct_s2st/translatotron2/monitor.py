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
    if saved is not None:
        if (set(saved) != {'version', 'base_warmup', 'warmup', 'adopted_at', 'mode'} or
                saved['version'] != 1 or saved['mode'] != 'paper_practical' or
                saved['base_warmup'] != base_warmup or
                type(saved['adopted_at']) is not int or
                type(saved['warmup']) is not int or
                not 0 < saved['adopted_at'] < saved['warmup'] < base_warmup or
                saved['adopted_at'] > completed):
            raise ValueError('invalid saved warmup transition')
        if requested is not None and requested != saved['warmup']:
            raise ValueError('saved practical warmup cannot change again')
        return saved['warmup'], deepcopy(saved)
    if requested is None:
        return base_warmup, None
    if not checkpoint or not 0 < completed < requested < base_warmup:
        raise ValueError('warmup shortening requires a checkpoint before the new warmup endpoint')
    if checkpoint.get('monitor_state') is not None:
        raise ValueError('adopt shortened warmup before adopting monitoring; existing history must not be reset')
    return requested, dict(version=1, base_warmup=base_warmup, warmup=requested,
                           adopted_at=completed, mode='paper_practical')


def policy_from_env():
    value = os.environ.get('S2ST_TT2_MONITOR_POLICY')
    return json.loads(value) if value else None


def validate_policy(policy):
    if not isinstance(policy, dict) or set(policy) != {'interval', 'patience', 'min_relative_improvement'}:
        raise ValueError('invalid TT2 monitor policy fields')
    for key in ('interval', 'patience'):
        if type(policy[key]) is not int or policy[key] < 1:
            raise ValueError('monitor interval/patience must be positive integers')
    delta = policy['min_relative_improvement']
    if isinstance(delta, bool) or not isinstance(delta, (int, float)) or not math.isfinite(delta) or not 0 <= delta < 1:
        raise ValueError('monitor relative improvement must be finite in [0, 1)')
    return deepcopy(policy)


class Monitor:
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
            if values is not None and any(not math.isfinite(float(values[k])) or float(values[k]) < 0 for k in METRICS):
                raise ValueError('monitor requires finite nonnegative losses')
        if not math.isfinite(lr) or lr <= 0 or (dev is not None) != self.due(update):
            raise ValueError('monitor learning rate/evaluation schedule mismatch')
        self.history.append(dict(update=update, train={k: float(train[k]) for k in METRICS},
                                 dev={k: float(dev[k]) for k in METRICS} if dev is not None else None,
                                 learning_rate=lr))
        improved = False
        if dev is not None:
            value = float(dev['loss'])
            improved = self.best_value is None or value < self.best_value
            if improved:
                self.best_value, self.best_update = value, update
                self.best_sha256 = None  # Caller must publish this exact checkpoint.
            meaningful = self.anchor is None or value < self.anchor * (1-self.policy['min_relative_improvement'])
            if meaningful or update <= self.warmup:
                self.anchor, self.bad_evaluations = value, 0
            else:
                self.bad_evaluations += 1
            self.stopped = update > self.warmup and self.bad_evaluations >= self.policy['patience']
        return improved

    def state(self):
        return dict(version=VERSION, metric='teacher_forced_dev_loss', policy=deepcopy(self.policy),
                    warmup=self.warmup, origin=self.origin, history=deepcopy(self.history),
                    best_value=self.best_value, best_update=self.best_update, best_sha256=self.best_sha256,
                    anchor=self.anchor, bad_evaluations=self.bad_evaluations, stopped=self.stopped)

    @classmethod
    def restore(cls, requested, saved, completed, warmup, work):
        if saved is None:
            return cls(requested, warmup, origin=completed) if requested is not None else None
        if saved.get('version') != VERSION or saved.get('warmup') != warmup:
            raise ValueError('monitor version/warmup changed')
        if requested is not None and validate_policy(requested) != saved['policy']:
            raise ValueError('monitor policy changed; retain the saved policy for this run')
        result = cls(saved['policy'], warmup, origin=saved['origin'])
        for row in saved['history']:
            result.observe(row['update'], row['train'], row['dev'], row['learning_rate'])
        result.best_sha256 = saved['best_sha256']
        if result.update != completed or result.state() != saved:
            raise ValueError('monitor history/counters do not match checkpoint')
        if result.best_update is not None:
            from ..drive_staging import raw_sha
            best = Path(work)/'checkpoints/checkpoint_best.pt'
            if not result.best_sha256 or not best.is_file() or raw_sha(best) != result.best_sha256:
                raise ValueError('best dev checkpoint missing or changed; restore its verified snapshot')
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
        fields = ['update', 'learning_rate'] + [f'{split}_{key}' for split in ('train', 'dev') for key in METRICS]
        writer = csv.DictWriter(table, fieldnames=fields)
        writer.writeheader()
        for row in self.history:
            writer.writerow(dict(update=row['update'], learning_rate=row['learning_rate'],
                **{f'{split}_{key}': row[split][key] if row[split] is not None else ''
                   for split in ('train', 'dev') for key in METRICS}))
        atomic_write_text(work/'training-monitor.csv', table.getvalue(), overwrite=True)
        atomic_write_text(work/'training-monitor.svg', self.svg(), overwrite=True)

    def svg(self):
        parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="880" height="680" viewBox="0 0 880 680">',
                 '<rect width="880" height="680" fill="white"/>',
                 '<text x="25" y="25" font-size="18">TT2: train / teacher-forced dev loss (not BLEU)</text>',
                 '<text x="25" y="48" fill="#2563eb">train</text><text x="95" y="48" fill="#dc2626">dev</text>']
        maximum_update = max(1, self.update)
        for index, key in enumerate(METRICS):
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
    if kind != 'tt2' or not path.is_file():
        return None
    saved = json.loads(path.read_text(encoding='utf-8'))
    if not saved.get('stopped'):
        return None
    monitor = Monitor.restore(policy_from_env(), saved, completed, saved['warmup'], work)
    return dict(status='EARLY_STOPPED', durable_updates=completed,
                best_update=monitor.best_update, best_dev_loss=monitor.best_value,
                selection_metric='teacher_forced_dev_loss')


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
            Monitor.restore(None, saved, completed, saved['warmup'], work)
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
