"""Held-out reconstruction monitoring, not adversarial equilibrium or MOS.

The GAN objectives/optimizers stay untouched. Validation uses a fixed central
segment per dev utterance, equal utterance weighting, and never fits on dev.
"""
import json
import os
from pathlib import Path

from ..journal import digest
from ..translatotron2.monitor import Monitor, validate_policy


class MelMonitor(Monitor):
    version = 'mel-vocoder-dev-monitor-v1'
    metrics = ('mel_loss',)
    score_key = 'mel_loss'
    metric = 'dev_central_segment_mel_l1'
    absolute_max_update = None  # Caller max-updates is the independent hard cap.
    title = 'Mel vocoder: train / fixed-segment dev reconstruction (not MOS)'


class UnitMonitor(MelMonitor):
    version = 'unit-vocoder-dev-monitor-v1'
    metrics = ('mel_loss', 'duration_loss')
    title = 'Unit vocoder: train / dev reconstruction and duration (not MOS)'


def monitor_class(kind):
    if kind not in ('mel', 'unit'):
        raise ValueError('unknown vocoder monitor kind')
    return UnitMonitor if kind == 'unit' else MelMonitor


def settings_from_env():
    raw = os.environ.get('S2ST_VOCODER_MONITOR_POLICY')
    if not raw:
        return None, None
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError('vocoder monitor policy must be an object')
    warmup = value.pop('warmup', 400)
    if type(warmup) is not int or warmup < 0:
        raise ValueError('vocoder monitor guard must be a nonnegative integer')
    policy = validate_policy(value)
    if policy.get('version') != 3:
        raise ValueError('vocoder monitoring requires independent-anchor policy v3')
    return policy, warmup


def validation_identity(verified, segment_frames):
    # Bind fixed ordering, content, and the deterministic segmentation protocol.
    # Physical cache paths/mtime may change between VMs, never audio bytes.
    return digest(dict(version='central-segment-v1', common=verified['common'],
        kind=verified['kind'], ids=[r['pair_id'] for r in verified['rows']],
        audio=verified['audio'], units=verified['units'],
        unit_lock_sha256=verified['unit_lock_sha256'], segment_frames=segment_frames))


def reject_overlap(train, dev):
    if ({r['pair_id'] for r in train['rows']} & {r['pair_id'] for r in dev['rows']}
            or set(train['audio'].values()) & set(dev['audio'].values())):
        raise ValueError('vocoder train/dev IDs or target audio overlap')


def restore_monitor(kind, state, completed, work, validation):
    policy, guard = settings_from_env()
    saved = state.get('monitor') if state else None
    if saved is not None:
        if state.get('monitor_validation') != validation:
            raise ValueError('vocoder validation inputs/segmentation changed on resume')
        guard = saved['warmup'] if guard is None else guard
    return monitor_class(kind).restore(policy, saved, completed, guard or 0, work)


def early_stop_result(work, completed):
    path = Path(work) / 'training-monitor.json'
    if not path.is_file():
        return None
    saved = json.loads(path.read_text(encoding='utf-8'))
    if not saved.get('stopped'):
        return None
    classes = {cls.version: cls for cls in (MelMonitor, UnitMonitor)}
    cls = classes.get(saved.get('version'))
    if cls is None:
        raise ValueError('unknown vocoder monitor state')
    policy, guard = settings_from_env()
    monitor = cls.restore(policy, saved, completed, saved['warmup'] if guard is None else guard, work)
    # Check that the durable resume checkpoint actually contains this state.
    import torch
    checkpoint = torch.load(Path(work)/'generator.pt', map_location='cpu', weights_only=True)
    if checkpoint.get('updates') != completed or checkpoint.get('monitor') != saved:
        raise ValueError('vocoder monitor differs from resume checkpoint')
    return dict(status='EARLY_STOPPED', durable_updates=completed,
        best_update=monitor.best_update, best_dev_loss=monitor.best_value,
        selection_metric=monitor.metric, stopping_metrics=list(monitor.metrics),
        stopping_rule='any_meaningful_resets_patience',
        bad_evaluations=monitor.bad_evaluations, anchors=monitor.anchors)


def select_generator(work):
    """Keep last for resume; a recorded best must exist and pass its checksum."""
    work = Path(work)
    path = work/'training-monitor.json'
    if not path.is_file():
        return work/'generator.pt'
    saved = json.loads(path.read_text(encoding='utf-8'))
    classes = {cls.version: cls for cls in (MelMonitor, UnitMonitor)}
    cls = classes.get(saved.get('version'))
    if cls is None:
        raise ValueError('unknown vocoder monitor state')
    completed = saved['history'][-1]['update'] if saved['history'] else saved['origin']
    restored = cls.restore(None, saved, completed, saved['warmup'], work)
    return work/'checkpoints/checkpoint_best.pt' if restored.best_update is not None else work/'generator.pt'
