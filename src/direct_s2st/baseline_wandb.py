"""Read-only S2UT metric export. No scientific runtime or W&B import here.

Only reports whose bytes match a published snapshot receipt are eligible. W&B
delivery lives in a separate process; this module also supports CPU inspection.
"""
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import tempfile

from .progress import operation

FILES = ('training-monitor.json', 'training-monitor.csv',
         'training-bleu.json', 'training-bleu.csv')
PROJECT = 'alignment-transfer-s2ut'
API_BASE = 'https://api.forge.coreweave.com'
VERSION = 1
LOSSES = ('loss', 'unit_loss', 'source_letter', 'target_letter', 'decoder_target_ctc')


def canonical(value):
    def normalize(item):
        if isinstance(item, float) and math.isfinite(item) and item.is_integer():
            return int(item)
        if isinstance(item, dict):
            return {k: normalize(v) for k, v in item.items()}
        if isinstance(item, (list, tuple)):
            return [normalize(v) for v in item]
        return item
    return json.dumps(normalize(value), sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    name = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                         delete=False) as stream:
            name = stream.name
            stream.write(canonical(value) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if name and Path(name).exists():
            Path(name).unlink()


def number(value):
    """Missing is missing; real zero is a value. Reject corrupt nonfinite data."""
    if value is None or value == '':
        return None
    if isinstance(value, bool):
        raise ValueError('boolean is not a measured number')
    result = float(value)
    if not math.isfinite(result):
        raise ValueError('nonfinite measured metric')
    return result


def update_number(value):
    result = number(value)
    if result is None or result < 0 or not result.is_integer():
        raise ValueError('invalid optimizer update')
    return int(result)


def merge(target, key, value):
    if value is None:
        return
    if key in target:
        old = target[key]
        equal = (math.isclose(old, value, rel_tol=1e-12, abs_tol=1e-12)
                 if isinstance(old, (int, float)) and isinstance(value, (int, float))
                 else old == value)
        if not equal:
            raise ValueError('conflicting duplicate metric: ' + key)
    else:
        target[key] = value


def metric_name(split, key):
    if not re.fullmatch(r'[A-Za-z0-9_.-]+', key):
        raise ValueError('invalid metric name: ' + key)
    # Transfer's native dev main loss is the same label-smoothed criterion sum /
    # sample_size / ln(2). Its TRAIN main loss is in nats: do not alias that.
    if split == 'dev' and key == 'unit_loss':
        return 'dev/main_loss'
    if key in LOSSES:
        return f'baseline/{split}/{key}_bits'
    return f'baseline/{split}/observed/{key}'


def _read_reports(source):
    return {name: (source / name).read_bytes() for name in FILES if (source / name).is_file()}


def _receipt(source, blobs, snapshot):
    candidates = [Path(snapshot)] if snapshot else list(source.glob('*/snapshot.json'))
    if (source / 'snapshot.json').is_file() and not snapshot:
        candidates.append(source / 'snapshot.json')
    receipts = []
    for path in candidates:
        try:
            receipt = json.loads(path.read_text(encoding='utf-8'))
            if update_number(receipt['updates']) > 0:
                receipts.append((update_number(receipt['updates']), str(path), receipt))
        except (KeyError, ValueError, OSError):
            if snapshot:
                raise
    if not receipts:
        raise ValueError('no published snapshot receipt; refusing unconfirmed metrics')
    maximum = max(item[0] for item in receipts)
    matching = []
    for update, path, receipt in receipts:
        if update != maximum:
            continue
        for name, payload in blobs.items():
            hashes = [v for k, v in receipt['files'].items() if Path(k).name == name]
            if len(hashes) != 1 or hashes[0] != hashlib.sha256(payload).hexdigest():
                break
        else:
            matching.append((path, receipt))
    if not matching:
        raise ValueError('reports do not match latest published snapshot')
    if len({digest(receipt['identity']) for _, receipt in matching}) != 1:
        raise ValueError('matching reports belong to ambiguous snapshot identities')
    path, receipt = sorted(matching, key=lambda item: item[0])[-1]
    identity = receipt['identity']
    if identity['config']['kind'] != 's2ut':
        raise ValueError('snapshot is not an S2UT baseline')
    return receipt, path


@operation('baseline W&B: read confirmed measured reports')
def collect(source, *, snapshot=None):
    source = Path(source)
    blobs = _read_reports(source)
    if not blobs:
        raise ValueError('no saved monitor/BLEU reports')
    receipt, receipt_path = _receipt(source, blobs, snapshot)
    durable = update_number(receipt['updates'])
    rows, summary, conditions = {}, {}, {}

    def row(update):
        update = update_number(update)
        return rows.setdefault(update, {}) if update <= durable else {}

    monitor = json.loads(blobs['training-monitor.json']) if 'training-monitor.json' in blobs else None
    if monitor:
        if monitor.get('metric') != 'teacher_forced_dev_unit_loss':
            raise ValueError('unrecognized S2UT loss definition')
        for item in monitor.get('history', []):
            target = row(item['update'])
            for split in ('train', 'dev'):
                for key, value in (item.get(split) or {}).items():
                    merge(target, metric_name(split, key), number(value))
            merge(target, 'train/learning_rate', number(item.get('learning_rate')))
        if update_number(monitor['update']) <= durable:
            for key in ('best_value', 'best_update', 'best_sha256', 'metric',
                        'best_total_value', 'best_total_update', 'stopped', 'early_stopping'):
                if monitor.get(key) is not None:
                    value = monitor[key]
                    if key in ('best_value', 'best_total_value'):
                        value = number(value)
                    if key in ('best_update', 'best_total_update'):
                        value = update_number(value)
                        if value > durable:
                            raise ValueError('best checkpoint metadata exceeds confirmed update')
                    if key == 'best_sha256' and not re.fullmatch('[0-9a-f]{64}', value):
                        raise ValueError('invalid saved best checkpoint hash')
                    summary['baseline/' + key] = value
        conditions['monitor'] = {k: monitor[k] for k in ('version', 'metric', 'policy', 'warmup') if k in monitor}

    if 'training-monitor.csv' in blobs:
        for item in csv.DictReader(io.StringIO(blobs['training-monitor.csv'].decode('utf-8-sig'))):
            target = row(item['update'])
            for key, value in item.items():
                if key.startswith(('train_', 'dev_')):
                    split, name = key.split('_', 1)
                    merge(target, metric_name(split, name), number(value))
            merge(target, 'train/learning_rate', number(item.get('learning_rate')))

    measured = {}
    if 'training-bleu.json' in blobs:
        bleu = json.loads(blobs['training-bleu.json'])
        conditions['bleu_purpose'] = bleu.get('purpose')
        for item in bleu.get('results', []):
            update = update_number(item['update'])
            target = row(update)
            for mode in ('aux', 'asr'):
                # Unscheduled ASR rows contain synthetic samples=0/failures=0.
                # An actual score (including 0.0) is evidence of a measurement.
                if item.get(mode + '_bleu') is None:
                    continue
                measured.setdefault(update, set()).add(mode)
                for key in ('bleu', 'samples', 'failures'):
                    merge(target, f'eval/{mode}_{key}', number(item.get(mode + '_' + key)))
                scoring = item.get(mode + '_scoring') or {}
                for key, value in scoring.items():
                    if key in ('bleu', 'samples', 'failures'):
                        merge(target, f'eval/{mode}_{key}', number(value))
                    if isinstance(value, (float, int)) and not isinstance(value, bool):
                        merge(target, f'baseline/eval/{mode}_scoring/{key}', number(value))
                    elif key in ('signature', 'normalization') and value is not None:
                        merge(target, f'baseline/eval/{mode}_{key}', value)
            if item.get('seconds') is not None:
                merge(target, 'baseline/eval/seconds', number(item['seconds']))
            if item.get('status') is not None:
                merge(target, 'baseline/eval/status', item['status'])
            identity = item.get('identity') or {}
            if identity:
                # Preserve exact conditions, not 500 private reference sentences.
                metadata = {k: v for k, v in identity.items() if k not in ('ids', 'references', 'weights', 'update')}
                conditions.setdefault('bleu_conditions', {})[digest(metadata)] = metadata
                merge(target, 'baseline/eval/conditions_sha256', digest(metadata))
                if identity.get('weights') is not None:
                    merge(target, 'baseline/eval/weights_sha256', identity['weights'])

    if 'training-bleu.csv' in blobs:
        for item in csv.DictReader(io.StringIO(blobs['training-bleu.csv'].decode('utf-8-sig'))):
            update = update_number(item['update'])
            target = row(update)
            for mode in ('aux', 'asr'):
                if number(item.get(mode + '_bleu')) is None:
                    continue
                if 'training-bleu.json' in blobs and mode not in measured.get(update, set()):
                    raise ValueError('CSV/JSON disagree about measured BLEU')
                for key in ('bleu', 'samples', 'failures'):
                    merge(target, f'eval/{mode}_{key}', number(item.get(mode + '_' + key)))
            if item.get('status'):
                merge(target, 'baseline/eval/status', item['status'])

    maximum = ((monitor or {}).get('policy', {}).get('max_update')
               or (monitor or {}).get('absolute_max_update'))
    if maximum is not None:
        summary['train/max_update'] = update_number(maximum)
    summary['train/durable_update'] = durable
    counts = {key: sum(key in values for values in rows.values())
              for key in sorted({k for values in rows.values() for k in values})}
    return dict(version=VERSION, identity=receipt['identity'], durable_update=durable,
                rows={str(k): v for k, v in sorted(rows.items()) if v}, summary=summary,
                conditions=conditions, counts=counts,
                sources={name: dict(sha256=hashlib.sha256(payload).hexdigest(), bytes=len(payload))
                         for name, payload in blobs.items()}, receipt=str(receipt_path),
                best_checkpoint_sha256=(monitor or {}).get('best_sha256'))


def binding(bundle, entity, project=PROJECT):
    if (not project or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', project)
            or not entity or not re.fullmatch(r'[A-Za-z0-9_-]+', entity)):
        raise ValueError('explicit W&B entity and a valid existing project name are required')
    return dict(version=VERSION, api_base=API_BASE, entity=entity, project=project, condition='baseline',
                source_identity_sha256=digest(bundle['identity']))


def run_id(contract):
    return 'baseline-' + digest(contract)[:24]


def plan(bundle, remote_rows):
    """Plan only missing metric cells. An existing cell may never be replaced."""
    observed, events, last = {}, set(), -1
    for item in remote_rows:
        if 'baseline/event_id' not in item:
            raise ValueError('foreign/unrecognized history in baseline run')
        update = update_number(item['update'])
        if update > bundle['durable_update']:
            raise ValueError('source is older than already published W&B metrics')
        fields = json.loads(item['baseline/fields'])
        event = dict(update=update, values={key: item[key] for key in fields},
                     provenance=json.loads(item['baseline/provenance']))
        if digest(event) != item['baseline/event_id']:
            raise ValueError('remote baseline event content differs')
        for key, value in event['values'].items():
            merge(observed.setdefault(str(update), {}), key, value)
        events.add(item['baseline/event_id'])
        last = max(last, int(item['_step']))
    pending = []
    for update, values in bundle['rows'].items():
        old = observed.get(update, {})
        for key in old.keys() & values.keys():
            merge(dict(old), key, values[key])
        missing = {k: v for k, v in values.items() if k not in old}
        if missing:
            event = dict(update=int(update), values=missing,
                         provenance=dict(sources=bundle['sources'], receipt=bundle['receipt'],
                                         durable_update=bundle['durable_update']))
            pending.append(dict(step=last + len(pending) + 1, event=event, id=digest(event)))
    return pending, events
