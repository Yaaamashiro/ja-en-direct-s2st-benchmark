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
from zipfile import BadZipFile
from . import baseline_wandb_v3 as v3

from .progress import operation
from .baseline_wandb_evidence import (PublishedFiles, baseline_wandb_run_id, publication_key,
                                     registered_fingerprint, snapshot_fingerprint, verify_bleu)

FILES = ('training-monitor.json', 'training-monitor.csv',
         'training-bleu.json', 'training-bleu.csv')
PROJECT = 'alignment-transfer-s2ut'
API_BASE = 'https://api.forge.coreweave.com'
VERSION = 2
LOSSES = ('loss', 'unit_loss', 'source_letter', 'target_letter', 'decoder_target_ctc')
TRAIN_SCHEMA_V2 = {'unit_loss': 'train/unit_ce_nats', 'loss': 'alignment/update/task_loss'}
LEARNING_RATE_SCHEMA_V2 = 'train/update/learning_rate'
DEV_SCHEMA_V2 = {'loss': 'dev/full/loss', **{
    key: f'dev/full/multitask_{key}_loss' for key in LOSSES[2:]
}}
SCHEMA_V2_METRICS = frozenset((*TRAIN_SCHEMA_V2.values(), LEARNING_RATE_SCHEMA_V2,
                              *DEV_SCHEMA_V2.values()))


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
    if split == 'dev' and key == 'unit_loss':
        return 'dev/main_loss'
    if key in LOSSES:
        return f'baseline/{split}/{key}_bits'
    return f'baseline/{split}/observed/{key}'


def record_loss(target, split, key, value):
    value = number(value)
    if value is None:
        return
    merge(target, metric_name(split, key), value)
    if split == 'train' and key in TRAIN_SCHEMA_V2:
        nats = value * math.log(2)
        merge(target, 'train/task_loss' if key == 'loss' else 'train/main_loss', nats)
        merge(target, TRAIN_SCHEMA_V2[key], nats)
    elif split == 'dev' and key == 'loss':
        merge(target, 'dev/loss', value)
        merge(target, DEV_SCHEMA_V2[key], value)
    elif key in LOSSES[2:]:
        name = f'train/aux/{key}_loss_bits' if split == 'train' else f'dev/multitask_{key}_loss'
        merge(target, name, value)
        if split == 'dev':
            merge(target, DEV_SCHEMA_V2[key], value)


def record_learning_rate(target, value):
    rate = number(value)
    merge(target, 'train/learning_rate', rate)
    merge(target, LEARNING_RATE_SCHEMA_V2, rate)


def _read_reports(source):
    return {name: (source / name).read_bytes() for name in FILES if (source / name).is_file()}


def _read_receipt(path, expected=None):
    receipt = json.loads(path.read_text(encoding='utf-8'))
    update = update_number(receipt['updates'])
    if not update or (expected is not None and update != expected):
        raise ValueError(f'invalid published snapshot update: {path}')
    return update, str(path), receipt


def _latest_receipts(source, snapshot):
    if snapshot:
        return [_read_receipt(Path(snapshot))]
    standalone = source/'snapshot.json'
    root_receipt = _read_receipt(standalone) if standalone.is_file() else None
    candidates = {}
    # Colab Drive mounts make individual reads expensive. Publication directories
    # encode the optimizer update, so open receipts only at the newest update.
    if source.is_dir():
        for directory in source.iterdir():
            match = re.fullmatch(r'([0-9]+)-.+', directory.name)
            if match:
                candidates.setdefault(int(match[1]), []).append(directory/'snapshot.json')
    for update in sorted(candidates, reverse=True):
        if root_receipt and update < root_receipt[0]:
            break
        receipts = []
        for path in candidates[update]:
            try:
                receipts.append(_read_receipt(path, expected=update))
            except (FileNotFoundError, NotADirectoryError):
                continue  # The publication marker is written last.
        if root_receipt and update == root_receipt[0]:
            receipts.append(root_receipt)
        if receipts:
            return receipts
    return [root_receipt] if root_receipt else []


def _receipt(source, blobs, snapshot):
    receipts = _latest_receipts(source, snapshot)
    if not receipts:
        raise ValueError('no published snapshot receipt; refusing unconfirmed metrics')
    maximum = max(item[0] for item in receipts)
    matching = []
    for update, path, receipt in receipts:
        if update != maximum:
            continue
        if not blobs or any(name in receipt['files'] and name not in blobs for name in FILES):
            continue
        for name, payload in blobs.items():
            hashes = [v for k, v in receipt['files'].items() if Path(k).name == name]
            if len(hashes) != 1 or hashes[0] != hashlib.sha256(payload).hexdigest():
                break
        else:
            matching.append((path, receipt, blobs))
    if not matching:
        # Read only the report members when the convenience mirror lags.
        for update, path, receipt in receipts:
            if update != maximum:
                continue
            try:
                with PublishedFiles(path, receipt) as files:
                    archived = {name: files.read(name) for name in FILES if name in receipt['files']}
            except (OSError, KeyError, ValueError, BadZipFile):
                continue
            if archived:
                matching.append((path, receipt, archived))
        if not matching:
            raise ValueError('reports do not match latest published snapshot')
    if len({digest(receipt['identity']) for _, receipt, _ in matching}) != 1:
        raise ValueError('matching reports belong to ambiguous snapshot identities')
    path, receipt, blobs = sorted(matching, key=lambda item: item[0])[-1]
    identity = receipt['identity']
    if identity['config']['kind'] != 's2ut':
        raise ValueError('snapshot is not an S2UT baseline')
    return receipt, path, blobs


@operation('baseline W&B: read confirmed measured reports')
def collect(source, *, snapshot=None, registration=None, audit_journals=False, metric_schema=2,
            baseline_run_key=''):
    source = Path(source)
    blobs = _read_reports(source)
    receipt, receipt_path, blobs = _receipt(source, blobs, snapshot)
    fingerprint = snapshot_fingerprint(receipt)
    if metric_schema == 3 and not registration:
        raise ValueError('Schema v3 requires the published Baseline registration receipt')
    if registration and (fingerprint is None or metric_schema == 3):
        registered = registered_fingerprint(registration, receipt)
        if fingerprint is not None and fingerprint != registered:
            raise ValueError('registered Baseline and snapshot fingerprints differ')
        fingerprint = registered
    with PublishedFiles(receipt_path, receipt) as files:
        result = _measurements(blobs, receipt, receipt_path, files if audit_journals else None,
                               schema_v3=metric_schema == 3)
    result['training_fingerprint'] = fingerprint
    if metric_schema == 3:
        # These member hashes are bound by the published receipt and registration.
        result['lineage'] = {name: receipt['files'][name] for name in (
            'checkpoints/checkpoint_last.pt', 'checkpoints/checkpoint_best.pt',
            'research-metadata.json', 's2ut-recipe-lock.json')}
        key = publication_key(baseline_run_key or source.as_posix())
        result['schema_identity'] = v3.identity(result, key)
        result['rows'] = v3.measurements(result)
        v3.validate_completeness(result['rows'])
        result['summary'] = dict(v3.SUMMARY)
        result['counts'], result['coverage'] = metric_coverage(
            (int(update), values) for update, values in result['rows'].items())
    elif metric_schema != 2:
        raise ValueError('metric_schema must be 2 or 3')
    return result


def _measurements(blobs, receipt, receipt_path, journals, *, schema_v3=False):
    durable = update_number(receipt['updates'])
    rows, summary, conditions = {}, {}, {}

    def at_update(metric, update, read, *args):
        try:
            return read(*args)
        except (ValueError, TypeError, OverflowError) as error:
            if not schema_v3:
                raise
            raise ValueError(f'Schema v3 {metric} at update {update!r}: {error}') from error

    def row(update):
        # Retain suspect coordinates so v3 can reject them explicitly.
        return rows.setdefault(update, {}) if schema_v3 or update <= durable else {}

    def monitor_metric(split, key):
        return v3.SOURCE_METRIC_NAMES.get(metric_name(split, key), f'{split}/{key}')

    def monitor_row(item, losses):
        coordinate = item.get('update')
        rate = item.get('learning_rate')
        try:
            update = update_number(coordinate)
        except (ValueError, TypeError, OverflowError) as error:
            if not schema_v3:
                raise
            metrics = [monitor_metric(split, key) for split, key, value in losses if value not in (None, '')]
            if rate not in (None, ''):
                metrics.append('train/learning_rate')
            label = ', '.join(metrics) or 'train/dev monitor metrics'
            raise ValueError(f'Schema v3 {label} at update {coordinate!r}: {error}') from error
        target = row(update)
        for split, key, value in losses:
            name = monitor_metric(split, key) if schema_v3 else key
            at_update(name, update, record_loss, target, split, key, value)
        at_update('train/learning_rate', update, record_learning_rate, target, rate)

    def bleu_number(item, update, mode, key):
        return at_update(f'eval/{mode}_{key}', update, number, item.get(f'{mode}_{key}'))

    monitor = json.loads(blobs['training-monitor.json']) if 'training-monitor.json' in blobs else None
    if monitor:
        if monitor.get('metric') != 'teacher_forced_dev_unit_loss':
            raise ValueError('unrecognized S2UT loss definition')
        for item in monitor.get('history', []):
            monitor_row(item, ((split, key, value) for split in ('train', 'dev')
                               for key, value in (item.get(split) or {}).items()))
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
            monitor_row(item, ((*key.split('_', 1), value) for key, value in item.items()
                               if key.startswith(('train_', 'dev_'))))

    measured = {}
    if 'training-bleu.json' in blobs:
        bleu = json.loads(blobs['training-bleu.json'])
        conditions['bleu_purpose'] = bleu.get('purpose')
        for item in bleu.get('results', []):
            update = at_update('eval/asr_bleu, eval/aux_bleu', item.get('update'), update_number, item.get('update'))
            target = row(update)
            for mode in ('aux', 'asr'):
                if schema_v3 and mode == 'asr' and (
                    item.get('asr_bleu') is not None
                    or any(item.get(f'asr_{key}') not in (None, 0) for key in ('samples', 'failures'))
                ):
                    if update not in v3.BLEU_UPDATES:
                        raise ValueError(f'Schema v3 eval/asr_bleu at unexpected update {update}')
                    for key in v3.BLEU_FIELDS:
                        if bleu_number(item, update, mode, key) is None:
                            raise ValueError(f'Schema v3 eval/asr_{key} missing at update {update}')
                # Unscheduled ASR rows contain synthetic samples=0/failures=0.
                # An actual score (including 0.0) is evidence of a measurement.
                if item.get(mode + '_bleu') is None:
                    continue
                journal = (journals if update <= durable and (not schema_v3 or update in v3.BLEU_UPDATES)
                           else None)
                eligibility = verify_bleu(item, mode, journal)
                suffix = '' if eligibility == 'comparable' else '_'+eligibility
                previous = measured.setdefault(update, {}).setdefault(mode, suffix)
                if previous != suffix:
                    raise ValueError('duplicate BLEU records disagree about eligibility')
                for key in ('bleu', 'samples', 'failures'):
                    merge(target, f'eval/{mode}_{key}{suffix}', bleu_number(item, update, mode, key))
                scoring = item.get(mode + '_scoring') or {}
                for key, value in scoring.items():
                    if key in ('bleu', 'samples', 'failures'):
                        checked = at_update(f'eval/{mode}_{key} scoring', update, number, value)
                        merge(target, f'eval/{mode}_{key}{suffix}', checked)
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
                if 'ids' in identity:
                    merge(target, 'baseline/eval/dev_subset_sha256', digest(identity['ids']))
                if 'references' in identity:
                    merge(target, 'baseline/eval/dev_references_sha256', digest(identity['references']))
                if identity.get('weights') is not None:
                    merge(target, 'baseline/eval/weights_sha256', identity['weights'])

    if 'training-bleu.csv' in blobs:
        for item in csv.DictReader(io.StringIO(blobs['training-bleu.csv'].decode('utf-8-sig'))):
            update = at_update('eval/asr_bleu, eval/aux_bleu', item.get('update'), update_number, item.get('update'))
            target = row(update)
            for mode in ('aux', 'asr'):
                if bleu_number(item, update, mode, 'bleu') is None:
                    continue
                if schema_v3 and mode == 'asr' and update not in v3.BLEU_UPDATES:
                    raise ValueError(f'Schema v3 eval/asr_bleu at unexpected update {update}')
                if 'training-bleu.json' in blobs and mode not in measured.get(update, set()):
                    if schema_v3:
                        raise ValueError(f'Schema v3 eval/{mode}_bleu at update {update}: CSV/JSON disagree')
                    raise ValueError('CSV/JSON disagree about measured BLEU')
                suffix = measured.get(update, {}).get(mode)
                if suffix is None:
                    # CSV alone cannot establish the scoring protocol or classify failures.
                    suffix = '_legacy_unclassified' if number(item.get(mode+'_failures')) != 0 else '_unmatched'
                for key in ('bleu', 'samples', 'failures'):
                    merge(target, f'eval/{mode}_{key}{suffix}', bleu_number(item, update, mode, key))
            if item.get('status'):
                merge(target, 'baseline/eval/status', item['status'])

    maximum = ((monitor or {}).get('policy', {}).get('max_update')
               or (monitor or {}).get('absolute_max_update'))
    if maximum is not None:
        summary['train/target_update'] = update_number(maximum)
    summary['train/max_update'] = durable
    summary['train/durable_update'] = durable
    counts, coverage = metric_coverage(rows.items())
    return dict(version=VERSION, identity=receipt['identity'], durable_update=durable,
                rows={str(k): v for k, v in sorted(rows.items()) if v}, summary=summary,
                conditions=conditions, counts=counts, coverage=coverage,
                sources={name: dict(sha256=hashlib.sha256(payload).hexdigest(), bytes=len(payload))
                         for name, payload in blobs.items()}, receipt=str(receipt_path),
                best_checkpoint_sha256=(monitor or {}).get('best_sha256'))


def metric_coverage(rows):
    coverage = {}
    for update, values in sorted(rows):
        for key in values:
            span = coverage.setdefault(key, dict(count=0, first_update=update, last_update=update))
            span['count'] += 1
            span['last_update'] = update
    return {key: span['count'] for key, span in sorted(coverage.items())}, coverage


def binding(bundle, entity, project=PROJECT, *, key, adopted_run_id=''):
    if (not project or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', project)
            or not entity or not re.fullmatch(r'[A-Za-z0-9_-]+', entity)):
        raise ValueError('explicit W&B entity and a valid existing project name are required')
    fingerprint = bundle.get('training_fingerprint')
    if not fingerprint:
        raise ValueError('verified training fingerprint required; select a published Baseline registration or verified native snapshot')
    key = publication_key(key)
    derived = baseline_wandb_run_id(key, fingerprint)
    if adopted_run_id and not re.fullmatch(r'[A-Za-z0-9_-]+', adopted_run_id):
        raise ValueError('invalid existing W&B Run ID')
    contract = dict(version=VERSION, api_base=API_BASE, entity=entity, project=project, condition='baseline',
                    baseline_run_key=key, training_fingerprint=fingerprint,
                    run_id=adopted_run_id or derived, adopted=bool(adopted_run_id))
    if bundle.get('schema_identity'):
        if project != v3.PROJECT or entity != v3.ENTITY or adopted_run_id:
            raise ValueError('invalid Schema v3 destination or Run ID override')
        contract.update(v3.identity(bundle, key))
    elif project == v3.PROJECT:
        raise ValueError('the Schema v3 project rejects legacy Baseline writes')
    return contract


def run_id(contract):
    return contract['run_id']


def plan(bundle, remote_rows, *, adopted=False, eligible=None, check_duplicates=True):
    """Plan only missing metric cells. An existing cell may never be replaced."""
    observed, events, last = {}, set(), -1
    for item in remote_rows:
        last = max(last, int(item['_step']))
        if 'baseline/event_id' not in item:
            if not adopted:
                raise ValueError('foreign/unrecognized history in baseline run')
            if item.get('update') is None:
                if set(item) & {key for row in bundle['rows'].values() for key in row}:
                    raise ValueError('existing measured history has no optimizer update')
                continue
            update = update_number(item['update'])
            if update > bundle['durable_update']:
                raise ValueError('source is older than already published W&B metrics')
            for key in item.keys() & bundle['rows'].get(str(update), {}).keys():
                merge(observed.setdefault(str(update), {}), key, item[key])
            continue
        update = update_number(item['update'])
        if update > bundle['durable_update']:
            raise ValueError('source is older than already published W&B metrics')
        fields = json.loads(item['baseline/fields'])
        event = dict(update=update, values={key: item[key] for key in fields},
                     provenance=json.loads(item['baseline/provenance']))
        if digest(event) != item['baseline/event_id']:
            raise ValueError('remote baseline event content differs')
        recorded = observed.setdefault(str(update), {})
        if bundle.get('schema_identity'):
            if event['values'].keys() - bundle['rows'].get(str(update), {}).keys():
                raise ValueError('remote metric has no matching Schema v3 source measurement')
            if check_duplicates and event['values'].keys() & recorded.keys():
                raise ValueError(f'duplicate Schema v3 history at update {update}')
        for key, value in event['values'].items():
            merge(recorded, key, value)
        events.add(item['baseline/event_id'])
    pending = []
    for update, values in bundle['rows'].items():
        old = observed.get(update, {})
        for key in old.keys() & values.keys():
            merge(dict(old), key, values[key])
        missing = {k: v for k, v in values.items() if k not in old and (eligible is None or k in eligible)}
        if missing:
            event = dict(update=int(update), values=missing,
                         provenance=dict(sources=bundle['sources'], receipt=bundle['receipt'],
                                         durable_update=bundle['durable_update']))
            pending.append(dict(step=last + len(pending) + 1, event=event, id=digest(event)))
    return pending, events
