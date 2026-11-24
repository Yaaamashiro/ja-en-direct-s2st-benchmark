"""Schema v3 mapping and destination for the registered 400k Baseline."""
from hashlib import sha256
import json
import math

from .baseline_wandb_evidence import baseline_wandb_run_id

ENTITY = 'yaaamashiro-national-institute-of-technology-okinawa-college'
PROJECT = 'direct-s2st-v3'
GROUP = 'experiment-001/transfer-fixed-v2'
STATE_NAME = 'wandb-baseline-v3'
BASELINE_KEY = 'alignment-transfer-s2ut-data/experiment-001/baseline'
FINGERPRINT = 'fe8b63b2f57c5c15f24f03ae9af4ec083ccb1eff6f1c897ac707b83e40b69e24'
FAIRSEQ_REVISION = '3d262bb25690e4eb2e7d3c1309b1e9c406ca4b99'
MAX_UPDATE = 400000
TRAIN_UPDATES = frozenset(range(100, MAX_UPDATE + 1, 100))
DEV_UPDATES = frozenset(range(1000, MAX_UPDATE + 1, 1000))
BLEU_UPDATES = frozenset((1000, 5000, 10000, 25000, 50000, 100000, 200000, 400000))
AUX = ('source_letter', 'target_letter', 'decoder_target_ctc')
TRAIN_LOSSES = {
    'train/main_loss': 'baseline/train/unit_loss_bits',
    'train/task_loss': 'baseline/train/loss_bits',
    'train/total_loss': 'baseline/train/loss_bits',
    **{f'train/aux/{key}_loss': f'baseline/train/{key}_bits' for key in AUX},
}
DEV_LOSSES = {
    'dev/main_loss': 'dev/main_loss',
    'dev/total_loss': 'baseline/dev/loss_bits',
    **{f'dev/aux/{key}_loss': f'baseline/dev/{key}_bits' for key in AUX},
}
BLEU_FIELDS = ('bleu', 'failures', 'samples')
ASR_METRICS = tuple(f'eval/asr_{field}' for field in BLEU_FIELDS)
AUX_BLEU_METRICS = tuple(f'eval/aux_{field}' for field in BLEU_FIELDS)
REQUIRED_UPDATES = {
    **dict.fromkeys((*TRAIN_LOSSES, 'train/learning_rate'), TRAIN_UPDATES),
    **dict.fromkeys(DEV_LOSSES, DEV_UPDATES),
    **dict.fromkeys(ASR_METRICS, BLEU_UPDATES),
}
SOURCE_METRIC_NAMES = {
    source: ', '.join(key for key, legacy in mapping.items() if legacy == source)
    for mapping in (TRAIN_LOSSES, DEV_LOSSES) for source in set(mapping.values())
}
SUMMARY = {f'train/{key}_update': MAX_UPDATE for key in ('target', 'max', 'durable')}


def validate_destination(entity, project, group, state, *, existing_run_id='', schema_v2_only=False):
    if (entity, project, group) != (ENTITY, PROJECT, GROUP):
        raise ValueError('Schema v3 requires the exact registered entity, direct-s2st-v3 project and group')
    if state.name != STATE_NAME or existing_run_id or schema_v2_only:
        raise ValueError('Schema v3 requires its own wandb-baseline-v3 state and derived Run ID')


def validate_source(bundle, key):
    config = bundle['identity']['config']
    metadata = config.get('research_metadata', {})
    if (key != BASELINE_KEY or bundle.get('training_fingerprint') != FINGERPRINT
            or config.get('kind') != 's2ut' or metadata.get('s2ut_recipe') != 'paper_fisher_v1'
            or metadata.get('fairseq_revision') != FAIRSEQ_REVISION
            or bundle['durable_update'] != MAX_UPDATE
            or any(bundle['summary'].get(key) != value for key, value in SUMMARY.items())):
        raise ValueError('Schema v3 requires the verified registered 400k Baseline lineage and summary')


def identity(bundle, key):
    validate_source(bundle, key)
    raw = json.dumps(['baseline', 3, PROJECT, key, FINGERPRINT],
                     separators=(',', ':'), ensure_ascii=False).encode('utf-8')
    return {
        'metric_schema': 3,
        'group': GROUP,
        'run_id': sha256(raw).hexdigest()[:24],
        'origin': {
            'entity': ENTITY, 'project': 'direct-s2st',
            'run_id': baseline_wandb_run_id(key, FINGERPRINT),
            'training_fingerprint': FINGERPRINT, 'baseline_run_key': key,
        },
        'lineage': bundle['lineage'],
    }


def measurements(bundle):
    """Convert only normalized losses and already qualified BLEU measurements."""
    rows = {}
    for coordinate, values in bundle['rows'].items():
        result = {key: values[source] * math.log(2)
                  for mapping in (TRAIN_LOSSES, DEV_LOSSES) for key, source in mapping.items()
                  if source in values}
        result.update({key: values[key] for key in ('train/learning_rate', *ASR_METRICS) if key in values})
        if int(coordinate) in BLEU_UPDATES and all(key in values for key in AUX_BLEU_METRICS):
            result.update({key: values[key] for key in AUX_BLEU_METRICS})
        if result:
            rows[coordinate] = result
    return rows


def _format_updates(updates, stride=1):
    spans = []
    for update in sorted(updates):
        if spans and update == spans[-1][1] + stride:
            spans[-1][1] = update
        else:
            spans.append([update, update])
    return ', '.join(str(start) if start == end else f'{start}–{end} (every {stride})'
                     for start, end in spans)


def validate_completeness(rows):
    """Require the exact coordinates of the registered 400k publication."""
    observed = {metric: set() for metric in REQUIRED_UPDATES}
    for coordinate, values in rows.items():
        for metric in values.keys() & REQUIRED_UPDATES.keys():
            observed[metric].add(int(coordinate))
    errors = []
    for metric, expected in REQUIRED_UPDATES.items():
        missing = expected - observed[metric]
        unexpected = observed[metric] - expected
        if missing:
            stride = min(expected) if metric.startswith(('train/', 'dev/')) else 1
            errors.append(f'{metric} missing updates: {_format_updates(missing, stride)}')
        if unexpected:
            errors.append(f'{metric} unexpected updates: {_format_updates(unexpected)}')
    if errors:
        raise ValueError('Incomplete Schema v3 Baseline measurements: ' + '; '.join(errors))
