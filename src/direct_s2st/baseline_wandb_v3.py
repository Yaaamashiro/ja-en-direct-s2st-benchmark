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
BLEU_METRICS = tuple(f'eval/{mode}_{stat}' for mode in ('aux', 'asr')
                     for stat in ('bleu', 'failures', 'samples'))
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
        update = int(coordinate)
        result = {}
        if update > 0 and update % 100 == 0:
            result.update({key: values[source] * math.log(2) for key, source in TRAIN_LOSSES.items()
                           if source in values})
            if 'train/learning_rate' in values:
                result['train/learning_rate'] = values['train/learning_rate']
        if update > 0 and update % 1000 == 0:
            result.update({key: values[source] * math.log(2) for key, source in DEV_LOSSES.items()
                           if source in values})
        if update in BLEU_UPDATES:
            result.update({key: values[key] for key in BLEU_METRICS if key in values})
        if result:
            rows[coordinate] = result
    return rows
