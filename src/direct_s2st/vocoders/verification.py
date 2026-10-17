"""Resumable train-only input verification, outside the training time budget.

Reuse assumes immutable inputs and matching size/mtime. S2ST_PREP_RECHECK=1
forces hashing again. Session receipts are private, not training checkpoints.
"""
import json
import os
from pathlib import Path
import subprocess

from ..hashing import sha256_file
from ..io import atomic_write_json
from ..journal import digest
from ..manifests.reader import read_common_manifest
from ..preparation import Checkpoints, checkpoint_map, file_stamp
from ..progress import operation
from ..s2ut.extract_units import load_unit_file
from ..s2ut.unit_storage import records, sequence, stamp


@operation('vocoder: resumable input verification')
def verify_inputs(common_root, kind, units_root=None, *, split='train'):
    if split not in ('train', 'dev'):
        raise ValueError('vocoder verification only supports train or dev')
    common_root = Path(common_root)
    rows = list(read_common_manifest(common_root / f'{split}.jsonl', parallel=True))
    if not rows or len({row['pair_id'] for row in rows}) != len(rows) or any(row['split'] != split for row in rows):
        raise ValueError(f'fitting/validation requires nonempty unique {split}-only rows')
    if kind not in ('mel', 'unit') or (kind == 'unit' and units_root is None):
        raise ValueError('unit fitting requires units-root')
    paths = {}
    unit_records = records(units_root) if kind == 'unit' else {}
    for row in rows:
        pair_id = row['pair_id']
        if not isinstance(pair_id, str) or not pair_id or Path(pair_id).name != pair_id or any(c in pair_id for c in '/\\:'):
            raise ValueError('fitting requires safe pair IDs')
        expected = row.get('en_sha256')
        if not isinstance(expected, str) or len(expected) != 64 or any(c not in '0123456789abcdefABCDEF' for c in expected):
            raise ValueError(f'{pair_id}: expected en_sha256 in the common manifest')
        if kind == 'unit':
            paths[pair_id] = Path(units_root) / split / 'original' / (pair_id + '.units')
            if pair_id in unit_records and unit_records[pair_id]['split'] != split:
                raise ValueError('unit split mismatch during fitting')

    def key(row):
        stamps = dict(audio=file_stamp(row['en_audio']))
        if kind == 'unit':
            record = unit_records.get(row['pair_id'])
            stamps['units'] = stamp(record, 'original') if record else file_stamp(paths[row['pair_id']])
        return digest([row['pair_id'], row['en_sha256'].lower(), stamps])

    def check(row):
        before = key(row)
        audio = sha256_file(Path(row['en_audio']))
        if audio.lower() != row['en_sha256'].lower():
            raise ValueError(f"{row['pair_id']}: target audio checksum mismatch")
        value = dict(audio=audio, audio_stamp=file_stamp(row['en_audio']))
        if kind == 'unit':
            path = paths[row['pair_id']]
            record = unit_records.get(row['pair_id'])
            if record and record.get('units_storage') == 'inline-v1':
                units = sequence(record, 'original', 100)
                from ..s2ut.extract_units import serialize_units
                from ..hashing import sha256_text
                value.update(units=sha256_text(serialize_units(units)), sequence=units,
                             unit_stamp=stamp(record, 'original'))
            else:
                value.update(units=sha256_file(path), sequence=load_unit_file(path, clusters=100),
                             unit_stamp=file_stamp(path))
        if key(row) != before:
            raise ValueError(f"{row['pair_id']}: training input changed during verification")
        return value

    with Checkpoints(common_root.parent / '.prep-checkpoints/vocoder-inputs',
                     dict(stage='vocoder-inputs-v1', kind=kind, **({'split': split} if split != 'train' else {})),
                     resume=os.environ.get('S2ST_PREP_RECHECK') != '1') as cache:
        values = list(checkpoint_map(check, rows, cache, key, f'vocoder: verify/reuse {split} inputs', total=len(rows)))
    for row, value in zip(rows, values):
        row['_verified_audio_stamp'] = value['audio_stamp']
        if kind == 'unit':
            row['_verified_unit_stamp'] = value['unit_stamp']
    return dict(format='direct-s2st-vocoder-inputs-v1', kind=kind,
                common=sha256_file(common_root / f'{split}.jsonl'), rows=rows,
                audio={row['pair_id']: value['audio'] for row, value in zip(rows, values)},
                units={row['pair_id']: value['units'] for row, value in zip(rows, values)} if kind == 'unit' else {},
                sequences={row['pair_id']: value['sequence'] for row, value in zip(rows, values)} if kind == 'unit' else {},
                unit_lock_sha256=sha256_file(Path(units_root) / 'unit-lock.json') if kind == 'unit' else None)


def save_receipt(path, value):
    corpus = os.environ.get('CORPUS_ROOT')
    if corpus and Path(path).resolve().is_relative_to(Path(corpus).resolve()):
        raise ValueError('verification receipts must be outside CORPUS_ROOT')
    atomic_write_json(Path(path), dict(value=value, sha256=digest(value)))


def load_receipt(path, common_root, kind, units_root=None, *, split='train'):
    if split not in ('train', 'dev'):
        raise ValueError('vocoder receipt split must be train or dev')
    document = json.loads(Path(path).read_text(encoding='utf-8'))
    value = document['value']
    expected_lock = sha256_file(Path(units_root) / 'unit-lock.json') if kind == 'unit' else None
    if (document['sha256'] != digest(value) or value['format'] != 'direct-s2st-vocoder-inputs-v1'
            or value['kind'] != kind or value['common'] != sha256_file(Path(common_root) / f'{split}.jsonl')
            or value['unit_lock_sha256'] != expected_lock
            or not value['rows'] or any(row['split'] != split for row in value['rows'])):
        raise ValueError('vocoder input verification receipt mismatch')
    return value


@operation('training: vocoder input preflight (before session timer)')
def preflight_command(command, directory, *, runner=None):
    if '-m' not in command or command[command.index('-m') + 1] != 'direct_s2st.vocoders.train':
        return command
    receipt = Path(directory) / 'vocoder-inputs.json'
    (runner or subprocess.run)(command + ['--verify-only', '--verification-result', str(receipt)], check=True)
    if not receipt.is_file():
        raise RuntimeError('vocoder preflight did not publish its verified input receipt')
    dev_receipt = receipt.with_suffix('.dev.json')
    return command + ['--verified-inputs', str(receipt)] + (
        ['--verified-dev-inputs', str(dev_receipt)] if dev_receipt.is_file() else [])
