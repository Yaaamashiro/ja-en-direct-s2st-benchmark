"""Consolidated Unit records; legacy per-sample files remain readable.

Sequences in immutable checkpoint chunks and JSONL manifests avoid hundreds of
thousands of tiny Drive files. Storage changes do not change model identity.
"""
from pathlib import Path

from ..io import read_jsonl
from ..journal import digest
from .reduce_units import reduce_consecutive_units, validate_units


def manifest_paths(root):
    root = Path(root)
    old = {p.name: p for p in root.glob('manifest.shard-*-of-*.jsonl')}
    # A migration is a sibling, never an overwrite of the legacy manifest.
    for path in root.glob('manifest.inline.shard-*-of-*.jsonl'):
        old[path.name.replace('manifest.inline.', 'manifest.')] = path
    return [old[name] for name in sorted(old)]


def records(root):
    result = {}
    for path in manifest_paths(root):
        for row in read_jsonl(path):
            if row['pair_id'] in result:
                raise ValueError(f'duplicate unit record: {row["pair_id"]}')
            result[row['pair_id']] = row
    return result


def sequence(row, kind, clusters):
    if row.get('units_storage') == 'inline-v1':
        original = validate_units(row['units_original'], clusters=clusters)
        reduced = validate_units(row['units_reduced'], clusters=clusters)
        body = dict(original=original, reduced=reduced)
        if (row['units_sha256'] != digest(body)
                or reduced != reduce_consecutive_units(original)
                or len(original) != row['unit_count_original']
                or len(reduced) != row['unit_count_reduced']):
            raise ValueError(f'corrupt inline units: {row["pair_id"]}')
        return original if kind == 'original' else reduced
    from .extract_units import load_unit_file
    return load_unit_file(Path(row[f'units_{kind}_path']), clusters=clusters)


def stamp(row, kind):
    if row.get('units_storage') == 'inline-v1':
        # Authenticate payload here too, even when preparation reuses a cache.
        sequence(row, kind, int(row['kmeans_clusters']))
        return dict(storage='inline-v1', sha256=row['units_sha256'])
    from ..preparation import file_stamp
    return file_stamp(row[f'units_{kind}_path'])
