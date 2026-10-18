"""Explicit, bounded inference selections; never rewrite prepared splits."""
import json
from pathlib import Path

from .io import read_jsonl
from .manifests.reader import read_common_manifest


def sample_ids(path):
    if path is None:
        return None
    ids = json.loads(Path(path).read_text(encoding='utf-8'))
    if (not isinstance(ids, list) or not 1 <= len(ids) <= 10
            or any(not isinstance(v, str) or not v for v in ids) or len(set(ids)) != len(ids)):
        raise ValueError('preview requires 1..10 explicit unique sample IDs')
    return ids


def common_rows(path, ids=None):
    if ids is None:
        return list(read_common_manifest(path)), None
    # Validate the FULL ID set using small metadata, but resolve/check original
    # WAV paths only for selected examples. No other source WAV is opened.
    raw = list(read_jsonl(path))
    by_id = {r['pair_id']: r for r in raw}
    if len(by_id) != len(raw) or not set(ids) <= by_id.keys():
        raise ValueError('preview IDs are missing or the common manifest has duplicates')
    from .drive_staging import read_common_manifest_row
    rows = []
    for pair_id in ids:
        row = by_id[pair_id]
        for language in ('ja', 'en'):
            row = read_common_manifest_row(row, language)
        rows.append(row)
    return rows, set(by_id)
