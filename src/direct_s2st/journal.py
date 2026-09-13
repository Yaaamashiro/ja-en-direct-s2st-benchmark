"""Atomic per-sample progress with immutable invocation identity."""
import hashlib
import json
from pathlib import Path
from .io import ExistingOutputError, atomic_write_json, atomic_write_jsonl, read_jsonl


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


class Journal:
    def __init__(self, path, identity, *, resume=False, overwrite=False):
        self.path = Path(path)
        self.lock = self.path.with_suffix('.lock.json')
        self.rows = {}
        if resume and overwrite:
            raise ValueError('resume and overwrite are mutually exclusive')
        if (self.path.exists() or self.lock.exists()) and not (resume or overwrite):
            raise ExistingOutputError(str(self.path))
        if resume and (self.path.exists() or self.lock.exists()):
            if not self.lock.is_file() or json.loads(self.lock.read_text(encoding='utf-8')) != identity:
                raise ValueError('resume identity mismatch')
            if self.path.exists():
                for row in read_jsonl(self.path):
                    if row['pair_id'] in self.rows:
                        raise ValueError('duplicate progress ID')
                    self.rows[row['pair_id']] = row
        atomic_write_json(self.lock, identity, resume=resume, overwrite=overwrite)

    def record(self, row):
        self.rows[row['pair_id']] = row
        atomic_write_jsonl(self.path, self.rows.values(), overwrite=True)
