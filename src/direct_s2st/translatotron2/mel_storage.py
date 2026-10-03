"""Non-destructive Mel storage migration; the extraction identity never changes.

Old checkpoint rows stay immutable. A verified sibling copy takes precedence
over the old flat folder, including when that folder cannot be read by DriveFS.
"""
import errno
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import time

from ..hashing import sha256_file
from ..io import atomic_write_json
from ..journal import digest
from ..preparation import file_stamp
from ..progress import operation, track


def guarded_root(root):
    root = Path(os.path.abspath(root))
    # Do not follow links to a different experiment or the read-only corpus.
    if any(p.is_symlink() for p in (root, *root.parents)):
        raise ValueError('Mel cache paths must not contain symlinks')
    corpus = os.environ.get('CORPUS_ROOT')
    if corpus and root.resolve().is_relative_to(Path(corpus).resolve()):
        raise ValueError('Mel cache must be outside CORPUS_ROOT')
    return root


def feature_path(root, name):
    if not re.fullmatch(r'[0-9a-f]{32}\.npy', name):
        raise ValueError(f'invalid Mel cache filename: {name}')
    root = guarded_root(root)
    path = root / 'features-v2' / name[:2] / name
    if any(p.is_symlink() for p in (path, path.parent, path.parent.parent)):
        raise ValueError('Mel feature paths must not contain symlinks')
    return path


def checked_row(root, row):
    source = Path(os.path.abspath(row['path']))
    target = feature_path(root, source.name)
    if source not in (guarded_root(root) / 'features' / source.name, target):
        raise ValueError(f'Mel checkpoint references a path outside its cache: {source}')
    if not re.fullmatch(r'[0-9a-f]{64}', row['sha256']) or not row.get('id'):
        raise ValueError('invalid Mel checkpoint result')
    # Deliberately do not stat/resolve the legacy source here: DriveFS may fail.
    return source, target


def retry_io(function, *, attempts=2):
    """One bounded retry for transient I/O, never for missing/corrupt data."""
    for attempt in range(attempts):
        try:
            return function()
        except OSError as error:
            if error.errno not in (errno.EIO, errno.ETIMEDOUT, errno.EAGAIN) or attempt + 1 == attempts:
                raise
            print(f'[mel-storage] transient I/O errno={error.errno}; retry={attempt+1}/{attempts-1}',
                  file=sys.stderr, flush=True)
            time.sleep(1)


def exists_checked(path):
    # Python versions differ in whether Path.exists suppresses I/O errors.
    # An unreadable published file is NOT an absent file we may replace.
    try:
        retry_io(lambda: Path(path).stat())
        return True
    except FileNotFoundError:
        return False


def verify_feature(path, checksum):
    """Hash once, then reuse only an authenticated receipt with the same stat.

As for corpus checkpoints, this assumes immutable artifacts. RECHECK=1 forces
content reads. Receipts are append-only; changes never replace an old receipt.
"""
    path = Path(path)
    stamp = retry_io(lambda: file_stamp(path))
    receipt = path.with_name(f'.{path.stem}.{digest(stamp)[:16]}.verified.json')
    if receipt.is_symlink():
        raise ValueError('Mel verification receipts must not be symlinks')
    identity = dict(stage='mel-file-verification-v1', sha256=checksum, stamp=stamp)
    if exists_checked(receipt) and os.environ.get('S2ST_PREP_RECHECK') != '1':
        document = json.loads(receipt.read_text(encoding='utf-8'))
        if document != dict(identity=identity, sha256=digest(identity)):
            raise ValueError(f'corrupt Mel verification receipt: {receipt}')
        return
    if retry_io(lambda: sha256_file(path)) != checksum:
        raise ValueError(f'corrupt cached Mel feature: {path}')
    if retry_io(lambda: file_stamp(path)) != stamp:
        raise ValueError(f'Mel feature changed during verification: {path}')
    atomic_write_json(receipt, dict(identity=identity, sha256=digest(identity)), resume=True)


def copy_feature(root, row, writer=None):
    """Publish only a complete checksum-matching copy. Never move/delete source.

writer(temp) may download by Drive file ID; in that mode the old mount path is
never opened. Existing destination conflicts stop rather than being overwritten.
"""
    source, target = checked_row(root, row)
    if exists_checked(target):
        verify_feature(target, row['sha256'])
        return target, False
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=target.parent, prefix='.recover-', suffix='.tmp')
    os.close(fd)
    temporary = Path(name)
    try:
        if writer is None:
            if source.is_symlink() or source.parent.is_symlink():
                raise ValueError('legacy Mel features must not be symlinks')
            retry_io(lambda: shutil.copyfile(source, temporary))
        else:
            writer(temporary)
        if sha256_file(temporary) != row['sha256']:
            raise ValueError(f'recovered Mel SHA256 mismatch: {row["id"]}')
        # No concurrent writer is supported; the second check catches a conflict.
        if exists_checked(target):
            verify_feature(target, row['sha256'])
            return target, False
        os.replace(temporary, target)
        verify_feature(target, row['sha256'])
        return target, True
    finally:
        # Only our unpublished temporary file is disposable, not old features.
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            print(f'[mel-storage] unpublished temporary retained: {temporary}',
                  file=sys.stderr, flush=True)


def resolve_feature(root, row):
    source, target = checked_row(root, row)
    try:
        if exists_checked(target):
            verify_feature(target, row['sha256'])
            return target
        # Resume readable legacy files by verified copy, leaving old rows intact.
        return copy_feature(root, row)[0]
    except OSError as error:
        raise RuntimeError(
            f'Mel cache I/O failed (errno={error.errno}): {source}. '
            f'Keep all features/checkpoints; do not regenerate or delete them. '
            f'Run scripts/colab/repair_mel_cache.py --cache-root "{root}" '
            '--drive-folder-id <legacy features folder ID> after Colab authentication. '
            'Drive quota/permissions/free-space errors must be resolved first.'
        ) from error


@operation('mel recovery: load immutable checkpoints')
def load_rows(root):
    root = guarded_root(root)
    rows, identity = {}, None
    for path in track(sorted(root.glob('chunk-*.json')), 'mel recovery: checkpoint chunks'):
        if path.is_symlink():
            raise ValueError('checkpoint chunks must not be symlinks')
        document = json.loads(path.read_text(encoding='utf-8'))
        if identity is None:
            identity = document['identity']
        if (document['identity'] != identity or identity.get('stage') != 'mel-v1'
                or document['sha256'] != digest(document['rows'])):
            raise ValueError(f'corrupt Mel checkpoint: {path}')
        for key, row in document['rows'].items():
            source, _ = checked_row(root, row)
            if not re.fullmatch(r'[0-9a-f]{64}', key) or source.name != key[:32] + '.npy':
                raise ValueError(f'Mel checkpoint filename/key mismatch: {path}')
            if key in rows and rows[key] != row:
                raise ValueError(f'conflicting Mel checkpoint: {key}')
            rows[key] = row
    if not rows:
        raise ValueError(f'no saved Mel checkpoint rows in {root}')
    # Accept the original identity folder or an explicitly generated child.
    identity_folder = digest(identity)[:24]
    if not (root.name == identity_folder or
            (root.parent.name == identity_folder and re.fullmatch('[0-9a-f]{12}', root.name))):
        raise ValueError('Mel cache root does not match checkpoint identity')
    return rows
