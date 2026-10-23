"""Small immutable ZIP recovery units for benchmark audio and journals.

Receipts are published last. Originals are never removed. Archive members and
checksums are validated before extraction, including model-free listening.
"""
import json
import os
from pathlib import Path, PurePosixPath
import re
import tempfile
import zipfile

from .drive_staging import copy_verified, raw_sha
from .io import atomic_write_json
from .journal import digest
from .progress import operation, track


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def safe_name(name):
    value = PurePosixPath(name)
    if (not name or value.is_absolute() or '..' in value.parts or '\\' in name
            or ':' in name or value.as_posix() != name):
        raise ValueError('unsafe benchmark archive member')
    return value


def guarded(root):
    root = Path(root).absolute()
    corpus = os.environ.get('CORPUS_ROOT')
    if corpus and root.resolve().is_relative_to(Path(corpus).resolve()):
        raise ValueError('benchmark outputs must be outside CORPUS_ROOT')
    if any(p.is_symlink() for p in (root, *root.parents)):
        raise ValueError('benchmark output must not contain symlinks')
    root.mkdir(parents=True, exist_ok=True)
    return root


@operation('benchmark: publish verified ZIP recovery unit')
def publish_tree(source, destination, label, identity):
    source, destination = guarded(source), guarded(destination)
    if not re.fullmatch('[A-Za-z0-9_-]+', label):
        raise ValueError('invalid benchmark archive label')
    files = {}
    with tempfile.TemporaryDirectory(prefix='benchmark-pack-', dir=source.parent) as temp:
        archive = Path(temp)/'part.zip'
        with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_STORED) as pack:
            for path in track(sorted(source.rglob('*')), 'benchmark: pack verified results'):
                if path.is_symlink():
                    raise ValueError('benchmark archive cannot contain symlinks')
                if not path.is_file():
                    continue
                name = path.relative_to(source).as_posix()
                safe_name(name)
                checksum = raw_sha(path)
                info = zipfile.ZipInfo(name)  # Stable timestamps for content-addressed reuse.
                with path.open('rb') as stream, pack.open(info, 'w') as output:
                    import shutil
                    shutil.copyfileobj(stream, output, 1024**2)
                if raw_sha(path) != checksum:
                    raise ValueError('benchmark file changed during backup')
                files[name] = dict(sha256=checksum, size=path.stat().st_size)
        checksum = raw_sha(archive)
        target = destination/(label+'-'+checksum+'.zip')
        if not target.is_file():
            copy_verified(archive, target, checksum)
        elif raw_sha(target) != checksum:
            raise ValueError('corrupt existing benchmark archive; original preserved')
    body = dict(version=1, identity=identity, archive=target.name, sha256=checksum, files=files)
    receipt = dict(body=body, sha256=digest(body))
    atomic_write_json(destination/(label+'.json'), receipt, overwrite=True)
    return receipt


@operation('benchmark: restore verified ZIP recovery unit')
def restore_tree(destination, label, target, identity, *, members=None):
    destination, target = Path(destination), guarded(target)
    receipt = read(destination/(label+'.json'))
    body = receipt['body']
    if receipt['sha256'] != digest(body) or body['identity'] != identity or body.get('version') != 1:
        raise ValueError('benchmark receipt identity/checksum mismatch')
    safe_name(body['archive'])
    if '/' in body['archive'] or not re.fullmatch('[0-9a-f]{64}', body['sha256']):
        raise ValueError('invalid benchmark archive identity')
    wanted = set(body['files']) if members is None else set(members)
    if not wanted <= body['files'].keys():
        raise ValueError('requested benchmark member missing')
    for name in body['files']:
        safe_name(name)
    with tempfile.TemporaryDirectory(prefix='benchmark-restore-', dir=target.parent) as temp:
        archive = Path(temp)/'part.zip'
        copy_verified(destination/body['archive'], archive, body['sha256'])
        with zipfile.ZipFile(archive) as pack:
            entries = pack.infolist()
            if len({e.filename for e in entries}) != len(entries) or set(pack.namelist()) != set(body['files']):
                raise ValueError('benchmark archive member set mismatch')
            for entry in entries:
                if (entry.file_size != body['files'][entry.filename]['size']
                        or entry.file_size > 2 * 1024**3 or (entry.external_attr >> 16) & 0o170000 == 0o120000):
                    raise ValueError('unsafe benchmark archive size/type')
            for name in track(sorted(wanted), 'benchmark: verify/extract saved results'):
                path = target.joinpath(*safe_name(name).parts)
                guarded(path.parent)
                if path.is_symlink():
                    raise ValueError('benchmark extraction target is a symlink')
                if path.is_file():
                    if raw_sha(path) == body['files'][name]['sha256']:
                        continue
                    raise ValueError('different local benchmark artifact retained: '+str(path))
                temporary = Path(temp)/'member'
                with pack.open(name) as source, temporary.open('wb') as output:
                    import shutil
                    shutil.copyfileobj(source, output, 1024**2)
                if raw_sha(temporary) != body['files'][name]['sha256']:
                    raise ValueError('benchmark member checksum mismatch')
                os.replace(temporary, path)
    return receipt
