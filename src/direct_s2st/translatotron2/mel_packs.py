"""Local working files, immutable bounded ZIPs on persistent storage.

Packing changes storage only, never Mel extraction identity or payloads. Legacy
rows/files remain untouched. A ZIP is durable only after readback SHA256 and its
receipt; an orphan ZIP is revalidated on restart, never silently replaced.
"""
import ast
from contextlib import closing
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import time
import zipfile

from ..hashing import sha256_file
from ..io import atomic_write_json
from ..journal import digest
from ..preparation import file_stamp
from ..progress import operation, track
from .mel_storage import _checked_row, exists_checked, guarded_root, verify_feature


def npy_frames(path):
    """Read a bounded numeric NPY header without requiring NumPy in host Python."""
    with Path(path).open('rb') as stream:
        if stream.read(6) != b'\x93NUMPY':
            raise ValueError('invalid Mel NPY magic')
        version = stream.read(2)
        if version not in (b'\x01\x00', b'\x02\x00', b'\x03\x00'):
            raise ValueError('unsupported Mel NPY version')
        length_bytes = 2 if version[0] == 1 else 4
        raw = stream.read(length_bytes)
        if len(raw) != length_bytes:
            raise ValueError('truncated Mel NPY header')
        length = int.from_bytes(raw, 'little')
        if length > 65536:
            raise ValueError('oversized Mel NPY header')
        header = ast.literal_eval(stream.read(length).decode('utf-8' if version[0] == 3 else 'latin1'))
        shape, dtype = header['shape'], header['descr']
        if (not isinstance(shape, tuple) or len(shape) != 2
                or any(type(n) is not int or n <= 0 for n in shape)
                or not isinstance(dtype, str) or not re.fullmatch(r'[<>=|]f[248]', dtype)):
            raise ValueError('Mel must be a nonempty numeric float matrix')
        if path.stat().st_size != stream.tell() + shape[0] * shape[1] * int(dtype[2:]):
            raise ValueError('Mel NPY size does not match header')
        return shape


def local_workspace(parent, persistent):
    parent = Path(parent or tempfile.gettempdir()).absolute()
    parent = guarded_root(parent)
    if (parent == persistent or parent.is_relative_to(persistent)
            or str(parent).replace('\\', '/').startswith('/content/drive/')):
        raise ValueError('Mel work directory must be local, not the persistent Drive cache')
    parent.mkdir(parents=True, exist_ok=True)
    return tempfile.TemporaryDirectory(prefix='mel-pack-', dir=parent)


class MelPacks:
    def __init__(self, root, *, local_parent=None, max_files=128, max_bytes=256 * 1024**2):
        if max_files <= 0 or max_bytes <= 0:
            raise ValueError('positive Mel pack limits required')
        self.root = guarded_root(root)
        self.directory = self.root / 'packs-v1'
        if self.directory.is_symlink():
            raise ValueError('Mel packs must not be symlinks')
        self.directory.mkdir(parents=True, exist_ok=True)
        self.owner = local_workspace(local_parent, self.root)
        self.local = Path(self.owner.name)
        self.max_files, self.max_bytes = max_files, max_bytes
        self.rows, self.packs, self.pending = {}, {}, {}
        self.size, self.last_flush = 0, time.monotonic()
        try:
            self._load()
        except BaseException:
            self.owner.cleanup()
            raise

    @operation('mel packs: load durable shard receipts')
    def _load(self):
        for receipt in track(sorted(self.directory.glob('*/pack-*.json')), 'mel packs: verify/reuse shards'):
            if receipt.is_symlink() or receipt.parent.is_symlink():
                raise ValueError('Mel pack receipt must not be a symlink')
            document = json.loads(receipt.read_text(encoding='utf-8'))
            body = {k: v for k, v in document.items() if k != 'receipt_sha256'}
            if (document.get('receipt_sha256') != digest(body)
                    or body.get('version') != 1 or body.get('cache') != self.root.name
                    or receipt.stem != 'pack-' + digest(body['rows'])[:24]
                    or receipt.parent.name != digest(body['rows'])[:2]):
                raise ValueError(f'corrupt Mel pack receipt: {receipt}')
            archive = receipt.with_suffix('.zip')
            if archive.is_symlink():
                raise ValueError('Mel pack archive must not be a symlink')
            if file_stamp(archive) != body['stamp'] or os.environ.get('S2ST_PREP_RECHECK') == '1':
                verify_feature(archive, body['sha256'])
            for key, entry in body['rows'].items():
                self._validate_entry(key, entry)
                if key in self.rows:
                    raise ValueError('duplicate Mel key across packs')
                self.rows[key] = (entry, archive)
            self.packs[archive.name] = body
        print(f'[mel-packs] durable={len(self.rows)} shards={len(self.packs)}', file=sys.stderr, flush=True)

    def _validate_entry(self, key, entry):
        source, _ = _checked_row(self.root, entry['row'])
        if (not re.fullmatch('[0-9a-f]{64}', key) or source.name != key[:32] + '.npy'
                or not re.fullmatch('[A-Za-z0-9_-]+', entry['row']['id'])
                or entry['name'] != entry['row']['id'] + '.npy'
                or type(entry['offset']) is not int or entry['offset'] < 30
                or type(entry['size']) is not int or entry['size'] <= 0
                or len(entry['shape']) != 2 or any(type(n) is not int or n <= 0 for n in entry['shape'])):
            raise ValueError('invalid Mel pack entry')

    def add(self, key, row, path):
        """Consume a PRIVATE local file. Do not pass original Drive files here."""
        path = Path(path)
        if path.parent != self.local or path.is_symlink():
            raise ValueError('pack input must be an owned local temporary file')
        if key in self.rows or key in self.pending:
            raise ValueError('duplicate Mel pack key')
        _checked_row(self.root, row)
        if sha256_file(path) != row['sha256']:
            raise ValueError(f'recovered Mel SHA256 mismatch: {row["id"]}')
        shape = npy_frames(path)
        size = path.stat().st_size
        if self.pending and self.size + size > self.max_bytes:
            self.flush()
        self.pending[key] = dict(row=row, local=path, shape=shape)
        self.size += size
        if len(self.pending) >= self.max_files or (len(self.pending) >= 32 and time.monotonic() - self.last_flush >= 60):
            self.flush()

    @operation('mel packs: publish verified ZIP checkpoint')
    def flush(self):
        if not self.pending:
            return
        local_zip = self.local / 'pending.zip'
        entries = {}
        with zipfile.ZipFile(local_zip, 'w', zipfile.ZIP_STORED, allowZip64=False) as archive:
            for key, item in self.pending.items():
                row, path = item['row'], item['local']
                name = row['id'] + '.npy'
                if name in {e['name'] for e in entries.values()}:
                    raise ValueError('duplicate Mel sample ID')
                archive.write(path, arcname=name)
                info = archive.getinfo(name)
                entry = dict(row=row, name=name, shape=item['shape'], size=info.file_size,
                             offset=info.header_offset + 30 + len(name.encode('utf-8')) + len(info.extra))
                self._validate_entry(key, entry)
                entries[key] = entry
        checksum = sha256_file(local_zip)
        pack_id = digest(entries)
        bucket = self.directory / pack_id[:2]
        if bucket.is_symlink():
            raise ValueError('Mel pack bucket must not be a symlink')
        bucket.mkdir(parents=True, exist_ok=True)
        archive_path = bucket / f'pack-{pack_id[:24]}.zip'
        # ZIP timestamps are nondeterministic across rebuilt workspaces; adopt an
        # orphan only after matching every member to the immutable expected SHA.
        if exists_checked(archive_path):
            if archive_path.is_symlink():
                raise ValueError('Mel pack archive must not be a symlink')
            with zipfile.ZipFile(archive_path) as existing:
                if existing.namelist() != [e['name'] for e in entries.values()]:
                    raise ValueError('conflicting orphan Mel pack')
                for entry in entries.values():
                    info = existing.getinfo(entry['name'])
                    if (info.compress_type != zipfile.ZIP_STORED or info.file_size != entry['size']
                            or info.header_offset + 30 + len(info.filename.encode('utf-8')) + len(info.extra) != entry['offset']
                            or hashlib.sha256(existing.read(info)).hexdigest() != entry['row']['sha256']):
                        raise ValueError('corrupt orphan Mel pack')
            checksum = sha256_file(archive_path)
        else:
            fd, name = tempfile.mkstemp(dir=bucket, prefix='.publish-', suffix='.tmp')
            os.close(fd)
            temporary = Path(name)
            try:
                shutil.copyfile(local_zip, temporary)
                if sha256_file(temporary) != checksum:
                    raise ValueError('Mel ZIP readback SHA256 mismatch')
                if exists_checked(archive_path):
                    raise FileExistsError(archive_path)
                os.replace(temporary, archive_path)
            finally:
                temporary.unlink(missing_ok=True)
        verify_feature(archive_path, checksum)
        body = dict(version=1, cache=self.root.name, rows=entries,
                    sha256=checksum, stamp=file_stamp(archive_path))
        atomic_write_json(archive_path.with_suffix('.json'),
                          {**body, 'receipt_sha256': digest(body)}, resume=True)
        for key, entry in entries.items():
            self.rows[key] = (entry, archive_path)
        self.packs[archive_path.name] = body
        print(f'[mel-packs] persisted={len(self.rows)} published={len(entries)} shards={len(self.packs)}',
              file=sys.stderr, flush=True)
        # Only files created in this instance's private local directory are removed.
        for item in self.pending.values():
            item['local'].unlink()
        local_zip.unlink()
        self.pending.clear()
        self.size, self.last_flush = 0, time.monotonic()

    def __enter__(self):
        return self

    def __exit__(self, kind, error, tb):
        try:
            # A normal interruption still saves completed entries, not the failing
            # one. A killed VM loses only its uncommitted local shard.
            self.flush()
        finally:
            self.owner.cleanup()


@operation('mel packs: export prepared training archives')
def export_packs(store, keys, zip_path, *, resume=False, overwrite=False):
    """Copy whole shards, never unpack NPY files onto Drive."""
    selected = {store.rows[key][1].name for key in keys}
    sources = {archive.name: archive for _, archive in store.rows.values()}
    names = sorted(selected)
    mapping = {name: (zip_path.name if i == 0 else f'{zip_path.stem}-packed-{Path(name).stem[5:]}.zip')
               for i, name in enumerate(names)}
    plan = dict(version=2, packs=mapping)
    atomic_write_json(zip_path.with_suffix('.shards.json'), plan, resume=resume, overwrite=overwrite)
    paths, lengths, hashes = {}, {}, {}
    for name in track(names, 'mel packs: export/reuse ZIP shards'):
        source, destination = sources[name], zip_path.parent / mapping[name]
        checksum = store.packs[name]['sha256']
        if overwrite or not exists_checked(destination):
            fd, tempname = tempfile.mkstemp(dir=destination.parent, prefix='.export-', suffix='.tmp')
            os.close(fd)
            temporary = Path(tempname)
            try:
                shutil.copyfile(source, temporary)
                if sha256_file(temporary) != checksum:
                    raise ValueError('exported Mel ZIP checksum mismatch')
                if exists_checked(destination) and not overwrite:
                    raise FileExistsError(destination)
                os.replace(temporary, destination)
            finally:
                temporary.unlink(missing_ok=True)
        elif not resume:
            raise FileExistsError(destination)
        verify_feature(destination, checksum)
        hashes[destination.name] = checksum
    for key in keys:
        entry, archive = store.rows[key]
        pair_id = entry['row']['id']
        if pair_id in paths:
            raise ValueError('duplicate prepared Mel sample ID')
        paths[pair_id] = f'{mapping[archive.name]}:{entry["offset"]}:{entry["size"]}'
        lengths[pair_id] = entry['shape'][0]
    return paths, lengths, hashes


@operation('mel: local extraction and packed checkpoints')
def prepare_packed(cache, items, extractor, settings, zip_path, *, resume=False, overwrite=False):
    from ..preparation import adaptive_map
    from .mel_storage import checked_row
    import numpy as np
    saved = dict(cache.rows)
    with MelPacks(cache.root, local_parent=os.environ.get('S2ST_MEL_LOCAL_WORK')) as store:
        if store.rows and not (resume or overwrite):
            from ..io import ExistingOutputError
            raise ExistingOutputError('packed Mel cache exists; pass --resume or --overwrite')
        durable = dict(store.rows)
        def work(item):
            pair_id, audio = item
            key = digest([pair_id, file_stamp(audio)])
            prior = saved.get(key)
            if key in durable:
                entry, _ = durable[key]
                if prior is not None and entry['row'] != prior:
                    raise ValueError('packed Mel differs from legacy checkpoint')
                if entry['row']['id'] != pair_id or entry['shape'][1] != settings['n_mels']:
                    raise ValueError('packed Mel does not match sample/frontend')
                return key, None, None, True
            fd, name = tempfile.mkstemp(dir=store.local, suffix='.npy')
            os.close(fd)
            local = Path(name)
            if prior is not None:
                source, distributed = checked_row(cache.root, prior)
                source = distributed if exists_checked(distributed) else source
                if source.is_symlink() or source.parent.is_symlink():
                    raise ValueError('Mel source must not be a symlink')
                try:
                    shutil.copyfile(source, local)
                except OSError as error:
                    raise RuntimeError(f'Keep existing Mel files. Repair this cache into ZIP packs first: {cache.root}') from error
                row = prior
            else:
                extractor(audio, local, settings)
                destination = cache.root / 'features-v2' / key[:2] / (key[:32] + '.npy')
                # This path is a legacy-compatible identity only; no NPY is
                # published there. Packed receipts are the durable checkpoint.
                row = dict(id=pair_id, path=str(destination), sha256=sha256_file(local))
            array = np.load(local, allow_pickle=False)
            if (array.ndim != 2 or array.shape[0] == 0 or array.shape[1] != settings['n_mels']
                    or not np.isfinite(array).all()):
                raise ValueError(f'invalid Mel feature: {pair_id}')
            return key, row, local, prior is not None

        keys, computed, reused = [], 0, 0
        reported = time.monotonic()
        # Join/cancel workers before releasing their owned local directory, even
        # when publishing a result fails while other native jobs are running.
        with closing(adaptive_map(work, items)) as workers:
            with closing(track(workers, 'mel: extract/reuse local features', total=len(items))) as progress:
                for key, row, local, reuse in progress:
                    keys.append(key)
                    if local is not None:
                        store.add(key, row, local)
                    reused += reuse
                    computed += not reuse
                    if time.monotonic() - reported >= 10:
                        print(f'[checkpoint-progress] mel packed: reused={reused} computed={computed} '
                              f'persisted={len(store.rows)} pending_local={len(store.pending)}', file=sys.stderr, flush=True)
                        reported = time.monotonic()
        store.flush()
        return export_packs(store, keys, zip_path, resume=resume, overwrite=overwrite)
