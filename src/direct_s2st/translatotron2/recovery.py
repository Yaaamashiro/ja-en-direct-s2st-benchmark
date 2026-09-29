"""Bounded Mel archives and per-example final validation recovery.

Reuse assumes immutable inputs: size/mtime are checked, not a fresh content hash
on every resume. S2ST_PREP_RECHECK=1 explicitly requests full verification.
"""
import json
import os
from pathlib import Path

from ..hashing import sha256_file
from ..io import atomic_write_json
from ..journal import digest
from ..preparation import Checkpoints, file_stamp
from ..progress import operation, track


@operation('mel: publish bounded archives')
def publish_archives(files, zip_path, *, resume=False, overwrite=False,
                     max_files=512, max_bytes=256 * 1024**2):
    from .prepare_fairseq import _write_feature_zip, _zip_manifest
    plan_path = zip_path.with_suffix('.shards.json')
    groups, current, size = [], [], 0
    for name, path in files:
        length = path.stat().st_size
        if current and (len(current) >= max_files or size + length > max_bytes):
            groups.append(current)
            current, size = [], 0
        current.append((name, path))
        size += length
    if current:
        groups.append(current)
    plan = dict(version=1, groups=[[name for name, _ in group] for group in groups])
    atomic_write_json(plan_path, plan, resume=resume, overwrite=overwrite)
    paths, lengths, hashes = {}, {}, {}
    for index, group in track(enumerate(groups), 'mel: archive shards', total=len(groups)):
        archive = zip_path if index == 0 else zip_path.with_name(f'{zip_path.stem}-{index:05d}.zip')
        marker = archive.with_suffix('.complete.json')
        inputs = [[name, file_stamp(path)] for name, path in group]
        prior = json.loads(marker.read_text()) if marker.is_file() and resume and not overwrite else None
        if prior and prior.get('receipt_sha256') != digest({k: v for k, v in prior.items() if k != 'receipt_sha256'}):
            raise ValueError(f'corrupt Mel archive receipt: {marker}')
        if prior and prior['inputs'] == inputs and archive.is_file() and prior['stamp'] == file_stamp(archive) and os.environ.get('S2ST_PREP_RECHECK') != '1':
            manifest, frames, checksum = prior['paths'], prior['lengths'], prior['sha256']
        else:
            if not archive.exists() or overwrite:
                _write_feature_zip(group[0][1].parent, archive, group)
            elif not resume:
                raise FileExistsError(archive)
            # A crash after ZIP publication but before its marker is recoverable,
            # but only after checking every archived payload against its source.
            import zipfile
            import hashlib
            with zipfile.ZipFile(archive) as handle:
                if handle.namelist() != [name for name, _ in group]:
                    raise ValueError(f'conflicting Mel archive: {archive}')
                for name, path in group:
                    if hashlib.sha256(handle.read(name)).hexdigest() != sha256_file(path):
                        raise ValueError(f'corrupt Mel archive: {archive}/{name}')
            manifest, frames = _zip_manifest(archive)
            checksum = sha256_file(archive)
            receipt = dict(inputs=inputs, stamp=file_stamp(archive),
                           paths=manifest, lengths=frames, sha256=checksum)
            atomic_write_json(marker, {**receipt, 'receipt_sha256': digest(receipt)}, overwrite=True)
        if paths.keys() & manifest.keys():
            raise ValueError('duplicate feature across archive shards')
        paths.update(manifest)
        lengths.update(frames)
        hashes[archive.name] = checksum
    return paths, lengths, hashes


@operation('tt2: resumable final validation')
def validate_prepared(root, *, resume=False, overwrite=False):
    from .data import PreparedDataset, fingerprint
    from ..manifests.paths import resolve_audio_path
    root = Path(root)
    datasets = {split: PreparedDataset(root, split) for split in ('train', 'dev', 'test')}
    ids = [row['id'] for dataset in datasets.values() for row in dataset.rows]
    if len(ids) != len(set(ids)):
        raise ValueError('duplicate IDs across splits')
    # Small config/TSV hashes bind model frontend, labels, split and archive locators.
    configs = {p.relative_to(root).as_posix(): sha256_file(p) for p in sorted(root.rglob('*'))
               if p.is_file() and p.suffix in ('.json', '.yaml', '.tsv', '.txt')
               and not p.name.endswith('.complete.json')}
    stamps = {p.name: file_stamp(p) for p in root.glob('*.zip')}
    reuse = resume and not overwrite and os.environ.get('S2ST_PREP_RECHECK') != '1'
    with Checkpoints(root.parent / '.prep-checkpoints/final-validation',
                     dict(stage='tt2-final-v1', config=configs), resume=reuse, overwrite=overwrite) as cache:
        for split, dataset in datasets.items():
            for index in track(range(len(dataset)), f'tt2: validate/reuse {split}'):
                row = dataset.rows[index]
                source = (resolve_audio_path(row['src_audio'], dataset.corpus_root)
                          if dataset.corpus_root else Path(row['src_audio']))
                name = row['tgt_audio'].rsplit(':', 2)[0]
                key = digest(dict(split=split, row=row, source=file_stamp(source), archive=stamps[name]))
                if cache.get(key) is None:
                    dataset[index]
                    cache.record(key, dict(verified=True))
    return dict(splits={split: len(dataset) for split, dataset in datasets.items()}, fingerprint=fingerprint(root))
