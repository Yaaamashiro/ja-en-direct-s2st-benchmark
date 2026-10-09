"""Verified, resumable local inputs. Never fall back to remote training I/O.

Source WAVs are read once into immutable derived ZIPs outside CORPUS_ROOT.
Later VMs download whole ZIPs, not thousands of individual WAVs. Original paths
and stamps stay logical identities; only actual reads are redirected locally.
"""
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
import zipfile

from .io import atomic_write_json, read_jsonl
from .journal import digest
from .progress import operation, track


_maps = {}


def logical(path):
    return os.path.abspath(path)


def mapping():
    name = os.environ.get('S2ST_INPUT_MAP')
    if not name:
        return {}
    if name not in _maps:
        document = json.loads(Path(name).read_text(encoding='utf-8'))
        if document['sha256'] != digest(document['rows']):
            raise ValueError('corrupt local input map')
        _maps[name] = document['rows']
    return _maps[name]


def local_path(path):
    entry = mapping().get(logical(path))
    if entry is None:
        return Path(path)
    target = Path(entry['local'])
    stat = target.stat()
    if target.is_symlink() or [stat.st_size, stat.st_mtime_ns] != entry['local_stamp']:
        raise ValueError(f'staged input changed: {path}')
    return target


def source_stamp(path):
    entry = mapping().get(logical(path))
    if entry is None:
        return None
    local_path(path)
    return entry['source_stamp']


def ensure_local(root):
    if str(root).replace('\\', '/').startswith('/content/drive/'):
        raise ValueError('input staging must be on the Colab VM, not Drive')
    root = Path(root).absolute()
    corpus = os.environ.get('CORPUS_ROOT')
    if corpus and root.resolve().is_relative_to(Path(corpus).resolve()):
        raise ValueError('staging must be outside CORPUS_ROOT')
    if any(p.is_symlink() for p in (root, *root.parents)):
        raise ValueError('staging path must not contain symlinks')
    root.mkdir(parents=True, exist_ok=True)
    return root


def capacity(root, amount, reserve=2 * 1024**3):
    if shutil.disk_usage(root).free < amount + reserve:
        raise RuntimeError(f'Local disk insufficient: need {amount + reserve} free bytes. '
                           'Training has not started; retain all Drive data and use a larger local disk.')


def raw_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(1024**2):
            h.update(block)
    return h.hexdigest()


@operation('Drive inputs: copy whole file and verify locally')
def copy_verified(source, target, expected=None):
    source, target = Path(source), Path(target)
    before = source.stat()
    target.parent.mkdir(parents=True, exist_ok=True)
    capacity(target.parent, before.st_size)
    fd, name = tempfile.mkstemp(dir=target.parent, prefix='.copy-')
    temp = Path(name)
    try:
        h = hashlib.sha256()
        with os.fdopen(fd, 'wb') as output, source.open('rb') as stream:
            while block := stream.read(1024**2):
                output.write(block)
                h.update(block)
        after = source.stat()
        if ((before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns)
                or (expected is not None and h.hexdigest() != expected)
                or raw_sha(temp) != h.hexdigest()):
            raise ValueError(f'input copy checksum/stamp mismatch: {source}')
        shutil.copystat(source, temp)
        os.replace(temp, target)
        return h.hexdigest()
    finally:
        temp.unlink(missing_ok=True)


def register(rows, source, target, stamp):
    stat = target.stat()
    rows[logical(source)] = dict(local=str(target), source_stamp=stamp,
                                 local_stamp=[stat.st_size, stat.st_mtime_ns])


def audio_archive_path(persistent, manifest_sha, split, language, start, pack_files):
    # Keep the 55a5a90 namespace/layout: interrupted 4B packs remain reusable.
    namespace = digest([manifest_sha, split, language, pack_files])[:24]
    return Path(persistent) / namespace[:2] / namespace / digest(start)[:2] / f'{start:08d}.zip'


def missing_audio_pack(archive):
    return RuntimeError(
        f'CPU audio packing incomplete: {archive}. '
        '先にCPUランタイムでセル4A.5（音声ZIP準備）を実行してください。'
        '完成したZIPは再利用します。4Bでは原本からZIPを作成しません。')


@operation('Drive inputs: require completed CPU audio packs')
def require_audio_packs(common, persistent, *, splits=('train', 'dev', 'test'),
                        languages=('en', 'ja'), pack_files=128):
    """Cheap preflight before downloads/models: no original WAV reads or writes.

    This checks publication, not payload integrity; staging checks receipt
    identity and archive/member SHA256 before passing inputs to HuBERT.
    """
    from .translatotron2.mel_storage import guarded_root
    persistent = guarded_root(persistent)
    if not 1 <= pack_files <= 512:
        raise ValueError('audio pack_files must be between 1 and 512')
    checked = 0
    for split in splits:
        manifest = Path(common) / f'{split}.jsonl'
        count = sum(1 for _ in read_jsonl(manifest))
        manifest_sha = raw_sha(manifest)
        for language in languages:
            for start in track(range(0, count, pack_files), f'require WAV ZIPs: {split}/{language}'):
                archive = audio_archive_path(persistent, manifest_sha, split, language, start, pack_files)
                if not archive.is_file() or not archive.with_suffix('.json').is_file():
                    raise missing_audio_pack(archive)
                checked += 1
    return checked


@operation('Drive inputs: resumable WAV archive staging')
def stage_audio(common, persistent, local, *, splits=('train', 'dev', 'test'), languages=('ja', 'en'),
                pack_files=128, create=True, materialize=True):
    from .preparation import file_stamp
    from .translatotron2.mel_storage import guarded_root, verify_feature
    common, persistent, local = Path(common), guarded_root(persistent), ensure_local(local)
    if pack_files < 1 or pack_files > 512:
        raise ValueError('audio pack_files must be between 1 and 512')
    if local.is_relative_to(persistent) or persistent.is_relative_to(local):
        raise ValueError('local inputs and persistent packs must be separate')
    if not create:
        require_audio_packs(common, persistent, splits=splits, languages=languages, pack_files=pack_files)
    output = {}
    built = reused = 0
    last_report = float('-inf')
    for split in splits:
        # Resolve paths only when BUILDING a new shard. Completed shards need no
        # stat/open of their original WAVs, including after a runtime reset.
        items = list(read_jsonl(common / f'{split}.jsonl'))
        input_hash = raw_sha(common / f'{split}.jsonl')
        for language in languages:
            namespace = digest([input_hash, split, language, pack_files])[:24]
            namespace_root = persistent / namespace[:2] / namespace
            if namespace_root.is_symlink():
                raise ValueError('audio pack directory must not be a symlink')
            for start in track(range(0, len(items), pack_files), f'stage WAV ZIPs: {split}/{language}'):
                directory = namespace_root / digest(start)[:2]
                if directory.is_symlink():
                    raise ValueError('audio pack bucket must not be a symlink')
                directory.mkdir(parents=True, exist_ok=True)
                selected = items[start:start + pack_files]
                expected = {str(row['pair_id']): row[f'{language}_sha256'].lower() for row in selected}
                if len(expected) != len(selected) or any(len(h) != 64 or any(c not in '0123456789abcdef' for c in h) for h in expected.values()):
                    raise ValueError('audio staging requires unique IDs and manifest SHA256')
                archive = audio_archive_path(persistent, input_hash, split, language, start, pack_files)
                receipt = archive.with_suffix('.json')
                plan = archive.with_suffix('.plan.json')
                if not create and not receipt.is_file():
                    raise missing_audio_pack(archive)
                if not receipt.exists() and archive.exists() and plan.is_file():
                    saved = json.loads(plan.read_text(encoding='utf-8'))
                    if saved['sha256'] != digest(saved['body']) or saved['body']['expected'] != expected:
                        raise ValueError('corrupt audio publication plan')
                    verify_feature(archive, saved['body']['archive_sha256'])
                    atomic_write_json(receipt, saved)
                if receipt.is_file():
                    document = json.loads(receipt.read_text(encoding='utf-8'))
                    body = document['body']
                    if document['sha256'] != digest(body) or body['expected'] != expected:
                        raise ValueError('corrupt/stale audio pack receipt')
                    verify_feature(archive, body['archive_sha256'])
                    reused += 1
                else:
                    if not create:
                        raise missing_audio_pack(archive)
                    # An unpublished archive is not trusted and is not replaced.
                    if archive.exists():
                        raise RuntimeError(f'Uncommitted audio pack retained: {archive}; inspect before retrying')
                    with tempfile.TemporaryDirectory(prefix='audio-pack-', dir=local) as work:
                        temp_zip = Path(work) / 'audio.zip'
                        entries = {}
                        with zipfile.ZipFile(temp_zip, 'w', zipfile.ZIP_STORED) as pack:
                            for raw in track(selected, 'stage WAVs: read/verify originals'):
                                row = read_common_manifest_row(raw, language)
                                source = Path(row[f'{language}_audio'])
                                stamp = file_stamp(source)
                                name = digest([raw['pair_id'], language])[:32] + '.wav'
                                h = hashlib.sha256()
                                with source.open('rb') as stream, pack.open(name, 'w') as target:
                                    while block := stream.read(1024**2):
                                        capacity(local, len(block))
                                        target.write(block)
                                        h.update(block)
                                if h.hexdigest() != expected[str(raw['pair_id'])] or file_stamp(source) != stamp:
                                    raise ValueError(f'audio checksum/stamp mismatch: {raw["pair_id"]}')
                                entries[name] = dict(source=str(source), stamp=stamp, sha256=h.hexdigest())
                        checksum = raw_sha(temp_zip)
                        body = dict(expected=expected, entries=entries, archive_sha256=checksum)
                        atomic_write_json(plan, dict(body=body, sha256=digest(body)), resume=True)
                        # Immutable publication: copied bytes are verified before
                        # the final name, then the completion receipt is written.
                        fd, name = tempfile.mkstemp(dir=directory, prefix='.publish-')
                        os.close(fd)
                        pending = Path(name)
                        try:
                            capacity(directory, temp_zip.stat().st_size)
                            shutil.copyfile(temp_zip, pending)
                            if raw_sha(pending) != checksum:
                                raise ValueError('audio pack readback mismatch')
                            if archive.exists():
                                raise FileExistsError(archive)
                            os.replace(pending, archive)
                        finally:
                            pending.unlink(missing_ok=True)
                    atomic_write_json(receipt, dict(body=body, sha256=digest(body)))
                    built += 1
                names = {digest([raw['pair_id'], language])[:32] + '.wav': expected[str(raw['pair_id'])]
                         for raw in selected}
                if (set(body['entries']) != set(names)
                        or any(e['sha256'] != names[name] for name, e in body['entries'].items())):
                    raise ValueError('audio pack entries differ from common manifest')
                now = time.monotonic()
                if now - last_report >= 10 or start + pack_files >= len(items):
                    print(f'[audio-packs] mode={"CPU publish" if not materialize else "local staging"} '
                          f'split={split}/{language} zip={start // pack_files + 1}/{(len(items) + pack_files - 1) // pack_files} '
                          f'audio={min(start + pack_files, len(items))}/{len(items)} '
                          f'built={built} reused={reused} zip_size_max={pack_files} originals_preserved=true',
                          file=sys.stderr, flush=True)
                    last_report = now
                if not materialize:
                    # CPU stage keeps only immutable Drive ZIPs, not all WAVs on
                    # this disposable VM. Switching runtimes cannot lose them.
                    continue
                pack_local = local / namespace / f'{start:08d}'
                pack_local.mkdir(parents=True, exist_ok=True)
                committed = pack_local / 'complete.json'
                reusable = False
                if committed.is_file():
                    state = json.loads(committed.read_text(encoding='utf-8'))
                    reusable = state.get('identity') == digest(body) and all(
                        (pack_local / name).is_file() and raw_sha(pack_local / name) == e['sha256']
                        for name, e in body['entries'].items())
                if not reusable:
                    capacity(local, sum(e['stamp']['size'] for e in body['entries'].values()) * 2)
                    zipped = pack_local / 'download.zip'
                    copy_verified(archive, zipped, body['archive_sha256'])
                    with zipfile.ZipFile(zipped) as pack:
                        if set(pack.namelist()) != set(body['entries']) or len(pack.namelist()) != len(body['entries']):
                            raise ValueError('audio pack member mismatch')
                        for name, entry in body['entries'].items():
                            if Path(name).name != name or not name.endswith('.wav'):
                                raise ValueError('unsafe audio pack member')
                            target = pack_local / name
                            with pack.open(name) as source, target.open('wb') as out:
                                shutil.copyfileobj(source, out)
                            if raw_sha(target) != entry['sha256']:
                                raise ValueError('unpacked audio checksum mismatch')
                    atomic_write_json(committed, dict(identity=digest(body)), overwrite=True)
                    zipped.unlink()  # our verified local download only
                for name, entry in body['entries'].items():
                    register(output, entry['source'], pack_local / name, entry['stamp'])
                corpus = os.environ.get('CORPUS_ROOT')
                for raw in selected:
                    name = digest([raw['pair_id'], language])[:32] + '.wav'
                    entry = body['entries'][name]
                    # Portable aliases are derived lexically from the accepted
                    # common row; no access to old/new Drive WAV paths is needed.
                    aliases = [Path(raw[f'{language}_audio'])]
                    relative = raw.get(f'{language}_audio_corpus_relative')
                    if corpus and relative:
                        if Path(relative).is_absolute() or '..' in Path(relative).parts:
                            raise ValueError('unsafe corpus-relative audio alias')
                        aliases.append(Path(corpus).absolute() / relative)
                    for alias in aliases:
                        if alias.is_absolute():
                            stamp = dict(entry['stamp'], path=str(alias))
                            register(output, alias, pack_local / name, stamp)
    print(f'[drive-staging] built_zip={built} reused_zip={reused} local_audio={len(output)} '
          'originals_preserved=true', file=sys.stderr, flush=True)
    return output


def read_common_manifest_row(row, language):
    """Same portable resolver without creating a temporary common manifest."""
    from .manifests.paths import resolve_audio_path
    value = dict(row)
    corpus = os.environ.get('CORPUS_ROOT')
    if corpus:
        raw = row.get(f'{language}_audio_corpus_relative') or row[f'{language}_audio']
        value[f'{language}_audio'] = str(resolve_audio_path(raw, Path(corpus)))
    return value


@contextmanager
def active_map(rows, local):
    name = Path(local) / 'input-map.json'
    atomic_write_json(name, dict(rows=rows, sha256=digest(rows)), overwrite=True)
    prior = os.environ.get('S2ST_INPUT_MAP')
    _maps.pop(str(name), None)
    os.environ['S2ST_INPUT_MAP'] = str(name)
    try:
        yield name
    finally:
        _maps.pop(str(name), None)
        if prior is None:
            os.environ.pop('S2ST_INPUT_MAP', None)
        else:
            os.environ['S2ST_INPUT_MAP'] = prior
