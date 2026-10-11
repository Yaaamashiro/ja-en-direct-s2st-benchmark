"""On-demand, verified TT2 inputs with bounded background shard prefetch.

Only metadata and declared archive identities are needed at startup. Payloads
are checked against those identities before being exposed to the trainer.
Logical paths, sample order and optimizer snapshots remain unchanged.
"""
import atexit
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import threading
import time
import zipfile

from .io import atomic_write_json, read_jsonl
from .journal import digest
from .progress import operation, track


def checksum(value):
    if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError('streaming requires a saved lowercase SHA256')
    return value


@operation('training inputs: plan verified on-demand TT2 shards')
def plan(config, work):
    from .drive_staging import (ensure_local, logical, audio_archive_path,
                                copy_verified, register, raw_sha)
    from .preparation import file_stamp
    if config['kind'] != 'tt2':
        raise ValueError('on-demand input planning currently supports TT2 only')
    files = [Path(p) for p in config['identity_files']]
    locks = [p for p in files if p.name == 'data-lock.json']
    if len(locks) != 1:
        raise ValueError('streaming requires one prepared TT2 data-lock.json')
    root = locks[0].parent
    common = root.parent.parent/'common'
    local = ensure_local(Path(work).parent/'.inputs'/digest([str(common), 'tt2-stream-v1'])[:16])
    # Small metadata is copied first, not deferred. Archive hashes come ONLY
    # from the prepared lock, never from a newly computed/unverified checksum.
    rows, jobs = {}, {}
    archives = [p for p in files if p.suffix == '.zip']
    for source in track([p for p in files if p.suffix != '.zip'], 'stream inputs: small metadata'):
        if source.is_symlink():
            raise ValueError('streaming metadata must not be symlinks')
        before = file_stamp(source)
        target = local/'prepared'/digest(str(source))[:24]/source.name
        copy_verified(source, target)
        if file_stamp(source) != before:
            raise ValueError('metadata changed during streaming setup')
        register(rows, source, target, before)
    lock = json.loads(Path(rows[logical(locks[0])]['local']).read_text(encoding='utf-8'))
    if raw_sha(common/'dataset-lock.json') != checksum(lock.get('common_dataset_lock_sha256')):
        raise ValueError('common dataset lock differs from prepared TT2 data')
    expected = lock.get('mel_shards_sha256')
    if expected is None:
        if len(archives) != 1:
            raise ValueError('prepared Mel shard hashes are missing; use full staging for this legacy layout')
        expected = {archives[0].name: lock.get('mel_zip_sha256')}
    if not isinstance(expected, dict) or set(expected) != {p.name for p in archives}:
        raise ValueError('prepared Mel archives differ from their saved lock')
    for source in archives:
        if source.is_symlink():
            raise ValueError('streaming archives must not be symlinks')
        before = file_stamp(source)
        target = local/'prepared'/digest(str(source))[:24]/source.name
        job_id = digest(['mel', str(source), expected[source.name]])[:24]
        jobs[job_id] = dict(kind='mel', source=str(source), target=str(target),
                            sha256=checksum(expected[source.name]), stamp=before)
        rows[logical(source)] = dict(local=str(target), job=job_id, member=source.name,
                                     source_stamp=before, declared_sha256=expected[source.name])
    corpus = os.environ.get('CORPUS_ROOT')
    for split in ('train', 'dev', 'test'):
        items = list(read_jsonl(common/f'{split}.jsonl'))
        manifest_sha = raw_sha(common/f'{split}.jsonl')
        for start in track(range(0, len(items), 128), f'stream inputs: index {split} WAV shards'):
            selected = items[start:start+128]
            archive = audio_archive_path(common.parent/'.drive-audio-packs', manifest_sha, split, 'ja', start, 128)
            job_id = digest(['audio', str(archive), manifest_sha])[:24]
            directory = local/'audio'/job_id
            pairs = {str(r['pair_id']): checksum(r['ja_sha256'].lower()) for r in selected}
            if len(pairs) != len(selected):
                raise ValueError('duplicate streaming audio IDs')
            members = {digest([r['pair_id'], 'ja'])[:32]+'.wav': pairs[str(r['pair_id'])] for r in selected}
            jobs[job_id] = dict(kind='audio', source=str(archive), target=str(directory),
                                pairs=pairs, members=members)
            for raw in selected:
                member = digest([raw['pair_id'], 'ja'])[:32]+'.wav'
                aliases = [Path(raw['ja_audio'])]
                relative = raw.get('ja_audio_corpus_relative')
                if relative and corpus:
                    if Path(relative).is_absolute() or '..' in Path(relative).parts:
                        raise ValueError('unsafe streaming corpus-relative alias')
                    aliases.append(Path(corpus).absolute()/relative)
                for alias in aliases:
                    if not alias.is_absolute():
                        if '..' in alias.parts:
                            raise ValueError('unsafe streaming relative audio path')
                        if not corpus:
                            raise ValueError('CORPUS_ROOT needed for relative streaming audio')
                        alias = Path(corpus).absolute()/alias
                    key = logical(alias)
                    entry = dict(local=str(directory/member), job=job_id, member=member)
                    if key in rows and rows[key] != entry:
                        raise ValueError('ambiguous streaming audio path')
                    rows[key] = entry
    body = dict(root=str(local), jobs=jobs)
    print(f'[stream-inputs] planned_audio_and_mel_shards={len(jobs)} '
          'payloads_downloaded=0 sample_order=unchanged', file=sys.stderr, flush=True)
    return rows, local, dict(body=body, sha256=digest(body))


class Downloader:
    """Demand has priority over queued prefetch; at most two live transfers.

    File locks coordinate independent trial/train processes. Ready markers are
    local and atomic; partial downloads never become training inputs.
    """
    def __init__(self, document):
        from .drive_staging import ensure_local
        body = document['body']
        if document['sha256'] != digest(body):
            raise ValueError('corrupt streaming job index')
        self.root = ensure_local(body['root'])
        self.jobs = body['jobs']
        for key, job in self.jobs.items():
            if len(key) != 24 or any(c not in '0123456789abcdef' for c in key):
                raise ValueError('unsafe streaming job ID')
            target = Path(job['target'])
            if not target.is_absolute() or not target.is_relative_to(self.root) or '..' in target.parts:
                raise ValueError('streaming target escapes the owned local cache')
            ensure_local(target.parent if job['kind'] == 'mel' else target)
            names = [target.name] if job['kind'] == 'mel' else list(job['members'])
            if any(Path(name).name != name or '/' in name or '\\' in name for name in names):
                raise ValueError('unsafe streaming member')
        self.pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix='s2st-download')
        self.transfers = threading.Condition()
        self.active_transfers = self.waiting_demands = 0
        self.lock = threading.RLock()
        self.ready, self.pending, self.errors = {}, {}, {}
        self.job_locks = {key: threading.Lock() for key in self.jobs}
        self.completed = 0
        atexit.register(self.close)

    def close(self):
        self.pool.shutdown(wait=False, cancel_futures=True)
        atexit.unregister(self.close)

    @contextmanager
    def transfer(self, background):
        with self.transfers:
            if not background:
                self.waiting_demands += 1
            try:
                while self.active_transfers >= 2 or (background and self.waiting_demands):
                    self.transfers.wait()
                self.active_transfers += 1
            finally:
                if not background:
                    self.waiting_demands -= 1
        try:
            yield
        finally:
            with self.transfers:
                self.active_transfers -= 1
                self.transfers.notify_all()

    def prefetch(self, keys):
        with self.lock:
            self.pending = {key: value for key, value in self.pending.items() if not value.done()}
            for key in keys:
                if key in self.ready or key in self.pending or key in self.errors:
                    continue
                if len(self.pending) >= 16:
                    break
                self.pending[key] = self.pool.submit(self.ensure, key, background=True)

    def ensure(self, key, *, background=False):
        if not background:
            # A demand must not sit behind up to sixteen unrelated transfers.
            with self.lock:
                queued = self.pending.get(key)
                if queued is not None:
                    queued.cancel()
        with self.job_locks[key]:
            if key in self.errors:
                raise self.errors[key]
            if key in self.ready:
                return self.ready[key]
            from filelock import FileLock
            directory = self.root/'jobs'/key
            from .drive_staging import ensure_local
            ensure_local(directory)
            started = time.monotonic()
            try:
                print(f'[stream-download] request={key} kind={self.jobs[key]["kind"]} '
                      f'background={background} verified_before_use=true', file=sys.stderr, flush=True)
                with FileLock(str(directory/'download.lock')), self.transfer(background):
                    result, reused = self.fetch(self.jobs[key], directory/'ready.json')
                with self.lock:
                    self.ready[key] = result
                    self.completed += 1
                print(f'[stream-download] ready={self.completed}/{len(self.jobs)} '
                      f'kind={self.jobs[key]["kind"]} reused={reused} '
                      f'elapsed={time.monotonic()-started:.1f}s', file=sys.stderr, flush=True)
                return result
            except Exception as error:
                self.errors[key] = error
                raise

    @operation('stream inputs: download/check one shard before use')
    def fetch(self, job, receipt):
        from .drive_staging import copy_verified, raw_sha, capacity, missing_audio_pack, ensure_local
        target_root = Path(job['target'])
        ensure_local(target_root.parent if job['kind'] == 'mel' else target_root)
        if receipt.is_symlink():
            raise ValueError('unsafe streaming completion receipt')
        identity = digest(job)
        if receipt.is_file():
            saved = json.loads(receipt.read_text(encoding='utf-8'))
            body = saved['body']
            if saved['sha256'] != digest(body) or body['identity'] != identity:
                raise ValueError('stale/corrupt streaming completion receipt')
            expected_names = {Path(job['target']).name} if job['kind'] == 'mel' else set(job['members'])
            if set(body['files']) != expected_names:
                raise ValueError('streaming completion members mismatch')
            for name, entry in body['files'].items():
                expected = job['sha256'] if job['kind'] == 'mel' else job['members'][name]
                if entry['sha256'] != expected:
                    raise ValueError('streaming completion checksum differs from the locked input')
                if Path(name).name != name or '\\' in name or '/' in name:
                    raise ValueError('unsafe streaming completion member')
                target = Path(job['target']) if job['kind'] == 'mel' else Path(job['target'])/name
                stat = target.stat()
                if target.is_symlink() or [stat.st_size, stat.st_mtime_ns] != entry['local_stamp'] or raw_sha(target) != entry['sha256']:
                    raise ValueError('verified streaming input changed')
            return body['files'], True
        source, target = Path(job['source']), Path(job['target'])
        if source.is_symlink() or target.is_symlink() or source.with_suffix('.json').is_symlink():
            raise ValueError('unsafe streaming payload path')
        files = {}
        if job['kind'] == 'mel':
            before = source.stat()
            if [before.st_size, before.st_mtime_ns] != [job['stamp']['size'], job['stamp']['mtime_ns']]:
                raise ValueError('locked Mel input changed before download')
            copy_verified(source, target, job['sha256'])
            stat = target.stat()
            files[target.name] = dict(local_stamp=[stat.st_size, stat.st_mtime_ns],
                                     sha256=job['sha256'], source_stamp=job['stamp'])
        elif job['kind'] == 'audio':
            if not source.is_file() or not source.with_suffix('.json').is_file():
                raise missing_audio_pack(source)
            document = json.loads(source.with_suffix('.json').read_text(encoding='utf-8'))
            body = document['body']
            if document['sha256'] != digest(body) or body['expected'] != job['pairs']:
                raise ValueError('audio receipt differs from the common manifest')
            if (set(body['entries']) != set(job['members']) or
                    any(entry['sha256'] != job['members'][name] for name, entry in body['entries'].items())):
                raise ValueError('audio receipt members differ from the common manifest')
            # No second full read of the mounted ZIP before copying it.
            capacity(self.root, source.stat().st_size + sum(e['stamp']['size'] for e in body['entries'].values()))
            target.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix='.incoming-', dir=target) as temporary:
                temporary = Path(temporary)
                archive = temporary/'audio.zip'
                copy_verified(source, archive, checksum(body['archive_sha256']))
                with zipfile.ZipFile(archive) as pack:
                    if set(pack.namelist()) != set(job['members']) or len(pack.namelist()) != len(job['members']):
                        raise ValueError('streaming audio ZIP member mismatch')
                    for name, expected in job['members'].items():
                        if Path(name).name != name or '\\' in name or '/' in name or not name.endswith('.wav'):
                            raise ValueError('unsafe streaming audio member')
                        pending = temporary/name
                        with pack.open(name) as src, pending.open('wb') as out:
                            shutil.copyfileobj(src, out)
                        if raw_sha(pending) != expected:
                            raise ValueError('streaming WAV checksum mismatch')
                        if pending.stat().st_size != body['entries'][name]['stamp']['size']:
                            raise ValueError('streaming WAV size mismatch')
                        os.replace(pending, target/name)
                        stat = (target/name).stat()
                        files[name] = dict(local_stamp=[stat.st_size, stat.st_mtime_ns], sha256=expected,
                                           source_stamp=body['entries'][name]['stamp'])
        else:
            raise ValueError('unknown streaming job kind')
        body = dict(identity=identity, files=files)
        atomic_write_json(receipt, dict(body=body, sha256=digest(body)), overwrite=True)
        return files, False


_instances = {}
_instances_lock = threading.Lock()


def close_current():
    key = (os.getpid(), os.environ.get('S2ST_INPUT_MAP'))
    with _instances_lock:
        manager = _instances.pop(key, None)
    if manager:
        manager.close()


def downloader():
    from .drive_staging import stream_document
    name, document = stream_document()
    if document is None:
        return None
    key = (os.getpid(), name)
    with _instances_lock:
        if key not in _instances:
            _instances[key] = Downloader(document)
        return _instances[key]


def materialize(path, entry):
    manager = downloader()
    if manager is None:
        raise ValueError('streaming row without a job index')
    job = manager.jobs[entry['job']]
    target = Path(job['target']) if job['kind'] == 'mel' else Path(job['target'])/entry['member']
    if Path(entry['local']) != target:
        raise ValueError('streaming row differs from its indexed target')
    saved = manager.ensure(entry['job'])[entry['member']]
    # Preserve the logical original/portable alias in file_stamp identities.
    return dict(entry, local_stamp=saved['local_stamp'],
                source_stamp=dict(saved['source_stamp'], path=str(Path(path).absolute())))


def declared_hash(path):
    """A locked identity, NOT proof that any not-yet-used payload is verified."""
    from .drive_staging import mapping, logical
    entry = mapping().get(logical(path))
    if entry is None or 'declared_sha256' not in entry:
        return None
    manager = downloader()
    job = manager.jobs[entry['job']]
    if job['kind'] != 'mel' or job['sha256'] != entry['declared_sha256']:
        raise ValueError('declared streaming hash differs from its locked job')
    return checksum(entry['declared_sha256'])


def prefetch_dataset(dataset, start, count):
    from .drive_staging import mapping, logical
    manager = downloader()
    if manager is None:
        return
    indexed = mapping()
    jobs = []
    seen = set()
    # Bound metadata scanning as well as queued transfers. Look ahead in the
    # EXISTING sampler order; downloaded availability never picks a sample.
    for step in range(min(count, len(dataset))):
        row = dataset.rows[(start+step) % len(dataset)]
        source = Path(row['src_audio'])
        if not source.is_absolute() and dataset.corpus_root is not None:
            source = dataset.corpus_root/source
        mel = dataset.root/row['tgt_audio'].rsplit(':', 2)[0]
        for path in (source, mel):
            entry = indexed.get(logical(path))
            if entry and entry.get('job') and entry['job'] not in seen:
                seen.add(entry['job'])
                if entry['job'] not in manager.ready:
                    jobs.append(entry['job'])
        if len(jobs) >= 16:
            break
    manager.prefetch(jobs)


@contextmanager
def streaming_map(config, work):
    from .drive_staging import active_map
    rows, local, document = plan(config, work)
    with active_map(rows, local, stream=document):
        try:
            yield
        finally:
            close_current()
