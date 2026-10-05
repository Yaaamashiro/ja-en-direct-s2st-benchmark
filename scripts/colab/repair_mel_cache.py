"""Rescue legacy Mel files into bounded ZIP packs without re-extraction.

Run with Colab's host Python after auth.authenticate_user() for API downloads.
The Drive API is used exclusively for listing and reading, never remote writes.
Temporary NPY/ZIP work stays local; verified ZIP checkpoints alone go to Drive.
--source-zip avoids per-file API requests; --storage files is legacy compatibility.
"""
import argparse
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import sys
import tempfile
import time
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from direct_s2st.io import atomic_write_json
from direct_s2st.journal import digest
from direct_s2st.progress import Progress, activity, operation, track
from direct_s2st.translatotron2.mel_storage import (
    _checked_row, checked_row, copy_feature, exists_checked, guarded_root, load_rows, verify_feature,
)


class DriveReader:
    """Read-only API adapter, serial and bounded (no shared HTTP client threads)."""

    def __init__(self, service, downloader, *, interval=1.0, max_retries=8):
        if not math.isfinite(interval) or interval < 0 or max_retries < 0:
            raise ValueError('invalid API pacing/retry limits')
        self.service, self.downloader = service, downloader
        self.interval, self.max_retries, self.last_request = interval, max_retries, None

    def call(self, function):
        """Pace ALL requests; retry only transient rate limits, not permissions."""
        for attempt in range(self.max_retries + 1):
            if self.last_request is not None:
                wait = self.interval - (time.monotonic() - self.last_request)
                if wait > 0:
                    # Bound each sleep, including user-supplied slow intervals.
                    while wait > 0:
                        pause = min(wait, 30)
                        time.sleep(pause)
                        wait -= pause
            self.last_request = time.monotonic()
            try:
                return function()
            except Exception as error:
                status = getattr(getattr(error, 'resp', None), 'status', None)
                try:
                    reasons = {e.get('reason') for e in json.loads(error.content)['error'].get('errors', [])}
                except (AttributeError, KeyError, TypeError, ValueError):
                    reasons = set()
                transient = status in (429, 500, 502, 503, 504) or (
                    status == 403 and bool(reasons & {'rateLimitExceeded', 'userRateLimitExceeded'}))
                if not transient or attempt == self.max_retries:
                    raise
                self.interval = min(30, max(2, self.interval * 2))
                pause = min(60, 2 ** attempt + random.random())
                print(f'[drive-rate] status={status} retry={attempt+1}/{self.max_retries} '
                      f'wait={pause:.1f}s request_interval={self.interval:.1f}s; durable packs preserved',
                      file=sys.stderr, flush=True)
                time.sleep(pause)

    def execute(self, request):
        return self.call(lambda: request.execute(num_retries=0))

    @operation('mel recovery: list Drive source folder (read-only)')
    def index(self, folder_id, cache_root, expected):
        if not re.fullmatch(r'[A-Za-z0-9_-]+', folder_id):
            raise ValueError('supply the folder ID, not a URL')
        folder = self.execute(self.service.files().get(
            fileId=folder_id, fields='id,name,mimeType,parents,trashed',
            supportsAllDrives=True))
        if (folder.get('name') != 'features' or folder.get('trashed')
                or folder.get('mimeType') != 'application/vnd.google-apps.folder'):
            raise ValueError('Drive source must be the original features folder')
        parents = folder.get('parents', [])
        if len(parents) != 1:
            raise ValueError('Drive source must have one checkpoint parent')
        parent = self.execute(self.service.files().get(
            fileId=parents[0], fields='name', supportsAllDrives=True))
        if parent.get('name') != cache_root.name:
            raise ValueError('Drive folder belongs to another Mel cache/generation')
        # Cache a complete authenticated filename->ID index, not credentials.
        index_path = cache_root / f'drive-source-{folder_id}.json'
        identity = dict(stage='mel-drive-index-v1', folder=folder_id,
                        expected=digest(sorted(expected)))
        if index_path.is_symlink():
            raise ValueError('Drive indexes must not be symlinks')
        if exists_checked(index_path):
            document = json.loads(index_path.read_text(encoding='utf-8'))
            if (document.get('identity') != identity or
                    document.get('sha256') != digest(document['files'])):
                raise ValueError('stale/corrupt Drive source index; keep it and use --refresh-index')
            return document['files']
        result, token, seen_tokens = {}, None, set()
        with Progress('mel recovery: Drive metadata pages', counted=True) as progress:
            while True:
                page = self.execute(self.service.files().list(
                    q=f"'{folder_id}' in parents and trashed=false", spaces='drive',
                    pageSize=1000, pageToken=token, supportsAllDrives=True,
                    includeItemsFromAllDrives=True,
                    fields='nextPageToken,incompleteSearch,files(id,name,mimeType,capabilities/canDownload)'
                ))
                if page.get('incompleteSearch'):
                    raise ValueError('incomplete Drive search; cannot trust missing-file results')
                for entry in page.get('files', []):
                    name = entry['name']
                    if name not in expected:
                        continue
                    if name in result:
                        raise ValueError(f'ambiguous duplicate Drive filename: {name}')
                    if (entry.get('mimeType', '').startswith('application/vnd.google-apps.')
                            or not entry.get('capabilities', {}).get('canDownload')):
                        raise ValueError(f'Drive file cannot be downloaded: {name}')
                    result[name] = entry['id']
                progress.count = len(result)
                progress.advanced = time.monotonic()
                progress.current = 'matched filenames'
                token = page.get('nextPageToken')
                if not token:
                    break
                if token in seen_tokens:
                    raise ValueError('Drive listing repeated a page token')
                seen_tokens.add(token)
        missing = expected - result.keys()
        if missing:
            raise ValueError(f'{len(missing)} saved Mel files missing from Drive source; '
                             f'first={sorted(missing)[0]}; nothing will be regenerated')
        atomic_write_json(index_path, dict(identity=identity, files=result, sha256=digest(result)))
        return result

    def download(self, file_id, temporary):
        request = self.service.files().get_media(fileId=file_id, supportsAllDrives=True)
        with temporary.open('wb') as handle:
            download = self.downloader(handle, request, chunksize=4 * 1024 * 1024)
            done = False
            while not done:
                with activity('Drive API: download Mel copy'):
                    _, done = self.call(lambda: download.next_chunk(num_retries=0))


def authenticated_reader(*, interval=1.0):
    try:
        import google.auth
        import httplib2
        from google_auth_httplib2 import AuthorizedHttp
        from googleapiclient.discovery import build
        from googleapiclient.http import MediaIoBaseDownload
        credentials, _ = google.auth.default(scopes=['https://www.googleapis.com/auth/drive.readonly'])
        transport = AuthorizedHttp(credentials, http=httplib2.Http(timeout=60))
        return DriveReader(build('drive', 'v3', http=transport, cache_discovery=False),
                           MediaIoBaseDownload, interval=interval)
    except ImportError as error:
        raise RuntimeError('use Colab host sys.executable (Google API libraries required)') from error


@operation('mel recovery: verified distributed copies')
def repair(cache_root, *, reader=None, folder_id=None, limit=None, refresh_index=False):
    if limit is not None and limit <= 0:
        raise ValueError('limit must be positive')
    cache_root = guarded_root(cache_root)
    rows = load_rows(cache_root)
    # No old features directory enumeration/stat/read in API mode.
    legacy = {checked_row(cache_root, row)[0].name for row in rows.values()
              if Path(row['path']).parent.name == 'features'}
    if reader is not None and not folder_id:
        raise ValueError('Drive API mode requires the exact legacy features folder ID')
    if refresh_index:
        if not folder_id or not re.fullmatch(r'[A-Za-z0-9_-]+', folder_id):
            raise ValueError('refresh-index requires a valid Drive folder ID')
        saved = cache_root / f'drive-source-{folder_id}.json'
        if saved.is_symlink():
            raise ValueError('Drive indexes must not be symlinks')
        if exists_checked(saved):
            # Preserve old indexes just like old features. No --overwrite needed.
            saved.rename(saved.with_name(f'{saved.stem}.before-{time.time_ns()}.json'))
    index = reader.index(folder_id, cache_root, legacy) if reader and legacy else {}
    copied = reused = 0
    last_log = time.monotonic()
    selected = list(sorted(rows.items()))
    if limit is not None:
        selected = selected[:limit]
    for row in track([value for _, value in selected], 'mel recovery: copy/reuse', total=len(selected)):
        source, target = checked_row(cache_root, row)
        if exists_checked(target):
            verify_feature(target, row['sha256'])
            reused += 1
        else:
            writer = None
            if reader and source.parent.name == 'features':
                writer = lambda temp, name=source.name: reader.download(index[name], temp)
            _, published = copy_feature(cache_root, row, writer)
            copied += published
            reused += not published
        if time.monotonic() - last_log >= 10:
            print(f'[mel-recovery] copied={copied} reused={reused} '
                  f'remaining={len(selected)-copied-reused}', file=sys.stderr, flush=True)
            last_log = time.monotonic()
    return dict(saved=len(rows), checked=len(selected), copied=copied, reused=reused,
                remaining=len(rows)-len(selected), originals_preserved=True)


class ZipSources:
    """Read supplied folder-download ZIPs; never extract paths from an archive."""
    def __init__(self, paths, expected):
        self.handles, self.entries = [], {}
        try:
            for path in paths:
                archive = zipfile.ZipFile(path)
                self.handles.append(archive)
                for info in track(archive.infolist(), 'mel recovery: index supplied ZIP'):
                    parts = info.filename.replace('\\', '/').split('/')
                    if '..' in parts or info.filename.startswith(('/', '\\')) or ':' in parts[0]:
                        raise ValueError('unsafe source ZIP member path')
                    name = parts[-1]
                    if info.is_dir() or name not in expected:
                        continue
                    if (name in self.entries or info.file_size > 256 * 1024**2
                            or info.flag_bits & 1 or (info.external_attr >> 16) & 0o170000 == 0o120000):
                        raise ValueError('duplicate/oversized/encrypted/symlink Mel ZIP member')
                    self.entries[name] = archive, info
        except BaseException:
            self.close()
            raise

    def download(self, name, destination):
        archive, info = self.entries[name]
        with archive.open(info) as source, destination.open('wb') as target:
            shutil.copyfileobj(source, target, length=1024 * 1024)

    def close(self):
        for archive in self.handles:
            archive.close()


@operation('mel recovery: local work and durable ZIP packs')
def repair_packed(cache_root, *, reader=None, folder_id=None, limit=None,
                  source_zips=(), local_parent=None, refresh_index=False, max_files=128):
    from direct_s2st.translatotron2.mel_packs import MelPacks
    if limit is not None and limit <= 0:
        raise ValueError('limit must be positive')
    root = guarded_root(cache_root)
    rows = load_rows(root)
    expected = {Path(row['path']).name for row in rows.values()}
    sources = ZipSources(source_zips, expected)
    copied = reused = 0
    try:
        with MelPacks(root, local_parent=local_parent, max_files=max_files) as store:
            # Packed lookup is in memory. Do not stat 140k old files on restart.
            selected = sorted(rows.items())
            if limit is not None:
                selected = selected[:limit]
            index = None
            reported = time.monotonic()
            for key, row in track(selected, 'mel recovery: pack/reuse', total=len(selected)):
                if key in store.rows:
                    if store.rows[key][0]['row'] != row:
                        raise ValueError('packed Mel differs from immutable checkpoint')
                    reused += 1
                    continue
                source, distributed = _checked_row(root, row)
                fd, name = tempfile.mkstemp(dir=store.local, suffix='.npy')
                os.close(fd)
                local = Path(name)
                if source.name in sources.entries:
                    sources.download(source.name, local)
                elif exists_checked(distributed):
                    if any(p.is_symlink() for p in (distributed, distributed.parent, distributed.parent.parent)):
                        raise ValueError('distributed Mel source must not be a symlink')
                    shutil.copyfile(distributed, local)
                elif reader and source.parent.name == 'features':
                    if not folder_id:
                        raise ValueError('Drive API mode requires the exact legacy folder ID')
                    if index is None:
                        if refresh_index:
                            if not re.fullmatch('[A-Za-z0-9_-]+', folder_id):
                                raise ValueError('invalid Drive folder ID')
                            saved = root / f'drive-source-{folder_id}.json'
                            if saved.is_symlink():
                                raise ValueError('Drive index must not be a symlink')
                            if exists_checked(saved):
                                saved.rename(saved.with_name(f'{saved.stem}.before-{time.time_ns()}.json'))
                        legacy = {Path(r['path']).name for r in rows.values() if Path(r['path']).parent.name == 'features'}
                        index = reader.index(folder_id, root, legacy)
                    reader.download(index[source.name], local)
                else:
                    if source.is_symlink() or source.parent.is_symlink():
                        raise ValueError('legacy Mel source must not be a symlink')
                    shutil.copyfile(source, local)
                store.add(key, row, local)
                copied += 1
                if time.monotonic() - reported >= 10:
                    print(f'[mel-recovery] packed={copied} reused={reused} persisted={len(store.rows)} '
                          f'pending_local={len(store.pending)} remaining={len(selected)-copied-reused}',
                          file=sys.stderr, flush=True)
                    reported = time.monotonic()
            store.flush()
            covered = len(rows.keys() & store.rows.keys())
            return dict(saved=len(rows), checked=len(selected), copied=copied, reused=reused,
                        persisted=covered, remaining=len(rows)-covered,
                        storage='zip-packs-v1', originals_preserved=True)
    finally:
        sources.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache-root', required=True, type=Path)
    parser.add_argument('--drive-folder-id')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--refresh-index', action='store_true')
    parser.add_argument('--storage', choices=('packed', 'files'), default='packed')
    parser.add_argument('--source-zip', type=Path, action='append', default=[])
    parser.add_argument('--local-work-root', type=Path)
    parser.add_argument('--api-interval', type=float, default=1.0)
    args = parser.parse_args()
    if args.storage == 'files' and (args.source_zip or args.local_work_root):
        parser.error('source-zip/local-work-root require packed storage')
    reader = authenticated_reader(interval=args.api_interval) if args.drive_folder_id else None
    options = dict(reader=reader, folder_id=args.drive_folder_id, limit=args.limit,
                   refresh_index=args.refresh_index)
    result = (repair_packed(args.cache_root, source_zips=args.source_zip,
                           local_parent=args.local_work_root, **options)
              if args.storage == 'packed' else repair(args.cache_root, **options))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
