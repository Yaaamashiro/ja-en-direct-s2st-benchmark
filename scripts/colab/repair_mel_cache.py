"""Copy legacy Mel files to distributed storage, without re-extracting anything.

Run with Colab's host Python after auth.authenticate_user() for API downloads.
The Drive API is used exclusively for listing and reading, never remote writes.
Destination copies use the mounted experiment path and remain resumable.
"""
import argparse
import json
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from direct_s2st.io import atomic_write_json
from direct_s2st.journal import digest
from direct_s2st.progress import Progress, activity, operation, track
from direct_s2st.translatotron2.mel_storage import (
    checked_row, copy_feature, exists_checked, guarded_root, load_rows, verify_feature,
)


class DriveReader:
    """Read-only API adapter, serial and bounded (no shared HTTP client threads)."""

    def __init__(self, service, downloader):
        self.service, self.downloader = service, downloader

    @operation('mel recovery: list Drive source folder (read-only)')
    def index(self, folder_id, cache_root, expected):
        if not re.fullmatch(r'[A-Za-z0-9_-]+', folder_id):
            raise ValueError('supply the folder ID, not a URL')
        folder = self.service.files().get(
            fileId=folder_id, fields='id,name,mimeType,parents,trashed',
            supportsAllDrives=True).execute(num_retries=2)
        if (folder.get('name') != 'features' or folder.get('trashed')
                or folder.get('mimeType') != 'application/vnd.google-apps.folder'):
            raise ValueError('Drive source must be the original features folder')
        parents = folder.get('parents', [])
        if len(parents) != 1:
            raise ValueError('Drive source must have one checkpoint parent')
        parent = self.service.files().get(
            fileId=parents[0], fields='name', supportsAllDrives=True).execute(num_retries=2)
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
                page = self.service.files().list(
                    q=f"'{folder_id}' in parents and trashed=false", spaces='drive',
                    pageSize=1000, pageToken=token, supportsAllDrives=True,
                    includeItemsFromAllDrives=True,
                    fields='nextPageToken,incompleteSearch,files(id,name,mimeType,capabilities/canDownload)'
                ).execute(num_retries=2)
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
                    _, done = download.next_chunk(num_retries=2)


def authenticated_reader():
    try:
        import google.auth
        import httplib2
        from google_auth_httplib2 import AuthorizedHttp
        from googleapiclient.discovery import build
        from googleapiclient.http import MediaIoBaseDownload
        credentials, _ = google.auth.default(scopes=['https://www.googleapis.com/auth/drive.readonly'])
        transport = AuthorizedHttp(credentials, http=httplib2.Http(timeout=60))
        return DriveReader(build('drive', 'v3', http=transport, cache_discovery=False),
                           MediaIoBaseDownload)
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache-root', required=True, type=Path)
    parser.add_argument('--drive-folder-id')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--refresh-index', action='store_true')
    args = parser.parse_args()
    reader = authenticated_reader() if args.drive_folder_id else None
    result = repair(args.cache_root, reader=reader, folder_id=args.drive_folder_id,
                    limit=args.limit, refresh_index=args.refresh_index)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
