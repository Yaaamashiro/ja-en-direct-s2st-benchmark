from __future__ import annotations

import hashlib
from pathlib import Path
from .progress import activity


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    from .drive_staging import local_path
    path = local_path(path)
    digest = hashlib.sha256()
    with activity(f'SHA256: {path.name}') as progress, path.open("rb") as handle:
        read_bytes = 0
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
            read_bytes += len(chunk)
            if progress:
                progress.activity = f'SHA256: {path.name} bytes={read_bytes}'
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
