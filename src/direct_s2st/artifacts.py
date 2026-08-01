from __future__ import annotations

import hashlib
import os
import urllib.request
import uuid
from pathlib import Path
from urllib.parse import urlparse

from .hashing import sha256_file
from .io import ExistingOutputError


def download_artifact(
    url: str,
    destination: Path,
    *,
    sha256: str,
    overwrite: bool = False,
) -> dict[str, str | bool]:
    parsed = urlparse(url)
    if parsed.scheme not in {"https", "file"}:
        raise ValueError("artifact URL must use https or file")
    expected = sha256.lower()
    if len(expected) != 64 or any(char not in "0123456789abcdef" for char in expected):
        raise ValueError("artifact sha256 must be a 64-character hexadecimal digest")
    if destination.is_file():
        actual = sha256_file(destination)
        if actual == expected:
            return {"path": str(destination), "sha256": actual, "reused": True}
        if not overwrite:
            raise ExistingOutputError(f"artifact exists with a different checksum: {destination}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    digest = hashlib.sha256()
    try:
        with urllib.request.urlopen(url) as response, temporary.open("xb") as handle:
            while chunk := response.read(1024 * 1024):
                handle.write(chunk)
                digest.update(chunk)
            handle.flush()
            os.fsync(handle.fileno())
        actual = digest.hexdigest()
        if actual != expected:
            raise ValueError(
                f"artifact checksum mismatch for {url}: expected {expected}, got {actual}"
            )
        if destination.exists() and not overwrite:
            raise ExistingOutputError(f"artifact appeared during download: {destination}")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return {"path": str(destination), "sha256": expected, "reused": False}
