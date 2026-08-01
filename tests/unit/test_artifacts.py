from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from direct_s2st.artifacts import download_artifact
from direct_s2st.io import ExistingOutputError


def test_artifact_download_is_checksum_locked_and_reusable(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"fixed artifact")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    destination = tmp_path / "cache" / "artifact.bin"

    first = download_artifact(source.as_uri(), destination, sha256=digest)
    second = download_artifact(source.as_uri(), destination, sha256=digest)

    assert first["reused"] is False
    assert second["reused"] is True
    assert destination.read_bytes() == b"fixed artifact"


def test_artifact_download_rejects_existing_mismatch(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"expected")
    destination = tmp_path / "artifact.bin"
    destination.write_bytes(b"different")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()

    with pytest.raises(ExistingOutputError, match="different checksum"):
        download_artifact(source.as_uri(), destination, sha256=digest)
