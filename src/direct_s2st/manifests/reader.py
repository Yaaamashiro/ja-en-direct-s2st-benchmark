"""Portable common manifest reader; resolved absolute paths are runtime caches."""
import os
from pathlib import Path

from ..io import read_jsonl
from .paths import resolve_audio_path


def read_common_manifest(path: Path, *, corpus_root: Path | None = None):
    configured = corpus_root or os.environ.get("CORPUS_ROOT")
    for row in read_jsonl(path):
        if configured:
            for language in ("ja", "en"):
                raw = row.get(f"{language}_audio_corpus_relative") or row[f"{language}_audio"]
                row[f"{language}_audio"] = str(resolve_audio_path(raw, Path(configured)))
        yield row


def portable_row(row, corpus_root: Path) -> dict:
    value = row.to_dict()
    root = corpus_root.resolve()
    for language in ("ja", "en"):
        audio = Path(value[f"{language}_audio"]).resolve()
        if audio.is_relative_to(root):
            value[f"{language}_audio_corpus_relative"] = audio.relative_to(root).as_posix()
    return value
