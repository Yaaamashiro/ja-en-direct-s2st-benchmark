from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any


class ExistingOutputError(FileExistsError):
    """Raised when a command would replace an existing output."""


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    from .drive_staging import local_path
    path = local_path(path)
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {error.msg}") from error
            if not isinstance(row, dict):
                raise ValueError(f"expected an object at {path}:{line_number}")
            yield row


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def atomic_write_text(
    path: Path, content: str, *, overwrite: bool = False, resume: bool = False
) -> bool:
    if path.exists():
        if resume and path.read_text(encoding="utf-8") == content:
            return False
        if not overwrite:
            raise ExistingOutputError(f"output already exists: {path}")
    _atomic_text(path, content)
    return True


def atomic_write_json(
    path: Path, value: Any, *, overwrite: bool = False, resume: bool = False
) -> bool:
    content = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    return atomic_write_text(path, content, overwrite=overwrite, resume=resume)


def atomic_write_jsonl(
    path: Path,
    rows: Iterable[dict[str, Any]],
    *,
    overwrite: bool = False,
    resume: bool = False,
) -> bool:
    content = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    )
    return atomic_write_text(path, content, overwrite=overwrite, resume=resume)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
