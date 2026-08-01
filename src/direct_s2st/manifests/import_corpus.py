from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..hashing import sha256_file
from ..io import atomic_write_json, atomic_write_jsonl, read_jsonl
from .schema import CommonManifestRow, SPLITS
from .split import group_by_original_split
from .validate import validate_rows


def import_corpus(
    accepted_manifest: Path,
    *,
    corpus_root: Path,
    output_root: Path,
    limit: int | None = None,
    resume: bool = False,
    overwrite: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    rows: list[CommonManifestRow] = []
    selected = {split: 0 for split in SPLITS}
    for raw in read_jsonl(accepted_manifest):
        row = CommonManifestRow.from_corpus_row(
            raw, corpus_root=corpus_root, manifest_parent=accepted_manifest.parent
        )
        if limit is not None and selected[row.split] >= limit:
            continue
        rows.append(row)
        selected[row.split] += 1
    summary = validate_rows(rows)
    lock = {
        "schema_version": 1,
        "source_manifest_modified_at": datetime.fromtimestamp(
            accepted_manifest.stat().st_mtime, timezone.utc
        ).isoformat(),
        "source_manifest": str(accepted_manifest.resolve()),
        "source_manifest_sha256": sha256_file(accepted_manifest),
        **summary,
    }
    if dry_run:
        return lock
    grouped = group_by_original_split(rows)
    output_root.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        atomic_write_jsonl(
            output_root / f"{split}.jsonl",
            (row.to_dict() for row in grouped[split]),
            resume=resume,
            overwrite=overwrite,
        )
    atomic_write_json(
        output_root / "dataset-lock.json", lock, resume=resume, overwrite=overwrite
    )
    return lock
