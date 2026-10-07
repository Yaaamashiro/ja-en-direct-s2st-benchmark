from __future__ import annotations

import wave
from ..progress import operation, track
from collections import Counter
from pathlib import Path
from typing import Any

from ..hashing import sha256_file
from .schema import CommonManifestRow, SPLITS
from .reader import read_common_manifest
from ..preparation import Checkpoints, adaptive_map, checkpoint_map, file_stamp
from ..journal import digest
import os


class ManifestValidationError(ValueError):
    pass


def inspect_wav(path: Path) -> dict[str, int | float]:
    from ..drive_staging import local_path
    path = local_path(path)
    if not path.is_file():
        raise ManifestValidationError(f"audio file does not exist: {path}")
    try:
        with wave.open(str(path), "rb") as handle:
            channels = handle.getnchannels()
            sample_width = handle.getsampwidth()
            sample_rate = handle.getframerate()
            frames = handle.getnframes()
    except (wave.Error, EOFError) as error:
        raise ManifestValidationError(f"invalid WAV file {path}: {error}") from error
    if channels != 1:
        raise ManifestValidationError(f"audio must be mono: {path}")
    if sample_width != 2:
        raise ManifestValidationError(f"audio must be PCM16: {path}")
    if sample_rate != 16000:
        raise ManifestValidationError(f"audio must be 16 kHz: {path}")
    return {
        "channels": channels,
        "sample_width": sample_width,
        "sample_rate": sample_rate,
        "frames": frames,
        "duration": frames / sample_rate,
    }


@operation('manifests/validate: validate_rows')
def validate_rows(
    rows: list[CommonManifestRow], *, duration_tolerance: float = 0.02,
    checkpoint_root: Path | None = None, resume: bool = False,
) -> dict[str, Any]:
    seen: dict[str, str] = {}
    counts: Counter[str] = Counter()
    ja_seconds: Counter[str] = Counter()
    en_seconds: Counter[str] = Counter()
    for row in rows:
        if row.pair_id in seen:
            raise ManifestValidationError(
                f"duplicate pair_id {row.pair_id!r} in {seen[row.pair_id]} and {row.split}"
            )
        seen[row.pair_id] = row.split
        counts[row.split] += 1
        ja_seconds[row.split] += row.ja_duration
        en_seconds[row.split] += row.en_duration

    def check(row):
        for language in ("ja", "en"):
            path = Path(getattr(row, f"{language}_audio"))
            metadata = inspect_wav(path)
            expected_hash = getattr(row, f"{language}_sha256")
            actual_hash = sha256_file(path)
            if actual_hash.lower() != expected_hash.lower():
                raise ManifestValidationError(f"SHA-256 mismatch for {path}")
            expected_duration = getattr(row, f"{language}_duration")
            if abs(float(metadata["duration"]) - expected_duration) > duration_tolerance:
                raise ManifestValidationError(f"duration mismatch for {path}")
        return {'verified': True}

    def key(row):
        return digest(dict(row=row.to_dict(), files=[file_stamp(getattr(row, f'{lang}_audio'))
                                                    for lang in ('ja', 'en')]))

    phase = 'corpus: validate WAV and SHA256'
    if checkpoint_root is None:
        for _ in track(adaptive_map(check, rows), phase, total=len(rows)):
            pass
    else:
        reuse = resume and os.environ.get('S2ST_PREP_RECHECK', '0') != '1'
        with Checkpoints(checkpoint_root, dict(stage='wav-sha256-v1', tolerance=duration_tolerance),
                         resume=reuse) as cache:
            for _ in checkpoint_map(check, rows, cache, key, phase, total=len(rows)):
                pass
    return {
        "total_pairs": len(rows),
        "splits": {
            split: {
                "pairs": counts[split],
                "ja_hours": ja_seconds[split] / 3600,
                "en_hours": en_seconds[split] / 3600,
            }
            for split in SPLITS
        },
    }


@operation('manifests/validate: validate_manifest_directory')
def validate_manifest_directory(root: Path, *, resume: bool = False) -> dict[str, Any]:
    rows: list[CommonManifestRow] = []
    for split in SPLITS:
        path = root / f"{split}.jsonl"
        if not path.is_file():
            raise ManifestValidationError(f"missing split manifest: {path}")
        for raw in track(read_common_manifest(path, parallel=True), f'corpus: read {split}'):
            row = CommonManifestRow.from_dict(raw)
            if row.split != split:
                raise ManifestValidationError(
                    f"row {row.pair_id!r} has split={row.split!r} in {path.name}"
                )
            rows.append(row)
    return validate_rows(rows, checkpoint_root=root.parent / '.prep-checkpoints/corpus', resume=resume)
