from __future__ import annotations

import wave
from collections import Counter
from pathlib import Path
from typing import Any

from ..hashing import sha256_file
from ..io import read_jsonl
from .schema import CommonManifestRow, SPLITS


class ManifestValidationError(ValueError):
    pass


def inspect_wav(path: Path) -> dict[str, int | float]:
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


def validate_rows(
    rows: list[CommonManifestRow], *, duration_tolerance: float = 0.02
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
        ja_seconds[row.split] += row.ja_duration
        en_seconds[row.split] += row.en_duration
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


def validate_manifest_directory(root: Path) -> dict[str, Any]:
    rows: list[CommonManifestRow] = []
    for split in SPLITS:
        path = root / f"{split}.jsonl"
        if not path.is_file():
            raise ManifestValidationError(f"missing split manifest: {path}")
        for raw in read_jsonl(path):
            row = CommonManifestRow.from_dict(raw)
            if row.split != split:
                raise ManifestValidationError(
                    f"row {row.pair_id!r} has split={row.split!r} in {path.name}"
                )
            rows.append(row)
    return validate_rows(rows)
