from __future__ import annotations

from collections.abc import Iterable

from .schema import CommonManifestRow, SPLITS


def group_by_original_split(
    rows: Iterable[CommonManifestRow],
) -> dict[str, list[CommonManifestRow]]:
    grouped = {split: [] for split in SPLITS}
    for row in rows:
        grouped[row.split].append(row)
    return grouped
