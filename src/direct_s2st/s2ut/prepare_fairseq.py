from __future__ import annotations

import wave
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from ..hashing import sha256_file
from ..io import atomic_write_json, atomic_write_text, read_jsonl
from .extract_units import load_unit_file


def _frames(path: Path) -> int:
    with wave.open(str(path), "rb") as handle:
        return handle.getnframes()


def prepare_fairseq(
    common_root: Path,
    units_root: Path,
    output_root: Path,
    *,
    clusters: int = 100,
    resume: bool = False,
    overwrite: bool = False,
) -> dict[str, Any]:
    unit_records: dict[str, dict[str, Any]] = {}
    manifests = sorted(units_root.glob("manifest.shard-*-of-*.jsonl"))
    if not manifests:
        raise FileNotFoundError(f"no unit shard manifests found below {units_root}")
    for manifest in manifests:
        for row in read_jsonl(manifest):
            pair_id = str(row["pair_id"])
            if pair_id in unit_records:
                raise ValueError(f"duplicate unit record: {pair_id}")
            unit_records[pair_id] = row

    frequencies: Counter[int] = Counter()
    split_counts: dict[str, int] = {}
    for split in ("train", "dev", "test"):
        lines = ["id\taudio\tn_frames\ttgt_text\n"]
        count = 0
        for row in read_jsonl(common_root / f"{split}.jsonl"):
            pair_id = str(row["pair_id"])
            unit_record = unit_records.get(pair_id)
            if unit_record is None:
                raise ValueError(f"missing units for {pair_id}")
            if unit_record.get("split") != split:
                raise ValueError(f"unit split mismatch for {pair_id}")
            units = load_unit_file(
                Path(unit_record["units_reduced_path"]), clusters=clusters
            )
            frequencies.update(units)
            lines.append(
                f"{pair_id}\t{Path(row['ja_audio']).resolve()}\t"
                f"{_frames(Path(row['ja_audio']))}\t{' '.join(map(str, units))}\n"
            )
            count += 1
        atomic_write_text(
            output_root / f"{split}.tsv",
            "".join(lines),
            resume=resume,
            overwrite=overwrite,
        )
        split_counts[split] = count

    dictionary = "".join(f"{unit} {frequencies[unit]}\n" for unit in range(clusters))
    atomic_write_text(
        output_root / "dict.txt", dictionary, resume=resume, overwrite=overwrite
    )
    fairseq_config = {
        "audio_root": "/",
        "sampling_rate": 16000,
        "vocab_filename": "dict.txt",
        "target_type": "units",
        "reduce_consecutive_units": True,
    }
    atomic_write_text(
        output_root / "config.yaml",
        yaml.safe_dump(fairseq_config, sort_keys=True),
        resume=resume,
        overwrite=overwrite,
    )
    lock = {
        "common_dataset_lock_sha256": sha256_file(common_root / "dataset-lock.json"),
        "unit_manifest_sha256": {str(path): sha256_file(path) for path in manifests},
        "clusters": clusters,
        "target": "reduced_units",
        "splits": split_counts,
    }
    atomic_write_json(
        output_root / "data-lock.json", lock, resume=resume, overwrite=overwrite
    )
    return lock
