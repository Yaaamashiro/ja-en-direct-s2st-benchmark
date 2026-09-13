from __future__ import annotations

import wave
import json
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from ..hashing import sha256_file
from ..io import atomic_write_json, atomic_write_text, read_jsonl
from .extract_units import load_unit_file
from .multitask import prepare_labels, validate_prepared
from ..manifests.reader import read_common_manifest


def _ten_ms_frames(path: Path) -> int:
    with wave.open(str(path), "rb") as handle:
        if handle.getframerate() != 16000:
            raise ValueError(f"expected 16 kHz source audio: {path}")
        return handle.getnframes() // 160


def prepare_fairseq(
    common_root: Path,
    units_root: Path,
    output_root: Path,
    *,
    clusters: int = 100,
    multitask: dict[str, Any] | None = None,
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

    rows_by_split = {split: list(read_common_manifest(common_root / f"{split}.jsonl")) for split in ("train", "dev", "test")}
    seen = set()
    for split, rows in rows_by_split.items():
        for row in rows:
            pair_id = row["pair_id"]
            if not isinstance(pair_id, str) or not pair_id or any(c in pair_id for c in "\t\r\n/\\") or pair_id in seen:
                raise ValueError(f"invalid or duplicate pair_id: {pair_id}")
            seen.add(pair_id)
            if row["split"] != split:
                raise ValueError(f"common split mismatch: {pair_id}")
    linguistic = prepare_labels(rows_by_split, output_root, settings=multitask, resume=resume, overwrite=overwrite)
    frequencies: Counter[int] = Counter()
    split_counts: dict[str, int] = {}
    for split in ("train", "dev", "test"):
        lines = ["id\tsrc_audio\tsrc_n_frames\ttgt_audio\ttgt_n_frames\n"]
        count = 0
        for row in rows_by_split[split]:
            pair_id = str(row["pair_id"])
            unit_record = unit_records.get(pair_id)
            if unit_record is None:
                raise ValueError(f"missing units for {pair_id}")
            if unit_record.get("split") != split:
                raise ValueError(f"unit split mismatch for {pair_id}")
            units = load_unit_file(
                Path(unit_record["units_reduced_path"]), clusters=clusters
            )
            if split == "train":
                frequencies.update(units)
            lines.append(
                f"{pair_id}\t{Path(row['ja_audio']).resolve()}\t"
                f"{_ten_ms_frames(Path(row['ja_audio']))}\t"
                f"{' '.join(map(str, units))}\t{len(units)}\n"
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
        "input_channels": 1,
        "input_feat_per_channel": 80,
        "specaugment": {
            "time_wrap_W": 0,
            "freq_mask_N": 1,
            "freq_mask_F": 27,
            "time_mask_N": 1,
            "time_mask_T": 100,
            "time_mask_p": 1.0,
        },
        "transforms": {
            "*": ["utterance_cmvn"],
            "_train": ["utterance_cmvn", "specaugment"],
        },
    }
    atomic_write_text(
        output_root / "config.yaml",
        yaml.safe_dump(fairseq_config, sort_keys=True),
        resume=resume,
        overwrite=overwrite,
    )
    lock = {
        "unit_configuration": json.loads((units_root / "unit-lock.json").read_text(encoding="utf-8")),
        "common_dataset_lock_sha256": sha256_file(common_root / "dataset-lock.json"),
        "unit_manifest_sha256": {str(path): sha256_file(path) for path in manifests},
        "clusters": clusters,
        "target": "reduced_units",
        "linguistic": linguistic,
        "splits": split_counts,
    }
    atomic_write_json(
        output_root / "data-lock.json", lock, resume=resume, overwrite=overwrite
    )
    validate_prepared(output_root, clusters=clusters)
    return lock
