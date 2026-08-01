from __future__ import annotations

import wave
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from ..hashing import sha256_file
from ..io import atomic_write_json, atomic_write_text, read_jsonl


def _read_phonemes(root: Path, split: str) -> dict[str, str]:
    direct = root / f"{split}.tsv"
    paths = [direct] if direct.is_file() else sorted(root.glob(f"{split}.shard-*-of-*.tsv"))
    if not paths:
        raise FileNotFoundError(f"no phoneme TSV found for {split}")
    values: dict[str, str] = {}
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                pair_id, separator, sequence = line.rstrip("\n").partition("\t")
                if not separator or not pair_id or not sequence:
                    raise ValueError(f"invalid phoneme row at {path}:{line_number}")
                if pair_id in values:
                    raise ValueError(f"duplicate phoneme row: {pair_id}")
                values[pair_id] = sequence
    return values


def _frames(path: Path) -> int:
    with wave.open(str(path), "rb") as handle:
        return handle.getnframes()


def prepare_fairseq(
    common_root: Path,
    phoneme_root: Path,
    output_root: Path,
    *,
    mel_config: dict[str, Any],
    resume: bool = False,
    overwrite: bool = False,
) -> dict[str, Any]:
    phonemes = {split: _read_phonemes(phoneme_root, split) for split in ("train", "dev", "test")}
    train_vocabulary = Counter(
        token for sequence in phonemes["train"].values() for token in sequence.split()
    )
    if not train_vocabulary:
        raise ValueError("train phoneme vocabulary is empty")
    known = set(train_vocabulary)
    for split in ("dev", "test"):
        unknown = sorted(
            {token for sequence in phonemes[split].values() for token in sequence.split()} - known
        )
        if unknown:
            raise ValueError(f"unknown phonemes in {split}: {', '.join(unknown)}")

    split_counts: dict[str, int] = {}
    target_root = output_root / "target_phoneme"
    for split in ("train", "dev", "test"):
        fairseq_lines = ["id\taudio\tn_frames\ttgt_audio\ttgt_n_frames\ttgt_text\tspeaker\n"]
        target_lines: list[str] = []
        count = 0
        for row in read_jsonl(common_root / f"{split}.jsonl"):
            pair_id = str(row["pair_id"])
            sequence = phonemes[split].get(pair_id)
            if sequence is None:
                raise ValueError(f"missing phonemes for {pair_id}")
            source = Path(row["ja_audio"]).resolve()
            target = Path(row["en_audio"]).resolve()
            fairseq_lines.append(
                f"{pair_id}\t{source}\t{_frames(source)}\t{target}\t{_frames(target)}\t"
                f"{sequence}\tfixed\n"
            )
            target_lines.append(f"{pair_id}\t{sequence}\n")
            count += 1
        atomic_write_text(
            output_root / f"{split}.tsv", "".join(fairseq_lines), resume=resume, overwrite=overwrite
        )
        atomic_write_text(
            target_root / f"{split}.tsv", "".join(target_lines), resume=resume, overwrite=overwrite
        )
        split_counts[split] = count

    dictionary = "".join(
        f"{token} {count}\n" for token, count in sorted(train_vocabulary.items())
    )
    atomic_write_text(target_root / "dict.txt", dictionary, resume=resume, overwrite=overwrite)
    base_config = {"audio_root": "/", "sampling_rate": 16000, "mel": mel_config}
    multitask = {
        "target_phoneme": {
            "type": "translation",
            "decoder_type": "transformer",
            "is_first_pass_decoder": True,
            "data": "target_phoneme",
        }
    }
    atomic_write_text(
        output_root / "config.yaml",
        yaml.safe_dump(base_config, sort_keys=True),
        resume=resume,
        overwrite=overwrite,
    )
    atomic_write_text(
        output_root / "config_multitask.yaml",
        yaml.safe_dump(multitask, sort_keys=True),
        resume=resume,
        overwrite=overwrite,
    )
    lock = {
        "common_dataset_lock_sha256": sha256_file(common_root / "dataset-lock.json"),
        "mel": mel_config,
        "phoneme_vocabulary_source": "train",
        "splits": split_counts,
    }
    atomic_write_json(output_root / "data-lock.json", lock, resume=resume, overwrite=overwrite)
    return lock
