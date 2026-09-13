from __future__ import annotations

import os
import tempfile
import wave
import zipfile
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from ..manifests.reader import read_common_manifest
from typing import Any

import yaml

from ..hashing import sha256_file
from ..io import ExistingOutputError, atomic_write_json, atomic_write_text, read_jsonl

FeatureExtractor = Callable[[Path, Path, dict[str, Any]], None]


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


def _read_inventory(root: Path) -> tuple[str, ...]:
    path = root / "inventory.txt"
    if not path.is_file():
        raise FileNotFoundError(f"phoneme inventory not found: {path}")
    tokens = tuple(line.strip() for line in path.read_text(encoding="utf-8").splitlines())
    if not tokens or any(not token or any(char.isspace() for char in token) for token in tokens):
        raise ValueError(f"invalid phoneme inventory: {path}")
    if len(tokens) != len(set(tokens)):
        raise ValueError(f"duplicate token in phoneme inventory: {path}")
    return tokens


def _ten_ms_frames(path: Path) -> int:
    with wave.open(str(path), "rb") as handle:
        if handle.getframerate() != 16000:
            raise ValueError(f"expected 16 kHz source audio: {path}")
        return handle.getnframes() // 160


def _mel_settings(values: dict[str, Any]) -> dict[str, Any]:
    settings = {
        "sample_rate": int(values.get("sample_rate", 16000)),
        "win_length": int(values.get("win_length", 1024)),
        "hop_length": int(values.get("hop_length", 256)),
        "n_fft": int(values.get("n_fft", 1024)),
        "n_mels": int(values.get("n_mels", 80)),
        "f_min": float(values.get("f_min", 20)),
        "f_max": float(values.get("f_max", 8000)),
        "eps": float(values.get("eps", 1e-5)),
        "normalize_volume": bool(values.get("normalize_volume", False)),
    }
    for key in ("sample_rate", "win_length", "hop_length", "n_fft", "n_mels"):
        if settings[key] <= 0:
            raise ValueError(f"mel.{key} must be positive")
    if settings["win_length"] > settings["n_fft"]:
        raise ValueError("mel.win_length must not exceed mel.n_fft")
    if not 0 <= settings["f_min"] < settings["f_max"] <= settings["sample_rate"] / 2:
        raise ValueError("mel frequency range must fit within the Nyquist frequency")
    return settings


def _extract_logmel_official(
    audio_path: Path, output_path: Path, settings: dict[str, Any]
) -> None:
    try:
        import torchaudio
        from examples.speech_synthesis.data_utils import extract_logmel_spectrogram
        from fairseq.data.audio.audio_utils import convert_waveform
    except ImportError as error:
        raise RuntimeError(
            "Translatotron 2 preparation requires the pinned fairseq Docker image"
        ) from error

    waveform, sample_rate = torchaudio.load(audio_path.as_posix())
    waveform, sample_rate = convert_waveform(
        waveform,
        sample_rate,
        normalize_volume=settings["normalize_volume"],
        to_sample_rate=settings["sample_rate"],
    )
    extract_logmel_spectrogram(
        waveform,
        sample_rate,
        output_path,
        win_length=settings["win_length"],
        hop_length=settings["hop_length"],
        n_fft=settings["n_fft"],
        n_mels=settings["n_mels"],
        f_min=settings["f_min"],
        f_max=settings["f_max"],
        eps=settings["eps"],
    )


def _write_feature_zip(feature_root: Path, zip_path: Path) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=zip_path.parent, prefix=f".{zip_path.name}.", suffix=".tmp"
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with zipfile.ZipFile(temporary, "w", zipfile.ZIP_STORED) as archive:
            for path in sorted(feature_root.glob("*.npy")):
                archive.write(path, arcname=path.name)
        os.replace(temporary, zip_path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _zip_manifest(zip_path: Path) -> tuple[dict[str, str], dict[str, int]]:
    try:
        import io
        import numpy as np
    except ImportError as error:
        raise RuntimeError("Translatotron 2 preparation requires NumPy") from error

    paths: dict[str, str] = {}
    lengths: dict[str, int] = {}
    with zipfile.ZipFile(zip_path, "r") as archive, zip_path.open("rb") as raw:
        for info in archive.infolist():
            sample_id = Path(info.filename).stem
            if sample_id in paths:
                raise ValueError(f"duplicate Mel feature in ZIP: {sample_id}")
            offset = (
                info.header_offset
                + 30
                + len(info.filename.encode("utf-8"))
                + len(info.extra)
            )
            raw.seek(offset)
            payload = raw.read(info.file_size)
            array = np.load(io.BytesIO(payload), allow_pickle=False)
            if array.ndim != 2:
                raise ValueError(f"expected a 2-D Mel feature for {sample_id}")
            paths[sample_id] = f"{zip_path.name}:{offset}:{info.file_size}"
            lengths[sample_id] = int(array.shape[0])
    return paths, lengths


def prepare_fairseq(
    common_root: Path,
    phoneme_root: Path,
    output_root: Path,
    *,
    mel_config: dict[str, Any],
    source_mel_config: dict[str, Any] | None = None,
    feature_extractor: FeatureExtractor | None = None,
    resume: bool = False,
    overwrite: bool = False,
) -> dict[str, Any]:
    settings = _mel_settings(mel_config)
    source_settings = _mel_settings(source_mel_config or mel_config)
    phonemes = {
        split: _read_phonemes(phoneme_root, split)
        for split in ("train", "dev", "test")
    }
    inventory = _read_inventory(phoneme_root)
    train_vocabulary = Counter(
        token for sequence in phonemes["train"].values() for token in sequence.split()
    )
    if not train_vocabulary:
        raise ValueError("train phoneme vocabulary is empty")
    known = set(inventory)
    for split in ("train", "dev", "test"):
        unknown = sorted(
            {token for sequence in phonemes[split].values() for token in sequence.split()} - known
        )
        if unknown:
            raise ValueError(f"unknown phonemes in {split}: {', '.join(unknown)}")

    rows_by_split: dict[str, list[dict[str, Any]]] = {}
    target_audio: dict[str, Path] = {}
    for split in ("train", "dev", "test"):
        rows_by_split[split] = []
        for row in read_common_manifest(common_root / f"{split}.jsonl"):
            pair_id = str(row["pair_id"])
            sequence = phonemes[split].get(pair_id)
            if sequence is None:
                raise ValueError(f"missing phonemes for {pair_id}")
            if pair_id in target_audio:
                raise ValueError(f"duplicate pair_id across splits: {pair_id}")
            target_audio[pair_id] = Path(row["en_audio"]).resolve()
            rows_by_split[split].append(
                {
                    "id": pair_id,
                    "src_audio": Path(row["ja_audio"]).resolve(),
                    "src_n_frames": _ten_ms_frames(Path(row["ja_audio"])),
                    "phonemes": sequence,
                }
            )

    for split, rows in rows_by_split.items():
        if not rows or {row['id'] for row in rows} != set(phonemes[split]):
            raise ValueError(f'empty split or extra phoneme IDs in {split}')
    output_root.mkdir(parents=True, exist_ok=True)
    feature_name = f"logmelspec{settings['n_mels']}"
    zip_path = output_root / f"{feature_name}.zip"
    if zip_path.exists() and not (resume or overwrite):
        raise ExistingOutputError(f"output already exists: {zip_path}")
    if overwrite or not zip_path.exists():
        extractor = feature_extractor or _extract_logmel_official
        with tempfile.TemporaryDirectory(
            dir=output_root, prefix=f".{feature_name}."
        ) as temporary_name:
            feature_root = Path(temporary_name)
            for pair_id, audio_path in sorted(target_audio.items()):
                extractor(audio_path, feature_root / f"{pair_id}.npy", settings)
            _write_feature_zip(feature_root, zip_path)

    target_paths, target_lengths = _zip_manifest(zip_path)
    missing_features = sorted(set(target_audio) - set(target_paths))
    if missing_features:
        raise ValueError(f"missing Mel features: {', '.join(missing_features)}")

    split_counts: dict[str, int] = {}
    target_root = output_root / "target_phoneme"
    for split, rows in rows_by_split.items():
        fairseq_lines = ["id\tsrc_audio\tsrc_n_frames\ttgt_audio\ttgt_n_frames\n"]
        target_lines = ["id\ttgt_text\n"]
        for row in rows:
            pair_id = row["id"]
            fairseq_lines.append(
                f"{pair_id}\t{row['src_audio']}\t{row['src_n_frames']}\t"
                f"{target_paths[pair_id]}\t{target_lengths[pair_id]}\n"
            )
            target_lines.append(f"{pair_id}\t{row['phonemes']}\n")
        atomic_write_text(
            output_root / f"{split}.tsv",
            "".join(fairseq_lines),
            resume=resume,
            overwrite=overwrite,
        )
        atomic_write_text(
            target_root / f"{split}.tsv",
            "".join(target_lines),
            resume=resume,
            overwrite=overwrite,
        )
        split_counts[split] = len(rows)

    dictionary = "".join(
        f"{token} {max(train_vocabulary.get(token, 0), 1)}\n" for token in inventory
    )
    atomic_write_text(
        target_root / "dict.txt", dictionary, resume=resume, overwrite=overwrite
    )

    sample_rate = settings["sample_rate"]
    base_config = {
        "audio_root": output_root.resolve().as_posix(),
        "input_channels": 1,
        "input_feat_per_channel": settings["n_mels"],
        "output_sample_rate": sample_rate,
        "specaugment": {
            "time_wrap_W": 0,
            "freq_mask_N": 1,
            "freq_mask_F": 27,
            "time_mask_N": 1,
            "time_mask_T": 100,
            "time_mask_p": 1.0,
        },
        "transforms": {
            "*": ["utterance_cmvn", "delta_deltas"],
            "_train": ["utterance_cmvn", "delta_deltas", "specaugment"],
        },
        "features": {
            "type": "spectrogram+melscale+log",
            "sample_rate": sample_rate,
            "eps": settings["eps"],
            "n_mels": settings["n_mels"],
            "n_fft": settings["n_fft"],
            "window_fn": "hann",
            "win_length": settings["win_length"],
            "hop_length": settings["hop_length"],
            "win_len_t": settings["win_length"] / sample_rate,
            "hop_len_t": settings["hop_length"] / sample_rate,
            "f_min": settings["f_min"],
            "f_max": settings["f_max"],
            "n_stft": settings["n_fft"] // 2 + 1,
        },
    }
    multitask = {
        "target_phoneme": {
            "decoder_type": "transformer",
            "dict": (target_root / "dict.txt").resolve().as_posix(),
            "data": target_root.resolve().as_posix(),
            "is_first_pass_decoder": True,
            "loss_weight": 1.0,
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
        "mel": settings,
        "source_mel": source_settings,
        "mel_zip_sha256": sha256_file(zip_path),
        "phoneme_vocabulary_source": "fixed_espeak_inventory",
        "phoneme_inventory_sha256": sha256_file(phoneme_root / "inventory.txt"),
        "preparation_reference": "fairseq prep_s2spect_data.py",
        "splits": split_counts,
    }
    atomic_write_json(
        output_root / "data-lock.json", lock, resume=resume, overwrite=overwrite
    )
    atomic_write_json(
        output_root / "mel-spec.json",
        {**settings, "log_transform": "natural_log_clamp_eps", "normalization": "none"},
        resume=resume, overwrite=overwrite,
    )
    atomic_write_json(output_root / 'source-mel-spec.json',
        {**source_settings, 'log_transform': 'natural_log_clamp_eps', 'normalization': 'utterance_cmvn'},
        resume=resume, overwrite=overwrite)
    return lock
