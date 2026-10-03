from __future__ import annotations

import os
import json
from ..progress import operation, track
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
from ..journal import digest
from ..preparation import Checkpoints, adaptive_map, checkpoint_map, file_stamp
from ..io import ExistingOutputError, atomic_write_json, atomic_write_text, read_jsonl

FeatureExtractor = Callable[[Path, Path, dict[str, Any]], None]

# Fixed supplement to the probe-derived eSpeak NG 1.52.0 inventory. These
# additional phones were identified in train data; dev/test never grow it.
# Keep original phoneme files/metadata and their checkpoint identity intact.
ESPEAK_152_INVENTORY_SUPPLEMENT = (
    'a\u200dɪ\u200də', 'a\u200dɪ\u200dɚ', 'o', 'r', 'ɐ', 'ɑ\u0303', 'ɔ',
)


def _training_inventory(phoneme_root, common_root, original):
    metadata_paths = sorted(phoneme_root.glob('metadata*.json'))
    if not metadata_paths:
        return original
    metadata = [json.loads(path.read_text(encoding='utf-8')) for path in metadata_paths]
    if not all(row.get('engine') == 'espeak-ng' and row.get('version') == '1.52.0'
               for row in metadata):
        return original
    checksum = sha256_file(phoneme_root / 'inventory.txt')
    for row in metadata:
        if row.get('inventory_sha256') != checksum:
            raise ValueError('phoneme inventory differs from saved metadata')
        for split, expected in row['common_manifests'].items():
            if sha256_file(common_root / f'{split}.jsonl') != expected:
                raise ValueError(f'phoneme metadata refers to stale {split} input')
        suffix = '' if row.get('num_shards', 1) == 1 else f".shard-{row['shard_index']:05d}-of-{row['num_shards']:05d}"
        for split, expected in row.get('phoneme_manifests', {}).items():
            if sha256_file(phoneme_root / f'{split}{suffix}.tsv') != expected:
                raise ValueError(f'phoneme labels differ from saved {split} metadata')
    return tuple(sorted(set(original) | set(ESPEAK_152_INVENTORY_SUPPLEMENT)))


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
                if not separator or not pair_id or not sequence.split():
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


@operation('phonemes: dictionary and saved-label preflight (no WAV reads)')
def validate_phonemes(common_root, phoneme_root):
    common_root, phoneme_root = Path(common_root), Path(phoneme_root)
    inventory = _training_inventory(phoneme_root, common_root, _read_inventory(phoneme_root))
    known, counts, problems = set(inventory), {}, []
    for split in ('train', 'dev', 'test'):
        labels = _read_phonemes(phoneme_root, split)
        ids = [row['pair_id'] for row in read_jsonl(common_root / f'{split}.jsonl')]
        if not ids or len(ids) != len(set(ids)) or set(ids) != labels.keys():
            raise ValueError(f'phoneme IDs do not match {split} common manifest')
        for pair_id, sequence in track(labels.items(), f'phonemes: validate {split} dictionary', total=len(labels)):
            unknown = set(sequence.split()) - known
            if unknown:
                problems.append(f'{split}/{pair_id}: {sorted(unknown)}')
        counts[split] = len(labels)
    if problems:
        raise ValueError(f'unknown phonemes in {len(problems)} samples: ' + '; '.join(problems[:20]))
    return dict(splits=counts, inventory_size=len(inventory), wav_reads=0)


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
        # Our caller reserves a new private temporary file with mkstemp.
        # fairseq otherwise sees that empty file and skips feature extraction.
        # This does not authorize replacing any published/cached feature.
        overwrite=True,
    )


@operation('translatotron2/prepare_fairseq: _write_feature_zip')
def _write_feature_zip(feature_root: Path, zip_path: Path, files=None) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=zip_path.parent, prefix=f".{zip_path.name}.", suffix=".tmp"
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with zipfile.ZipFile(temporary, "w", zipfile.ZIP_STORED) as archive:
            entries = files if files is not None else [(p.name, p) for p in sorted(feature_root.glob('*.npy'))]
            for name, path in track(entries, 'mel: write ZIP'):
                archive.write(path, arcname=name)
        os.replace(temporary, zip_path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


@operation('translatotron2/prepare_fairseq: _zip_manifest')
def _zip_manifest(zip_path: Path, *, resume=False) -> tuple[dict[str, str], dict[str, int]]:
    try:
        import io
        import numpy as np
    except ImportError as error:
        raise RuntimeError("Translatotron 2 preparation requires NumPy") from error

    paths: dict[str, str] = {}
    lengths: dict[str, int] = {}
    with Checkpoints(zip_path.parent.parent / '.prep-checkpoints' / f'{zip_path.stem}-index',
                     dict(stage='mel-index-v1'), resume=resume) as cache, \
            zipfile.ZipFile(zip_path, "r") as archive, zip_path.open("rb") as raw:
        stamp = file_stamp(zip_path)
        for info in track(archive.infolist(), 'mel: validate ZIP'):
            sample_id = Path(info.filename).stem
            if sample_id in paths:
                raise ValueError(f"duplicate Mel feature in ZIP: {sample_id}")
            offset = (
                info.header_offset
                + 30
                + len(info.filename.encode("utf-8"))
                + len(info.extra)
            )
            key = digest([stamp, info.filename, offset, info.file_size])
            saved = cache.get(key) if os.environ.get('S2ST_PREP_RECHECK') != '1' else None
            if saved is None:
                raw.seek(offset)
                payload = raw.read(info.file_size)
                array = np.load(io.BytesIO(payload), allow_pickle=False)
                if array.ndim != 2:
                    raise ValueError(f"expected a 2-D Mel feature for {sample_id}")
                saved = dict(frames=int(array.shape[0]))
                cache.record(key, saved)
            paths[sample_id] = f"{zip_path.name}:{offset}:{info.file_size}"
            lengths[sample_id] = saved['frames']
    return paths, lengths


@operation('translatotron2/prepare_fairseq: prepare_fairseq')
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
    original_inventory = _read_inventory(phoneme_root)
    inventory = _training_inventory(phoneme_root, common_root, original_inventory)
    train_vocabulary = Counter(
        token for sequence in phonemes["train"].values() for token in sequence.split()
    )
    if not train_vocabulary:
        raise ValueError("train phoneme vocabulary is empty")
    known = set(inventory)
    unknown_by_split = {}
    for split in ("train", "dev", "test"):
        unknown = sorted(
            {token for sequence in phonemes[split].values() for token in sequence.split()} - known
        )
        if unknown:
            unknown_by_split[split] = unknown
    if unknown_by_split:
        raise ValueError('; '.join(f"unknown phonemes in {split}: {', '.join(tokens)}"
                                  for split, tokens in unknown_by_split.items()))
    # Catch missing/extra/empty labels before resolving/opening all WAVs.
    validate_phonemes(common_root, phoneme_root)

    rows_by_split: dict[str, list[dict[str, Any]]] = {}
    target_audio: dict[str, Path] = {}
    for split in ("train", "dev", "test"):
        rows_by_split[split] = []
        for row in track(read_common_manifest(common_root / f"{split}.jsonl", parallel=True), f'mel: read {split}'):
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
                    "phonemes": sequence,
                }
            )

    for split, rows in rows_by_split.items():
        def source_frames(row):
            return {'frames': _ten_ms_frames(row['src_audio'])}
        with Checkpoints(output_root.parent / '.prep-checkpoints' / f'source-headers-{split}',
                         dict(stage='source-headers-v1'), resume=resume and not overwrite and os.environ.get('S2ST_PREP_RECHECK') != '1',
                         overwrite=overwrite) as cache:
            frames = list(checkpoint_map(source_frames, rows, cache,
                                    lambda row: digest(file_stamp(row['src_audio'])),
                                    f'mel: source headers {split}', total=len(rows)))
            rows_by_split[split] = [{**row, 'src_n_frames': saved['frames']}
                                   for row, saved in zip(rows, frames)]
        if not rows or {row['id'] for row in rows} != set(phonemes[split]):
            raise ValueError(f'empty split or extra phoneme IDs in {split}')
    output_root.mkdir(parents=True, exist_ok=True)
    feature_name = f"logmelspec{settings['n_mels']}"
    zip_path = output_root / f"{feature_name}.zip"
    if zip_path.exists() and not (resume or overwrite):
        raise ExistingOutputError(f"output already exists: {zip_path}")
    identity = dict(stage='mel-v1', settings=settings, source_settings=source_settings,
                    dataset=sha256_file(common_root / 'dataset-lock.json'),
                    manifests={split: sha256_file(common_root / f'{split}.jsonl')
                               for split in ('train', 'dev', 'test')},
                    output=str(output_root.resolve()))
    identity_path = output_root.parent / '.prep-checkpoints' / f'{output_root.name}-mel-input.json'
    if zip_path.exists() and not identity_path.exists() and not overwrite:
        old_lock_path = output_root / 'data-lock.json'
        if not old_lock_path.is_file():
            raise ValueError('orphan Mel ZIP without configuration lock; use a new output root or --overwrite')
        old_lock = json.loads(old_lock_path.read_text(encoding='utf-8'))
        if (old_lock['mel'] != settings or old_lock['source_mel'] != source_settings
                or old_lock['common_dataset_lock_sha256'] != identity['dataset']):
            raise ValueError('existing Mel ZIP configuration mismatch')
    atomic_write_json(identity_path, identity, resume=True, overwrite=overwrite)
    sharded = zip_path.with_suffix('.shards.json').is_file()
    if overwrite or not zip_path.exists() or sharded:
        extractor = feature_extractor or _extract_logmel_official
        if feature_extractor is None:
            # Load native pools before adaptive_map limits their thread counts.
            import torchaudio  # noqa: F401
        with Checkpoints(output_root.parent / '.prep-checkpoints/mel', identity,
                         resume=resume and not overwrite, overwrite=overwrite) as cache:
            from .mel_storage import feature_path, resolve_feature
            def key(item):
                return digest([item[0], file_stamp(item[1])])
            def extract(item):
                import numpy as np
                pair_id, audio_path = item
                destination = feature_path(cache.root, f'{key(item)[:32]}.npy')
                destination.parent.mkdir(parents=True, exist_ok=True)
                fd, temporary_name = tempfile.mkstemp(dir=destination.parent, suffix='.npy')
                os.close(fd)
                temporary = Path(temporary_name)
                try:
                    extractor(audio_path, temporary, settings)
                    array = np.load(temporary, allow_pickle=False)
                    if array.ndim != 2 or array.shape[1] != settings['n_mels'] or not np.isfinite(array).all():
                        raise ValueError(f'invalid Mel feature: {pair_id}')
                    checksum = sha256_file(temporary)
                    if destination.exists() and not overwrite:
                        if sha256_file(destination) != checksum:
                            raise ValueError(f'conflicting cached Mel feature: {pair_id}')
                    else:
                        os.replace(temporary, destination)
                    return dict(id=pair_id, path=str(destination), sha256=checksum)
                finally:
                    temporary.unlink(missing_ok=True)
            files = []
            for result in checkpoint_map(extract, sorted(target_audio.items()), cache, key,
                                         'mel: extract/reuse features', total=len(target_audio)):
                path = resolve_feature(cache.root, result)
                files.append((result['id'] + '.npy', path))
            from .recovery import publish_archives
            target_paths, target_lengths, archive_hashes = publish_archives(
                files, zip_path, resume=resume, overwrite=overwrite)
    else:
        # Existing single-ZIP datasets retain their locators and lock format.
        target_paths, target_lengths = _zip_manifest(zip_path, resume=resume)
        archive_hashes = {zip_path.name: sha256_file(zip_path)}
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
        "mel_zip_sha256": archive_hashes[zip_path.name],
        "phoneme_vocabulary_source": "fixed_espeak_inventory",
        "phoneme_inventory_sha256": sha256_file(phoneme_root / "inventory.txt"),
        "preparation_reference": "fairseq prep_s2spect_data.py",
        "splits": split_counts,
    }
    if inventory != original_inventory:
        lock['phoneme_vocabulary_source'] = 'fixed_espeak_inventory_1.52.0_supplement_v1'
        lock['phoneme_inventory_supplement'] = list(ESPEAK_152_INVENTORY_SUPPLEMENT)
        lock['prepared_phoneme_inventory_sha256'] = digest(list(inventory))
    if len(archive_hashes) > 1:
        lock['mel_shards_sha256'] = archive_hashes
    atomic_write_json(output_root / "data-lock.json", lock, resume=resume, overwrite=overwrite)
    atomic_write_json(
        output_root / "mel-spec.json",
        {**settings, "log_transform": "natural_log_clamp_eps", "normalization": "none"},
        resume=resume, overwrite=overwrite,
    )
    atomic_write_json(output_root / 'source-mel-spec.json',
        {**source_settings, 'log_transform': 'natural_log_clamp_eps', 'normalization': 'utterance_cmvn'},
        resume=resume, overwrite=overwrite)
    return lock
