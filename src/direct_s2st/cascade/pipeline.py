from __future__ import annotations

import wave
from collections.abc import Callable
from pathlib import Path
from ..manifests.reader import read_common_manifest
from time import perf_counter
from typing import Any, Protocol

from ..io import ExistingOutputError, atomic_write_jsonl, read_jsonl
from ..s2ut.extract_units import stable_shard


class TTS(Protocol):
    def __call__(self, text: str, output_path: Path) -> None: ...


def _duration(path: Path) -> float:
    with wave.open(str(path), "rb") as handle:
        return handle.getnframes() / handle.getframerate()


def _fatal_accelerator_error(error: BaseException) -> bool:
    message = str(error).lower()
    return isinstance(error, RuntimeError) and any(
        marker in message
        for marker in ("cuda", "out of memory", "device-side assert", "cublas", "cudnn")
    )


def _run_text_pipeline(
    common_manifest: Path,
    output_root: Path,
    *,
    run_id: str,
    text_stages: list,
    system_id: str,
    strict: bool = False,
    tts: TTS,
    split: str = "test",
    shard_index: int = 0,
    num_shards: int = 1,
    limit: int | None = None,
    resume: bool = False,
    overwrite: bool = False,
    model_identity: dict | None = None,
) -> dict[str, Any]:
    if num_shards < 1 or not 0 <= shard_index < num_shards or (limit is not None and limit < 1):
        raise ValueError('invalid shard/limit')
    prediction_path = output_root / "predictions.jsonl"
    if prediction_path.exists() and not (resume or overwrite):
        raise ExistingOutputError(f"output already exists: {prediction_path}")
    from ..journal import Journal
    from ..hashing import sha256_file
    common_rows = list(read_common_manifest(common_manifest))
    if len({r['pair_id'] for r in common_rows}) != len(common_rows) or any(r['split'] != split for r in common_rows):
        raise ValueError('invalid cascade common IDs/split')
    identity = dict(common=sha256_file(common_manifest), models=model_identity, run_id=run_id,
                    split=split, shard_index=shard_index, num_shards=num_shards, limit=limit,
                    source={r['pair_id']: sha256_file(Path(r['ja_audio'])) for r in common_rows})
    if strict:
        identity.update(system_id=system_id,
                        reference={r['pair_id']: sha256_file(Path(r['en_audio'])) for r in common_rows})
    journal = Journal(prediction_path, identity, resume=resume, overwrite=overwrite)
    prior = journal.rows.copy()
    records: list[dict[str, Any]] = []
    selected = 0
    for row in common_rows:
        pair_id = str(row["pair_id"])
        if stable_shard(pair_id, num_shards) != shard_index:
            continue
        if limit is not None and selected >= limit:
            break
        selected += 1
        old = prior.get(pair_id)
        if old and old.get("status") == "success" and Path(old["output_audio"]).is_file() and old.get('output_sha256') == sha256_file(Path(old['output_audio'])):
            records.append(old)
            continue
        if strict and old and old.get('status') == 'success':
            raise ValueError(f'output SHA/missing audio on resume: {pair_id}; review or explicitly overwrite')
        output_audio = output_root / "audio" / f"{pair_id}.wav"
        if not output_audio.resolve().is_relative_to((output_root / 'audio').resolve()):
            raise ValueError('unsafe cascade pair ID')
        record: dict[str, Any] = {
            "pair_id": pair_id,
            "system_id": system_id,
            "run_id": run_id,
            "split": split,
            "source_audio": str(Path(row["ja_audio"]).resolve()),
            "reference_audio": str(Path(row["en_audio"]).resolve()),
            "reference_text": row["en_text"],
            "output_audio": str(output_audio.resolve()),
            "output_duration": None,
            "status": "failed",
            "error": None,
        }
        if strict:
            record.update(s2t_en_text=None, s2t_seconds=0.0, tts_seconds=0.0, output_sha256=None)
        total_started = perf_counter()
        fatal_error = None
        try:
            text = Path(row["ja_audio"])
            for field, timing, component in text_stages:
                started = perf_counter()
                try:
                    text = component(text)
                    record[field] = text
                finally:
                    record[timing] = perf_counter() - started
            if strict and (not isinstance(text, str) or not text.strip()):
                raise ValueError('empty translated text')
            started = perf_counter()
            tts(text, output_audio)
            record["tts_seconds"] = perf_counter() - started
            record["output_duration"] = _duration(output_audio)
            if strict:
                import numpy as np
                import soundfile as sf
                waveform, _ = sf.read(output_audio, always_2d=True)
                if waveform.shape[1] != 1 or not waveform.size or not np.isfinite(waveform).all() or not np.any(waveform):
                    raise ValueError('invalid/silent generated WAV')
            record["status"] = "success"
            record['output_sha256'] = sha256_file(output_audio)
        except Exception as error:
            if _fatal_accelerator_error(error):
                fatal_error = error
            record["error"] = f"{type(error).__name__}: {error}"
        record["total_seconds"] = perf_counter() - total_started
        record["inference_seconds"] = record["total_seconds"]
        duration = float(record.get("output_duration") or 0.0)
        record["real_time_factor"] = (
            record["total_seconds"] / duration if duration > 0 else None
        )
        records.append(record)
        journal.record(record)
        if fatal_error is not None:
            raise fatal_error
    atomic_write_jsonl(prediction_path, records, overwrite=True)
    return {
        "predictions": str(prediction_path),
        "samples": len(records),
        "successes": sum(record["status"] == "success" for record in records),
        "failures": sum(record["status"] == "failed" for record in records),
    }


def run_pipeline(common_manifest, output_root, *, run_id, asr, mt, tts, **kwargs):
    """Existing ASR→MT→TTS contract, including original journal identity."""
    return _run_text_pipeline(common_manifest, output_root, run_id=run_id, tts=tts,
        text_stages=[('asr_ja_text', 'asr_seconds', asr), ('mt_en_text', 'mt_seconds', mt)],
        system_id='cascade', **kwargs)


def components_from_config(config: dict[str, Any]) -> tuple[WhisperASR, NllbTranslator, QwenTTS]:
    from .asr import WhisperASR
    from .mt import NllbTranslator
    from .tts import QwenTTS

    asr_values = config["asr"]
    mt_values = config["mt"]
    tts_values = config["tts"]
    return (
        WhisperASR(model_id=asr_values["model"], revision=asr_values["revision"]),
        NllbTranslator(
            model_id=mt_values["model"],
            revision=mt_values["revision"],
            source_language=mt_values["source_language"],
            target_language=mt_values["target_language"],
        ),
        QwenTTS(
            model_id=tts_values["model"],
            revision=tts_values["revision"],
            speaker=tts_values["speaker"],
            language=tts_values["language"],
        ),
    )
