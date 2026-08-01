from __future__ import annotations

import wave
from collections.abc import Callable
from pathlib import Path
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


def run_pipeline(
    common_manifest: Path,
    output_root: Path,
    *,
    run_id: str,
    asr: Callable[[Path], str],
    mt: Callable[[str], str],
    tts: TTS,
    split: str = "test",
    shard_index: int = 0,
    num_shards: int = 1,
    limit: int | None = None,
    resume: bool = False,
    overwrite: bool = False,
) -> dict[str, Any]:
    prediction_path = output_root / "predictions.jsonl"
    if prediction_path.exists() and not (resume or overwrite):
        raise ExistingOutputError(f"output already exists: {prediction_path}")
    prior = {
        str(row["pair_id"]): row for row in read_jsonl(prediction_path)
    } if resume and prediction_path.is_file() else {}
    records: list[dict[str, Any]] = []
    selected = 0
    for row in read_jsonl(common_manifest):
        pair_id = str(row["pair_id"])
        if stable_shard(pair_id, num_shards) != shard_index:
            continue
        if limit is not None and selected >= limit:
            break
        selected += 1
        old = prior.get(pair_id)
        if old and old.get("status") == "success" and Path(old["output_audio"]).is_file():
            records.append(old)
            continue
        output_audio = output_root / "audio" / f"{pair_id}.wav"
        record: dict[str, Any] = {
            "pair_id": pair_id,
            "system_id": "cascade",
            "run_id": run_id,
            "source_audio": str(Path(row["ja_audio"]).resolve()),
            "reference_audio": str(Path(row["en_audio"]).resolve()),
            "reference_text": row["en_text"],
            "output_audio": str(output_audio.resolve()),
            "output_duration": None,
            "status": "failed",
            "error": None,
        }
        total_started = perf_counter()
        try:
            started = perf_counter()
            record["asr_ja_text"] = asr(Path(row["ja_audio"]))
            record["asr_seconds"] = perf_counter() - started
            started = perf_counter()
            record["mt_en_text"] = mt(record["asr_ja_text"])
            record["mt_seconds"] = perf_counter() - started
            started = perf_counter()
            tts(record["mt_en_text"], output_audio)
            record["tts_seconds"] = perf_counter() - started
            record["output_duration"] = _duration(output_audio)
            record["status"] = "success"
        except Exception as error:
            if _fatal_accelerator_error(error):
                raise
            record["error"] = f"{type(error).__name__}: {error}"
        record["total_seconds"] = perf_counter() - total_started
        record["inference_seconds"] = record["total_seconds"]
        duration = float(record.get("output_duration") or 0.0)
        record["real_time_factor"] = (
            record["total_seconds"] / duration if duration > 0 else None
        )
        records.append(record)
        atomic_write_jsonl(prediction_path, records, overwrite=True)
    atomic_write_jsonl(prediction_path, records, overwrite=True)
    return {
        "predictions": str(prediction_path),
        "samples": len(records),
        "successes": sum(record["status"] == "success" for record in records),
        "failures": sum(record["status"] == "failed" for record in records),
    }


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
