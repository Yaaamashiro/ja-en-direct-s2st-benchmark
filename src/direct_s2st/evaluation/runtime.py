from __future__ import annotations

import array
import wave
from pathlib import Path
from typing import Any


def audio_quality(path: Path, *, silence_threshold: float = 0.01) -> dict[str, float]:
    with wave.open(str(path), "rb") as handle:
        if handle.getsampwidth() != 2:
            raise ValueError(f"quality metrics require PCM16 WAV: {path}")
        channels = handle.getnchannels()
        rate = handle.getframerate()
        frames = handle.getnframes()
        samples = array.array("h", handle.readframes(frames))
    if channels > 1:
        samples = array.array("h", samples[::channels])
    count = len(samples)
    if count == 0:
        return {"duration": 0.0, "silence_ratio": 1.0, "clipping_ratio": 0.0}
    silence_limit = int(32767 * silence_threshold)
    silent = sum(abs(sample) <= silence_limit for sample in samples)
    clipped = sum(abs(sample) >= 32767 for sample in samples)
    return {
        "duration": frames / rate,
        "silence_ratio": silent / count,
        "clipping_ratio": clipped / count,
    }


def aggregate_runtime(records: list[dict[str, Any]]) -> dict[str, float | None]:
    successful = [record for record in records if record.get("status") == "success"]
    return {
        "mean_inference_seconds": (
            sum(float(record["inference_seconds"]) for record in successful) / len(successful)
            if successful
            else None
        ),
        "mean_rtf": (
            sum(float(record["real_time_factor"]) for record in successful) / len(successful)
            if successful
            else None
        ),
        "failure_rate": 1 - len(successful) / len(records) if records else 0.0,
    }
