from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .io import read_jsonl


SYSTEMS = ("s2ut", "translatotron2", "cascade", "s2t_tts")


@dataclass(frozen=True)
class Prediction:
    pair_id: str
    system_id: str
    run_id: str
    source_audio: str
    reference_audio: str
    reference_text: str
    output_audio: str | None
    output_duration: float | None
    inference_seconds: float
    real_time_factor: float | None
    status: str
    error: str | None

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Prediction":
        missing = [field for field in cls.__dataclass_fields__ if field not in value]
        if missing:
            raise ValueError(f"missing prediction fields: {', '.join(missing)}")
        prediction = cls(**{field: value[field] for field in cls.__dataclass_fields__})
        if prediction.system_id not in SYSTEMS:
            raise ValueError(f"invalid system_id: {prediction.system_id}")
        if prediction.status not in ("success", "failed"):
            raise ValueError(f"invalid prediction status: {prediction.status}")
        if prediction.status == "success":
            if not prediction.output_audio or not Path(prediction.output_audio).is_file():
                raise ValueError(f"successful prediction has no output audio: {prediction.pair_id}")
            if prediction.error is not None:
                raise ValueError("successful prediction must have error=null")
        elif not prediction.error:
            raise ValueError("failed prediction must retain an error")
        return prediction


def validate_predictions(path: Path) -> dict[str, int]:
    seen: set[str] = set()
    success = 0
    failed = 0
    for row in read_jsonl(path):
        prediction = Prediction.from_dict(row)
        if prediction.pair_id in seen:
            raise ValueError(f"duplicate prediction pair_id: {prediction.pair_id}")
        seen.add(prediction.pair_id)
        if prediction.status == "success":
            success += 1
        else:
            failed += 1
    return {"samples": len(seen), "successes": success, "failures": failed}
