from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

from .io import append_jsonl


@dataclass
class JsonlLogger:
    path: Path
    command: str

    def log(self, status: str, **fields: Any) -> None:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "command": self.command,
            "status": status,
            **fields,
        }
        append_jsonl(self.path, record)

    def sample(self, pair_id: str, **fields: Any) -> "SampleTimer":
        return SampleTimer(self, pair_id, fields)


class SampleTimer:
    def __init__(self, logger: JsonlLogger, pair_id: str, fields: dict[str, Any]):
        self.logger = logger
        self.pair_id = pair_id
        self.fields = fields
        self.started = 0.0

    def __enter__(self) -> "SampleTimer":
        self.started = perf_counter()
        self.logger.log("started", pair_id=self.pair_id, **self.fields)
        return self

    def __exit__(self, kind: object, error: object, traceback: object) -> bool:
        duration = perf_counter() - self.started
        if error is None:
            self.logger.log(
                "completed", pair_id=self.pair_id, duration_seconds=duration, **self.fields
            )
        else:
            self.logger.log(
                "failed",
                pair_id=self.pair_id,
                duration_seconds=duration,
                error_type=type(error).__name__,
                error=str(error),
                **self.fields,
            )
        return False
