from __future__ import annotations

from pathlib import Path


class WhisperASR:
    def __init__(self, *, model_id: str, revision: str, device: int = 0) -> None:
        from .whisper import load_whisper_pipeline
        self.pipeline = load_whisper_pipeline(model_id, revision, device)

    def __call__(self, audio_path: Path) -> str:
        result = self.pipeline(
            str(audio_path),
            generate_kwargs={
                "language": "japanese",
                "task": "transcribe",
                "do_sample": False,
                "condition_on_prev_tokens": False,
            },
        )
        return str(result["text"]).strip()
