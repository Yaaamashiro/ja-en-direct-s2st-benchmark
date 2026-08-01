from __future__ import annotations

from pathlib import Path


class EvaluationASR:
    def __init__(self, *, model_id: str, revision: str, device: int = 0) -> None:
        from direct_s2st.cascade.asr import WhisperASR

        self.model = WhisperASR(model_id=model_id, revision=revision, device=device)

    def __call__(self, audio_path: Path) -> str:
        pipeline = self.model.pipeline
        result = pipeline(
            str(audio_path),
            generate_kwargs={
                "language": "english",
                "task": "transcribe",
                "do_sample": False,
                "condition_on_prev_tokens": False,
            },
        )
        return str(result["text"]).strip()
