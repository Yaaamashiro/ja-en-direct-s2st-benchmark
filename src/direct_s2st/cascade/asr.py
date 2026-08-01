from __future__ import annotations

from pathlib import Path


class WhisperASR:
    def __init__(self, *, model_id: str, revision: str, device: int = 0) -> None:
        import torch
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

        dtype = torch.float16 if device >= 0 else torch.float32
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            revision=revision,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        )
        if device >= 0:
            model = model.to(f"cuda:{device}")
        processor = AutoProcessor.from_pretrained(model_id, revision=revision)
        self.pipeline = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=dtype,
            device=device,
        )

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
