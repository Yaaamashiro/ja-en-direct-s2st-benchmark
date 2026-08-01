from __future__ import annotations

import os
import tempfile
from pathlib import Path


class QwenTTS:
    def __init__(
        self,
        *,
        model_id: str,
        revision: str,
        speaker: str,
        language: str,
        device: str = "cuda:0",
    ) -> None:
        import torch
        from qwen_tts import Qwen3TTSModel

        self.model = Qwen3TTSModel.from_pretrained(
            model_id,
            revision=revision,
            device_map=device,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        self.speaker = speaker
        self.language = language

    def __call__(self, text: str, output_path: Path) -> None:
        import soundfile as sf

        waveforms, sample_rate = self.model.generate_custom_voice(
            text=text,
            language=self.language,
            speaker=self.speaker,
            do_sample=False,
            subtalker_dosample=False,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            dir=output_path.parent, prefix=f".{output_path.stem}.", suffix=".wav"
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            sf.write(temporary, waveforms[0], sample_rate, subtype="PCM_16")
            os.replace(temporary, output_path)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
