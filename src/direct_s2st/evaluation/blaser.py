from __future__ import annotations

from pathlib import Path
from typing import Protocol
from ..progress import operation


class BlaserScorer(Protocol):
    def __call__(self, source_audio: Path, output_audio: Path) -> float: ...


class UnavailableBlaser:
    """Explicit marker used when the pinned BLASER runtime is unavailable."""

    def __call__(self, source_audio: Path, output_audio: Path) -> float:
        raise RuntimeError("BLASER runtime is unavailable; ASR-BLEU remains evaluable")


class SonarBlaser:
    """BLASER 2.0 QE over language-specific SONAR speech embeddings."""

    @operation('model: load SONAR/BLASER')
    def __init__(
        self,
        *,
        source_encoder: str = "sonar_speech_encoder_jpn",
        target_encoder: str = "sonar_speech_encoder_eng",
    ) -> None:
        import torch
        from sonar.inference_pipelines.speech import SpeechToEmbeddingModelPipeline
        from sonar.models.blaser.loader import load_blaser_model

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.source = SpeechToEmbeddingModelPipeline(
            encoder=source_encoder, device=self.device
        )
        self.target = SpeechToEmbeddingModelPipeline(
            encoder=target_encoder, device=self.device
        )
        self.model = load_blaser_model("blaser_2_0_qe").to(self.device).eval()

    def __call__(self, source_audio: Path, output_audio: Path) -> float:
        source_embedding = self.source.predict([str(source_audio)])
        target_embedding = self.target.predict([str(output_audio)])
        return float(self.model(src=source_embedding, mt=target_embedding).item())
