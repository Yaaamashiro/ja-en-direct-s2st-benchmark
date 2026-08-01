from __future__ import annotations

from pathlib import Path


class EcapaSimilarity:
    def __init__(self, *, model_id: str, revision: str, cache_root: Path) -> None:
        import torch
        from huggingface_hub import snapshot_download
        from speechbrain.inference.speaker import EncoderClassifier

        self.torch = torch
        snapshot = snapshot_download(
            repo_id=model_id,
            revision=revision,
            cache_dir=cache_root / "huggingface",
        )
        self.model = EncoderClassifier.from_hparams(
            source=snapshot,
            savedir=str(cache_root / model_id.replace("/", "--") / revision),
            run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"},
        )

    def __call__(self, first: Path, second: Path) -> float:
        first_embedding = self.model.encode_file(str(first)).squeeze()
        second_embedding = self.model.encode_file(str(second)).squeeze()
        return float(
            self.torch.nn.functional.cosine_similarity(
                first_embedding, second_embedding, dim=0
            ).item()
        )
