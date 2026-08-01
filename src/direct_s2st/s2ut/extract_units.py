from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from ..hashing import sha256_file, sha256_text
from ..io import atomic_write_json, atomic_write_jsonl, atomic_write_text, read_jsonl
from .reduce_units import reduce_consecutive_units, validate_units


class UnitExtractor(Protocol):
    def __call__(self, audio_path: Path) -> list[int]: ...


def stable_shard(pair_id: str, num_shards: int) -> int:
    return int(sha256_text(pair_id)[:16], 16) % num_shards


def serialize_units(units: list[int]) -> str:
    return " ".join(str(unit) for unit in units) + "\n"


def load_unit_file(path: Path, *, clusters: int) -> list[int]:
    try:
        values = [int(value) for value in path.read_text(encoding="utf-8").split()]
    except ValueError as error:
        raise ValueError(f"invalid unit file: {path}") from error
    return validate_units(values, clusters=clusters)


class HubertKMeansExtractor:
    """Lazy Transformers HuBERT extractor with NumPy centroid assignment."""

    def __init__(
        self,
        *,
        model_id: str,
        revision: str,
        layer: int,
        centroids_path: Path,
        device: str = "cuda",
    ) -> None:
        import numpy as np
        import torch
        from transformers import AutoFeatureExtractor, HubertModel

        self.np = np
        self.torch = torch
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.processor = AutoFeatureExtractor.from_pretrained(model_id, revision=revision)
        self.model = HubertModel.from_pretrained(model_id, revision=revision).to(self.device)
        self.model.eval()
        self.layer = layer
        centers = np.load(centroids_path)
        if centers.ndim != 2:
            raise ValueError("k-means centroids must be a [clusters, features] array")
        self.centers = centers.astype(np.float32, copy=False)

    def __call__(self, audio_path: Path) -> list[int]:
        import soundfile as sf

        waveform, sample_rate = sf.read(audio_path, dtype="float32", always_2d=False)
        if sample_rate != 16000 or waveform.ndim != 1:
            raise ValueError(f"HuBERT input must be 16 kHz mono: {audio_path}")
        inputs = self.processor(waveform, sampling_rate=16000, return_tensors="pt")
        values = inputs.input_values.to(self.device)
        with self.torch.inference_mode():
            output = self.model(values, output_hidden_states=True)
        features = output.hidden_states[self.layer][0].float().cpu().numpy()
        if features.shape[1] != self.centers.shape[1]:
            raise ValueError("HuBERT feature width does not match k-means centroids")
        distances = (
            (features**2).sum(axis=1, keepdims=True)
            - 2 * features @ self.centers.T
            + (self.centers**2).sum(axis=1)
        )
        return distances.argmin(axis=1).astype(int).tolist()


def extract_units(
    common_root: Path,
    output_root: Path,
    *,
    extractor: UnitExtractor,
    split: str | None,
    clusters: int,
    hubert_model: str,
    hubert_revision: str,
    hubert_layer: int,
    kmeans_sha256: str,
    shard_index: int = 0,
    num_shards: int = 1,
    limit: int | None = None,
    resume: bool = False,
    overwrite: bool = False,
) -> dict[str, Any]:
    splits = (split,) if split else ("train", "dev", "test")
    records: list[dict[str, Any]] = []
    processed = 0
    for current_split in splits:
        split_processed = 0
        for row in read_jsonl(common_root / f"{current_split}.jsonl"):
            pair_id = str(row["pair_id"])
            if stable_shard(pair_id, num_shards) != shard_index:
                continue
            if limit is not None and split_processed >= limit:
                break
            original_path = output_root / current_split / "original" / f"{pair_id}.units"
            reduced_path = output_root / current_split / "reduced" / f"{pair_id}.units"
            if resume and original_path.is_file() and reduced_path.is_file():
                original = load_unit_file(original_path, clusters=clusters)
                reduced = load_unit_file(reduced_path, clusters=clusters)
            else:
                original = validate_units(extractor(Path(row["en_audio"])), clusters=clusters)
                reduced = reduce_consecutive_units(original)
                atomic_write_text(original_path, serialize_units(original), overwrite=overwrite)
                atomic_write_text(reduced_path, serialize_units(reduced), overwrite=overwrite)
            if len(reduced) > len(original):
                raise ValueError(f"reduced unit sequence grew for {pair_id}")
            records.append(
                {
                    "pair_id": pair_id,
                    "split": current_split,
                    "unit_count_original": len(original),
                    "unit_count_reduced": len(reduced),
                    "units_original_path": str(original_path.resolve()),
                    "units_reduced_path": str(reduced_path.resolve()),
                    "hubert_model": hubert_model,
                    "hubert_revision": hubert_revision,
                    "hubert_layer": hubert_layer,
                    "kmeans_clusters": clusters,
                    "kmeans_sha256": kmeans_sha256,
                }
            )
            processed += 1
            split_processed += 1
    manifest = output_root / f"manifest.shard-{shard_index:05d}-of-{num_shards:05d}.jsonl"
    atomic_write_jsonl(manifest, records, resume=resume, overwrite=overwrite)
    lock = {
        "hubert_model": hubert_model,
        "hubert_revision": hubert_revision,
        "hubert_layer": hubert_layer,
        "kmeans_clusters": clusters,
        "kmeans_sha256": kmeans_sha256,
        "num_shards": num_shards,
    }
    atomic_write_json(
        output_root / "unit-lock.json", lock, resume=True, overwrite=overwrite
    )
    return {"processed": processed, "manifest": str(manifest), **lock}


def extractor_from_config(config: dict[str, Any], cache_root: Path) -> HubertKMeansExtractor:
    hubert = config["hubert"]
    kmeans = config["kmeans"]
    centroids = cache_root / kmeans["path"]
    expected = kmeans["sha256"]
    actual = sha256_file(centroids)
    if actual.lower() != expected.lower():
        raise ValueError(f"k-means checksum mismatch: {centroids}")
    return HubertKMeansExtractor(
        model_id=hubert["model"],
        revision=hubert["revision"],
        layer=int(config["hubert_layer"]),
        centroids_path=centroids,
        device=str(config.get("device", "cuda")),
    )
