from __future__ import annotations

from pathlib import Path
import os
import sys
import time
from ..progress import operation, track
from ..manifests.reader import read_common_manifest
from typing import Any, Protocol

from ..hashing import sha256_file, sha256_text
from ..journal import digest
from ..preparation import Checkpoints, adaptive_map, file_stamp
from ..io import atomic_write_json, atomic_write_jsonl, atomic_write_text
from .reduce_units import reduce_consecutive_units, validate_units


class UnitExtractor(Protocol):
    def __call__(self, audio_path: Path) -> list[int]: ...


def retain_hubert_layers(model, layer):
    """Later encoder layers cannot affect the requested earlier hidden state."""
    if not 1 <= layer <= len(model.encoder.layers):
        raise ValueError('HuBERT extraction layer outside the pretrained encoder')
    model.encoder.layers = model.encoder.layers[:layer]
    return model


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
    """Lazy Transformers HuBERT extractor with fairseq joblib k-means."""

    def __init__(
        self,
        *,
        model_id: str,
        revision: str,
        layer: int,
        kmeans_path: Path,
        expected_clusters: int,
        device: str = "cuda",
    ) -> None:
        import numpy as np
        import torch
        from transformers import AutoFeatureExtractor, HubertModel

        self.np = np
        self.torch = torch
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.processor = AutoFeatureExtractor.from_pretrained(model_id, revision=revision)
        self.model = retain_hubert_layers(HubertModel.from_pretrained(model_id, revision=revision), layer).to(self.device)
        self.model.eval()
        self.layer = layer
        self.centers = load_fairseq_kmeans_centers(
            kmeans_path, expected_clusters=expected_clusters
        )

    def __call__(self, audio_path: Path) -> list[int]:
        return self.extract_many([audio_path])[0]

    @staticmethod
    def _read_audio(audio_path):
        import soundfile as sf
        from ..drive_staging import local_path
        audio_path = local_path(audio_path)
        waveform, sample_rate = sf.read(audio_path, dtype="float32", always_2d=False)
        if sample_rate != 16000 or waveform.ndim != 1:
            raise ValueError(f"HuBERT input must be 16 kHz mono: {audio_path}")
        return waveform

    def _batch(self, waves):
        # No padding: HuBERT's group-normalized convolution changes with padding.
        inputs = self.processor(waves, sampling_rate=16000, return_tensors='pt', padding=False)
        with self.torch.inference_mode():
            output = self.model(inputs.input_values.to(self.device), output_hidden_states=True)
        features = output.hidden_states[self.layer].float().cpu().numpy()
        return [assign_kmeans_units(feature, self.centers).astype(int).tolist() for feature in features]

    def extract_many(self, paths):
        maximum = int(os.environ.get('S2ST_HUBERT_BATCH_MAX', '8'))
        if not 1 <= maximum <= 32:
            raise ValueError('S2ST_HUBERT_BATCH_MAX must be between 1 and 32')
        waves = list(adaptive_map(self._read_audio, paths))
        groups = {}
        for index, waveform in enumerate(waves):
            groups.setdefault(len(waveform), []).append(index)
        results = [None] * len(waves)
        batch_size = min(getattr(self, '_batch_size', 1), maximum)
        for indices in groups.values():
            offset = 0
            while offset < len(indices):
                chosen = indices[offset:offset + batch_size]
                started = time.monotonic()
                try:
                    values = self._batch([waves[index] for index in chosen])
                except self.torch.cuda.OutOfMemoryError:
                    if len(chosen) == 1:
                        raise
                    batch_size = max(1, len(chosen) // 2)
                    self.torch.cuda.empty_cache()
                    print(f'[adaptive] HuBERT OOM: retry batch={batch_size}', file=sys.stderr, flush=True)
                    continue
                for index, value in zip(chosen, values):
                    results[index] = value
                offset += len(chosen)
                rate = len(chosen) / max(time.monotonic() - started, .001)
                if self.device.type == 'cuda':
                    free, total = self.torch.cuda.mem_get_info(self.device)
                    rates = getattr(self, '_batch_rates', {})
                    previous = rates.get(len(waves[chosen[0]]))
                    slower = (previous is not None and len(chosen) > previous[0]
                              and rate < previous[1] * .85)
                    if free / total < .15 or slower:
                        batch_size = max(1, batch_size // 2)
                    elif free / total > .3:
                        batch_size = min(maximum, batch_size + 1)
                    rates[len(waves[chosen[0]])] = (len(chosen), rate)
                    self._batch_rates = rates
                    now = time.monotonic()
                    if now - getattr(self, '_last_report', float('-inf')) >= 10:
                        print(f'[adaptive] HuBERT actual_batch={len(chosen)} next_batch={batch_size} '
                              f'items_per_s={rate:.3f} free_vram={free}/{total} '
                              f'lookahead={len(waves)} unique_lengths={len(groups)} '
                              'padding=false', file=sys.stderr, flush=True)
                        self._last_report = now
        self._batch_size = batch_size
        return results


def _recoverable_units(rows, extractor, cache, clusters):
    rows = iter(rows)
    maximum = int(os.environ.get('S2ST_HUBERT_BATCH_MAX', '8'))
    if not 1 <= maximum <= 32:
        raise ValueError('S2ST_HUBERT_BATCH_MAX must be between 1 and 32')
    # Larger lookahead groups identical lengths without changing output order or
    # introducing padding. Bound estimated float32 audio storage and row count.
    def windows():
        pending = None
        while True:
            window, estimate = [], 0
            while len(window) < 256:
                row = pending if pending is not None else next(rows, None)
                pending = None
                if row is None:
                    break
                size = max(1., float(row.get('en_duration', 1.))) * 16000 * 4
                if window and estimate + size > 128 * 1024**2:
                    pending = row
                    break
                window.append(row)
                estimate += size
            if not window:
                return
            yield window
    for window in windows():
        keys = [digest([row, file_stamp(row['en_audio'])]) for row in window]
        values = [cache.get(key) for key in keys]
        missing = [index for index, value in enumerate(values) if value is None]
        paths = [Path(window[index]['en_audio']) for index in missing]
        extracted = ((extractor.extract_many(paths) if hasattr(extractor, 'extract_many')
                      else map(extractor, paths)) if paths else ())
        for index, units in zip(missing, extracted):
            values[index] = validate_units(units, clusters=clusters)
            cache.record(keys[index], values[index])
        for row, value in zip(window, values):
            if value is None:
                raise ValueError('extractor returned too few results')
            yield row, validate_units(value, clusters=clusters)


def load_fairseq_kmeans_centers(path: Path, *, expected_clusters: int) -> Any:
    import joblib
    import numpy as np

    model = joblib.load(path)
    if not hasattr(model, "cluster_centers_"):
        raise ValueError("fairseq k-means artifact must expose cluster_centers_")
    centers = np.asarray(model.cluster_centers_, dtype=np.float32)
    if centers.ndim != 2:
        raise ValueError("k-means centers must be a [clusters, features] array")
    if centers.shape[0] != expected_clusters:
        raise ValueError(
            f"k-means cluster count mismatch: expected {expected_clusters}, got {centers.shape[0]}"
        )
    return centers


def assign_kmeans_units(features: Any, centers: Any) -> Any:
    """Match fairseq ApplyKmeans' squared-Euclidean nearest-center rule."""

    if features.ndim != 2 or centers.ndim != 2:
        raise ValueError("features and centers must both be rank-2 arrays")
    if features.shape[1] != centers.shape[1]:
        raise ValueError("HuBERT feature width does not match k-means centers")
    distances = (
        (features**2).sum(axis=1, keepdims=True)
        - 2 * features @ centers.T
        + (centers**2).sum(axis=1)
    )
    return distances.argmin(axis=1)


@operation('s2ut/extract_units: extract_units')
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
    kmeans_artifact: str | None = None,
    shard_index: int = 0,
    num_shards: int = 1,
    limit: int | None = None,
    resume: bool = False,
    overwrite: bool = False,
    storage: str = 'inline',
) -> dict[str, Any]:
    if storage not in ('inline', 'files'):
        raise ValueError('unit storage must be inline or files')
    splits = (split,) if split else ("train", "dev", "test")
    records: list[dict[str, Any]] = []
    processed = 0
    lock = dict(hubert_model=hubert_model, hubert_revision=hubert_revision,
                hubert_layer=hubert_layer, kmeans_clusters=clusters,
                kmeans_sha256=kmeans_sha256, kmeans_artifact=kmeans_artifact, num_shards=num_shards)
    # Publish identity BEFORE outputs, so partial runs cannot mix model revisions.
    atomic_write_json(output_root / 'unit-lock.json', lock, resume=True, overwrite=overwrite)
    with Checkpoints(output_root / '.checkpoints', dict(stage='units-v1', **lock),
                     resume=resume and not overwrite, overwrite=overwrite) as cache:
        for current_split in splits:
            selected = [row for row in read_common_manifest(common_root / f"{current_split}.jsonl", parallel=True)
                        if stable_shard(str(row['pair_id']), num_shards) == shard_index]
            if limit is not None:
                selected = selected[:limit]
            for row, original in track(_recoverable_units(selected, extractor, cache, clusters),
                                       f'units: {current_split} (extract/reuse)', total=len(selected)):
                pair_id = str(row['pair_id'])
                original_path = output_root / current_split / "original" / f"{pair_id}.units"
                reduced_path = output_root / current_split / "reduced" / f"{pair_id}.units"
                reduced = reduce_consecutive_units(original)
                if storage == 'files':
                    atomic_write_text(original_path, serialize_units(original), resume=resume, overwrite=overwrite)
                    atomic_write_text(reduced_path, serialize_units(reduced), resume=resume, overwrite=overwrite)
                records.append(
                    {
                        "pair_id": pair_id,
                        "split": current_split,
                        "unit_count_original": len(original),
                        "unit_count_reduced": len(reduced),
                        "hubert_model": hubert_model,
                        "hubert_revision": hubert_revision,
                        "hubert_layer": hubert_layer,
                        "kmeans_clusters": clusters,
                        "kmeans_sha256": kmeans_sha256,
                        "kmeans_artifact": kmeans_artifact,
                    }
                )
                if storage == 'inline':
                    record = records[-1]
                    record.update(units_storage='inline-v1', units_original=original,
                                  units_reduced=reduced,
                                  units_sha256=digest(dict(original=original, reduced=reduced)))
                else:
                    records[-1].update(units_original_path=str(original_path.resolve()),
                                       units_reduced_path=str(reduced_path.resolve()))
                processed += 1
    manifest = output_root / f"manifest.shard-{shard_index:05d}-of-{num_shards:05d}.jsonl"
    if storage == 'inline' and manifest.is_file():
        from ..io import read_jsonl
        if any(row.get('units_storage') != 'inline-v1' for row in read_jsonl(manifest)):
            manifest = output_root / manifest.name.replace('manifest.', 'manifest.inline.')
    atomic_write_jsonl(manifest, records, resume=resume, overwrite=overwrite)
    return {"processed": processed, "manifest": str(manifest), **lock}


@operation('s2ut/extract_units: extractor_from_config')
def extractor_from_config(config: dict[str, Any], cache_root: Path) -> HubertKMeansExtractor:
    hubert = config["hubert"]
    kmeans = config["kmeans"]
    if kmeans.get("format") != "fairseq-joblib":
        raise ValueError("kmeans.format must be fairseq-joblib")
    artifact = cache_root / kmeans["path"]
    expected = kmeans["sha256"]
    actual = sha256_file(artifact)
    if actual.lower() != expected.lower():
        raise ValueError(f"k-means checksum mismatch: {artifact}")
    return HubertKMeansExtractor(
        model_id=hubert["model"],
        revision=hubert["revision"],
        layer=int(config["hubert_layer"]),
        kmeans_path=artifact,
        expected_clusters=int(config["kmeans_clusters"]),
        device=str(config.get("device", "cuda")),
    )
