from __future__ import annotations

import os
from pathlib import Path

import pytest

from direct_s2st.config import load_config
from direct_s2st.s2ut.extract_units import (
    HubertKMeansExtractor,
    assign_kmeans_units,
    load_fairseq_kmeans_centers,
)


@pytest.mark.gpu
def test_hubert_units_match_fairseq_for_the_same_audio() -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    checkpoint_value = os.environ.get("FAIRSEQ_HUBERT_CHECKPOINT")
    kmeans_value = os.environ.get("S2ST_KMEANS_ARTIFACT")
    audio_value = os.environ.get("S2ST_PARITY_AUDIO")
    if not all((checkpoint_value, kmeans_value, audio_value)):
        pytest.skip("fairseq HuBERT checkpoint, k-means artifact, and parity audio are required")

    checkpoint = Path(checkpoint_value)
    kmeans = Path(kmeans_value)
    audio = Path(audio_value)
    for artifact in (checkpoint, kmeans, audio):
        if not artifact.is_file():
            pytest.fail(f"required parity artifact is missing: {artifact}")

    root = Path(__file__).resolve().parents[2]
    config = load_config(root / "configs" / "s2ut" / "prepare.yaml")
    extractor = HubertKMeansExtractor(
        model_id=str(config["hubert"]["model"]),
        revision=str(config["hubert"]["revision"]),
        layer=int(config["hubert_layer"]),
        kmeans_path=kmeans,
        expected_clusters=int(config["kmeans_clusters"]),
    )
    transformers_units = extractor(audio)

    from fairseq import checkpoint_utils
    from fairseq.data.audio.audio_utils import get_features_or_waveform
    import torch.nn.functional as functional

    models, _, task = checkpoint_utils.load_model_ensemble_and_task([str(checkpoint)])
    model = models[0].eval().cuda()
    waveform = get_features_or_waveform(
        str(audio), need_waveform=True, use_sample_rate=task.cfg.sample_rate
    )
    if waveform.ndim == 2:
        waveform = waveform.mean(-1)
    values = torch.from_numpy(waveform).float().cuda()
    if task.cfg.normalize:
        values = functional.layer_norm(values, values.shape)
    with torch.inference_mode():
        features, _ = model.extract_features(
            source=values.view(1, -1),
            padding_mask=None,
            mask=False,
            output_layer=int(config["hubert_layer"]),
        )
    centers = load_fairseq_kmeans_centers(
        kmeans, expected_clusters=int(config["kmeans_clusters"])
    )
    fairseq_units = assign_kmeans_units(
        features.squeeze(0).float().cpu().numpy(), centers
    ).tolist()
    assert transformers_units == fairseq_units
