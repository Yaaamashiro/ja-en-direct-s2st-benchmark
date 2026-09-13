"""Opt-in tests against real checkpoints. No download or fabricated weights."""
import json
import os
from pathlib import Path

import pytest

from direct_s2st.io import atomic_write_jsonl
from direct_s2st.predictions import validate_predictions
from direct_s2st.vocoders.inference import vocode

pytestmark = pytest.mark.gpu


def required_path(name):
    value = os.environ.get(name)
    if not value:
        pytest.skip(f"configure {name} for real-checkpoint test")
    path = Path(value)
    assert path.is_file(), f"configured artifact does not exist: {path}"
    return path


def test_real_unit_vocoder(tmp_path):
    checkpoint = required_path("S2ST_UNIT_VOCODER_CHECKPOINT")
    config = required_path("S2ST_UNIT_VOCODER_CONFIG")
    inputs = tmp_path / "units.jsonl"
    atomic_write_jsonl(inputs, [{"pair_id": "unit-fixture", "system_id": "s2ut", "run_id": "vocoder-test",
                                "source_audio": "unused", "reference_audio": "unused", "reference_text": "fixture",
                                "units": [10, 20, 35, 18, 42, 50, 60, 7], "inference_seconds": 0}])
    vocode("unit", inputs, tmp_path / "out", checkpoint, config, sample_rate=16000,
           device=os.environ.get("S2ST_TEST_DEVICE", "cuda"))
    assert validate_predictions(tmp_path / "out/predictions.jsonl")["successes"] == 1


def test_real_mel_vocoder(tmp_path):
    checkpoint = required_path("S2ST_MEL_VOCODER_CHECKPOINT")
    config = required_path("S2ST_MEL_VOCODER_CONFIG")
    feature = required_path("S2ST_MEL_FIXTURE")
    spec = json.loads(required_path("S2ST_MEL_SPEC").read_text())
    inputs = tmp_path / "mels.jsonl"
    atomic_write_jsonl(inputs, [{"pair_id": "mel-fixture", "system_id": "translatotron2", "run_id": "vocoder-test",
                                "source_audio": "unused", "reference_audio": "unused", "reference_text": "fixture",
                                "mel_path": str(feature), "inference_seconds": 0}])
    vocode("mel", inputs, tmp_path / "out", checkpoint, config, sample_rate=spec["sample_rate"],
           mel=spec, device=os.environ.get("S2ST_TEST_DEVICE", "cuda"))
    assert validate_predictions(tmp_path / "out/predictions.jsonl")["successes"] == 1
