from __future__ import annotations

import hashlib
import json
import wave
from pathlib import Path

import pytest

from direct_s2st.io import atomic_write_json
from direct_s2st.s2ut.extract_units import assign_kmeans_units, extract_units, stable_shard
from direct_s2st.s2ut.prepare_fairseq import prepare_fairseq
from direct_s2st.s2ut.reduce_units import reduce_consecutive_units, validate_units
from direct_s2st.vocoders.runner import run_vocoder_command


def _wav(path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\0\0" * 1600)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _common(root: Path) -> None:
    root.mkdir(parents=True)
    for split in ("train", "dev", "test"):
        ja = root / f"{split}-ja.wav"
        en = root / f"{split}-en.wav"
        _wav(ja)
        _wav(en)
        row = {
            "pair_id": f"pair-{split}",
            "split": split,
            "corpus": "fixture",
            "source_id": f"fixture:{split}",
            "ja_audio": str(ja),
            "en_audio": str(en),
            "ja_text_raw": "日本語",
            "en_text_raw": "English",
            "ja_text": "日本語",
            "en_text": "English",
            "ja_tts_text": "日本語",
            "en_tts_text": "English",
            "ja_duration": 0.1,
            "en_duration": 0.1,
            "ja_sha256": "unused",
            "en_sha256": "unused",
        }
        (root / f"{split}.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    atomic_write_json(root / "dataset-lock.json", {"fixture": True})


def test_reduce_units_and_range_validation() -> None:
    assert reduce_consecutive_units([1, 1, 2, 2, 1]) == [1, 2, 1]
    with pytest.raises(ValueError, match="range"):
        validate_units([100], clusters=100)


def test_stable_sharding_is_deterministic() -> None:
    assert stable_shard("pair", 7) == stable_shard("pair", 7)


def test_kmeans_assignment_matches_fairseq_distance_rule() -> None:
    np = pytest.importorskip("numpy")
    features = np.asarray([[0.1, 0.2], [9.0, 8.0]], dtype=np.float32)
    centers = np.asarray([[0.0, 0.0], [10.0, 10.0]], dtype=np.float32)
    assert assign_kmeans_units(features, centers).tolist() == [0, 1]


def test_extract_and_prepare_fairseq_fixture(tmp_path: Path) -> None:
    common = tmp_path / "common"
    units = tmp_path / "units"
    fairseq = tmp_path / "fairseq"
    _common(common)
    result = extract_units(
        common,
        units,
        extractor=lambda _: [1, 1, 2, 2, 3],
        split=None,
        clusters=100,
        hubert_model="fixture/hubert",
        hubert_revision="a" * 40,
        hubert_layer=6,
        kmeans_sha256="b" * 64,
    )
    assert result["processed"] == 3
    lock = prepare_fairseq(common, units, fairseq)
    assert lock["splits"] == {"train": 1, "dev": 1, "test": 1}
    manifest = (fairseq / "train.tsv").read_text(encoding="utf-8").splitlines()
    assert manifest[0] == "id\tsrc_audio\tsrc_n_frames\ttgt_audio\ttgt_n_frames"
    assert manifest[1].split("\t")[2:] == ["10", "1 2 3", "3"]
    from direct_s2st.s2ut.unit_storage import records, sequence
    assert sequence(records(units)['pair-train'], 'reduced', 100) == [1, 2, 3]
    assert not list(units.rglob('*.units'))


def test_vocoder_training_rejects_non_train_split(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="train split"):
        run_vocoder_command(
            ["fixture"],
            output_root=tmp_path,
            split="test",
            training=True,
            dry_run=True,
        )
