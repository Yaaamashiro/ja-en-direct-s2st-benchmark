from __future__ import annotations

import hashlib
import json
import wave
from pathlib import Path

import pytest

from direct_s2st.io import atomic_write_json
from direct_s2st.translatotron2.phonemize import normalize_phonemes, phonemize_manifests
from direct_s2st.translatotron2.prepare_fairseq import prepare_fairseq


def _wav(path: Path) -> None:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\0\0" * 1600)


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
            "ja_audio": str(ja),
            "en_audio": str(en),
            "en_tts_text": "The book is new.",
        }
        (root / f"{split}.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    atomic_write_json(root / "dataset-lock.json", {"fixture": True})


def test_phoneme_normalization_is_stable() -> None:
    assert normalize_phonemes("  ˈðə   |  bʊk ") == "ðə | bʊk"


def test_phonemize_and_prepare_fixture(tmp_path: Path) -> None:
    common = tmp_path / "common"
    phonemes = tmp_path / "phonemes"
    fairseq = tmp_path / "fairseq"
    _common(common)
    result = phonemize_manifests(
        common,
        phonemes,
        phonemizer=lambda _: "DH AH | B UH K",
        engine="fixture",
        version="1",
    )
    assert result["processed"] == 3
    lock = prepare_fairseq(
        common,
        phonemes,
        fairseq,
        mel_config={"n_mels": 80, "hop_length": 160},
    )
    assert lock["phoneme_vocabulary_source"] == "train"
    assert "target_phoneme" in (fairseq / "config_multitask.yaml").read_text()


def test_unknown_dev_phoneme_is_rejected(tmp_path: Path) -> None:
    common = tmp_path / "common"
    phonemes = tmp_path / "phonemes"
    _common(common)
    phonemes.mkdir()
    (phonemes / "train.tsv").write_text("pair-train\tA B\n")
    (phonemes / "dev.tsv").write_text("pair-dev\tA X\n")
    (phonemes / "test.tsv").write_text("pair-test\tA B\n")
    with pytest.raises(ValueError, match="unknown phonemes"):
        prepare_fairseq(
            common,
            phonemes,
            tmp_path / "out",
            mel_config={"n_mels": 80},
        )
