from __future__ import annotations

import hashlib
import json
import wave
from pathlib import Path

import pytest

from direct_s2st.io import atomic_write_json
from direct_s2st.translatotron2.phonemize import (
    normalize_phonemes,
    phonemize_manifests,
    tokenize_espeak_ipa,
)
from direct_s2st.translatotron2.prepare_fairseq import prepare_fairseq


def _wav(path: Path) -> None:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\0\0" * 1600)


def _mel_fixture(_: Path, output: Path, settings: dict[str, object]) -> None:
    np = pytest.importorskip("numpy")
    np.save(output, np.zeros((11, int(settings["n_mels"])), dtype=np.float32))


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
    assert tokenize_espeak_ipa("d\u200dʒˈʌd\u200dʒ") == ["d\u200dʒ", "ʌ", "d\u200dʒ"]
    assert tokenize_espeak_ipa("bˈʌʔn̩") == ["b", "ʌ", "ʔ", "n̩"]


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
        fixed_vocabulary=("DH", "AH", "|", "B", "UH", "K", "X"),
    )
    assert result["processed"] == 3
    lock = prepare_fairseq(
        common,
        phonemes,
        fairseq,
        mel_config={"n_mels": 80, "hop_length": 160},
        feature_extractor=_mel_fixture,
    )
    assert lock["phoneme_vocabulary_source"] == "fixed_espeak_inventory"
    assert (fairseq / "logmelspec80.zip").is_file()
    manifest = (fairseq / "train.tsv").read_text(encoding="utf-8").splitlines()
    assert manifest[0] == "id\tsrc_audio\tsrc_n_frames\ttgt_audio\ttgt_n_frames"
    assert manifest[1].split("\t")[2:] == [
        "10",
        manifest[1].split("\t")[3],
        "11",
    ]
    assert manifest[1].split("\t")[3].startswith("logmelspec80.zip:")
    target_manifest = (fairseq / "target_phoneme" / "train.tsv").read_text()
    assert target_manifest.startswith("id\ttgt_text\n")
    multitask = (fairseq / "config_multitask.yaml").read_text()
    assert (fairseq / "target_phoneme" / "dict.txt").resolve().as_posix() in multitask
    assert (fairseq / "target_phoneme").resolve().as_posix() in multitask
    dictionary = (fairseq / "target_phoneme" / "dict.txt").read_text()
    assert "X 1\n" in dictionary


def test_unknown_dev_phoneme_is_rejected(tmp_path: Path) -> None:
    common = tmp_path / "common"
    phonemes = tmp_path / "phonemes"
    _common(common)
    phonemes.mkdir()
    (phonemes / "train.tsv").write_text("pair-train\tA B\n")
    (phonemes / "dev.tsv").write_text("pair-dev\tA X\n")
    (phonemes / "test.tsv").write_text("pair-test\tA B\n")
    (phonemes / "inventory.txt").write_text("A\nB\n")
    with pytest.raises(ValueError, match="unknown phonemes"):
        prepare_fairseq(
            common,
            phonemes,
            tmp_path / "out",
            mel_config={"n_mels": 80},
        )


def test_espeak_skips_only_unspoken_symbols(monkeypatch, capsys):
    from types import SimpleNamespace
    from direct_s2st.translatotron2 import phonemize as module
    def run(args, **kwargs):
        output = 'eSpeak NG 1.52.0' if '--version' in args else (
            '' if args[-1] in {'⋯', 'broken', '123'} else 'a')
        return SimpleNamespace(stdout=output, stderr='')
    monkeypatch.setattr(module.subprocess, 'run', run)
    engine = module.EspeakNgPhonemizer(expected_version='1.52.0')
    assert engine("Police organization ⋯ prefecture's police department") == 'a | a | a | a | a | a'
    assert "ignored non-spoken symbol='⋯'" in capsys.readouterr().err
    # Pronounced symbols still contribute phones. No leading/trailing empty separator.
    assert engine('⋯ word ⋯') == 'a'
    assert engine('word + word') == 'a | a | a'
    for text in ('broken', '123'):
        with pytest.raises(ValueError, match='empty IPA for word'):
            engine(text)
    with pytest.raises(ValueError, match='no spoken phonemes'):
        engine('⋯')


def test_phonemization_error_reports_exact_pair_and_text(tmp_path):
    common = tmp_path / 'common'
    _common(common)
    def fail(_):
        raise ValueError('empty IPA')
    with pytest.raises(ValueError, match="pair_id='pair-train'.*The book is new"):
        phonemize_manifests(common, tmp_path / 'phones', phonemizer=fail,
                           engine='fixture', version='1', fixed_vocabulary=['a'], resume=True)
