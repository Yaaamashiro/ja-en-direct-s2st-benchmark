from __future__ import annotations

import hashlib
import json
import wave
from pathlib import Path

import pytest

from direct_s2st.manifests.import_corpus import import_corpus
from direct_s2st.manifests.schema import CommonManifestRow
from direct_s2st.manifests.validate import ManifestValidationError, validate_manifest_directory


def _wav(path: Path, frames: int = 1600) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\0\0" * frames)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _accepted_row(root: Path, pair_id: str, split: str) -> dict[str, object]:
    ja = root / "production" / "audio" / "16k" / "ja" / f"{pair_id}.wav"
    en = root / "production" / "audio" / "16k" / "en" / f"{pair_id}.wav"
    ja_hash = _wav(ja)
    en_hash = _wav(en)
    return {
        "pair_id": pair_id,
        "split": split,
        "corpus": "fixture",
        "source_id": f"fixture:{pair_id}",
        "ja_wav_16k": str(ja.relative_to(root)),
        "en_wav_16k": str(en.relative_to(root)),
        "ja_text_raw": "原文",
        "en_text_raw": "Raw text.",
        "ja_text": "日本語です。",
        "en_text": "English text.",
        "ja_tts_text": "日本語です。",
        "en_tts_text": "English text.",
        "ja_duration": 0.1,
        "en_duration": 0.1,
        "ja_sha256": ja_hash,
        "en_sha256": en_hash,
    }


def test_import_preserves_splits_and_creates_lock(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    manifest = corpus / "production" / "manifests" / "releases" / "accepted.jsonl"
    manifest.parent.mkdir(parents=True)
    rows = [_accepted_row(corpus, f"pair-{split}", split) for split in ("train", "dev", "test")]
    manifest.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    output = tmp_path / "benchmark" / "common"
    lock = import_corpus(manifest, corpus_root=corpus, output_root=output)
    assert lock["total_pairs"] == 3
    assert lock["source_manifest_sha256"]
    assert validate_manifest_directory(output)["total_pairs"] == 3
    assert '"split": "test"' in (output / "test.jsonl").read_text(encoding="utf-8")


def test_duplicate_pair_id_across_splits_is_rejected(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    first = _accepted_row(corpus, "same", "train")
    second = {**first, "split": "test"}
    manifest = tmp_path / "accepted.jsonl"
    manifest.write_text(
        json.dumps(first) + "\n" + json.dumps(second) + "\n", encoding="utf-8"
    )
    with pytest.raises(ManifestValidationError, match="duplicate pair_id"):
        import_corpus(manifest, corpus_root=corpus, output_root=tmp_path / "out")


def test_schema_never_uses_asr_text_as_teacher(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    raw = _accepted_row(corpus, "pair", "train")
    raw["ja_asr_text"] = "incorrect"
    row = CommonManifestRow.from_corpus_row(
        raw, corpus_root=corpus, manifest_parent=tmp_path
    )
    assert row.ja_text == "日本語です。"


def test_limit_is_applied_per_original_split(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    manifest = tmp_path / "accepted.jsonl"
    rows = [
        _accepted_row(corpus, f"{split}-{index}", split)
        for split in ("train", "dev", "test")
        for index in range(2)
    ]
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
    lock = import_corpus(
        manifest,
        corpus_root=corpus,
        output_root=tmp_path / "out",
        limit=1,
    )
    assert lock["total_pairs"] == 3
    assert all(value["pairs"] == 1 for value in lock["splits"].values())
