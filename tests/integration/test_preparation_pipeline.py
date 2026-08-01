from __future__ import annotations

import hashlib
import json
import wave
from pathlib import Path

from direct_s2st.manifests.import_corpus import import_corpus
from direct_s2st.s2ut.extract_units import extract_units
from direct_s2st.s2ut.prepare_fairseq import prepare_fairseq as prepare_s2ut
from direct_s2st.translatotron2.phonemize import phonemize_manifests
from direct_s2st.translatotron2.prepare_fairseq import prepare_fairseq as prepare_t2


def _wav(path: Path) -> tuple[float, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\0\0" * 1600)
    return 0.1, hashlib.sha256(path.read_bytes()).hexdigest()


def test_corpus_to_both_direct_model_manifests(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    accepted = corpus / "production" / "manifests" / "releases" / "accepted.jsonl"
    accepted.parent.mkdir(parents=True)
    rows = []
    for split in ("train", "dev", "test"):
        ja = corpus / "production" / "audio" / "16k" / "ja" / f"{split}.wav"
        en = corpus / "production" / "audio" / "16k" / "en" / f"{split}.wav"
        ja_duration, ja_hash = _wav(ja)
        en_duration, en_hash = _wav(en)
        rows.append(
            {
                "pair_id": f"fixture-{split}",
                "split": split,
                "corpus": "fixture",
                "source_id": f"fixture:{split}",
                "ja_wav_16k": str(ja.relative_to(corpus)),
                "en_wav_16k": str(en.relative_to(corpus)),
                "ja_text": "日本語です。",
                "en_text": "English text.",
                "ja_tts_text": "日本語です。",
                "en_tts_text": "English text.",
                "ja_duration": ja_duration,
                "en_duration": en_duration,
                "ja_sha256": ja_hash,
                "en_sha256": en_hash,
            }
        )
    accepted.write_text("".join(json.dumps(row) + "\n" for row in rows))

    common = tmp_path / "experiment" / "common"
    units = tmp_path / "experiment" / "s2ut" / "units"
    s2ut = tmp_path / "experiment" / "s2ut" / "fairseq"
    phonemes = tmp_path / "experiment" / "translatotron2" / "phonemes"
    t2 = tmp_path / "experiment" / "translatotron2" / "fairseq"

    import_corpus(accepted, corpus_root=corpus, output_root=common)
    extract_units(
        common,
        units,
        extractor=lambda _: [1, 1, 2, 3],
        split=None,
        clusters=100,
        hubert_model="fixture",
        hubert_revision="a" * 40,
        hubert_layer=6,
        kmeans_sha256="b" * 64,
    )
    prepare_s2ut(common, units, s2ut)
    phonemize_manifests(
        common,
        phonemes,
        phonemizer=lambda _: "EH NG | G L IH SH",
        engine="fixture",
        version="1",
    )
    prepare_t2(common, phonemes, t2, mel_config={"n_mels": 80, "hop_length": 160})

    assert (s2ut / "test.tsv").is_file()
    assert (t2 / "config_multitask.yaml").is_file()
