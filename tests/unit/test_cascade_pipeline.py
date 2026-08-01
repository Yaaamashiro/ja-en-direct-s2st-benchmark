from __future__ import annotations

import json
import wave
from pathlib import Path

from direct_s2st.io import read_jsonl
from direct_s2st.cascade.pipeline import run_pipeline


def _wav(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\0\0" * 1600)


def _manifest(path: Path) -> None:
    rows = []
    for pair_id in ("ok", "bad"):
        ja = path.parent / f"{pair_id}-ja.wav"
        en = path.parent / f"{pair_id}-en.wav"
        _wav(ja)
        _wav(en)
        rows.append(
            {
                "pair_id": pair_id,
                "ja_audio": str(ja),
                "en_audio": str(en),
                "en_text": "Reference.",
            }
        )
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_pipeline_keeps_intermediates_and_failures(tmp_path: Path) -> None:
    manifest = tmp_path / "test.jsonl"
    output = tmp_path / "predictions"
    _manifest(manifest)

    def mt(text: str) -> str:
        if text == "bad":
            raise ValueError("fixture failure")
        return "English output."

    def tts(text: str, path: Path) -> None:
        _wav(path)

    result = run_pipeline(
        manifest,
        output,
        run_id="fixture",
        asr=lambda path: path.stem.split("-")[0],
        mt=mt,
        tts=tts,
    )
    assert result["successes"] == 1
    assert result["failures"] == 1
    records = list(read_jsonl(output / "predictions.jsonl"))
    assert records[0]["asr_ja_text"] == "ok"
    assert records[0]["mt_en_text"] == "English output."
    assert records[0]["real_time_factor"] is not None
    assert records[1]["status"] == "failed"
    assert "fixture failure" in records[1]["error"]


def test_resume_reuses_verified_success(tmp_path: Path) -> None:
    manifest = tmp_path / "test.jsonl"
    output = tmp_path / "predictions"
    _manifest(manifest)
    calls = 0

    def tts(text: str, path: Path) -> None:
        nonlocal calls
        calls += 1
        _wav(path)

    run_pipeline(
        manifest,
        output,
        run_id="fixture",
        asr=lambda _: "ok",
        mt=lambda _: "English",
        tts=tts,
        limit=1,
    )
    run_pipeline(
        manifest,
        output,
        run_id="fixture",
        asr=lambda _: "ok",
        mt=lambda _: "English",
        tts=tts,
        limit=1,
        resume=True,
    )
    assert calls == 1
