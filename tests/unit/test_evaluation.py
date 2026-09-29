from __future__ import annotations

import json
import wave
from pathlib import Path

from direct_s2st.evaluation.aggregate import aggregate_runs
from direct_s2st.evaluation.bleu import normalize_english
from direct_s2st.evaluation.run import evaluate_predictions
from direct_s2st.evaluation.runtime import audio_quality
from direct_s2st.io import read_jsonl


def _wav(path: Path, sample: int = 0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(int(sample).to_bytes(2, "little", signed=True) * 1600)


def _prediction(pair_id: str, output: Path | None, status: str) -> dict[str, object]:
    return {
        "pair_id": pair_id,
        "system_id": "s2ut",
        "run_id": "fixture-run",
        "source_audio": str(output or "source.wav"),
        "reference_audio": str(output or "reference.wav"),
        "reference_text": "The book is new.",
        "output_audio": str(output) if output else None,
        "output_duration": 0.1 if output else None,
        "inference_seconds": 0.05,
        "real_time_factor": 0.5 if output else None,
        "status": status,
        "error": None if status == "success" else "generation failed",
    }


def test_normalization_and_quality() -> None:
    assert normalize_english("  Hello,  WORLD! ") == "hello world"


def test_evaluation_keeps_generation_failures(tmp_path: Path) -> None:
    audio = tmp_path / "output.wav"
    _wav(audio)
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text(
        json.dumps(_prediction("ok", audio, "success"))
        + "\n"
        + json.dumps(_prediction("bad", None, "failed"))
        + "\n"
    )
    output = tmp_path / "metrics"

    def unavailable_speaker(first: Path, second: Path) -> float:
        raise RuntimeError("speaker model unavailable")

    metrics = evaluate_predictions(
        predictions,
        output,
        transcribe=lambda _: "The book is new.",
        speaker_similarity=unavailable_speaker,
    )
    assert metrics["samples"] == 2
    assert metrics["generation_failures"] == 1
    assert metrics["bleu"] == 100.0
    rows = list(read_jsonl(output / "per_sample.jsonl"))
    assert len(rows) == 2
    assert rows[1]["evaluation_error"] == "generation failed"
    assert rows[0]["evaluation_status"] == "success"
    assert "speaker model unavailable" in rows[0]["speaker_similarity_error"]
    assert audio_quality(audio)["silence_ratio"] == 1.0


def test_aggregate_writes_comparison_report(tmp_path: Path) -> None:
    run = tmp_path / "run"
    metrics = run / "metrics"
    metrics.mkdir(parents=True)
    (metrics / "metrics.json").write_text(
        json.dumps(
            {
                "system_id": "s2ut",
                "run_id": "fixture",
                "bleu": 10.0,
                "blaser": None,
                "speaker_similarity_source": None,
                "speaker_similarity_reference": None,
                "mean_rtf": 0.5,
                "failure_rate": 0.0,
            }
        )
    )
    (metrics / "per_sample.jsonl").write_text('{"pair_id":"one"}\n')
    output = tmp_path / "results"
    aggregate_runs([run], output)
    assert "s2ut" in (output / "comparison.csv").read_text()
    assert "# S2ST comparison" in (output / "report.md").read_text()
    (output / 'report.md').unlink()
    aggregate_runs([run], output, resume=True)
    assert (output / 'report.md').is_file()
    aggregate_runs([run], output, resume=True)


def test_resume_retries_failed_optional_metric_then_reuses_success(tmp_path):
    audio = tmp_path / 'output.wav'
    _wav(audio)
    predictions = tmp_path / 'predictions.jsonl'
    predictions.write_text(json.dumps(_prediction('ok', audio, 'success')) + '\n')
    calls = []
    def metric(*args):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError('temporary failure')
        return .9
    kwargs = dict(transcribe=lambda _: 'The book is new.', blaser=metric, resume=True)
    output = tmp_path / 'metrics'
    assert evaluate_predictions(predictions, output, **kwargs)['blaser'] is None
    assert evaluate_predictions(predictions, output, **kwargs)['blaser'] == .9
    assert evaluate_predictions(predictions, output, **kwargs)['blaser'] == .9
    assert len(calls) == 2
