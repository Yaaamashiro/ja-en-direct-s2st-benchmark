from __future__ import annotations

import json
import sys
import wave
from datetime import date
from pathlib import Path

import pytest

from direct_s2st.predictions import Prediction, validate_predictions
from direct_s2st.runs import make_run_id, redact_command, run_external_experiment
from direct_s2st.training import build_training_command
from direct_s2st.training import train_system


def _lock(path: Path) -> None:
    path.write_text('{"sha": "fixture"}\n', encoding="utf-8")


def _wav(path: Path) -> None:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\0\0" * 10)


def test_run_id_generation() -> None:
    assert make_run_id("s2ut", "reduced", 1, date(2026, 8, 2)) == "s2ut-reduced-seed1-20260802"


def test_command_redaction() -> None:
    assert redact_command(["tool", "--token", "secret"])[2] == "<redacted>"
    assert redact_command(["tool", "--api-key=value"])[1] == "--api-key=<redacted>"


def test_training_command_uses_resolved_profile_values(tmp_path: Path) -> None:
    command = build_training_command(
        tmp_path / "run",
        tmp_path / "data",
        config={
            "training": {
                "max_updates": 10_000,
                "save_interval_updates": 1_000,
                "command": [
                    "train",
                    "--max-update",
                    "{max_updates}",
                    "--save-interval-updates",
                    "{save_interval_updates}",
                ],
            }
        },
    )
    assert command[-3:] == ["10000", "--save-interval-updates", "1000"]


def test_full_training_requires_docker_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("S2ST_EXECUTION_ENV", raising=False)
    monkeypatch.delenv("S2ST_DOCKER_IMAGE_DIGEST", raising=False)
    with pytest.raises(RuntimeError, match="production Docker"):
        train_system(
            tmp_path,
            tmp_path / "run",
            tmp_path / "data",
            run_id="full-run",
            config={"confirm_full": True},
            profile="full",
        )


def test_full_training_requires_image_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("S2ST_EXECUTION_ENV", "docker")
    monkeypatch.delenv("S2ST_DOCKER_IMAGE_DIGEST", raising=False)
    with pytest.raises(RuntimeError, match="S2ST_DOCKER_IMAGE_DIGEST"):
        train_system(
            tmp_path,
            tmp_path / "run",
            tmp_path / "data",
            run_id="full-run",
            config={"confirm_full": True},
            profile="full",
        )


def test_external_run_records_environment_and_status(tmp_path: Path) -> None:
    data = tmp_path / "data-lock.json"
    _lock(data)
    run = tmp_path / "run"
    code = "from pathlib import Path; Path(r'%s').write_text('ok')" % (run / "done.txt")
    result = run_external_experiment(
        tmp_path,
        run,
        command=[sys.executable, "-c", code],
        config={"seed": 1},
        dataset_lock=data,
        seed=1,
    )
    assert result["status"] == "completed"
    assert json.loads((run / "status.json").read_text())["returncode"] == 0
    assert (run / "environment.json").is_file()


def test_prediction_schema_retains_failed_samples(tmp_path: Path) -> None:
    audio = tmp_path / "output.wav"
    _wav(audio)
    success = {
        "pair_id": "ok",
        "system_id": "s2ut",
        "run_id": "run",
        "source_audio": "source.wav",
        "reference_audio": "reference.wav",
        "reference_text": "Reference",
        "output_audio": str(audio),
        "output_duration": 1.0,
        "inference_seconds": 0.2,
        "real_time_factor": 0.2,
        "status": "success",
        "error": None,
    }
    failed = {
        **success,
        "pair_id": "failed",
        "output_audio": None,
        "output_duration": None,
        "real_time_factor": None,
        "status": "failed",
        "error": "fixture",
    }
    path = tmp_path / "predictions.jsonl"
    path.write_text(json.dumps(success) + "\n" + json.dumps(failed) + "\n")
    assert validate_predictions(path) == {"samples": 2, "successes": 1, "failures": 1}


def test_success_requires_audio(tmp_path: Path) -> None:
    value = {
        "pair_id": "missing",
        "system_id": "cascade",
        "run_id": "run",
        "source_audio": "source.wav",
        "reference_audio": "reference.wav",
        "reference_text": "Reference",
        "output_audio": str(tmp_path / "missing.wav"),
        "output_duration": 1.0,
        "inference_seconds": 1.0,
        "real_time_factor": 1.0,
        "status": "success",
        "error": None,
    }
    with pytest.raises(ValueError, match="no output audio"):
        Prediction.from_dict(value)
