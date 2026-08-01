from __future__ import annotations

from pathlib import Path
from typing import Any

from .predictions import validate_predictions
from .runs import resolve_command, run_external_experiment, validate_run_id


def infer_system(
    repository_root: Path,
    run_root: Path,
    data_root: Path,
    *,
    run_id: str,
    config: dict[str, Any],
    resume: bool = False,
    overwrite: bool = False,
) -> dict[str, Any]:
    validate_run_id(run_id)
    inference = config.get("inference")
    if not isinstance(inference, dict) or not isinstance(inference.get("command"), list):
        raise ValueError("inference.command must be an explicit argument list")
    predictions = run_root / "predictions" / "predictions.jsonl"
    command = resolve_command(
        inference["command"],
        {"data_root": data_root, "run_root": run_root, "predictions": predictions},
    )
    result = run_external_experiment(
        repository_root,
        run_root,
        command=command,
        config=config,
        dataset_lock=data_root / "data-lock.json",
        seed=int(config.get("seed", 1)),
        resume=resume,
        overwrite=overwrite,
    )
    return {**result, **validate_predictions(predictions)}
