from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .runs import resolve_command, run_external_experiment, validate_run_id


def build_training_command(
    run_root: Path,
    data_root: Path,
    *,
    config: dict[str, Any],
    resume: bool = False,
) -> list[str]:
    training = config.get("training")
    if not isinstance(training, dict) or not isinstance(training.get("command"), list):
        raise ValueError("training.command must be an explicit argument list")
    max_updates = training.get("max_updates")
    save_interval = training.get("save_interval_updates")
    if not isinstance(max_updates, int) or max_updates < 1:
        raise ValueError("training.max_updates must be a positive integer")
    if not isinstance(save_interval, int) or save_interval < 1:
        raise ValueError("training.save_interval_updates must be a positive integer")
    checkpoint = run_root / "checkpoints" / "checkpoint_last.pt"
    command = resolve_command(
        training["command"],
        {
            "data_root": data_root,
            "run_root": run_root,
            "checkpoint_last": checkpoint,
            "max_updates": str(max_updates),
            "save_interval_updates": str(save_interval),
        },
    )
    if resume and "--restore-file" not in command:
        command.extend(["--restore-file", str(checkpoint)])
    return command


def train_system(
    repository_root: Path,
    run_root: Path,
    data_root: Path,
    *,
    run_id: str,
    config: dict[str, Any],
    profile: str,
    resume: bool = False,
    overwrite: bool = False,
) -> dict[str, Any]:
    validate_run_id(run_id)
    if profile == "full" and config.get("confirm_full") is not True:
        raise ValueError("full training requires confirm_full: true in the explicit config")
    if profile == "full" and os.environ.get("S2ST_EXECUTION_ENV") != "docker":
        raise RuntimeError("full training must run inside the production Docker environment")
    if profile == "full" and not os.environ.get("S2ST_DOCKER_IMAGE_DIGEST"):
        raise RuntimeError("full training requires S2ST_DOCKER_IMAGE_DIGEST")
    command = build_training_command(
        run_root,
        data_root,
        config=config,
        resume=resume,
    )
    return run_external_experiment(
        repository_root,
        run_root,
        command=command,
        config=config,
        dataset_lock=data_root / "data-lock.json",
        seed=int(config.get("seed", 1)),
        resume=resume,
        overwrite=overwrite,
        require_checkpoint_for_resume=True,
    )
