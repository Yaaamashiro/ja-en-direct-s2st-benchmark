from __future__ import annotations

from pathlib import Path
from typing import Any

from .runs import resolve_command, run_external_experiment, validate_run_id


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
    training = config.get("training")
    if not isinstance(training, dict) or not isinstance(training.get("command"), list):
        raise ValueError("training.command must be an explicit argument list")
    checkpoint = run_root / "checkpoints" / "checkpoint_last.pt"
    command = resolve_command(
        training["command"],
        {"data_root": data_root, "run_root": run_root, "checkpoint_last": checkpoint},
    )
    if resume and "--restore-file" not in command:
        command.extend(["--restore-file", str(checkpoint)])
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
