from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from ..io import atomic_write_json


def run_vocoder_command(
    command: list[str],
    *,
    output_root: Path,
    split: str,
    training: bool,
    dry_run: bool = False,
) -> dict[str, Any]:
    if not command or any(not part for part in command):
        raise ValueError("vocoder command must not be empty")
    if training and split != "train":
        raise ValueError("vocoder training may only use the train split")
    result: dict[str, Any] = {"command": command, "split": split}
    if dry_run:
        return result
    completed = subprocess.run(command, check=False)
    result["returncode"] = completed.returncode
    atomic_write_json(output_root / "status.json", result, overwrite=True)
    if completed.returncode:
        raise RuntimeError(f"vocoder command failed with exit code {completed.returncode}")
    return result
