from __future__ import annotations

import json
import os
import platform
import re
import socket
import subprocess
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from .hashing import sha256_file
from .io import ExistingOutputError, atomic_write_json, atomic_write_text


RUN_ID = re.compile(r"^[a-z0-9][a-z0-9._-]{2,127}$")


def make_run_id(system: str, variant: str, seed: int, day: date | None = None) -> str:
    value = f"{system}-{variant}-seed{seed}-{(day or date.today()):%Y%m%d}".lower()
    value = re.sub(r"[^a-z0-9._-]+", "-", value).strip("-")
    if not RUN_ID.fullmatch(value):
        raise ValueError(f"invalid generated run_id: {value!r}")
    return value


def validate_run_id(value: str) -> str:
    if not RUN_ID.fullmatch(value):
        raise ValueError("run_id must contain only lowercase letters, numbers, '.', '_' or '-'")
    return value


def collect_model_revisions(config: Any, path: str = "config") -> dict[str, str]:
    result: dict[str, str] = {}
    if isinstance(config, dict):
        if isinstance(config.get("model"), str) and isinstance(config.get("revision"), str):
            result[path] = f"{config['model']}@{config['revision']}"
        for key, value in config.items():
            result.update(collect_model_revisions(value, f"{path}.{key}"))
    elif isinstance(config, list):
        for index, value in enumerate(config):
            result.update(collect_model_revisions(value, f"{path}[{index}]"))
    return result


def _git_revision(path: Path) -> str | None:
    completed = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def environment_metadata(
    repository_root: Path,
    *,
    config: dict[str, Any],
    dataset_lock: Path,
    seed: int,
) -> dict[str, Any]:
    torch_values: dict[str, Any] = {"torch": None, "cuda": None, "gpu": None, "gpu_vram": None}
    try:
        import torch

        torch_values["torch"] = torch.__version__
        torch_values["cuda"] = torch.version.cuda
        if torch.cuda.is_available():
            properties = torch.cuda.get_device_properties(0)
            torch_values["gpu"] = properties.name
            torch_values["gpu_vram"] = properties.total_memory
    except ImportError:
        pass
    fairseq_root = repository_root / "third_party" / "fairseq"
    return {
        "git_commit": _git_revision(repository_root),
        "fairseq_commit": _git_revision(fairseq_root) if fairseq_root.exists() else None,
        "docker_image_digest": os.environ.get("S2ST_DOCKER_IMAGE_DIGEST"),
        "python": platform.python_version(),
        **torch_values,
        "hostname": socket.gethostname(),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "model_revisions": collect_model_revisions(config),
        "dataset_manifest_sha256": sha256_file(dataset_lock),
    }


def resolve_command(command: list[str], values: dict[str, Path | str]) -> list[str]:
    if not command:
        raise ValueError("command must be a non-empty list")
    return [part.format_map({key: str(value) for key, value in values.items()}) for part in command]


def redact_command(command: list[str]) -> list[str]:
    redacted = list(command)
    hide_next = False
    for index, value in enumerate(redacted):
        lowered = value.lower()
        if hide_next:
            redacted[index] = "<redacted>"
            hide_next = False
        elif any(marker in lowered for marker in ("token", "password", "secret", "api-key", "api_key")):
            if "=" in value:
                redacted[index] = value.split("=", 1)[0] + "=<redacted>"
            else:
                hide_next = True
    return redacted


def run_external_experiment(
    repository_root: Path,
    run_root: Path,
    *,
    command: list[str],
    config: dict[str, Any],
    dataset_lock: Path,
    seed: int,
    resume: bool = False,
    overwrite: bool = False,
    require_checkpoint_for_resume: bool = False,
) -> dict[str, Any]:
    status_path = run_root / "status.json"
    if run_root.exists() and any(run_root.iterdir()) and not (resume or overwrite):
        raise ExistingOutputError(f"run already exists: {run_root}")
    checkpoint = run_root / "checkpoints" / "checkpoint_last.pt"
    if resume and require_checkpoint_for_resume and not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint_last is required for resume: {checkpoint}")
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "logs").mkdir(exist_ok=True)
    (run_root / "checkpoints").mkdir(exist_ok=True)
    atomic_write_text(
        run_root / "resolved-config.yaml",
        yaml.safe_dump(config, allow_unicode=True, sort_keys=True),
        overwrite=True,
    )
    atomic_write_json(
        run_root / "environment.json",
        environment_metadata(
            repository_root, config=config, dataset_lock=dataset_lock, seed=seed
        ),
        overwrite=True,
    )
    atomic_write_text(
        run_root / "dataset-lock.json",
        dataset_lock.read_text(encoding="utf-8"),
        overwrite=True,
    )
    atomic_write_text(
        run_root / "command.txt",
        subprocess.list2cmdline(redact_command(command)) + "\n",
        overwrite=True,
    )
    atomic_write_json(status_path, {"status": "running"}, overwrite=True)
    log_path = run_root / "logs" / "process.log"
    with log_path.open("a" if resume else "w", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            cwd=repository_root,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    result = {"status": "completed" if completed.returncode == 0 else "failed", "returncode": completed.returncode}
    atomic_write_json(status_path, result, overwrite=True)
    if completed.returncode:
        raise RuntimeError(f"experiment command failed with exit code {completed.returncode}")
    return result
