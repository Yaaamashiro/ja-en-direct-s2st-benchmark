from __future__ import annotations

import os
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


PROFILES = ("smoke", "pilot", "full")
ROOT_VARIABLES = ("CORPUS_ROOT", "EXPERIMENT_DATA_ROOT", "RUNS_ROOT", "CACHE_ROOT")


@dataclass(frozen=True)
class RootPaths:
    corpus: Path
    experiment_data: Path
    runs: Path
    cache: Path

    def validate_output_roots(self) -> None:
        corpus = self.corpus.resolve()
        for root in (self.experiment_data, self.runs, self.cache):
            if root.resolve().is_relative_to(corpus):
                raise ValueError(f"benchmark output root must not be inside CORPUS_ROOT: {root}")

    @classmethod
    def from_environment(cls, environment: dict[str, str] | None = None) -> "RootPaths":
        values = os.environ if environment is None else environment
        missing = [name for name in ROOT_VARIABLES if not values.get(name)]
        if missing:
            raise ValueError(f"missing required environment variables: {', '.join(missing)}")
        return cls(
            corpus=Path(values["CORPUS_ROOT"]).expanduser(),
            experiment_data=Path(values["EXPERIMENT_DATA_ROOT"]).expanduser(),
            runs=Path(values["RUNS_ROOT"]).expanduser(),
            cache=Path(values["CACHE_ROOT"]).expanduser(),
        )


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping")
    return value


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def _validate_revisions(value: Any, path: str = "config") -> None:
    if isinstance(value, dict):
        if "model" in value and isinstance(value["model"], str):
            revision = value.get("revision")
            if not isinstance(revision, str) or not revision.strip():
                raise ValueError(f"{path}.revision is required when {path}.model is set")
        for key, child in value.items():
            _validate_revisions(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _validate_revisions(child, f"{path}[{index}]")


def load_config(path: Path | None, *, profile: str = "smoke") -> dict[str, Any]:
    if profile not in PROFILES:
        raise ValueError(f"profile must be one of: {', '.join(PROFILES)}")
    raw: dict[str, Any] = {}
    if path is not None:
        with path.open("r", encoding="utf-8") as handle:
            raw = _mapping(yaml.safe_load(handle), "config")
        common_path = path.parent.parent / "common" / f"{profile}.yaml"
        if common_path.is_file():
            with common_path.open("r", encoding="utf-8") as handle:
                common = _mapping(yaml.safe_load(handle), f"common profile {profile}")
            raw = _merge(common, raw)
    profiles = _mapping(raw.pop("profiles", {}), "profiles")
    selected = _mapping(profiles.get(profile, {}), f"profiles.{profile}")
    resolved = _merge(raw, selected)
    resolved["profile"] = profile
    _validate_revisions(resolved)
    return resolved
