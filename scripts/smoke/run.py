"""Configurable Docker smoke driver. Planning is default; --execute runs real stages."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY / "src"))

from direct_s2st.config import load_config
from direct_s2st.io import atomic_write_json, atomic_write_text
from direct_s2st.runs import validate_run_id
import yaml


def build_plan(run_id: str, limit: int, updates: int, env_file: Path,
               docker_context: str | None = None) -> tuple[dict, list[dict]]:
    validate_run_id(run_id)
    if not 1 <= limit <= 100 or not 1 <= updates <= 10:
        raise ValueError("smoke permits 1..100 rows per split and 1..10 updates")
    settings = {}
    for name, source in {"train": "s2ut/train.yaml", "infer": "s2ut/infer.yaml",
                         "unit": "vocoder/unit.yaml", "evaluation": "evaluation/default.yaml"}.items():
        cfg = load_config(REPOSITORY / "configs" / source, profile="smoke")
        cfg["run_id"] = run_id
        if name == "train":
            cfg["training"]["max_updates"] = updates
            cfg["training"]["save_interval_updates"] = 1
        settings[name] = cfg
    prefix = ["docker"]
    if docker_context:
        prefix += ["--context", docker_context]
    prefix += ["compose", "--env-file", str(env_file.resolve())]
    config_root = f"/workspace/configs/local/{run_id}"
    specs = [
        ("common_import", "common", ["corpus", "import", "--limit", str(limit)]),
        ("common_validate", "common", ["corpus", "validate"]),
        ("unit_artifact", "fairseq", ["s2ut", "fetch-artifacts"]),
        ("unit_extraction", "fairseq", ["s2ut", "extract-units", "--limit", str(limit), "--resume"]),
        ("multitask_preparation", "fairseq", ["s2ut", "prepare"]),
        ("multitask_validation", "fairseq", ["s2ut", "validate"]),
        ("training", "fairseq", ["s2ut", "train", "--config", f"{config_root}/train.yaml"]),
        ("unit_inference", "fairseq", ["s2ut", "infer", "--config", f"{config_root}/infer.yaml", "--resume"]),
        ("unit_vocoder", "fairseq", ["vocoder", "unit", "infer", "--config", f"{config_root}/unit.yaml"]),
        ("evaluation", "evaluation", ["evaluate", "run", "--config", f"{config_root}/evaluation.yaml"]),
    ]
    return settings, [{"stage": name, "status": "NOT_RUN", "command": prefix + ["run", "--rm", service] + args + ["--profile", "smoke"]}
                      for name, service, args in specs]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--run-id", default="s2ut-smoke")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--max-updates", type=int, default=2)
    parser.add_argument("--docker-context")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    settings, stages = build_plan(args.run_id, args.limit, args.max_updates, args.env_file, args.docker_context)
    if not args.execute:
        print(json.dumps({"status": "PLAN_ONLY", "stages": stages}, indent=2))
        return 0
    if not args.env_file.is_file():
        raise FileNotFoundError(args.env_file)
    if shutil.which("docker") is None:
        raise RuntimeError("Docker is unavailable; configure the execution environment first")
    root = REPOSITORY / "configs" / "local" / args.run_id
    for name, cfg in settings.items():
        atomic_write_text(root / f"{name}.yaml", yaml.safe_dump(cfg, allow_unicode=True), resume=True)
    report = root / "execution.json"
    if report.exists():
        raise FileExistsError("choose a new run-id/environment data root; an execution report already exists")
    atomic_write_json(report, {"status": "RUNNING", "stages": stages})
    for stage in stages:
        start = time.perf_counter()
        try:
            completed = subprocess.run(stage["command"], cwd=REPOSITORY, check=False)
            stage["returncode"] = completed.returncode
            stage["status"] = "PASS" if completed.returncode == 0 else "FAIL"
        except OSError as error:
            stage.update(status="FAIL", error=str(error))
        stage["seconds"] = time.perf_counter() - start
        atomic_write_json(report, {"status": "RUNNING", "stages": stages}, overwrite=True)
        if stage["status"] != "PASS":
            atomic_write_json(report, {"status": "FAIL", "stages": stages}, overwrite=True)
            return 1
    # Stage command completion is necessary but not a research-quality claim.
    atomic_write_json(report, {"status": "STAGES_COMPLETED_REVIEW_METRICS", "stages": stages}, overwrite=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
