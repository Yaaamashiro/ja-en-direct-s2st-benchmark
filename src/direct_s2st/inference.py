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
    if config.get("implementation_status") == "not_implemented":
        raise NotImplementedError('configured inference architecture is not implemented')
    inference = config.get("inference")
    if not isinstance(inference, dict) or not isinstance(inference.get("command"), list):
        raise ValueError("inference.command must be an explicit argument list")
    unit_output = config.get("prediction_type") == "units"
    mel_output = config.get('prediction_type') == 'mels'
    predictions = run_root / "predictions" / ("units.jsonl" if unit_output else "mels.jsonl" if mel_output else "predictions.jsonl")
    command = resolve_command(
        inference["command"],
        {"data_root": data_root, "run_root": run_root, "predictions": predictions,
         "common_root": data_root.parent.parent / "common"},
    )
    if overwrite:
        command.append("--overwrite")
    if resume and (config.get('architecture') == 'translatotron2' or unit_output):
        command.append('--resume')
    result = run_external_experiment(
        repository_root,
        run_root / "inference-state",
        command=command,
        config=config,
        dataset_lock=data_root / "data-lock.json",
        seed=int(config.get("seed", 1)),
        resume=resume,
        overwrite=overwrite,
    )
    if mel_output:
        import numpy as np
        from .io import read_jsonl
        rows = list(read_jsonl(predictions))
        if not rows or len({r['pair_id'] for r in rows}) != len(rows):
            raise ValueError('empty or duplicate mel predictions')
        for row in rows:
            if row.get('status') == 'failed':
                if not row.get('error'):
                    raise ValueError('failed mel prediction missing error')
                continue
            values = np.load(row['mel_path'], allow_pickle=False)
            if values.ndim != 2 or values.size == 0 or not np.isfinite(values).all():
                raise ValueError('invalid predicted mel')
        return {**result, 'mel_predictions': str(predictions), 'samples': len(rows)}
    if unit_output:
        from .io import read_jsonl
        from .s2ut.reduce_units import validate_units
        rows = list(read_jsonl(predictions))
        if not rows or len({row['pair_id'] for row in rows}) != len(rows):
            raise ValueError("empty or duplicate unit predictions")
        for row in rows:
            if row.get('status') == 'failed':
                if not row.get('error'):
                    raise ValueError('failed unit prediction missing error')
            else:
                validate_units(row["units"], clusters=100)
        return {**result, "unit_predictions": str(predictions), "samples": len(rows)}
    return {**result, **validate_predictions(predictions)}
