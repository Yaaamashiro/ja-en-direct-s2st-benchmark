from __future__ import annotations

import csv
from ..progress import operation
import io
import json
from pathlib import Path
from typing import Any

from ..io import atomic_write_json, atomic_write_jsonl, atomic_write_text, read_jsonl


FIELDS = (
    "system_id",
    "run_id",
    "bleu",
    "blaser",
    "speaker_similarity_source",
    "speaker_similarity_reference",
    "mean_rtf",
    "failure_rate",
)


@operation('evaluation/aggregate: aggregate_runs')
def aggregate_runs(
    run_roots: list[Path],
    output_root: Path,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    metrics: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []
    for root in run_roots:
        metrics_path = root / "metrics" / "metrics.json"
        sample_path = root / "metrics" / "per_sample.jsonl"
        metrics.append(json.loads(metrics_path.read_text(encoding="utf-8")))
        samples.extend(read_jsonl(sample_path))
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=FIELDS, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    writer.writerows(metrics)
    atomic_write_text(output_root / "comparison.csv", stream.getvalue(), overwrite=overwrite)
    atomic_write_jsonl(output_root / "per_sample.jsonl", samples, overwrite=overwrite)
    payload = {"runs": metrics}
    atomic_write_json(output_root / "metrics.json", payload, overwrite=overwrite)
    lines = ["# S2ST comparison", "", "| System | Run | BLEU | BLASER | RTF | Failure rate |", "|---|---|---:|---:|---:|---:|"]
    for row in metrics:
        lines.append(
            f"| {row.get('system_id')} | {row.get('run_id')} | {row.get('bleu')} | "
            f"{row.get('blaser')} | {row.get('mean_rtf')} | {row.get('failure_rate')} |"
        )
    atomic_write_text(output_root / "report.md", "\n".join(lines) + "\n", overwrite=overwrite)
    return payload
