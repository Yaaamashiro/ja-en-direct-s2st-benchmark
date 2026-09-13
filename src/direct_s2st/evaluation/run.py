from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..io import atomic_write_json, atomic_write_jsonl, read_jsonl
from ..predictions import Prediction
from .bleu import corpus_bleu, normalize_english, sentence_bleu
from .runtime import aggregate_runtime, audio_quality


def evaluate_predictions(
    predictions_path: Path,
    output_root: Path,
    *,
    transcribe: Callable[[Path], str],
    blaser: Callable[[Path, Path], float] | None = None,
    speaker_similarity: Callable[[Path, Path], float] | None = None,
    resume: bool = False,
    overwrite: bool = False,
    evaluation_identity: dict | None = None,
) -> dict[str, Any]:
    per_sample_path = output_root / "per_sample.jsonl"
    predictions = list(read_jsonl(predictions_path))
    if not predictions or len({r['pair_id'] for r in predictions}) != len(predictions):
        raise ValueError('empty or duplicate evaluation inputs')
    if len({(r['system_id'], r['run_id']) for r in predictions}) != 1:
        raise ValueError('evaluation cannot mix runs/systems')
    from ..journal import Journal
    from ..hashing import sha256_file
    identity = {'predictions': sha256_file(predictions_path), 'evaluation': evaluation_identity,
                'speaker_enabled': speaker_similarity is not None, 'blaser_enabled': blaser is not None,
                'audio': {}}
    for raw in predictions:
        Prediction.from_dict(raw)
        for field in ('output_audio', 'source_audio', 'reference_audio'):
            if raw.get(field) and Path(raw[field]).is_file():
                identity['audio'][raw[field]] = sha256_file(Path(raw[field]))
    journal = Journal(per_sample_path, identity, resume=resume, overwrite=overwrite)
    prior = journal.rows.copy()
    results: list[dict[str, Any]] = []
    hypotheses: list[str] = []
    references: list[str] = []
    for raw in predictions:
        prediction = Prediction.from_dict(raw)
        old = prior.get(prediction.pair_id)
        if old and old.get("evaluation_status") == "success":
            result = old
        else:
            result = {
                "pair_id": prediction.pair_id,
                "system_id": prediction.system_id,
                "run_id": prediction.run_id,
                "generation_status": prediction.status,
                "evaluation_status": "failed",
                "evaluation_error": prediction.error,
                "reference_raw": prediction.reference_text,
                "reference_normalized": normalize_english(prediction.reference_text),
                "hypothesis_raw": None,
                "hypothesis_normalized": None,
                "sentence_bleu": None,
                "blaser": None,
                "speaker_similarity_source": None,
                "speaker_similarity_reference": None,
                "inference_seconds": prediction.inference_seconds,
                "real_time_factor": prediction.real_time_factor,
            }
            if prediction.status == "success" and prediction.output_audio:
                output_audio = Path(prediction.output_audio)
                try:
                    hypothesis = transcribe(output_audio)
                    hypothesis_normalized = normalize_english(hypothesis)
                    result.update(
                        {
                            "evaluation_status": "success",
                            "evaluation_error": None,
                            "hypothesis_raw": hypothesis,
                            "hypothesis_normalized": hypothesis_normalized,
                            "sentence_bleu": sentence_bleu(
                                hypothesis_normalized, result["reference_normalized"]
                            ),
                            **audio_quality(output_audio),
                        }
                    )
                    if blaser is not None:
                        try:
                            result["blaser"] = blaser(Path(prediction.source_audio), output_audio)
                        except Exception as error:
                            result["blaser_error"] = f"{type(error).__name__}: {error}"
                    if speaker_similarity is not None:
                        try:
                            result["speaker_similarity_source"] = speaker_similarity(
                                Path(prediction.source_audio), output_audio
                            )
                            result["speaker_similarity_reference"] = speaker_similarity(
                                Path(prediction.reference_audio), output_audio
                            )
                        except Exception as error:
                            result["speaker_similarity_error"] = (
                                f"{type(error).__name__}: {error}"
                            )
                except Exception as error:
                    result["evaluation_error"] = f"{type(error).__name__}: {error}"
        if result.get("evaluation_status") == "success":
            hypotheses.append(str(result["hypothesis_normalized"]))
            references.append(str(result["reference_normalized"]))
        results.append(result)
        journal.record(result)

    def mean(field: str) -> float | None:
        values = [float(row[field]) for row in results if row.get(field) is not None]
        return sum(values) / len(values) if values else None

    metrics = {
        "system_id": predictions[0]["system_id"] if predictions else None,
        "run_id": predictions[0]["run_id"] if predictions else None,
        "samples": len(predictions),
        "evaluation_identity": evaluation_identity,
        "generation_failures": sum(row.get("status") == "failed" for row in predictions),
        "asr_success_rate": len(hypotheses) / len(predictions) if predictions else 0.0,
        "bleu": corpus_bleu(hypotheses, references),
        "blaser": mean("blaser"),
        "speaker_similarity_source": mean("speaker_similarity_source"),
        "speaker_similarity_reference": mean("speaker_similarity_reference"),
        "silence_ratio": mean("silence_ratio"),
        "clipping_ratio": mean("clipping_ratio"),
        **aggregate_runtime(predictions),
    }
    atomic_write_json(output_root / "metrics.json", metrics, overwrite=overwrite or resume)
    return metrics
