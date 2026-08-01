from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .config import PROFILES, RootPaths, load_config
from .runs import make_run_id


Handler = Callable[[argparse.Namespace, dict[str, Any]], dict[str, Any] | None]


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path)
    parser.add_argument("--profile", choices=PROFILES, default="smoke")
    parser.add_argument("--split", choices=("train", "dev", "test"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")


def _leaf(parent: argparse._SubParsersAction, name: str, help_text: str) -> argparse.ArgumentParser:
    parser = parent.add_parser(name, help=help_text)
    _common(parser)
    return parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="s2st-benchmark", description="Japanese-to-English S2ST benchmark")
    systems = parser.add_subparsers(dest="system", required=True)

    corpus = systems.add_parser("corpus").add_subparsers(dest="action", required=True)
    _leaf(corpus, "import", "Import accepted corpus rows")
    _leaf(corpus, "validate", "Validate normalized manifests")

    for system, actions in {
        "s2ut": ("extract-units", "prepare", "train", "infer"),
        "translatotron2": ("phonemize", "prepare", "train", "infer"),
    }.items():
        commands = systems.add_parser(system).add_subparsers(dest="action", required=True)
        for action in actions:
            _leaf(commands, action, f"Run {system} {action}")

    cascade = systems.add_parser("cascade").add_subparsers(dest="action", required=True)
    _leaf(cascade, "run", "Run the cascade pipeline")

    vocoder = systems.add_parser("vocoder").add_subparsers(dest="vocoder_type", required=True)
    for kind in ("unit", "mel"):
        commands = vocoder.add_parser(kind).add_subparsers(dest="action", required=True)
        for action in ("train", "infer"):
            _leaf(commands, action, f"Run {kind} vocoder {action}")

    evaluate = systems.add_parser("evaluate").add_subparsers(dest="action", required=True)
    _leaf(evaluate, "run", "Evaluate standardized predictions")
    _leaf(evaluate, "aggregate", "Aggregate experiment metrics")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")
    if args.num_shards < 1:
        raise ValueError("--num-shards must be positive")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard-index must be in [0, --num-shards)")
    if args.resume and args.overwrite:
        raise ValueError("--resume and --overwrite are mutually exclusive")


def command_key(args: argparse.Namespace) -> str:
    pieces = [args.system]
    if getattr(args, "vocoder_type", None):
        pieces.append(args.vocoder_type)
    pieces.append(args.action)
    return ".".join(pieces)


def _default_config(args: argparse.Namespace) -> Path | None:
    root = Path(__file__).resolve().parents[2] / "configs"
    if args.system == "corpus":
        return None
    if args.system in ("s2ut", "translatotron2"):
        name = "prepare" if args.action in ("extract-units", "phonemize", "prepare") else args.action
        return root / args.system / f"{name}.yaml"
    if args.system == "cascade":
        return root / "cascade" / "default.yaml"
    if args.system == "evaluate":
        return root / "evaluation" / "default.yaml"
    if args.system == "vocoder":
        return root / "vocoder" / f"{args.vocoder_type}.yaml"
    return None


def _run_corpus(args: argparse.Namespace) -> dict[str, Any]:
    from .manifests.import_corpus import import_corpus
    from .manifests.validate import validate_manifest_directory

    roots = RootPaths.from_environment()
    common_root = roots.experiment_data / "common"
    if args.action == "import":
        accepted = (
            roots.corpus / "production" / "manifests" / "releases" / "accepted.jsonl"
        )
        return import_corpus(
            accepted,
            corpus_root=roots.corpus,
            output_root=common_root,
            limit=args.limit,
            resume=args.resume,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
        )
    return validate_manifest_directory(common_root)


def _run_s2ut(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    from .s2ut.extract_units import extract_units, extractor_from_config
    from .s2ut.prepare_fairseq import prepare_fairseq

    roots = RootPaths.from_environment()
    common = roots.experiment_data / "common"
    units = roots.experiment_data / "s2ut" / "units"
    if args.action == "extract-units":
        plan = {
            "common_root": str(common),
            "output_root": str(units),
            "split": args.split,
            "shard_index": args.shard_index,
            "num_shards": args.num_shards,
        }
        if args.dry_run:
            return plan
        extractor = extractor_from_config(config, roots.cache)
        return extract_units(
            common,
            units,
            extractor=extractor,
            split=args.split,
            clusters=int(config["kmeans_clusters"]),
            hubert_model=str(config["hubert"]["model"]),
            hubert_revision=str(config["hubert"]["revision"]),
            hubert_layer=int(config["hubert_layer"]),
            kmeans_sha256=str(config["kmeans"]["sha256"]),
            shard_index=args.shard_index,
            num_shards=args.num_shards,
            limit=args.limit,
            resume=args.resume,
            overwrite=args.overwrite,
        )
    if args.action == "prepare":
        output = roots.experiment_data / "s2ut" / "fairseq"
        if args.dry_run:
            return {"common_root": str(common), "units_root": str(units), "output_root": str(output)}
        return prepare_fairseq(
            common,
            units,
            output,
            clusters=int(config.get("kmeans_clusters", 100)),
            resume=args.resume,
            overwrite=args.overwrite,
        )
    if args.action in ("train", "infer"):
        return _run_direct_model("s2ut", args, config, roots)
    raise NotImplementedError(f"unsupported s2ut action: {args.action}")


def _run_translatotron2(
    args: argparse.Namespace, config: dict[str, Any]
) -> dict[str, Any]:
    from .translatotron2.phonemize import phonemize_manifests, phonemizer_from_config
    from .translatotron2.prepare_fairseq import prepare_fairseq

    roots = RootPaths.from_environment()
    common = roots.experiment_data / "common"
    phonemes = roots.experiment_data / "translatotron2" / "phonemes"
    if args.action == "phonemize":
        if args.dry_run:
            return {"common_root": str(common), "output_root": str(phonemes)}
        engine = phonemizer_from_config(config)
        values = config["phonemizer"]
        return phonemize_manifests(
            common,
            phonemes,
            phonemizer=engine,
            engine=str(values["engine"]),
            version=str(values["version"]),
            split=args.split,
            shard_index=args.shard_index,
            num_shards=args.num_shards,
            limit=args.limit,
            resume=args.resume,
            overwrite=args.overwrite,
        )
    if args.action == "prepare":
        output = roots.experiment_data / "translatotron2" / "fairseq"
        if args.dry_run:
            return {"phoneme_root": str(phonemes), "output_root": str(output)}
        return prepare_fairseq(
            common,
            phonemes,
            output,
            mel_config=config["mel"],
            resume=args.resume,
            overwrite=args.overwrite,
        )
    if args.action in ("train", "infer"):
        return _run_direct_model("translatotron2", args, config, roots)
    raise NotImplementedError(f"unsupported translatotron2 action: {args.action}")


def _run_direct_model(
    system: str,
    args: argparse.Namespace,
    config: dict[str, Any],
    roots: RootPaths,
) -> dict[str, Any]:
    from .inference import infer_system
    from .training import train_system

    repository = Path(__file__).resolve().parents[2]
    data_root = roots.experiment_data / system / "fairseq"
    variant = str(config.get("variant", "default"))
    seed = int(config.get("seed", 1))
    run_id = str(config.get("run_id", make_run_id(system, variant, seed)))
    run_root = roots.runs / run_id
    if args.dry_run:
        return {
            "run_id": run_id,
            "run_root": str(run_root),
            "data_root": str(data_root),
            "action": args.action,
        }
    if args.action == "train":
        return train_system(
            repository,
            run_root,
            data_root,
            run_id=run_id,
            config=config,
            profile=args.profile,
            resume=args.resume,
            overwrite=args.overwrite,
        )
    return infer_system(
        repository,
        run_root,
        data_root,
        run_id=run_id,
        config=config,
        resume=args.resume,
        overwrite=args.overwrite,
    )


def _run_cascade(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    from .cascade.pipeline import components_from_config, run_pipeline

    roots = RootPaths.from_environment()
    split = args.split or "test"
    run_id = str(config.get("run_id", f"cascade-{args.profile}"))
    output = roots.runs / run_id / "predictions"
    if args.num_shards > 1:
        output = output / f"shard-{args.shard_index:05d}-of-{args.num_shards:05d}"
    plan = {
        "run_id": run_id,
        "split": split,
        "manifest": str(roots.experiment_data / "common" / f"{split}.jsonl"),
        "output_root": str(output),
    }
    if args.dry_run:
        return plan
    asr, mt, tts = components_from_config(config)
    return run_pipeline(
        roots.experiment_data / "common" / f"{split}.jsonl",
        output,
        run_id=run_id,
        asr=asr,
        mt=mt,
        tts=tts,
        split=split,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
        limit=args.limit,
        resume=args.resume,
        overwrite=args.overwrite,
    )


def _run_evaluate(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    from .evaluation.aggregate import aggregate_runs
    from .evaluation.run import evaluate_predictions

    roots = RootPaths.from_environment()
    if args.action == "aggregate":
        run_ids = config.get("run_ids")
        if not isinstance(run_ids, list) or not run_ids:
            raise ValueError("run_ids must be a non-empty list for evaluation aggregation")
        run_roots = [roots.runs / str(run_id) for run_id in run_ids]
        output = roots.experiment_data / "results"
        if args.dry_run:
            return {"run_roots": [str(path) for path in run_roots], "output_root": str(output)}
        return aggregate_runs(run_roots, output, overwrite=args.overwrite)

    run_id = config.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("run_id is required for evaluate run")
    run_root = roots.runs / run_id
    predictions = run_root / "predictions" / "predictions.jsonl"
    output = run_root / "metrics"
    if args.dry_run:
        return {"run_id": run_id, "predictions": str(predictions), "output_root": str(output)}
    from .evaluation.transcribe import EvaluationASR

    asr_values = config["asr"]
    transcriber = EvaluationASR(
        model_id=asr_values["model"], revision=asr_values["revision"]
    )
    speaker = None
    speaker_values = config.get("speaker_encoder", {})
    if speaker_values.get("enabled", True):
        from .evaluation.speaker_similarity import EcapaSimilarity

        speaker = EcapaSimilarity(
            model_id=speaker_values["model"],
            revision=speaker_values["revision"],
            cache_root=roots.cache,
        )
    blaser = None
    blaser_values = config.get("blaser", {})
    if blaser_values.get("enabled", False):
        from .evaluation.blaser import SonarBlaser

        blaser = SonarBlaser(
            source_encoder=str(
                blaser_values.get("source_encoder", "sonar_speech_encoder_jpn")
            ),
            target_encoder=str(
                blaser_values.get("target_encoder", "sonar_speech_encoder_eng")
            ),
        )
    return evaluate_predictions(
        predictions,
        output,
        transcribe=transcriber,
        blaser=blaser,
        speaker_similarity=speaker,
        resume=args.resume,
        overwrite=args.overwrite,
    )


def _run_vocoder(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    from .vocoders.runner import run_vocoder_command

    roots = RootPaths.from_environment()
    kind = args.vocoder_type
    action = args.action
    values = config.get(action)
    if not isinstance(values, dict) or not isinstance(values.get("command"), list):
        raise ValueError(f"{action}.command must be an explicit argument list")
    split = args.split or ("train" if action == "train" else "test")
    output = roots.experiment_data / "artifacts" / "vocoders" / kind
    command = [
        str(part).format_map(
            {
                "experiment_data_root": str(roots.experiment_data),
                "output_root": str(output),
                "split": split,
            }
        )
        for part in values["command"]
    ]
    return run_vocoder_command(
        command,
        output_root=output,
        split=split,
        training=action == "train",
        dry_run=args.dry_run,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        _validate_args(args)
        config = load_config(args.config or _default_config(args), profile=args.profile)
        key = command_key(args)
        if args.system == "corpus":
            payload = _run_corpus(args)
        elif args.system == "s2ut":
            payload = _run_s2ut(args, config)
        elif args.system == "translatotron2":
            payload = _run_translatotron2(args, config)
        elif args.system == "cascade":
            payload = _run_cascade(args, config)
        elif args.system == "evaluate":
            payload = _run_evaluate(args, config)
        elif args.system == "vocoder":
            payload = _run_vocoder(args, config)
        elif args.dry_run:
            payload = {"config": config}
        else:
            raise NotImplementedError(
                f"{key} is scaffolded but its task implementation is not installed"
            )
        result = {
            "command": key,
            "profile": args.profile,
            "dry_run": args.dry_run,
            **payload,
        }
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, default=str))
        return 0
    except (ValueError, KeyError, OSError, NotImplementedError) as error:
        print(json.dumps({"status": "failed", "error": str(error)}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
