from __future__ import annotations

import argparse
from .progress import operation, track
import json
import os
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
        "s2ut": ("fetch-artifacts", "extract-units", "prepare", "validate", "train", "infer"),
        "translatotron2": ("phonemize", "validate-phonemes", "prepare", "validate", "train", "infer"),
    }.items():
        commands = systems.add_parser(system).add_subparsers(dest="action", required=True)
        for action in actions:
            _leaf(commands, action, f"Run {system} {action}")

    cascade = systems.add_parser("cascade").add_subparsers(dest="action", required=True)
    _leaf(cascade, "run", "Run the cascade pipeline")
    s2t_tts = systems.add_parser("s2t-tts").add_subparsers(dest="action", required=True)
    _leaf(s2t_tts, "run", "Run direct speech translation followed by TTS")

    vocoder = systems.add_parser("vocoder").add_subparsers(dest="vocoder_type", required=True)
    for kind in ("unit", "mel"):
        commands = vocoder.add_parser(kind).add_subparsers(dest="action", required=True)
        for action in ("train", "infer"):
            _leaf(commands, action, f"Run {kind} vocoder {action}")

    evaluate = systems.add_parser("evaluate").add_subparsers(dest="action", required=True)
    _leaf(evaluate, "run", "Evaluate standardized predictions")
    _leaf(evaluate, "aggregate", "Aggregate experiment metrics")
    _leaf(evaluate, 'verify', 'Verify four-system real artifact completion')
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
    configured_root = os.environ.get("S2ST_CONFIG_ROOT")
    root = (
        Path(configured_root).expanduser()
        if configured_root
        else Path(__file__).resolve().parents[2] / "configs"
    )
    if args.system == "corpus":
        return None
    if args.system in ("s2ut", "translatotron2"):
        name = "prepare" if args.action in ("fetch-artifacts", "extract-units", "phonemize", "validate-phonemes", "prepare", "validate") else args.action
        return root / args.system / f"{name}.yaml"
    if args.system == "cascade":
        return root / "cascade" / "default.yaml"
    if args.system == "s2t-tts":
        return root / "s2t_tts" / "default.yaml"
    if args.system == "evaluate":
        return root / "evaluation" / "default.yaml"
    if args.system == "vocoder":
        return root / "vocoder" / f"{args.vocoder_type}.yaml"
    return None


@operation('direct_s2st/cli: _run_corpus')
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
    return validate_manifest_directory(common_root, resume=args.resume)


@operation('direct_s2st/cli: _run_s2ut')
def _run_s2ut(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    from .artifacts import download_artifact
    from .s2ut.extract_units import extract_units, extractor_from_config
    from .s2ut.prepare_fairseq import prepare_fairseq

    roots = RootPaths.from_environment()
    common = roots.experiment_data / "common"
    units = roots.experiment_data / "s2ut" / "units"
    if args.action == "validate":
        from .s2ut.multitask import validate_prepared
        from .s2ut.migrate_labels import paper_data_root
        return validate_prepared(paper_data_root(roots.experiment_data), clusters=int(config.get("kmeans_clusters", 100)))
    if args.action == "fetch-artifacts":
        values = config["kmeans"]
        destination = roots.cache / str(values["path"])
        plan = {
            "artifact": str(values["artifact"]),
            "url": str(values["source_url"]),
            "path": str(destination),
            "sha256": str(values["sha256"]),
        }
        if args.dry_run:
            return plan
        return {
            **plan,
            **download_artifact(
                str(values["source_url"]),
                destination,
                sha256=str(values["sha256"]),
                overwrite=args.overwrite,
            ),
        }
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
            kmeans_artifact=str(config["kmeans"]["artifact"]),
            shard_index=args.shard_index,
            num_shards=args.num_shards,
            limit=args.limit,
            resume=args.resume,
            overwrite=args.overwrite,
        )
    if args.action == "prepare":
        from .s2ut.migrate_labels import paper_data_root, current_labels, migrate
        output = paper_data_root(roots.experiment_data)
        if args.dry_run:
            return {"common_root": str(common), "units_root": str(units), "output_root": str(output)}
        if args.resume and not args.overwrite and (output/'data-lock.json').is_file():
            prior = json.loads((output/'data-lock.json').read_text(encoding='utf-8'))
            if not current_labels(prior):
                # Use the same original source as the cell-5 migration CLI;
                # intermediate Unigram-v1 siblings are preserved as well.
                migrated = migrate(common, roots.experiment_data / 's2ut/fairseq')
                return json.loads((migrated/'data-lock.json').read_text(encoding='utf-8'))
        return prepare_fairseq(
            common,
            units,
            output,
            clusters=int(config.get("kmeans_clusters", 100)),
            multitask=config.get("multitask"),
            resume=args.resume,
            overwrite=args.overwrite,
        )
    if args.action in ("train", "infer"):
        return _run_direct_model("s2ut", args, config, roots)
    raise NotImplementedError(f"unsupported s2ut action: {args.action}")


@operation('direct_s2st/cli: _run_translatotron2')
def _run_translatotron2(
    args: argparse.Namespace, config: dict[str, Any]
) -> dict[str, Any]:
    from .translatotron2.phonemize import phonemize_manifests, phonemizer_from_config
    from .translatotron2.prepare_fairseq import prepare_fairseq

    roots = RootPaths.from_environment()
    common = roots.experiment_data / "common"
    phonemes = roots.experiment_data / "translatotron2" / "phonemes"
    if args.action == 'validate-phonemes':
        from .translatotron2.prepare_fairseq import validate_phonemes
        if args.dry_run:
            return dict(phoneme_root=str(phonemes), wav_reads=0)
        return validate_phonemes(roots.experiment_data / 'common', phonemes)
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
            source_mel_config=config.get('source_mel'),
            resume=args.resume,
            overwrite=args.overwrite,
        )
    if args.action == 'validate':
        root = roots.experiment_data / 'translatotron2' / 'fairseq'
        if args.dry_run:
            return {'data_root': str(root), 'action': 'validate'}
        from .translatotron2.recovery import validate_prepared
        return validate_prepared(root, resume=args.resume, overwrite=args.overwrite)
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

    repository = Path(os.environ.get("S2ST_CONFIG_ROOT", str(Path(__file__).resolve().parents[2] / "configs"))).resolve().parent
    data_root = roots.experiment_data / system / "fairseq"
    if system == 's2ut':
        from .s2ut.migrate_labels import paper_data_root
        data_root = paper_data_root(roots.experiment_data)
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


@operation('direct_s2st/cli: _run_cascade')
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
        model_identity=config,
        split=split,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
        limit=args.limit,
        resume=args.resume,
        overwrite=args.overwrite,
    )


@operation('direct_s2st/cli: _run_s2t_tts')
def _run_s2t_tts(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    from .s2t_tts.pipeline import components_from_config, run_pipeline
    from .s2t_tts.s2t import validate_s2t_config
    from .runs import validate_run_id, collect_model_revisions
    from .io import atomic_write_json
    validate_s2t_config(config)
    roots = RootPaths.from_environment()
    split = args.split or 'test'
    run_id = validate_run_id(str(config.get('run_id', f's2t_tts-{args.profile}')))
    output = roots.runs / run_id / 'predictions'
    if args.num_shards > 1:
        output = output / f'shard-{args.shard_index:05d}-of-{args.num_shards:05d}'
    manifest = roots.experiment_data / 'common' / f'{split}.jsonl'
    plan = dict(system_id='s2t_tts', run_id=run_id, split=split, manifest=str(manifest),
                output_root=str(output), config=config, limit=args.limit,
                shard_index=args.shard_index, num_shards=args.num_shards)
    if args.dry_run:
        return plan
    atomic_write_json(output / 'run-metadata.json',
        dict(plan=plan, model_revisions=collect_model_revisions(config)),
        resume=args.resume, overwrite=args.overwrite)
    s2t, tts = components_from_config(config)
    try:
        return run_pipeline(manifest, output, run_id=run_id, s2t=s2t, tts=tts,
            model_identity=config, split=split, shard_index=args.shard_index,
            num_shards=args.num_shards, limit=args.limit, resume=args.resume, overwrite=args.overwrite)
    finally:
        del s2t, tts
        import gc
        gc.collect()
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


@operation('direct_s2st/cli: _run_evaluate')
def _run_evaluate(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    from .evaluation.aggregate import aggregate_runs
    from .evaluation.run import evaluate_predictions

    roots = RootPaths.from_environment()
    if args.action in ('aggregate', 'verify'):
        run_ids = config.get("run_ids")
        if not isinstance(run_ids, list) or not run_ids:
            raise ValueError("run_ids must be a non-empty list for evaluation aggregation")
        run_roots = [roots.runs / str(run_id) for run_id in run_ids]
        output = roots.experiment_data / "results"
        if args.dry_run:
            return {"run_roots": [str(path) for path in run_roots], "output_root": str(output)}
        if args.action == 'verify':
            from .evaluation.acceptance import verify_suite
            return verify_suite(roots.experiment_data / 'common', run_roots, output, overwrite=args.overwrite, resume=args.resume)
        return aggregate_runs(run_roots, output, overwrite=args.overwrite, resume=args.resume)

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
        evaluation_identity=config,
        resume=args.resume,
        overwrite=args.overwrite,
    )


@operation('direct_s2st/cli: _run_vocoder')
def _run_vocoder(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    from .vocoders.runner import run_vocoder_command

    roots = RootPaths.from_environment()
    kind = args.vocoder_type
    action = args.action
    values = config.get(action)
    if not isinstance(values, dict) or not isinstance(values.get("command"), list):
        raise ValueError(f"{action}.command must be an explicit argument list")
    split = args.split or ("train" if action == "train" else "test")
    run_id = config.get("run_id")
    if not isinstance(run_id, str):
        raise ValueError("vocoder config requires run_id matching the direct-model inference run")
    from .runs import validate_run_id
    validate_run_id(run_id)
    output = roots.runs / run_id / ('vocoder-' + kind if action == 'train' else 'predictions')
    command = [
        str(part).format_map(
            {
                "experiment_data_root": str(roots.experiment_data),
                "output_root": str(output),
                "run_root": str(roots.runs / run_id),
                "cache_root": str(roots.cache),
                "split": split,
                "config_root": str(Path(os.environ.get('S2ST_CONFIG_ROOT', Path(__file__).resolve().parents[2] / 'configs'))),
            }
        )
        for part in values["command"]
    ]
    if args.overwrite:
        command.append("--overwrite")
    if args.resume:
        command.append('--resume')
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
        if not args.dry_run and all(os.environ.get(name) for name in ("CORPUS_ROOT", "EXPERIMENT_DATA_ROOT", "RUNS_ROOT", "CACHE_ROOT")):
            RootPaths.from_environment().validate_output_roots()
        key = command_key(args)
        if args.system == "corpus":
            payload = _run_corpus(args)
        elif args.system == "s2ut":
            payload = _run_s2ut(args, config)
        elif args.system == "translatotron2":
            payload = _run_translatotron2(args, config)
        elif args.system == "cascade":
            payload = _run_cascade(args, config)
        elif args.system == "s2t-tts":
            payload = _run_s2t_tts(args, config)
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
    except (ValueError, KeyError, OSError, RuntimeError) as error:
        print(json.dumps({"status": "failed", "error": str(error)}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
