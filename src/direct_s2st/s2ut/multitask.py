"""Versioned character supervision and strict pinned-fairseq TSV validation."""
from collections import Counter
import csv
import math
from pathlib import Path

import yaml

from ..io import atomic_write_text

TASKS = {"source_letter": "ja_text", "target_letter": "en_text", "decoder_target_ctc": "en_text"}
TOKENIZER_VERSION = "unicode-codepoint-v1"


def tokenize(text: str) -> list[str]:
    # No Unicode normalization/casing tables: identical codepoints on every Python version.
    if not isinstance(text, str) or not text.strip(" \t\r\n"):
        raise ValueError("empty linguistic sequence")
    tokens = []
    for char in text.strip(" \t\r\n"):
        token = "<space>" if char in " \t\r\n" else char
        forbidden_spaces = "\u00a0\u1680\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a\u2028\u2029\u202f\u205f\u3000"
        if token != "<space>" and (ord(char) < 32 or 127 <= ord(char) <= 159 or 0xD800 <= ord(char) <= 0xDFFF or char in forbidden_spaces):
            raise ValueError(f"unsupported whitespace/control character U+{ord(char):04X}")
        if token != "<space>" or not tokens or tokens[-1] != token:
            tokens.append(token)
    return tokens


def load_settings() -> dict:
    # Packaged defaults, also copied into each prepared data lock.
    return yaml.safe_load(Path(__file__).with_name("multitask.yaml").read_text(encoding="utf-8"))


def validate_task_config(task: str, cfg: dict) -> None:
    expected_decoder = "ctc" if task == "decoder_target_ctc" else "transformer"
    layer_key = "decoder_layer" if expected_decoder == "ctc" else "encoder_layer"
    other_key = "encoder_layer" if expected_decoder == "ctc" else "decoder_layer"
    if cfg.get("decoder_type") != expected_decoder or type(cfg.get(layer_key)) is not int or cfg[layer_key] < 1 or other_key in cfg:
        raise ValueError(f"invalid attachment for {task}")
    if not math.isfinite(float(cfg.get("loss_weight", 0))) or float(cfg.get("loss_weight", 0)) <= 0:
        raise ValueError(f"positive finite loss_weight required for {task}")


def prepare_labels(rows_by_split: dict, output_root: Path, *, settings: dict | None = None,
                   resume: bool = False, overwrite: bool = False) -> dict:
    settings = load_settings() if settings is None else settings
    if set(settings) != set(TASKS):
        raise ValueError("all three S2UT auxiliary tasks are required")
    config, artifacts = {}, {}
    differences = []
    for split, rows in rows_by_split.items():
        for row in rows:
            for language in ("ja", "en"):
                text, spoken = row[f"{language}_text"], row.get(f"{language}_tts_text", row[f"{language}_text"])
                if tokenize(text) != tokenize(spoken):
                    raise ValueError(f"{row['pair_id']}: {language}_text differs from {language}_tts_text; review audio/transcript alignment")
                if text != spoken:
                    differences.append({"pair_id": row["pair_id"], "language": language, "difference": "ASCII whitespace only"})
    for task, field in TASKS.items():
        cfg = dict(settings[task])
        validate_task_config(task, cfg)
        counts = Counter(token for row in rows_by_split["train"] for token in tokenize(row[field]))
        if not counts:
            raise ValueError(f"empty train dictionary for {task}")
        for split, rows in rows_by_split.items():
            lines = ["id\ttgt_text\n"]
            for row in rows:
                tokens = tokenize(row[field])
                unknown = set(tokens) - counts.keys()
                if unknown:
                    raise ValueError(f"unknown {task} tokens in {split}/{row['pair_id']}: {sorted(unknown)}; provide a reviewed training vocabulary")
                lines.append(f"{row['pair_id']}\t{' '.join(tokens)}\n")
            artifacts[output_root / task / f"{split}.tsv"] = "".join(lines)
        artifacts[output_root / task / "dict.txt"] = "".join(f"{token} {counts[token]}\n" for token in sorted(counts))
        config[task] = {**cfg, "data": (output_root / task).resolve().as_posix(),
                        "dict": (output_root / task / "dict.txt").resolve().as_posix()}
    # Validate every task before emitting any labels.
    for path, text in artifacts.items():
        atomic_write_text(path, text, resume=resume, overwrite=overwrite)
    atomic_write_text(output_root / "config_multitask.yaml", yaml.safe_dump(config), resume=resume, overwrite=overwrite)
    return {"tokenizer": TOKENIZER_VERSION, "vocabulary_source": "train_only", "tasks": settings,
            "text_fields": TASKS, "text_tts_differences": differences}


def read_tsv(path: Path, columns: tuple[str, ...]) -> dict[str, dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if tuple(reader.fieldnames or ()) != columns:
            raise ValueError(f"invalid TSV header: {path}")
        rows = {}
        for row in reader:
            pair_id = row["id"]
            if not pair_id or pair_id in rows or None in row or any(v is None for v in row.values()):
                raise ValueError(f"invalid or duplicate sample ID: {path}: {pair_id}")
            rows[pair_id] = row
    return rows


def validate_prepared(root: Path, *, clusters: int = 100) -> dict:
    from .reduce_units import validate_units
    cfg = yaml.safe_load((root / "config_multitask.yaml").read_text(encoding="utf-8"))
    if set(cfg) != set(TASKS):
        raise ValueError("all three S2UT auxiliary tasks are required")
    for name, settings in cfg.items():
        validate_task_config(name, settings)
    seen, counts = set(), {}
    for split in ("train", "dev", "test"):
        main = read_tsv(root / f"{split}.tsv", ("id", "src_audio", "src_n_frames", "tgt_audio", "tgt_n_frames"))
        if not main or seen.intersection(main):
            raise ValueError(f"empty split or duplicate pair IDs across splits: {split}")
        seen.update(main)
        for row in main.values():
            units = [int(x) for x in row["tgt_audio"].split()]
            validate_units(units, clusters=clusters)
            if len(units) != int(row["tgt_n_frames"]) or int(row["src_n_frames"]) <= 0:
                raise ValueError(f"invalid frame counts: {row['id']}")
            if not Path(row["src_audio"]).is_file():
                raise FileNotFoundError(row["src_audio"])
        targets = {}
        for task in TASKS:
            task_root, dictionary = Path(cfg[task]["data"]), Path(cfg[task]["dict"])
            vocabulary = set()
            for line in dictionary.read_text(encoding="utf-8").splitlines():
                token, count = line.rsplit(" ", 1)
                if not token or token in vocabulary or int(count) < 1:
                    raise ValueError(f"invalid dictionary: {dictionary}")
                vocabulary.add(token)
            aux = read_tsv(task_root / f"{split}.tsv", ("id", "tgt_text"))
            targets[task] = aux
            if main.keys() != aux.keys():
                raise ValueError(f"sample ID alignment failure: {split}/{task}")
            for pair_id, row in aux.items():
                tokens = row["tgt_text"].split()
                if not tokens or set(tokens) - vocabulary:
                    raise ValueError(f"empty sequence or dictionary coverage failure: {task}/{pair_id}")
        if targets["target_letter"] != targets["decoder_target_ctc"]:
            raise ValueError(f"target linguistic / decoder CTC label mismatch: {split}")
        counts[split] = len(main)
    return {"splits": counts, "multitask": list(TASKS)}
