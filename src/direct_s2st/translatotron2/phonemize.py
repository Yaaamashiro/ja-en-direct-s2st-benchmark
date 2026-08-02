from __future__ import annotations

import re
import subprocess
import unicodedata
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Protocol

from ..hashing import sha256_file
from ..io import atomic_write_json, atomic_write_text, read_jsonl
from ..s2ut.extract_units import stable_shard


class Phonemizer(Protocol):
    def __call__(self, text: str) -> str: ...


_ESPEAK_EN_US_INVENTORY_PROBES = (
    "pea bee tea day key guy fee view thin this see zoo she vision high cheese judge "
    "me no sing lay red yes we fleece kit face dress trap palm lot thought goat foot "
    "goose strut nurse comma price mouth choice near square cure happy letter atom "
    "writer rider button hidden sudden kitten little cattle rhythm loch genre rouge "
    "jalapeno resume deja oeuvre zero one two three four five six seven eight nine "
    "a b c d e f g h i j k l m n o p q r s t u v w x y z"
)


def tokenize_espeak_ipa(value: str, *, with_stress: bool = False) -> list[str]:
    """Split eSpeak IPA while preserving tied phones and combining marks."""
    normalized = unicodedata.normalize("NFKC", value).strip()
    if not with_stress:
        normalized = normalized.replace("ˈ", "").replace("ˌ", "")
    tokens: list[str] = []
    join_next = False
    for character in normalized:
        if character.isspace():
            if join_next:
                raise ValueError("invalid trailing IPA joiner")
            continue
        if character == "\u200d":
            if not tokens or join_next:
                raise ValueError("invalid IPA joiner")
            tokens[-1] += character
            join_next = True
            continue
        if join_next:
            tokens[-1] += character
            join_next = False
            continue
        if unicodedata.combining(character) or character in {"ː", "ˑ"}:
            if not tokens:
                raise ValueError("IPA modifier has no base phone")
            tokens[-1] += character
            continue
        tokens.append(character)
    if join_next:
        raise ValueError("invalid trailing IPA joiner")
    if not tokens:
        raise ValueError("IPA sequence must not be empty")
    return tokens


def normalize_phonemes(
    value: str, *, with_stress: bool = False, word_separator: str = "|"
) -> str:
    normalized = unicodedata.normalize("NFKC", value)
    if not with_stress:
        normalized = normalized.replace("ˈ", "").replace("ˌ", "")
    normalized = re.sub(r"\s+", " ", normalized).strip()
    normalized = re.sub(rf"\s*{re.escape(word_separator)}\s*", f" {word_separator} ", normalized)
    tokens = normalized.split()
    if not tokens:
        raise ValueError("phoneme sequence must not be empty")
    if any("\t" in token or "\n" in token for token in tokens):
        raise ValueError("phoneme tokens may not contain tabs or newlines")
    return " ".join(tokens)


def strip_punctuation(text: str) -> str:
    return "".join(" " if unicodedata.category(char).startswith("P") else char for char in text)


class EspeakNgPhonemizer:
    def __init__(
        self,
        *,
        executable: str = "espeak-ng",
        expected_version: str,
        language: str = "en-us",
        preserve_punctuation: bool = False,
        with_stress: bool = False,
        word_separator: str = "|",
    ) -> None:
        self.executable = executable
        self.language = language
        self.preserve_punctuation = preserve_punctuation
        self.with_stress = with_stress
        self.word_separator = word_separator
        version = subprocess.run(
            [executable, "--version"], check=True, capture_output=True, text=True
        ).stdout
        if expected_version not in version:
            raise RuntimeError(
                f"espeak-ng version mismatch: expected {expected_version!r}, got {version.strip()!r}"
            )

    def __call__(self, text: str) -> str:
        source = text if self.preserve_punctuation else strip_punctuation(text)
        words = source.split()
        if not words:
            raise ValueError("text must not be empty after punctuation removal")
        rendered: list[str] = []
        for word in words:
            result = subprocess.run(
                [self.executable, "-q", "--ipa", "--tie=z", "-v", self.language, word],
                check=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
            )
            rendered.append(
                " ".join(
                    tokenize_espeak_ipa(result.stdout, with_stress=self.with_stress)
                )
            )
        return normalize_phonemes(
            f" {self.word_separator} ".join(rendered),
            with_stress=self.with_stress,
            word_separator=self.word_separator,
        )

    def fixed_vocabulary(self) -> tuple[str, ...]:
        if self.language.lower() != "en-us":
            raise ValueError(
                "the checked inventory probes currently support only the en-us voice"
            )
        return tuple(sorted(set(self(_ESPEAK_EN_US_INVENTORY_PROBES).split())))


def _resolve_fixed_vocabulary(
    phonemizer: Phonemizer, fixed_vocabulary: Iterable[str] | None
) -> tuple[str, ...]:
    if fixed_vocabulary is None:
        provider = getattr(phonemizer, "fixed_vocabulary", None)
        if not callable(provider):
            raise ValueError(
                "fixed_vocabulary is required when the phonemizer does not provide one"
            )
        fixed_vocabulary = provider()
    vocabulary = tuple(sorted(set(fixed_vocabulary)))
    if not vocabulary:
        raise ValueError("fixed phoneme vocabulary must not be empty")
    for token in vocabulary:
        if not token or any(character.isspace() for character in token):
            raise ValueError(f"invalid fixed phoneme token: {token!r}")
    return vocabulary


def phonemize_manifests(
    common_root: Path,
    output_root: Path,
    *,
    phonemizer: Phonemizer,
    engine: str,
    version: str,
    fixed_vocabulary: Iterable[str] | None = None,
    split: str | None = None,
    shard_index: int = 0,
    num_shards: int = 1,
    limit: int | None = None,
    resume: bool = False,
    overwrite: bool = False,
) -> dict[str, Any]:
    vocabulary = _resolve_fixed_vocabulary(phonemizer, fixed_vocabulary)
    inventory_content = "".join(f"{token}\n" for token in vocabulary)
    atomic_write_text(
        output_root / "inventory.txt",
        inventory_content,
        resume=True,
        overwrite=overwrite,
    )
    splits = (split,) if split else ("train", "dev", "test")
    counts: dict[str, int] = {}
    token_counts: Counter[str] = Counter()
    processed = 0
    suffix = "" if num_shards == 1 else f".shard-{shard_index:05d}-of-{num_shards:05d}"
    for current_split in splits:
        lines: list[str] = []
        split_processed = 0
        for row in read_jsonl(common_root / f"{current_split}.jsonl"):
            pair_id = str(row["pair_id"])
            if stable_shard(pair_id, num_shards) != shard_index:
                continue
            if limit is not None and split_processed >= limit:
                break
            sequence = normalize_phonemes(
                phonemizer(str(row["en_tts_text"])), with_stress=True
            )
            lines.append(f"{pair_id}\t{sequence}\n")
            token_counts.update(sequence.split())
            processed += 1
            split_processed += 1
        atomic_write_text(
            output_root / f"{current_split}{suffix}.tsv",
            "".join(lines),
            resume=resume,
            overwrite=overwrite,
        )
        counts[current_split] = len(lines)

    metadata = {
        "engine": engine,
        "version": version,
        "num_shards": num_shards,
        "shard_index": shard_index,
        "counts": counts,
        "common_manifests": {
            current_split: sha256_file(common_root / f"{current_split}.jsonl")
            for current_split in splits
        },
        "token_counts": dict(sorted(token_counts.items())),
        "inventory_sha256": sha256_file(output_root / "inventory.txt"),
    }
    atomic_write_json(
        output_root / f"metadata{suffix}.json",
        metadata,
        resume=resume,
        overwrite=overwrite,
    )
    return {"processed": processed, **metadata}


def phonemizer_from_config(config: dict[str, Any]) -> EspeakNgPhonemizer:
    values = config["phonemizer"]
    return EspeakNgPhonemizer(
        executable=str(values.get("executable", "espeak-ng")),
        expected_version=str(values["version"]),
        language=str(values["language"]),
        preserve_punctuation=bool(values["preserve_punctuation"]),
        with_stress=bool(values["with_stress"]),
        word_separator=str(values["word_separator"]),
    )
