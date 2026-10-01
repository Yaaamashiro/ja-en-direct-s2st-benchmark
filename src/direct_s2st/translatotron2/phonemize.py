from __future__ import annotations
from ..preparation import Checkpoints, checkpoint_map
from ..journal import digest

import re
from ..progress import operation
import subprocess
import sys
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

# Phoneme mnemonics in base1 -> en -> en-us from the immutable 1.52.0
# phsource/{phonemes,ph_english,ph_english_us} at commit
# 4870adfa25b1a32b4361592f1be8a40337c58d6c. Exclude stress/pause/virtual
# controls. Query the pinned engine's IPA rendering, including aliases and
# rare phones; do not infer the inventory from train/dev/test text.
_ESPEAK_EN_US_PHONEMES = (
    '? @ @- a e i o u m- n- N- r- l- r r/ R R2 R3 r" '
    'l l/ l/2 l/3 l^ l. L/ L w j ; m n n. n^ N ** * r. '
    'b d d[ dZ dZ; J g B v v# D z Z z. z; Z; J^ Q Q^ Q" '
    'p t t[ tS tS; c k q f T s S s. s; S; l# C x X h '
    't2 t# d# z# z/2 w# 3 @2 @5 @L a2 a# aa E E# E2 I I2 I# I2# '
    '0 0# 02 O2 V U A: A@ A# 3: i: O: O O@ o@ u: '
    'aU oU oU# aI eI OI e@ i@ i@3 U@ aI@ aI3 aU@ IR VR o: A~ O~ e: e# a#2 @#'
).split()
TEXT_PROCESSING_VERSION = 'espeak-lexical-punctuation-v2'
INVENTORY_VERSION = 'espeak-en-us-1.52.0-phoneme-table-v2'


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
    """Drop sentence punctuation, not lexical apostrophes/decimal points.

    eSpeak receives contractions, possessives, hyphenated words, abbreviations
    and numbers intact. Never rewrite the common manifest or TTS source text.
    """
    result = []
    for index, char in enumerate(text):
        left = text[index - 1] if index else ''
        right = text[index + 1] if index + 1 < len(text) else ''
        if char in "'’" and left.isalnum() and (right.isalnum() or left.lower() == 's'):
            result.append("'")
        elif char in '-‐‑' and left.isalnum() and right.isalnum():
            result.append('-')
        elif char in '.,:/' and left.isdigit() and right.isdigit():
            result.append(char)
        elif char == '-' and right.isdigit() and (not left or left.isspace()):
            result.append(char)
        elif char == '.' and left.isalpha() and (right.isalpha() or (index >= 2 and text[index - 2] == '.')):
            result.append(char)
        else:
            result.append(' ' if unicodedata.category(char).startswith('P') else char)
    return ''.join(result)


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
        text_processing_version: str = TEXT_PROCESSING_VERSION,
    ) -> None:
        self.executable = executable
        self.language = language
        self.preserve_punctuation = preserve_punctuation
        self.with_stress = with_stress
        self.word_separator = word_separator
        if text_processing_version not in ('legacy-v1', TEXT_PROCESSING_VERSION):
            raise ValueError('unsupported eSpeak text processing version')
        self.text_processing_version = text_processing_version
        self.inventory_version = ('probe-v1' if text_processing_version == 'legacy-v1' else INVENTORY_VERSION)
        version = subprocess.run(
            [executable, "--version"], check=True, capture_output=True, text=True
        ).stdout
        if expected_version not in version:
            raise RuntimeError(
                f"espeak-ng version mismatch: expected {expected_version!r}, got {version.strip()!r}"
            )

    def __call__(self, text: str) -> str:
        source = text if self.preserve_punctuation else (
            ''.join(' ' if unicodedata.category(char).startswith('P') else char for char in text)
            if self.text_processing_version == 'legacy-v1' else strip_punctuation(text))
        words = source.split()
        if not words:
            raise ValueError("text must not be empty after punctuation removal")
        rendered: list[str] = []
        for word in words:
            result = subprocess.run(
                # '--' prevents lexical minus signs becoming CLI options.
                [self.executable, "-q", "--ipa", "--tie=z", "-v", self.language, '--', word],
                check=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
            )
            if not result.stdout.strip():
                # Unicode symbols such as U+22EF (⋯, category Sm) survive
                # strip_punctuation but may have no spoken form in eSpeak.
                # Preserve all previously successful pronunciations, including
                # symbols that eSpeak does pronounce; never drop lexical words.
                if all(unicodedata.category(char)[0] in {'P', 'S'} for char in word):
                    print(f'[phonemize] ignored non-spoken symbol={word!r}',
                          file=sys.stderr, flush=True)
                    continue
                raise ValueError(f'eSpeak returned empty IPA for word={word!r}; stderr={result.stderr!r}')
            rendered.append(
                " ".join(
                    tokenize_espeak_ipa(result.stdout, with_stress=self.with_stress)
                )
            )
        if not rendered:
            raise ValueError('text contains no spoken phonemes')
        return normalize_phonemes(
            f" {self.word_separator} ".join(rendered),
            with_stress=self.with_stress,
            word_separator=self.word_separator,
        )

    @operation('phonemes: fixed eSpeak definition inventory')
    def fixed_vocabulary(self) -> tuple[str, ...]:
        if self.language.lower() != "en-us":
            raise ValueError(
                "the checked inventory probes currently support only the en-us voice"
            )
        tokens = set(self(_ESPEAK_EN_US_INVENTORY_PROBES).split())
        if self.text_processing_version != 'legacy-v1':
            from ..progress import track
            for mnemonic in track(_ESPEAK_EN_US_PHONEMES, 'phonemes: inspect pinned definitions'):
                result = subprocess.run(
                    [self.executable, '-q', '--ipa', '--tie=z', '-v', self.language,
                     '--', f'[[{mnemonic}]]'], check=True, capture_output=True,
                    text=True, encoding='utf-8')
                # Some linking aliases are silent without surrounding vowels.
                if result.stdout.strip():
                    tokens.update(tokenize_espeak_ipa(result.stdout, with_stress=self.with_stress))
            tokens.add(self.word_separator)
        return tuple(sorted(tokens))


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


@operation('translatotron2/phonemize: phonemize_manifests')
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
    suffix = "" if num_shards == 1 else f".shard-{shard_index:05d}-of-{num_shards:05d}"
    metadata_path = output_root / f'metadata{suffix}.json'
    if resume and not overwrite and isinstance(phonemizer, EspeakNgPhonemizer) and metadata_path.is_file():
        import json
        previous = json.loads(metadata_path.read_text(encoding='utf-8'))
        saved_version = previous.get('text_processing_version', 'legacy-v1')
        if saved_version != phonemizer.text_processing_version:
            if saved_version != 'legacy-v1':
                raise ValueError('unsupported saved text processing version; use an explicit migration')
            # Keep an existing run reproducible, not a mixture of old/new G2P.
            # Explicit regeneration is provided by the preparation recovery tool.
            phonemizer.text_processing_version = saved_version
            phonemizer.inventory_version = previous.get('inventory_version', 'probe-v1')
            print('[phonemize] retaining saved legacy text processing; '
                  'use resume_preparation.py --regenerate-phonemes for the corrected labels',
                  file=sys.stderr, flush=True)
    vocabulary = _resolve_fixed_vocabulary(phonemizer, fixed_vocabulary)
    inventory_content = "".join(f"{token}\n" for token in vocabulary)
    inventory_path = output_root / 'inventory.txt'
    if (resume and not overwrite and isinstance(phonemizer, EspeakNgPhonemizer)
            and inventory_path.is_file()
            and inventory_path.read_text(encoding='utf-8') != inventory_content):
        raise ValueError('saved phoneme inventory uses a different generation; '
                         'use resume_preparation.py --regenerate-phonemes explicitly; old data is retained')
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
    identity = dict(stage='phonemes-v1', engine=engine, version=version, inventory=list(vocabulary),
                    options={name: getattr(phonemizer, name, None) for name in
                             ('language', 'preserve_punctuation', 'with_stress', 'word_separator')})
    processing = getattr(phonemizer, 'text_processing_version', None)
    if processing and processing != 'legacy-v1':
        identity['text_processing_version'] = processing
        identity['inventory_version'] = phonemizer.inventory_version
    # Retain compatibility with the known old probe dictionary without changing
    # its cache keys or files. New engines use the definition-based inventory.
    from .prepare_fairseq import ESPEAK_152_INVENTORY_SUPPLEMENT
    allowed = set(vocabulary)
    if engine == 'espeak-ng' and version == '1.52.0':
        allowed.update(ESPEAK_152_INVENTORY_SUPPLEMENT)
    for current_split in splits:
        lines: list[str] = []
        # Phonemization needs text, not another stat/open of every WAV on Drive.
        rows = [row for row in read_jsonl(common_root / f'{current_split}.jsonl')
                if stable_shard(str(row['pair_id']), num_shards) == shard_index]
        if limit is not None:
            rows = rows[:limit]
        def phonemize(row):
            try:
                sequence = normalize_phonemes(phonemizer(str(row['en_tts_text'])), with_stress=True)
                unknown = set(sequence.split()) - allowed
                if unknown:
                    raise ValueError(f'unknown phonemes: {sorted(unknown)}; '
                                     'review the fixed engine inventory (do not grow it from dev/test)')
                return dict(pair_id=str(row['pair_id']), sequence=sequence)
            except (ValueError, subprocess.CalledProcessError) as error:
                raise ValueError(f"phonemization failed: pair_id={row['pair_id']!r} "
                                 f"text={row['en_tts_text']!r}: {error}") from error
        with Checkpoints(output_root / '.checkpoints', identity, resume=resume and not overwrite,
                         overwrite=overwrite and current_split == splits[0]) as cache:
            for result in checkpoint_map(phonemize, rows, cache,
                    lambda row: digest([row['pair_id'], row['en_tts_text']]),
                    f'phonemize: {current_split} (generate/reuse)', total=len(rows)):
                sequence = result['sequence']
                if not sequence.split() or set(sequence.split()) - allowed:
                    raise ValueError(f"invalid/unknown cached phonemes: pair_id={result['pair_id']!r}")
                lines.append(f"{result['pair_id']}\t{sequence}\n")
                token_counts.update(sequence.split())
                processed += 1
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
    if processing and processing != 'legacy-v1':
        metadata.update(text_processing_version=processing, inventory_version=phonemizer.inventory_version,
                        phoneme_manifests={current_split: sha256_file(output_root / f'{current_split}{suffix}.tsv')
                                           for current_split in splits})
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
