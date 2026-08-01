from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


SPLITS = ("train", "dev", "test")


def _text(row: dict[str, Any], key: str, *, fallback: str | None = None) -> str:
    value = row.get(key)
    if value is None and fallback is not None:
        value = row.get(fallback)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value.strip()


def _number(row: dict[str, Any], key: str) -> float:
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ValueError(f"{key} must be a positive number")
    return float(value)


@dataclass(frozen=True)
class CommonManifestRow:
    pair_id: str
    split: str
    corpus: str
    source_id: str
    ja_audio: str
    en_audio: str
    ja_text_raw: str
    en_text_raw: str
    ja_text: str
    en_text: str
    ja_tts_text: str
    en_tts_text: str
    ja_duration: float
    en_duration: float
    ja_sha256: str
    en_sha256: str

    @classmethod
    def from_corpus_row(
        cls,
        row: dict[str, Any],
        *,
        corpus_root: Path,
        manifest_parent: Path,
    ) -> "CommonManifestRow":
        split = _text(row, "split")
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}, received {split!r}")

        def audio(language: str) -> str:
            raw = row.get(f"{language}_audio", row.get(f"{language}_wav_16k"))
            if not isinstance(raw, str) or not raw.strip():
                raise ValueError(f"{language}_audio or {language}_wav_16k is required")
            path = Path(raw)
            if not path.is_absolute():
                rooted = corpus_root / path
                path = rooted if rooted.exists() else manifest_parent / path
            return str(path.resolve())

        return cls(
            pair_id=_text(row, "pair_id"),
            split=split,
            corpus=_text(row, "corpus"),
            source_id=_text(row, "source_id"),
            ja_audio=audio("ja"),
            en_audio=audio("en"),
            ja_text_raw=_text(row, "ja_text_raw", fallback="ja_text"),
            en_text_raw=_text(row, "en_text_raw", fallback="en_text"),
            ja_text=_text(row, "ja_text"),
            en_text=_text(row, "en_text"),
            ja_tts_text=_text(row, "ja_tts_text", fallback="ja_text"),
            en_tts_text=_text(row, "en_tts_text", fallback="en_text"),
            ja_duration=_number(row, "ja_duration"),
            en_duration=_number(row, "en_duration"),
            ja_sha256=_text(row, "ja_sha256"),
            en_sha256=_text(row, "en_sha256"),
        )

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "CommonManifestRow":
        missing = [name for name in cls.__dataclass_fields__ if name not in row]
        if missing:
            raise ValueError(f"missing required fields: {', '.join(missing)}")
        value = cls(**{name: row[name] for name in cls.__dataclass_fields__})
        if value.split not in SPLITS:
            raise ValueError(f"invalid split: {value.split}")
        for name in ("pair_id", "corpus", "source_id", "ja_text", "en_text"):
            if not isinstance(getattr(value, name), str) or not getattr(value, name).strip():
                raise ValueError(f"{name} must be a non-empty string")
        return value

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
