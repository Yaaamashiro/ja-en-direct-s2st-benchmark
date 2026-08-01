from __future__ import annotations

import re
import unicodedata


def normalize_english(text: str) -> str:
    value = unicodedata.normalize("NFKC", text).lower()
    value = re.sub(r"[^\w'\s]", " ", value, flags=re.UNICODE)
    return re.sub(r"\s+", " ", value).strip()


def sentence_bleu(hypothesis: str, reference: str) -> float:
    import sacrebleu

    return round(float(sacrebleu.sentence_bleu(hypothesis, [reference]).score), 12)


def corpus_bleu(hypotheses: list[str], references: list[str]) -> float | None:
    if not hypotheses:
        return None
    import sacrebleu

    return round(float(sacrebleu.corpus_bleu(hypotheses, [references]).score), 12)
