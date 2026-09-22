"""Explicit, versioned evaluation rules; no substring or token-F1 credit."""
from __future__ import annotations

import re
from functools import lru_cache

CHOICE_PROTOCOL = "choice-set-exact-or-exact-option-text-v1"
WORDNET_PROTOCOL = "lowercase-exact-or-wordnet-lemma-path>=0.8-v1"


def normalize_text(value: str) -> str:
    value = re.sub(r"</?answer>", "", str(value), flags=re.I)
    return " ".join(value.strip().casefold().split())


def open_ended_match(pred: str, gt: str) -> tuple[bool, str]:
    p, g = normalize_text(pred), normalize_text(gt)
    return (bool(p and g and p == g), "normalized_exact")


def _choice_labels(text: str, option_count: int) -> set[str]:
    valid = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ"[:option_count])
    value = re.sub(r"</?answer>", "", str(text), flags=re.I).strip()
    value = re.sub(
        r"^(?:the\s+)?(?:best\s+|correct\s+)?(?:answer|option|choice)s?\s*(?:is|are|:)?\s*",
        "", value, flags=re.I,
    ).strip()
    compact = value.strip("<>[](){} .")
    compact = re.sub(r"\band\b", ",", compact, flags=re.I)
    if re.fullmatch(r"[A-Za-z](?:[\s,;/+&]+[A-Za-z])*", compact):
        labels = {x.upper() for x in re.findall(r"[A-Za-z]", compact)}
        return labels if labels <= valid else set()
    match = re.match(r"^[<(\[]?([A-Z])\s*(?:[)>\].:]|\s+-\s+)", value)
    if match and match.group(1) in valid:
        return {match.group(1)}
    return set()


def _option_text_labels(value: str, options: list[str]) -> set[str]:
    normalized = normalize_text(value)
    hits = [chr(65 + i) for i, option in enumerate(options)
            if normalized and normalize_text(option) == normalized]
    return set(hits) if len(hits) == 1 else set()


def multiple_choice_match(pred: str, gt: str, options: list[str]) -> tuple[bool, str]:
    if not options or len(options) > 26:
        raise ValueError("Choice scoring requires 1--26 options")
    target = _option_text_labels(gt, options) or _choice_labels(gt, len(options))
    if not target:
        raise ValueError(f"Ground truth is neither option labels nor unique option text: {gt!r}")
    predicted = _option_text_labels(pred, options) or _choice_labels(pred, len(options))
    return bool(predicted) and predicted == target, "choice_set_exact"


@lru_cache(maxsize=1)
def _wordnet_resources():
    from nltk.corpus import wordnet
    from nltk.stem import WordNetLemmatizer
    try:
        wordnet.ensure_loaded()
    except LookupError as exc:
        raise RuntimeError("Install WordNet: python -m nltk.downloader wordnet omw-1.4") from exc
    return wordnet, WordNetLemmatizer()


@lru_cache(maxsize=65536)
def wordnet_match(pred: str, gt: str) -> tuple[bool, str]:
    p, g = str(pred).lower().strip(), str(gt).lower().strip()
    if not p or not g:
        return False, "empty"
    if p == g:
        return True, "exact"
    wn, lemmatizer = _wordnet_resources()
    left, right = lemmatizer.lemmatize(p), lemmatizer.lemmatize(g)
    for a in wn.synsets(left):
        for b in wn.synsets(right):
            if (a.path_similarity(b) or 0.0) >= 0.8:
                return True, "wordnet"
    return False, "mismatch"


def build_scorer(dataset_name: str):
    key = dataset_name.lower().replace("_", "-")
    if key in {"lrs-gro", "lrsgro"}:
        _wordnet_resources()
        return (lambda pred, sample: wordnet_match(pred, sample.answer)), WORDNET_PROTOCOL
    if key in {"xlrs-bench", "xlrs", "xlrs-bench-lite", "mme-realworld-rs", "mme-rs", "mme-realworld"}:
        return (lambda pred, sample: multiple_choice_match(pred, sample.answer, sample.options)), CHOICE_PROTOCOL
    raise ValueError(f"No scoring protocol for {dataset_name!r}")
