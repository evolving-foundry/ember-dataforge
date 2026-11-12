"""Deterministic single-turn answer matchers — Python re-expressions of the
Evaluator ``score.ts`` semantics (gsm8k / math500 / mgsm / drop / mmlu /
mmlu_pro / gpqa / triviaqa).

Trusted derivation material: ``hf_single_turn`` source import embeds this
module's source text verbatim into frozen ``unit_test`` ``test_script``
payloads, so the file must stay self-contained (stdlib only, no repository
imports) and import-free at module level beyond the standard library.

Score semantics (R1): these are semantic re-expressions for this repository's
offline scoring trust model — never to be conflated with official NeMo/HF or
Evaluator leaderboard numbers.
"""

from __future__ import annotations

import re
import string
import unicodedata

_LETTER_RE = re.compile(
    r"(?is)answer\s*(?:is|:)?\s*\(?\s*([a-j])\s*\)?"
)
_PAREN_LETTER_RE = re.compile(r"(?is)\(([a-j])\)")
_STANDALONE_LETTER_RE = re.compile(r"(?im)^\s*\**([a-j])\**\s*\.?\s*$")
_NUMBER_RE = re.compile(r"(?i)-?\d[\d,\.\s]*\d|\d")
_BOXED_RE = re.compile(r"(?s)\\boxed\{")


def normalize_text(value: str) -> str:
    """Lowercase, strip punctuation/articles, collapse whitespace."""
    text = unicodedata.normalize("NFKC", str(value)).lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = [
        token for token in text.split() if token not in ("a", "an", "the")
    ]
    return " ".join(tokens)


def extract_boxed(text: str) -> str:
    """Content of the last ``\\boxed{...}`` with brace matching; '' if absent."""
    matches = list(_BOXED_RE.finditer(text or ""))
    if not matches:
        return ""
    start = matches[-1].end()
    depth = 1
    for index in range(start, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[start:index]
    return ""


def extract_after_marker(text: str, marker: str) -> str:
    """Text after the LAST occurrence of ``marker`` (answer-line semantics)."""
    lowered = (text or "").lower()
    position = lowered.rfind(marker.lower())
    if position < 0:
        return ""
    return text[position + len(marker):].strip()


def extract_last_number(text: str) -> str:
    """Last number-like token in the text; '' if absent."""
    matches = _NUMBER_RE.findall(text or "")
    if not matches:
        return ""
    return matches[-1].strip()


def normalize_numeric(value: str) -> str:
    """Canonical numeric string: strip $/%/spaces, drop thousands separators,
    trim trailing zeros after a decimal point."""
    token = (value or "").strip().replace("$", "").replace("%", "")
    token = token.replace(",", "").replace(" ", "")
    if not token:
        return ""
    match = re.fullmatch(r"(-?\d+)(?:\.(\d*))?", token)
    if not match:
        return token
    sign = "-" if match.group(1).startswith("-") else ""
    integer = match.group(1).lstrip("-").lstrip("0") or "0"
    fraction = (match.group(2) or "").rstrip("0")
    if sign == "-" and integer == "0" and not fraction:
        return "0"
    return f"{sign}{integer}.{fraction}" if fraction else f"{sign}{integer}"


def numeric_answer_equal(prediction: str, gold: str) -> bool:
    left = normalize_numeric(extract_last_number(prediction or ""))
    right = normalize_numeric(gold or "")
    return bool(left) and left == right


def multichoice_letter(prediction: str, allowed: str) -> str:
    """Extract the chosen option letter ('' if none) from a model reply."""
    text = prediction or ""
    match = _LETTER_RE.search(text)
    if match and match.group(1).upper() in allowed:
        return match.group(1).upper()
    for pattern in (_PAREN_LETTER_RE, _STANDALONE_LETTER_RE):
        for candidate in pattern.findall(text):
            if candidate.upper() in allowed:
                return candidate.upper()
    return ""


def multichoice_correct(prediction: str, gold_letter: str, allowed: str) -> bool:
    return multichoice_letter(prediction, allowed) == (gold_letter or "").upper()


def bag_of_words_f1(prediction: str, gold: str) -> float:
    """Token-level F1 between normalized texts (drop fuzzy-span semantics)."""
    predicted_tokens = normalize_text(prediction or "").split()
    gold_tokens = normalize_text(gold or "").split()
    if not predicted_tokens or not gold_tokens:
        return 0.0
    common: dict[str, int] = {}
    for token in predicted_tokens:
        common[token] = common.get(token, 0) + 1
    overlap = 0
    for token in gold_tokens:
        if common.get(token, 0) > 0:
            overlap += 1
            common[token] -= 1
    if overlap == 0:
        return 0.0
    precision = overlap / len(predicted_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def fuzzy_span_correct(prediction: str, gold_spans, threshold: float = 0.5) -> bool:
    spans = gold_spans if isinstance(gold_spans, list) else [gold_spans]
    return any(
        normalize_text(prediction or "") == normalize_text(span)
        or bag_of_words_f1(prediction or "", str(span)) >= threshold
        for span in spans
        if str(span).strip()
    )


def trivia_answer_correct(prediction: str, aliases) -> bool:
    """Normalized exact membership in the trusted alias table."""
    normalized = normalize_text(prediction or "")
    if not normalized:
        return False
    candidates = aliases if isinstance(aliases, list) else [aliases]
    return any(normalize_text(alias) == normalized for alias in candidates)


__all__ = [
    "normalize_text",
    "extract_boxed",
    "extract_after_marker",
    "extract_last_number",
    "normalize_numeric",
    "numeric_answer_equal",
    "multichoice_letter",
    "multichoice_correct",
    "bag_of_words_f1",
    "fuzzy_span_correct",
    "trivia_answer_correct",
]
