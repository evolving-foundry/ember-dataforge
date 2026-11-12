"""Text overlap soft scoring (Q2 ``CandidateTextOverlapScore``): reuses the filtering operator family implementation.

Scoring logic mirrors the implementations of ``BleuSampleFilterOperation`` / ``CiderSampleFilterOperation``
(same tokenizer backend ``get_nlp_backend``, same BLEU/CIDEr math), changing only
'threshold filtering' into 'return per-pair scores'; ngram is the clipped n-gram precision
of candidate against reference. Pure Python pairwise computation, no Ray dependency, byte-identical on recomputation for the same input.
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Callable, List

from curator.utils.lazy_imports import (
    get_nlp_backend,
    simple_ngrams,
    simple_tokenize,
)

METRICS = ("bleu", "cider", "ngram")


_TOKENIZER: Callable[[str], List[str]] | None = None


def _make_tokenizer() -> Callable[[str], List[str]]:
    """Same as BleuSampleFilterOperation: use nltk when available; fall back to simple_tokenize on missing data/errors."""
    global _TOKENIZER
    if _TOKENIZER is not None:
        return _TOKENIZER
    try:
        nlp = get_nlp_backend()
        have_nltk = nlp.ensure_nltk()
    except Exception:
        # The nltk import chain (nltk→scipy) may fail depending on the host environment (for example, conflicts between a torch stub and scipy's
        # getattr checks); fall back to simple_tokenize, matching the fault-tolerance semantics of the operator family.
        have_nltk = False
    if have_nltk:

        def tokenize(text: str) -> List[str]:
            try:
                return nlp.word_tokenize(str(text).lower())
            except Exception:
                return simple_tokenize(str(text))

    else:

        def tokenize(text: str) -> List[str]:
            return simple_tokenize(str(text))

    _TOKENIZER = tokenize
    return _TOKENIZER


def score_pair(metric: str, candidate: str, reference: str, n: int) -> float:
    """Return a [0, 1] similarity score; an empty candidate / reference records 0.0."""
    if not candidate or not reference:
        return 0.0
    tokenize = _make_tokenizer()
    cand_tokens = tokenize(candidate)
    ref_tokens = tokenize(reference)
    if not cand_tokens or not ref_tokens:
        return 0.0

    if metric == "bleu":
        return _bleu(cand_tokens, ref_tokens, n)
    if metric == "cider":
        return _cider(cand_tokens, ref_tokens)
    return _ngram_precision(cand_tokens, ref_tokens, n)


def _ngrams(tokens: List[str], n: int) -> List[tuple]:
    try:
        from nltk import ngrams as nltk_ngrams

        return list(nltk_ngrams(tokens, n))
    except Exception:
        return simple_ngrams(tokens, n)


def _bleu(cand_tokens: List[str], ref_tokens: List[str], n: int) -> float:
    """Same as BleuSampleFilterOperation: nltk sentence_bleu + method4 smoothing."""
    try:
        from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

        weights = tuple([1.0 / n] * n)
        return float(
            sentence_bleu(
                [ref_tokens],
                cand_tokens,
                weights=weights,
                smoothing_function=SmoothingFunction().method4,
            )
        )
    except Exception:
        # Same shape as the operator family's fallback: simple word hit rate
        ref_set = set(ref_tokens)
        matches = sum(1 for w in cand_tokens if w in ref_set)
        return matches / max(len(cand_tokens), 1)


def _cider(cand_tokens: List[str], ref_tokens: List[str]) -> float:
    """Same as CiderSampleFilterOperation: n=4, sigma=6.0, constant-IDF fallback."""
    n = 4
    sigma = 6.0
    idf_value = math.log(100000.0)  # log of the operator family's default corpus_size
    score_levels = []
    for k in range(1, n + 1):
        cand_counts = Counter(_ngrams(cand_tokens, k))
        ref_counts = Counter(_ngrams(ref_tokens, k))
        dot = 0.0
        norm_cand = 0.0
        norm_ref = 0.0
        for gram in set(cand_counts) | set(ref_counts):
            cand_c = cand_counts.get(gram, 0)
            ref_c = ref_counts.get(gram, 0)
            dot += min(cand_c, ref_c) * (idf_value ** 2)
            norm_cand += (cand_c * idf_value) ** 2
            norm_ref += (ref_c * idf_value) ** 2
        norm_cand = math.sqrt(norm_cand)
        norm_ref = math.sqrt(norm_ref)
        score_levels.append(
            dot / (norm_cand * norm_ref) if norm_cand > 0 and norm_ref > 0 else 0.0
        )
    avg = sum(score_levels) / len(score_levels)
    penalty = math.exp(-((len(cand_tokens) - len(ref_tokens)) ** 2) / (2 * sigma ** 2))
    return avg * penalty


def _ngram_precision(cand_tokens: List[str], ref_tokens: List[str], n: int) -> float:
    """Clipped precision of candidate n-grams against the reference."""
    cand_grams = _ngrams(cand_tokens, n)
    if not cand_grams:
        return 0.0
    ref_counts = Counter(_ngrams(ref_tokens, n))
    cand_counts = Counter(cand_grams)
    clipped = sum(min(c, ref_counts.get(g, 0)) for g, c in cand_counts.items())
    return clipped / len(cand_grams)
