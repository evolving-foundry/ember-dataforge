"""Seed contamination detection (design §8.2-1): n-gram similarity between new items and real seeds.

The purpose of expansion is to fill blind spots, not to copy original items indirectly (otherwise evaluation measures memorization, not capability).
Detection: character 3-gram Jaccard similarity; if a new item exceeds the threshold against any seed, mark
``derivative=true`` and quarantine by default (audit auto_fail), with the report saved in the manifest.

The design recommends reusing the deduplication operator family's similarity tier, but that tier
(``SimilarityBasedDedupOperation`` / ``FuzzyMatchingDedupOperation``)
is currently a NotImplementedError stub (verified 2026-09-20), so this module uses a built-in zero-dependency implementation;
it can be replaced seamlessly when those operators land.
"""

from __future__ import annotations

from typing import Any, Dict, List

DEFAULT_CONTAMINATION_THRESHOLD = 0.55
NGRAM_SIZE = 3


def _char_ngrams(text: str, n: int = NGRAM_SIZE) -> set:
    normalized = "".join((text or "").lower().split())
    if len(normalized) < n:
        return {normalized} if normalized else set()
    return {normalized[i:i + n] for i in range(len(normalized) - n + 1)}


def jaccard_similarity(a: str, b: str) -> float:
    ga, gb = _char_ngrams(a), _char_ngrams(b)
    if not ga or not gb:
        return 0.0
    return len(ga & gb) / len(ga | gb)


def contamination_report(
    new_items: List[Dict[str, Any]],
    seed_items: List[Dict[str, Any]],
    threshold: float = DEFAULT_CONTAMINATION_THRESHOLD,
) -> Dict[str, Any]:
    """Compute similarity for each new item against all seeds; items above the threshold go into flagged.

    new_items/seed_items elements must contain task_text; identifier keys prefer
    task_id / seed_id respectively. Returns a manifest-serializable report dict.
    """
    flagged: List[Dict[str, Any]] = []
    for new in new_items or []:
        new_id = str(new.get("task_id") or new.get("seed_id") or "")
        best_sim, best_seed = 0.0, None
        for seed in seed_items or []:
            sim = jaccard_similarity(str(new.get("task_text") or ""),
                                     str(seed.get("task_text") or ""))
            if sim > best_sim:
                best_sim, best_seed = sim, str(
                    seed.get("seed_id") or seed.get("task_id") or "")
        if best_sim >= threshold:
            flagged.append({
                "task_id": new_id,
                "max_similarity": round(best_sim, 4),
                "matched_seed_id": best_seed,
            })
    return {
        "threshold": float(threshold),
        "ngram_size": NGRAM_SIZE,
        "checked": len(new_items or []),
        "seed_count": len(seed_items or []),
        "flagged": flagged,
        "flagged_count": len(flagged),
    }


__all__ = [
    "contamination_report",
    "jaccard_similarity",
    "DEFAULT_CONTAMINATION_THRESHOLD",
    "NGRAM_SIZE",
]
