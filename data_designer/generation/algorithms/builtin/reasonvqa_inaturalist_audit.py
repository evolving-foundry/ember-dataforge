from __future__ import annotations

from collections import Counter
from typing import Any

from data_designer.generation.algorithms.builtin.reasonvqa_inaturalist_templates import (
    inaturalist_question_template_policy,
    is_inaturalist_question_template_usable_path,
)


def summarize_inaturalist_kg_coverage(
    *,
    sampled_count: int,
    linked_records: list[dict[str, Any]],
    root_claim_triples: list[dict[str, Any]],
    paths: list[dict[str, Any]],
) -> dict[str, Any]:
    linked_qids = {
        str(record.get("root_concept", {}).get("wikidata_qid") or "").strip()
        for record in linked_records
    }
    linked_qids.discard("")
    linked_count = len(linked_qids)
    paths_by_root: dict[str, list[dict[str, Any]]] = {}
    for path in paths:
        root_qid = str(path.get("root", {}).get("qid") or "").strip()
        if root_qid:
            paths_by_root.setdefault(root_qid, []).append(path)

    roots_by_hop = {
        hop: {
            root_qid
            for root_qid, root_paths in paths_by_root.items()
            if any(int(path.get("hop") or 0) == hop for path in root_paths)
        }
        for hop in (1, 2, 3)
    }
    root_claim_counts = Counter(
        str(triple.get("property", {}).get("id") or "")
        for triple in root_claim_triples
    )
    path_claim_counts: Counter[str] = Counter()
    path_group_counts: Counter[str] = Counter()
    for path in paths:
        for triple in path.get("path") or []:
            path_claim_counts[str(triple.get("property", {}).get("id") or "")] += 1
            path_group_counts[str(triple.get("property", {}).get("group") or "")] += 1

    terminal_answer_type_distribution = Counter(
        _terminal_answer_type(path) for path in paths
    )
    answer_counts = Counter(_terminal_answer_key(path) for path in paths)
    answer_counts.pop("", None)
    duplicate_answer_paths = sum(
        count for count in answer_counts.values() if count > 1
    )
    geography_roots = {
        str(triple.get("root", {}).get("qid") or "").strip()
        for triple in root_claim_triples
        if str(triple.get("property", {}).get("group") or "") == "geography"
    }
    geography_roots.discard("")
    usable_paths = [
        path for path in paths if is_inaturalist_question_template_usable_path(path)
    ]
    path_count = len(paths)
    return {
        "sampled_count": sampled_count,
        "linked_root_count": linked_count,
        "path_count": path_count,
        "rates": {
            "link_success_rate": _safe_rate(linked_count, sampled_count),
            "one_hop_path_rate": _safe_rate(len(roots_by_hop[1]), linked_count),
            "two_hop_path_rate": _safe_rate(len(roots_by_hop[2]), linked_count),
            "three_hop_path_rate": _safe_rate(len(roots_by_hop[3]), linked_count),
            "geography_coverage": _safe_rate(
                len(geography_roots & linked_qids),
                linked_count,
            ),
            "duplicate_answer_rate": _safe_rate(
                duplicate_answer_paths,
                path_count,
            ),
            "question_template_usable_rate": _safe_rate(
                len(usable_paths),
                path_count,
            ),
        },
        "root_counts_by_hop": {
            "1": len(roots_by_hop[1]),
            "2": len(roots_by_hop[2]),
            "3": len(roots_by_hop[3]),
        },
        "property_hit_distribution": {
            "root_claim_counts": dict(
                sorted(_drop_empty_counts(root_claim_counts).items())
            ),
            "path_claim_counts": dict(
                sorted(_drop_empty_counts(path_claim_counts).items())
            ),
            "path_group_counts": dict(
                sorted(_drop_empty_counts(path_group_counts).items())
            ),
        },
        "terminal_answer_type_distribution": dict(
            sorted(_drop_empty_counts(terminal_answer_type_distribution).items())
        ),
        "duplicate_answer_count": duplicate_answer_paths,
        "question_template_usable_path_count": len(usable_paths),
        "question_template_policy": inaturalist_question_template_policy(),
    }


def _safe_rate(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def _terminal_answer_type(path: dict[str, Any]) -> str:
    path_triples = path.get("path") or []
    if path_triples:
        answer_type = str(
            path_triples[-1].get("property", {}).get("answer_type") or ""
        ).strip()
        if answer_type:
            return answer_type
    return str(path.get("final_answer", {}).get("type") or "").strip()


def _terminal_answer_key(path: dict[str, Any]) -> str:
    final_answer = path.get("final_answer") or {}
    value = final_answer.get("label") or final_answer.get("value") or ""
    return " ".join(str(value).casefold().split())


def _drop_empty_counts(counter: Counter[str]) -> Counter[str]:
    counter.pop("", None)
    return counter


__all__ = [
    "summarize_inaturalist_kg_coverage",
]
