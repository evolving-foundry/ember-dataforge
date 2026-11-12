from __future__ import annotations

from decimal import Decimal, InvalidOperation
import hashlib
import re
from typing import Any


LEGACY_LITERAL_STRATEGIES = {
    "same_route_property",
    "typed_same_property",
    "typed_same_route_property",
    "legacy_same_route_property",
}
TAXONOMY_LITERAL_STRATEGIES = {
    "wikidata_taxonomy_neighbors",
    "taxonomy_hard",
    "typed_taxonomic_neighbors",
    "fine_grained_taxonomy",
}
FIXED_OPTION_SETS = {
    "P141": [
        "least concern",
        "near threatened",
        "vulnerable",
        "endangered",
        "critically endangered",
        "extinct in the wild",
        "extinct",
        "data deficient",
        "not evaluated",
    ],
}
TAXONOMY_NEIGHBOR_FIELDS = (
    "genus",
    "family",
    "order",
    "class",
    "phylum",
    "kingdom",
)
TAXONOMY_RELATION_DISTANCE = {
    "visual_neighbor": 0,
    "same_genus": 0,
    "same_family": 1,
    "same_order": 2,
    "same_class": 3,
    "same_phylum": 4,
    "same_kingdom": 5,
}


def typed_answer_options(
    *,
    answer: str,
    answer_type: str,
    property_id: str | None = None,
    kg_paths: list[dict[str, Any]] | None = None,
    reference_path_record: dict[str, Any] | None = None,
    candidate_values: list[Any] | None = None,
    fixed_values: list[str] | None = None,
    negative_strategy: str | None = None,
    max_taxonomy_distance: str | int | None = None,
    limit: int = 3,
) -> list[str]:
    distractors = typed_distractors(
        answer=answer,
        answer_type=answer_type,
        property_id=property_id,
        kg_paths=kg_paths,
        reference_path_record=reference_path_record,
        candidate_values=candidate_values,
        fixed_values=fixed_values,
        negative_strategy=negative_strategy,
        max_taxonomy_distance=max_taxonomy_distance,
        limit=limit,
    )
    options = [str(answer)] + distractors
    if len(set(_norm(option) for option in options)) != len(options):
        raise ValueError(f"Typed answer options are not unique for {property_id}.")
    return options


def typed_distractors(
    *,
    answer: str,
    answer_type: str,
    property_id: str | None = None,
    kg_paths: list[dict[str, Any]] | None = None,
    reference_path_record: dict[str, Any] | None = None,
    candidate_values: list[Any] | None = None,
    fixed_values: list[str] | None = None,
    negative_strategy: str | None = None,
    max_taxonomy_distance: str | int | None = None,
    limit: int = 3,
) -> list[str]:
    normalized_type = str(answer_type or "").strip().casefold()
    if normalized_type == "external_id":
        raise ValueError("external-id answers are not used for QA distractors.")
    if normalized_type == "literal":
        strategy = _literal_negative_strategy(negative_strategy)
        max_distance = _max_taxonomy_distance(max_taxonomy_distance)
        if strategy == "taxonomy_neighbors":
            candidates = [
                *_candidate_values_by_taxonomy_relation(
                    candidate_values or [],
                    max_distance=max_distance,
                    reference_path_record=reference_path_record,
                    property_id=property_id,
                    answer=answer,
                ),
                *_taxonomic_literal_values_for_property(
                    kg_paths or [],
                    property_id,
                    reference_path_record=reference_path_record,
                    max_distance=max_distance,
                ),
            ]
            return _first_unique_distractors(answer, candidates, limit)

        taxonomic_candidates = _taxonomic_literal_values_for_property(
            kg_paths or [],
            property_id,
            reference_path_record=reference_path_record,
        )
        literal_candidates = _literal_values_for_property(
            kg_paths or [],
            property_id,
            reference_path_record=reference_path_record,
        )
        taxonomic_seen = {_norm(candidate) for candidate in taxonomic_candidates}
        literal_candidates = _shape_matched_literal_values(
            [
                candidate
                for candidate in literal_candidates
                if _norm(candidate) not in taxonomic_seen
            ],
            answer=answer,
            property_id=property_id,
            limit=limit,
        )
        fallback_candidates = _stable_literal_candidate_order(
            literal_candidates,
            answer=answer,
            property_id=property_id,
            reference_path_record=reference_path_record,
        )
        candidates = [
            *taxonomic_candidates,
            *fallback_candidates,
            *_plain_candidate_values(candidate_values or []),
        ]
        return _first_unique_distractors(answer, candidates, limit)
    if normalized_type == "fixed":
        candidates = fixed_values or FIXED_OPTION_SETS.get(str(property_id or ""), [])
        return _first_unique_distractors(answer, candidates, limit)
    if normalized_type == "number":
        return _number_distractors(answer, limit)
    if normalized_type == "date":
        return _date_distractors(answer, limit)
    raise ValueError(f"Unsupported typed distractor answer_type: {answer_type}")


def balanced_answer_view(
    answer: str,
    answers: list[str],
    question_id: int,
) -> tuple[list[str], list[int]]:
    target_index = question_id % len(answers)
    negatives = [candidate for candidate in answers if candidate != answer]
    balanced_answers = negatives[:target_index] + [answer] + negatives[target_index:]
    correct = [1 if candidate == answer else 0 for candidate in balanced_answers]
    if sum(correct) != 1:
        raise ValueError(
            f"Balanced answer view lost unique answer for qid={question_id}."
        )
    return balanced_answers, correct


def _literal_values_for_property(
    kg_paths: list[dict[str, Any]],
    property_id: str | None,
    *,
    reference_path_record: dict[str, Any] | None = None,
) -> list[str]:
    reference_route = (
        _route_key(reference_path_record) if reference_path_record is not None else None
    )
    values: list[str] = []
    for path in kg_paths:
        route = _route_key(path)
        if property_id and route and route[-1] != property_id:
            continue
        if reference_route is not None and route != reference_route:
            continue
        if _same_final_target(path, reference_path_record):
            continue
        final_answer = path.get("final_answer") or {}
        value = final_answer.get("label") or final_answer.get("value")
        if value is not None:
            values.append(str(value))
    return values


def _taxonomic_literal_values_for_property(
    kg_paths: list[dict[str, Any]],
    property_id: str | None,
    *,
    reference_path_record: dict[str, Any] | None = None,
    max_distance: int | None = None,
) -> list[str]:
    if reference_path_record is None:
        return []
    reference_route = _route_key(reference_path_record)
    seed = _stable_seed(
        answer=str(
            (reference_path_record.get("final_answer") or {}).get("label")
            or (reference_path_record.get("final_answer") or {}).get("value")
            or ""
        ),
        property_id=property_id,
        reference_path_record=reference_path_record,
    )
    ranked: list[tuple[int, str, str]] = []
    for path in kg_paths:
        route = _route_key(path)
        if property_id and route and route[-1] != property_id:
            continue
        if route != reference_route:
            continue
        if _same_final_target(path, reference_path_record):
            continue
        distance = _taxonomic_distance(reference_path_record, path)
        if distance is None:
            continue
        if max_distance is not None and distance > max_distance:
            continue
        final_answer = path.get("final_answer") or {}
        value = final_answer.get("label") or final_answer.get("value")
        if value is None:
            continue
        ranked.append(
            (
                distance,
                _stable_digest(seed, f"{distance}\0{_norm(str(value))}"),
                str(value),
            )
        )
    return [value for _, _, value in sorted(ranked)]


def _literal_negative_strategy(strategy: str | None) -> str:
    if strategy is None:
        return "same_route_property"
    normalized = str(strategy).strip().casefold()
    if normalized in LEGACY_LITERAL_STRATEGIES:
        return "same_route_property"
    if normalized in TAXONOMY_LITERAL_STRATEGIES:
        return "taxonomy_neighbors"
    raise ValueError(f"Unsupported literal negative strategy: {strategy}")


def _max_taxonomy_distance(value: str | int | None) -> int | None:
    if value is None or value == "":
        return None
    if isinstance(value, int):
        if value < 0:
            raise ValueError("max_taxonomy_distance must be non-negative.")
        return value
    normalized = str(value).strip().casefold()
    if normalized.isdigit():
        return int(normalized)
    aliases = {
        "genus": 0,
        "same_genus": 0,
        "family": 1,
        "same_family": 1,
        "order": 2,
        "same_order": 2,
        "class": 3,
        "same_class": 3,
        "phylum": 4,
        "same_phylum": 4,
        "kingdom": 5,
        "same_kingdom": 5,
    }
    try:
        return aliases[normalized]
    except KeyError as exc:
        raise ValueError(f"Unknown max_taxonomy_distance: {value}") from exc


def _candidate_values_by_taxonomy_relation(
    candidate_values: list[Any],
    *,
    max_distance: int | None,
    reference_path_record: dict[str, Any] | None,
    property_id: str | None,
    answer: str,
) -> list[str]:
    seed = (
        _stable_seed(
            answer=answer,
            property_id=property_id,
            reference_path_record=reference_path_record,
        )
        if reference_path_record is not None
        else f"{property_id}\0{answer}"
    )
    ranked: list[tuple[int, str, str]] = []
    for candidate in candidate_values:
        value, relation = _candidate_value_and_relation(candidate)
        if not value:
            continue
        distance = TAXONOMY_RELATION_DISTANCE.get(relation, 999)
        if max_distance is not None and distance > max_distance:
            continue
        ranked.append(
            (
                distance,
                _stable_digest(seed, f"{distance}\0{_norm(value)}"),
                value,
            )
        )
    return [value for _, _, value in sorted(ranked)]


def _plain_candidate_values(candidate_values: list[Any]) -> list[str]:
    values = []
    for candidate in candidate_values:
        value, _relation = _candidate_value_and_relation(candidate)
        if value:
            values.append(value)
    return values


def _candidate_value_and_relation(candidate: Any) -> tuple[str, str]:
    if isinstance(candidate, dict):
        value = (
            candidate.get("label")
            or candidate.get("value")
            or candidate.get("taxon_name")
            or candidate.get("common_name")
            or ""
        )
        return str(value).strip(), str(candidate.get("relation") or "").strip()
    return str(candidate).strip(), ""


def _route_key(path_record: dict[str, Any]) -> tuple[str, ...]:
    route = path_record.get("route_properties")
    if not isinstance(route, list):
        route = [
            str(triple.get("property", {}).get("id") or "")
            for triple in path_record.get("path") or []
            if isinstance(triple, dict)
        ]
    return tuple(str(item).strip() for item in route if str(item).strip())


def _same_final_target(
    path_record: dict[str, Any],
    reference_path_record: dict[str, Any] | None,
) -> bool:
    if reference_path_record is None:
        return False
    return _final_target_key(path_record) == _final_target_key(reference_path_record)


def _final_target_key(path_record: dict[str, Any]) -> tuple[str, str]:
    path = path_record.get("path") or []
    final_triple = path[-1] if path else {}
    final_root = final_triple.get("root") or {}
    final_triple_value = final_triple.get("triple") or {}
    qid = (
        final_root.get("qid")
        or final_triple_value.get("subject")
        or (path_record.get("root") or {}).get("qid")
        or (path_record.get("root_concept") or {}).get("wikidata_qid")
        or ""
    )
    route = " > ".join(_route_key(path_record))
    return route, str(qid)


def _taxonomic_distance(
    reference_path_record: dict[str, Any],
    candidate_path_record: dict[str, Any],
) -> int | None:
    reference_taxonomy = _taxonomy(reference_path_record)
    candidate_taxonomy = _taxonomy(candidate_path_record)
    if not reference_taxonomy or not candidate_taxonomy:
        return None
    for distance, field in enumerate(TAXONOMY_NEIGHBOR_FIELDS):
        reference_value = _norm(reference_taxonomy.get(field, ""))
        candidate_value = _norm(candidate_taxonomy.get(field, ""))
        if reference_value and reference_value == candidate_value:
            return distance
    return None


def _taxonomy(path_record: dict[str, Any]) -> dict[str, Any]:
    inaturalist = path_record.get("inaturalist") or {}
    taxonomy = inaturalist.get("taxonomy") or {}
    return taxonomy if isinstance(taxonomy, dict) else {}


def _first_unique_distractors(
    answer: str,
    candidates: list[str],
    limit: int,
) -> list[str]:
    distractors: list[str] = []
    seen = {_norm(answer)}
    for candidate in candidates:
        candidate_text = str(candidate)
        key = _norm(candidate_text)
        if not key or key in seen:
            continue
        seen.add(key)
        distractors.append(candidate_text)
        if len(distractors) >= limit:
            return distractors
    raise ValueError(f"Not enough typed distractors for answer {answer}.")


def _stable_literal_candidate_order(
    candidates: list[str],
    *,
    answer: str,
    property_id: str | None,
    reference_path_record: dict[str, Any] | None,
) -> list[str]:
    if reference_path_record is None:
        return candidates
    seed = _stable_seed(
        answer=answer,
        property_id=property_id,
        reference_path_record=reference_path_record,
    )
    return sorted(
        candidates,
        key=lambda candidate: _stable_digest(seed, _norm(candidate)),
    )


def _shape_matched_literal_values(
    candidates: list[str],
    *,
    answer: str,
    property_id: str | None,
    limit: int,
) -> list[str]:
    if property_id != "P1843":
        return candidates
    target_shape = _common_name_shape(answer)
    if not target_shape:
        return candidates
    shaped = [
        candidate
        for candidate in candidates
        if _common_name_shape(candidate) == target_shape
    ]
    if _unique_distractor_count(answer, shaped) >= limit:
        return shaped
    return candidates


def _common_name_shape(value: str) -> str:
    tokens = _norm(value).split()
    if not tokens:
        return ""
    last = tokens[-1]
    if last in {"family", "families"}:
        return "rank:family"
    if len(last) > 3 and last.endswith("s"):
        last = last[:-1]
    return f"head:{last}"


def _unique_distractor_count(answer: str, candidates: list[str]) -> int:
    seen = {_norm(answer)}
    count = 0
    for candidate in candidates:
        key = _norm(candidate)
        if not key or key in seen:
            continue
        seen.add(key)
        count += 1
    return count


def _stable_seed(
    *,
    answer: str,
    property_id: str | None,
    reference_path_record: dict[str, Any],
) -> str:
    root = reference_path_record.get("root") or {}
    root_concept = reference_path_record.get("root_concept") or {}
    image = reference_path_record.get("image") or {}
    return "\0".join(
        [
            str(property_id or ""),
            " > ".join(_route_key(reference_path_record)),
            str(answer),
            str(root.get("qid") or root_concept.get("wikidata_qid") or ""),
            str(root.get("label") or root_concept.get("label") or ""),
            str(
                image.get("image_id")
                or image.get("file_name")
                or image.get("url")
                or ""
            ),
        ]
    )


def _stable_digest(seed: str, candidate: str) -> str:
    return hashlib.sha256(f"{seed}\0{candidate}".encode("utf-8")).hexdigest()


def _number_distractors(answer: str, limit: int) -> list[str]:
    try:
        value = Decimal(str(answer).strip())
    except InvalidOperation as exc:
        raise ValueError(f"Number answer is not numeric: {answer}") from exc
    decimal_places = max(0, -value.as_tuple().exponent)
    step = Decimal(1).scaleb(-decimal_places)
    candidates = [
        _format_decimal(value - step),
        _format_decimal(value + step),
        _format_decimal(value + (step * 2)),
        _format_decimal(value - (step * 2)),
    ]
    return _first_unique_distractors(str(answer), candidates, limit)


def _date_distractors(answer: str, limit: int) -> list[str]:
    year_match = re.search(r"-?\d{1,4}", str(answer))
    if not year_match:
        raise ValueError(f"Date answer does not contain a year: {answer}")
    year = int(year_match.group(0))
    candidates = [
        str(year - 1),
        str(year + 1),
        str(year - 2),
        str(year + 2),
    ]
    return _first_unique_distractors(str(answer), candidates, limit)


def _format_decimal(value: Decimal) -> str:
    if value == value.to_integral_value():
        return str(value.quantize(Decimal(1)))
    return format(value.normalize(), "f")


def _norm(value: str) -> str:
    return " ".join(str(value).casefold().split())


__all__ = [
    "balanced_answer_view",
    "typed_answer_options",
    "typed_distractors",
]
