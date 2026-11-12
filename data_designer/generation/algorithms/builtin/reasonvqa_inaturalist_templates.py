from __future__ import annotations

from copy import deepcopy
from typing import Any


INATURALIST_TEMPLATE_VERSION = "inat-reasonvqa-template-v1"
INATURALIST_INTERMEDIATE_PROPERTY_IDS = ("P171", "P183")
INATURALIST_LINKING_ONLY_PROPERTY_IDS = ("P3151",)


def inaturalist_reasonvqa_template_defaults() -> dict[str, Any]:
    return {
        "template_version": INATURALIST_TEMPLATE_VERSION,
        "visual_anchor_template": "the species shown in the image",
        "difficulty_policy": {
            "negative_strategy": "wikidata_taxonomy_neighbors",
            "max_taxonomy_distance": "family",
            "drop_rows_without_hard_negatives": True,
        },
        "question_templates": {
            "P141": {
                "main": "What is the IUCN conservation status of {subject}?",
                "sub_clause": "the IUCN conservation status of {subject}",
            },
            "P171": {
                "main": "Which parent taxon contains {subject}?",
                "sub_clause": "the parent taxon containing {subject}",
            },
            "P183": {
                "main": "Where is {subject} endemic?",
                "sub_clause": "the place where {subject} is endemic",
            },
            "P1843": {
                "main": "What is a common English name for {subject}?",
                "sub_clause": "a common English name for {subject}",
            },
            "P225": {
                "main": "What is the scientific name of {subject}?",
                "sub_clause": "the scientific name of {subject}",
            },
        },
        "route_question_templates": {
            "P171 -> P141": (
                "What is the IUCN conservation status of the parent taxon of "
                "{subject}?"
            ),
            "P171 -> P1843": (
                "What is a common English name for the parent taxon of {subject}?"
            ),
            "P171 -> P225": (
                "What is the scientific name of the parent taxon of {subject}?"
            ),
            "P171 -> P171 -> P141": (
                "What is the IUCN conservation status of the taxonomic group "
                "two levels above {subject}?"
            ),
            "P171 -> P171 -> P1843": (
                "What is a common English name for the taxonomic group two "
                "levels above {subject}?"
            ),
            "P171 -> P171 -> P225": (
                "What is the scientific name of the taxonomic group two levels "
                "above {subject}?"
            ),
            "P183 -> P1843": (
                "What is a common English name for the place where {subject} "
                "is endemic?"
            ),
            "P183 -> P225": (
                "What is the scientific name associated with the place where "
                "{subject} is endemic?"
            ),
        },
    }


def with_inaturalist_reasonvqa_template_defaults(
    config: dict[str, Any],
) -> dict[str, Any]:
    merged = inaturalist_reasonvqa_template_defaults()
    incoming = deepcopy(config)
    for key, value in incoming.items():
        if key in {"question_templates", "route_question_templates"}:
            base_templates = dict(merged.get(key) or {})
            if isinstance(value, dict):
                base_templates.update(value)
            merged[key] = base_templates
        else:
            merged[key] = value
    return merged


def inaturalist_question_template_policy(
    config: dict[str, Any] | None = None,
) -> dict[str, list[str]]:
    templates = (config or inaturalist_reasonvqa_template_defaults()).get(
        "question_templates",
        {},
    )
    template_property_ids = [str(property_id) for property_id in templates]
    intermediate_property_ids = list(INATURALIST_INTERMEDIATE_PROPERTY_IDS)
    linking_only_property_ids = list(INATURALIST_LINKING_ONLY_PROPERTY_IDS)
    final_property_ids = [
        property_id
        for property_id in template_property_ids
        if property_id not in intermediate_property_ids
        and property_id not in linking_only_property_ids
    ]
    return {
        "intermediate_property_ids": intermediate_property_ids,
        "final_property_ids": final_property_ids,
        "linking_only_property_ids": linking_only_property_ids,
    }


def is_inaturalist_question_template_usable_path(
    path: dict[str, Any],
    policy: dict[str, list[str]] | None = None,
) -> bool:
    template_policy = policy or inaturalist_question_template_policy()
    route = [str(item) for item in path.get("route_properties") or []]
    if not route:
        return False
    if route[-1] not in set(template_policy["final_property_ids"]):
        return False
    return all(
        property_id in set(template_policy["intermediate_property_ids"])
        for property_id in route[:-1]
    )


__all__ = [
    "INATURALIST_TEMPLATE_VERSION",
    "inaturalist_reasonvqa_template_defaults",
    "inaturalist_question_template_policy",
    "is_inaturalist_question_template_usable_path",
    "with_inaturalist_reasonvqa_template_defaults",
]
