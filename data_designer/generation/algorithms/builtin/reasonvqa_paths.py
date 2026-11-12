from __future__ import annotations

from pathlib import Path
from typing import Any

from data_designer.generation.algorithms.builtin.reasonvqa_choices import (
    balanced_answer_view,
    typed_answer_options,
)
from data_designer.generation.algorithms.builtin.reasonvqa_inaturalist_templates import (
    is_inaturalist_question_template_usable_path,
    with_inaturalist_reasonvqa_template_defaults,
)
from data_designer.generation.algorithms.builtin.reasonvqa_templates import (
    TemplateEntity,
    TemplateProperty,
    question_for_route,
)


def generate_reasonvqa_rows_from_wikidata_paths(
    *,
    path_records: list[dict[str, Any]],
    config: dict[str, Any],
    recipe: str,
    generator_prompt_version: str,
    schema_version: str,
    params: dict[str, Any],
) -> list[dict[str, Any]]:
    config = with_inaturalist_reasonvqa_template_defaults(config)
    usable_paths = [
        path
        for path in path_records
        if _path_is_question_usable(path, config)
    ]
    policy = difficulty_policy(config)
    drop_without_hard_negatives = bool(
        policy.get("drop_rows_without_hard_negatives", False)
    )
    rows = []
    for index, path in enumerate(usable_paths):
        try:
            rows.append(
                generate_reasonvqa_row_from_wikidata_path(
                    path_record=dict(path),
                    all_path_records=usable_paths,
                    row_index=index,
                    config=config,
                    recipe=recipe,
                    generator_prompt_version=generator_prompt_version,
                    schema_version=schema_version,
                    params=params,
                )
            )
        except ValueError as exc:
            if drop_without_hard_negatives and "Not enough typed distractors" in str(exc):
                continue
            raise
    return rows


def generate_reasonvqa_row_from_wikidata_path(
    *,
    path_record: dict[str, Any],
    all_path_records: list[dict[str, Any]],
    row_index: int,
    config: dict[str, Any],
    recipe: str,
    generator_prompt_version: str,
    schema_version: str,
    params: dict[str, Any],
) -> dict[str, Any]:
    config = with_inaturalist_reasonvqa_template_defaults(config)
    route = property_route(path_record)
    path = path_triples(path_record)
    final_property = property_from_triple(path[-1])
    final_answer = answer_text(path_record)
    question_id = int(path_record.get("question_id", row_index))
    root_entity = root_template_entity(path_record, config)
    route_entities = route_template_entities(path_record, config)
    question_variant = int(path_record.get("question_variant", row_index))
    property_catalog = PathPropertyCatalog(path)
    question = question_for_route(
        root_entity,
        route,
        property_catalog,
        question_variant,
        config,
        route_entities,
    )
    answer_type = answer_type_from_triple(path[-1])
    policy = difficulty_policy(config)
    negative_strategy = str(
        path_record.get("synthetic_strategy")
        or policy.get("negative_strategy")
        or "same_route_property"
    )
    answers = typed_answer_options(
        answer=final_answer,
        answer_type=answer_type,
        property_id=final_property.property_id,
        kg_paths=all_path_records,
        reference_path_record=path_record,
        candidate_values=choice_candidates(path_record, route),
        negative_strategy=negative_strategy,
        max_taxonomy_distance=policy.get("max_taxonomy_distance"),
    )
    answers, correct = balanced_answer_view(
        answer=final_answer,
        answers=answers,
        question_id=question_id,
    )
    categories = category_ids(path_record, config, final_property.property_id)
    image_id, image_name, image_path, image_url = image_fields(
        path_record,
        question_id,
        default_image_dir(config),
    )
    source = image_source(config)
    route_labels = [entity.label for entity in route_entities]
    kg_source = str(path_record.get("source") or "wikidata_sparql_whitelist_paths")
    image_ref = {
        "image_id": image_id,
        "image_path": image_path,
        "image_url": image_url,
        "source": source,
    }
    image_attribution = path_record.get("image_attribution")
    if isinstance(image_attribution, dict):
        image_ref["attribution"] = image_attribution
    evidence_refs = {
        "image": image_ref,
        "kg_route": route_labels,
        "kg_property_route": route,
        "kg_path": path,
    }
    quality_scores = {
        "generation_success": 1.0,
        "route_verified": 1.0,
        "answer_unique": 1.0,
        "hard_negative_count": max(0, len(answers) - 1),
    }
    template_version = str(config.get("template_version") or generator_prompt_version)
    qae_triplet = {
        "triplet_id": f"krvqa_tri_{question_id:06d}",
        "question": question,
        "answer": final_answer,
        "choices": answers,
        "correct": correct,
        "triplet_type": "single_hop" if len(route) == 1 else "multihop",
        "question_type": "knowledge_reasoning_vqa",
        "evidence_refs": {
            "image": image_ref,
            "kg": {
                "entity_id": root_entity.entity_id,
                "property_route": route,
                "route": route_labels,
                "path": path,
            },
        },
    }
    provenance = {
        "sample_id": str(path_record.get("sample_id") or image_id),
        "source_id": image_id,
        "source_uri_or_hash": image_url,
        "modality": ["image", "text"],
        "recipe": recipe,
        "template_version": template_version,
        "generator_prompt_version": generator_prompt_version,
        "schema_version": schema_version,
        "kg_adapter": kg_source,
        "difficulty_policy": policy,
        "image_source": {
            "source": source,
            "url": image_url,
            "entity_id": root_entity.entity_id,
            "entity_name": root_entity.label,
        },
        "evidence_refs": evidence_refs,
        "quality_scores": quality_scores,
        "filter_versions": list(params.get("filter_versions") or []),
        "audit_status": "auto_pass",
        "intended_split": params.get("intended_split"),
    }
    row = dict(path_record)
    row.update(
        {
            "question_id": question_id,
            "question": question,
            "image_id": image_id,
            "image_name": image_name,
            "image_path": image_path,
            "image_url": image_url,
            "source": source,
            "answers": answers,
            "correct": correct,
            "categories": categories,
            "hop": len(route),
            "property_id": final_property.property_id,
            "property_label": final_property.label,
            "entity_id": root_entity.entity_id,
            "entity_name": root_entity.label,
            "route": route_labels,
            "has_scene_graph": bool(path_record.get("has_scene_graph", False)),
            "program_answer": final_answer,
            "route_verified": True,
            "difficulty": {
                "negative_strategy": negative_strategy,
                "max_taxonomy_distance": policy.get("max_taxonomy_distance"),
                "hop": len(route),
                "property_route": route,
                "question_variant": question_variant,
            },
            "reasoning_units": [
                {
                    "reasoning_unit_id": f"krvqa_reason_{question_id:06d}",
                    "hop": len(route),
                    "entity_id": root_entity.entity_id,
                    "entity_name": root_entity.label,
                    "property_route": route,
                    "route": route_labels,
                    "program_answer": final_answer,
                    "kg_path": path,
                }
            ],
            "qae_triplets": [qae_triplet],
            "evidence_refs": evidence_refs,
            "quality_scores": quality_scores,
            "audit_status": "auto_pass",
            "provenance": provenance,
            "recipe": recipe,
            "schema_version": schema_version,
            "generator_model": None,
            "generator_prompt_version": generator_prompt_version,
        }
    )
    return row


class PathPropertyCatalog:
    def __init__(self, path: list[dict[str, Any]]):
        self._properties = {
            property_from_triple(triple).property_id: property_from_triple(triple)
            for triple in path
        }

    def property(self, property_id: str) -> TemplateProperty:
        try:
            return self._properties[property_id]
        except KeyError as exc:
            raise ValueError(f"Unknown path property: {property_id}") from exc


def property_route(path_record: dict[str, Any]) -> list[str]:
    route = path_record.get("route_properties")
    if not isinstance(route, list):
        route = [
            str(triple.get("property", {}).get("id") or "")
            for triple in path_triples(path_record)
        ]
    route = [str(item).strip() for item in route if str(item).strip()]
    if not route:
        raise ValueError("Wikidata path record requires route_properties.")
    if not all(item.startswith("P") for item in route):
        raise ValueError("Wikidata path route must contain property ids.")
    return route


def path_triples(path_record: dict[str, Any]) -> list[dict[str, Any]]:
    raw_path = path_record.get("path")
    if not isinstance(raw_path, list) or not raw_path:
        raise ValueError("Wikidata path record requires a non-empty path.")
    return [dict(triple) for triple in raw_path]


def root_template_entity(
    path_record: dict[str, Any],
    config: dict[str, Any],
) -> TemplateEntity:
    root = path_record.get("root") or {}
    root_concept = path_record.get("root_concept") or {}
    qid = str(
        root.get("qid")
        or root_concept.get("wikidata_qid")
        or root_concept.get("qid")
        or ""
    ).strip()
    if not qid:
        raise ValueError("Wikidata path record requires root.qid.")
    label = str(
        root.get("label")
        or root_concept.get("label")
        or root_concept.get("taxon_name")
        or qid
    )
    entity_type = str(
        config.get("root_entity_type")
        or root_concept.get("taxon_rank")
        or "taxon"
    )
    return TemplateEntity(
        entity_id=qid,
        label=label,
        entity_type=entity_type,
        domain=str(config.get("root_entity_domain") or "inaturalist"),
    )


def route_template_entities(
    path_record: dict[str, Any],
    config: dict[str, Any],
) -> list[TemplateEntity]:
    entities = [root_template_entity(path_record, config)]
    for index, triple in enumerate(path_triples(path_record), start=1):
        tail = triple.get("tail") or {}
        tail_qid = str(tail.get("qid") or "").strip()
        label = str(tail.get("label") or tail.get("value") or tail_qid or f"answer_{index}")
        entity_id = tail_qid or f"literal:{index}:{label}"
        entity_type = (
            "wikidata_entity"
            if str(tail.get("type") or "") == "wikidata_entity"
            else answer_type_from_triple(triple)
        )
        entities.append(
            TemplateEntity(
                entity_id=entity_id,
                label=label,
                entity_type=entity_type,
                domain=str(config.get("root_entity_domain") or "inaturalist"),
            )
        )
    return entities


def property_from_triple(triple: dict[str, Any]) -> TemplateProperty:
    raw_property = triple.get("property") or {}
    property_id = str(raw_property.get("id") or "").strip()
    if not property_id:
        raise ValueError(f"Path triple lacks property id: {triple}")
    label = str(raw_property.get("label") or property_id)
    return TemplateProperty(
        property_id=property_id,
        label=label,
        question_label=str(raw_property.get("question_label") or label),
    )


def answer_type_from_triple(triple: dict[str, Any]) -> str:
    raw_property = triple.get("property") or {}
    answer_type = str(raw_property.get("answer_type") or "").strip()
    if answer_type:
        return answer_type
    tail = triple.get("tail") or {}
    return str(tail.get("type") or "literal")


def answer_text(path_record: dict[str, Any]) -> str:
    final_answer = path_record.get("final_answer") or path_triples(path_record)[-1].get("tail") or {}
    value = final_answer.get("label") or final_answer.get("value") or final_answer.get("qid")
    if value is None or not str(value).strip():
        raise ValueError("Wikidata path record requires a final_answer label/value.")
    return str(value)


def category_ids(
    path_record: dict[str, Any],
    config: dict[str, Any],
    property_id: str,
) -> list[int]:
    row_categories = path_record.get("categories")
    if row_categories:
        return [int(category_id) for category_id in row_categories]
    category_ids_by_property = config.get("category_ids_by_property")
    if (
        isinstance(category_ids_by_property, dict)
        and property_id in category_ids_by_property
    ):
        category_values = category_ids_by_property[property_id]
        if category_values:
            return [int(category_id) for category_id in category_values]
    raise ValueError(
        "recipe_config.category_ids_by_property must provide category ids "
        f"for property {property_id}."
    )


def default_image_dir(config: dict[str, Any]) -> Path:
    image_dir = config.get("image_dir")
    if image_dir:
        return Path(image_dir)
    export_dir = config.get("export_dir")
    if export_dir:
        return Path(export_dir) / "images"
    raise ValueError("recipe_config.image_dir or recipe_config.export_dir is required.")


def image_fields(
    path_record: dict[str, Any],
    question_id: int,
    default_image_dir: Path,
) -> tuple[str, str, str, str]:
    image = path_record.get("image") or {}
    metadata = image.get("metadata") or {}
    image_id = str(
        image.get("image_id")
        or image.get("id")
        or metadata.get("id")
        or path_record.get("image_id")
        or question_id
    )
    file_name = str(
        image.get("file_name")
        or metadata.get("file_name")
        or path_record.get("image_name")
        or f"{image_id}.jpg"
    )
    image_name = Path(file_name).name
    image_path = str(
        image.get("image_path")
        or image.get("path")
        or path_record.get("image_path")
        or default_image_dir / file_name
    )
    image_url = str(
        image.get("url")
        or image.get("image_url")
        or metadata.get("url")
        or path_record.get("image_url")
        or file_name
    )
    return image_id, image_name, image_path, image_url


def image_source(config: dict[str, Any]) -> str:
    source = config.get("image_source")
    if not source:
        raise ValueError("recipe_config.image_source is required.")
    return str(source)


def choice_candidates(
    path_record: dict[str, Any],
    route: list[str],
) -> list[Any]:
    raw_candidates = path_record.get("choice_candidates")
    if not raw_candidates:
        return []
    if not isinstance(raw_candidates, list):
        raise ValueError("path_record.choice_candidates must be a list.")
    final_property = route[-1] if route else ""
    candidates = []
    for candidate in raw_candidates:
        if not isinstance(candidate, dict):
            candidates.append(candidate)
            continue
        property_id = str(candidate.get("property_id") or final_property)
        candidate_route = candidate.get("route_properties")
        if property_id != final_property:
            continue
        if isinstance(candidate_route, list) and [
            str(item) for item in candidate_route
        ] != route:
            continue
        candidates.append(candidate)
    return candidates


def difficulty_policy(config: dict[str, Any]) -> dict[str, Any]:
    incoming = config.get("difficulty_policy")
    if incoming is None:
        return {}
    if isinstance(incoming, dict):
        return dict(incoming)
    raise ValueError("recipe_config.difficulty_policy must be a dict.")


def _path_is_question_usable(
    path_record: dict[str, Any],
    config: dict[str, Any],
) -> bool:
    if bool(config.get("filter_template_usable_paths", True)):
        return is_inaturalist_question_template_usable_path(path_record)
    return True


__all__ = [
    "PathPropertyCatalog",
    "answer_text",
    "generate_reasonvqa_row_from_wikidata_path",
    "generate_reasonvqa_rows_from_wikidata_paths",
    "property_route",
    "route_template_entities",
    "root_template_entity",
]
