from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol
from typing import Any


@dataclass(frozen=True)
class TemplateEntity:
    entity_id: str
    label: str
    entity_type: str
    domain: str


@dataclass(frozen=True)
class TemplateProperty:
    property_id: str
    label: str
    question_label: str


class TemplatePropertyLookup(Protocol):
    def property(self, property_id: str) -> TemplateProperty:
        ...


def question_for_route(
    root_entity: TemplateEntity,
    property_route: list[str],
    kg: TemplatePropertyLookup,
    variant_index: int = 0,
    config: dict[str, Any] | None = None,
    route_entities: list[TemplateEntity] | None = None,
) -> str:
    config = config or {}
    route_entities = route_entities or [root_entity]
    template_bank = question_template_bank(config)
    subject = visual_anchor_subject(
        config,
        root_entity,
        property_route,
        route_entities,
        variant_index,
    )
    route_template = route_question_template(config, property_route, variant_index)
    if route_template:
        final_property = kg.property(property_route[-1])
        return render_question_template(
            route_template,
            question_template_slots(
                subject,
                root_entity,
                final_property,
                property_route,
                route_entities,
            ),
            "recipe_config.route_question_templates."
            + " -> ".join(property_route),
        )
    for property_id in property_route[:-1]:
        template = property_question_template(
            template_bank,
            property_id,
            "sub_clause",
            variant_index,
        )
        property_object = kg.property(property_id)
        subject = render_question_template(
            template,
            question_template_slots(
                subject,
                root_entity,
                property_object,
                property_route,
                route_entities,
            ),
            property_id,
        )
    final_property_id = property_route[-1]
    final_template = property_question_template(
        template_bank,
        final_property_id,
        "main",
        variant_index,
    )
    final_property = kg.property(final_property_id)
    return render_question_template(
        final_template,
        question_template_slots(
            subject,
            root_entity,
            final_property,
            property_route,
            route_entities,
        ),
        final_property_id,
    )


def question_template_bank(config: dict[str, Any]) -> dict[str, Any]:
    templates = config.get("question_templates")
    if not isinstance(templates, dict) or not templates:
        raise ValueError("recipe_config.question_templates is required.")
    return templates


def route_question_template(
    config: dict[str, Any],
    property_route: list[str],
    variant_index: int,
) -> str | None:
    templates = config.get("route_question_templates")
    if not isinstance(templates, dict) or not templates:
        return None
    route_keys = [
        " -> ".join(property_route),
        "/".join(property_route),
    ]
    for route_key in route_keys:
        raw_template = templates.get(route_key)
        if raw_template:
            return select_question_template(
                raw_template,
                f"recipe_config.route_question_templates.{route_key}",
                variant_index,
            )
    return None


def visual_anchor_subject(
    config: dict[str, Any],
    root_entity: TemplateEntity,
    property_route: list[str],
    route_entities: list[TemplateEntity],
    variant_index: int,
) -> str:
    template = select_question_template(
        config.get("visual_anchor_template"),
        "recipe_config.visual_anchor_template",
        variant_index,
    )
    slots = {
        "root_entity_label": root_entity.label,
        "root_entity_type": root_entity.entity_type,
        "root_entity_domain": root_entity.domain,
        "hop": len(property_route),
        "property_route": " -> ".join(property_route),
        "route_labels": " -> ".join(entity.label for entity in route_entities),
    }
    return render_question_template(
        template,
        slots,
        "visual_anchor_template",
    )


def property_question_template(
    template_bank: dict[str, Any],
    property_id: str,
    template_role: str,
    variant_index: int,
) -> str:
    entry = template_bank.get(property_id)
    if not isinstance(entry, dict):
        raise ValueError(
            f"recipe_config.question_templates must define {property_id}.{template_role}."
        )
    return select_question_template(
        entry.get(template_role),
        f"recipe_config.question_templates.{property_id}.{template_role}",
        variant_index,
    )


def select_question_template(
    raw_template: Any,
    field_name: str,
    variant_index: int,
) -> str:
    if isinstance(raw_template, str) and raw_template.strip():
        return raw_template.strip()
    if isinstance(raw_template, (list, tuple)) and raw_template:
        selected = raw_template[variant_index % len(raw_template)]
        if isinstance(selected, str) and selected.strip():
            return selected.strip()
    raise ValueError(f"{field_name} must be a non-empty string or list.")


def question_template_slots(
    subject: str,
    root_entity: TemplateEntity,
    property_object: TemplateProperty,
    property_route: list[str],
    route_entities: list[TemplateEntity],
) -> dict[str, Any]:
    slots: dict[str, Any] = {
        "subject": subject,
        "root_entity_label": root_entity.label,
        "root_entity_type": root_entity.entity_type,
        "root_entity_domain": root_entity.domain,
        "property_id": property_object.property_id,
        "property_label": property_object.label,
        "property_question_label": property_object.question_label,
        "answer_type": property_object.question_label,
        "hop": len(property_route),
        "property_route": " -> ".join(property_route),
        "route_labels": " -> ".join(entity.label for entity in route_entities),
    }
    for index, entity in enumerate(route_entities):
        slots[f"entity_{index}_label"] = entity.label
        slots[f"entity_{index}_type"] = entity.entity_type
        slots[f"entity_{index}_domain"] = entity.domain
    return slots


def render_question_template(
    template: str,
    slots: dict[str, Any],
    template_id: str,
) -> str:
    try:
        rendered = template.format(**slots)
    except KeyError as exc:
        missing_slot = exc.args[0]
        raise ValueError(
            f"Question template {template_id} references unknown slot "
            f"{missing_slot!r}."
        ) from exc
    rendered = " ".join(rendered.split())
    if not rendered:
        raise ValueError(f"Question template {template_id} rendered empty text.")
    return rendered


__all__ = [
    "property_question_template",
    "question_for_route",
    "question_template_bank",
    "question_template_slots",
    "render_question_template",
    "route_question_template",
    "select_question_template",
    "TemplateEntity",
    "TemplateProperty",
    "TemplatePropertyLookup",
    "visual_anchor_subject",
]
