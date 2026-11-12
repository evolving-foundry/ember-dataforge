from __future__ import annotations

from pathlib import Path
from typing import Any

from data_designer.storage.writers import write_jsonl


def write_reasonvqa_exports(
    rows: list[dict[str, Any]],
    export_dir: Path,
    split: str,
    categories: list[dict[str, Any]],
) -> None:
    if not categories:
        raise ValueError(
            "recipe_config.categories is required when export is enabled."
        )
    qa_fields = (
        "question_id",
        "question",
        "image_id",
        "image_name",
        "image_path",
        "image_url",
        "source",
        "answers",
        "correct",
        "categories",
    )
    ann_fields = (
        "question_id",
        "hop",
        "property_id",
        "property_label",
        "entity_id",
        "entity_name",
        "route",
        "has_scene_graph",
    )
    export_dir.mkdir(parents=True, exist_ok=True)
    qa_rows = [{field: row[field] for field in qa_fields} for row in rows]
    ann_rows = [{field: row[field] for field in ann_fields} for row in rows]
    write_jsonl(export_dir / f"{split}.jsonl", qa_rows)
    write_jsonl(export_dir / f"{split}_ann.jsonl", ann_rows)
    write_jsonl(export_dir / "knowledge_reasoning_vqa_merged.jsonl", rows)
    manifest_rows = [image_manifest_row(row) for row in rows]
    write_jsonl(export_dir / "image_manifest.jsonl", manifest_rows)
    write_jsonl(export_dir / "categories.jsonl", categories)


def image_manifest_row(row: dict[str, Any]) -> dict[str, Any]:
    manifest = {
        "question_id": row["question_id"],
        "image_id": row["image_id"],
        "image_name": row["image_name"],
        "image_path": row["image_path"],
        "image_url": row["image_url"],
        "source": row["source"],
        "entity_id": row["entity_id"],
        "entity_name": row["entity_name"],
        "hop": row["hop"],
        "route": row["route"],
    }
    attribution = (
        row.get("provenance", {}).get("image_source", {}).get("attribution")
    )
    if attribution:
        manifest["image_attribution"] = attribution
    return manifest


__all__ = [
    "image_manifest_row",
    "write_jsonl",
    "write_reasonvqa_exports",
]
