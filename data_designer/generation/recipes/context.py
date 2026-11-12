from __future__ import annotations

import hashlib
import json
from typing import Any, Callable, TYPE_CHECKING

MultimodalDataset = Any

class RecipeExecutor:
    """Shared recipe utilities; owns a dependency context, never an operator."""

    def __init__(self, context):
        self.context = context

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self.context, name)

    def execute(self, dataset: MultimodalDataset) -> MultimodalDataset:
        raise NotImplementedError

    def section_config(self, key: str) -> dict[str, Any]:
        section = self.recipe_config.get(key)
        if section is None:
            return dict(self.recipe_config)
        if not isinstance(section, dict):
            raise ValueError(f"recipe_config.{key} must be a dict.")
        return dict(section)

    def step_enabled(self, key: str, default: bool) -> bool:
        enabled = self.recipe_config.get(f"{key}_enabled", default)
        return bool(enabled)

    def call_with_retries(
        self,
        callback: Callable[[dict[str, Any]], Any],
        row: dict[str, Any],
    ) -> Any:
        last_error: Exception | None = None
        for _ in range(self.max_retries + 1):
            try:
                return callback(row)
            except Exception as exc:
                last_error = exc
        raise last_error or RuntimeError("captioner failed")

    def annotate_dataset(
        self,
        dataset: MultimodalDataset,
        output_keys: list[str],
        modality: list[str],
        generator_model: str | None,
    ) -> MultimodalDataset:
        def annotate_row(row):
            row = dict(row)
            evidence_refs = self.collect_evidence_refs(row)
            quality_scores = self.build_quality_scores(row, output_keys)
            audit_status = self.audit_status(row, quality_scores)
            provenance = self.build_provenance(
                row=row,
                modality=modality,
                evidence_refs=evidence_refs,
                quality_scores=quality_scores,
                audit_status=audit_status,
                generator_model=generator_model,
            )
            row["recipe"] = self.recipe
            row["schema_version"] = self.schema_version
            row["generator_model"] = generator_model
            row["generator_prompt_version"] = self.generator_prompt_version
            row[self.evidence_refs_key] = evidence_refs
            row[self.quality_scores_key] = quality_scores
            row[self.audit_status_key] = audit_status
            row[self.provenance_key] = provenance
            return row

        return dataset.map(annotate_row)

    def build_provenance(
        self,
        row: dict[str, Any],
        modality: list[str],
        evidence_refs: Any,
        quality_scores: dict[str, Any],
        audit_status: str,
        generator_model: str | None,
    ) -> dict[str, Any]:
        return {
            "sample_id": str(row.get("sample_id") or row.get("id") or self.row_hash(row)[:16]),
            "source_id": self.source_id(row),
            "source_uri_or_hash": self.source_uri_or_hash(row),
            "modality": modality,
            "recipe": self.recipe,
            "generator_model": generator_model,
            "generator_prompt_version": self.generator_prompt_version,
            "schema_version": self.schema_version,
            "evidence_refs": evidence_refs,
            "quality_scores": quality_scores,
            "filter_versions": list(self.params.get("filter_versions") or []),
            "audit_status": audit_status,
            "intended_split": self.params.get("intended_split"),
            "generation_error": row.get("_synthetic_generation_error"),
            "generator_model_path": row.get("generator_model_path"),
        }

    def resolve_generator_model(self, config: dict[str, Any]) -> str | None:
        if self.params.get("generator_model"):
            return str(self.params["generator_model"])
        for key in ("vlm_model_path", "model_path", "llm_model", "judge_model"):
            if self.params.get(key):
                return str(self.params[key])
        for key in ("generator_model", "vlm_model_path", "model_path", "llm_model", "judge_model"):
            if config.get(key):
                return str(config[key])
        if self.recipe == "dense_caption":
            return "Salesforce/blip-image-captioning-base"
        return None

    def build_quality_scores(
        self,
        row: dict[str, Any],
        output_keys: list[str],
    ) -> dict[str, Any]:
        scores: dict[str, Any] = {
            "generation_success": 1.0 if self.has_any_output(row, output_keys) else 0.0
        }

        materialization_metrics = row.get("materialization_metrics")
        if isinstance(materialization_metrics, dict):
            scores.update(
                {
                    f"materialization_{key}": value
                    for key, value in materialization_metrics.items()
                }
            )

        quality_records = row.get("quality_records")
        if isinstance(quality_records, list):
            total = len(quality_records)
            keep_count = sum(
                1 for item in quality_records if str(item.get("decision")) == "keep"
            )
            drop_count = sum(
                1 for item in quality_records if str(item.get("decision")) == "drop"
            )
            scores["quality_record_count"] = total
            scores["quality_keep_rate"] = round(keep_count / total, 6) if total else 0.0
            scores["quality_drop_rate"] = round(drop_count / total, 6) if total else 0.0
        return scores

    def audit_status(self, row: dict[str, Any], quality_scores: dict[str, Any]) -> str:
        if row.get(self.audit_status_key):
            return str(row[self.audit_status_key])
        if row.get("_synthetic_generation_error"):
            return "auto_fail"
        if quality_scores.get("generation_success", 0.0) <= 0.0:
            return "auto_fail"
        return "auto_pass"

    def collect_evidence_refs(self, row: dict[str, Any]) -> Any:
        existing = row.get(self.evidence_refs_key)
        if existing:
            return existing
        qae_triplets = row.get("qae_triplets")
        if isinstance(qae_triplets, list):
            refs = []
            for triplet in qae_triplets:
                if not isinstance(triplet, dict) or not triplet.get("evidence_refs"):
                    continue
                refs.append(
                    {
                        "target_triplet_id": str(
                            triplet.get("triplet_id") or triplet.get("id") or ""
                        ),
                        "evidence_refs": triplet["evidence_refs"],
                    }
                )
            if refs:
                return refs
        return {}

    def mark_failed(self, row: dict[str, Any], exc: Exception) -> dict[str, Any]:
        row["_synthetic_generation_error"] = str(exc)
        row[self.audit_status_key] = "auto_fail"
        return row

    @staticmethod
    def has_any_output(row: dict[str, Any], output_keys: list[str]) -> bool:
        for key in output_keys:
            if not key:
                continue
            value = row.get(key)
            if isinstance(value, (list, tuple, dict)) and value:
                return True
            if isinstance(value, str) and value.strip():
                return True
            if value not in (None, "", [], {}):
                return True
        return False

    @classmethod
    def row_cache_key(cls, row: dict[str, Any], image_key: str) -> str:
        for key in ("sample_id", "id", "source_id", "image_id", "image_path", image_key):
            if row.get(key) is not None:
                return str(row[key])
        return cls.row_hash(row)

    @classmethod
    def source_id(cls, row: dict[str, Any]) -> str:
        for key in ("source_id", "sample_id", "id", "video_id", "image_id"):
            if row.get(key) is not None:
                return str(row[key])
        return cls.row_hash(row)[:16]

    @classmethod
    def source_uri_or_hash(cls, row: dict[str, Any]) -> str:
        for key in ("source_uri", "uri", "url", "image_path", "video_path"):
            if row.get(key):
                return str(row[key])
        return cls.row_hash(row)

    @staticmethod
    def row_hash(row: dict[str, Any]) -> str:
        payload = json.dumps(row, ensure_ascii=False, sort_keys=True, default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


    # Base class name from before the refactor; the four executors under recipes/ and the recipes registry still reference it by this name.
BaseSyntheticRecipeExecutor = RecipeExecutor
