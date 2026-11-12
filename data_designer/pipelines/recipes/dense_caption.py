from __future__ import annotations

import json
from typing import Any, Callable

from base.dataset import MultimodalDataset
from curator.operations.augmentation.image_captioning_operation import ImageCaptioningOperation
from curator.utils.local_qwen_vlm_client import extract_json_object
from data_designer.integrations.curator.recipe_parameters import (
    DEFAULT_DENSE_CAPTION_FALLBACK_IMAGE_KEYS,
    DENSE_CAPTION_OPERATION_PARAM_KEYS,
)
from data_designer.generation.recipes.context import RecipeExecutor


DENSE_CAPTION_SCHEMA_VERSION = "dense-caption-v1"
DENSE_CAPTION_RAW_OUTPUT_KEY = "_dense_caption_raw"
DENSE_CAPTION_TEXT_KEY = "dense_caption_text"
DENSE_CAPTION_DEFAULT_PROMPT = (
    "Generate a fine-grained dense caption for the image. Return valid JSON only "
    "with this schema: {"
    '"summary": "one short global caption", '
    '"scene": {"setting": "", "background": "", "viewpoint": "", "image_quality": ""}, '
    '"objects": [{"name": "", "count": null, "attributes": [], "location": ""}], '
    '"relations": [{"subject": "", "predicate": "", "object": ""}], '
    '"actions": [{"actor": "", "action": "", "target": "", "state": ""}], '
    '"text_in_image": [], '
    '"uncertainties": []'
    "}. Use only visible evidence; do not guess hidden facts."
)


class DenseCaptionRecipeExecutor(RecipeExecutor):
    """dense_caption recipe executor."""

    def execute(self, dataset: MultimodalDataset) -> MultimodalDataset:
        config = self.section_config("dense_caption")
        image_key = config.get("image_key") or self.params.get("input_key") or "image_path"
        output_key = config.get("output_key") or self.params.get("output_key") or "dense_caption"
        caption_key = config.get("caption_key") or self.params.get("caption_key") or "caption"
        caption_text_key = (
            config.get("caption_text_key")
            or self.params.get("caption_text_key")
            or DENSE_CAPTION_TEXT_KEY
        )
        raw_output_key = (
            config.get("raw_output_key")
            or self.params.get("raw_output_key")
            or DENSE_CAPTION_RAW_OUTPUT_KEY
        )
        schema_version = str(
            config.get("dense_caption_schema_version")
            or self.params.get("dense_caption_schema_version")
            or DENSE_CAPTION_SCHEMA_VERSION
        )
        captioner = config.get("captioner") or self.params.get("captioner")
        fallback_image_keys = (
            config.get("fallback_image_keys")
            or self.params.get("fallback_image_keys")
            or list(DEFAULT_DENSE_CAPTION_FALLBACK_IMAGE_KEYS)
        )

        if captioner is not None:
            if not callable(captioner):
                raise ValueError("dense_caption captioner must be callable.")
            dataset = dataset.map(
                self._build_captioner_row_func(
                    captioner=captioner,
                    image_key=image_key,
                    output_key=output_key,
                    raw_output_key=raw_output_key,
                    fallback_image_keys=fallback_image_keys,
                )
            )
        else:
            caption_params = self._dense_caption_operation_params(config, image_key, raw_output_key)
            dataset = ImageCaptioningOperation(caption_params).execute(dataset)

        def normalize_caption_row(row):
            row = dict(row)
            raw_caption = row.get(output_key)
            if not self._has_value(raw_caption):
                raw_caption = row.get(raw_output_key)
            if not self._has_value(raw_caption):
                return row

            dense_caption = self._canonical_dense_caption(
                raw_caption,
                schema_version=schema_version,
            )
            row[output_key] = dense_caption

            caption_text = self._dense_caption_to_text(dense_caption)
            if caption_text_key:
                row[caption_text_key] = caption_text

            summary = str(dense_caption.get("summary") or "").strip()
            if caption_key and caption_key != output_key and row.get(caption_key) is None and summary:
                row[caption_key] = summary
            return row

        generator_model = self.resolve_generator_model(config)
        dataset = dataset.map(normalize_caption_row)
        return self.annotate_dataset(
            dataset,
            output_keys=[output_key, caption_key, caption_text_key],
            modality=["image", "text"],
            generator_model=generator_model,
        )

    def _dense_caption_operation_params(
        self,
        config: dict[str, Any],
        image_key: str,
        raw_output_key: str,
    ) -> dict[str, Any]:
        params = {"image_key": image_key, "output_key": raw_output_key}
        for key in DENSE_CAPTION_OPERATION_PARAM_KEYS:
            if key in self.params and self.params[key] is not None:
                params[key] = self.params[key]
            if key in config and config[key] is not None:
                params[key] = config[key]
        params.setdefault("prompt_text", DENSE_CAPTION_DEFAULT_PROMPT)
        params.setdefault("max_new_tokens", 512)
        params.setdefault(
            "fallback_image_keys",
            list(DEFAULT_DENSE_CAPTION_FALLBACK_IMAGE_KEYS),
        )
        params.setdefault("error_key", "_synthetic_generation_error")
        params.setdefault("resolved_model_path_key", "generator_model_path")
        return params

    def _build_captioner_row_func(
        self,
        captioner: Callable[[dict[str, Any]], Any],
        image_key: str,
        output_key: str,
        raw_output_key: str,
        fallback_image_keys: list[str],
    ):
        def caption_row(row):
            row = dict(row)
            image_candidates = [image_key] + list(fallback_image_keys)
            image_value = next(
                (row.get(key) for key in image_candidates if row.get(key) is not None),
                None,
            )
            if image_value is None:
                return row
            cache_key = self.row_cache_key(row, image_key)
            if cache_key in self.context.caption_cache:
                caption = self.context.caption_cache[cache_key]
            else:
                try:
                    caption = self.call_with_retries(captioner, row)
                except Exception as exc:
                    if self.on_error == "mark_failed":
                        return self.mark_failed(row, exc)
                    raise
                self.context.caption_cache[cache_key] = caption
            if isinstance(caption, dict) and output_key not in caption:
                row[output_key] = caption
            elif isinstance(caption, dict):
                row.update(caption)
            elif caption is not None:
                row[raw_output_key] = str(caption)
            return row

        return caption_row

    @classmethod
    def _canonical_dense_caption(
        cls,
        value: Any,
        schema_version: str,
    ) -> dict[str, Any]:
        payload = cls._parse_dense_caption_payload(value)

        summary = cls._string_value(
            payload.get("summary")
            or payload.get("caption")
            or payload.get("global_caption")
            or payload.get("description")
            or value
        )
        scene = cls._normalize_scene(payload.get("scene"))
        objects = cls._normalize_items(payload.get("objects"), default_key="name")
        attributes = cls._normalize_items(payload.get("attributes"), default_key="attribute")
        relations = cls._normalize_items(payload.get("relations"), default_key="relation")
        actions = cls._normalize_items(payload.get("actions"), default_key="action")
        text_in_image = cls._normalize_string_list(payload.get("text_in_image"))
        uncertainties = cls._normalize_string_list(payload.get("uncertainties"))

        return {
            "schema_version": str(payload.get("schema_version") or schema_version),
            "summary": summary,
            "scene": scene,
            "objects": objects,
            "attributes": attributes,
            "relations": relations,
            "actions": actions,
            "text_in_image": text_in_image,
            "uncertainties": uncertainties,
        }

    @classmethod
    def _parse_dense_caption_payload(cls, value: Any) -> dict[str, Any]:
        if isinstance(value, dict):
            return dict(value)
        text = cls._string_value(value)
        parsed = extract_json_object(text)
        if parsed is not None:
            return parsed
        return {"summary": text}

    @classmethod
    def _normalize_scene(cls, value: Any) -> dict[str, str]:
        scene = dict(value) if isinstance(value, dict) else {}
        return {
            "setting": cls._string_value(scene.get("setting")),
            "background": cls._string_value(scene.get("background")),
            "viewpoint": cls._string_value(scene.get("viewpoint")),
            "image_quality": cls._string_value(scene.get("image_quality")),
        }

    @classmethod
    def _normalize_items(cls, value: Any, default_key: str) -> list[dict[str, Any]]:
        if not cls._has_value(value):
            return []
        items = value if isinstance(value, list) else [value]
        normalized = []
        for item in items:
            if isinstance(item, dict):
                cleaned = {
                    str(key): cls._normalize_nested_value(item_value)
                    for key, item_value in item.items()
                    if cls._has_value(item_value)
                }
                if cleaned:
                    normalized.append(cleaned)
            else:
                text = cls._string_value(item)
                if text:
                    normalized.append({default_key: text})
        return normalized

    @classmethod
    def _normalize_nested_value(cls, value: Any) -> Any:
        if isinstance(value, list):
            return [cls._normalize_nested_value(item) for item in value if cls._has_value(item)]
        if isinstance(value, dict):
            return {
                str(key): cls._normalize_nested_value(item)
                for key, item in value.items()
                if cls._has_value(item)
            }
        if isinstance(value, (int, float, bool)):
            return value
        return cls._string_value(value)

    @classmethod
    def _normalize_string_list(cls, value: Any) -> list[str]:
        if not cls._has_value(value):
            return []
        items = value if isinstance(value, list) else [value]
        return [text for item in items if (text := cls._string_value(item))]

    @classmethod
    def _dense_caption_to_text(cls, dense_caption: dict[str, Any]) -> str:
        parts = [cls._string_value(dense_caption.get("summary"))]
        scene = dense_caption.get("scene")
        if isinstance(scene, dict):
            parts.extend(cls._string_value(scene.get(key)) for key in ("setting", "background"))
        parts.extend(
            cls._item_text(item, ("name", "object", "attribute"))
            for item in dense_caption.get("objects") or []
        )
        parts.extend(
            cls._item_text(item, ("relation", "predicate"))
            for item in dense_caption.get("relations") or []
        )
        parts.extend(
            cls._item_text(item, ("action", "actor", "target"))
            for item in dense_caption.get("actions") or []
        )
        parts.extend(cls._normalize_string_list(dense_caption.get("text_in_image")))
        return " ".join(part for part in parts if part).strip()

    @classmethod
    def _item_text(cls, item: Any, preferred_keys: tuple[str, ...]) -> str:
        if not isinstance(item, dict):
            return cls._string_value(item)
        values = []
        for key in preferred_keys:
            value = item.get(key)
            if isinstance(value, list):
                values.extend(cls._string_value(part) for part in value)
            else:
                values.append(cls._string_value(value))
        text = " ".join(part for part in values if part).strip()
        if text:
            return text
        return " ".join(cls._string_value(value) for value in item.values() if value).strip()

    @staticmethod
    def _string_value(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, (dict, list)):
            return json.dumps(value, ensure_ascii=False)
        return str(value).strip()

    @staticmethod
    def _has_value(value: Any) -> bool:
        if value is None:
            return False
        if isinstance(value, str):
            return bool(value.strip())
        if isinstance(value, (dict, list, tuple, set)):
            return bool(value)
        return True
