"""Image OCR loading operator: png/jpg -> C-IR-2."""
from __future__ import annotations

from typing import Any, Dict, Optional

from .loading_operator import LoadingOperation
from .image_ocr import parse_image_inputs


class ImageOcrOperation(LoadingOperation):
    """Run OCR over local images and emit ``parsed-doc-v2`` rows."""

    PARAMS_SCHEMA = {
        "category": "loading",
        "description": "Run OCR on png/jpg images -> ParsedDocument(parsed-doc-v2)",
        "params": {
            "data_paths": {"type": "array", "items": {"type": "string"}, "required": True},
            "output_key": {"type": "string", "required": False, "default": "parsed_doc"},
            "backend": {
                "type": "string",
                "enum": ["auto", "mindocr", "paddleocr"],
                "required": False,
                "default": "auto",
            },
            "device": {
                "type": "string",
                "enum": ["auto", "npu", "cuda", "cpu"],
                "required": False,
                "default": "auto",
            },
            "lang": {
                "type": "string",
                "enum": ["zh", "en", "mixed", "ch"],
                "required": False,
                "default": "zh",
            },
            "layout_analysis": {"type": "boolean", "required": False, "default": True},
            "max_file_mb": {"type": "number", "required": False, "default": 100},
            "max_image_megapixels": {"type": "number", "required": False, "default": 100},
            "parse_timeout_s": {"type": "number", "required": False, "default": 300},
            "max_memory_mb": {"type": "integer", "required": False, "default": 4096},
            "on_error": {
                "type": "string",
                "enum": ["raise", "skip"],
                "required": False,
                "default": "raise",
            },
            "allowed_roots": {
                "type": "array",
                "items": {"type": "string"},
                "required": False,
                "default": [],
            },
            "backend_ranking_path": {"type": "string", "required": False},
            "model_paths": {"type": "object", "required": False, "default": {}},
            "allow_model_download": {"type": "boolean", "required": False, "default": False},
        },
        "example": {"data_paths": ["./data/images/"]},
    }

    def execute(self, dataset: Optional[Any] = None) -> Any:
        if dataset is not None:
            return dataset
        data_paths = self.params.get("data_paths", [])
        if isinstance(data_paths, str):
            data_paths = [data_paths]
        if not data_paths:
            raise ValueError("ImageOcrOperation requires 'data_paths' parameter")
        params: Dict[str, Any] = dict(self.params)
        params["data_paths"] = list(data_paths)
        rows = parse_image_inputs(data_paths, params)

        import ray

        from base.dataset import MultimodalDataset

        return MultimodalDataset(dataset=ray.data.from_items(rows), modalities=["image"])
