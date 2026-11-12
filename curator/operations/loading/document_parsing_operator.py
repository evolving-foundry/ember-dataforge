"""Document parsing loading operator: docx/xlsx/pdf → C-IR-2."""
from __future__ import annotations

from typing import Any, Dict, Optional

from .loading_operator import LoadingOperation
from .document_parsing import parse_inputs


class DocumentParsingOperation(LoadingOperation):
    """Parse local documents into ``parsed-doc-v2`` rows."""

    PARAMS_SCHEMA = {
        "category": "loading",
        "description": "Parse docx/xlsx/pdf -> ParsedDocument(parsed-doc-v2)",
        "params": {
            "data_paths": {"type": "array", "items": {"type": "string"}, "required": True},
            "output_key": {"type": "string", "required": False, "default": "parsed_doc"},
            "extract_style": {"type": "boolean", "required": False, "default": False},
            "max_file_mb": {"type": "number", "required": False, "default": 100},
            "max_pages": {"type": "integer", "required": False, "default": 500},
            "pdf_min_text_chars_per_page": {"type": "integer", "required": False, "default": 20},
            "parse_timeout_s": {"type": "number", "required": False, "default": 60},
            "max_memory_mb": {"type": "integer", "required": False, "default": 2048},
            "on_error": {"type": "string", "enum": ["raise", "skip"], "required": False, "default": "raise"},
            "allowed_roots": {"type": "array", "items": {"type": "string"}, "required": False, "default": []},
        },
        "example": {"data_paths": ["./data/tickets/"]},
    }

    def execute(self, dataset: Optional[Any] = None) -> Any:
        if dataset is not None:
            return dataset
        data_paths = self.params.get("data_paths", [])
        if isinstance(data_paths, str):
            data_paths = [data_paths]
        if not data_paths:
            raise ValueError("DocumentParsingOperation requires 'data_paths' parameter")
        params: Dict[str, Any] = dict(self.params)
        params["data_paths"] = list(data_paths)
        rows = parse_inputs(data_paths, params)
        import ray

        from base.dataset import MultimodalDataset

        return MultimodalDataset(dataset=ray.data.from_items(rows), modalities=["document"])
