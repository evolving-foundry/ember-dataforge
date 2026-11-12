from typing import Iterable, List, Any, Dict, Optional
from .filtering_operator import FilteringOperation
from base.dataset import MultimodalDataset
from ...tracking.tracking import OperationType


class SpecialCharFilterOperation(FilteringOperation):
    """Filter text based on special characters

    Filter text samples that contain specified special characters. If the text contains any of the specified special characters, the sample is filtered out.
    Commonly used to filter text containing HTML tags, path separators, and other special characters.
    """

    operation_type = OperationType.FILTERING

    PARAMS_SCHEMA = {
        "category": "filtering",
        "description": "Filter text samples containing specified special characters",
        "params": {
            "special_chars": {
                "type": "array",
                "required": False,
                "default": ["<", ">", "`", "/"],
                "description": "Special-character list"
            },
            "text_key": {
                "type": "string",
                "required": False,
                "default": "text",
                "description": "Text field to filter"
            }
        },
        "example": {
            "special_chars": ["<", ">", "`"],
            "text_key": "text"
        }
    }

    def execute(self, dataset: MultimodalDataset) -> MultimodalDataset:
        return self.filter(dataset)

    def filter(self, dataset):
        special_chars = self.params.get('special_chars', ["<", ">", "`", "/"])
        text_key = self.params.get('text_key', 'text')

        special_chars_list = list(special_chars)

        def special_char_filter(row):
            text = row.get(text_key, "")
            if text is None:
                return False
            text = str(text)

            for ch in special_chars_list:
                if ch and (ch in text):
                    return False
            return True

        return dataset.filter(special_char_filter)
