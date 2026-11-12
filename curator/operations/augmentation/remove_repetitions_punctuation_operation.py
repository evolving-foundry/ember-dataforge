import re
from typing import Any, Dict, Optional
from .augmentation_operator import AugmentationOperation
from base.dataset import MultimodalDataset
from ...tracking.tracking import OperationType


_PUNC_RUN_RE = re.compile(r"([!！?？,，。\.、;；:：])\1{1,}")
_MULTI_DOTS_RE = re.compile(r"\.{2,}")
_SPACE_RE = re.compile(r"\s+")
_REPEAT_WORD_RE = re.compile(r"\b(\w+)(\s+\1\b){1,}", flags=re.IGNORECASE)
_REPEAT_CHAR_RE = re.compile(r"(.)\1{2,}")


class RemoveRepetitionsPunctuationOperation(AugmentationOperation):
    """Remove repeated punctuation and repeated words

    Clean up repeated punctuation marks in text (such as multiple consecutive question marks or commas) and repeated words.
    Also fix extra spaces. This helps normalize text formatting and improve readability.
    """

    operation_type = OperationType.AUGMENTATION

    PARAMS_SCHEMA = {
        "category": "augmentation",
        "description": "Remove repeated punctuation and repeated words",
        "params": {
            "text_key": {
                "type": "string",
                "required": False,
                "default": "text",
                "description": "Text field to process"
            },
            "keep_repeat_char": {
                "type": "integer",
                "required": False,
                "default": 2,
                "description": "Maximum repetitions of a character to retain"
            }
        },
        "example": {
            "text_key": "text",
            "keep_repeat_char": 2
        }
    }

    def execute(self, dataset: MultimodalDataset) -> MultimodalDataset:
        text_key = self.params.get('text_key', 'text')
        keep_repeat_char = self.params.get('keep_repeat_char', 2)

        def remove_repetitions(row):
            text = row.get(text_key, "")
            text = "" if text is None else str(text)

            text = _PUNC_RUN_RE.sub(r"\1", text)
            text = _MULTI_DOTS_RE.sub("...", text)

            text = _SPACE_RE.sub(" ", text).strip()

            text = _REPEAT_WORD_RE.sub(r"\1", text)

            if keep_repeat_char >= 1:
                text = _REPEAT_CHAR_RE.sub(lambda m: m.group(1) * keep_repeat_char, text)

            row[text_key] = text
            return row

        return dataset.map(remove_repetitions)
