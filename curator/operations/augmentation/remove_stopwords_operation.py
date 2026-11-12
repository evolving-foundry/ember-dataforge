from typing import Any, Dict, Optional, List

from .augmentation_operator import AugmentationOperation
from base.dataset import MultimodalDataset
from ...tracking.tracking import OperationType
from ...utils.lazy_imports import get_nlp_backend, get_default_stopwords


class RemoveStopwordsOperation(AugmentationOperation):
    """Remove stop words

    Remove English stop words from text (such as 'the', 'is', 'at', 'which', etc.).
    Supports custom stop-word lists, allowing additional stop words to be added or certain stop words to be retained.
    """

    operation_type = OperationType.AUGMENTATION

    PARAMS_SCHEMA = {
        "category": "augmentation",
        "description": "Remove English stopwords from text",
        "params": {
            "text_key": {
                "type": "string",
                "required": False,
                "default": "text",
                "description": "Input text field name"
            },
            "custom_stopwords": {
                "type": "array",
                "required": False,
                "default": [],
                "description": "Custom stopword list"
            }
        },
        "example": {
            "text_key": "text",
            "custom_stopwords": []
        }
    }

    def _get_stopwords(self, custom_stopwords: List[str]):
        """Load stop words on demand"""
        # Use the shared NLP backend to obtain stop words
        nlp = get_nlp_backend()
        stopwords = nlp.get_stopwords()

        if stopwords:
            base_stopwords = set(stopwords.words('english'))
        else:
            # Fallback strategy
            base_stopwords = get_default_stopwords()

        if custom_stopwords:
            base_stopwords.update(set(custom_stopwords))

        return base_stopwords

    def execute(self, dataset: MultimodalDataset) -> MultimodalDataset:
        text_key = self.params.get('text_key', 'text')
        custom_stopwords = self.params.get('custom_stopwords', [])

        # Load stop words on demand
        stopwords_set = self._get_stopwords(custom_stopwords)

        def remove_stopwords(row):
            text = row.get(text_key, "")
            if not text:
                row['output_text'] = text
                return row

            tokens = str(text).split()
            filtered_tokens = [word for word in tokens if word.lower() not in stopwords_set]

            row['output_text'] = ' '.join(filtered_tokens)
            return row

        return dataset.map(remove_stopwords)
