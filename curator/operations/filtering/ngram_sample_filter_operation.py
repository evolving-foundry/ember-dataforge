from typing import Any, Dict, Optional

from .filtering_operator import FilteringOperation
from base.dataset import MultimodalDataset
from ...tracking.tracking import OperationType


def _simple_tokenize(text: str) -> list:
    return text.lower().split()


def _simple_ngrams(tokens: list, n: int) -> list:
    if len(tokens) < n:
        return []
    return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]


def _calculate_repetition_ratio(ngrams: list) -> float:
    if not ngrams:
        return 0.0
    total_ngrams = len(ngrams)
    unique_ngrams = len(set(ngrams))
    return (total_ngrams - unique_ngrams) / total_ngrams


class NgramSampleFilterOperation(FilteringOperation):
    """Filter text based on N-gram repetition rate

    Filter samples by computing the N-gram repetition rate in text. A high repetition rate usually indicates low-quality text or templated content.
    Supports English (using NLTK tokenization) and Chinese (single-character tokenization).
    """

    operation_type = OperationType.FILTERING

    PARAMS_SCHEMA = {
        "category": "filtering",
        "description": "Filter text by N-gram repetition rate to detect templated or repetitive content",
        "params": {
            "text_key": {
                "type": "string",
                "required": False,
                "default": "input_text",
                "description": "Text field to filter"
            },
            "n_gram": {
                "type": "integer",
                "required": False,
                "default": 2,
                "description": "N-gram order"
            },
            "threshold": {
                "type": "number",
                "required": False,
                "default": 0.5,
                "description": "Repetition-rate threshold; samples above it are filtered"
            },
            "lang": {
                "type": "string",
                "required": False,
                "default": "en",
                "description": "Language: 'en' English or 'zh' Chinese"
            }
        },
        "example": {
            "text_key": "text",
            "n_gram": 2,
            "threshold": 0.5,
            "lang": "en"
        }
    }

    def execute(self, dataset: MultimodalDataset) -> MultimodalDataset:
        return self.filter(dataset)

    def filter(self, dataset):
        def ngram_filter_batch(batch):
            text_key = self.params.get('text_key', 'input_text')
            n_gram = self.params.get('n_gram', 2)
            threshold = self.params.get('threshold', 0.5)
            lang = self.params.get('lang', 'en').lower()
            
            texts = batch.get(text_key, [])
            mask = []
            
            for text in texts:
                if not text:
                    mask.append(True)
                    continue
                
                if lang == "en":
                    tokens = _simple_tokenize(str(text))
                else:
                    tokens = list(str(text))
                
                ngrams = _simple_ngrams(tokens, n_gram)
                repetition_ratio = _calculate_repetition_ratio(ngrams)
                mask.append(repetition_ratio <= threshold)
            
            filtered_batch = {}
            for key, values in batch.items():
                if len(values) == len(mask):
                    filtered_batch[key] = [v for v, m in zip(values, mask) if m]
                else:
                    filtered_batch[key] = values
            
            return filtered_batch

        return dataset.map_batches(ngram_filter_batch, batch_size=32)
