from typing import Any, Dict, Optional

from .filtering_operator import FilteringOperation
from base.dataset import MultimodalDataset
from ...tracking.tracking import OperationType
from ...utils.lazy_imports import get_nlp_backend, simple_tokenize


class MeanWordLengthFilterOperation(FilteringOperation):
    """Filter text based on average word length

    Filter based on the average word length of text. Supports English (using NLTK tokenization) and Chinese (using jieba tokenization).
    Can be used to filter text with words that are too short or too long, helping control text quality.
    """

    operation_type = OperationType.FILTERING

    PARAMS_SCHEMA = {
        "category": "filtering",
        "description": "Filter text by mean word length; supports English and Chinese",
        "params": {
            "text_key": {
                "type": "string",
                "required": False,
                "default": "input_text",
                "description": "Text field to filter"
            },
            "min_len": {
                "type": "number",
                "required": False,
                "default": 3,
                "description": "Minimum mean word length"
            },
            "max_len": {
                "type": "number",
                "required": False,
                "default": 10,
                "description": "Maximum mean word length"
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
            "min_len": 3,
            "max_len": 10,
            "lang": "en"
        }
    }

    def execute(self, dataset: MultimodalDataset) -> MultimodalDataset:
        return self.filter(dataset)

    def filter(self, dataset):
        text_key = self.params.get('text_key', 'input_text')
        min_len = self.params.get('min_len', 3)
        max_len = self.params.get('max_len', 10)
        lang = self.params.get('lang', 'en').lower()

        # Use the shared NLP backend
        nlp = get_nlp_backend()
        have_nltk = nlp.ensure_nltk() if lang == "en" else False
        have_jieba = nlp.ensure_jieba() if lang == "zh" else False

        def _tokenize_en(text):
            if have_nltk:
                words = nlp.word_tokenize(text.lower())
                return [word for word in words if word.isalnum()]
            else:
                return simple_tokenize(text)

        def _tokenize_zh(text):
            if have_jieba:
                return nlp.jieba_cut(text)
            else:
                return list(text)

        def mean_word_length_filter(row):
            text = row.get(text_key)
            if not text:
                return False

            if lang == "en":
                tokens = _tokenize_en(text)
            else:
                tokens = _tokenize_zh(text)

            if not tokens:
                return False

            total_chars = sum(len(word) for word in tokens)
            num_words = len(tokens)

            mean_word_length = total_chars / num_words

            return min_len <= mean_word_length <= max_len

        return dataset.filter(mean_word_length_filter)
