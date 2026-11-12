from typing import Any, Dict

from .filtering_operator import FilteringOperation
from base.dataset import MultimodalDataset
from ...tracking.tracking import OperationType
from ...utils.lazy_imports import get_nlp_backend, simple_tokenize


class BleuSampleFilterOperation(FilteringOperation):
    """Filter low-similarity text based on BLEU score

    Used to assess text quality by computing the BLEU score between source text and reference text to measure their similarity.
    """

    operation_type = OperationType.FILTERING

    PARAMS_SCHEMA = {
        "category": "filtering",
        "description": "Filter low-similarity text by BLEU score for translation or text-generation quality",
        "params": {
            "n_grams": {
                "type": "integer",
                "required": False,
                "default": 1,
                "description": "n-gram order used to compute BLEU"
            },
            "reference_key": {
                "type": "string",
                "required": False,
                "default": "reference_text",
                "description": "Reference text field name"
            },
            "text_key": {
                "type": "string",
                "required": False,
                "default": "input_text",
                "description": "Source text field name"
            },
            "threshold": {
                "type": "number",
                "required": False,
                "default": 0.5,
                "description": "BLEU score threshold; samples below it are filtered"
            }
        },
        "example": {
            "n_grams": 1,
            "reference_key": "reference",
            "text_key": "text",
            "threshold": 0.9
        }
    }

    def execute(self, dataset: MultimodalDataset) -> MultimodalDataset:
        return self.filter(dataset)

    def filter(self, dataset):
        n_grams = self.params.get('n_grams', 1)
        reference_key = self.params.get('reference_key', 'reference_text')
        text_key = self.params.get('text_key', 'input_text')
        threshold = self.params.get('threshold', 0.5)

        # Use the shared NLP backend
        nlp = get_nlp_backend()
        have_nltk = nlp.ensure_nltk()

        def bleu_filter(row):
            reference = row.get(reference_key)
            prediction = row.get(text_key)

            if not reference or not prediction:
                return False

            if have_nltk:
                try:
                    reference_tokens = nlp.word_tokenize(str(reference))
                    prediction_tokens = nlp.word_tokenize(str(prediction))
                except Exception:
                    reference_tokens = simple_tokenize(str(reference))
                    prediction_tokens = simple_tokenize(str(prediction))

                if not reference_tokens or not prediction_tokens:
                    return False

                try:
                    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
                    smoothie = SmoothingFunction().method4
                    weights = tuple([1.0 / n_grams] * n_grams)
                    bleu_score = sentence_bleu(
                        [reference_tokens],
                        prediction_tokens,
                        weights=weights,
                        smoothing_function=smoothie,
                    )
                except Exception:
                    bleu_score = 0.0

                return bleu_score >= threshold
            else:
                # Fallback strategy: use simple tokenization
                ref_tokens = simple_tokenize(str(reference))
                pred_tokens = simple_tokenize(str(prediction))
                if not ref_tokens or not pred_tokens:
                    return False
                matches = sum(1 for w in pred_tokens if w in ref_tokens)
                score = matches / max(len(pred_tokens), 1)
                return score >= threshold

        return dataset.filter(bleu_filter)
