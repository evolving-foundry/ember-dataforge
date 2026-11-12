import os
import math
import pickle
from collections import Counter
from typing import Any, Dict, Optional

from .filtering_operator import FilteringOperation
from base.dataset import MultimodalDataset
from ...tracking.tracking import OperationType
from ...utils.lazy_imports import get_nlp_backend, simple_tokenize, simple_ngrams


class CiderSampleFilterOperation(FilteringOperation):
    """Filter low-similarity text based on CIDEr score

    Used for image captioning tasks, measuring similarity by computing the CIDEr score between candidate captions and reference captions.
    CIDEr is an evaluation metric designed specifically for image captioning and accounts for TF-IDF weighting.
    """

    operation_type = OperationType.FILTERING

    PARAMS_SCHEMA = {
        "category": "filtering",
        "description": "Filter low-similarity text by CIDEr score for image-captioning tasks",
        "params": {
            "threshold": {
                "type": "number",
                "required": False,
                "default": 0.5,
                "description": "CIDEr score threshold; samples below it are filtered"
            },
            "n": {
                "type": "integer",
                "required": False,
                "default": 4,
                "description": "Maximum n-gram order"
            },
            "sigma": {
                "type": "number",
                "required": False,
                "default": 6.0,
                "description": "Gaussian penalty coefficient"
            },
            "df_mode": {
                "type": "string",
                "required": False,
                "default": "coco-val-df",
                "description": "Document-frequency mode"
            },
            "idf_path": {
                "type": "string",
                "required": False,
                "default": None,
                "description": "Custom IDF dictionary file path"
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
            }
        },
        "example": {
            "threshold": 0.5,
            "n": 4,
            "reference_key": "reference",
            "text_key": "prediction"
        }
    }

    def execute(self, dataset: MultimodalDataset) -> MultimodalDataset:
        return self.filter(dataset)

    def filter(self, dataset):
        threshold = self.params.get('threshold', 0.5)
        n = self.params.get('n', 4)
        sigma = self.params.get('sigma', 6.0)
        idf_path = self.params.get('idf_path')
        reference_key = self.params.get('reference_key', 'reference_text')
        text_key = self.params.get('text_key', 'input_text')

        # Load IDF data
        idf_dict, corpus_size = self._load_idf_data(idf_path)

        # Use the shared NLP backend
        nlp = get_nlp_backend()
        have_nltk = nlp.ensure_nltk()

        def _tokenize(text):
            if have_nltk:
                return nlp.word_tokenize(text.lower())
            else:
                return simple_tokenize(text)

        def _get_ngrams_count(tokens, n_val):
            if have_nltk:
                nltk_ngrams = nlp.get_ngrams()
                if nltk_ngrams:
                    return Counter(nltk_ngrams(tokens, n_val))
            # Fallback strategy
            return Counter(simple_ngrams(tokens, n_val))

        def _get_idf_value(ngram):
            if idf_dict and ngram in idf_dict:
                return idf_dict[ngram]
            return math.log(max(corpus_size, 1.0))

        def cider_filter(row):
            reference_text = row.get(reference_key)
            prediction_text = row.get(text_key)

            if not reference_text or not prediction_text:
                return False

            ref_tokens = _tokenize(reference_text)
            pred_tokens = _tokenize(prediction_text)

            l_ref = len(ref_tokens)
            l_pred = len(pred_tokens)

            cider_scores_per_n_gram_level = []

            for k in range(1, n + 1):
                ref_counts = _get_ngrams_count(ref_tokens, k)
                pred_counts = _get_ngrams_count(pred_tokens, k)

                dot_product = 0.0
                norm_ref = 0.0
                norm_pred = 0.0

                all_ngrams = set(ref_counts.keys()) | set(pred_counts.keys())

                for ngram in all_ngrams:
                    idf_val = _get_idf_value(ngram)

                    pred_count = pred_counts.get(ngram, 0)
                    ref_count = ref_counts.get(ngram, 0)

                    clipped_match = min(pred_count, ref_count) * (idf_val ** 2)

                    val_pred = pred_count * idf_val
                    val_ref = ref_count * idf_val

                    dot_product += clipped_match
                    norm_pred += (val_pred ** 2)
                    norm_ref += (val_ref ** 2)

                norm_pred = math.sqrt(norm_pred)
                norm_ref = math.sqrt(norm_ref)

                if norm_pred > 0 and norm_ref > 0:
                    score_k = dot_product / (norm_pred * norm_ref)
                else:
                    score_k = 0.0

                cider_scores_per_n_gram_level.append(score_k)

            if cider_scores_per_n_gram_level:
                avg_score = sum(cider_scores_per_n_gram_level) / len(cider_scores_per_n_gram_level)
            else:
                avg_score = 0.0
            penalty = math.exp(-(l_pred - l_ref) ** 2 / (2 * sigma ** 2))

            final_cider_score = avg_score * penalty

            return final_cider_score >= threshold

        return dataset.filter(cider_filter)

    def _load_idf_data(self, idf_path):
        """Load an IDF pickle from a trusted local path.

        Pickle can execute arbitrary code. ``idf_path`` is therefore a trusted
        operator input, not an untrusted user-facing upload path.
        """
        default_size = 100000

        if idf_path and os.path.exists(idf_path):
            try:
                with open(idf_path, 'rb') as f:
                    data = pickle.load(f, encoding='latin1')

                if isinstance(data, tuple) and len(data) == 2:
                    return data[0], int(data[1])
                elif isinstance(data, dict):
                    if data:
                        max_idf = max(data['df'].values())
                        try:
                            estimated_N = int(math.exp(max_idf))
                            return data, estimated_N
                        except:
                            return data, default_size
                    else:
                        return data, default_size
                else:
                    return None, default_size
            except Exception:
                return None, default_size
        else:
            return None, default_size
