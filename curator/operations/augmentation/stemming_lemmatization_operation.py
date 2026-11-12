import re
from typing import Any, Dict, Optional

from .augmentation_operator import AugmentationOperation
from base.dataset import MultimodalDataset
from ...tracking.tracking import OperationType
from ...utils.lazy_imports import get_nlp_backend


_WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")


class StemmingLemmatizationOperation(AugmentationOperation):
    """Stemming and lemmatization

    Provides two text normalization modes:
    - stem: use PorterStemmer for stemming, removing affixes
    - lemma: use WordNetLemmatizer for lemmatization, taking part-of-speech tags into account

    Stemming is more aggressive, while lemmatization is more accurate.
    """

    operation_type = OperationType.AUGMENTATION

    PARAMS_SCHEMA = {
        "category": "augmentation",
        "description": "Normalize text with stemming or lemmatization",
        "params": {
            "text_key": {
                "type": "string",
                "required": False,
                "default": "text",
                "description": "Input text field name"
            },
            "mode": {
                "type": "string",
                "required": False,
                "default": "stem",
                "description": "Mode: 'stem' or 'lemma'"
            }
        },
        "example": {
            "text_key": "text",
            "mode": "stem"
        }
    }

    def _wn_pos(self, wordnet, treebank_tag: str):
        """Convert part-of-speech tags"""
        if not treebank_tag or not wordnet:
            return None
        c = treebank_tag[0].upper()
        if c == "J":
            return wordnet.ADJ
        if c == "V":
            return wordnet.VERB
        if c == "R":
            return wordnet.ADV
        return wordnet.NOUN

    def _stem(self, stemmer, w: str) -> str:
        if stemmer:
            out = stemmer.stem(w)
            if w[:1].isupper():
                out = out[:1].upper() + out[1:]
            return out
        return w

    def _lemma(self, lemmatizer, wordnet, w: str, pos_tag: str) -> str:
        if lemmatizer:
            wn_pos = self._wn_pos(wordnet, pos_tag)
            if wn_pos:
                out = lemmatizer.lemmatize(w.lower(), pos=wn_pos)
            else:
                out = lemmatizer.lemmatize(w.lower())
            if w[:1].isupper():
                out = out[:1].upper() + out[1:]
            return out
        return w

    def execute(self, dataset: MultimodalDataset) -> MultimodalDataset:
        text_key = self.params.get('text_key', 'text')
        mode = self.params.get('mode', 'stem').lower().strip()

        # Use the shared NLP backend
        nlp = get_nlp_backend()
        have_nltk = nlp.ensure_nltk()

        # Get the required tools
        stemmer = nlp.get_porter_stemmer() if have_nltk else None
        lemmatizer = nlp.get_wordnet_lemmatizer() if have_nltk else None
        wordnet = nlp.get_wordnet() if have_nltk else None

        def stemming_lemmatization(row):
            text = row.get(text_key, "")
            text = "" if text is None else str(text)

            words = _WORD_RE.findall(text)
            if not words:
                row[text_key] = text
                return row

            if not have_nltk or not stemmer or not lemmatizer:
                # Fallback strategy: return the original text
                row[text_key] = text
                return row

            if mode == "lemma":
                try:
                    tags = nlp.pos_tag(words)
                except Exception:
                    row[text_key] = text
                    return row

                it = iter(tags)

                def repl(m):
                    w, tag = next(it)
                    return self._lemma(lemmatizer, wordnet, w, tag)

                text = _WORD_RE.sub(repl, text)
            else:
                text = _WORD_RE.sub(lambda m: self._stem(stemmer, m.group(0)), text)

            row[text_key] = text
            return row

        return dataset.map(stemming_lemmatization)
