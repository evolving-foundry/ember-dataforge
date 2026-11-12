import random
from typing import Any, Dict, Optional

from .augmentation_operator import AugmentationOperation
from base.dataset import MultimodalDataset
from ...tracking.tracking import OperationType
from ...utils.lazy_imports import get_nlp_backend


class RandomNounAugmentationOperation(AugmentationOperation):
    """Random noun replacement augmentation

    Randomly select nouns in the text and replace them with their synonyms. This can increase text diversity
    while preserving semantic consistency.
    """

    operation_type = OperationType.AUGMENTATION

    PARAMS_SCHEMA = {
        "category": "augmentation",
        "description": "Randomly select nouns in text and replace them with synonyms",
        "params": {
            "text_key": {
                "type": "string",
                "required": False,
                "default": "text",
                "description": "Input text field name"
            },
            "replacement_prob": {
                "type": "number",
                "required": False,
                "default": 0.5,
                "description": "Noun replacement probability"
            }
        },
        "example": {
            "text_key": "text",
            "replacement_prob": 0.5
        }
    }

    def execute(self, dataset: MultimodalDataset) -> MultimodalDataset:
        text_key = self.params.get('text_key', 'text')
        replacement_prob = self.params.get('replacement_prob', 0.5)

        # Use the shared NLP backend
        nlp = get_nlp_backend()
        have_nltk = nlp.ensure_nltk()

        def get_noun_synonyms(word):
            """Get synonyms for a noun"""
            if not have_nltk:
                return []

            wordnet = nlp.get_wordnet()
            if not wordnet:
                return []

            synonyms = set()
            for syn in wordnet.synsets(word, pos=wordnet.NOUN):
                for lemma in syn.lemmas():
                    synonym = lemma.name().replace('_', ' ')
                    if synonym.lower() != word.lower():
                        synonyms.add(synonym)
            return list(synonyms)

        def augment_text(row):
            text = row.get(text_key, "")
            if not text:
                row['output_text'] = text
                return row

            if have_nltk:
                tokens = nlp.word_tokenize(str(text))
                tagged_words = nlp.pos_tag(tokens)
            else:
                # Fallback strategy
                row['output_text'] = str(text)
                return row

            result_words = []

            for word, tag in tagged_words:
                # Check whether the token is a noun (NN, NNS, NNP, NNPS)
                if tag.startswith('NN') and random.random() < replacement_prob:
                    synonyms = get_noun_synonyms(word)

                    if synonyms:
                        replacement = random.choice(synonyms)
                        result_words.append(replacement)
                    else:
                        result_words.append(word)
                else:
                    result_words.append(word)

            row['output_text'] = ' '.join(result_words).replace(" ,", ",").replace(" .", ".")
            return row

        return dataset.map(augment_text)
