import os
import urllib.request
import importlib.resources
from typing import Any, Dict, Optional
from .augmentation_operator import AugmentationOperation
from base.dataset import MultimodalDataset
from ...tracking.tracking import OperationType

try:
    from symspellpy import SymSpell
    SYM_SPELL_AVAILABLE = True
except ImportError:
    SYM_SPELL_AVAILABLE = False


_DEFAULT_DICT_URL = (
    "https://raw.githubusercontent.com/mammothb/symspellpy/master/"
    "symspellpy/frequency_dictionary_en_82_765.txt"
)

_FALLBACK = {
    "teh": "the",
    "adress": "address",
    "wierd": "weird",
    "definately": "definitely",
    "recieve": "receive",
}


def _ensure_dictionary_file(dictionary_path: str) -> str:
    if os.path.exists(dictionary_path):
        return dictionary_path
    os.makedirs(os.path.dirname(dictionary_path) or ".", exist_ok=True)
    urllib.request.urlretrieve(_DEFAULT_DICT_URL, dictionary_path)
    return dictionary_path


class SpellingCorrectionOperation(AugmentationOperation):
    """Spelling correction

    Automatically correct spelling errors in text using the SymSpell library. Based on an edit-distance algorithm,
    it can quickly find and correct spelling mistakes. Supports loading custom dictionaries.
    """

    operation_type = OperationType.AUGMENTATION

    PARAMS_SCHEMA = {
        "category": "augmentation",
        "description": "Automatically correct spelling errors in text",
        "params": {
            "text_key": {
                "type": "string",
                "required": False,
                "default": "text",
                "description": "Text field to correct"
            },
            "max_dictionary_edit_distance": {
                "type": "integer",
                "required": False,
                "default": 2,
                "description": "Maximum dictionary edit distance"
            },
            "max_edit_distance": {
                "type": "integer",
                "required": False,
                "default": 2,
                "description": "Maximum edit distance during correction"
            },
            "prefix_length": {
                "type": "integer",
                "required": False,
                "default": 7,
                "description": "Prefix length"
            },
            "dictionary_path": {
                "type": "string",
                "required": False,
                "default": None,
                "description": "Custom dictionary file path"
            }
        },
        "example": {
            "text_key": "text",
            "max_edit_distance": 2
        }
    }

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__(params)
        if not SYM_SPELL_AVAILABLE:
            raise ImportError("symspellpy is required for SpellingCorrectionOperation. "
                            "Install it with: pip install symspellpy")

        self._init_symspell()

    def _init_symspell(self):
        max_dictionary_edit_distance = self.params.get('max_dictionary_edit_distance', 2)
        max_edit_distance = self.params.get('max_edit_distance', 2)
        prefix_length = self.params.get('prefix_length', 7)
        dictionary_path = self.params.get('dictionary_path')

        if max_edit_distance > max_dictionary_edit_distance:
            max_edit_distance = max_dictionary_edit_distance
        self.max_edit_distance = max_edit_distance

        self.sym_spell = SymSpell(
            max_dictionary_edit_distance=max_dictionary_edit_distance,
            prefix_length=int(prefix_length),
        )

        if dictionary_path:
            dictionary_path = _ensure_dictionary_file(dictionary_path)
        else:
            try:
                dictionary_path = str(
                    importlib.resources.files("symspellpy") / "frequency_dictionary_en_82_765.txt"
                )
            except Exception:
                dictionary_path = None

        if dictionary_path:
            loaded = self.sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)
            if not loaded:
                raise RuntimeError(f"Failed to load SymSpell dictionary from: {dictionary_path}")

    def execute(self, dataset: MultimodalDataset) -> MultimodalDataset:
        text_key = self.params.get('text_key', 'text')

        def correct_spelling(row):
            text = row.get(text_key, "")
            text = "" if text is None else str(text)

            lowered = text.lower()
            for k, v in _FALLBACK.items():
                lowered = lowered.replace(k, v)

            suggestions = self.sym_spell.lookup_compound(lowered, max_edit_distance=self.max_edit_distance)
            corrected = suggestions[0].term if suggestions else lowered

            if text[:1].isupper() and corrected:
                corrected = corrected[:1].upper() + corrected[1:]

            row[text_key] = corrected
            return row

        return dataset.map(correct_spelling)
