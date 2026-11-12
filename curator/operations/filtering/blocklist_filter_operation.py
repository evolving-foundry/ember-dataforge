import re
from typing import Iterable, Callable, Optional, List, Any, Dict
from .filtering_operator import FilteringOperation
from base.dataset import MultimodalDataset
from ...tracking.tracking import OperationType


def _default_tokenize_en(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9']+", (text or "").lower())


class BlocklistFilterOperation(FilteringOperation):
    """Filter samples based on a blocklist

    Filter text samples according to blocklist terms. Supports substring matching and token matching modes.
    - match_mode='substring': count keywords that appear as substrings in the text
    - match_mode='token': count tokens that exactly match keywords
    """

    operation_type = OperationType.FILTERING

    PARAMS_SCHEMA = {
        "category": "filtering",
        "description": "Filter text samples with a blocklist vocabulary; supports substring and token matching",
        "params": {
            "blocklist": {
                "type": "array",
                "required": True,
                "description": "Blocklist vocabulary"
            },
            "max_hits": {
                "type": "integer",
                "required": False,
                "default": 0,
                "description": "Maximum allowed matches before filtering"
            },
            "match_mode": {
                "type": "string",
                "required": False,
                "default": "substring",
                "description": "Match mode: 'substring' or 'token'"
            },
            "case_insensitive": {
                "type": "boolean",
                "required": False,
                "default": True,
                "description": "Ignore case"
            },
            "text_key": {
                "type": "string",
                "required": False,
                "default": "text",
                "description": "Text field to filter"
            }
        },
        "example": {
            "blocklist": ["spam", "abuse"],
            "max_hits": 0,
            "match_mode": "substring",
            "text_key": "text"
        }
    }

    def execute(self, dataset: MultimodalDataset) -> MultimodalDataset:
        return self.filter(dataset)

    def filter(self, dataset):
        blocklist = self.params.get('blocklist', [])
        max_hits = self.params.get('max_hits', 0)
        match_mode = self.params.get('match_mode', 'substring')
        case_insensitive = self.params.get('case_insensitive', True)
        text_key = self.params.get('text_key', 'text')

        # Preprocess the blocklist
        bl = list(blocklist) if blocklist is not None else []
        if case_insensitive:
            bl = [str(x).lower() for x in bl]
        else:
            bl = [str(x) for x in bl]
        blockset = set(bl)

        def blocklist_filter(row):
            text = row.get(text_key, "")
            if text is None:
                return False
            text = str(text)
            t = text.lower() if case_insensitive else text

            hits = 0
            if len(bl) == 0:
                return True

            if match_mode == "token":
                tokens = _default_tokenize_en(t)
                for tok in tokens:
                    if tok in blockset:
                        hits += 1
                        if hits > max_hits:
                            return False
                return True

            # substring (default)
            for kw in bl:
                if kw and (kw in t):
                    hits += 1
                    if hits > max_hits:
                        return False
            return True

        return dataset.filter(blocklist_filter)
