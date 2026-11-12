import re
from typing import Any, Dict, Optional
from .augmentation_operator import AugmentationOperation
from base.dataset import MultimodalDataset
from ...tracking.tracking import OperationType


# Date format regular expressions
_RE_MDYYYY = re.compile(r"\b(0?[1-9]|1[0-2])/(0?[1-9]|[12]\d|3[01])/(\d{4})\b")

_MONTH_MAP = {
    "jan": "01", "january": "01",
    "feb": "02", "february": "02",
    "mar": "03", "march": "03",
    "apr": "04", "april": "04",
    "may": "05",
    "jun": "06", "june": "06",
    "jul": "07", "july": "07",
    "aug": "08", "august": "08",
    "sep": "09", "sept": "09", "september": "09",
    "oct": "10", "october": "10",
    "nov": "11", "november": "11",
    "dec": "12", "december": "12",
}
_RE_MONTH_DD_YYYY = re.compile(
    r"\b("
    r"Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|"
    r"Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?"
    r")\s+([0-3]?\d),\s*(\d{4})\b",
    flags=re.IGNORECASE,
)

_RE_USD = re.compile(r"\$(\d+(?:\.\d+)?)\b")


def _norm_mdyyyy(m: re.Match) -> str:
    mm = int(m.group(1))
    dd = int(m.group(2))
    yyyy = m.group(3)
    return f"{yyyy}-{mm:02d}-{dd:02d}"


def _norm_month_dd_yyyy(m: re.Match) -> str:
    mon_raw = m.group(1).lower()
    dd = int(m.group(2))
    yyyy = m.group(3)
    mon_key = mon_raw[:3] if mon_raw not in _MONTH_MAP else mon_raw
    mon = _MONTH_MAP.get(mon_raw, _MONTH_MAP.get(mon_key, None))
    if not mon:
        return m.group(0)
    return f"{yyyy}-{mon}-{dd:02d}"


class TextNormalizationOperation(AugmentationOperation):
    """Text normalization

    Normalize date and currency formats in text data:
    - Convert date formats to YYYY-MM-DD (supports MM/DD/YYYY and Month DD, YYYY formats)
    - Convert US dollar currency formats to a number plus USD (for example, $50 -> 50 USD)
    """

    operation_type = OperationType.AUGMENTATION

    PARAMS_SCHEMA = {
        "category": "augmentation",
        "description": "Normalize date and currency formats in text data",
        "params": {
            "text_key": {
                "type": "string",
                "required": False,
                "default": "text",
                "description": "Text field to normalize"
            }
        },
        "example": {
            "text_key": "text"
        }
    }

    def execute(self, dataset: MultimodalDataset) -> MultimodalDataset:
        text_key = self.params.get('text_key', 'text')

        def normalize_text(row):
            text = row.get(text_key, "")
            text = "" if text is None else str(text)

            text = _RE_MDYYYY.sub(_norm_mdyyyy, text)
            text = _RE_MONTH_DD_YYYY.sub(_norm_month_dd_yyyy, text)
            text = _RE_USD.sub(r"\1 USD", text)

            row[text_key] = text
            return row

        return dataset.map(normalize_text)
