import logging

logger = logging.getLogger(__name__)

"""
Text token-count mapping operation.

Counts tokens in dataset text fields using these tokenizers:
  - whitespace: split on whitespace (default; no extra dependencies)
  - char:       count characters
  - tiktoken:   OpenAI tiktoken encoding (requires tiktoken)
  - huggingface: HuggingFace tokenizers (requires transformers)
"""

from typing import Any, Dict, List, Optional

from .mapping_operator import MappingOperation
from base.dataset import MultimodalDataset


def _build_row_processor(
    tokenizer_type: str,
    model_name: str,
    text_field: str,
    token_count_field: str,
    char_count_field: Optional[str],
):
    """
    Build the row-processing function.

    Encapsulate tokenization logic in a plain function to avoid serializing large objects
    (such as Tokenizer instances) to Ray workers, reducing memory usage and serialization overhead.
    All dependencies are imported on demand inside the worker.
    """



    def process_row(row: Dict[str, Any]) -> Dict[str, Any]:
        text = row.get(text_field) or ""
        text = str(text)

        if tokenizer_type == "tiktoken":
            try:
                import tiktoken
                try:
                    enc = tiktoken.encoding_for_model(model_name)
                except KeyError:
                    enc = tiktoken.get_encoding(model_name)
                count = len(enc.encode(text))
            except ImportError:
                # tiktoken is not installed; fall back to whitespace tokenization
                count = len(text.split())

        elif tokenizer_type == "huggingface":
            try:
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                count = len(tokenizer.encode(text, add_special_tokens=False))
            except ImportError:
                # transformers is not installed; fall back to whitespace tokenization
                count = len(text.split())

        elif tokenizer_type == "char":
            count = len(text)

        else:
            # whitespace (default)
            count = len(text.split())

        row[token_count_field] = count
        if char_count_field:
            row[char_count_field] = len(text)
        return row

    return process_row


class TokenCountMappingOperation(MappingOperation):
    """
    Text token count mapping operation.

    Compute the token count for the specified text field of each record in the dataset
    and write the result to a new field.

    Parameters (passed via params):
        text_field (str):
            Name of the text field to count, default 'text'.
        token_count_field (str):
            Target field for the token count, default 'token_count'.
        char_count_field (str | None):
            Target field for the character count; do not write when None, default None.
        tokenizer (str):
            Tokenization method, one of 'whitespace' (default) / 'char' /
            'tiktoken' / 'huggingface'.
        model_name (str):
            When tokenizer is 'tiktoken', specify the encoding name (such as
            'cl100k_base' or 'gpt-4');
            when tokenizer is 'huggingface', specify the model name (such as
            'bert-base-chinese').
            Default 'cl100k_base'.
        print_stats (bool):
            Whether to print aggregate statistics after execution, default True.

    Example::

        op = TokenCountMappingOperation({
            'text_field': 'content',
            'token_count_field': 'content_tokens',
            'char_count_field': 'content_chars',
            'tokenizer': 'whitespace',
            'print_stats': True,
        })
        result_dataset = op.execute(dataset)
    """


    PARAMS_SCHEMA = {
        "category":    "mapping",
        "description": "Count tokens in a text field and write the result to a new field. Supports whitespace, char, tiktoken, and huggingface tokenizers",
        "params": {
            "text_field": {
                "type":        "string",
                "required":    False,
                "default":     "text",
                "description": "Text field to count",
            },
            "token_count_field": {
                "type":        "string",
                "required":    False,
                "default":     "token_count",
                "description": "Target field for the token count",
            },
            "char_count_field": {
                "type":        ["string", "null"],
                "required":    False,
                "default":     None,
                "description": "Target field for character count; no character count is written when null",
            },
            "tokenizer": {
                "type":        "string",
                "required":    False,
                "enum":        ["whitespace", "char", "tiktoken", "huggingface"],
                "default":     "whitespace",
                "description": "Tokenizer: whitespace / char (suitable for Chinese) / tiktoken (OpenAI BPE) / huggingface (HF tokenizer)",
            },
            "model_name": {
                "type":        "string",
                "required":    False,
                "default":     "cl100k_base",
                "description": "Encoding name for tiktoken (for example cl100k_base), or model name for huggingface",
            },
            "print_stats": {
                "type":        "boolean",
                "required":    False,
                "default":     True,
                "description": "Print aggregate token and character statistics after execution",
            },
        },
        "example": {
            "text_field":        "content",
            "token_count_field": "token_count",
            "char_count_field":  "char_count",
            "tokenizer":         "whitespace",
            "print_stats":       True,
        },
    }
    def execute(self, dataset: MultimodalDataset) -> MultimodalDataset:
        """Run token counting on the dataset and return the extended dataset."""
        text_field = self.params.get("text_field", "text")
        token_count_field = self.params.get("token_count_field", "token_count")
        char_count_field = self.params.get("char_count_field", None)
        tokenizer_type = self.params.get("tokenizer", "whitespace")
        model_name = self.params.get("model_name", "cl100k_base")
        print_stats = self.params.get("print_stats", True)

        row_fn = _build_row_processor(
            tokenizer_type=tokenizer_type,
            model_name=model_name,
            text_field=text_field,
            token_count_field=token_count_field,
            char_count_field=char_count_field,
        )

        new_ray_dataset = dataset.dataset.map(row_fn)

        result = MultimodalDataset(
            dataset=new_ray_dataset,
            modalities=dataset.modalities,
        )

        if print_stats:
            self._print_stats(result, token_count_field, char_count_field)

        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _print_stats(
        self,
        dataset: MultimodalDataset,
        token_count_field: str,
        char_count_field: Optional[str],
    ) -> None:
        """Collect and print token / character count statistics."""
        try:
            rows = dataset.dataset.take_all()
            total = len(rows)
            if total == 0:
                logger.info("[TokenCountMappingOperation] Dataset is empty; no statistics are available.")
                return

            token_counts: List[int] = [
                r.get(token_count_field, 0) for r in rows
            ]
            total_tokens = sum(token_counts)
            avg_tokens = total_tokens / total
            min_tokens = min(token_counts)
            max_tokens = max(token_counts)

            logger.info(
                "\n[TokenCountMappingOperation] Token statistics summary\n"
                "  Tokenizer    : %s\n"
                "  Text field   : %s\n"
                "  Records      : %s\n"
                "  Total tokens : %s\n"
                "  Mean tokens  : %.2f\n"
                "  Min tokens   : %s\n"
                "  Max tokens   : %s",
                self.params.get('tokenizer', 'whitespace'),
                self.params.get('text_field', 'text'),
                total,
                total_tokens,
                avg_tokens,
                min_tokens,
                max_tokens
            )

            if char_count_field:
                char_counts: List[int] = [
                    r.get(char_count_field, 0) for r in rows
                ]
                total_chars = sum(char_counts)
                logger.info(
                    "  Total chars  : %s\n"
                    "  Mean chars   : %.2f",
                    total_chars,
                    total_chars / total
                )

            logger.debug("")
        except Exception as exc:
            logger.error("[TokenCountMappingOperation] Failed to collect statistics: %s", exc)
