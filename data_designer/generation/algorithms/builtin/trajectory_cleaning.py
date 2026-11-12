"""Evidence-backed preprocessing operators (FR-B): B1 cleaning / B2 PII and secrets / B4 omission is declaration.

B1: Remove model-internal reasoning blocks, truncate tool receipt text (default 300 characters, configurable),
    and preserve original message indices and error markers.
B2: PII / secret cleaning (configurable, auditable rules).
B4: Any lossy transformation must write filtering statistics to the omission log (OmissionLog);
    omitted content must not appear in the "known evidence" set of later analyses.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

from .trajectory_data import OmissionEntry, TrajectoryCorpus
from .trajectory_processing import _Processor

#: Model-internal reasoning block markers (<think>...</think>, reasoning: line blocks)
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_THINK_OPEN = re.compile(r"<think>.*$", re.DOTALL | re.IGNORECASE)
_REASONING_FIELD = re.compile(r"(?m)^\s*reasoning\s*[:=].*$")

#: B2 default secret rules (configurable, auditable)
_DEFAULT_SECRET_PATTERNS = [
    ("api_key", re.compile(r"(?i)(sk-[A-Za-z0-9]{16,}|api[_-]?key\s*[:=]\s*\S+)")),
    ("bearer_token", re.compile(r"(?i)bearer\s+[A-Za-z0-9._\-]{16,}")),
    ("aws_access_key", re.compile(r"AKIA[0-9A-Z]{16}")),
    ("private_key", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("password", re.compile(r"(?i)(password|passwd|pwd)\s*[:=]\s*\S+")),
]

#: B2 default PII rules
_DEFAULT_PII_PATTERNS = [
    ("email", re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")),
    ("phone_cn", re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")),
    ("ipv4", re.compile(r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])")),
]


class ReasoningStripOperation(_Processor):
    """B1: Remove model-internal reasoning blocks; preserve indices and error markers; record omissions."""

    name = "reasoning_strip"
    PARAMS_SCHEMA = {"params": {}}

    def execute(self, corpus: TrajectoryCorpus) -> TrajectoryCorpus:
        for rec in corpus.records:
            for step in rec.steps:
                stripped = _THINK_BLOCK.sub("", step.observation)
                stripped = _THINK_OPEN.sub("", stripped)
                stripped = _REASONING_FIELD.sub("", stripped)
                if stripped != step.observation:
                    corpus.omissions.add(OmissionEntry(
                        op=self.name, trajectory_id=rec.trajectory_id,
                        step_index=step.index, field="observation",
                        reason="model_internal_reasoning", chars_removed=len(step.observation) - len(stripped),
                    ))
                    step.observation = stripped
                    step.redacted = True
        return corpus


class ToolReceiptTruncationOperation(_Processor):
    """B1: Truncate tool receipt text (default 300 characters, configurable); preserve indices and error markers."""

    name = "tool_receipt_truncation"
    PARAMS_SCHEMA = {
        "params": {
            "max_chars": {
                "type": "int", "required": False, "default": 300,
                "description": "Maximum characters to retain in tool-result observations",
            },
        }
    }

    def execute(self, corpus: TrajectoryCorpus) -> TrajectoryCorpus:
        max_chars = int(self.params["max_chars"])
        for rec in corpus.records:
            for step in rec.steps:
                if step.tool and len(step.observation) > max_chars:
                    removed = len(step.observation) - max_chars
                    corpus.omissions.add(OmissionEntry(
                        op=self.name, trajectory_id=rec.trajectory_id,
                        step_index=step.index, field="observation",
                        reason=f"tool_receipt_truncated_{max_chars}", chars_removed=removed,
                    ))
                    step.observation = step.observation[:max_chars]
                    step.redacted = True
        return corpus


class PIIScrubbingOperation(_Processor):
    """B2: Clean PII / secrets; configurable, auditable rules (log every hit in the omission log)."""

    name = "pii_secret_scrubbing"
    PARAMS_SCHEMA = {
        "params": {
            "extra_patterns": {
                "type": "list", "required": False, "default": [],
                "description": "Additional rules; elements are [name, pattern] (regular-expression strings)",
            },
            "scrub_task": {
                "type": "bool", "required": False, "default": True,
                "description": "Whether to also clean the task field",
            },
        }
    }

    def __init__(self, params: Dict[str, Any] | None = None):
        super().__init__(params)
        self._patterns: List[tuple] = list(_DEFAULT_SECRET_PATTERNS) + list(_DEFAULT_PII_PATTERNS)
        for name, pattern in self.params["extra_patterns"]:
            self._patterns.append((str(name), re.compile(str(pattern))))

    def execute(self, corpus: TrajectoryCorpus) -> TrajectoryCorpus:
        scrub_task = bool(self.params["scrub_task"])
        for rec in corpus.records:
            if scrub_task:
                rec.task = self._scrub_text(corpus, rec, -1, "task", rec.task)
            for step in rec.steps:
                step.observation = self._scrub_text(corpus, rec, step.index, "observation", step.observation)
                step.action = self._scrub_text(corpus, rec, step.index, "action", step.action)
        return corpus

    def _scrub_text(self, corpus: TrajectoryCorpus, rec, step_index: int, field: str, text: str) -> str:
        if not text:
            return text
        result = text
        for name, pattern in self._patterns:
            def _mask(match: re.Match, _name: str = name) -> str:
                corpus.omissions.add(OmissionEntry(
                    op=self.name, trajectory_id=rec.trajectory_id,
                    step_index=step_index, field=field,
                    reason=f"pii_or_secret:{_name}", chars_removed=len(match.group(0)),
                ))
                return f"[REDACTED:{_name}]"

            result = pattern.sub(_mask, result)
        return result
