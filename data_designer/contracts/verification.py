from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Tuple


@dataclass(frozen=True)
class VerificationResult:
    status: str
    verifier: str
    version: str = "1"
    passed: bool | None = None
    score: float | None = None
    detail: str = ""
    evidence: Tuple[str, ...] = ()

    def __post_init__(self):
        if self.status not in {"completed", "error", "unavailable"}:
            raise ValueError("unknown verification status")
        if self.status == "completed":
            if type(self.passed) is not bool or isinstance(self.score, bool) or not isinstance(self.score, (int, float)) or not math.isfinite(self.score) or not 0 <= self.score <= 1:
                raise ValueError("completed verification requires a finite score and boolean verdict")
        elif self.passed is not None or self.score is not None:
            raise ValueError("incomplete verification cannot carry a verdict or score")

    def as_dict(self):
        value = {"status": self.status, "verifier": self.verifier, "version": self.version,
                 "detail": self.detail, "evidence": list(self.evidence)}
        if self.status == "completed":
            value.update(passed=self.passed, score=self.score)
        return value
