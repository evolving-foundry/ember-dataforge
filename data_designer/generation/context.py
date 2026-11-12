from __future__ import annotations

import random
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict


@dataclass
class AlgorithmContext:
    seed: int = 42
    providers: Dict[str, Any] = field(default_factory=dict)
    work_directory: Path | None = None
    cancelled: threading.Event = field(default_factory=threading.Event)
    registry: Any = None

    def __post_init__(self):
        self.random = random.Random(self.seed)

    def check_cancelled(self):
        if self.cancelled.is_set():
            raise InterruptedError("generation cancelled")

    def provider(self, name):
        try:
            return self.providers[name]
        except KeyError as error:
            raise ValueError(f"required provider is not configured: {name}") from error
