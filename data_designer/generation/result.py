from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List
from data_designer.contracts.datasets import DatasetPackage, RecordEnvelope

@dataclass
class GenerationResult:
    records: List[RecordEnvelope]
    generated_count: int
    input_count: int
    status: str
    report: Dict[str, Any]
    provenance: List[Dict[str, Any]]
    package: DatasetPackage | None = None

    def save(self, uri, *, overwrite=False):
        from data_designer.storage.local import save_result
        self.package = save_result(self, uri, overwrite=overwrite)
        return self.package
