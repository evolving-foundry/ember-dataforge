from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Mapping, Protocol, Tuple

from .datasets import RecordEnvelope

ALGORITHM_CONTRACT_VERSION = "dataforge-algorithm-v1"


@dataclass(frozen=True)
class AlgorithmSpec:
    id: str
    implementation_version: str
    config_schema: Dict[str, Any]
    input_schema: str = "record-v1"
    output_schema: str = "record-v1"
    scope: str = "record"
    capabilities: Tuple[str, ...] = ("transform",)
    cardinality: str = "one-to-one"
    execution: Tuple[str, ...] = ("python",)
    resources: Dict[str, Any] = field(default_factory=dict)
    determinism: str = "deterministic"
    upstream: Dict[str, Any] = field(default_factory=dict)
    contract_version: str = ALGORITHM_CONTRACT_VERSION

    def __post_init__(self):
        if not self.id or "." not in self.id:
            raise ValueError("algorithm id must be namespaced")
        if self.scope not in {"record", "batch", "dataset"}:
            raise ValueError("unsupported algorithm scope")
        if self.contract_version != ALGORITHM_CONTRACT_VERSION:
            raise ValueError("unsupported algorithm contract version")
        if not self.implementation_version:
            raise ValueError("algorithm implementation version is required")


@dataclass(frozen=True)
class AlgorithmDescriptor:
    spec: AlgorithmSpec
    factory: str
    package_version: str = "builtin"


@dataclass(frozen=True)
class AlgorithmRequest:
    records: Tuple[RecordEnvelope, ...]
    config: Mapping[str, Any]
    run_id: str
    target_new_records: int | None = None
    max_attempts: int = 10000
    source_refs: Tuple[str, ...] = ()


@dataclass
class AlgorithmResult:
    records: Iterable[RecordEnvelope]
    rejected: Dict[str, int] = field(default_factory=dict)
    diagnostics: Dict[str, Any] = field(default_factory=dict)


class Algorithm(Protocol):
    def describe(self) -> AlgorithmSpec: ...
    def run(self, request: AlgorithmRequest, context: Any) -> AlgorithmResult: ...
