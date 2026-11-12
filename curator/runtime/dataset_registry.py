"""DatasetRegistry (doc §14, Phase 2) — the single namespace for datasets.

TS never holds Ray Datasets; it holds DatasetRef {id, runtimeId, backend}.
Both the native Python path and the ZMQ path can register datasets here so
that the two entry points share one dataset namespace (doc §41.3 constraint 1).
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)


class DatasetRegistry:
    def __init__(
        self,
        on_created: Optional[Callable[[str, str], None]] = None,
        on_released: Optional[Callable[[str, str, str], None]] = None,
    ) -> None:
        self._datasets: Dict[str, Any] = {}
        self._backends: Dict[str, str] = {}
        self._on_created = on_created
        self._on_released = on_released

    def register(
        self,
        dataset: Any,
        dataset_id: Optional[str] = None,
        backend: str = "ray",
    ) -> str:
        """Register a dataset object; returns its reference id (ds-*)."""
        if dataset_id is None:
            dataset_id = f"ds-{uuid.uuid4().hex[:12]}"
        self._datasets[dataset_id] = dataset
        self._backends[dataset_id] = backend
        if self._on_created is not None:
            self._on_created(dataset_id, backend)
        return dataset_id

    def get(self, dataset_id: str) -> Any:
        return self._datasets[dataset_id]

    def exists(self, dataset_id: str) -> bool:
        return dataset_id in self._datasets

    def inspect(self, dataset_id: str) -> Dict[str, Any]:
        """Best-effort metadata snapshot (num_rows may cost a Ray pass)."""
        if dataset_id not in self._datasets:
            raise KeyError(f"unknown dataset: {dataset_id}")
        dataset = self._datasets[dataset_id]
        metadata: Dict[str, Any] = {
            "datasetId": dataset_id,
            "backend": self._backends.get(dataset_id, "ray"),
        }
        modalities = getattr(dataset, "modalities", None)
        if modalities:
            metadata["modalities"] = list(modalities)
        try:
            inner = getattr(dataset, "dataset", None)
            count = inner.count() if inner is not None else None
            if count is not None:
                metadata["numRows"] = int(count)
        except Exception as exc:  # noqa: BLE001 — metadata is best-effort
            logger.debug("row count unavailable for %s: %s", dataset_id, exc)
        stats = getattr(dataset, "get_stats", None)
        if callable(stats):
            try:
                metadata["extra"] = stats()
            except Exception:  # noqa: BLE001
                pass
        return metadata

    def release(self, dataset_id: str, reason: str = "released") -> bool:
        """Drop the reference; the Ray object store reclaims on GC."""
        if dataset_id not in self._datasets:
            return False
        del self._datasets[dataset_id]
        backend = self._backends.pop(dataset_id, "ray")
        if self._on_released is not None:
            self._on_released(dataset_id, backend, reason)
        return True

    def ids(self) -> list:
        return list(self._datasets.keys())
