from __future__ import annotations

import json
import os
import threading
from typing import Any, Dict, List, Optional

from .resource_models import OperationResourceProfile, OperationResourceStats


class ResourceHistoryDB:
    """Operation resource history store, persisted to a local JSON file and thread-safe.

    The default file path is .cache/resource_history.json.
    Each operation retains at most the latest 100 records.
    """

    DEFAULT_PATH = ".cache/resource_history.json"
    MAX_RECORDS_PER_OP = 100

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or self.DEFAULT_PATH
        os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
        self._lock = threading.Lock()
        self._data: Dict[str, List[Dict[str, Any]]] = self._load()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def record(self, profile: OperationResourceProfile) -> None:
        """Append a resource usage record; keep only the latest 100 records when the limit is exceeded."""
        with self._lock:
            op = profile.operation_name
            records = self._data.setdefault(op, [])
            records.append(profile.model_dump(mode="json"))
            if len(records) > self.MAX_RECORDS_PER_OP:
                self._data[op] = records[-self.MAX_RECORDS_PER_OP:]
            self._save()

    def get_stats(self, operation_name: str) -> Optional[OperationResourceStats]:
        """Aggregate and compute historical statistics."""
        with self._lock:
            records = self._data.get(operation_name, [])
        if not records:
            return None

        profiles = [OperationResourceProfile.model_validate(r) for r in records]
        n = len(profiles)

        cpu_values = sorted(p.cpu_usage_cores for p in profiles)
        mem_values = sorted(p.memory_usage_mb for p in profiles)
        time_values = sorted(p.execution_time_ms for p in profiles)
        gpu_values = [p.gpu_usage for p in profiles]

        def mean(lst: list) -> float:
            return sum(lst) / len(lst) if lst else 0.0

        def percentile(sorted_lst: list, p: float) -> float:
            if not sorted_lst:
                return 0.0
            idx = int(len(sorted_lst) * p / 100)
            idx = min(idx, len(sorted_lst) - 1)
            return float(sorted_lst[idx])

        avg_gpu = mean(gpu_values)
        avg_mem = mean(mem_values)

        return OperationResourceStats(
            operation_name=operation_name,
            sample_count=n,
            avg_cpu_cores=round(mean(cpu_values), 4),
            avg_memory_mb=round(avg_mem, 2),
            p90_memory_mb=round(percentile(mem_values, 90), 2),
            p90_execution_time_ms=round(percentile(time_values, 90), 2),
            is_gpu_intensive=avg_gpu > 0.5,
            is_memory_intensive=avg_mem > 2048,
        )

    def clear(self, operation_name: Optional[str] = None) -> None:
        """Clear history records (all or for a specified operation)."""
        with self._lock:
            if operation_name:
                self._data.pop(operation_name, None)
            else:
                self._data.clear()
            self._save()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load(self) -> Dict[str, List[Dict[str, Any]]]:
        if not os.path.exists(self.db_path):
            return {}
        try:
            with open(self.db_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def _save(self) -> None:
        try:
            with open(self.db_path, "w", encoding="utf-8") as f:
                json.dump(self._data, f, ensure_ascii=False, default=str)
        except Exception:
            pass
