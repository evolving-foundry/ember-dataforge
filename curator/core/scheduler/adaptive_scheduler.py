from __future__ import annotations

import time
from datetime import datetime
from typing import Any, Dict, Optional

from .resource_history import ResourceHistoryDB
from .resource_models import (
    OperationResourceHint,
    OperationResourceProfile,
    OperationResourceStats,
)

# Conservative defaults
_DEFAULT_NUM_CPUS = 1.0
_DEFAULT_BATCH_SIZE = 1000
_MAX_BATCH_SIZE = 10000


class AdaptiveResourceScheduler:
    """Adaptive resource scheduler.

    Resource recommendation priority:
      user hint > historical statistics (ResourceHistoryDB) > conservative defaults

    batch_size estimation formula:
      safe_memory_mb = min(available_mem * 0.7, p90_memory_mb * 2)
      batch_size = min(safe_mem / bytes_per_row, 10000)
    """

    def __init__(self, history_db: Optional[ResourceHistoryDB] = None):
        self._history = history_db or ResourceHistoryDB()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def recommend_resources(
        self,
        operation,
        input_dataset,
        hint: Optional[OperationResourceHint] = None,
    ) -> Dict[str, Any]:
        """Recommend resource configuration.

        Returns:
            {'num_cpus': x, 'num_gpus': y, 'batch_size': z}
        """
        op_name = getattr(operation, "name", type(operation).__name__)

        # 1. User hint (highest priority)
        if hint is None:
            hint = getattr(operation, "RESOURCE_HINT", None)

        if hint is not None:
            return {
                "num_cpus": hint.num_cpus or _DEFAULT_NUM_CPUS,
                "num_gpus": hint.num_gpus or 0.0,
                "batch_size": self._estimate_batch_size(
                    hint.memory_mb, None
                ) if hint.memory_mb else _DEFAULT_BATCH_SIZE,
            }

        # 2. Historical statistics
        stats = self._history.get_stats(op_name)
        if stats:
            return {
                "num_cpus": max(stats.avg_cpu_cores, 1.0),
                "num_gpus": 1.0 if stats.is_gpu_intensive else 0.0,
                "batch_size": self._estimate_batch_size(
                    stats.p90_memory_mb, None
                ),
            }

        # 3. Conservative defaults
        return {
            "num_cpus": _DEFAULT_NUM_CPUS,
            "num_gpus": 0.0,
            "batch_size": _DEFAULT_BATCH_SIZE,
        }

    def measure_and_record(
        self,
        operation,
        input_dataset,
        output_dataset,
        start_time: float,
        end_time: float,
    ) -> OperationResourceProfile:
        """Collect CPU/memory usage data and write it to ResourceHistoryDB."""
        op_name = getattr(operation, "name", type(operation).__name__)

        cpu_cores, memory_mb = self._sample_process_resources()
        gpu_usage = self._sample_gpu_usage()

        input_rows = self._safe_count(input_dataset)
        output_rows = self._safe_count(output_dataset)

        profile = OperationResourceProfile(
            operation_name=op_name,
            execution_time_ms=(end_time - start_time) * 1000,
            cpu_usage_cores=cpu_cores,
            memory_usage_mb=memory_mb,
            gpu_usage=gpu_usage,
            input_rows=input_rows,
            output_rows=output_rows,
            timestamp=datetime.now(),
        )
        self._history.record(profile)
        return profile

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _sample_process_resources() -> tuple[float, float]:
        """Collect CPU and memory usage of the current process using psutil."""
        try:
            import psutil
            proc = psutil.Process()
            cpu_cores = proc.cpu_percent(interval=0.05) / 100.0
            memory_mb = proc.memory_info().rss / (1024 ** 2)
            return round(cpu_cores, 4), round(memory_mb, 2)
        except Exception:
            return 0.0, 0.0

    @staticmethod
    def _sample_gpu_usage() -> float:
        """Collect GPU utilization using pynvml (optional)."""
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            return round(util.gpu / 100.0, 4)
        except Exception:
            return 0.0

    @staticmethod
    def _estimate_batch_size(
        available_mb: Optional[float],
        bytes_per_row: Optional[float],
    ) -> int:
        """Estimate a safe batch_size."""
        if not available_mb:
            return _DEFAULT_BATCH_SIZE
        safe_mb = min(available_mb * 0.7, (available_mb or 512) * 2)
        if bytes_per_row:
            size = int(safe_mb * 1024 * 1024 / bytes_per_row)
        else:
            # Conservative estimate: assume about 1KB per row
            size = int(safe_mb * 1024)
        return min(max(size, 1), _MAX_BATCH_SIZE)

    @staticmethod
    def _safe_count(dataset) -> int:
        if dataset is None:
            return 0
        try:
            return dataset.dataset.count()
        except Exception:
            return 0
