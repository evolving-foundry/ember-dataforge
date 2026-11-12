"""PipelineExecutor (doc §17/§41 Phase 1) — wraps the existing kernel.

Phase 1 semantics: the whole pipeline is still scheduled as one unit, and
single operations can be executed against registered datasets. The kernel
(curator) is imported lazily and never modified.

Blocking kernel calls run in a worker thread (asyncio.to_thread) so the
ZMQ message loop stays responsive (doc §18).
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class PipelineExecutor:
    def __init__(self, dataset_registry, publisher=None, runtime_id: str = "") -> None:
        self.datasets = dataset_registry
        self.publisher = publisher
        self.runtime_id = runtime_id

    async def execute_operation(
        self,
        name: str,
        params: Optional[Dict[str, Any]] = None,
        input_dataset_id: Optional[str] = None,
        output_dataset_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute one registered operation against a registered dataset."""
        from curator.core.operation_registry import OperationRegistry

        started = time.time()
        if OperationRegistry.get_operation_class(name) is None:
            raise ValueError(f"Unknown operation: {name}")
        op = OperationRegistry.create_operation(name, params or {})
        input_dataset = None
        if input_dataset_id:
            input_dataset = self.datasets.get(input_dataset_id)

        result = await asyncio.to_thread(op.execute, input_dataset)
        out_id = self.datasets.register(result, dataset_id=output_dataset_id, backend="ray")

        metrics = {"durationMs": int((time.time() - started) * 1000)}
        return {"datasetId": out_id, "operation": name, "metrics": metrics}

    async def execute_pipeline(
        self,
        config: Any,
        is_path: bool = False,
        output_dataset_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run an entire MultimodalPipeline (Phase 1 whole-pipeline mode).

        config: pipeline configuration dict or YAML path.
        """
        from curator.core.pipeline import MultimodalPipeline

        started = time.time()

        def _build_and_run():
            pipeline = MultimodalPipeline(config if not is_path else str(config))
            return pipeline.run()

        result = await asyncio.to_thread(_build_and_run)
        out_id = self.datasets.register(result, dataset_id=output_dataset_id, backend="ray")

        metrics = {"durationMs": int((time.time() - started) * 1000)}
        return {"datasetId": out_id, "metrics": metrics}
