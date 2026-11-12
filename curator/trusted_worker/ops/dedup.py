"""Exact deduplication op: shared core (used by both evaluation mode and op mode).

Reuse ``ExactDuplicateRemovalOperation.execute`` (Ray execution; output order is frozen by the operator
— Evaluator §4.4-3 determinism requirement) without rewriting the deduplication logic.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

from curator.operations.deduplication import ExactDuplicateRemovalOperation


def execute_exact_dedup(rows: List[Dict[str, Any]]) -> Tuple[int, Any]:
    """rows → MultimodalDataset → exact deduplication; return ``(input_count, deduped_dataset)``."""
    import ray

    from base.dataset import MultimodalDataset

    dataset = MultimodalDataset(dataset=ray.data.from_items(rows))
    deduped = ExactDuplicateRemovalOperation().execute(dataset)
    return len(rows), deduped
