#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ExactDuplicateRemovalOperation example script

Function:
    Exact deduplication: remove completely identical data records (calls Ray Dataset.distinct())

Usage:
    python examples/curator/single_ops/deduplication/exact_duplicate_removal_example.py
"""

import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)); PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR))))
sys.path.insert(0, PROJECT_ROOT)

from curator.core.pipeline import MultimodalPipeline
from curator.operations.loading import MultiModalLoadingOperation
from curator.operations.deduplication import ExactDuplicateRemovalOperation
from curator.operations.saving import MultiModalSavingOperation
from curator.checkpoint.checkpoint_models import CheckpointConfig

os.environ['PYTHONPATH'] = PROJECT_ROOT + os.pathsep + os.environ.get('PYTHONPATH', '')
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'examples', 'curator'))
from common import init_ray, shutdown_ray

init_ray()

_NO_CHECKPOINT = CheckpointConfig(enabled=False)
DATA_FILE = os.path.join(PROJECT_ROOT, "data", "texts", "test_duplicate.txt")


def example_exact_duplicate_removal():
    """Exact deduplication example"""
    print("=" * 60)
    print("Example: exact deduplication")
    print("=" * 60)
    
    pipeline = MultimodalPipeline(
        enable_tracking=False,
        checkpoint_config=_NO_CHECKPOINT
    )
    
    # Load data
    pipeline.add_operation(MultiModalLoadingOperation({
        "data_paths": [DATA_FILE],
        "modalities": ["text", "image"]
    }))
    
    # Exact deduplication
    pipeline.add_operation(ExactDuplicateRemovalOperation({}))

    output_path = os.path.join(PROJECT_ROOT, "data", "output", "dedup_output.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pipeline.add_operation(MultiModalSavingOperation({
        "output_path": output_path,
        "format": "json"
    }))

    result = pipeline.run()
    print(f"Deduplicated {result.dataset.count()} records")
    
    print("\nSample deduplicated data:")
    for i, item in enumerate(result.dataset.iter_rows()):
        print(f"  Record {i+1}: {item}")
        if i >= 2:
            break
    
    pipeline.cleanup()
    return result


if __name__ == "__main__":
    example_exact_duplicate_removal()

shutdown_ray()
