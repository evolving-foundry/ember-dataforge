#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
BlocklistFilterOperation example script

Function:
    Filter text samples based on blocklist terms, supporting both substring matching and token matching modes

Usage:
    python examples/curator/single_ops/filtering/blocklist_filter_example.py
"""

import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)); PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR))))
sys.path.insert(0, PROJECT_ROOT)

from curator.core.pipeline import MultimodalPipeline
from curator.operations.loading import MultiModalLoadingOperation
from curator.operations.filtering import BlocklistFilterOperation
from curator.operations.saving import TextSavingOperation
from curator.checkpoint.checkpoint_models import CheckpointConfig

os.environ['PYTHONPATH'] = PROJECT_ROOT + os.pathsep + os.environ.get('PYTHONPATH', '')
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'examples', 'curator'))
from common import init_ray, shutdown_ray

init_ray()

_NO_CHECKPOINT = CheckpointConfig(enabled=False)
DATA_FILE = os.path.join(PROJECT_ROOT, "data", "blocklist_test_data.json")


def example_blocklist_filter():
    """Blocklist filtering example"""
    print("=" * 60)
    print("Example: blocklist filtering")
    print("=" * 60)
    
    pipeline = MultimodalPipeline(
        enable_tracking=False,
        checkpoint_config=_NO_CHECKPOINT
    )
    
    # Load data
    pipeline.add_operation(MultiModalLoadingOperation({
        "data_paths": [DATA_FILE],
        "modalities": ["text"]
    }))
    
    # Blocklist filtering
    pipeline.add_operation(BlocklistFilterOperation({
        "blocklist": ["spam", "ad", "promotion"],
        "max_hits": 0,
        "match_mode": "substring",
        "case_insensitive": True,
        "text_key": "text"
    }))

    # Save results
    output_path = os.path.join(PROJECT_ROOT, "data", "output", "blocklist_filter_output.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pipeline.add_operation(TextSavingOperation({
        "output_path": output_path,
        "format": "json"
    }))

    result = pipeline.run()
    print(f"Filtering complete; remaining {result.dataset.count()} records")
    
    print("\nSample filtered data:")
    for i, item in enumerate(result.dataset.iter_rows()):
        print(f"  Record {i+1}: {item}")
        if i >= 2:
            break
    
    pipeline.cleanup()
    return result


if __name__ == "__main__":
    example_blocklist_filter()

shutdown_ray()
