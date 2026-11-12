#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SpecialCharFilterOperation example script

Function:
    Filter text samples containing specified special characters, commonly used to remove text containing HTML tags, path separators, and similar special characters

Usage:
    python examples/curator/single_ops/filtering/special_char_filter_example.py
"""

import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)); PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR))))
sys.path.insert(0, PROJECT_ROOT)

from curator.core.pipeline import MultimodalPipeline
from curator.operations.loading import MultiModalLoadingOperation
from curator.operations.filtering import SpecialCharFilterOperation
from curator.operations.saving import TextSavingOperation
from curator.checkpoint.checkpoint_models import CheckpointConfig

os.environ['PYTHONPATH'] = PROJECT_ROOT + os.pathsep + os.environ.get('PYTHONPATH', '')
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'examples', 'curator'))
from common import init_ray, shutdown_ray

init_ray()

_NO_CHECKPOINT = CheckpointConfig(enabled=False)
DATA_FILE = os.path.join(PROJECT_ROOT, "data", "special_char_test_data.json")


def example_special_char_filter():
    """Special character filtering example"""
    print("=" * 60)
    print("Example: special character filtering")
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
    
    # Special character filtering
    pipeline.add_operation(SpecialCharFilterOperation({
        "special_chars": ["<", ">", "`", "/"],
        "text_key": "text"
    }))

    # Save results
    output_path = os.path.join(PROJECT_ROOT, "data", "output", "special_char_filter_output.json")
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
    example_special_char_filter()

shutdown_ray()
