#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TextLoadingOperation example script

Function:
    Load data from text files, supporting .txt / .json / .jsonl / .csv formats

Usage:
    python examples/curator/single_ops/loading/text_loading_example.py
"""

import sys
import os

# Get the project root directory (the script is under examples/curator/single_ops/loading/)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR))))
sys.path.insert(0, PROJECT_ROOT)

from curator.core.pipeline import MultimodalPipeline
from curator.operations.loading import TextLoadingOperation
from curator.operations.saving import TextSavingOperation
from curator.checkpoint.checkpoint_models import CheckpointConfig

os.environ['PYTHONPATH'] = PROJECT_ROOT + os.pathsep + os.environ.get('PYTHONPATH', '')
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'examples', 'curator'))
from common import init_ray, shutdown_ray

init_ray()

_NO_CHECKPOINT = CheckpointConfig(enabled=False)
DATA_FILE = os.path.join(PROJECT_ROOT, "data", "sample.json")


def example_text_loading():
    """Text loading example"""
    print("=" * 60)
    print("Example: text data loading")
    print("=" * 60)
    
    pipeline = MultimodalPipeline(
        enable_tracking=False,
        checkpoint_config=_NO_CHECKPOINT
    )
    
    pipeline.add_operation(TextLoadingOperation({
        "data_paths": [DATA_FILE]
    }))

    output_path = os.path.join(PROJECT_ROOT, "data", "output", "text_loaded_output.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pipeline.add_operation(TextSavingOperation({
        "output_path": output_path,
        "format": "json"
    }))

    result = pipeline.run()
    print(f"Loaded {result.dataset.count()} records")
    
    # Print a sample of the loaded data
    print("\nSample loaded data:")
    for i, item in enumerate(result.dataset.iter_rows()):
        print(f"  Record {i+1}: {item}")
        if i >= 2:
            break
    
    pipeline.cleanup()
    return result


if __name__ == "__main__":
    example_text_loading()

shutdown_ray()
