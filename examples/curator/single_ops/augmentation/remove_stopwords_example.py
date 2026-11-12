#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RemoveStopwordsOperation example script

Function:
    Remove stopwords, with support for a custom stopword list

Usage:
    python examples/curator/single_ops/augmentation/remove_stopwords_example.py
"""

import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)); PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR))))
sys.path.insert(0, PROJECT_ROOT)

from curator.core.pipeline import MultimodalPipeline
from curator.operations.loading import MultiModalLoadingOperation
from curator.operations.augmentation import RemoveStopwordsOperation
from curator.operations.saving import TextSavingOperation
from curator.checkpoint.checkpoint_models import CheckpointConfig

os.environ['PYTHONPATH'] = PROJECT_ROOT + os.pathsep + os.environ.get('PYTHONPATH', '')
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'examples', 'curator'))
from common import init_ray, shutdown_ray

init_ray()

_NO_CHECKPOINT = CheckpointConfig(enabled=False)
DATA_FILE = os.path.join(PROJECT_ROOT, "data", "texts", "sample.txt")


def example_remove_stopwords():
    """Remove stopwords example"""
    print("=" * 60)
    print("Example: remove stopwords")
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
    
    # Remove stopwords
    pipeline.add_operation(RemoveStopwordsOperation({
        "text_key": "text",
        "custom_stopwords": []  # Custom stopword list
    }))
    
    # Save results
    output_path = os.path.join(PROJECT_ROOT, "data", "output", "remove_stopwords_output.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pipeline.add_operation(TextSavingOperation({
        "output_path": output_path,
        "format": "json"
    }))
    
    result = pipeline.run()
    print(f"Processing complete: {result.dataset.count()} records")
    
    print("\nSample processed data:")
    for i, item in enumerate(result.dataset.iter_rows()):
        print(f"  Record {i+1}: {item}")
        if i >= 2:
            break
    
    pipeline.cleanup()
    return result


if __name__ == "__main__":
    example_remove_stopwords()

shutdown_ray()
