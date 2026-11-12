#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RemoveRepetitionsPunctuationOperation example script

Function:
    Remove repeated punctuation and repeated characters

Usage:
    python examples/curator/single_ops/augmentation/remove_repetitions_punctuation_example.py
"""

import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)); PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR))))
sys.path.insert(0, PROJECT_ROOT)

from curator.core.pipeline import MultimodalPipeline
from curator.operations.loading import MultiModalLoadingOperation
from curator.operations.augmentation import RemoveRepetitionsPunctuationOperation
from curator.operations.saving import TextSavingOperation
from curator.checkpoint.checkpoint_models import CheckpointConfig

os.environ['PYTHONPATH'] = PROJECT_ROOT + os.pathsep + os.environ.get('PYTHONPATH', '')
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'examples', 'curator'))
from common import init_ray, shutdown_ray

init_ray()

_NO_CHECKPOINT = CheckpointConfig(enabled=False)
DATA_FILE = os.path.join(PROJECT_ROOT, "data", "texts", "test_repetitions.txt")


def example_remove_repetitions_punctuation():
    """Remove repeated punctuation and characters example"""
    print("=" * 60)
    print("Example: remove repeated punctuation and characters")
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
    
    # Remove repeated punctuation and characters
    pipeline.add_operation(RemoveRepetitionsPunctuationOperation({
        "text_key": "text",
        "keep_repeat_char": 1
    }))
    
    # Save results
    output_path = os.path.join(PROJECT_ROOT, "data", "output", "remove_repetitions_output.json")
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
    example_remove_repetitions_punctuation()

shutdown_ray()
