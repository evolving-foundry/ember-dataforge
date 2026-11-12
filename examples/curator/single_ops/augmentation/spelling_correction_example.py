#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SpellingCorrectionOperation example script

Function:
    Spelling correction: automatically correct spelling errors in text using a dictionary and edit-distance algorithm

Usage:
    python examples/curator/single_ops/augmentation/spelling_correction_example.py
"""

import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)); PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR))))
sys.path.insert(0, PROJECT_ROOT)

from curator.core.pipeline import MultimodalPipeline
from curator.operations.loading import MultiModalLoadingOperation
from curator.operations.augmentation import SpellingCorrectionOperation
from curator.operations.saving import TextSavingOperation
from curator.checkpoint.checkpoint_models import CheckpointConfig

os.environ['PYTHONPATH'] = PROJECT_ROOT + os.pathsep + os.environ.get('PYTHONPATH', '')
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'examples', 'curator'))
from common import init_ray, shutdown_ray

init_ray()

_NO_CHECKPOINT = CheckpointConfig(enabled=False)
DATA_FILE = os.path.join(PROJECT_ROOT, "data", "texts", "test_spelling.txt")


def example_spelling_correction():
    """Spelling correction example"""
    print("=" * 60)
    print("Example: spelling correction")
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
    
    # Spelling correction
    pipeline.add_operation(SpellingCorrectionOperation({
        "text_key": "text",
        "max_dictionary_edit_distance": 2,
        "max_edit_distance": 2,
        "prefix_length": 7,
        "dictionary_path": None
    }))
    
    # Save results
    output_path = os.path.join(PROJECT_ROOT, "data", "output", "spelling_correction_output.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pipeline.add_operation(TextSavingOperation({
        "output_path": output_path,
        "format": "json"
    }))
    
    result = pipeline.run()
    print(f"Correction complete: {result.dataset.count()} records")
    
    print("\nSample processed data:")
    for i, item in enumerate(result.dataset.iter_rows()):
        print(f"  Record {i+1}: {item}")
        if i >= 2:
            break
    
    pipeline.cleanup()
    return result


if __name__ == "__main__":
    example_spelling_correction()

shutdown_ray()
