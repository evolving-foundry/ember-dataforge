#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
StemmingLemmatizationOperation example script

Function:
    Stemming and lemmatization, supporting multiple modes

Usage:
    python examples/curator/single_ops/augmentation/stemming_lemmatization_example.py
"""

import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)); PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR))))
sys.path.insert(0, PROJECT_ROOT)

from curator.core.pipeline import MultimodalPipeline
from curator.operations.loading import MultiModalLoadingOperation
from curator.operations.augmentation import StemmingLemmatizationOperation
from curator.operations.saving import TextSavingOperation
from curator.checkpoint.checkpoint_models import CheckpointConfig

os.environ['PYTHONPATH'] = PROJECT_ROOT + os.pathsep + os.environ.get('PYTHONPATH', '')
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'examples', 'curator'))
from common import init_ray, shutdown_ray

init_ray()

_NO_CHECKPOINT = CheckpointConfig(enabled=False)
DATA_FILE = os.path.join(PROJECT_ROOT, "data", "texts", "test_stemming.txt")


def example_stemming_lemmatization():
    """Stemming and lemmatization example"""
    print("=" * 60)
    print("Example: stemming and lemmatization")
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
    
    # Stemming and lemmatization
    pipeline.add_operation(StemmingLemmatizationOperation({
        "text_key": "text",
        "mode": "stem"  # Optional: stem, lemma
    }))
    
    # Save results
    output_path = os.path.join(PROJECT_ROOT, "data", "output", "stemming_output.json")
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
    example_stemming_lemmatization()

shutdown_ray()
