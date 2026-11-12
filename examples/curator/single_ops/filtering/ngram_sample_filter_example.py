#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
NgramSampleFilterOperation example script

Function:
    Filter text based on N-gram repetition rate, used to detect templated or repetitive content

Usage:
    python examples/curator/single_ops/filtering/ngram_sample_filter_example.py
"""

import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)); PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR))))
sys.path.insert(0, PROJECT_ROOT)

from curator.core.pipeline import MultimodalPipeline
from curator.operations.loading import MultiModalLoadingOperation
from curator.operations.filtering import NgramSampleFilterOperation
from curator.operations.saving import TextSavingOperation
from curator.checkpoint.checkpoint_models import CheckpointConfig

os.environ['PYTHONPATH'] = PROJECT_ROOT + os.pathsep + os.environ.get('PYTHONPATH', '')
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'examples', 'curator'))
from common import init_ray, shutdown_ray

init_ray()

_NO_CHECKPOINT = CheckpointConfig(enabled=False)
DATA_FILE = os.path.join(PROJECT_ROOT, "data", "ngram_test_data.json")


def example_ngram_filter():
    """N-gram repetition rate filtering example"""
    print("=" * 60)
    print("Example: N-gram repetition filtering")
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
    
    # N-gram filtering
    pipeline.add_operation(NgramSampleFilterOperation({
        "text_key": "input_text",
        "n_gram": 2,
        "threshold": 0.5,
        "lang": "en"
    }))

    output_path = os.path.join(PROJECT_ROOT, "data", "output", "ngram_filter_output.parquet")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pipeline.add_operation(TextSavingOperation({
        "output_path": output_path,
        "format": "parquet"
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
    example_ngram_filter()

shutdown_ray()
