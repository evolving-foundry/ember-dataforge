#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
BleuSampleFilterOperation example script

Function:
    Filter low-similarity text based on BLEU scores, used to evaluate the accuracy and fluency of translation or text generation

Usage:
    python examples/curator/single_ops/filtering/bleu_sample_filter_example.py
"""

import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)); PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR))))
sys.path.insert(0, PROJECT_ROOT)

from curator.core.pipeline import MultimodalPipeline
from curator.operations.loading import MultiModalLoadingOperation
from curator.operations.filtering import BleuSampleFilterOperation
from curator.operations.saving import TextSavingOperation
from curator.checkpoint.checkpoint_models import CheckpointConfig

os.environ['PYTHONPATH'] = PROJECT_ROOT + os.pathsep + os.environ.get('PYTHONPATH', '')
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'examples', 'curator'))
from common import init_ray, shutdown_ray

init_ray()

_NO_CHECKPOINT = CheckpointConfig(enabled=False)
DATA_FILE = os.path.join(PROJECT_ROOT, "data", "bleu_test_data.json")


def example_bleu_filter():
    """BLEU score filtering example"""
    print("=" * 60)
    print("Example: BLEU score filtering")
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
    
    # BLEU filtering
    pipeline.add_operation(BleuSampleFilterOperation({
        "n_grams": 1,
        "reference_key": "reference_text",
        "text_key": "input_text",
        "threshold": 0.9
    }))

    # Save results
    output_path = os.path.join(PROJECT_ROOT, "data", "output", "bleu_filter_output.json")
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
    example_bleu_filter()

shutdown_ray()
