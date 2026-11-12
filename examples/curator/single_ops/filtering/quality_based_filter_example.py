#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
QualityBasedFilterOperation example script

Function:
    Filter data based on a quality score field, keeping records whose score is not below the threshold

Usage:
    python examples/curator/single_ops/filtering/quality_based_filter_example.py
"""

import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)); PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR))))
sys.path.insert(0, PROJECT_ROOT)

from curator.core.pipeline import MultimodalPipeline
from curator.operations.loading import MultiModalLoadingOperation
from curator.operations.filtering import QualityBasedFilterOperation
from curator.operations.saving import MultiModalSavingOperation
from curator.checkpoint.checkpoint_models import CheckpointConfig

os.environ['PYTHONPATH'] = PROJECT_ROOT + os.pathsep + os.environ.get('PYTHONPATH', '')
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'examples', 'curator'))
from common import init_ray, shutdown_ray

init_ray()

_NO_CHECKPOINT = CheckpointConfig(enabled=False)
DATA_FILE = os.path.join(PROJECT_ROOT, "data", "multimodal_sample.json")


def example_quality_filter():
    """Quality score filtering example"""
    print("=" * 60)
    print("Example: quality score filtering")
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
    
    # Quality filtering
    pipeline.add_operation(QualityBasedFilterOperation({
        "quality_threshold": 0.9,
        "quality_column": "quality"
    }))

    # Save results
    output_path = os.path.join(PROJECT_ROOT, "data", "output", "quality_filter_output.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pipeline.add_operation(MultiModalSavingOperation({
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
    example_quality_filter()

shutdown_ray()
