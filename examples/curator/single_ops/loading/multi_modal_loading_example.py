#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MultiModalLoadingOperation example script

Function:
    Load multimodal data, automatically detect file types or filter by a specified modality
    Supports text / image / audio / video modalities

Usage:
    python examples/curator/single_ops/loading/multi_modal_loading_example.py
"""

import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)); PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR))))
sys.path.insert(0, PROJECT_ROOT)

from curator.core.pipeline import MultimodalPipeline
from curator.operations.loading import MultiModalLoadingOperation
from curator.operations.saving import MultiModalSavingOperation
from curator.checkpoint.checkpoint_models import CheckpointConfig

os.environ['PYTHONPATH'] = PROJECT_ROOT + os.pathsep + os.environ.get('PYTHONPATH', '')
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'examples', 'curator'))
from common import init_ray, shutdown_ray

init_ray()

_NO_CHECKPOINT = CheckpointConfig(enabled=False)
DATA_FILE = os.path.join(PROJECT_ROOT, "data", "sample.json")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")


def example_multi_modal_loading():
    """Multimodal loading example"""
    print("=" * 60)
    print("Example: multimodal data loading")
    print("=" * 60)
    
    pipeline = MultimodalPipeline(
        enable_tracking=False,
        checkpoint_config=_NO_CHECKPOINT
    )
    
    # Option 1: Automatically detect the modality
    print("\nMethod 1: automatic modality detection")
    pipeline.add_operation(MultiModalLoadingOperation({
        "data_paths": [DATA_FILE]
    }))

    output_path = os.path.join(PROJECT_ROOT, "data", "output", "multimodal_loaded_output.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pipeline.add_operation(MultiModalSavingOperation({
        "output_path": output_path,
        "format": "json"
    }))

    result = pipeline.run()
    print(f"Automatic detection loaded {result.dataset.count()} records")
    
    # Print a sample of the loaded data
    print("\nSample data loaded by method 1:")
    for i, item in enumerate(result.dataset.iter_rows()):
        print(f"  Record {i+1}: {item}")
        if i >= 2:
            break
    
    pipeline.cleanup()
    
    # Option 2: Specify the modality
    print("\nMethod 2: explicitly specify text modality")
    pipeline2 = MultimodalPipeline(
        enable_tracking=False,
        checkpoint_config=_NO_CHECKPOINT
    )
    
    pipeline2.add_operation(MultiModalLoadingOperation({
        "data_paths": [DATA_FILE],
        "modalities": ["text"]
    }))
    
    result2 = pipeline2.run()
    print(f"Specified modality loaded {result2.dataset.count()} records")
    pipeline2.cleanup()
    
    return result


if __name__ == "__main__":
    example_multi_modal_loading()

shutdown_ray()
