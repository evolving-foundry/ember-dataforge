#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FusionOperation example script

Function:
    Fusion operation: merge multiple Operation instances into one atomic operation to reduce intermediate transfer overhead
    Note: Operation instances must be passed through the Python API; plain string configuration is not supported

Usage:
    python examples/curator/single_ops/common/fusion_example.py
"""

import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)); PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR))))
sys.path.insert(0, PROJECT_ROOT)

from curator.core.pipeline import MultimodalPipeline
from curator.operations.loading import MultiModalLoadingOperation
from curator.operations.filtering import QualityBasedFilterOperation
from curator.operations.augmentation import TextNormalizationOperation
from curator.operations.common import FusionOperation
from curator.operations.saving import TextSavingOperation
from curator.checkpoint.checkpoint_models import CheckpointConfig

os.environ['PYTHONPATH'] = PROJECT_ROOT + os.pathsep + os.environ.get('PYTHONPATH', '')
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'examples', 'curator'))
from common import init_ray, shutdown_ray

init_ray()

_NO_CHECKPOINT = CheckpointConfig(enabled=False)
DATA_FILE = os.path.join(PROJECT_ROOT, "data", "test_fusion.json")


def example_fusion():
    """Fusion operation example"""
    print("=" * 60)
    print("Example: fusion operation")
    print("=" * 60)
    
    # Create the list of operations to fuse
    operations = [
        QualityBasedFilterOperation({
            "quality_threshold": 0.5
        }),
        TextNormalizationOperation({
            "text_key": "text"
        })
    ]
    
    pipeline = MultimodalPipeline(
        enable_tracking=False,
        checkpoint_config=_NO_CHECKPOINT
    )
    
    # Load data
    pipeline.add_operation(MultiModalLoadingOperation({
        "data_paths": [DATA_FILE],
        "modalities": ["text"]
    }))
    
    # Fusion operation
    pipeline.add_operation(FusionOperation(operations, {}))

    output_path = os.path.join(PROJECT_ROOT, "data", "output", "fusion_output.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pipeline.add_operation(TextSavingOperation({
        "output_path": output_path,
        "format": "json"
    }))

    result = pipeline.run()
    print(f"Fusion processing complete: {result.dataset.count()} records")
    
    print("\nSample fused data:")
    for i, item in enumerate(result.dataset.iter_rows()):
        print(f"  Record {i+1}: {item}")
        if i >= 2:
            break
    
    pipeline.cleanup()
    return result


if __name__ == "__main__":
    example_fusion()

shutdown_ray()
