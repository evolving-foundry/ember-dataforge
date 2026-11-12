#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MultimodalValidationMappingOperation example script

Function:
    Multimodal validation mapping: validate the quality and matching degree of image-text pairs

Usage:
    python examples/curator/single_ops/mapping/multimodal_validation_mapping_example.py
"""

import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)); PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR))))
sys.path.insert(0, PROJECT_ROOT)

from curator.core.pipeline import MultimodalPipeline
from curator.operations.loading import MultiModalLoadingOperation
from curator.operations.mapping import MultimodalValidationMappingOperation
from curator.operations.saving import MultiModalSavingOperation
from curator.checkpoint.checkpoint_models import CheckpointConfig

os.environ['PYTHONPATH'] = PROJECT_ROOT + os.pathsep + os.environ.get('PYTHONPATH', '')
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'examples', 'curator'))
from common import init_ray, shutdown_ray

init_ray()

_NO_CHECKPOINT = CheckpointConfig(enabled=False)
DATA_FILE = os.path.join(PROJECT_ROOT, "data", "multimodal_validation_test.json")


def example_multimodal_validation_mapping():
    """Multimodal validation mapping example"""
    print("=" * 60)
    print("Example: multimodal validation mapping")
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
    
    # Multimodal validation mapping
    pipeline.add_operation(MultimodalValidationMappingOperation({
        "image_key": "image_path",
        "text_key": "text",
        "validation_rules": {
            "min_text_length": 10,
            "max_text_length": 1000,
            "require_image": True
        }
    }))

    output_path = os.path.join(PROJECT_ROOT, "data", "output", "multimodal_validation_output.parquet")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pipeline.add_operation(MultiModalSavingOperation({
        "output_path": output_path,
        "format": "parquet"
    }))

    result = pipeline.run()
    print(f"Validated {result.dataset.count()} records")
    
    print("\nSample validated data:")
    for i, item in enumerate(result.dataset.iter_rows()):
        print(f"  Record {i+1}: {item}")
        if i >= 2:
            break
    
    pipeline.cleanup()
    return result


if __name__ == "__main__":
    example_multimodal_validation_mapping()

shutdown_ray()
