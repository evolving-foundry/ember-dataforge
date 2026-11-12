#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ImageRandomFlipOperation example script

Function:
    Random image flip augmentation, supporting horizontal flip, vertical flip, and rotation

Usage:
    python examples/curator/single_ops/augmentation/image_random_flip_example.py
"""

import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)); PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR))))
sys.path.insert(0, PROJECT_ROOT)

from curator.core.pipeline import MultimodalPipeline
from curator.operations.loading import ImageLoadingOperation
from curator.operations.augmentation import ImageRandomFlipOperation
from curator.operations.saving import ImageSavingOperation
from curator.checkpoint.checkpoint_models import CheckpointConfig
from curator.core.operation import Operation
from curator.tracking.tracking import OperationType
os.environ['PYTHONPATH'] = PROJECT_ROOT + os.pathsep + os.environ.get('PYTHONPATH', '')
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'examples', 'curator'))
from common import init_ray, shutdown_ray, display_image_results


init_ray()

_NO_CHECKPOINT = CheckpointConfig(enabled=False)
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "images")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "output", "random_flip")


def example_image_random_flip():
    """Random image flip example"""
    print("=" * 60)
    print("Example: random image flipping")
    print("=" * 60)
    
    pipeline = MultimodalPipeline(
        enable_tracking=False,
        checkpoint_config=_NO_CHECKPOINT
    )
    
    pipeline.add_operation(ImageLoadingOperation({
        "data_paths": [DATA_DIR]
    }))
    
    pipeline.add_operation(ImageRandomFlipOperation({
        "image_key": "image",
        "flip_horizontal_prob": 0.3,
        "flip_vertical_prob": 0.3,
        "rotation_prob": 0.3,
        "num_augmentations": 3,
        "output_key": "flip_image_data_list"
    }))
    
    output_path = os.path.join(OUTPUT_DIR, "random_flip_output.parquet")
    pipeline.add_operation(ImageSavingOperation({
        "output_path": output_path,
        "format": "parquet"
    }))
    
    result = pipeline.run()
    print(f"Augmentation complete: {result.dataset.count()} records")
    
    print("\nSample processed data:")
    for i, item in enumerate(result.dataset.iter_rows()):
        flip_list = item.get('flip_image_data_list', [])
        print(f"  Record {i+1}: generated {len(flip_list)} flipped images")
        if i >= 2:
            break
    
    print("\nFlipped image information:")
    display_image_results(output_path, image_field_name='flip_image_data_list')
    
    pipeline.cleanup()
    return result


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    try:
        example_image_random_flip()
    finally:
        shutdown_ray()
