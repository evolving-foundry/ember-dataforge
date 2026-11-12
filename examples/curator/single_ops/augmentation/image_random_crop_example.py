#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ImageRandomCropOperation example script

Function:
    Random image crop augmentation

Usage:
    python examples/curator/single_ops/augmentation/image_random_crop_example.py
"""

import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)); PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR))))
sys.path.insert(0, PROJECT_ROOT)

from curator.core.pipeline import MultimodalPipeline
from curator.operations.loading import ImageLoadingOperation
from curator.operations.augmentation import ImageRandomCropOperation
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
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "output", "random_crop")


def example_image_random_crop():
    """Random image crop example"""
    print("=" * 60)
    print("Example: random image cropping")
    print("=" * 60)
    
    pipeline = MultimodalPipeline(
        enable_tracking=False,
        checkpoint_config=_NO_CHECKPOINT
    )
    
    pipeline.add_operation(ImageLoadingOperation({
        "data_paths": [DATA_DIR],
        "modalities": ["image"]
    }))
    
    pipeline.add_operation(ImageRandomCropOperation({
        "image_key": "image",
        "crop_height": 224,
        "crop_width": 224,
        "num_crops": 3,
        "output_key": "crop_image_data_list"
    }))
    
    output_path = os.path.join(OUTPUT_DIR, "random_crop_output.parquet")
    pipeline.add_operation(ImageSavingOperation({
        "output_path": output_path,
        "format": "parquet"
    }))
    
    result = pipeline.run()
    total_count = result.dataset.count()
    print(f"Processed {total_count} records")

    print(f"\nFinal statistics:")
    print(f"  Total records processed: {total_count}")
    
    print("\nCropped image information:")
    display_image_results(output_path, image_field_name='crop_image_data_list')
    
    pipeline.cleanup()
    return result


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    try:
        example_image_random_crop()
    finally:
        shutdown_ray()
