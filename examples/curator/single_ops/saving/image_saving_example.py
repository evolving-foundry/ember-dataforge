#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ImageSavingOperation example script

Function:
    Save an image dataset to the specified path, using parquet format by default

Usage:
    python examples/curator/single_ops/saving/image_saving_example.py
"""

import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)); PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR))))
sys.path.insert(0, PROJECT_ROOT)

from curator.core.pipeline import MultimodalPipeline
from curator.operations.loading import ImageLoadingOperation
from curator.operations.saving import ImageSavingOperation
from curator.checkpoint.checkpoint_models import CheckpointConfig

os.environ['PYTHONPATH'] = PROJECT_ROOT + os.pathsep + os.environ.get('PYTHONPATH', '')
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'examples', 'curator'))
from common import init_ray, shutdown_ray, display_image_results

init_ray()

_NO_CHECKPOINT = CheckpointConfig(enabled=False)
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "images")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "output", "image_saving")


def example_image_saving():
    """Image saving example"""
    print("=" * 60)
    print("Example: image data saving")
    print("=" * 60)
    
    if not os.path.exists(DATA_DIR):
        print(f"Error: data directory {DATA_DIR} does not exist")
        return None

    image_files = [f for f in os.listdir(DATA_DIR) if
                   f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp', '.gif'))]
    print(f"Found {len(image_files)} image files: {image_files}")

    if not image_files:
        print("Warning: no image files found")
        return None

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output directory created: {OUTPUT_DIR}")
    
    pipeline = MultimodalPipeline(
        enable_tracking=False,
        checkpoint_config=_NO_CHECKPOINT
    )
    
    # Step 1: Load image data
    pipeline.add_operation(ImageLoadingOperation({
        "data_paths": [DATA_DIR]
    }))
    
    # Step 2: Save data
    output_path = os.path.join(OUTPUT_DIR, "image_output.parquet")
    pipeline.add_operation(ImageSavingOperation({
        "output_path": output_path,
        "format": "parquet"
    }))
    
    result = pipeline.run()
    print(f"Saved {result.dataset.count()} records")
    
    # Display the saved image results
    display_image_results(output_path, image_field_name='image')
    
    pipeline.cleanup()
    return result


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    example_image_saving()

shutdown_ray()
