#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ImageLoadingOperation example script

Function:
    Load image data from local paths, supporting .jpg / .png / .bmp / .tiff / .webp / .gif / .svg formats

Usage:
    python examples/curator/single_ops/loading/image_loading_example.py
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
from common import init_ray, shutdown_ray

init_ray()

_NO_CHECKPOINT = CheckpointConfig(enabled=False)
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "images")


def example_image_loading():
    """Image loading example"""
    print("=" * 60)
    print("Example: image data loading")
    print("=" * 60)
    
    pipeline = MultimodalPipeline(
        enable_tracking=False,
        checkpoint_config=_NO_CHECKPOINT
    )
    
    pipeline.add_operation(ImageLoadingOperation({
        "data_paths": [DATA_DIR]
    }))

    output_path = os.path.join(PROJECT_ROOT, "data", "output", "image_loaded_output.parquet")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pipeline.add_operation(ImageSavingOperation({
        "output_path": output_path,
        "format": "parquet"
    }))

    result = pipeline.run()
    print(f"Loaded {result.dataset.count()} records")
    
    # Print a sample of the loaded data
    print("\nSample loaded data:")
    for i, item in enumerate(result.dataset.iter_rows()):
        item_size = {k: f"{len(v)} bytes" if isinstance(v, (bytes, bytearray)) else v for k, v in item.items()}
        print(f"  Record {i+1}: {item_size}")
        if i >= 2:
            break
    
    pipeline.cleanup()
    return result


if __name__ == "__main__":
    try:
        example_image_loading()
    finally:
        shutdown_ray()
