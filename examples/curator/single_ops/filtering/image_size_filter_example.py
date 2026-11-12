#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ImageSizeFilterOperation example script

Function:
    Filter images by image file size, supporting multiple units (bytes, kb, mb, gb, tb)

Usage:
    python examples/curator/single_ops/filtering/image_size_filter_example.py
"""

import sys
import os
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR))))
sys.path.insert(0, PROJECT_ROOT)

from curator.core.pipeline import MultimodalPipeline
from curator.operations.loading import ImageLoadingOperation
from curator.operations.filtering import ImageSizeFilterOperation
from curator.operations.saving import ImageSavingOperation
from curator.checkpoint.checkpoint_models import CheckpointConfig

os.environ['PYTHONPATH'] = PROJECT_ROOT + os.pathsep + os.environ.get('PYTHONPATH', '')

sys.path.insert(0, os.path.join(PROJECT_ROOT, 'examples', 'curator'))
from common import init_ray, shutdown_ray, display_image_results

init_ray()

_NO_CHECKPOINT = CheckpointConfig(enabled=False)
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "images")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "output", "image_size")


def example_image_size_filter():
    """Image file size filtering example"""
    print("=" * 60)
    print("Example: image file size filtering")
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

    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        print(f"Output directory created: {OUTPUT_DIR}")

        pipeline = MultimodalPipeline(
            enable_tracking=False,
            checkpoint_config=_NO_CHECKPOINT
        )
        
        pipeline.add_operation(ImageLoadingOperation({
            "data_paths": [DATA_DIR]
        }))
        
        pipeline.add_operation(ImageSizeFilterOperation({
            "image_key": "image_path",
            "min_size": 50,
            "max_size": 80,
            "size_unit": "kb",
            "any_or_all": "any"
        }))

        output_path = os.path.join(OUTPUT_DIR, "image_size_filter_output.parquet")
        pipeline.add_operation(ImageSavingOperation({
            "output_path": output_path,
            "format": "parquet"
        }))

        run_start_time = time.time()
        result = pipeline.run()
        run_time = time.time() - run_start_time

        total_count = result.dataset.count()
        print(f"Filtering complete (elapsed: {run_time:.2f}s), remaining {total_count} records")
        
        display_image_results(output_path, image_field_name='image')
        
        pipeline.cleanup()
        return result

    except Exception as e:
        print(f"\nError during execution: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        shutdown_ray()


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    example_image_size_filter()
