#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ImageCannyAugmentationOperation example script

Function:
    Canny edge detection: extract image edges

Usage:
    python examples/curator/single_ops/augmentation/image_canny_example.py
"""

import sys
import os
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR))))
sys.path.insert(0, PROJECT_ROOT)

from curator.core.pipeline import MultimodalPipeline
from curator.operations.loading import ImageLoadingOperation
from curator.operations.augmentation import ImageCannyAugmentationOperation
from curator.operations.saving import ImageSavingOperation
from curator.checkpoint.checkpoint_models import CheckpointConfig

os.environ['PYTHONPATH'] = PROJECT_ROOT + os.pathsep + os.environ.get('PYTHONPATH', '')

# Import common functions
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'examples', 'curator'))
from common import init_ray, shutdown_ray, display_image_results

init_ray()

_NO_CHECKPOINT = CheckpointConfig(enabled=False)
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "images")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "output", "canny")


def example_image_canny():
    """Canny edge detection and result saving example"""
    print("=" * 60)
    print("Example: Canny edge detection")
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

        print("\nCreating pipeline...")
        pipeline = MultimodalPipeline(
            enable_tracking=False,
            checkpoint_config=_NO_CHECKPOINT
        )
        print("Pipeline created successfully")

        print("\nAdding image loading operation...")
        loading_config = {
            "data_paths": [DATA_DIR],
            "modalities": ["image"]
        }
        print(f"Load configuration: {loading_config}")

        pipeline.add_operation(ImageLoadingOperation(loading_config))
        print("Image loading operation added")

        print("\nAdding Canny edge detection operation...")
        canny_config = {
            "low_threshold": 50,
            "high_threshold": 150
        }
        pipeline.add_operation(ImageCannyAugmentationOperation(canny_config))
        print("Canny edge detection operation added")

        output_path = os.path.join(OUTPUT_DIR, "canny_output.parquet")
        pipeline.add_operation(ImageSavingOperation({
            "output_path": output_path,
            "format": "parquet"
        }))

        print("\nRunning pipeline...")
        run_start_time = time.time()
        result = pipeline.run()
        run_time = time.time() - run_start_time

        total_count = result.dataset.count()
        print(f"Pipeline run complete (elapsed: {run_time:.2f}s)")
        print(f"Processed {total_count} records")

        print(f"\nFinal statistics:")
        print(f"  Total records processed: {total_count}")

        # Display image results using common functions
        display_image_results(output_path, image_field_name='canny_image_data')

        pipeline.cleanup()
        return total_count

    except Exception as e:
        print(f"\nError during execution: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        shutdown_ray()


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    example_image_canny()
