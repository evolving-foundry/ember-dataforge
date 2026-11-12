#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ImageCaptioningOperation example script

Function:
    Image caption generation: use an AI model to generate text descriptions for images
    Note: the BLIP model is large and may fail due to insufficient memory

Usage:
    python examples/curator/single_ops/augmentation/image_captioning_example.py
"""

import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)); PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR))))
sys.path.insert(0, PROJECT_ROOT)

from curator.core.pipeline import MultimodalPipeline
from curator.operations.augmentation import ImageCaptioningOperation
from curator.operations.loading import ImageLoadingOperation
from curator.operations.saving import ImageSavingOperation
from curator.checkpoint.checkpoint_models import CheckpointConfig

os.environ['PYTHONPATH'] = PROJECT_ROOT + os.pathsep + os.environ.get('PYTHONPATH', '')
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'examples', 'curator'))
from common import init_ray, shutdown_ray, display_image_results

init_ray()

_NO_CHECKPOINT = CheckpointConfig(enabled=False)
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "images")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "output", "image_captioning")


def example_image_captioning():
    """Image caption generation example (simplified)"""
    print("=" * 60)
    print("Example: image caption generation (simplified)")
    print("=" * 60)
    print("Note: this example skips actual caption generation because the model is too large")
    print("      It only demonstrates image loading, saving, and result display")
    
    pipeline = MultimodalPipeline(
        enable_tracking=False,
        checkpoint_config=_NO_CHECKPOINT
    )
    
    # Load image data
    pipeline.add_operation(ImageLoadingOperation({
        "data_paths": [DATA_DIR],
        "modalities": ["image"]
    }))
    
    # Generate image captions
    pipeline.add_operation(ImageCaptioningOperation({
        "image_key": "image_path",
        "output_key": "caption",
        "model_path": "Salesforce/blip-image-captioning-base",
        "device_preference": "auto",
        "prompt_text": "a photography of"
    }))

    # Save results
    output_path = os.path.join(OUTPUT_DIR, "caption_output.parquet")
    pipeline.add_operation(ImageSavingOperation({
        "output_path": output_path,
        "format": "parquet"
    }))

    result = pipeline.run()
    print(f"Processing complete: {result.dataset.count()} records")
    
    # Display input image information
    print("\nInput image information:")
    display_image_results(output_path, image_field_name='image')
    
    pipeline.cleanup()
    return result


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    try:
        example_image_captioning()
    finally:
        shutdown_ray()
