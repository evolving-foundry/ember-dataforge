#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MultiModalSavingOperation example script

Function:
    Save a multimodal dataset to the specified path, supporting json / parquet / csv formats

Usage:
    python examples/curator/single_ops/saving/multi_modal_saving_example.py
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
from common import init_ray, shutdown_ray, display_image_results, display_audio_results, display_video_results, display_text_results

init_ray()

_NO_CHECKPOINT = CheckpointConfig(enabled=False)
DATA_FILE = os.path.join(PROJECT_ROOT, "data", "multimodal_sample.json")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "output", "multimodal_saving")


def example_multi_modal_saving():
    """Multimodal saving example"""
    print("=" * 60)
    print("Example: multimodal data saving")
    print("=" * 60)
    
    if not os.path.exists(DATA_FILE):
        print(f"Error: data file {DATA_FILE} does not exist")
        return None

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output directory created: {OUTPUT_DIR}")
    
    pipeline = MultimodalPipeline(
        enable_tracking=False,
        checkpoint_config=_NO_CHECKPOINT
    )
    
    # Step 1: Load multimodal data
    pipeline.add_operation(MultiModalLoadingOperation({
        "data_paths": [DATA_FILE]
    }))
    
    # Step 2: Save data
    output_path = os.path.join(OUTPUT_DIR, "multimodal_output.parquet")
    pipeline.add_operation(MultiModalSavingOperation({
        "output_path": output_path,
        "format": "parquet"
    }))
    
    result = pipeline.run()
    print(f"Saved {result.dataset.count()} records")
    
    # Display the saved multimodal results
    print("\n" + "=" * 60)
    print("Multimodal data saving result")
    print("=" * 60)
    
    # Try to display image results
    try:
        display_image_results(output_path, image_field_name='image', max_files=2, save_images=False)
    except Exception as e:
        print(f"  (unable to display image: {e})")
    
    # Try to display audio results
    try:
        display_audio_results(output_path, audio_field_name='audio', max_files=2, save_audio=False)
    except Exception as e:
        print(f"  (unable to display audio: {e})")
    
    # Try to display video results
    try:
        display_video_results(output_path, video_field_name='video', max_files=2, save_videos=False)
    except Exception as e:
        print(f"  (unable to display video: {e})")
    
    # Try to display text results
    try:
        display_text_results(output_path, text_field_name='text', max_files=2)
    except Exception as e:
        print(f"  (unable to display text: {e})")
    
    pipeline.cleanup()
    return result


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    example_multi_modal_saving()

shutdown_ray()
