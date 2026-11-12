#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
VideoSavingOperation example script

Function:
    Save a video dataset to the specified path

Usage:
    python examples/curator/single_ops/saving/video_saving_example.py
"""

import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)); PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR))))
sys.path.insert(0, PROJECT_ROOT)

from curator.core.pipeline import MultimodalPipeline
from curator.operations.loading import VideoLoadingOperation
from curator.operations.saving import VideoSavingOperation
from curator.checkpoint.checkpoint_models import CheckpointConfig

os.environ['PYTHONPATH'] = PROJECT_ROOT + os.pathsep + os.environ.get('PYTHONPATH', '')
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'examples', 'curator'))
from common import init_ray, shutdown_ray, display_video_results

init_ray()

_NO_CHECKPOINT = CheckpointConfig(enabled=False)
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "videos")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "output", "video_saving")


def example_video_saving():
    """Video saving example"""
    print("=" * 60)
    print("Example: video data saving")
    print("=" * 60)
    
    if not os.path.exists(DATA_DIR):
        print(f"Error: data directory {DATA_DIR} does not exist")
        return None

    video_files = [f for f in os.listdir(DATA_DIR) if
                   f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm'))]
    print(f"Found {len(video_files)} video files: {video_files}")

    if not video_files:
        print("Warning: no video files found")
        return None

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output directory created: {OUTPUT_DIR}")
    
    pipeline = MultimodalPipeline(
        enable_tracking=False,
        checkpoint_config=_NO_CHECKPOINT
    )
    
    # Step 1: Load video data
    pipeline.add_operation(VideoLoadingOperation({
        "data_paths": [DATA_DIR]
    }))
    
    # Step 2: Save data
    output_path = os.path.join(OUTPUT_DIR, "video_output.parquet")
    pipeline.add_operation(VideoSavingOperation({
        "output_path": output_path,
        "format": "parquet"
    }))
    
    result = pipeline.run()
    print(f"Saved {result.dataset.count()} records")
    
    # Display the saved video results
    display_video_results(output_path, video_field_name='path')
    
    pipeline.cleanup()
    return result


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    example_video_saving()

shutdown_ray()
