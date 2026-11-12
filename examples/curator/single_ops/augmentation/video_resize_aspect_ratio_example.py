#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
VideoResizeAspectRatioOperation example script

Function:
    Adjust the video aspect ratio, resizing the video into the target aspect ratio range

Usage:
    python examples/curator/single_ops/augmentation/video_resize_aspect_ratio_example.py
"""

import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)); PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR))))
sys.path.insert(0, PROJECT_ROOT)

from curator.core.pipeline import MultimodalPipeline
from curator.operations.loading import MultiModalLoadingOperation
from curator.operations.augmentation import VideoResizeAspectRatioOperation
from curator.operations.saving import VideoSavingOperation
from curator.checkpoint.checkpoint_models import CheckpointConfig

os.environ['PYTHONPATH'] = PROJECT_ROOT + os.pathsep + os.environ.get('PYTHONPATH', '')
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'examples', 'curator'))
from common import init_ray, shutdown_ray, display_video_results

init_ray()

_NO_CHECKPOINT = CheckpointConfig(enabled=False)
DATA_FILE = os.path.join(PROJECT_ROOT, "data", "videos", "Megamind.avi")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "output", "videos_resize")


def example_video_resize_aspect_ratio():
    """Video aspect ratio adjustment example"""
    print("=" * 60)
    print("Example: video aspect-ratio resizing")
    print("=" * 60)
    
    pipeline = MultimodalPipeline(
        enable_tracking=False,
        checkpoint_config=_NO_CHECKPOINT
    )
    
    # Load data
    pipeline.add_operation(MultiModalLoadingOperation({
        "data_paths": [DATA_FILE],
        "modalities": ["video"]
    }))
    
    # Video aspect ratio adjustment
    pipeline.add_operation(VideoResizeAspectRatioOperation({
        "video_key": "video_path",
        "output_key": "resized_video_path",
        "output_folder": OUTPUT_DIR,
        "target_aspect_ratio_range": [0.5, 2.0]
    }))
    
    # Save results
    output_path = os.path.join(OUTPUT_DIR, "video_resize_output.parquet")
    pipeline.add_operation(VideoSavingOperation({
        "output_path": output_path,
        "format": "parquet"
    }))
    
    result = pipeline.run()
    print(f"Resize complete: {result.dataset.count()} records")
    
    print("\nSample processed data:")
    for i, item in enumerate(result.dataset.iter_rows()):
        print(f"  Record {i+1}: resized_video_path={item.get('resized_video_path', 'N/A')}")
        if i >= 2:
            break
    
    # Display video result information
    print("\nProcessed video information:")
    display_video_results(
        output_path=OUTPUT_DIR,
        video_field_name='resized_video_path',
        max_files=3,
        save_videos=True,
        videos_output_dir=os.path.join(OUTPUT_DIR, 'exported_videos'),
        video_format='mp4'
    )
    
    pipeline.cleanup()
    return result


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    try:
        example_video_resize_aspect_ratio()
    finally:
        shutdown_ray()
