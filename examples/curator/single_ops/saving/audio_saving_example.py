#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AudioSavingOperation example script

Function:
    Save an audio dataset to the specified path

Usage:
    python examples/curator/single_ops/saving/audio_saving_example.py
"""

import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)); PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR))))
sys.path.insert(0, PROJECT_ROOT)

from curator.core.pipeline import MultimodalPipeline
from curator.operations.loading import AudioLoadingOperation
from curator.operations.saving import AudioSavingOperation
from curator.checkpoint.checkpoint_models import CheckpointConfig

os.environ['PYTHONPATH'] = PROJECT_ROOT + os.pathsep + os.environ.get('PYTHONPATH', '')
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'examples', 'curator'))
from common import init_ray, shutdown_ray, display_audio_results

init_ray()

_NO_CHECKPOINT = CheckpointConfig(enabled=False)
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "audios")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "output", "audio_saving")


def example_audio_saving():
    """Audio saving example"""
    print("=" * 60)
    print("Example: audio data saving")
    print("=" * 60)
    
    if not os.path.exists(DATA_DIR):
        print(f"Error: data directory {DATA_DIR} does not exist")
        return None

    audio_files = [f for f in os.listdir(DATA_DIR) if
                  f.lower().endswith(('.mp3', '.wav', '.aac', '.flac', '.ogg', '.m4a'))]
    print(f"Found {len(audio_files)} audio files: {audio_files}")

    if not audio_files:
        print("Warning: no audio files found")
        return None

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output directory created: {OUTPUT_DIR}")
    
    pipeline = MultimodalPipeline(
        enable_tracking=False,
        checkpoint_config=_NO_CHECKPOINT
    )
    
    # Step 1: Load audio data
    pipeline.add_operation(AudioLoadingOperation({
        "data_paths": [DATA_DIR]
    }))
    
    # Step 2: Save data
    output_path = os.path.join(OUTPUT_DIR, "audio_output.parquet")
    pipeline.add_operation(AudioSavingOperation({
        "output_path": output_path,
        "format": "parquet"
    }))
    
    result = pipeline.run()
    print(f"Saved {result.dataset.count()} records")
    
    # Display the saved audio results
    display_audio_results(output_path, audio_field_name='path')
    
    pipeline.cleanup()
    return result


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    example_audio_saving()

shutdown_ray()
