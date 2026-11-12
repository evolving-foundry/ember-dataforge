#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AudioLoadingOperation example script

Function:
    Load audio data from local paths, supporting .mp3 / .wav / .aac / .flac / .ogg and other formats

Usage:
    python examples/curator/single_ops/loading/audio_loading_example.py
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
from common import init_ray, shutdown_ray

init_ray()

_NO_CHECKPOINT = CheckpointConfig(enabled=False)
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "audios")


def example_audio_loading():
    """Audio loading example"""
    print("=" * 60)
    print("Example: audio data loading")
    print("=" * 60)
    
    pipeline = MultimodalPipeline(
        enable_tracking=False,
        checkpoint_config=_NO_CHECKPOINT
    )
    
    pipeline.add_operation(AudioLoadingOperation({
        "data_paths": [DATA_DIR]
    }))

    output_path = os.path.join(PROJECT_ROOT, "data", "output", "audio_loaded_output.parquet")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pipeline.add_operation(AudioSavingOperation({
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
    example_audio_loading()

shutdown_ray()
