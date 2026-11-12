#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TokenCountMappingOperation example script

Function:
    Token count mapping: count the number of tokens in each record

Usage:
    python examples/curator/single_ops/mapping/token_count_mapping_example.py
"""

import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)); PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR))))
sys.path.insert(0, PROJECT_ROOT)

from curator.core.pipeline import MultimodalPipeline
from curator.operations.loading import MultiModalLoadingOperation
from curator.operations.mapping import TokenCountMappingOperation
from curator.operations.saving import TextSavingOperation
from curator.checkpoint.checkpoint_models import CheckpointConfig

os.environ['PYTHONPATH'] = PROJECT_ROOT + os.pathsep + os.environ.get('PYTHONPATH', '')
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'examples', 'curator'))
from common import init_ray, shutdown_ray

init_ray()

_NO_CHECKPOINT = CheckpointConfig(enabled=False)
DATA_FILE = os.path.join(PROJECT_ROOT, "data", "token_count_sample.json")


def example_token_count_mapping():
    """Token count mapping example"""
    print("=" * 60)
    print("Example: token count mapping")
    print("=" * 60)
    
    pipeline = MultimodalPipeline(
        enable_tracking=False,
        checkpoint_config=_NO_CHECKPOINT
    )
    
    # Load data
    pipeline.add_operation(MultiModalLoadingOperation({
        "data_paths": [DATA_FILE],
        "modalities": ["text"]
    }))
    
    # Token count mapping
    pipeline.add_operation(TokenCountMappingOperation({
        "text_field": "content",
        "token_count_field": "token_count",
        "char_count_field": "char_count",
        "tokenizer": "whitespace",
        "print_stats": True
    }))

    output_path = os.path.join(PROJECT_ROOT, "data", "output", "token_count_output.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pipeline.add_operation(TextSavingOperation({
        "output_path": output_path,
        "format": "json"
    }))

    result = pipeline.run()
    print(f"Mapped {result.dataset.count()} records")
    
    print("\nSample mapped data:")
    for i, item in enumerate(result.dataset.iter_rows()):
        print(f"  Record {i+1}: token_count={item.get('token_count', 'N/A')}, char_count={item.get('char_count', 'N/A')}")
        if i >= 2:
            break
    
    pipeline.cleanup()
    return result


if __name__ == "__main__":
    example_token_count_mapping()

shutdown_ray()
