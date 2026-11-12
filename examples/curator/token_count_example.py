#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Complete text token counting workflow example

Demonstrates reading text data from a JSON file, processing it through a pipeline,
counting the number of tokens in each record, and outputting a detailed statistics report.

Usage:
    cd /path/to/DataForge
    python examples/curator/token_count_example.py
"""

# Configure encoding before importing other modules (especially on Windows)
import sys
import os

if sys.platform == 'win32':
    # Fix stdout/stderr encoding before all imports to avoid breaking PyCharm wrappers
    for _stream_attr in ('stdout', 'stderr'):
        _s = getattr(sys, _stream_attr)
        try:
            if hasattr(_s, 'reconfigure'):
                _s.reconfigure(encoding='utf-8', errors='replace')
            elif hasattr(_s, 'buffer'):
                import io
                setattr(sys, _stream_attr,
                        io.TextIOWrapper(_s.buffer, encoding='utf-8', errors='replace'))
        except Exception:
            pass

# Ensure the project root is on the Python path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

# Apply the logging monkey patch before importing Ray-related modules
# (windows_compat only depends on the standard library)
from curator.utils.windows_compat import patch_windows_unicode_logging
patch_windows_unicode_logging()

from curator.core.pipeline import MultimodalPipeline
from curator.operations.loading import TextLoadingOperation
from curator.operations.mapping import TokenCountMappingOperation
from curator.checkpoint.checkpoint_models import CheckpointConfig
from curator.core.config.config import ResourcesConfig, PipelineConfiguration
# This example does not need checkpoints; disable them to avoid an extra Ray
# execution triggered by raw_ds.count()
_NO_CHECKPOINT = CheckpointConfig(enabled=False)


# ──────────────────────────────────────────────
# Data paths (relative to the project root directory)
# ──────────────────────────────────────────────
DATA_FILE = os.path.join(PROJECT_ROOT, "data", "token_count_sample.json")


def example_whitespace_tokenizer():
    """
    Example 1: Whitespace tokenization (default, no additional dependencies)

    Flow: load JSON → count tokens (split on spaces) → also count characters
    """
    print("=" * 60)
    print("Example: configure NPU resource usage")
    print("=" * 60)

    # Configure resources, including the NPU
    resources_config = ResourcesConfig(
        cpu=1,  # Number of CPU cores
        gpu=0,  # Number of GPUs
        npu=1,  # Number of NPUs, if the system has NPU devices to use
        memory="1GB",  # Memory size
        adaptive_scheduling=False  # Whether to enable adaptive resource scheduling
    )

    # Create the complete pipeline configuration
    pipeline_config = PipelineConfiguration(
        resources=resources_config
    )

    # Pass the configuration when initializing the pipeline
    pipeline = MultimodalPipeline(
        config=pipeline_config,
        enable_tracking=False,
        checkpoint_config=_NO_CHECKPOINT
    )

    print("=" * 60)
    print("Example 1: whitespace tokenization (whitespace)")
    print("=" * 60)

    pipeline = MultimodalPipeline(enable_tracking=False, checkpoint_config=_NO_CHECKPOINT)

    # Step 1: Load text data
    pipeline.add_operation(TextLoadingOperation({
        "data_paths": [DATA_FILE]
    }))

    # Step 2: Count tokens
    pipeline.add_operation(TokenCountMappingOperation({
        "text_field": "content",          # Text field to count
        "token_count_field": "token_count",  # Field to write the token count to
        "char_count_field": "char_count",    # Field to write the character count to (optional)
        "tokenizer": "whitespace",           # Split on whitespace
        "print_stats": True,                 # Print the statistics summary
    }))

    result = pipeline.run()
    _show_records(result, fields=["id", "title", "token_count", "char_count"])
    pipeline.cleanup()  # Clean up resources
    return result


def example_char_tokenizer():
    """
    Example 2: Count by character (suitable for Chinese and other languages without spaces)

    Flow: load JSON → count tokens by character
    """
    print("=" * 60)
    print("Example 2: character counting (char)")
    print("=" * 60)

    pipeline = MultimodalPipeline(enable_tracking=False, checkpoint_config=_NO_CHECKPOINT)

    pipeline.add_operation(TextLoadingOperation({
        "data_paths": [DATA_FILE]
    }))

    pipeline.add_operation(TokenCountMappingOperation({
        "text_field": "content",
        "token_count_field": "char_token_count",
        "tokenizer": "char",
        "print_stats": True,
    }))

    result = pipeline.run()
    _show_records(result, fields=["id", "title", "char_token_count"])
    pipeline.cleanup()  # Clean up resources
    return result


def example_tiktoken_tokenizer():
    """
    Example 3: Use tiktoken (OpenAI BPE encoding)

    Requires: pip install tiktoken
    Automatically falls back to whitespace tokenization when not installed.

    Flow: load JSON → count tokens using the cl100k_base encoding (same encoding as GPT-3.5/4)
    """
    print("=" * 60)
    print("Example 3: tiktoken encoding (cl100k_base, compatible with GPT-3.5/4)")
    print("=" * 60)

    pipeline = MultimodalPipeline(enable_tracking=False, checkpoint_config=_NO_CHECKPOINT)

    pipeline.add_operation(TextLoadingOperation({
        "data_paths": [DATA_FILE]
    }))

    pipeline.add_operation(TokenCountMappingOperation({
        "text_field": "content",
        "token_count_field": "tiktoken_count",
        "tokenizer": "tiktoken",
        "model_name": "cl100k_base",   # tiktoken encoding name
        "print_stats": True,
    }))

    result = pipeline.run()
    _show_records(result, fields=["id", "title", "tiktoken_count"])
    pipeline.cleanup()  # Clean up resources
    return result


def example_string_based_registration():
    """
    Example 4: Use a registered operator by string name

    Demonstrates how to reference an operator by the string 'token_count_mapping' in a pipeline,
    suitable for configuration-file-driven scenarios.
    """
    print("=" * 60)
    print("Example 4: use an operator by string name (configuration-driven)")
    print("=" * 60)

    pipeline = MultimodalPipeline(enable_tracking=False, checkpoint_config=_NO_CHECKPOINT)

    pipeline.add_operation("text_loading", {
        "data_paths": [DATA_FILE]
    })

    pipeline.add_operation("token_count_mapping", {
        "text_field": "content",
        "token_count_field": "ws_tokens",
        "char_count_field": "chars",
        "tokenizer": "whitespace",
        "print_stats": True,
    })

    result = pipeline.run()
    _show_records(result, fields=["id", "title", "ws_tokens", "chars"])
    pipeline.cleanup()  # Clean up resources
    return result


# ──────────────────────────────────────────────
# Helper: print records
# ──────────────────────────────────────────────

def _show_records(dataset, fields=None):
    """Print the specified fields of the first few records in the dataset."""
    try:
        rows = dataset.dataset.take_all()
        print(f"\n  Total {len(rows)} records; field details:")
        for row in rows:
            if fields:
                display = {k: row.get(k) for k in fields if k in row}
            else:
                display = row
            # Truncate overly long field values
            for k, v in display.items():
                if isinstance(v, str) and len(v) > 40:
                    display[k] = v[:37] + "..."
            print(f"    {display}")
        print()
    except Exception as exc:
        print(f"  [warning] Unable to display record details: {exc}\n")


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────

if __name__ == "__main__":
    print("\nMultimodal data processing framework - text token counting example\n")

    if not os.path.exists(DATA_FILE):
        print(f"[error] Data file does not exist: {DATA_FILE}")
        sys.exit(1)
    try:
        example_whitespace_tokenizer()
        example_char_tokenizer()
        example_tiktoken_tokenizer()
        example_string_based_registration()
        print("All examples completed!")
    except Exception as exc:
        import traceback
        print(f"\n[error] Execution failed: {exc}")
        traceback.print_exc()
        sys.exit(1)
