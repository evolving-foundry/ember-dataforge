#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Example code for building a pipeline from YAML

Demonstrates how to define and execute a data processing pipeline through a YAML configuration file,
including text processing, image processing, and multimodal data processing scenarios.

Note: this example only uses implemented operators (loading, saving, filtering, deduplication)
"""

import os
import sys
import tempfile
import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'examples', 'curator'))

from curator import MultimodalPipeline
from curator.core.config import PipelineConfiguration
from common import init_ray, shutdown_ray


def example_1_basic_yaml_pipeline():
    """Example 1: Basic YAML-configured pipeline - uses only implemented operators"""
    print("\n" + "=" * 60)
    print("Example 1: basic YAML-configured pipeline")
    print("=" * 60)

    yaml_content = """
global:
  logging_level: info
  temp_dir: ./temp
  cache_dir: ./cache

data:
  input_path: ./data/input/text_data.json
  output_path: ./data/output/processed_text
  modalities:
    - text

resources:
  cpu: 2
  memory: 2GB

pipeline:
  operations:
    - name: text_loading
      params:
        data_paths:
          - ./data/input/text_data.json
        modalities:
          - text

    - name: exact_duplicate_removal
      params:
        fields:
          - text
        ignore_case: true

    - name: blocklist_filter
      params:
        blocklist:
          - spam
          - advertisement
        case_sensitive: false
        match_mode: any

    - name: text_saving
      params:
        output_path: ./data/output/processed_text
        format: json
"""

    temp_yaml_path = os.path.join(tempfile.gettempdir(), "basic_pipeline_config.yaml")
    with open(temp_yaml_path, 'w', encoding='utf-8') as f:
        f.write(yaml_content)

    print(f"\nCreated temporary YAML configuration file")

    # Create example data
    input_dir = "./data/input"
    output_dir = "./data/output/processed_text"
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    import json
    sample_data = [
        {"id": 1, "text": "This is a sample text for processing."},
        {"id": 2, "text": "Another example with some content."},
        {"id": 3, "text": "This is a sample text for processing."},
    ]
    input_file = os.path.join(input_dir, "text_data.json")
    with open(input_file, 'w', encoding='utf-8') as f:
        json.dump(sample_data, f, ensure_ascii=False, indent=2)
    print(f"Created example input data: {input_file}")

    try:
        print("\n>>> Creating pipeline...")
        pipeline = MultimodalPipeline(config=temp_yaml_path)
        print(f"Pipeline created successfully with {len(pipeline.operations)} operations")
        for i, op in enumerate(pipeline.operations):
            print(f"  [{i+1}] {op.name}")

        print("\n>>> Executing pipeline...")
        result = pipeline.run()
        print(f"Pipeline execution complete!")

        # Check the output and organize files
        output_files = os.listdir(output_dir) if os.path.exists(output_dir) else []
        if output_files:
            print(f"\nOutput file generated: {output_dir}/")

            # JSON files saved by Ray Data are actually in JSONL format; organize file extensions
            for f in output_files:
                if f.endswith('.json') and not f.endswith('.jsonl'):
                    old_path = os.path.join(output_dir, f)
                    new_path = os.path.join(output_dir, f.replace('.json', '.jsonl'))
                    os.rename(old_path, new_path)
                    print(f"  - {f} -> {f.replace('.json', '.jsonl')} (renamed to JSONL format)")

            # Re-list the files
            output_files = os.listdir(output_dir)
            for f in sorted(output_files):
                file_path = os.path.join(output_dir, f)
                file_size = os.path.getsize(file_path) if os.path.isfile(file_path) else 0
                print(f"  - {f} ({file_size} bytes)")

            # Read and display the content
            for f in output_files:
                if f.endswith('.jsonl'):
                    print(f"\n{f} File contents:")
                    with open(os.path.join(output_dir, f), 'r', encoding='utf-8') as rf:
                        content = rf.read()
                        print(content)
                    break
        else:
            print(f"\nNote: output directory is empty: {output_dir}")

    except Exception as e:
        print(f"Error executing pipeline: {e}")
        import traceback
        traceback.print_exc()

    finally:
        if os.path.exists(temp_yaml_path):
            os.remove(temp_yaml_path)


def main():
    """Main function"""
    print("=" * 60)
    print("Example code for building a pipeline from YAML")
    print("=" * 60)

    init_ray(num_cpus=2)

    try:
        example_1_basic_yaml_pipeline()
        print("\n" + "=" * 60)
        print("Example completed!")
        print("=" * 60)
    finally:
        shutdown_ray()


if __name__ == "__main__":
    main()
