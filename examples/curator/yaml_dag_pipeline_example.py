#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Example code for building a DAG (directed acyclic graph) pipeline from YAML

Demonstrates how to define and execute a DAG-style data processing pipeline through a YAML configuration file.

Note: this example only uses implemented operators (loading, saving, filtering, deduplication)
"""

import os
import sys
import tempfile
import yaml
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'examples', 'curator'))

from curator import MultimodalPipeline
from curator.core.pipeline import DAGPipeline
from curator.core.dag.dag_models import DAGDefinition
from common import init_ray, shutdown_ray


def example_1_simple_dag():
    """
    Example 1: A simple serial DAG pipeline

    Uses only implemented operators and runs serially to ensure stability.
    """
    print("\n" + "=" * 60)
    print("Example 1: simple serial DAG pipeline")
    print("=" * 60)

    # Create input data
    input_dir = "./data/input"
    output_dir = "./data/output/dag_simple"
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    import json
    sample_data = [
        {"id": 1, "text": "This is a sample text for processing."},
        {"id": 2, "text": "Another example with advertisement spam."},
        {"id": 3, "text": "This is a sample text for processing."},
        {"id": 4, "text": "Short"},
    ]
    input_file = os.path.join(input_dir, "simple_data.json")
    with open(input_file, 'w', encoding='utf-8') as f:
        json.dump(sample_data, f, ensure_ascii=False, indent=2)
    print(f"Created example input data: {input_file}")

    yaml_content = """
dag:
  entry_nodes:
    - load
  exit_nodes:
    - save

  nodes:
    - id: load
      operation: text_loading
      params:
        data_paths:
          - ./data/input/simple_data.json
        modalities:
          - text

    - id: filter1
      operation: blocklist_filter
      depends_on:
        - load
      params:
        blocklist:
          - spam
          - advertisement
        case_sensitive: false
        match_mode: any

    - id: filter2
      operation: ngram_sample_filter
      depends_on:
        - filter1
      params:
        n: 3
        max_repetition_ratio: 0.3

    - id: dedup
      operation: exact_duplicate_removal
      depends_on:
        - filter2
      params:
        fields:
          - text
        ignore_case: true

    - id: save
      operation: text_saving
      depends_on:
        - dedup
      params:
        output_path: ./data/output/dag_simple
        format: json
"""

    temp_yaml_path = os.path.join(tempfile.gettempdir(), "simple_dag.yaml")
    with open(temp_yaml_path, 'w', encoding='utf-8') as f:
        f.write(yaml_content)

    print("\nDAG structure (serial):")
    print("  load → filter1 → filter2 → dedup → save")
    print("\nNode description:")
    print("  1. load: load text data")
    print("  2. filter1: blocklist filtering")
    print("  3. filter2: N-gram repetition filtering")
    print("  4. dedup: exact deduplication")
    print("  5. save: save results")

    try:
        # Use serial mode (enable_parallel=False) to ensure stability
        dag_pipeline = DAGPipeline.from_yaml(temp_yaml_path, enable_parallel=False)
        print(f"\nDAG pipeline created successfully")
        print(f"Visualized structure:\n{dag_pipeline.visualize()}")

        print("\n>>> Executing DAG pipeline (serial mode)...")
        start_time = time.time()
        results = dag_pipeline.run()
        elapsed = time.time() - start_time
        print(f"DAG pipeline execution complete! Elapsed: {elapsed:.2f}s")

        # Check the output
        output_files = os.listdir(output_dir) if os.path.exists(output_dir) else []
        if output_files:
            print(f"\nOutput file generated: {output_dir}/")

            # Organize file extensions
            for f in output_files:
                if f.endswith('.json') and not f.endswith('.jsonl'):
                    os.rename(os.path.join(output_dir, f),
                             os.path.join(output_dir, f.replace('.json', '.jsonl')))

            output_files = os.listdir(output_dir)
            for f in sorted(output_files):
                file_path = os.path.join(output_dir, f)
                file_size = os.path.getsize(file_path)
                print(f"  - {f} ({file_size} bytes)")

            # Display content
            for f in output_files:
                if f.endswith('.jsonl'):
                    print(f"\n{f} File contents:")
                    with open(os.path.join(output_dir, f), 'r', encoding='utf-8') as rf:
                        print(rf.read())
                    break
        else:
            print(f"\nNote: output directory is empty")

    except Exception as e:
        print(f"Error executing DAG pipeline: {e}")
        import traceback
        traceback.print_exc()

    finally:
        if os.path.exists(temp_yaml_path):
            os.remove(temp_yaml_path)


def example_2_multimodal_dag():
    """
    Example 2: A multimodal DAG pipeline

    Demonstrates how to process text and image data in parallel.
    """
    print("\n" + "=" * 60)
    print("Example 2: multimodal DAG pipeline")
    print("=" * 60)

    # Create input data
    input_dir = "./data/input"
    os.makedirs(input_dir, exist_ok=True)

    import json
    sample_data = [
        {"id": 1, "text": "Sample text for multimodal processing.", "image_path": "./data/images/1.jpg"},
        {"id": 2, "text": "Another text sample.", "image_path": "./data/images/2.jpg"},
    ]
    input_file = os.path.join(input_dir, "multimodal_data.json")
    with open(input_file, 'w', encoding='utf-8') as f:
        json.dump(sample_data, f, ensure_ascii=False, indent=2)
    print(f"Created example input data: {input_file}")

    yaml_content = """
dag:
  entry_nodes:
    - load
  exit_nodes:
    - save

  nodes:
    - id: load
      operation: multimodal_loading
      params:
        data_paths:
          - ./data/input/multimodal_data.json
        modalities:
          - text
          - image

    - id: filter_text
      operation: blocklist_filter
      depends_on:
        - load
      params:
        blocklist: []
        case_sensitive: false

    - id: filter_image
      operation: image_size_filter
      depends_on:
        - load
      params:
        min_width: 100
        min_height: 100

    - id: dedup_text
      operation: exact_duplicate_removal
      depends_on:
        - filter_text
      params:
        fields:
          - text
        ignore_case: true

    - id: save
      operation: multimodal_saving
      depends_on:
        - dedup_text
        - filter_image
      params:
        output_path: ./data/output/multimodal_dag
        format: parquet
"""

    temp_yaml_path = os.path.join(tempfile.gettempdir(), "multimodal_dag.yaml")
    with open(temp_yaml_path, 'w', encoding='utf-8') as f:
        f.write(yaml_content)

    print("\nDAG structure:")
    print("  load")
    print("   ↓")
    print("  [filter_text ∥ filter_image] (parallel branches)")
    print("   ↓")
    print("  dedup_text → save (merge and save)")

    try:
        # Use serial mode
        dag_pipeline = DAGPipeline.from_yaml(temp_yaml_path, enable_parallel=False)
        print(f"\nDAG pipeline created successfully")
        print(f"Visualized structure:\n{dag_pipeline.visualize()}")

        print("\n>>> Executing DAG pipeline...")
        start_time = time.time()
        results = dag_pipeline.run()
        elapsed = time.time() - start_time
        print(f"DAG pipeline execution complete! Elapsed: {elapsed:.2f}s")

        for node_id, ds in results.items():
            print(f"  Exit node {node_id}: {type(ds).__name__}")

    except Exception as e:
        print(f"Error executing DAG pipeline: {e}")
        import traceback
        traceback.print_exc()

    finally:
        if os.path.exists(temp_yaml_path):
            os.remove(temp_yaml_path)


def main():
    """Main function"""
    print("=" * 60)
    print("Example code for building a DAG pipeline from YAML")
    print("=" * 60)
    print("\nUses only implemented operators; runs and generates output files")

    init_ray(num_cpus=2)

    try:
        example_1_simple_dag()
        example_2_multimodal_dag()

        print("\n" + "=" * 60)
        print("All examples completed!")
        print("=" * 60)

        # Display the generated output files
        print("\nGenerated output files:")
        for root, dirs, files in os.walk("./data/output"):
            for f in files:
                if f.endswith('.jsonl'):
                    path = os.path.join(root, f)
                    size = os.path.getsize(path)
                    print(f"  {path} ({size} bytes)")

    finally:
        shutdown_ray()


if __name__ == "__main__":
    main()
