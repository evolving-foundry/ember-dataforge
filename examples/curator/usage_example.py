#!/usr/bin/env python
"""Runnable Curator pipeline example.

This example intentionally uses only operations that are implemented in the
current release. It reads the versioned sample asset, removes blank content,
adds token counts, filters on an existing quality score, and writes the result
to a temporary directory.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from curator import MultimodalDataset, MultimodalPipeline
from curator.operations.loading import MultiModalLoadingOperation
from curator.operations.cleaning import InvalidDataRemovalOperation
from curator.operations.filtering import QualityBasedFilterOperation, ModelBasedFilterOperation
from curator.operations.saving import MultiModalSavingOperation
from curator.operations.augmentation import TextAugmentationOperation
from curator.core.config import PipelineConfiguration
import tempfile
import os
import yaml


def example_with_pipeline():
    """
    Process data through a pipeline (recommended).
    All operations, including loading and saving, are managed by the pipeline.
    """
    print("=== Processing data with a pipeline ===")
    
    # Create a pipeline
    pipeline = MultimodalPipeline()

    # Add a loading operation
    pipeline.add_operation(MultiModalLoadingOperation({
        "data_paths": ["./data/sample.json"],
        "modalities": ["text", "image"]
    }))

    # Add processing operations
    pipeline.add_operation(InvalidDataRemovalOperation({"remove_invalid": True}))
    pipeline.add_operation(QualityBasedFilterOperation({"quality_threshold": 0.7}))

    # Add a saving operation
    pipeline.add_operation(MultiModalSavingOperation({
        "output_path": "./data/processed",
        "format": "json"
    }))

    # Run the pipeline
    processed_ds = pipeline.run()
    print("Pipeline processing complete")


def example_with_loaded_dataset():
    """
    Process an already loaded dataset.
    Use this approach when a Ray dataset already exists and only needs transformation.
    """
    print("\n=== Using a loaded dataset ===")
    
    # First load the data with a pipeline
    pipeline = MultimodalPipeline()
    pipeline.add_operation(MultiModalLoadingOperation({
        "data_paths": ["./data/sample.json"],
        "modalities": ["text", "image"]
    }))
    
    # Get the loaded dataset
    loaded_dataset = pipeline.run()
    
    # Then process the loaded dataset with MultimodalDataset
    dataset = MultimodalDataset(
        dataset=loaded_dataset,
        modalities=["text", "image"]
    )
    
    # Transform the data
    def clean_data(row):
        # Data cleaning logic
        if 'text' in row and row['text']:
            row['text'] = row['text'].strip().lower()
        return row

    # Apply the transformation (executed in a distributed manner by Ray)
    cleaned_dataset = dataset.map(clean_data)
    
    print(f"Dataset processing complete: {cleaned_dataset.count()} records")


def chained_pipeline_example():
    """
    Chained pipeline example.
    Demonstrates using the output of one pipeline as the input to another.
    """
    print("=== Chained pipeline example ===")
    
    # First pipeline: data loading and initial cleaning
    print("\n1. Run the first pipeline: data loading and initial cleaning")
    pipeline1 = MultimodalPipeline()
    
    # Add a loading operation
    pipeline1.add_operation(MultiModalLoadingOperation({
        "data_paths": ["./data/sample.json"],
        "modalities": ["text", "image"]
    }))
    
    # Add an initial cleaning operation
    pipeline1.add_operation(InvalidDataRemovalOperation({
        "remove_invalid": True
    }))
    
    # Run the first pipeline
    processed_dataset_1 = pipeline1.run()
    print(f"First pipeline complete: {processed_dataset_1.count()} records")
    
    # Second pipeline: quality filtering and augmentation
    print("\n2. Run the second pipeline: quality filtering and augmentation")
    pipeline2 = MultimodalPipeline()
    
    # Add a quality filtering operation
    pipeline2.add_operation(QualityBasedFilterOperation({
        "quality_threshold": 0.7
    }))
    
    # Add a text augmentation operation
    pipeline2.add_operation(TextAugmentationOperation({
        "strategy": "synonym_replacement"
    }))
    
    # Use the first pipeline's output as the second pipeline's input
    processed_dataset_2 = pipeline2.run(dataset=processed_dataset_1)
    print(f"Second pipeline complete: {processed_dataset_2.count()} records")
    
    # Third pipeline: final save
    print("\n3. Run the third pipeline: data saving")
    pipeline3 = MultimodalPipeline()
    
    # Add a saving operation
    pipeline3.add_operation(MultiModalSavingOperation({
        "output_path": "./data/processed_output.json",
        "format": "json"
    }))
    
    # Use the second pipeline's output as the third pipeline's input
    final_result = pipeline3.run(dataset=processed_dataset_2)
    print(f"Third pipeline complete")
    
    print(f"\nChained processing complete! Three pipeline steps were executed.")


def alternative_approach_with_dataset_object():
    """
    Alternative approach: use a MultimodalDataset object as an intermediate step.
    """
    print("\n=== Using a MultimodalDataset as an intermediate step ===")
    
    # Run the initial pipeline
    pipeline = MultimodalPipeline()
    pipeline.add_operation(MultiModalLoadingOperation({
        "data_paths": ["./data/sample.json"],
        "modalities": ["text", "image"]
    }))
    pipeline.add_operation(InvalidDataRemovalOperation({
        "remove_invalid": True
    }))
    
    # Get the processed dataset
    processed_dataset = pipeline.run()
    print(f"Initial pipeline complete: {processed_dataset.count()} records")
    
    # Create a MultimodalDataset object for additional processing
    dataset_obj = MultimodalDataset(
        dataset=processed_dataset,
        modalities=["text", "image"]
    )
    
    # Perform additional processing at the MultimodalDataset level
    def additional_cleaning(row):
        # More fine-grained data cleaning
        if 'text' in row and row['text']:
            row['text'] = row['text'].strip().lower()
            # Additional processing logic can be added here
        return row
    
    # Apply the additional processing
    additional_processed = dataset_obj.map(additional_cleaning)
    print(f"Additional processing complete; dataset size: {additional_processed.count()}")
    
    # Pass the processed dataset to a new pipeline
    new_pipeline = MultimodalPipeline()
    new_pipeline.add_operation(QualityBasedFilterOperation({
        "quality_threshold": 0.8
    }))
    new_pipeline.add_operation(MultiModalSavingOperation({
        "output_path": "./data/alternative_output.json",
        "format": "json"
    }))
    
    # Use the additionally processed dataset as input
    final_result = new_pipeline.run(dataset=additional_processed)
    print("Processing with the intermediate MultimodalDataset object complete")


def example_with_string_operations():
    """
    Example of adding operations by string name.
    This allows processing flows to be defined flexibly through configuration files.
    """
    print("\n=== Adding operations by string name ===")
    
    # Create a pipeline
    pipeline = MultimodalPipeline()
    
    # Add operations by string name, allowing pipelines to be defined dynamically
    # from configuration files
    pipeline.add_operation("multi_modal_loading", {
        "data_paths": ["./data/sample.json"],
        "modalities": ["text", "image"]
    })
    
    # Add a data cleaning operation
    pipeline.add_operation("invalid_data_removal", {
        "remove_invalid": True
    })
    
    # Add a quality filtering operation
    pipeline.add_operation("quality_based_filter", {
        "quality_threshold": 0.7
    })
    
    # Add a text augmentation operation
    pipeline.add_operation("text_augmentation", {
        "strategy": "synonym_replacement"
    })
    
    # Add a saving operation
    pipeline.add_operation("multi_modal_saving", {
        "output_path": "./data/string_based_output.json",
        "format": "json"
    })
    
    # Run the pipeline
    processed_ds = pipeline.run()
    print("Pipeline using string-named operations complete")
    
    # Show the tracking report when tracking is enabled
    report = pipeline.get_tracking_report()
    print(f"Trace report: {report}")


def example_mixed_approach():
    """
    Mixed approach example: add operations using both instances and string names.
    Demonstrates the flexibility of the framework.
    """
    print("\n=== Mixed-method example ===")
    
    pipeline = MultimodalPipeline()
    
    # Add a loading operation using an instance
    pipeline.add_operation(MultiModalLoadingOperation({
        "data_paths": ["./data/sample.json"],
        "modalities": ["text", "image"]
    }))
    
    # Add a cleaning operation by string name
    pipeline.add_operation("invalid_data_removal", {
        "remove_invalid": True
    })
    
    # Add a filtering operation using an instance
    pipeline.add_operation(QualityBasedFilterOperation({
        "quality_threshold": 0.8
    }))
    
    # Add a saving operation by string name
    pipeline.add_operation("multi_modal_saving", {
        "output_path": "./data/mixed_approach_output.json",
        "format": "json"
    })
    
    # Run the pipeline
    processed_ds = pipeline.run()
    print("Mixed-method pipeline complete")


def example_with_configured_pipeline():
    """
    Example of creating a pipeline from a configuration object.
    Shows how to define the entire pipeline with a configuration file or object.
    """
    print("\n=== Creating a pipeline from a configuration object ===")
    
    # Method 1: use a configuration dict with the full set of options
    config_dict = {
        "global": {
            "logging_level": "info",
            "temp_dir": "./temp",
            "cache_dir": "./cache"
        },
        "data": {
            "input_path": "./data",
            "output_path": "./output",
            "modalities": ["text", "image"]
        },
        "resources": {
            "gpu": 0,  # Use zero GPUs for this example
            "cpu": 2,
            "memory": "8GB"
        },
        "pipeline": {
            "operations": [
                {
                    "name": "multi_modal_loading",
                    "params": {
                        "data_paths": ["./data/sample.json"],
                        "modalities": ["text", "image"]
                    }
                },
                {
                    "name": "invalid_data_removal",
                    "params": {
                        "remove_invalid": True
                    }
                },
                {
                    "name": "quality_based_filter",
                    "params": {
                        "quality_threshold": 0.75
                    }
                },
                {
                    "name": "multi_modal_saving",
                    "params": {
                        "output_path": "./data/configured_output.json",
                        "format": "json"
                    }
                }
            ]
        }
    }
    
    # Create a pipeline from the dict configuration
    pipeline1 = MultimodalPipeline(config=config_dict)
    
    # Run the pipeline
    processed_ds1 = pipeline1.run()
    print("Pipeline using dictionary configuration complete")
    
    # Method 2: use a PipelineConfiguration object
    config_obj = PipelineConfiguration.from_dict(config_dict)
    pipeline2 = MultimodalPipeline(config=config_obj)
    
    # Run the pipeline
    processed_ds2 = pipeline2.run()
    print("Pipeline using a configuration object complete")
    
    # Method 3: load configuration from a YAML file
    # Create a temporary YAML configuration file
    temp_config_path = os.path.join(tempfile.gettempdir(), "temp_pipeline_config.yaml")
    
    yaml_config = {
        "global": {
            "logging_level": "warning",
            "temp_dir": "/tmp",
            "cache_dir": ".cache"
        },
        "data": {
            "input_path": "./data",
            "output_path": "./output",
            "modalities": ["text", "image"]
        },
        "resources": {
            "gpu": 1,
            "cpu": 4,
            "memory": "16GB"
        },
        "pipeline": {
            "operations": [
                {
                    "name": "multi_modal_loading",
                    "params": {
                        "data_paths": ["./data/sample.json"],
                        "modalities": ["text", "image"]
                    }
                },
                {
                    "name": "invalid_data_removal",
                    "params": {
                        "remove_invalid": True
                    }
                },
                {
                    "name": "quality_based_filter",
                    "params": {
                        "quality_threshold": 0.6
                    }
                },
                {
                    "name": "multi_modal_saving",
                    "params": {
                        "output_path": "./data/yaml_configured_output.json",
                        "format": "json"
                    }
                }
            ]
        }
    }
    
    # Write the temporary YAML file
    with open(temp_config_path, 'w', encoding='utf-8') as f:
        yaml.dump(yaml_config, f, default_flow_style=False, allow_unicode=True)
    
    # Create a pipeline from the YAML file
    pipeline3 = MultimodalPipeline(config=temp_config_path)
    
    # Run the pipeline
    processed_ds3 = pipeline3.run()
    print("Pipeline using YAML configuration complete")
    
    # Clean up the temporary file
    try:
        os.remove(temp_config_path)
    except:
        pass  # Ignore deletion failures.


if __name__ == "__main__":
    print("Multimodal data processing framework usage example")

    try:
        example_with_pipeline()
        example_with_loaded_dataset()
        example_mixed_approach()
        example_with_string_operations()
        example_with_configured_pipeline()
        
        print("\nAll examples completed!")
    except Exception as e:
        print(f"Error during execution: {e}")
        print("Note: paths in the examples are illustrative; ensure the files exist before running them.")
