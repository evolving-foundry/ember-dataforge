"""
Data saver factory for dynamically selecting the appropriate Ray data saving method
"""
from typing import Dict, Any, List, Union
import ray
from ray.data import Dataset
import os
import numpy as np


class DataSaverFactory:
    """Data saver factory class for dynamically selecting the appropriate save method based on output format"""

    # Mapping from format to Ray save method
    FORMAT_TO_SAVER = {
        'json': 'write_json',
        'parquet': 'write_parquet',
        'csv': 'write_csv',
        'text': 'write_text'
    }

    @classmethod
    def detect_format_from_path(cls, output_path: str) -> str:
        """
        Detect the save format from the output path
        
        Args:
            output_path: Output path
            
        Returns:
            Format type ('json', 'parquet', 'csv', 'text')
        """
        _, ext = os.path.splitext(output_path.lower())
        format_map = {
            '.json': 'json',
            '.jsonl': 'json',
            '.parquet': 'parquet',
            '.csv': 'csv',
            '.txt': 'text'
        }
        return format_map.get(ext, 'json')  # Default to json

    @classmethod
    def _prepare_tensor_columns(cls, dataset: Dataset) -> Dataset:
        """
        Convert numpy array columns to Ray Data tensor format to preserve multi-dimensional shapes
        
        Args:
            dataset: Input dataset
            
        Returns:
            Processed dataset with numpy array columns converted to tensor format
        """
        # Inspect the schema to determine which columns need conversion
        try:
            schema = dataset.schema()
            # Get the list of column names
            field_names = schema.names if hasattr(schema, 'names') else []
            field_types = {str(schema.field(name).type) for name in field_names}
        except Exception:
            return dataset
        
        # Get columns with numpy array types
        tensor_columns = [name for name in field_names 
                        if 'object' in str(schema.field(name).type).lower() 
                        or 'null' in str(schema.field(name).type).lower()]
        
        if not tensor_columns:
            return dataset
        
        # Use map_batches to convert numpy arrays to tensor format
        def convert_to_tensor(batch: dict) -> dict:
            for col in tensor_columns:
                if col in batch and batch[col] is not None:
                    arr = batch[col]
                    if isinstance(arr, np.ndarray):
                        # Keep numpy arrays as-is; Ray Data handles them automatically
                        batch[col] = arr
            return batch
        
        try:
            return dataset.map_batches(convert_to_tensor, batch_format='numpy')
        except Exception:
            return dataset

    @classmethod
    def save_dataset(cls, dataset: Dataset, output_path: str, format_type: str = None):
        """
        Save the dataset to the specified path
        
        Args:
            dataset: Dataset to save
            output_path: Output path
            format_type: Format type; inferred from the path when None
        """
        if format_type is None:
            format_type = cls.detect_format_from_path(output_path)

        saver_method = cls.FORMAT_TO_SAVER.get(format_type)
        if not saver_method:
            raise ValueError(f"Unsupported format: {format_type}")

        # Create the output directory if it does not exist
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # For parquet format, first ensure tensor columns are handled correctly
        if format_type == 'parquet':
            dataset = cls._prepare_tensor_columns(dataset)

        # Use the corresponding save method
        saver = getattr(dataset, saver_method)
        saver(output_path)

    @classmethod
    def get_supported_formats(cls) -> List[str]:
        """
        Get the list of supported save formats
        
        Returns:
            List of supported formats
        """
        return list(cls.FORMAT_TO_SAVER.keys())