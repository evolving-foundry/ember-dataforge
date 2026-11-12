from typing import List, Dict, Any, Optional, Union
import ray
from ray.data import Dataset
try:
    from ray.data import read_images, read_audio, read_videos
except ImportError:
    # Preserve backward compatibility
    pass
from pydantic import BaseModel, ValidationError, validator
from typing import Literal


class DataValidationConfig(BaseModel):
    """Configuration model for data validation."""
    required_fields: Optional[List[str]] = []
    field_types: Optional[Dict[str, str]] = {}  # Field type mapping
    text_min_length: Optional[int] = 1
    text_max_length: Optional[int] = 10000
    image_min_width: Optional[int] = 1
    image_min_height: Optional[int] = 1
    audio_min_duration: Optional[float] = 0.0
    video_min_duration: Optional[float] = 0.0


class ValidationResult(BaseModel):
    """Validation result model."""
    is_valid: bool
    errors: List[str] = []
    warnings: List[str] = []


class MultimodalDataset:
    """Distributed multimodal dataset focused on data processing and management."""

    def __init__(self, dataset: Dataset, modalities: List[str] = None, validation_config: Optional[DataValidationConfig] = None, **kwargs):
        """
        Initialize a multimodal dataset.

        Args:
            dataset: Required Ray dataset instance.
            modalities: Modalities such as ``["text", "image", "video", "audio"]``.
            validation_config: Data validation configuration.
            **kwargs: Additional arguments.
        """
        if dataset is None:
            raise ValueError("Dataset cannot be None. Use pipeline operations to load data.")
        
        self.dataset = dataset
        self.modalities = modalities or []
        self.validation_config = validation_config or DataValidationConfig()
        self.kwargs = kwargs

    def validate_data(self) -> ValidationResult:
        """Validate the data in the dataset."""
        errors = []
        warnings = []
        
        # Check required fields
        if self.validation_config.required_fields:
            schema = self.dataset.schema()
            schema_fields = set(schema.names if hasattr(schema, 'names') else [])
            missing_fields = set(self.validation_config.required_fields) - schema_fields
            if missing_fields:
                errors.append(f"Missing required fields: {missing_fields}")

        try:
            sample_row = self.dataset.take(1)[0] if self.dataset.count() > 0 else None
            if sample_row:
                # Validate field-specific requirements such as text length
                for field_name, field_type in (self.validation_config.field_types or {}).items():
                    if field_name in sample_row:
                        value = sample_row[field_name]
                        if field_type == "text":
                            if self.validation_config.text_min_length and len(str(value)) < self.validation_config.text_min_length:
                                errors.append(f"Text field '{field_name}' has length {len(str(value))} which is less than minimum {self.validation_config.text_min_length}")
                            if self.validation_config.text_max_length and len(str(value)) > self.validation_config.text_max_length:
                                warnings.append(f"Text field '{field_name}' has length {len(str(value))} which exceeds maximum {self.validation_config.text_max_length}")
        except Exception as e:
            warnings.append(f"Could not validate sample data: {str(e)}")

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )

    def map(self, func):
        """Apply a transformation function."""
        return MultimodalDataset(self.dataset.map(func), self.modalities, self.validation_config)

    def filter(self, func):
        """Apply a filter function."""
        return MultimodalDataset(
            self.dataset.filter(func),
            self.modalities,
            self.validation_config,
            **self.kwargs
        )

    def map_batches(self, func, batch_size: Optional[int] = None, num_gpus: int = 0):
        """Apply a function to batches."""
        transformed_dataset = self.dataset.map_batches(
            func,
            batch_size=batch_size,
            num_gpus=num_gpus
        )
        return MultimodalDataset(
            transformed_dataset,
            self.modalities,
            self.validation_config,
            **self.kwargs
        )

    def shuffle(self):
        """Shuffle the data."""
        return MultimodalDataset(
            self.dataset.random_shuffle(),
            self.modalities,
            self.validation_config,
            **self.kwargs
        )

    def split(self, ratios: List[float]) -> List['MultimodalDataset']:
        """Split the dataset."""
        split_datasets = self.dataset.train_test_split(ratios)
        return [
            MultimodalDataset(
                ds,
                self.modalities,
                self.validation_config,
                **self.kwargs
            ) for ds in split_datasets
        ]

    def get_stats(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        stats = {
            "num_rows": self.dataset.count(),
            "modalities": self.modalities,
        }
        
        # Collect additional statistics
        schema = self.dataset.schema()
        stats["schema"] = str(schema)
        
        return stats

    def distinct(self):
        """Return a deduplicated dataset."""
        return MultimodalDataset(
            self.dataset.distinct(),
            self.modalities,
            self.validation_config,
            **self.kwargs
        )
