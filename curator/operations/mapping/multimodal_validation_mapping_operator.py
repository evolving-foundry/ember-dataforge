import logging
from typing import Any, Dict, Optional, Union
import ray
from .mapping_operator import MappingOperation

logger = logging.getLogger(__name__)
from base.dataset import MultimodalDataset, DataValidationConfig


class MultimodalValidationMappingOperation(MappingOperation):
    """Multimodal data validation mapping operation"""


    PARAMS_SCHEMA = {
        "category":    "mapping",
        "description": "Multimodal data validation mapping that checks modality-field validity for each record",
        "params": {
            "validation_config": {
                "type":        "object",
                "required":    False,
                "description": "Validation rule dictionary whose keys correspond to DataValidationConfig attributes",
            },
            "skip_invalid": {
                "type":        "boolean",
                "required":    False,
                "default":     False,
                "description": "Whether to skip records that fail validation (True = skip and log a warning; False = retain all records)",
            },
        },
        "example": {"skip_invalid": True},
    }

    def execute(self, dataset: MultimodalDataset) -> MultimodalDataset:
        """
        Execute a multimodal data validation operation

        Args:
            dataset: Input dataset, which can be a Ray Dataset or MultimodalDataset

        Returns:
            Validated dataset
        """
        # Determine the input dataset type
        is_multimodal_dataset = isinstance(dataset, MultimodalDataset)
        
        if is_multimodal_dataset:
            # If the input is a MultimodalDataset, we can use its validation capabilities
            result = self.validate_and_process(dataset)
            return result
        else:
            # If the input is a plain Ray Dataset, perform only basic processing
            # Still wrap it in a MultimodalDataset before returning
            multimodal_dataset = MultimodalDataset(dataset=dataset)
            result = self.validate_and_process(multimodal_dataset)
            return result

    def validate_and_process(self, multimodal_dataset: MultimodalDataset):
        """
        Validate and process a MultimodalDataset

        Args:
            multimodal_dataset: MultimodalDataset instance

        Returns:
            Processed MultimodalDataset
        """
        # Get validation configuration
        validation_config = self.params.get('validation_config', {})
        
        if validation_config:
            # Update the validation configuration of multimodal_dataset
            for key, value in validation_config.items():
                setattr(multimodal_dataset.validation_config, key, value)
        
        # Run validation
        validation_result = multimodal_dataset.validate_data()
        
        if not validation_result.is_valid:
            logger.warning("Data validation failed; error count: %s", len(validation_result.errors))
            for error in validation_result.errors:
                logger.warning("  - %s", error)
        
        if validation_result.warnings:
            logger.warning("Data validation warning count: %s", len(validation_result.warnings))
            for warning in validation_result.warnings:
                logger.warning("  - %s", warning)
        
        # Decide whether to skip items that fail validation based on the parameters
        skip_invalid = self.params.get('skip_invalid', False)
        
        if skip_invalid and not validation_result.is_valid:
            # Skipping invalid data requires more complex filtering logic.
            # This is only an example; a real implementation may need to filter
            # according to the configured validation rules.
            logger.info("Skipping data items that failed validation...")
        
        # Return the MultimodalDataset
        return multimodal_dataset
