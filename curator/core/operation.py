from typing import Any, Dict, Optional
from abc import ABC, abstractmethod

from base.dataset import MultimodalDataset
from ..tracking.tracking import OperationType


class Operation(ABC):
    """Operation base class; parent class of all data processing operations.

    Subclass conventions:
      - Set the class variable operation_type (an OperationType enum value)
      - Optionally declare PARAMS_SCHEMA to describe parameter specifications, in this format:
          {
              'params': {
                  'field_name': {
                      'type': 'str',
                      'required': True/False,
                      'default': <value>,   # Valid only when required=False
                      'description': '...',
                  },
                  ...
              }
          }
        __init__ automatically fills defaults from PARAMS_SCHEMA and validates required parameters.
      - Optionally declare RESOURCE_HINT (OperationResourceHint) to override the default recommendation for adaptive scheduling.
    """

    # Must be set by subclasses
    operation_type: OperationType = None

    # Subclasses may optionally declare a parameter schema for automatic validation and documentation generation
    PARAMS_SCHEMA: Dict[str, Any] = {}

    # Subclasses may optionally declare a resource hint for the adaptive resource scheduler (highest priority)
    RESOURCE_HINT: Optional[Any] = None  # OperationResourceHint

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        if self.operation_type is None:
            raise NotImplementedError(
                f"{self.__class__.__name__} must set class variable 'operation_type'. "
                f"Example: operation_type = OperationType.CLEANING"
            )
        self.params = self._apply_param_defaults(params or {})
        self.name = self.__class__.__name__

    @classmethod
    def _apply_param_defaults(cls, params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate required parameters from PARAMS_SCHEMA and fill in default values.

        - required=True and not provided → raise ValueError
        - Not provided and schema has a default → fill automatically
        - PARAMS_SCHEMA is empty or has no 'params' key → return unchanged
        """
        schema_params = cls.PARAMS_SCHEMA.get("params", {})
        if not schema_params:
            return dict(params)

        result = dict(params)
        for key, spec in schema_params.items():
            if key in result:
                continue
            if spec.get("required", False):
                raise ValueError(
                    f"{cls.__name__}: missing required parameter '{key}'. "
                    f"{spec.get('description', '')}"
                )
            if "default" in spec:
                result[key] = spec["default"]
        return result

    @abstractmethod
    def execute(self, dataset: Optional[MultimodalDataset]) -> MultimodalDataset:
        """Execute the operation and return the processed MultimodalDataset.

        Args:
            dataset: Input dataset; LoadingOperation may accept None.
        """

    def get_params(self) -> Dict[str, Any]:
        """Return a copy of the parameter dictionary."""
        return self.params.copy()

    def set_params(self, params: Dict[str, Any]) -> None:
        """Merge and update operation parameters."""
        self.params.update(params)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(params={self.params})"
