"""
Tracking manager - an independent utility for handling operation tracking logic
"""
from typing import Optional

from .data_tracker import DataTracker
from .tracking import OperationType
from base.dataset import MultimodalDataset


class _NullTracker:
    """Null-object implementation: acts as a DataTracker placeholder when tracking is disabled.

    All methods are no-ops, eliminating None checks in callers.
    """

    def track_operation(self, *args, **kwargs) -> None:
        pass

    def get_tracking_metadata(self):
        return None


class TrackingManager:
    """Tracking manager that encapsulates tracker-related operations.

    When tracker=None, _NullTracker is used automatically, so callers do not need to check whether tracking is enabled.
    """

    def __init__(self, tracker: Optional[DataTracker] = None):
        self.tracker = tracker if tracker is not None else _NullTracker()

    def track_operation(
        self,
        operation,
        input_dataset: Optional[MultimodalDataset],
        output_dataset: Optional[MultimodalDataset],
    ) -> None:
        """Track data changes before and after operation execution."""
        self.tracker.track_operation(
            operation_name=operation.name,
            operation_type=operation.operation_type,
            input_dataset=input_dataset,
            output_dataset=output_dataset,
            parameters=operation.get_params(),
        )

    def get_tracking_report(self):
        """Get the tracking report; return None when tracking is not enabled."""
        return self.tracker.get_tracking_metadata()
