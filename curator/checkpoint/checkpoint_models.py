"""
Checkpoint data models for saving and restoring data processing pipeline state
"""
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field, ConfigDict, computed_field
import uuid


class CheckpointStatus(Enum):
    """Checkpoint status enum"""
    ACTIVE = "active"          # Active state, available for resume
    COMPLETED = "completed"    # Completed, indicating the pipeline finished successfully
    FAILED = "failed"          # Failed, indicating pipeline execution failed
    EXPIRED = "expired"        # Expired, past the retention period


class CheckpointType(Enum):
    """Checkpoint type enum"""
    OPERATION_LEVEL = "operation_level"  # Operation level: save after each operation completes
    BATCH_LEVEL = "batch_level"          # Batch level: save after processing a batch of data
    TIME_BASED = "time_based"            # Time based: save at time intervals
    CUSTOM = "custom"                    # Custom: save according to user-defined conditions


class OperationState(BaseModel):
    """Operation status model that records the execution status of a single operation"""
    operation_id: str = Field(..., description="Operation ID")
    operation_name: str = Field(..., description="Operation name")
    operation_type: str = Field(..., description="Operation type")
    start_time: datetime = Field(default_factory=datetime.now, description="Start time")
    end_time: Optional[datetime] = Field(default=None, description="End time")
    status: str = Field(default="pending", description="Execution status")
    parameters: Optional[Dict[str, Any]] = Field(default=None, description="Operation parameters")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Metadata")
    output_location: Optional[str] = Field(default=None, description="Output data storage location")


class CheckpointMetadata(BaseModel):
    """Checkpoint metadata"""
    model_config = ConfigDict(
        use_enum_values=False,  # Keep the Enum instance so code can compare it directly
        extra="ignore",         # Ignore fields migrated to computed_field when loading old checkpoints
    )

    checkpoint_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Checkpoint ID")
    pipeline_id: str = Field(..., description="Pipeline ID")
    session_id: str = Field(..., description="Session ID")
    checkpoint_type: CheckpointType = Field(..., description="Checkpoint type")
    status: CheckpointStatus = Field(default=CheckpointStatus.ACTIVE, description="Checkpoint status")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation time")
    last_updated: datetime = Field(default_factory=datetime.now, description="Last update time")
    completed_operations: List[OperationState] = Field(default_factory=list, description="Completed operations")
    remaining_operations: List[Dict[str, Any]] = Field(default_factory=list, description="Remaining operation configuration")
    data_location: Optional[str] = Field(default=None, description="Data storage location")
    storage_path: str = Field(..., description="Checkpoint storage path")
    total_operations: int = Field(default=0, description="Total operation count")

    # ── Derived fields (computed automatically from completed_operations; no longer stored separately) ──────────

    @computed_field  # type: ignore[misc]
    @property
    def current_operation_index(self) -> int:
        """Current operation index, equal to the number of completed operations."""
        return len(self.completed_operations)

    @computed_field  # type: ignore[misc]
    @property
    def progress_percentage(self) -> float:
        """Progress percentage, computed automatically from completed_operations and total_operations."""
        return (len(self.completed_operations) / max(self.total_operations, 1)) * 100

    # ── State mutation methods ────────────────────────────────────────────────

    def update_status(self, status: CheckpointStatus) -> None:
        """Update the Checkpoint status."""
        self.status = status
        self.last_updated = datetime.now()

    def add_completed_operation(self, operation_state: OperationState) -> None:
        """Append a completed operation; progress is reflected automatically by computed_field, so no manual sync is needed."""
        self.completed_operations.append(operation_state)
        self.last_updated = datetime.now()


class CheckpointConfig(BaseModel):
    """Checkpoint configuration"""
    enabled: bool = Field(default=True, description="Enable checkpoints")
    auto_save_interval_ops: Optional[int] = Field(default=1, description="Autosave interval (operations)")
    auto_save_interval_time: Optional[int] = Field(default=None, description="Autosave interval (seconds)")
    max_checkpoints_to_keep: int = Field(default=5, description="Maximum checkpoints to retain")
    checkpoint_dir: str = Field(default="./checkpoints", description="Checkpoint storage directory")
    checkpoint_on_failure: bool = Field(default=True, description="Save checkpoint on failure")
    checkpoint_on_success: bool = Field(default=False, description="Save checkpoint on success")
    resume_from_latest: bool = Field(default=True, description="Resume from the latest checkpoint")
