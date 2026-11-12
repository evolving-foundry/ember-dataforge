"""
Data processing tracking model for recording the impact of operations on datasets
"""
from datetime import datetime
from enum import Enum
from typing import List, Dict, Any, Optional, Union

from pydantic import BaseModel, Field



class OperationType(Enum):
    """Operation type enum"""
    LOADING = "loading"
    FILTERING = "filtering"
    CLEANING = "cleaning"
    AUGMENTATION = "augmentation"
    DEDUPLICATION = "deduplication"
    FORMATTING = "formatting"
    GENERATION = "generation"
    AGGREGATION = "aggregation"
    GROUPING = "grouping"
    MAPPING = "mapping"
    SELECTION = "selection"
    SAVING = "saving"
    UTILITY = "utility"


class ChangeType(Enum):
    """Change type enum"""
    ROW_ADDED = "row_added"
    ROW_REMOVED = "row_removed"
    ROW_MODIFIED = "row_modified"
    COLUMN_ADDED = "column_added"
    COLUMN_REMOVED = "column_removed"
    COLUMN_MODIFIED = "column_modified"
    DUPLICATE_FOUND = "duplicate_found"
    DUPLICATE_REMOVED = "duplicate_removed"


class DataChangeRecord(BaseModel):
    """Data change record"""
    change_type: ChangeType = Field(..., description="Change type")
    timestamp: datetime = Field(default_factory=datetime.now, description="Change timestamp")
    record_ids: Optional[List[Union[str, int]]] = Field(default=None, description="Affected record IDs")
    columns_affected: Optional[List[str]] = Field(default=None, description="Affected columns")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")
    count: int = Field(default=0, description="Change count")


class OperationTrackingInfo(BaseModel):
    """Operation tracking information"""
    operation_id: str = Field(..., description="Unique operation identifier")
    operation_name: str = Field(..., description="Operation name")
    operation_type: OperationType = Field(..., description="Operation type")
    start_time: datetime = Field(default_factory=datetime.now, description="Start time")
    end_time: Optional[datetime] = Field(default=None, description="End time")
    input_stats: Optional[Dict[str, Any]] = Field(default=None, description="Input data statistics")
    output_stats: Optional[Dict[str, Any]] = Field(default=None, description="Output data statistics")
    changes: List[DataChangeRecord] = Field(default=[], description="Data change records")
    parameters: Optional[Dict[str, Any]] = Field(default=None, description="Operation parameters")
    success: bool = Field(default=True, description="Whether the operation succeeded")
    error_message: Optional[str] = Field(default=None, description="Error message")
    
    def finish_tracking(self, success: bool = True, error_message: Optional[str] = None):
        """Mark the operation as complete"""
        self.end_time = datetime.now()
        self.success = success
        if error_message:
            self.error_message = error_message


class TrackingMetadata(BaseModel):
    """Tracking metadata"""
    pipeline_id: str = Field(..., description="Pipeline ID")
    session_id: str = Field(..., description="Session ID")
    operations: List[OperationTrackingInfo] = Field(default=[], description="Operation tracking information list")
    total_input_records: int = Field(default=0, description="Total input records")
    total_output_records: int = Field(default=0, description="Total output records")
    
    def add_operation(self, operation_info: OperationTrackingInfo):
        """Add operation tracking information"""
        self.operations.append(operation_info)
        
        # Update total counts
        if operation_info.output_stats and 'num_rows' in operation_info.output_stats:
            self.total_output_records = operation_info.output_stats['num_rows']