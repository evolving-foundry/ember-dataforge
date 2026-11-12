from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class OperationResourceProfile(BaseModel):
    """Resource usage snapshot for a single operation execution."""
    operation_name: str
    execution_time_ms: float
    cpu_usage_cores: float          # psutil estimate
    memory_usage_mb: float          # Peak RSS
    gpu_usage: float = 0.0          # 0-1; 0 when no GPU is available
    input_rows: int = 0
    output_rows: int = 0
    timestamp: datetime = Field(default_factory=datetime.now)


class OperationResourceStats(BaseModel):
    """Historical statistics (aggregated across multiple executions)."""
    operation_name: str
    sample_count: int
    avg_cpu_cores: float
    avg_memory_mb: float
    p90_memory_mb: float
    p90_execution_time_ms: float
    is_gpu_intensive: bool = False   # avg_gpu_usage > 0.5
    is_memory_intensive: bool = False  # avg_memory_mb > 2048


class OperationResourceHint(BaseModel):
    """User-declared resource hint, with the highest priority."""
    num_cpus: Optional[float] = None
    num_gpus: Optional[float] = None
    memory_mb: Optional[float] = None
    max_parallelism: Optional[int] = None
