from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class RunStatus(str, Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class PipelineRun(BaseModel):
    run_id: str
    experiment_id: str
    pipeline_config: Dict[str, Any] = Field(default_factory=dict)  # Full configuration snapshot
    config_hash: str = ""                                           # MD5, for fast deduplication
    status: RunStatus = RunStatus.RUNNING
    started_at: datetime = Field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    # Quantitative metrics
    metrics: Dict[str, float] = Field(default_factory=dict)
    quality_scores: Dict[str, float] = Field(default_factory=dict)
    # Metadata
    tags: Dict[str, str] = Field(default_factory=dict)
    notes: str = ""
    git_commit: Optional[str] = None
    artifacts: Dict[str, str] = Field(default_factory=dict)
    # Performance data
    total_execution_time_ms: float = 0.0
    operation_times: Dict[str, float] = Field(default_factory=dict)


class Experiment(BaseModel):
    experiment_id: str
    name: str
    description: str = ""
    created_at: datetime = Field(default_factory=datetime.now)
    tags: Dict[str, str] = Field(default_factory=dict)
    run_ids: List[str] = Field(default_factory=list)
    best_run_id: Optional[str] = None
    primary_metric: str = "overall_quality_score"
