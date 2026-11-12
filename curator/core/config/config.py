from typing import List, Dict, Any, Literal, Optional, Union
from pydantic import BaseModel, Field, field_validator, model_validator
import yaml
from pathlib import Path


DEFAULT_CONFIG_PATH = "config/curator/global.yaml"


class GlobalConfig(BaseModel):
    """Global configuration model"""
    logging_level: str = Field(default="info", description="Log level")
    temp_dir: Optional[str] = Field(default="/tmp", description="Temporary directory")
    cache_dir: Optional[str] = Field(default=".cache", description="Cache directory")
    # Ray lazy evaluation control: when force_full_snapshot=True, force count() at every step (debugging only)
    force_full_snapshot: bool = Field(
        default=False,
        description="Force full count() materialization after every operation (breaks Ray lazy evaluation; debug only)",
    )
    # Media processing acceleration backend: auto (probe MultimodalSDK, accelerate if available) / mm_sdk (force; error if probing fails) / default (always use the default implementation)
    media_backend: Literal["auto", "mm_sdk", "default"] = Field(
        default="auto",
        description="Media processing acceleration backend (see docs/MMSDK_多模态处理加速集成设计.md)",
    )


class OperationConfig(BaseModel):
    """Configuration model for a single operation"""
    name: str = Field(..., description="Operation name")
    params: Optional[Dict[str, Any]] = Field(default={}, description="Operation parameters")

    @field_validator('name', mode='before')
    @classmethod
    def validate_operation_name(cls, v):
        """Validate the operation name format"""
        if not v or not isinstance(v, str):
            raise ValueError("Operation name must be a non-empty string")
        return v


class PipelineConfig(BaseModel):
    """Pipeline configuration model"""
    operations: List[OperationConfig] = Field(default_factory=list, description="Operation list")


class DataConfig(BaseModel):
    """Data configuration model"""
    input_path: str = Field(..., description="Input path")
    output_path: str = Field(..., description="Output path")
    modalities: List[str] = Field(..., description="Data modality list")


class ResourcesConfig(BaseModel):
    """Resource configuration model"""
    gpu: Optional[int] = Field(default=0, description="GPU count")
    npu: Optional[int] = Field(default=0, description="NPU count")
    cpu: Optional[int] = Field(default=1, description="CPU count")
    memory: Optional[str] = Field(default="4GB", description="Memory size")
    # Direction 4: adaptive resource scheduling
    adaptive_scheduling: bool = Field(default=False, description="Enable adaptive resource scheduling")
    history_db_path: Optional[str] = Field(default=None, description="Resource history database path")


class PipelineConfiguration(BaseModel):
    """Complete pipeline configuration model"""
    global_config: Optional[GlobalConfig] = Field(alias="global", default=None, description="Global configuration")
    pipeline: Optional[PipelineConfig] = Field(default=None, description="Pipeline configuration (mutually exclusive with dag)")
    dag: Optional[Dict[str, Any]] = Field(default=None, description="DAG configuration (mutually exclusive with pipeline)")
    data: Optional[DataConfig] = Field(default=None, description="Data configuration")
    resources: Optional[ResourcesConfig] = Field(default=None, description="Resource configuration")

    @field_validator('dag', mode='before')
    @classmethod
    def validate_dag_not_empty(cls, v):
        """If DAG nodes are configured, entry_nodes and exit_nodes cannot be empty."""
        if v is not None:
            if not v.get('nodes'):
                raise ValueError("dag configuration must contain a nodes field")
        return v

    @model_validator(mode='after')
    def validate_pipeline_or_dag(self):
        """pipeline and dag cannot be configured at the same time; choose one."""
        if self.pipeline is not None and self.dag is not None:
            raise ValueError("pipeline and dag cannot both be configured; choose one")
        return self

    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> 'PipelineConfiguration':
        """Load configuration from a YAML file"""
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        return cls(**data)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'PipelineConfiguration':
        """Load configuration from a dictionary"""
        return cls(**config_dict)

    def to_yaml(self, yaml_path: Union[str, Path]):
        """Save configuration to a YAML file"""
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.dict(by_alias=True), f, default_flow_style=False, allow_unicode=True)

    def dict(self, *args, **kwargs):
        """Override the dict method to support aliases"""
        if 'by_alias' not in kwargs:
            kwargs['by_alias'] = True
        return super().dict(*args, **kwargs)
