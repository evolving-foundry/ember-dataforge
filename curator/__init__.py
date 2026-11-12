"""Curator: a multimodal data processing framework."""

from .core import *

# Import utility modules
from .utils.io.data_loader_factory import DataLoaderFactory
from .utils.io.data_saver_factory import DataSaverFactory

# Import the tracking module
from . import tracking

# Import the checkpoint module
from . import checkpoint

# Import extension modules
from . import quality
from . import experiments
from . import autopipeline

__all__ = [
    # Core classes
    'MultimodalDataset',
    'MultimodalPipeline',
    'DAGPipeline',
    'Operation',
    'OperationRegistry',
    'register_operation',
    # Dataset helper classes
    'DataValidationConfig',
    'ValidationResult',
    # Configuration classes
    'PipelineConfiguration',
    'GlobalConfig',
    'OperationConfig',
    'PipelineConfig',
    'DataConfig',
    'ResourcesConfig',
    'DEFAULT_CONFIG_PATH',
    # Utility classes
    'DataLoaderFactory',
    'DataSaverFactory',
    # Data quality assessment
    'quality',
    # Experiment management
    'experiments',
    # AutoPipeline
    'autopipeline',
    # Plugin interfaces
    'pipeline_plugin',
]

# Lazily initialized operation registry singleton. It is initialized on the
# first create_operation() call, so register_default_operations() is not needed.

__version__ = "0.2.0"
__author__ = "Multimodal Processing Framework"

# ─────────────────────────────────────────────────────────────────────────────
# Global logging configuration
# ─────────────────────────────────────────────────────────────────────────────

import logging
import os
import sys
from pathlib import Path


def _resolve_config_path():
    configured = os.environ.get("MDP_CONFIG")
    candidates = [Path(configured).expanduser()] if configured else []
    candidates.extend(
        [
            Path(DEFAULT_CONFIG_PATH),
            Path(sys.prefix) / DEFAULT_CONFIG_PATH,
            Path(__file__).resolve().parent.parent / DEFAULT_CONFIG_PATH,
        ]
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _configure_logging():
    """Configure logging from GlobalConfig.logging_level once for the root logger."""
    level_str = "INFO"
    try:
        config_path = _resolve_config_path()
        if config_path is not None:
            with open(config_path, encoding="utf-8") as f:
                import yaml
                cfg = yaml.safe_load(f)
                level_str = cfg.get("global", {}).get("logging_level", "INFO")
    except Exception:
        pass

    logging.basicConfig(
        level=getattr(logging, level_str.upper(), logging.INFO),
        format="[%(asctime)s] %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )


_configure_logging()
