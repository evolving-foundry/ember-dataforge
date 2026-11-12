"""
Core Pipeline implementation.

Supports two execution modes:
  - Linear Pipeline (MultimodalPipeline): execute Operations sequentially
  - DAG Pipeline (DAGPipeline): execute in topological order with parallel scheduling

All extension capabilities (Tracking, Checkpoint, Quality, Experiment, AdaptiveScheduling)
are injected through the PipelinePlugin interface.
"""
from typing import Any, Dict, List, Optional, Union
import logging as _logging
import ray
import uuid
import os
from ray.data import Dataset

from .operation import Operation
from .operation_registry import OperationRegistry
from base.dataset import MultimodalDataset
from .config.config import PipelineConfiguration
from ..checkpoint.checkpoint_manager import CheckpointConfig
from .dag.dag_models import DAGDefinition
from .dag.dag_executor import DAGExecutor

logger = _logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Utility functions
# ─────────────────────────────────────────────────────────────────────────────

def _parse_memory_bytes(memory_str: str) -> int:
    """Convert a memory-size string (such as '2GB' or '512MB') to bytes."""
    s = memory_str.strip().upper()
    if s.endswith("GB"):
        return int(float(s[:-2]) * 1024 ** 3)
    if s.endswith("MB"):
        return int(float(s[:-2]) * 1024 ** 2)
    if s.endswith("KB"):
        return int(float(s[:-2]) * 1024)
    return int(s)


# ─────────────────────────────────────────────────────────────────────────────
# MultimodalPipeline — linear execution Pipeline (plugin architecture)
# ─────────────────────────────────────────────────────────────────────────────

class MultimodalPipeline:
    """Multimodal processing pipeline with plugin-based extensibility.

    Extension capabilities are injected through PipelinePlugin:
      - TrackingPlugin (data tracking)
      - CheckpointPlugin (checkpoints)
      - QualityPlugin (quality assessment)
      - ExperimentPlugin (experiment management)
      - AdaptiveSchedulingPlugin (adaptive scheduling)
    """

    def __init__(
        self,
        config: Optional[Union[Dict[str, Any], str, PipelineConfiguration]] = None,
        # ── Legacy signature compatibility parameters (deprecated) ─────────────────
        enable_tracking: bool = True,
        checkpoint_config: CheckpointConfig = None,
        quality_config: Optional[Any] = None,
        experiment_tracker: Optional[Any] = None,
        experiment_id: Optional[str] = None,
        experiment_tags: Optional[Dict[str, str]] = None,
        experiment_notes: str = "",
        enable_adaptive_scheduling: bool = False,
        history_db_path: Optional[str] = None,
        # ── New signature: plugin list ──────────────────────────────────────────
        plugins: Optional[List[Any]] = None,
    ):
        self.config_model, self.config = self._parse_config(config)
        # Synchronize global.media_backend with the media backend selector (consumed by operators via get_media_backend())
        if self.config_model and self.config_model.global_config:
            from ..utils.media_backend import set_configured_media_backend
            set_configured_media_backend(self.config_model.global_config.media_backend)
        self.operations: List[Operation] = []
        self.stats: Dict[str, Any] = {}
        self.pipeline_id = str(uuid.uuid4())
        self.session_id = str(uuid.uuid4())

        # ── Plugin system ────────────────────────────────────────────────────────
        self._plugins: Dict[str, Any] = {}
        if plugins:
            for p in plugins:
                self._plugins[p.name] = p
                p.on_pipeline_init(self)
        else:
            # Legacy signature compatibility: automatically create default plugins
            self._init_default_plugins(
                enable_tracking=enable_tracking,
                checkpoint_config=checkpoint_config,
                quality_config=quality_config,
                experiment_tracker=experiment_tracker,
                experiment_id=experiment_id,
                experiment_tags=experiment_tags,
                experiment_notes=experiment_notes,
                enable_adaptive_scheduling=enable_adaptive_scheduling,
                history_db_path=history_db_path,
            )

        # ── Attributes assigned directly by the legacy signature (backward compatible) ───────────────────────
        self.enable_tracking = enable_tracking

        # ── Automatically build the operation chain from configuration (YAML-driven) ────────────────────────────────
        self._auto_build_from_config()

    @property
    def tracker(self):
        """Backward-compatible property: return the DataTracker inside the data_tracking plugin."""
        tp = self._plugins.get("data_tracking")
        if tp:
            return tp.get_tracker()
        return None

    def _init_default_plugins(
        self,
        enable_tracking: bool,
        checkpoint_config,
        quality_config,
        experiment_tracker,
        experiment_id,
        experiment_tags,
        experiment_notes,
        enable_adaptive_scheduling,
        history_db_path,
    ) -> None:
        """Build the default plugin set when plugins are not provided (legacy signature compatibility)."""
        from .pipeline_plugin import (
            TrackingPlugin,
            CheckpointPlugin,
            QualityPlugin,
            ExperimentPlugin,
            AdaptiveSchedulingPlugin,
        )

        if enable_tracking:
            self._plugins["data_tracking"] = TrackingPlugin(enabled=True)
            self._plugins["data_tracking"].on_pipeline_init(self)

        self._plugins["checkpoint"] = CheckpointPlugin(
            checkpoint_config=checkpoint_config,
            pipeline_id=self.pipeline_id,
            session_id=self.session_id,
        )
        self._plugins["checkpoint"].on_pipeline_init(self)

        if quality_config is not None:
            self._plugins["quality_assessment"] = QualityPlugin(quality_config=quality_config)
            self._plugins["quality_assessment"].on_pipeline_init(self)

        if experiment_tracker is not None and experiment_id is not None:
            self._plugins["experiment_tracking"] = ExperimentPlugin(
                experiment_tracker=experiment_tracker,
                experiment_id=experiment_id,
                experiment_tags=experiment_tags,
                experiment_notes=experiment_notes,
            )
            self._plugins["experiment_tracking"].on_pipeline_init(self)

        if enable_adaptive_scheduling:
            self._plugins["adaptive_scheduling"] = AdaptiveSchedulingPlugin(
                history_db_path=history_db_path
            )
            self._plugins["adaptive_scheduling"].on_pipeline_init(self)

    @staticmethod
    def _parse_config(
        config: Optional[Union[Dict[str, Any], str, PipelineConfiguration]],
    ):
        if isinstance(config, PipelineConfiguration):
            return config, config.dict(by_alias=True)
        if isinstance(config, dict):
            return PipelineConfiguration(**config), config
        if isinstance(config, str):
            model = PipelineConfiguration.from_yaml(config)
            return model, model.dict(by_alias=True)
        return None, {}

    def _auto_build_from_config(self) -> None:
        """Automatically build the operation chain from PipelineConfiguration (YAML configuration driven)."""
        if self.config_model is None:
            return
        # Build from pipeline.operations
        if self.config_model.pipeline and self.config_model.pipeline.operations:
            for op_cfg in self.config_model.pipeline.operations:
                self.add_operation(op_cfg.name, op_cfg.params or {})
        # Automatically inject IO operators from data.input_path/output_path
        self._auto_inject_io_operations()

    def _auto_inject_io_operations(self) -> None:
        """Automatically inject Loading/Saving operators based on config.data."""
        if self.config_model is None or self.config_model.data is None:
            return
        data_cfg = self.config_model.data
        # Automatically inject Loading (before the first non-Loading operator)
        if data_cfg.input_path and (
            not self.operations or not getattr(self.operations[0], "IS_IO_OPERATION", False)
        ):
            loader = self._auto_create_loader(data_cfg.input_path, data_cfg.modalities)
            if loader:
                self.operations.insert(0, loader)
                logger.info("[Pipeline] Auto-injected loader: %s", loader.name)
        # Automatically inject Saving (after the last non-Saving operator)
        if data_cfg.output_path and (
            not self.operations or not getattr(self.operations[-1], "IS_IO_OPERATION", False)
        ):
            saver = self._auto_create_saver(data_cfg.output_path, data_cfg.modalities)
            if saver:
                self.operations.append(saver)
                logger.info("[Pipeline] Auto-injected saver: %s", saver.name)

    def _auto_create_loader(self, input_path: str, modalities: List[str]):
        """Automatically create the corresponding Loading operator based on input_path and modalities."""
        mod = (modalities or ["text"])[0]
        op_map = {
            "text": "text_loading",
            "image": "image_loading",
            "video": "video_loading",
            "audio": "audio_loading",
        }
        op_name = op_map.get(mod, "text_loading")
        try:
            return OperationRegistry.create_operation(op_name, {"data_paths": input_path})
        except Exception:
            return None

    def _auto_create_saver(self, output_path: str, modalities: List[str]):
        """Automatically create the corresponding Saving operator based on output_path and modalities."""
        mod = (modalities or ["text"])[0]
        op_map = {
            "text": "text_saving",
            "image": "image_saving",
            "video": "video_saving",
            "audio": "audio_saving",
        }
        op_name = op_map.get(mod, "text_saving")
        try:
            return OperationRegistry.create_operation(op_name, {"output_path": output_path})
        except Exception:
            return None

    # ── Ray initialization ───────────────────────────────────────────────────

    def _init_ray(self) -> None:
        from ..utils.windows_compat import patch_windows_unicode_logging

        resources = self.config_model.resources if self.config_model else None
        if ray.is_initialized():
            self._ray_owned = False
            if resources:
                logger.warning("Ray is already initialized; cpu/gpu/memory settings in ResourcesConfig will not take effect.")
            patch_windows_unicode_logging()
            return

        memory_bytes = (
            _parse_memory_bytes(resources.memory)
            if (resources and resources.memory)
            else 2 * 1024 ** 3
        )
        ray_num_cpus = resources.cpu if (resources and resources.cpu) else None
        ray_num_gpus = resources.gpu if (resources and resources.gpu is not None) else None
        ray_num_npus = resources.npu if (resources and resources.npu is not None) else None

        custom_resources = {}
        if ray_num_npus and ray_num_npus > 0:
            custom_resources["NPU"] = ray_num_npus

        try:
            ray.init(
                ignore_reinit_error=True,
                include_dashboard=False,
                object_store_memory=memory_bytes,
                num_cpus=ray_num_cpus,
                num_gpus=ray_num_gpus,
                resources=custom_resources,
                _temp_dir=os.path.join(os.environ.get("TEMP", "/tmp"), "ray"),
                _system_config={"object_spilling_threshold": 0.8},
            )
        except Exception as e:
            logger.error("Ray initialization failed: %s", e)
            try:
                ray.init(ignore_reinit_error=True, include_dashboard=False)
            except Exception as e2:
                logger.error("Ray could not be initialized: %s", e2)
                raise RuntimeError(
                    "Ray initialization failed. Please check Ray installation and system resources."
                ) from e2

        self._ray_owned = True
        patch_windows_unicode_logging()

    # ── Operation management ─────────────────────────────────────────────────

    def add_operation(
        self, operation: Union[Operation, str], params: Optional[Dict[str, Any]] = None
    ):
        if isinstance(operation, Operation):
            op_instance = operation
        elif isinstance(operation, str):
            op_instance = OperationRegistry.create_operation(operation, params or {})
        else:
            raise ValueError(f"Invalid operation type: {type(operation)}")
        self.operations.append(op_instance)

    # ── Pipeline execution ───────────────────────────────────────────────────

    def run(self, dataset=None, resume_from_checkpoint: bool = None):
        import time as _time

        self._init_ray()

        # Plugin hook: pipeline start
        for p in self._plugins.values():
            if hasattr(p, "before_pipeline"):
                p.before_pipeline(self)

        # ── Checkpoint resume ──────────────────────────────────────────────────
        checkpoint_wrapper = self._plugins.get("checkpoint")
        if checkpoint_wrapper:
            should_resume, resumed_dataset, start_index, _ = (
                checkpoint_wrapper.get_wrapper().should_resume_from_checkpoint(
                    resume_from_checkpoint
                )
            )
        else:
            should_resume, resumed_dataset, start_index = False, None, 0

        if should_resume:
            ops_to_run = self.operations[start_index:]
            processed_dataset = resumed_dataset
        else:
            ops_to_run = self.operations
            processed_dataset = None
            start_index = 0

        # ── Dataset input handling ───────────────────────────────────────────────
        if dataset is not None:
            if isinstance(dataset, Dataset):
                processed_dataset = MultimodalDataset(dataset=dataset)
            elif isinstance(dataset, MultimodalDataset):
                processed_dataset = dataset
            else:
                raise ValueError(f"Invalid dataset type: {type(dataset)}")

        # ── Execution loop ──────────────────────────────────────────────────────
        for i, operation in enumerate(ops_to_run):
            operation_idx = start_index + i
            logger.info(
                "Executing operation %s/%s: %s",
                operation_idx + 1,
                len(self.operations),
                operation.name,
            )

            # Plugin hook: before operation
            input_dataset = processed_dataset
            for p in self._plugins.values():
                p.before_operation(self, operation, operation_idx, input_dataset)

            try:
                # Adaptive scheduling recommendation
                sched = self._plugins.get("adaptive_scheduling")
                if sched and sched.get_scheduler():
                    try:
                        res = sched.get_scheduler().recommend_resources(
                            operation, input_dataset
                        )
                        logger.info(
                            "[AdaptiveScheduler] Recommended resources: CPU=%s, GPU=%s, batch_size=%s",
                            res.get("num_cpus"),
                            res.get("num_gpus"),
                            res.get("batch_size"),
                        )
                    except Exception:
                        pass

                # Core execution
                op_start = _time.time()
                processed_dataset = operation.execute(processed_dataset)
                op_end = _time.time()
                duration = op_end - op_start

                if not isinstance(processed_dataset, MultimodalDataset):
                    raise TypeError(
                        f"Operation {operation.name} must return MultimodalDataset, "
                        f"but returned {type(processed_dataset)}"
                    )

                # Plugin hook: after operation
                for p in self._plugins.values():
                    p.after_operation(
                        self, operation, operation_idx, input_dataset,
                        processed_dataset, duration,
                    )

                # Checkpoint saving (IO operators are already skipped inside the plugin)
                if checkpoint_wrapper:
                    wrapper = checkpoint_wrapper.get_wrapper()
                    if wrapper:
                        completed = list(range(start_index, operation_idx + 1))
                        remaining = list(range(operation_idx + 1, len(self.operations)))
                        wrapper.save_checkpoint_if_needed(
                            operations=self.operations,
                            current_operation_idx=operation_idx,
                            processed_dataset=processed_dataset,
                            completed_ops_indices=completed,
                            remaining_ops_indices=remaining,
                        )

            except Exception as e:
                logger.error(
                    "Error executing operation %s: %s - %s",
                    operation_idx + 1, operation.name, str(e),
                )
                if checkpoint_wrapper:
                    wrapper = checkpoint_wrapper.get_wrapper()
                    if wrapper:
                        wrapper.save_checkpoint_on_failure(
                            operations=self.operations,
                            operation_idx=operation_idx,
                            processed_dataset=processed_dataset,
                            error=e,
                        )
                for p in self._plugins.values():
                    p.on_pipeline_failure(self, operation, operation_idx, e)
                raise e

        # ── Completion ───────────────────────────────────────────────────────────
        if checkpoint_wrapper:
            wrapper = checkpoint_wrapper.get_wrapper()
            if wrapper:
                wrapper.save_final_checkpoint(
                    operations=self.operations,
                    processed_dataset=processed_dataset,
                )

        for p in self._plugins.values():
            p.on_pipeline_complete(self, processed_dataset)

        return processed_dataset

    # ── Helper methods ───────────────────────────────────────────────────────

    def get_tracking_report(self):
        """Get the tracking report."""
        tp = self._plugins.get("data_tracking")
        if tp:
            return tp.get_manager().get_tracking_report()
        return None

    def save_checkpoint_now(self, checkpoint_name: str = None):
        cw = self._plugins.get("checkpoint")
        if cw and cw.get_wrapper():
            cw.get_wrapper().save_manual_checkpoint(checkpoint_name)

    def list_checkpoints(self):
        cw = self._plugins.get("checkpoint")
        if cw and cw.get_wrapper():
            return cw.get_wrapper().list_checkpoints()
        return []

    def cleanup(self):
        try:
            if getattr(self, "_ray_owned", False) and ray.is_initialized():
                ray.shutdown()
                self._ray_owned = False
                logger.info("Ray shutdown successfully")
        except Exception as e:
            logger.warning("Error during Ray shutdown: %s", e)

    # ── DAG factory method ───────────────────────────────────────────────────

    @classmethod
    def from_dag(cls, dag_def, enable_parallel: bool = True, **kwargs):
        """Create a DAGPipeline from a DAGDefinition (legacy signature compatibility; delegates to DAGPipeline)."""
        return DAGPipeline.from_dag(dag_def, enable_parallel=enable_parallel, **kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# DAGPipeline — a DAG Pipeline that inherits all plugin capabilities
# ─────────────────────────────────────────────────────────────────────────────

class DAGPipeline:
    """Pipeline that supports DAG topological execution.

    Inherits all MultimodalPipeline plugin capabilities (Tracking/Checkpoint/Quality, etc.)
    and replaces only the execution layer with DAGExecutor.
    """

    def __init__(
        self,
        dag_def: DAGDefinition,
        enable_parallel: bool = True,
        plugins: Optional[List[Any]] = None,
        # ── Legacy signature compatibility parameters ──────────────────────
        enable_tracking: bool = True,
        checkpoint_config: CheckpointConfig = None,
        quality_config: Optional[Any] = None,
        experiment_tracker: Optional[Any] = None,
        experiment_id: Optional[str] = None,
        experiment_tags: Optional[Dict[str, str]] = None,
        experiment_notes: str = "",
        enable_adaptive_scheduling: bool = False,
        history_db_path: Optional[str] = None,
    ):
        self.dag_def = dag_def
        self.dag_executor = DAGExecutor(dag_def, enable_parallel=enable_parallel)
        self.pipeline_id = str(uuid.uuid4())
        self.session_id = str(uuid.uuid4())
        self.config_model = None  # DAGPipeline does not use PipelineConfiguration

        # Plugin system
        self._plugins: Dict[str, Any] = {}
        if plugins:
            for p in plugins:
                self._plugins[p.name] = p
                p.on_pipeline_init(self)
        else:
            self._init_default_plugins(
                enable_tracking=enable_tracking,
                checkpoint_config=checkpoint_config,
                quality_config=quality_config,
                experiment_tracker=experiment_tracker,
                experiment_id=experiment_id,
                experiment_tags=experiment_tags,
                experiment_notes=experiment_notes,
                enable_adaptive_scheduling=enable_adaptive_scheduling,
                history_db_path=history_db_path,
            )

    def _init_default_plugins(
        self,
        enable_tracking: bool,
        checkpoint_config,
        quality_config,
        experiment_tracker,
        experiment_id,
        experiment_tags,
        experiment_notes,
        enable_adaptive_scheduling,
        history_db_path,
    ) -> None:
        from .pipeline_plugin import (
            TrackingPlugin,
            CheckpointPlugin,
            QualityPlugin,
            ExperimentPlugin,
            AdaptiveSchedulingPlugin,
        )

        if enable_tracking:
            self._plugins["data_tracking"] = TrackingPlugin(enabled=True)
            self._plugins["data_tracking"].on_pipeline_init(self)

        self._plugins["checkpoint"] = CheckpointPlugin(
            checkpoint_config=checkpoint_config,
            pipeline_id=self.pipeline_id,
            session_id=self.session_id,
        )
        self._plugins["checkpoint"].on_pipeline_init(self)

        if quality_config is not None:
            self._plugins["quality_assessment"] = QualityPlugin(quality_config=quality_config)
            self._plugins["quality_assessment"].on_pipeline_init(self)

        if experiment_tracker is not None and experiment_id is not None:
            self._plugins["experiment_tracking"] = ExperimentPlugin(
                experiment_tracker=experiment_tracker,
                experiment_id=experiment_id,
                experiment_tags=experiment_tags,
                experiment_notes=experiment_notes,
            )
            self._plugins["experiment_tracking"].on_pipeline_init(self)

        if enable_adaptive_scheduling:
            self._plugins["adaptive_scheduling"] = AdaptiveSchedulingPlugin(
                history_db_path=history_db_path
            )
            self._plugins["adaptive_scheduling"].on_pipeline_init(self)

    def _init_ray(self) -> None:
        from ..utils.windows_compat import patch_windows_unicode_logging
        if ray.is_initialized():
            self._ray_owned = False
            patch_windows_unicode_logging()
            return
        try:
            ray.init(ignore_reinit_error=True, include_dashboard=False)
        except Exception:
            pass
        self._ray_owned = ray.is_initialized()
        patch_windows_unicode_logging()

    @classmethod
    def from_yaml(cls, yaml_path: str, **kwargs) -> "DAGPipeline":
        dag_def = DAGDefinition.from_yaml(yaml_path)
        return cls(dag_def=dag_def, **kwargs)

    @classmethod
    def from_dag(cls, dag_def, enable_parallel: bool = True, **kwargs) -> "DAGPipeline":
        if isinstance(dag_def, str):
            dag_def = DAGDefinition.from_yaml(dag_def)
        return cls(dag_def=dag_def, enable_parallel=enable_parallel, **kwargs)

    def run(
        self,
        initial_datasets: Optional[Dict[str, MultimodalDataset]] = None,
    ) -> Dict[str, MultimodalDataset]:
        """Execute the DAG and return {exit_node_id: MultimodalDataset}."""
        self._init_ray()

        # Plugin hook: pipeline start
        for p in self._plugins.values():
            if hasattr(p, "before_pipeline"):
                p.before_pipeline(self)

        # Start an experiment Run
        exp = self._plugins.get("experiment_tracking")
        if exp:
            try:
                exp.before_operation(self, None, -1, None)
            except Exception:
                pass

        try:
            outputs = self.dag_executor.execute(initial_datasets)

            # Plugin hook: exit nodes completed
            for node_id, ds in outputs.items():
                for p in self._plugins.values():
                    if hasattr(p, "after_dag_node"):
                        p.after_dag_node(self, node_id, ds)
                    p.on_pipeline_complete(self, ds)

            return outputs
        except Exception as e:
            for p in self._plugins.values():
                p.on_pipeline_failure(self, None, -1, e)
            raise

    def visualize(self) -> str:
        return self.dag_executor.visualize()

    def cleanup(self):
        try:
            if getattr(self, "_ray_owned", False) and ray.is_initialized():
                ray.shutdown()
                self._ray_owned = False
                logger.info("Ray shutdown successfully")
        except Exception:
            pass
