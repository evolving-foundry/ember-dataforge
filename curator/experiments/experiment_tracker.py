from __future__ import annotations

import hashlib
import json
import os
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

import logging

logger = logging.getLogger(__name__)

from .experiment_models import Experiment, PipelineRun, RunStatus


class ExperimentTracker:
    """Pipeline version control and experiment manager.

    Data storage layout (local filesystem):
        .experiments/
            experiments.json               # Experiment list index
            {experiment_id}/
                metadata.json              # Experiment object
                {run_id}/
                    run.json               # PipelineRun object
                    pipeline_config.yaml   # Configuration snapshot
                    tracking_report.json   # DataTracker output
                    quality_report.json    # Quality assessment output (if enabled)
    """

    BASE_DIR = ".experiments"

    def __init__(self, base_dir: Optional[str] = None):
        self.base_dir = base_dir or self.BASE_DIR
        os.makedirs(self.base_dir, exist_ok=True)
        self._index_path = os.path.join(self.base_dir, "experiments.json")

    # ------------------------------------------------------------------
    # Experiment management
    # ------------------------------------------------------------------

    def create_experiment(
        self,
        name: str,
        description: str = "",
        primary_metric: str = "overall_quality_score",
        tags: Optional[Dict[str, str]] = None,
    ) -> Experiment:
        experiment = Experiment(
            experiment_id=str(uuid.uuid4()),
            name=name,
            description=description,
            created_at=datetime.now(),
            tags=tags or {},
            primary_metric=primary_metric,
        )
        exp_dir = os.path.join(self.base_dir, experiment.experiment_id)
        os.makedirs(exp_dir, exist_ok=True)
        self._save_json(
            os.path.join(exp_dir, "metadata.json"),
            experiment.model_dump(mode="json"),
        )
        self._update_index(experiment)
        return experiment

    def get_experiment(self, experiment_id: str) -> Optional[Experiment]:
        path = os.path.join(self.base_dir, experiment_id, "metadata.json")
        data = self._load_json(path)
        if data is None:
            return None
        return Experiment.model_validate(data)

    def list_experiments(self) -> List[Experiment]:
        index = self._load_json(self._index_path) or []
        experiments = []
        for exp_id in index:
            exp = self.get_experiment(exp_id)
            if exp:
                experiments.append(exp)
        return experiments

    # ------------------------------------------------------------------
    # Run lifecycle
    # ------------------------------------------------------------------

    def start_run(
        self,
        experiment_id: str,
        pipeline_config: Optional[Dict[str, Any]] = None,
        tags: Optional[Dict[str, str]] = None,
        notes: str = "",
    ) -> PipelineRun:
        config = pipeline_config or {}
        config_hash = self._hash_config(config)

        run = PipelineRun(
            run_id=str(uuid.uuid4()),
            experiment_id=experiment_id,
            pipeline_config=config,
            config_hash=config_hash,
            status=RunStatus.RUNNING,
            started_at=datetime.now(),
            tags=tags or {},
            notes=notes,
            git_commit=self._get_git_commit(),
        )

        run_dir = self._run_dir(experiment_id, run.run_id)
        os.makedirs(run_dir, exist_ok=True)
        self._save_json(os.path.join(run_dir, "run.json"), run.model_dump(mode="json"))

        # Update experiment run_ids
        exp = self.get_experiment(experiment_id)
        if exp:
            exp.run_ids.append(run.run_id)
            self._save_json(
                os.path.join(self.base_dir, experiment_id, "metadata.json"),
                exp.model_dump(mode="json"),
            )

        # Warn on duplicate config
        if self._find_duplicate_run(experiment_id, config_hash, run.run_id):
            logger.warning(
                "[ExperimentTracker] Warning: found a previous run with the same configuration (hash=%s);"
                "check whether this experiment was duplicated.",
                config_hash[:8]
            )

        return run

    def log_metrics(self, run_id: str, experiment_id: str, metrics: Dict[str, float]) -> None:
        run = self._load_run(experiment_id, run_id)
        if run:
            run.metrics.update(metrics)
            self._save_run(run)

    def log_quality_scores(
        self, run_id: str, experiment_id: str, quality_scores: Dict[str, float]
    ) -> None:
        run = self._load_run(experiment_id, run_id)
        if run:
            run.quality_scores.update(quality_scores)
            self._save_run(run)

    def log_artifact(
        self, run_id: str, experiment_id: str, name: str, path: str
    ) -> None:
        run = self._load_run(experiment_id, run_id)
        if run:
            run.artifacts[name] = path
            self._save_run(run)

    def complete_run(
        self,
        run_id: str,
        experiment_id: str,
        status: RunStatus = RunStatus.COMPLETED,
        tracking_report: Optional[Any] = None,
        quality_report: Optional[Any] = None,
    ) -> Optional[PipelineRun]:
        run = self._load_run(experiment_id, run_id)
        if run is None:
            return None

        run.status = status
        run.completed_at = datetime.now()
        elapsed = (run.completed_at - run.started_at).total_seconds() * 1000
        run.total_execution_time_ms = elapsed

        # Persist tracking report
        if tracking_report is not None:
            run_dir = self._run_dir(experiment_id, run_id)
            try:
                report_data = (
                    tracking_report.model_dump(mode="json")
                    if hasattr(tracking_report, "model_dump")
                    else tracking_report
                )
                self._save_json(
                    os.path.join(run_dir, "tracking_report.json"), report_data
                )
                # Extract operation times
                ops = report_data.get("operations", [])
                for op in ops:
                    op_name = op.get("operation_name", "unknown")
                    start = op.get("start_time")
                    end = op.get("end_time")
                    if start and end:
                        try:
                            from datetime import datetime as dt
                            s = dt.fromisoformat(str(start))
                            e = dt.fromisoformat(str(end))
                            run.operation_times[op_name] = (
                                (e - s).total_seconds() * 1000
                            )
                        except Exception:
                            pass
            except Exception:
                pass

        # Persist quality report
        if quality_report is not None:
            run_dir = self._run_dir(experiment_id, run_id)
            try:
                qr_data = (
                    quality_report.model_dump(mode="json")
                    if hasattr(quality_report, "model_dump")
                    else quality_report
                )
                self._save_json(
                    os.path.join(run_dir, "quality_report.json"), qr_data
                )
                # Auto-populate quality_scores
                for dim, ds in qr_data.get("dimension_scores", {}).items():
                    key = dim if isinstance(dim, str) else str(dim)
                    run.quality_scores[key] = ds.get("score", 0.0)
                run.quality_scores["overall_quality_score"] = qr_data.get(
                    "overall_score", 0.0
                )
            except Exception:
                pass

        self._save_run(run)
        self._update_best_run(experiment_id)
        return run

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def compare_runs(
        self,
        run_ids: List[str],
        experiment_id: str,
        metrics: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Compare multiple Runs and return a metrics comparison table, configuration differences, and rankings."""
        runs = [self._load_run(experiment_id, rid) for rid in run_ids]
        runs = [r for r in runs if r is not None]
        if not runs:
            return {}

        all_metrics = metrics or list(
            {k for r in runs for k in list(r.metrics) + list(r.quality_scores)}
        )

        comparison_table: Dict[str, Dict[str, Any]] = {}
        for run in runs:
            combined = {**run.metrics, **run.quality_scores}
            comparison_table[run.run_id] = {
                m: combined.get(m, None) for m in all_metrics
            }

        config_diff = self._diff_configs(runs)

        # Ranking by first metric
        ranking = []
        if all_metrics:
            primary = all_metrics[0]
            sortable = [
                (r.run_id, comparison_table[r.run_id].get(primary) or 0.0)
                for r in runs
            ]
            sortable.sort(key=lambda x: x[1], reverse=True)
            ranking = [{"run_id": rid, primary: score} for rid, score in sortable]

        return {
            "comparison_table": comparison_table,
            "config_diff": config_diff,
            "ranking": ranking,
        }

    def get_experiment_summary(self, experiment_id: str) -> Dict[str, Any]:
        exp = self.get_experiment(experiment_id)
        if exp is None:
            return {}
        runs = [
            self._load_run(experiment_id, rid) for rid in exp.run_ids
        ]
        runs = [r for r in runs if r is not None]
        completed = [r for r in runs if r.status == RunStatus.COMPLETED]
        best_run = (
            self._load_run(experiment_id, exp.best_run_id)
            if exp.best_run_id
            else None
        )
        return {
            "experiment_id": experiment_id,
            "name": exp.name,
            "total_runs": len(runs),
            "completed_runs": len(completed),
            "best_run_id": exp.best_run_id,
            "best_run_metrics": (
                {**best_run.metrics, **best_run.quality_scores} if best_run else {}
            ),
            "runs_summary": [
                {
                    "run_id": r.run_id,
                    "status": r.status,
                    "started_at": str(r.started_at),
                    "metrics": r.metrics,
                    "quality_scores": r.quality_scores,
                }
                for r in runs
            ],
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_dir(self, experiment_id: str, run_id: str) -> str:
        return os.path.join(self.base_dir, experiment_id, run_id)

    def _load_run(self, experiment_id: str, run_id: str) -> Optional[PipelineRun]:
        path = os.path.join(self._run_dir(experiment_id, run_id), "run.json")
        data = self._load_json(path)
        if data is None:
            return None
        return PipelineRun.model_validate(data)

    def _save_run(self, run: PipelineRun) -> None:
        run_dir = self._run_dir(run.experiment_id, run.run_id)
        os.makedirs(run_dir, exist_ok=True)
        self._save_json(
            os.path.join(run_dir, "run.json"), run.model_dump(mode="json")
        )

    def _update_best_run(self, experiment_id: str) -> None:
        exp = self.get_experiment(experiment_id)
        if not exp:
            return
        runs = [
            self._load_run(experiment_id, rid) for rid in exp.run_ids
        ]
        completed = [
            r for r in runs if r is not None and r.status == RunStatus.COMPLETED
        ]
        if not completed:
            return
        metric = exp.primary_metric
        best = max(
            completed,
            key=lambda r: r.quality_scores.get(metric)
            or r.metrics.get(metric)
            or 0.0,
        )
        exp.best_run_id = best.run_id
        self._save_json(
            os.path.join(self.base_dir, experiment_id, "metadata.json"),
            exp.model_dump(mode="json"),
        )

    def _diff_configs(self, runs: List[PipelineRun]) -> Dict[str, Any]:
        if len(runs) < 2:
            return {}
        diff: Dict[str, Any] = {}
        base = runs[0].pipeline_config
        for run in runs[1:]:
            for key in set(list(base.keys()) + list(run.pipeline_config.keys())):
                bv = base.get(key)
                av = run.pipeline_config.get(key)
                if bv != av:
                    diff.setdefault(key, {})[run.run_id] = av
        return diff

    def _find_duplicate_run(
        self, experiment_id: str, config_hash: str, current_run_id: str
    ) -> Optional[str]:
        exp = self.get_experiment(experiment_id)
        if not exp:
            return None
        for rid in exp.run_ids:
            if rid == current_run_id:
                continue
            run = self._load_run(experiment_id, rid)
            if run and run.config_hash == config_hash:
                return rid
        return None

    @staticmethod
    def _hash_config(config: Dict[str, Any]) -> str:
        try:
            raw = json.dumps(config, sort_keys=True, default=str)
            return hashlib.md5(raw.encode()).hexdigest()
        except Exception:
            return ""

    @staticmethod
    def _get_git_commit() -> Optional[str]:
        try:
            import subprocess
            result = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True,
                text=True,
                timeout=3,
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass
        return None

    def _update_index(self, experiment: Experiment) -> None:
        index = self._load_json(self._index_path) or []
        if experiment.experiment_id not in index:
            index.append(experiment.experiment_id)
        self._save_json(self._index_path, index)

    @staticmethod
    def _save_json(path: str, data: Any) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)

    @staticmethod
    def _load_json(path: str) -> Optional[Any]:
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
