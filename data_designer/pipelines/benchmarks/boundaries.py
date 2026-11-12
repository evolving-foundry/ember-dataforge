"""Trusted case validation extracted from the retired whole-run evaluator."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, Optional
from .contracts import BenchmarkError, file_sha256, require_sha

CAPABILITIES = {
    "tau": {"dialogue", "tools"},
    "appworld": {"api", "code"},
    "webarena": {"browser"},
    "osworld": {"desktop"},
    "gaia": {"browser", "attachments"},
    "swe": {"terminal", "code"},
    "deepswe": {"terminal", "code"},
    "workbuddy": {"files"},
    "terminal": {"terminal", "code"},
    "reasoning_qa": {"single_turn"},
    "knowledge_qa": {"single_turn"},
    "code_gen": {"single_turn", "code"},
    "health_qa": {"single_turn"},
}

# verifier_spec.kind whitelist for the four HF single-turn families (M4 manifest validation):
# health_qa/simpleqa use judge (llm_rubric), deterministic benchmarks use unit_test,
# knowledge_qa accepts either (simpleqa uses llm_rubric, the rest use unit_test).
_SINGLE_TURN_FAMILIES = ("reasoning_qa", "knowledge_qa", "code_gen", "health_qa")
_SINGLE_TURN_ALLOWED_KINDS = {
    "reasoning_qa": ("unit_test",),
    "knowledge_qa": ("unit_test", "llm_rubric"),
    "code_gen": ("unit_test",),
    "health_qa": ("llm_rubric",),
}


class EvaluationFailure(BenchmarkError):
    """Categorized failure retained in the trusted run ledger."""

    def __init__(
        self, category: str, message: str, *,
        family: str | None = None, task_id: str | None = None,
    ):
        super().__init__(message)
        self.category = category
        self.family = family
        self.task_id = task_id


def _within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _assert_private_boundary(
    candidate_root: Path, store_path: str, payload: Dict[str, Any],
    runtime: Dict[str, Any], task_id: str, family: str,
) -> None:
    trusted_dir = Path(payload["trusted_manifest_dir"])
    if _within(candidate_root, trusted_dir) or _within(trusted_dir, candidate_root):
        raise EvaluationFailure(
            "private_boundary", "candidate root overlaps trusted manifests",
            family=family, task_id=task_id,
        )
    paths = [
        Path(store_path),
        Path(runtime["environment"]["path"]),
        Path(runtime["verifier"]["path"]),
        Path(runtime["verifier"]["launcher"]["path"]),
        *[Path(asset["path"]) for asset in runtime["assets"]],
    ]
    if any(_within(path, candidate_root) for path in paths):
        raise EvaluationFailure(
            "private_boundary", "candidate root contains trusted runtime data",
            family=family, task_id=task_id,
        )


def _reject_candidate_root_refs(
    value: Any, root_text: str, family: str, task_id: str,
) -> None:
    if isinstance(value, str) and root_text in value:
        raise EvaluationFailure(
            "private_boundary",
            f"{family} verifier_spec must not reference candidateRoot",
            family=family, task_id=task_id,
        )
    if isinstance(value, dict):
        for child in value.values():
            _reject_candidate_root_refs(child, root_text, family, task_id)
    elif isinstance(value, list):
        for child in value:
            _reject_candidate_root_refs(child, root_text, family, task_id)


def _workbuddy_spec_boundary(
    spec: Dict[str, Any], task_id: str, candidate_root: Path,
) -> None:
    payload = spec.get("payload")
    if spec.get("kind") == "wb_composite":
        artifacts = payload.get("artifacts") if isinstance(payload, dict) else None
        relatives = [
            artifact.get("path")
            for artifact in (artifacts or [])
            if isinstance(artifact, dict)
        ]
        if not relatives or any(
            not isinstance(relative, str) or not relative.strip()
            for relative in relatives
        ):
            raise BenchmarkError(
                f"workbuddy wb_composite spec lacks artifact paths: {task_id}"
            )
    else:
        artifact = payload.get("artifact") if isinstance(payload, dict) else None
        relative = artifact.get("path") if isinstance(artifact, dict) else None
        if not isinstance(relative, str) or not relative.strip():
            raise BenchmarkError(
                f"workbuddy verifier_spec lacks a deliverable path: {task_id}"
            )
        relatives = [relative]
    _reject_candidate_root_refs(spec, str(candidate_root), "workbuddy", task_id)
    for relative in relatives:
        if Path(relative).is_absolute():
            raise EvaluationFailure(
                "private_boundary",
                "workbuddy deliverable path must be relative",
                family="workbuddy", task_id=task_id,
            )
        deliverable = candidate_root / relative
        if ".." in Path(relative).parts or not _within(deliverable, candidate_root):
            raise EvaluationFailure(
                "private_boundary",
                "workbuddy deliverable path must resolve inside candidateRoot",
                family="workbuddy", task_id=task_id,
            )


def _swe_spec_boundary(
    spec: Dict[str, Any], task_id: str, candidate_root: Path,
) -> None:
    payload = spec.get("payload")
    relative = payload.get("deliverable") if isinstance(payload, dict) else None
    if not isinstance(relative, str) or not relative.strip():
        raise BenchmarkError(
            f"swe verifier_spec lacks a deliverable path: {task_id}"
        )
    _reject_candidate_root_refs(spec, str(candidate_root), "swe", task_id)
    if Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise EvaluationFailure(
            "private_boundary",
            "swe deliverable path must be a clean relative path",
            family="swe", task_id=task_id,
        )
    if not _within(candidate_root / relative, candidate_root):
        raise EvaluationFailure(
            "private_boundary",
            "swe deliverable path must resolve inside candidateRoot",
            family="swe", task_id=task_id,
        )


def _deepswe_spec_boundary(
    spec: Dict[str, Any], task_id: str, candidate_root: Path,
) -> None:
    payload = spec.get("payload")
    relative = payload.get("deliverable") if isinstance(payload, dict) else None
    if not isinstance(relative, str) or not relative.strip():
        raise BenchmarkError(
            f"deepswe verifier_spec lacks a deliverable path: {task_id}"
        )
    _reject_candidate_root_refs(spec, str(candidate_root), "deepswe", task_id)
    if Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise EvaluationFailure(
            "private_boundary",
            "deepswe deliverable path must be a clean relative path",
            family="deepswe", task_id=task_id,
        )
    if not _within(candidate_root / relative, candidate_root):
        raise EvaluationFailure(
            "private_boundary",
            "deepswe deliverable path must resolve inside candidateRoot",
            family="deepswe", task_id=task_id,
        )


def _osworld_spec_boundary(
    spec: Dict[str, Any], task_id: str, candidate_root: Path,
) -> None:
    payload = spec.get("payload")
    artifact = payload.get("artifact") if isinstance(payload, dict) else None
    relative = artifact.get("path") if isinstance(artifact, dict) else None
    if not isinstance(relative, str) or not relative.strip():
        raise BenchmarkError(
            f"osworld verifier_spec lacks a deliverable path: {task_id}"
        )
    _reject_candidate_root_refs(spec, str(candidate_root), "osworld", task_id)
    if Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise EvaluationFailure(
            "private_boundary",
            "osworld deliverable path must be a clean relative path",
            family="osworld", task_id=task_id,
        )
    if not _within(candidate_root / relative, candidate_root):
        raise EvaluationFailure(
            "private_boundary",
            "osworld deliverable path must resolve inside candidateRoot",
            family="osworld", task_id=task_id,
        )


def _terminal_spec_boundary(
    spec: Dict[str, Any], task_id: str, candidate_root: Path,
) -> None:
    payload = spec.get("payload")
    artifact = payload.get("artifact") if isinstance(payload, dict) else None
    relative = artifact.get("path") if isinstance(artifact, dict) else None
    if not isinstance(relative, str) or not relative.strip():
        raise BenchmarkError(
            f"terminal verifier_spec lacks a deliverable path: {task_id}"
        )
    _reject_candidate_root_refs(spec, str(candidate_root), "terminal", task_id)
    if Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise EvaluationFailure(
            "private_boundary",
            "terminal deliverable path must be a clean relative path",
            family="terminal", task_id=task_id,
        )
    if not _within(candidate_root / relative, candidate_root):
        raise EvaluationFailure(
            "private_boundary",
            "terminal deliverable path must resolve inside candidateRoot",
            family="terminal", task_id=task_id,
        )


def _tau_spec_boundary(
    spec: Dict[str, Any], task_id: str, candidate_root: Path,
) -> None:
    payload = spec.get("payload")
    artifact = payload.get("artifact") if isinstance(payload, dict) else None
    relative = artifact.get("path") if isinstance(artifact, dict) else None
    if not isinstance(relative, str) or not relative.strip():
        raise BenchmarkError(
            f"tau verifier_spec lacks a deliverable path: {task_id}"
        )
    _reject_candidate_root_refs(spec, str(candidate_root), "tau", task_id)
    if Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise EvaluationFailure(
            "private_boundary",
            "tau deliverable path must be a clean relative path",
            family="tau", task_id=task_id,
        )
    if not _within(candidate_root / relative, candidate_root):
        raise EvaluationFailure(
            "private_boundary",
            "tau deliverable path must resolve inside candidateRoot",
            family="tau", task_id=task_id,
        )


def _verify_manifest(
    path: Path, task_id: str, expected_sha: str, snapshot_cache: Dict[str, str],
    candidate_root: Path,
) -> Optional[Dict[str, Any]]:
    if file_sha256(path) != expected_sha:
        raise BenchmarkError(f"trusted case manifest drift: {task_id}")
    try:
        manifest = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except (OSError, ValueError) as error:
        raise BenchmarkError(f"trusted case manifest invalid: {task_id}") from error
    if not isinstance(manifest, dict) or manifest.get("task_id") != task_id:
        raise BenchmarkError(f"trusted case manifest identity drift: {task_id}")
    if manifest.get("family") == "workbuddy":
        private = manifest.get("private")
        spec = private.get("verifier_spec") if isinstance(private, dict) else None
        if not isinstance(spec, dict):
            raise BenchmarkError(
                f"trusted case manifest lacks verifier_spec: {task_id}"
            )
        _workbuddy_spec_boundary(spec, task_id, candidate_root)
    if manifest.get("family") == "swe":
        private = manifest.get("private")
        spec = private.get("verifier_spec") if isinstance(private, dict) else None
        if not isinstance(spec, dict) or spec.get("kind") != "swe_offline_patch":
            raise BenchmarkError(
                "swe trusted case manifest lacks a swe_offline_patch "
                f"verifier_spec: {task_id}"
            )
        _swe_spec_boundary(spec, task_id, candidate_root)
    if manifest.get("family") == "deepswe":
        private = manifest.get("private")
        spec = private.get("verifier_spec") if isinstance(private, dict) else None
        if not isinstance(spec, dict) or spec.get("kind") != "deepswe_offline_patch":
            raise BenchmarkError(
                "deepswe trusted case manifest lacks a deepswe_offline_patch "
                f"verifier_spec: {task_id}"
            )
        _deepswe_spec_boundary(spec, task_id, candidate_root)
    if manifest.get("family") == "osworld":
        private = manifest.get("private")
        spec = private.get("verifier_spec") if isinstance(private, dict) else None
        if (
            not isinstance(spec, dict)
            or spec.get("kind") not in ("unit_test", "artifact_compare")
        ):
            raise BenchmarkError(
                "osworld trusted case manifest lacks a unit_test/"
                f"artifact_compare verifier_spec: {task_id}"
            )
        _osworld_spec_boundary(spec, task_id, candidate_root)
    if manifest.get("family") == "tau":
        private = manifest.get("private")
        spec = private.get("verifier_spec") if isinstance(private, dict) else None
        if (
            not isinstance(spec, dict)
            or spec.get("kind") not in ("unit_test", "artifact_compare")
        ):
            raise BenchmarkError(
                "tau trusted case manifest lacks a unit_test/"
                f"artifact_compare verifier_spec: {task_id}"
            )
        _tau_spec_boundary(spec, task_id, candidate_root)
    if manifest.get("family") == "terminal":
        private = manifest.get("private")
        spec = private.get("verifier_spec") if isinstance(private, dict) else None
        if (
            not isinstance(spec, dict)
            or spec.get("kind") not in ("unit_test", "artifact_compare")
        ):
            raise BenchmarkError(
                "terminal trusted case manifest lacks a unit_test/"
                f"artifact_compare verifier_spec: {task_id}"
            )
        source_case = manifest.get("source_case")
        metadata = (
            source_case.get("terminal") if isinstance(source_case, dict) else None
        )
        if not isinstance(metadata, dict) or (
            metadata.get("expressibility") != "expressible"
        ):
            raise BenchmarkError(
                f"terminal trusted case is not an expressible derivation: {task_id}"
            )
        _terminal_spec_boundary(spec, task_id, candidate_root)
    single_turn_spec: Optional[Dict[str, Any]] = None
    if manifest.get("family") in _SINGLE_TURN_FAMILIES:
        family_name = manifest["family"]
        private = manifest.get("private")
        spec = private.get("verifier_spec") if isinstance(private, dict) else None
        allowed = _SINGLE_TURN_ALLOWED_KINDS[family_name]
        if not isinstance(spec, dict) or spec.get("kind") not in allowed:
            raise BenchmarkError(
                f"{family_name} trusted case manifest lacks a "
                f"{'/'.join(allowed)} verifier_spec: {task_id}"
            )
        payload = spec.get("payload")
        if not isinstance(payload, dict):
            raise BenchmarkError(
                f"{family_name} verifier_spec lacks a payload: {task_id}"
            )
        if spec["kind"] == "llm_rubric":
            rubric = payload.get("rubric")
            if not isinstance(rubric, str) or not rubric.strip():
                raise BenchmarkError(
                    f"{family_name} llm_rubric payload lacks rubric text: {task_id}"
                )
            threshold = payload.get("pass_threshold")
            if (
                isinstance(threshold, bool)
                or not isinstance(threshold, (int, float))
                or not 0.0 < float(threshold) <= 1.0
            ):
                raise BenchmarkError(
                    f"{family_name} llm_rubric payload lacks a valid "
                    f"pass_threshold: {task_id}"
                )
        artifact = payload.get("artifact")
        relative = artifact.get("path") if isinstance(artifact, dict) else None
        if not isinstance(relative, str) or not relative.strip():
            raise BenchmarkError(
                f"{family_name} verifier_spec lacks a deliverable path: {task_id}"
            )
        _reject_candidate_root_refs(spec, str(candidate_root), family_name, task_id)
        if Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise EvaluationFailure(
                "private_boundary",
                f"{family_name} deliverable path must be a clean relative path",
                family=family_name, task_id=task_id,
            )
        if not _within(candidate_root / relative, candidate_root):
            raise EvaluationFailure(
                "private_boundary",
                f"{family_name} deliverable path must resolve inside candidateRoot",
                family=family_name, task_id=task_id,
            )
        single_turn_spec = spec
    snapshot = manifest.get("source_snapshot")
    if snapshot is None:
        return
    if not isinstance(snapshot, dict):
        raise BenchmarkError(f"source snapshot pin invalid: {task_id}")
    if not isinstance(snapshot.get("path"), str) or not snapshot["path"].strip():
        raise BenchmarkError(f"source snapshot path invalid: {task_id}")
    snapshot_path = Path(snapshot["path"])
    if _within(snapshot_path, candidate_root):
        raise EvaluationFailure(
            "private_boundary", "candidate root contains a trusted source snapshot",
            family=manifest.get("family"), task_id=task_id,
        )
    snapshot_sha = require_sha(snapshot.get("sha256"), "source snapshot SHA-256")
    if not snapshot_path.is_absolute() or not snapshot_path.is_file():
        raise BenchmarkError(f"source snapshot unavailable: {task_id}")
    key = str(snapshot_path)
    if key not in snapshot_cache:
        snapshot_cache[key] = file_sha256(snapshot_path)
    if snapshot_cache[key] != snapshot_sha:
        raise BenchmarkError(f"source snapshot drift: {task_id}")
    if manifest.get("family") == "osworld":
        from data_designer.pipelines.benchmarks.families.osworld import (
            verify_osworld_asset_snapshot,
        )

        asset_key = "osworld-assets:" + json.dumps(
            {
                "root": snapshot["asset_root"],
                "inventory": snapshot["files"]["asset_inventory"],
            },
            sort_keys=True,
        )
        if asset_key not in snapshot_cache:
            verify_osworld_asset_snapshot(snapshot)
            snapshot_cache[asset_key] = "verified"
    return single_turn_spec


