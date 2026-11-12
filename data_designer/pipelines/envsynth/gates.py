"""Pre-freeze gates: anchor replay, distribution, expert review (design §3.5)."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from data_designer.pipelines.envsynth import grounding
from data_designer.pipelines.envsynth.package import EnvPackageBundle
from data_designer.verifiers import (
    VerifierDispatchError,
    run_verifier,
    validate_spec,
)


GATE_REPORT_SCHEMA_VERSION = "env-gate-report-v1"


class GateError(RuntimeError):
    """Gate inputs are malformed (as opposed to a gate verdict of failed)."""


def _state_projection(bundle: EnvPackageBundle):
    model = bundle.model
    if model is None or len(model.state_collections) != 1:
        return None
    collection = model.state_collections[0]
    state_field = collection.state_field
    if not isinstance(state_field, str) or not state_field:
        state_field = next(
            (
                name
                for name, spec in collection.fields.items()
                if isinstance(spec, dict) and spec.get("enum")
            ),
            "",
        )
    if not state_field:
        return None
    return collection, state_field


def _expected_anchor_artifacts(
    anchor: Dict[str, Any], bundle: EnvPackageBundle
) -> Optional[tuple[Dict[str, Any], Dict[str, Any]]]:
    gold = anchor.get("gold_solution")
    negative = anchor.get("negative_solution")
    if isinstance(gold, dict) and isinstance(negative, dict):
        return dict(gold), dict(negative)

    expected = anchor.get("expected")
    negative = anchor.get("negative")
    if not isinstance(expected, dict) or not isinstance(negative, dict):
        return None
    state = _state_projection(bundle)
    if state is None:
        return None
    collection, state_field = state
    object_id = expected.get(collection.key)
    status = expected.get(state_field)
    negative_status = negative.get(state_field)
    if not all(
        isinstance(value, str) and value
        for value in (object_id, status, negative_status)
    ):
        return None
    for action in (bundle.model.actions if bundle.model else ()):
        for transition in action.transitions:
            if transition.from_state != status:
                continue
            return (
                {
                    collection.key: object_id,
                    "action": action.name,
                    "next_status": transition.to_state,
                },
                {
                    collection.key: object_id,
                    "action": action.name,
                    "next_status": negative_status,
                },
            )
    return (
        {collection.key: object_id, state_field: status},
        {collection.key: object_id, state_field: negative_status},
    )


def _anchor_gate(bundle: EnvPackageBundle) -> Dict[str, Any]:
    checks: List[Dict[str, Any]] = []
    failed: List[str] = []
    anchor_ids = {str(anchor.get("anchor_id")) for anchor in bundle.anchors}
    anchors_by_id = {
        str(anchor.get("anchor_id")): anchor for anchor in bundle.anchors
    }
    if not anchor_ids:
        return {
            "passed": False,
            "checks": checks,
            "failedChecks": ["anchor_gate:no_anchors"],
        }
    for task in bundle.tasks:
        anchor_id = str(task.get("anchor_id") or "")
        if not anchor_id:
            continue
        try:
            validate_spec(task.get("verifier_spec"))
            positive = run_verifier(task["verifier_spec"], task.get("gold_solution"))
            negative = run_verifier(task["verifier_spec"], task.get("negative_solution"))
        except (VerifierDispatchError, KeyError, TypeError) as error:
            checks.append(
                {"taskId": task.get("task_id"), "anchorId": anchor_id, "error": str(error)}
            )
            failed.append("anchor_gate:{}:{}".format(task.get("task_id"), error))
            continue
        positive_passed = bool(positive.get("passed"))
        negative_passed = bool(negative.get("passed"))
        anchor = anchors_by_id.get(anchor_id) or {}
        artifacts = _expected_anchor_artifacts(anchor, bundle)
        gold_solution = task.get("gold_solution")
        negative_solution = task.get("negative_solution")
        expected_matches = artifacts is not None and gold_solution == artifacts[0]
        negative_matches = artifacts is not None and negative_solution == artifacts[1]
        check = {
            "taskId": task.get("task_id"),
            "anchorId": anchor_id,
            "positivePassed": positive_passed,
            "negativePassed": negative_passed,
            "expectedMatches": expected_matches,
            "negativeMatches": negative_matches,
        }
        checks.append(check)
        if not positive_passed:
            failed.append("anchor_gate:{}:positive_failed".format(task.get("task_id")))
        if negative_passed:
            failed.append("anchor_gate:{}:negative_passed".format(task.get("task_id")))
        if not expected_matches:
            failed.append(
                "anchor_gate:{}:expected_mismatch".format(task.get("task_id"))
            )
        if not negative_matches:
            failed.append(
                "anchor_gate:{}:negative_mismatch".format(task.get("task_id"))
            )
    seen_anchor_ids = {str(check.get("anchorId")) for check in checks}
    for anchor_id in sorted(anchor_ids - seen_anchor_ids):
        failed.append("anchor_gate:{}:missing_task".format(anchor_id))
    return {"passed": not failed, "checks": checks, "failedChecks": failed}


def _mean_vector(vectors: Sequence[Sequence[float]]) -> List[float]:
    if not vectors:
        return []
    dim = len(vectors[0])
    sums = [0.0] * dim
    for vector in vectors:
        for index, value in enumerate(vector[:dim]):
            sums[index] += float(value)
    return [value / len(vectors) for value in sums]


def _distribution_gate(
    bundle: EnvPackageBundle, *, quality_threshold: float
) -> Dict[str, Any]:
    failed: List[str] = []
    schema_errors: List[str] = []
    schema_checks = 0
    for task in bundle.tasks:
        schema_checks += 2
        try:
            validate_spec(task.get("verifier_spec"))
        except VerifierDispatchError as error:
            schema_errors.append("{}: {}".format(task.get("task_id"), error))
        if not isinstance(task.get("instruction"), str) or not task["instruction"].strip():
            schema_errors.append("{}: empty instruction".format(task.get("task_id")))
    schema_pass_rate = (
        1.0
        if not schema_checks
        else round(max(0, schema_checks - len(schema_errors)) / schema_checks, 4)
    )
    if schema_errors:
        failed.append("distribution_gate:schema")

    source_text = " ".join(str(anchor.get("text") or "") for anchor in bundle.anchors)
    glossary = []
    rules = bundle.files.get("rules.yaml", b"").decode("utf-8", errors="ignore")
    prompt = bundle.files.get("simulator_prompt.md", b"").decode("utf-8", errors="ignore")
    generated_text = " ".join(
        [str(task.get("instruction") or "") for task in bundle.tasks]
        + [rules, prompt]
    )
    for term in bundle.manifest.get("glossary") or []:
        if term in source_text or term in generated_text:
            glossary.append(term)
    covered = [term for term in glossary if term in generated_text]
    term_coverage = 1.0 if not glossary else round(len(covered) / len(glossary), 4)
    term_threshold = max(0.5, min(0.95, float(quality_threshold)))
    if term_coverage < term_threshold:
        failed.append("distribution_gate:term_coverage")

    backend = grounding.CharNgramEmbeddingBackend()
    generated_vectors = backend.embed(
        [str(task.get("instruction") or "") for task in bundle.tasks]
    )
    anchor_vectors = backend.embed([str(anchor.get("text") or "") for anchor in bundle.anchors])
    similarity = grounding.cosine_similarity(
        _mean_vector(generated_vectors), _mean_vector(anchor_vectors)
    )
    similarity = round(max(-1.0, min(1.0, similarity)), 6)
    similarity_threshold = max(0.30, float(quality_threshold) - 0.25)
    if similarity < similarity_threshold:
        failed.append("distribution_gate:embedding_similarity")

    return {
        "passed": not failed,
        "schemaPassRate": schema_pass_rate,
        "schemaErrors": schema_errors,
        "termCoverage": term_coverage,
        "termThreshold": term_threshold,
        "distributionSimilarity": similarity,
        "similarityThreshold": round(similarity_threshold, 6),
        "generatedTaskCount": len(bundle.tasks),
        "anchorCount": len(bundle.anchors),
        "failedChecks": failed,
    }


def _load_expert_evidence(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        raise GateError("evidencePath is required when expertReview.mode=required")
    target = Path(path)
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise GateError(f"Expert-review evidence is unreadable: {error}") from error
    if not isinstance(payload, dict) or payload.get("schemaVersion") != 1:
        raise GateError("expert evidence must be {schemaVersion:1, ...}")
    for key in ("reviewer", "reviewedAt"):
        if not isinstance(payload.get(key), str) or not payload[key].strip():
            raise GateError(f"expert evidence is missing {key}")
    if not isinstance(payload.get("sampleRate"), (int, float)) or isinstance(
        payload.get("sampleRate"), bool
    ):
        raise GateError("expert evidence.sampleRate must be numeric")
    if not isinstance(payload.get("verdicts"), list):
        raise GateError("expert evidence.verdicts must be an array")
    return payload


def _expert_gate(
    bundle: EnvPackageBundle,
    *,
    expert_review: Dict[str, Any],
    evidence_path: Optional[str],
) -> Dict[str, Any]:
    mode = expert_review.get("mode", "waived")
    if mode == "waived":
        return {
            "passed": True,
            "status": "waived",
            "sampleRate": float(expert_review.get("sampleRate") or 0.0),
            "note": "Explicitly waived in the first release; marked in the manifest",
        }
    if mode != "required":
        raise GateError("expertReview.mode must be required or waived")
    evidence = _load_expert_evidence(evidence_path)
    requested_rate = float(expert_review.get("sampleRate") or 0.0)
    if float(evidence["sampleRate"]) < requested_rate:
        raise GateError("Expert-review sampleRate is below the requested value")
    task_ids = {str(task.get("task_id")) for task in bundle.tasks}
    verdicts = evidence["verdicts"]
    sampled = []
    failed: List[str] = []
    for entry in verdicts:
        if not isinstance(entry, dict) or str(entry.get("taskId")) not in task_ids:
            raise GateError("expert evidence contains an unknown taskId")
        if entry.get("verdict") not in ("pass", "fail"):
            raise GateError("expert verdict must be pass/fail")
        sampled.append(str(entry["taskId"]))
        if entry.get("verdict") != "pass":
            failed.append(str(entry["taskId"]))
    required_count = int(math.ceil(requested_rate * len(bundle.tasks)))
    if len(set(sampled)) < required_count:
        raise GateError(
            "Expert-review sample is insufficient: required {}, actual {}".format(required_count, len(set(sampled)))
        )
    if failed:
        return {
            "passed": False,
            "status": "failed",
            "sampleRate": float(evidence["sampleRate"]),
            "reviewer": evidence["reviewer"],
            "failedTaskIds": sorted(failed),
        }
    return {
        "passed": True,
        "status": "passed",
        "sampleRate": float(evidence["sampleRate"]),
        "reviewer": evidence["reviewer"],
        "reviewedAt": evidence["reviewedAt"],
        "reviewedTaskCount": len(set(sampled)),
    }



def run_gate(
    kind: str,
    bundle: EnvPackageBundle,
    *,
    quality_threshold: float = 0.6,
    expert_review: Optional[Dict[str, Any]] = None,
    expert_evidence_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Run one gate for the agent self_check tool."""
    if kind == "anchor_replay":
        result = _anchor_gate(bundle)
    elif kind == "distribution":
        result = _distribution_gate(bundle, quality_threshold=quality_threshold)
    elif kind == "expert_review":
        result = _expert_gate(
            bundle,
            expert_review=dict(expert_review or {"mode": "waived", "sampleRate": 0.0}),
            evidence_path=expert_evidence_path,
        )
    else:
        raise GateError("Unknown gate: {!r}".format(kind))
    return {"gate": kind, **result}


def run_gates(
    bundle: EnvPackageBundle,
    *,
    quality_threshold: float = 0.6,
    expert_review: Optional[Dict[str, Any]] = None,
    expert_evidence_path: Optional[str] = None,
) -> Dict[str, Any]:
    expert_review = dict(expert_review or {"mode": "waived", "sampleRate": 0.0})
    anchor = run_gate(
        "anchor_replay",
        bundle,
        quality_threshold=quality_threshold,
        expert_review=expert_review,
        expert_evidence_path=expert_evidence_path,
    )
    distribution = run_gate(
        "distribution",
        bundle,
        quality_threshold=quality_threshold,
        expert_review=expert_review,
        expert_evidence_path=expert_evidence_path,
    )
    expert = run_gate(
        "expert_review",
        bundle,
        quality_threshold=quality_threshold,
        expert_review=expert_review,
        expert_evidence_path=expert_evidence_path,
    )
    failed_checks = list(anchor["failedChecks"]) + list(distribution["failedChecks"])
    if not expert["passed"]:
        failed_checks.append("expert_review:failed")
    gates = {
        "anchor_replay": "passed" if anchor["passed"] else "failed",
        "distribution": "passed" if distribution["passed"] else "failed",
        "expert_review": expert["status"],
    }
    return {
        "schemaVersion": GATE_REPORT_SCHEMA_VERSION,
        "passed": not failed_checks,
        "gates": gates,
        "failedChecks": failed_checks,
        "anchorReplay": anchor,
        "distribution": distribution,
        "expertReview": expert,
    }


__all__ = ["GATE_REPORT_SCHEMA_VERSION", "GateError", "run_gate", "run_gates"]
