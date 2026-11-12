"""Immutable suite pins, publication status, and regression comparability."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

from data_designer.contracts.provenance import content_digest as fingerprint_of
from data_designer.storage.catalog import ArtifactCatalog

from .contracts import (
    BenchmarkError,
    assert_public_item,
    canonical_sha256,
    file_sha256,
    require_nonempty,
    require_sha,
)
from .runtime import runtime_digests

SUITE_KIND = "benchmark_suite"
STATUS_KIND = "benchmark_publication"
OFFICIAL_EVIDENCE_FIELDS = frozenset(
    {
        "source_release",
        "license",
        "native_environment",
        "reference",
        "negative",
        "isolation",
    }
)


def _checked_taskset(store: ArtifactCatalog, pin: Dict[str, Any]) -> Dict[str, Any]:
    if (
        not isinstance(pin, dict)
        or not isinstance(pin.get("ref_id"), str)
        or not pin["ref_id"].strip()
        or type(pin.get("version")) is not int
        or pin["version"] < 1
    ):
        raise BenchmarkError("taskset pin requires ref_id and positive version")
    art = store.get_artifact("taskset", pin["ref_id"], pin["version"])
    if art is None:
        raise BenchmarkError(f"missing taskset {pin['ref_id']}/v{pin['version']}")
    payload = art["payload"]
    items = payload.get("taskset")
    if not isinstance(items, list) or not items:
        raise BenchmarkError("taskset must be non-empty")
    actual = fingerprint_of(items)
    if actual != pin.get("fingerprint") or actual != payload.get("taskset_fingerprint"):
        raise BenchmarkError("taskset fingerprint drift")
    if canonical_sha256(items) != payload.get("taskset_sha256"):
        raise BenchmarkError("taskset full SHA-256 drift")
    for item in items:
        assert_public_item(item)
    family = items[0]["benchmark"]["family"]
    if any(item["benchmark"]["family"] != family for item in items):
        raise BenchmarkError("taskset mixes families")
    if payload.get("family") != family:
        raise BenchmarkError("taskset family metadata mismatch")
    actual_runtime = runtime_digests(items, payload.get("runtime_by_task"))
    if any(payload.get(key) != digest for key, digest in actual_runtime.items()):
        raise BenchmarkError("taskset runtime aggregate SHA-256 drift")
    return payload


def create_suite(
    store: ArtifactCatalog,
    ref_id: str,
    pins: List[Dict[str, Any]],
    *,
    repeats: int = 1,
    seed: int = 0,
    timeout_seconds: float = 600,
    weights: Optional[Dict[str, float]] = None,
    resource_limits: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    if not ref_id or not isinstance(ref_id, str):
        raise BenchmarkError("suite ref_id is required")
    if (
        type(repeats) is not int
        or repeats < 1
        or type(seed) is not int
        or seed < 0
        or seed >= 2**63
        or type(timeout_seconds) not in (int, float)
        or not math.isfinite(timeout_seconds)
        or timeout_seconds <= 0
    ):
        raise BenchmarkError("repeats/seed/timeout configuration invalid")
    if not isinstance(pins, list) or not pins:
        raise BenchmarkError("suite needs at least one taskset")
    families: Dict[str, Dict[str, Any]] = {}
    ids: set[str] = set()
    locked_pins: List[Dict[str, Any]] = []
    suite_split: str | None = None
    for pin in pins:
        payload = _checked_taskset(store, pin)
        family = payload["family"]
        if suite_split is None:
            suite_split = payload["split"]
        elif payload["split"] != suite_split:
            raise BenchmarkError("suite tasksets must share one split")
        if family in families:
            raise BenchmarkError(f"duplicate suite family {family}")
        families[family] = payload
        for item in payload["taskset"]:
            if item["task_id"] in ids:
                raise BenchmarkError(f"duplicate task ID {item['task_id']}")
            ids.add(item["task_id"])
        locked_pins.append(
            {
                "family": family,
                "ref_id": pin["ref_id"],
                "version": pin["version"],
                "fingerprint": pin["fingerprint"],
                "taskset_sha256": require_sha(
                    payload.get("taskset_sha256"), "taskset_sha256"
                ),
                "source_sha256": require_sha(
                    payload.get("source_sha256"), "source_sha256"
                ),
                "augment_config_sha256": require_sha(
                    payload.get("augment_config_sha256"), "augment_config_sha256"
                ),
                "environment_sha256": require_sha(
                    payload.get("environment_sha256"), "environment_sha256"
                ),
                "verifier_sha256": require_sha(
                    payload.get("verifier_sha256"), "verifier_sha256"
                ),
                "asset_manifest_sha256": require_sha(
                    payload.get("asset_manifest_sha256"), "asset_manifest_sha256"
                ),
            }
        )
    expected = set(families)
    if weights is None:
        weights = {family: 1 / len(expected) for family in expected}
    if (
        not isinstance(weights, dict)
        or set(weights) != expected
        or any(
            type(v) not in (float, int) or not math.isfinite(v) or v <= 0
            for v in weights.values()
        )
        or abs(sum(weights.values()) - 1) > 1e-9
    ):
        raise BenchmarkError("weights must cover all suite families and sum to one")
    if resource_limits is not None:
        if not isinstance(resource_limits, dict) or set(resource_limits) != expected:
            raise BenchmarkError("resource limits must cover every suite family")
        for family, limit in resource_limits.items():
            required_fields = {
                "max_tasks", "timeout_seconds", "wall_seconds",
                "max_concurrency", "retry_policy",
            }
            if (
                not isinstance(limit, dict)
                or not required_fields <= set(limit)
                or not set(limit) <= required_fields | {"max_cost_usd"}
            ):
                raise BenchmarkError(f"resource limit fields invalid: {family}")
            if (
                type(limit["max_tasks"]) is not int
                or not 1 <= limit["max_tasks"] <= len(families[family]["taskset"])
                or type(limit["timeout_seconds"]) not in (int, float)
                or not math.isfinite(limit["timeout_seconds"])
                or limit["timeout_seconds"] <= 0
                or type(limit["wall_seconds"]) not in (int, float)
                or not math.isfinite(limit["wall_seconds"])
                or limit["wall_seconds"] <= 0
                or type(limit["max_concurrency"]) is not int
                or limit["max_concurrency"] != 1
                or limit["retry_policy"] != "none"
            ):
                raise BenchmarkError(f"resource limit values invalid: {family}")
            if "max_cost_usd" in limit and (
                type(limit["max_cost_usd"]) not in (int, float)
                or not math.isfinite(limit["max_cost_usd"])
                or limit["max_cost_usd"] < 0
            ):
                raise BenchmarkError(f"resource cost limit invalid: {family}")
    body = {
        "schema_version": "benchmark-suite-v1",
        "split": suite_split,
        "tasksets": locked_pins,
        "weights": weights,
        "repeats": repeats,
        "seed": seed,
        "timeout_seconds": timeout_seconds,
    }
    if resource_limits is not None:
        body["resource_limits"] = resource_limits
    sha = canonical_sha256(body)
    version = store.next_artifact_version(SUITE_KIND, ref_id)
    payload = {**body, "ref_id": ref_id, "version": version, "suite_sha256": sha}
    store.put_artifact(SUITE_KIND, ref_id, version, payload, fingerprint=sha)
    set_publication_status(store, ref_id, version, "candidate", "awaiting review")
    return payload


def verify_audit_records(payload: Dict[str, Any]) -> None:
    """Audit-coverage recheck for one taskset payload: every item must carry a
    legal native audit record (reference completed+passed, negative completed+
    not passed)."""
    items = payload["taskset"]
    report = payload.get("build_report")
    audits = report.get("audit") if isinstance(report, dict) else None
    if (
        not isinstance(audits, list)
        or len(audits) != len(items)
        or any(not isinstance(audit, dict) for audit in audits)
        or {audit.get("task_id") for audit in audits} != {
            item["task_id"] for item in items
        }
    ):
        raise BenchmarkError("suite taskset native audit record incomplete")
    for audit in audits:
        if (
            not isinstance(audit, dict)
            or not isinstance(audit.get("reference"), dict)
            or audit["reference"].get("status") != "completed"
            or audit["reference"].get("passed") is not True
            or not isinstance(audit.get("negative"), dict)
            or audit["negative"].get("status") != "completed"
            or audit["negative"].get("passed") is not False
        ):
            raise BenchmarkError("suite taskset native audit did not pass")


def _review_evidence(
    store: ArtifactCatalog, suite: Dict[str, Any], evidence: Any
) -> Dict[str, str]:
    if not isinstance(evidence, dict):
        raise BenchmarkError("suite approval requires review evidence")
    path = Path(require_nonempty(evidence.get("path"), "review evidence path"))
    sha = require_sha(evidence.get("sha256"), "review evidence SHA-256")
    if not path.is_absolute() or not path.is_file() or file_sha256(path) != sha:
        raise BenchmarkError("suite review evidence unavailable or drifted")
    try:
        review = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except (OSError, ValueError) as error:
        raise BenchmarkError(f"suite review evidence invalid: {error}") from error
    if not isinstance(review, dict) or (
        review.get("schema_version") != "benchmark-suite-review-v1"
        or review.get("suite_sha256") != suite["suite_sha256"]
        or review.get("tasksets") != suite["tasksets"]
        or not isinstance(review.get("reviewer"), str)
        or not review["reviewer"].strip()
        or not isinstance(review.get("evidence_refs"), list)
        or not review["evidence_refs"]
        or not all(
            isinstance(ref, str) and ref.strip()
            for ref in review["evidence_refs"]
        )
        or any(
            review.get(check) is not True
            for check in (
                "native_positive_negative_verified",
                "private_boundary_verified",
                "license_checked",
                "environment_reset_verified",
            )
        )
    ):
        raise BenchmarkError("suite review evidence does not approve frozen pins")
    scope = review.get("scope")
    if scope not in (None, "synthetic_fixture", "owned_native", "official_native"):
        raise BenchmarkError("suite review scope invalid")
    if scope in ("owned_native", "official_native"):
        pins = review.get("evidence_pins")
        if not isinstance(pins, dict) or set(pins) != OFFICIAL_EVIDENCE_FIELDS:
            raise BenchmarkError("native review requires six evidence pins")
        project_root = Path(__file__).resolve().parents[3]
        if path.resolve().is_relative_to(project_root):
            raise BenchmarkError("native review must stay outside the repository")
        evidence_paths: set[Path] = set()
        for label, pin in pins.items():
            if not isinstance(pin, dict):
                raise BenchmarkError(f"native {label} evidence pin invalid")
            evidence_path = Path(
                require_nonempty(pin.get("path"), f"{label} evidence path")
            )
            evidence_sha = require_sha(pin.get("sha256"), f"{label} evidence SHA")
            if (
                not evidence_path.is_absolute()
                or not evidence_path.is_file()
                or evidence_path.stat().st_size == 0
                or file_sha256(evidence_path) != evidence_sha
            ):
                raise BenchmarkError(
                    f"native {label} evidence unavailable or drifted"
                )
            resolved = evidence_path.resolve()
            if resolved.is_relative_to(project_root) or resolved in evidence_paths:
                raise BenchmarkError(
                    f"native {label} evidence must be distinct and external"
                )
            evidence_paths.add(resolved)
    for pin in suite["tasksets"]:
        payload = _checked_taskset(store, pin)
        verify_audit_records(payload)
    return {"path": str(path), "sha256": sha}


def set_publication_status(
    store: ArtifactCatalog,
    ref_id: str,
    version: int,
    status: str,
    reason: str,
    *,
    review_evidence: Optional[Dict[str, str]] = None,
) -> None:
    if (
        status not in ("candidate", "approved", "quarantined")
        or not isinstance(reason, str)
        or not reason.strip()
    ):
        raise BenchmarkError("invalid publication status or empty reason")
    suite_artifact = store.get_artifact(SUITE_KIND, ref_id, version)
    if suite_artifact is None:
        raise BenchmarkError("suite does not exist")
    key = f"{ref_id}/v{version}"
    prior = store.get_artifact(STATUS_KIND, key)
    previous_status = prior["payload"]["status"] if prior is not None else None
    allowed = {
        None: {"candidate"},
        "candidate": {"approved", "quarantined"},
        "approved": {"quarantined"},
        "quarantined": set(),
    }
    if status not in allowed.get(previous_status, set()):
        raise BenchmarkError("invalid suite publication transition")
    locked_review = (
        _review_evidence(store, suite_artifact["payload"], review_evidence)
        if status == "approved" else None
    )
    revision = store.next_artifact_version(STATUS_KIND, key)
    record = {
        "suite_ref_id": ref_id,
        "suite_version": version,
        "status": status,
        "reason": reason,
        "review_evidence": locked_review,
    }
    store.put_artifact(STATUS_KIND, key, revision, record)


def load_suite(
    store: ArtifactCatalog,
    ref_id: str,
    version: int,
    sha256: str,
    *,
    require_approved: bool = True,
) -> Dict[str, Any]:
    art = store.get_artifact(SUITE_KIND, ref_id, version)
    if art is None:
        raise BenchmarkError("suite not found")
    suite = art["payload"]
    body = {
        key: suite[key]
        for key in (
            "schema_version",
            "split",
            "tasksets",
            "weights",
            "repeats",
            "timeout_seconds",
        )
    }
    if "seed" in suite:
        body["seed"] = suite["seed"]
    if "resource_limits" in suite:
        body["resource_limits"] = suite["resource_limits"]
    if canonical_sha256(body) != sha256 or suite.get("suite_sha256") != sha256:
        raise BenchmarkError("suite SHA-256 drift")
    if require_approved:
        status = store.get_artifact(STATUS_KIND, f"{ref_id}/v{version}")
        if status is None or status["payload"]["status"] != "approved":
            raise BenchmarkError("suite is not approved for new evaluation")
        _review_evidence(store, suite, status["payload"].get("review_evidence"))
    for pin in suite["tasksets"]:
        payload = _checked_taskset(store, pin)
        if payload["split"] != suite["split"]:
            raise BenchmarkError("suite child split drift")
        for key in (
            "taskset_sha256",
            "source_sha256",
            "augment_config_sha256",
            "environment_sha256",
            "verifier_sha256",
            "asset_manifest_sha256",
        ):
            if pin[key] != payload[key]:
                raise BenchmarkError(f"suite child {key} drift")
    return suite


def assert_comparable(left: Dict[str, Any], right: Dict[str, Any]) -> None:
    if left.get("suite_sha256") != right.get("suite_sha256") or left.get(
        "tasksets"
    ) != right.get("tasksets"):
        raise BenchmarkError("candidate and baseline use different frozen suites")


def assert_comparable_evaluations(
    left: Dict[str, Any], right: Dict[str, Any]
) -> None:
    """Require the same frozen suite and exact ordered task selection."""
    if not isinstance(left, dict) or not isinstance(right, dict) or any(
        left.get(key) != right.get(key)
        for key in ("suite_sha256", "suite_ref", "tasksets", "split", "tasks")
    ):
        raise BenchmarkError(
            "candidate and baseline evaluations use different suites or tasks"
        )
    if not isinstance(left.get("tasks"), list) or not left["tasks"]:
        raise BenchmarkError("comparable evaluation requires a non-empty task list")


def read_suite(
    store: ArtifactCatalog,
    ref_id: str,
    version: int,
    *,
    require_approved: bool = False,
) -> Dict[str, Any]:
    """Load a suite using the SHA-256 recorded in its catalog payload, so the
    caller does not have to carry the digest. Full pin re-verification still
    runs via ``load_suite``."""
    art = store.get_artifact(SUITE_KIND, ref_id, version)
    if art is None:
        raise BenchmarkError("suite not found")
    sha = require_sha(art["payload"].get("suite_sha256"), "suite_sha256")
    return load_suite(store, ref_id, version, sha, require_approved=require_approved)


def _taskset_overview(store: ArtifactCatalog, pin: Dict[str, Any]) -> Dict[str, Any]:
    payload = store.get_artifact("taskset", pin["ref_id"], pin["version"])["payload"]
    items = payload["taskset"]
    distribution: Dict[str, int] = {}
    for item in items:
        difficulty = item["difficulty"]
        distribution[difficulty] = distribution.get(difficulty, 0) + 1
    return {
        "family": pin["family"],
        "ref_id": pin["ref_id"],
        "version": pin["version"],
        "fingerprint": pin["fingerprint"],
        "task_count": len(items),
        "difficulty_distribution": dict(sorted(distribution.items())),
    }


def _publication_overview(
    store: ArtifactCatalog, ref_id: str, version: int
) -> Dict[str, Any]:
    record = store.get_artifact(STATUS_KIND, f"{ref_id}/v{version}")
    if record is None:
        return {"status": "candidate", "reason": "awaiting review"}
    payload = record["payload"]
    return {"status": payload["status"], "reason": payload["reason"]}


def suite_overview(
    store: ArtifactCatalog, suite: Dict[str, Any]
) -> Dict[str, Any]:
    """Read-only show body (suite/publication/tasksets/review_evidence)."""
    publication_record = store.get_artifact(
        STATUS_KIND, f"{suite['ref_id']}/v{suite['version']}"
    )
    review_evidence = (
        publication_record["payload"].get("review_evidence")
        if publication_record is not None
        else None
    )
    return {
        "suite": {
            "ref_id": suite["ref_id"],
            "version": suite["version"],
            "suite_sha256": suite["suite_sha256"],
            "split": suite["split"],
            "weights": suite["weights"],
            "repeats": suite["repeats"],
            "timeout_seconds": suite["timeout_seconds"],
        },
        "publication": _publication_overview(
            store, suite["ref_id"], suite["version"]
        ),
        "tasksets": [
            _taskset_overview(store, pin) for pin in suite["tasksets"]
        ],
        "review_evidence": review_evidence,
    }


def show_suite(
    store: ArtifactCatalog,
    ref_id: str,
    version: int,
    *,
    require_approved: bool = False,
) -> Dict[str, Any]:
    return suite_overview(
        store, read_suite(store, ref_id, version, require_approved=require_approved)
    )


def verify_suite(store: ArtifactCatalog, ref_id: str, version: int) -> Dict[str, Any]:
    """Full-chain recheck: suite SHA-256, per-taskset double pins, public face,
    runtime digests, audit coverage, and publication/evidence consistency.

    Quarantined is a legal terminal state — verification succeeds but surfaces
    ``publication.status = "quarantined"``."""
    suite = read_suite(store, ref_id, version, require_approved=False)
    for pin in suite["tasksets"]:
        verify_audit_records(_checked_taskset(store, pin))
    publication = _publication_overview(store, ref_id, version)
    if publication["status"] == "approved":
        record = store.get_artifact(STATUS_KIND, f"{ref_id}/v{version}")
        _review_evidence(store, suite, record["payload"].get("review_evidence"))
    return {**suite_overview(store, suite), "verified": True}
