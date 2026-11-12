"""Snapshot import, semantic augmentation, native audit, and taskset freeze."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List

from data_designer.contracts.provenance import content_digest as fingerprint_of
from data_designer.storage.catalog import ArtifactCatalog

from .families.registry import get_family
from .augment import materialize_json_state_variant
from .contracts import (
    BenchmarkError,
    canonical_sha256,
    file_sha256,
    make_public_item,
    normalized_text_sha256,
    require_family,
    require_nonempty,
    require_sha,
    require_split,
)
from .native import run_case
from .quality import assess_taskset
from .runtime import lock_runtime, runtime_digests

# The operation names define an adapter contract. Upstream environment mutations
# are supplied by an authorized snapshot and must pass that runner's native audit.
AUGMENTATIONS = {
    "tau": {"policy_boundary", "confirmation", "tool_fault", "collaboration"},
    "appworld": {"cross_app", "temporal_state", "side_effect", "recovery"},
    "webarena": {"multi_site", "navigation_recovery", "network_fault"},
    "osworld": {"cross_app", "file_artifact", "state_recovery"},
    "gaia": {"multi_tool", "attachment", "temporal", "scenario"},
    "swe": {"issue_fix", "cross_file", "regression"},
    "deepswe": {"issue_fix", "cross_file", "regression"},
    "workbuddy": {"mutation", "taxonomy", "failure_probe", "native_ingest"},
    "terminal": {"setup_fault", "constraint_tightening", "failure_probe"},
    "reasoning_qa": {
        "numeric_perturbation", "constraint_tightening", "language_transfer",
        "distractor_injection",
    },
    "knowledge_qa": {
        "option_permutation", "distractor_replacement", "paraphrase",
        "alias_expansion",
    },
    "code_gen": {
        "signature_variation", "edge_case_tightening", "docstring_perturbation",
    },
    "health_qa": {"rubric_perturbation", "scenario_variation"},
}


def _failure_trace(value: Any) -> Dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, dict) or value.get("sanitized") is not True:
        raise BenchmarkError("failure trace requires sanitization attestation")
    if value.get("split") != "development":
        raise BenchmarkError("only development failure traces may seed augmentation")
    trace_path = Path(require_nonempty(value.get("path"), "failure_trace.path"))
    trace_sha = require_sha(value.get("sha256"), "failure_trace.sha256")
    if not trace_path.is_absolute() or not trace_path.is_file():
        raise BenchmarkError("failure trace snapshot unavailable")
    if file_sha256(trace_path) != trace_sha:
        raise BenchmarkError("failure trace SHA-256 drift")
    return {
        "trace_ref": require_nonempty(value.get("trace_ref"), "trace_ref"),
        "path": str(trace_path),
        "sha256": trace_sha,
        "split": "development",
        "sanitized": True,
        "lesson_ref": value.get("lesson_ref"),
        "cluster_ref": value.get("cluster_ref"),
    }


def _load_json(path: str) -> Dict[str, Any]:
    try:
        value = json.loads(
            Path(path).read_text(encoding="utf-8"),
            parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)),
        )
    except (OSError, ValueError) as error:
        raise BenchmarkError(f"cannot read JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise BenchmarkError(f"JSON {path} must be an object")
    return value


def _augment(
    source: Dict[str, Any], variant: Dict[str, Any], family: str, release: str
) -> Dict[str, Any]:
    if not isinstance(source, dict) or not isinstance(variant, dict):
        raise BenchmarkError("source and variant must be objects")
    operation = variant.get("operation")
    if operation not in AUGMENTATIONS[family]:
        raise BenchmarkError(
            f"unsupported semantic augmentation {operation!r} for {family}"
        )
    source_id = require_nonempty(source.get("source_task_id"), "source_task_id")
    if variant.get("source_task_id") != source_id:
        raise BenchmarkError("variant source_task_id mismatch")
    variant_id = require_nonempty(variant.get("variant_id"), "variant_id")
    public = variant.get("public")
    private = variant.get("private")
    if not isinstance(public, dict) or not isinstance(private, dict):
        raise BenchmarkError("variant must provide public and private objects")
    for key in ("initial_state_sha256", "expected_state_sha256"):
        require_sha(private.get(key), f"private.{key}")
        require_sha(source.get(key), f"source.{key}")
    if (
        private["initial_state_sha256"] == source["initial_state_sha256"]
        or private["expected_state_sha256"] == source["expected_state_sha256"]
    ):
        raise BenchmarkError("augmentation must change initial and expected state")
    task_id = f"{family}:{release}:{source_id}:{variant_id}"
    fields = {
        "task_id": task_id,
        "family": family,
        "source_release": release,
        "source_task_id": source_id,
        "variant_id": variant_id,
        "task_text": public.get("task_text"),
        "tools_available": public.get("tools_available"),
        "success_criteria": public.get("success_criteria"),
        "difficulty": public.get("difficulty"),
    }
    # Source and variant metadata are intentionally retained only on the trusted
    # side. The item is constructed from an allowlist, never a source dict copy.
    return {
        "public": fields,
        "private": private,
        "operation": operation,
        "source_ref": source_id,
        "audit": variant.get("audit"),
    }


def build_benchmark_taskset(
    store: ArtifactCatalog,
    source_path: str,
    config_path: str,
    trusted_dir: str,
    ref_id: str,
    *,
    quality_threshold: float = 0.6,
    expected_family: str | None = None,
    expected_release: str | None = None,
    expected_split: str | None = None,
) -> Dict[str, Any]:
    if (
        type(quality_threshold) not in (int, float)
        or not math.isfinite(quality_threshold)
        or not 0 <= quality_threshold <= 1
    ):
        raise BenchmarkError("quality_threshold must be in [0,1]")
    source = _load_json(source_path)
    config = _load_json(config_path)
    if (
        source.get("schema_version") != "benchmark-source-v1"
        or config.get("schema_version") != "benchmark-augment-v1"
    ):
        raise BenchmarkError("wrong source or augmentation schema")
    family = require_family(source.get("family"))
    rules = get_family(family)
    release = require_nonempty(source.get("release"), "release")
    split = require_split(source.get("split"))
    if (
        (expected_family is not None and family != expected_family)
        or (expected_release is not None and release != expected_release)
        or (expected_split is not None and split != expected_split)
    ):
        raise BenchmarkError("source pins do not match build request")
    if (config.get("family"), config.get("release"), config.get("split")) != (
        family,
        release,
        split,
    ):
        raise BenchmarkError("source and augment config pins differ")
    sources = source.get("cases")
    variants = config.get("variants")
    if not isinstance(sources, list) or not isinstance(variants, list) or not variants:
        raise BenchmarkError("source cases and at least one variant required")
    operations = {
        variant.get("operation")
        for variant in variants
        if isinstance(variant, dict)
    }
    native_build = "native_ingest" in operations
    if native_build and operations != {"native_ingest"}:
        raise BenchmarkError(
            "augment config must not mix native_ingest with other operations"
        )
    license_record = source.get("license")
    if (
        not isinstance(license_record, dict)
        or not require_nonempty(license_record.get("id"), "license.id")
    ):
        raise BenchmarkError("source license decision required")
    if native_build:
        if license_record.get("native_ingest_allowed") is not True:
            raise BenchmarkError("source license does not authorize native ingest")
    elif license_record.get("derive_allowed") is not True:
        raise BenchmarkError("source license does not authorize derivation")
    if split == "holdout":
        raise BenchmarkError("holdout cannot be an augmentation source")
    source_snapshot = source.get("source_snapshot")
    if source_snapshot is not None:
        if not isinstance(source_snapshot, dict):
            raise BenchmarkError("source_snapshot must be an object")
        snapshot_path = Path(
            require_nonempty(source_snapshot.get("path"), "source_snapshot.path")
        )
        snapshot_sha = require_sha(
            source_snapshot.get("sha256"), "source_snapshot.sha256"
        )
        if (
            not snapshot_path.is_absolute()
            or not snapshot_path.is_file()
            or file_sha256(snapshot_path) != snapshot_sha
        ):
            raise BenchmarkError("source snapshot missing or SHA-256 drift")
    source_sha = file_sha256(source_path)
    config_sha = file_sha256(config_path)
    by_id = {
        case.get("source_task_id"): case for case in sources if isinstance(case, dict)
    }
    if len(by_id) != len(sources):
        raise BenchmarkError("duplicate or invalid source case IDs")
    rules.verify_source(source)
    excluded = source.get("exclusion_text_sha256", [])
    if not isinstance(excluded, list):
        raise BenchmarkError("exclusion_text_sha256 must be a list")
    seen_texts = {require_sha(value, "exclusion_text_sha256") for value in excluded}
    published_texts: set[str] = set()
    source_text_by_id: Dict[str, str] = {}
    for case in sources:
        text_sha = normalized_text_sha256(case.get("task_text"))
        seen_texts.add(text_sha)
        source_text_by_id[case.get("source_task_id")] = text_sha
    if not Path(trusted_dir).is_absolute():
        raise BenchmarkError("trusted_dir must be an absolute directory")
    trusted_root = Path(trusted_dir).resolve()
    taskset: List[Dict[str, Any]] = []
    audit: List[Dict[str, Any]] = []
    runtime_by_task: Dict[str, Dict[str, Any]] = {}
    seen_ids: set[str] = set()
    for previous in store.list_artifacts("taskset"):
        for old in previous["payload"].get("taskset", []):
            metadata = old.get("benchmark") if isinstance(old, dict) else None
            if isinstance(metadata, dict):
                seen_ids.add(str(old.get("task_id")))
                text_sha = normalized_text_sha256(old.get("task_text"))
                seen_texts.add(text_sha)
                published_texts.add(text_sha)
    for variant in variants:
        source_case = (
            by_id.get(variant.get("source_task_id"))
            if isinstance(variant, dict)
            else None
        )
        if source_case is None:
            raise BenchmarkError("variant references an unknown source case")
        if "state_patch" in variant:
            if not rules.state_patch_allowed:
                raise BenchmarkError(
                    f"{family} derivation cannot use JSON state_patch"
                )
            variant = {
                **variant,
                "private": materialize_json_state_variant(
                    source_case, variant, trusted_root
                ),
            }
        variant = rules.materialize(source, source_case, variant)
        case = _augment(source_case, variant, family, release)
        public = case["public"]
        runtime_record = (
            variant["runtime"]
            if "runtime" in variant
            else source_case["runtime"]
            if "runtime" in source_case
            else source
        )
        runtime = lock_runtime(runtime_record)
        rules.check_build(source, source_case, runtime, case["private"])
        environment_sha = runtime["environment"]["sha256"]
        verifier_sha = runtime["verifier"]["sha256"]
        verifier_version = runtime["verifier"]["version"]
        command = runtime["verifier"]["command"]
        asset_sha = canonical_sha256(runtime["assets"])
        text_sha = normalized_text_sha256(public["task_text"])
# Native ingest re-publishes its own seed text by definition (design §5);
        # the exemption is bounded by that one source case and still rejects any
# fingerprint collision with previously published tasksets (cross-release).
        own_native_text = native_build and (
            text_sha == source_text_by_id.get(case["source_ref"])
        )
        duplicate_text = text_sha in seen_texts and not (
            own_native_text and text_sha not in published_texts
        )
        if public["task_id"] in seen_ids or duplicate_text:
            raise BenchmarkError("duplicate derived task ID or task text")
        seen_ids.add(public["task_id"])
        seen_texts.add(text_sha)
        trace = _failure_trace(variant.get("failure_trace"))
        case_audit = case["audit"]
        if (
            not isinstance(case_audit, dict)
            or case_audit.get("reviewed") is not True
            or not isinstance(case_audit.get("reference_root"), str)
            or not isinstance(case_audit.get("negative_root"), str)
        ):
            raise BenchmarkError(
                "independent review, reference and negative roots required"
            )
        for root in (case_audit["reference_root"], case_audit["negative_root"]):
            if not Path(root).is_absolute() or not Path(root).is_dir():
                raise BenchmarkError(f"audit root unavailable: {root}")
        manifest = {
            "schema_version": "benchmark-case-v1",
            "task_id": public["task_id"],
            "family": family,
            "source_release": release,
            "split": split,
            "source_sha256": source_sha,
            "augment_config_sha256": config_sha,
            "source_ref": case["source_ref"],
            "source_case": source_case,
            "source_snapshot": source_snapshot,
            "failure_trace": trace,
            "operation": case["operation"],
            "license_id": license_record["id"],
            "private": case["private"],
            "environment_sha256": environment_sha,
            "verifier_sha256": verifier_sha,
            "asset_manifest_sha256": asset_sha,
            "audit": case_audit,
        }
        manifest_sha = canonical_sha256(manifest)
        manifest_path = trusted_root / "cases" / f"{manifest_sha}.json"
        contents = json.dumps(
            manifest,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        # Both native calls read a trusted manifest. The candidate sees only the
        # public task item; source oracle and hidden assets remain outside it.
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        if manifest_path.exists() and manifest_path.read_bytes() != contents:
            raise BenchmarkError("trusted manifest content mismatch")
        manifest_path.write_bytes(contents)
        timeout_value = config.get("audit_timeout_seconds", 600)
        if (
            type(timeout_value) not in (int, float)
            or not math.isfinite(timeout_value)
            or timeout_value <= 0
        ):
            raise BenchmarkError("audit_timeout_seconds must be finite and positive")
        timeout = float(timeout_value)
        item = make_public_item(public, manifest_sha, environment_sha, verifier_sha)
        positive = run_case(
            command,
            public["task_id"],
            str(manifest_path),
            case_audit["reference_root"],
            phase="reference",
            timeout=timeout,
            public_item=item,
        )
        negative = run_case(
            command,
            public["task_id"],
            str(manifest_path),
            case_audit["negative_root"],
            phase="negative",
            timeout=timeout,
            public_item=item,
        )
        if (
            positive["status"] != "completed"
            or not positive["passed"]
            or negative["status"] != "completed"
            or negative["passed"]
            or positive["verifier_version"] != verifier_version
            or negative["verifier_version"] != verifier_version
        ):
            raise BenchmarkError(
                f"positive/negative native audit failed: {public['task_id']}"
            )
        taskset.append(item)
        runtime_by_task[item["task_id"]] = runtime
        audit.append(
            {
                "task_id": item["task_id"],
                "operation": case["operation"],
                "reference": positive,
                "negative": negative,
                "case_manifest_sha256": manifest_sha,
            }
        )
    quality_report = assess_taskset(taskset)
    if quality_report["quality_overall_score"] < quality_threshold:
        raise BenchmarkError(
            "benchmark quality score "
            f"{quality_report['quality_overall_score']:.4f} below threshold "
            f"{quality_threshold:.4f}: {quality_report['quality_issues']}"
        )
    native_pins: Dict[str, Any] | None = None
    if native_build:
        if not rules.native_pins:
            raise BenchmarkError("family does not support native ingest")
        native_pins = rules._call(rules.native_pins,
            config,
            {variant["source_task_id"] for variant in variants},
        )
    taskset_fp = fingerprint_of(taskset)
    taskset_sha = canonical_sha256(taskset)
    runtime_hashes = runtime_digests(taskset, runtime_by_task)
    version = store.next_artifact_version("taskset", ref_id)
    payload = {
        "ref_id": ref_id,
        "version": version,
        "family": family,
        "split": split,
        "source_release": release,
        "source_sha256": source_sha,
        "augment_config_sha256": config_sha,
        **runtime_hashes,
        "case_manifest_sha256": canonical_sha256(
            [x["case_manifest_sha256"] for x in audit]
        ),
        "taskset": taskset,
        "taskset_fingerprint": taskset_fp,
        "taskset_sha256": taskset_sha,
        "item_count": len(taskset),
        "build_report": {
            "recipe": "benchmark_case",
            "schema_version": "benchmark-case-v1",
            **quality_report,
            "audit": audit,
        },
        # Trusted-side lookup only; never passed through enumerate or task_file.
        "trusted_manifest_dir": str(trusted_root / "cases"),
        "runtime_by_task": runtime_by_task,
    }
    if native_pins is not None:
        # Public-input manifest is path+sha references only (D5); the report pin
        # freezes the per-task ingest evidence. Keys are source_task_id.
        payload["ingest_report_sha256"] = native_pins["ingest_report_sha256"]
        payload["public_inputs_dir"] = native_pins["public_inputs_dir"]
        payload["public_inputs_by_task"] = native_pins["public_inputs_by_task"]
    store.put_artifact("taskset", ref_id, version, payload, fingerprint=taskset_fp)
    reread = store.get_artifact("taskset", ref_id, version)
    if (
        reread is None
        or reread["payload"]["taskset_sha256"] != taskset_sha
        or reread["payload"]["taskset_fingerprint"] != taskset_fp
    ):
        raise BenchmarkError("taskset readback failed")
    return {
        "catalog_uri": store.uri,
        "taskset": {"ref_id": ref_id, "version": version, "fingerprint": taskset_fp},
        "item_count": len(taskset),
        "build_report": payload["build_report"],
        "source_release": release,
        "case_manifest_sha256": payload["case_manifest_sha256"],
        **runtime_hashes,
        "taskset_sha256": taskset_sha,
    }
