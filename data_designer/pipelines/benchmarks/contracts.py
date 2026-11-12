"""Stable, public/private benchmark-case-v1 contract.

This module deliberately contains no upstream benchmark runtime. Native runners run
in separately provisioned trusted environments and speak the JSON file protocol.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, Iterable

from data_designer.contracts.provenance import content_digest as fingerprint_of

FAMILIES = (
    "tau",
    "appworld",
    "webarena",
    "osworld",
    "gaia",
    "swe",
    "workbuddy",
    "terminal",
    "deepswe",
    "reasoning_qa",
    "knowledge_qa",
    "code_gen",
    "health_qa",
)
SPLITS = ("development", "regression", "holdout")
PUBLIC_KEYS = frozenset(
    {
        "task_id",
        "task_text",
        "tools_available",
        "success_criteria",
        "difficulty",
        "item_fingerprint",
        "benchmark",
    }
)
PRIVATE_MARKERS = (
    "answer",
    "gold",
    "secret",
    "hidden",
    "patch",
    "test_code",
    "verifier_path",
    "manifest_path",
    "password",
    "token",
    "api_key",
    "expected",
    "aliases",
    "rubric",
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
RESOURCE_USAGE_KEYS = frozenset(
    {
        "input_tokens",
        "output_tokens",
        "browser_actions",
        "cpu_seconds",
        "gpu_seconds",
        "vm_seconds",
        "cost_usd",
    }
)
RESOURCE_USAGE_COUNTS = frozenset(
    {"input_tokens", "output_tokens", "browser_actions"}
)


class BenchmarkError(RuntimeError):
    """A requested benchmark operation cannot be published or scored."""


class NativeFailure(BenchmarkError):
    """A native runner failure with a stable category for the run ledger."""

    def __init__(self, category: str, message: str):
        super().__init__(message)
        self.category = category


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def normalized_text_sha256(value: Any) -> str:
    """Hash presentation-insensitive text for exact contamination screening."""
    text = require_nonempty(value, "task_text")
    normalized = " ".join(unicodedata.normalize("NFKC", text).casefold().split())
    return canonical_sha256(normalized)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_sha(value: Any, name: str) -> str:
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        raise BenchmarkError(f"{name} must be a complete lowercase SHA-256")
    return value


def require_nonempty(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise BenchmarkError(f"{name} must be a non-empty string")
    return value


def require_family(family: Any) -> str:
    if family not in FAMILIES:
        raise BenchmarkError(f"family must be one of {FAMILIES}: {family!r}")
    return family


def require_split(split: Any) -> str:
    if split not in SPLITS:
        raise BenchmarkError(f"split must be one of {SPLITS}: {split!r}")
    return split


def assert_public_item(item: Dict[str, Any]) -> None:
    if not isinstance(item, dict) or set(item) != PUBLIC_KEYS:
        raise BenchmarkError(
            "public case must contain exactly the benchmark-case-v1 keys"
        )
    for key in ("task_id", "task_text", "success_criteria", "difficulty"):
        require_nonempty(item[key], key)
    if not isinstance(item["tools_available"], list) or not all(
        isinstance(tool, str) for tool in item["tools_available"]
    ):
        raise BenchmarkError("tools_available must be a list of strings")
    benchmark = item["benchmark"]
    if not isinstance(benchmark, dict) or set(benchmark) != {
        "family",
        "source_release",
        "source_task_id",
        "variant_id",
        "case_manifest_sha256",
        "environment_sha256",
        "verifier_sha256",
    }:
        raise BenchmarkError("public benchmark metadata has an invalid key set")
    family = require_family(benchmark["family"])
    if not item["task_id"].startswith(f"{family}:"):
        raise BenchmarkError("task_id must begin with its family namespace")
    for key in ("source_release", "source_task_id", "variant_id"):
        require_nonempty(benchmark[key], key)
    for key in ("case_manifest_sha256", "environment_sha256", "verifier_sha256"):
        require_sha(benchmark[key], key)
    if (
        not isinstance(item["item_fingerprint"], str)
        or len(item["item_fingerprint"]) != 64
    ):
        raise BenchmarkError("item_fingerprint must be a complete SHA-256")
    expected = canonical_sha256(
        {k: v for k, v in item.items() if k != "item_fingerprint"}
    )
    if item["item_fingerprint"] != expected:
        raise BenchmarkError("public item fingerprint mismatch")
    # Field names inside text cannot be recognized reliably; only structural leaks
    # can be rejected here. The source audit owns natural-language disclosure.
    if any(marker in str(key).lower() for key in item for marker in PRIVATE_MARKERS):
        raise BenchmarkError("private field in public item")


def make_public_item(
    public: Dict[str, Any], manifest_sha: str, environment_sha: str, verifier_sha: str
) -> Dict[str, Any]:
    expected = {
        "task_id",
        "task_text",
        "tools_available",
        "success_criteria",
        "difficulty",
        "family",
        "source_release",
        "source_task_id",
        "variant_id",
    }
    if not isinstance(public, dict) or set(public) != expected:
        raise BenchmarkError("source case public fields have an invalid key set")
    item = {
        key: public[key]
        for key in (
            "task_id",
            "task_text",
            "tools_available",
            "success_criteria",
            "difficulty",
        )
    }
    item["benchmark"] = {
        "family": public["family"],
        "source_release": public["source_release"],
        "source_task_id": public["source_task_id"],
        "variant_id": public["variant_id"],
        "case_manifest_sha256": require_sha(manifest_sha, "case manifest"),
        "environment_sha256": require_sha(environment_sha, "environment"),
        "verifier_sha256": require_sha(verifier_sha, "verifier"),
    }
    item["item_fingerprint"] = canonical_sha256(item)
    assert_public_item(item)
    return item


def validate_native_result(result: Any, task_id: str) -> Dict[str, Any]:
    if not isinstance(result, dict) or result.get("task_id") != task_id:
        raise BenchmarkError(f"native result identity mismatch for {task_id}")
    status = result.get("status")
    if status not in ("completed", "invalid_candidate", "infra_failed", "cancelled"):
        raise BenchmarkError(f"invalid native status for {task_id}: {status!r}")
    if status in ("infra_failed", "cancelled"):
        raise NativeFailure(status, f"native evaluation failed for {task_id}: {status}")
    if type(result.get("passed")) is not bool:
        raise BenchmarkError("native passed must be boolean")
    score = result.get("score")
    if (
        isinstance(score, bool)
        or not isinstance(score, (int, float))
        or not math.isfinite(score)
    ):
        raise BenchmarkError("native score must be finite")
    if not 0 <= score <= 1:
        raise BenchmarkError("native score must be in [0,1]")
    if status == "invalid_candidate" and (result["passed"] or score != 0):
        raise BenchmarkError(
            "invalid_candidate cannot pass or receive a positive score"
        )
    if not isinstance(result.get("native_metrics"), dict) or not all(
        isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
        for v in result["native_metrics"].values()
    ):
        raise BenchmarkError("native_metrics must contain finite numbers")
    require_nonempty(result.get("verifier_version"), "verifier_version")
    if not isinstance(result.get("evidence_refs"), list) or not all(
        isinstance(v, str) for v in result["evidence_refs"]
    ):
        raise BenchmarkError("evidence_refs must be strings")
    runtime = result.get("runtime_seconds")
    if (
        isinstance(runtime, bool)
        or not isinstance(runtime, (int, float))
        or not math.isfinite(runtime)
        or runtime < 0
    ):
        raise BenchmarkError("runtime_seconds must be finite and non-negative")
    usage = result.get("resource_usage")
    if "resource_usage" in result:
        if (
            not isinstance(usage, dict)
            or not usage
            or not set(usage) <= RESOURCE_USAGE_KEYS
        ):
            raise BenchmarkError("resource_usage has invalid or empty fields")
        for key, value in usage.items():
            if key in RESOURCE_USAGE_COUNTS:
                if type(value) is not int or value < 0:
                    raise BenchmarkError(f"resource_usage.{key} must be a count")
            elif (
                type(value) not in (int, float)
                or not math.isfinite(value)
                or value < 0
            ):
                raise BenchmarkError(
                    f"resource_usage.{key} must be finite and non-negative"
                )
    return result


def taskset_digest(items: Iterable[Dict[str, Any]]) -> str:
    return fingerprint_of(list(items))
