"""Read-only adapter for pinned, text-mode tau-bench domain snapshots.

The upstream task and simulator instructions remain in the trusted files. A
source manifest contains row digests and references, never reference actions.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..augment import _apply_patch
from ..contracts import (
    BenchmarkError,
    canonical_sha256,
    file_sha256,
    require_nonempty,
    require_sha,
    require_split,
)
from ..runtime import lock_runtime
from ..runners.tau_offline import VERSION as OFFLINE_RUNNER_VERSION
from .gaia import in_generation_checkout

UPSTREAM_SPLITS = frozenset({"train", "test", "base"})

OFFLINE_SPEC_KINDS = ("unit_test", "artifact_compare")


def _pinned_file(path_value: Any, sha_value: Any, name: str) -> Dict[str, str]:
    path = Path(require_nonempty(path_value, f"{name}.path"))
    sha = require_sha(sha_value, f"{name}.sha256")
    if not path.is_absolute() or not path.is_file():
        raise BenchmarkError(f"tau {name} must be an existing absolute file")
    if in_generation_checkout(path):
        raise BenchmarkError(f"tau {name} cannot live in this repository")
    if file_sha256(path) != sha:
        raise BenchmarkError(f"tau {name} SHA-256 drift")
    return {"path": str(path), "sha256": sha}


def _json_file(path: str, name: str) -> Any:
    try:
        return json.loads(
            Path(path).read_text(encoding="utf-8"),
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except (OSError, ValueError) as error:
        raise BenchmarkError(f"cannot read tau {name}: {error}") from error


def _offline_admission(basis: List[str], criteria: Dict[str, Any]) -> Dict[str, Any]:
    """TAU4 subset admission from pinned task metadata only.

    Reward bases outside {DB, COMMUNICATE} depend on live tool semantics or
    user-simulator dynamics and cannot be re-expressed as terminal-state
    file assertions; COMMUNICATE rewards additionally need the required
    message strings to be assertable strings.
    """
    if not set(basis) <= {"DB", "COMMUNICATE"}:
        return {"admission": "excluded", "reason": "simulator_required"}
    if "COMMUNICATE" in basis:
        info = criteria.get("communicate_info")
        if (
            not isinstance(info, list)
            or not info
            or not all(isinstance(item, str) and item.strip() for item in info)
        ):
            return {
                "admission": "excluded",
                "reason": "communicate_info_not_assertable",
            }
    return {
        "admission": "expressible",
        "mode": "db_communicate" if "COMMUNICATE" in basis else "db",
    }


def _task_case(
    row: Any, index: int, domain: str, files: Dict[str, Dict[str, str]]
) -> Dict[str, Any]:
    if not isinstance(row, dict):
        raise BenchmarkError(f"tau task row {index} must be an object")
    task_id = require_nonempty(row.get("id"), "tau task id")
    scenario = row.get("user_scenario")
    criteria = row.get("evaluation_criteria")
    if not isinstance(scenario, dict) or not isinstance(criteria, dict):
        raise BenchmarkError(f"tau task {task_id} needs scenario and criteria")
    if not isinstance(scenario.get("instructions"), (str, dict)):
        raise BenchmarkError(f"tau task {task_id} needs user instructions")
    instructions = scenario["instructions"]
    if isinstance(instructions, dict) and instructions.get("domain") not in (
        None, domain,
    ):
        raise BenchmarkError(f"tau task {task_id} domain mismatch")
    actions = criteria.get("actions")
    if actions is not None and not isinstance(actions, list):
        raise BenchmarkError(f"tau task {task_id} actions must be a list")
    basis = criteria.get("reward_basis", ["DB", "COMMUNICATE"])
    if not isinstance(basis, list) or not basis or not all(
        isinstance(value, str) and value for value in basis
    ):
        raise BenchmarkError(f"tau task {task_id} reward_basis invalid")
    return {
        "source_task_id": f"{domain}:{task_id}",
        "task_text": f"Simulated {domain} support interaction: task {task_id}",
        "initial_state_sha256": canonical_sha256(
            {
                "initial_state": row.get("initial_state"),
                "db_sha256": files["db"]["sha256"],
                "user_db_sha256": files.get("user_db", {}).get("sha256"),
            }
        ),
        "expected_state_sha256": canonical_sha256(
            {"evaluation_criteria": criteria, "db_sha256": files["db"]["sha256"]}
        ),
        "source_row": {"index": index, "sha256": canonical_sha256(row)},
        "tau": {
            "domain": domain,
            "upstream_task_id": task_id,
            "user_scenario_sha256": canonical_sha256(scenario),
            "evaluation_criteria_sha256": canonical_sha256(criteria),
            "reward_basis": basis,
            "reference_action_count": len(actions or []),
            "offline": _offline_admission(basis, criteria),
        },
    }


def _source_files(
    tasks_path: str,
    tasks_sha256: str,
    split_path: str,
    split_sha256: str,
    policy_path: str,
    policy_sha256: str,
    db_path: str,
    db_sha256: str,
    user_db_path: Optional[str],
    user_db_sha256: Optional[str],
) -> Dict[str, Dict[str, str]]:
    files = {
        "tasks": _pinned_file(tasks_path, tasks_sha256, "tasks"),
        "split_tasks": _pinned_file(split_path, split_sha256, "split_tasks"),
        "policy": _pinned_file(policy_path, policy_sha256, "policy"),
        "db": _pinned_file(db_path, db_sha256, "db"),
    }
    if (user_db_path is None) != (user_db_sha256 is None):
        raise BenchmarkError("tau user_db path and SHA-256 must be paired")
    if user_db_path is not None:
        files["user_db"] = _pinned_file(
            user_db_path, user_db_sha256, "user_db"
        )
    if len({item["path"] for item in files.values()}) != len(files):
        raise BenchmarkError("tau source files must have distinct paths")
    return files


def assert_tau_runtime_assets(
    runtime: Dict[str, Any], source_snapshot: Dict[str, Any]
) -> None:
    """Every executable runtime must preflight all upstream source files."""
    assets = runtime.get("assets") if isinstance(runtime, dict) else None
    files = source_snapshot.get("files") if isinstance(source_snapshot, dict) else None
    if not isinstance(assets, list) or not isinstance(files, dict):
        raise BenchmarkError("tau runtime/source assets missing")
    for name, pin in files.items():
        if pin not in assets:
            raise BenchmarkError(f"tau runtime missing pinned {name} asset")


def import_tau_snapshot(
    tasks_path: str,
    tasks_sha256: str,
    *,
    split_path: str,
    split_sha256: str,
    policy_path: str,
    policy_sha256: str,
    db_path: str,
    db_sha256: str,
    domain: str,
    release: str,
    split: str,
    upstream_split: str,
    license_record: Dict[str, Any],
    user_db_path: Optional[str] = None,
    user_db_sha256: Optional[str] = None,
    runtime_registry_path: Optional[str] = None,
    runtime_registry_sha256: Optional[str] = None,
) -> Dict[str, Any]:
    """Select one official text split from a fixed local domain snapshot."""
    files = _source_files(
        tasks_path, tasks_sha256, split_path, split_sha256,
        policy_path, policy_sha256, db_path, db_sha256,
        user_db_path, user_db_sha256,
    )
    domain = require_nonempty(domain, "tau domain")
    if not re.fullmatch(r"[a-z][a-z0-9_]*", domain):
        raise BenchmarkError("tau domain must be a simple lowercase identifier")
    release = require_nonempty(release, "tau release")
    if Path(tasks_path).name == "tasks_voice.json":
        raise BenchmarkError("tau voice tasks need a separate source adapter")
    split = require_split(split)
    if upstream_split not in UPSTREAM_SPLITS:
        raise BenchmarkError("tau upstream split must be train, test, or base")
    if upstream_split != "train" and split != "holdout":
        raise BenchmarkError("tau test/base cannot seed development or regression")
    if upstream_split == "train" and split == "holdout":
        raise BenchmarkError("tau train cannot be labeled holdout")
    if (
        not isinstance(license_record, dict)
        or not isinstance(license_record.get("id"), str)
        or not license_record["id"].strip()
        or type(license_record.get("derive_allowed")) is not bool
    ):
        raise BenchmarkError("explicit tau license decision required")
    tasks = _json_file(files["tasks"]["path"], "tasks")
    splits = _json_file(files["split_tasks"]["path"], "split_tasks")
    if not isinstance(tasks, list) or not tasks or not isinstance(splits, dict):
        raise BenchmarkError("tau tasks must be a list and splits an object")
    if "base" not in splits or upstream_split not in splits:
        raise BenchmarkError("tau split_tasks needs base and selected split")
    for key, value in splits.items():
        if not isinstance(value, list) or not all(
            isinstance(task_id, str) and task_id for task_id in value
        ) or len(value) != len(set(value)):
            raise BenchmarkError(f"tau split {key} has invalid task IDs")
    all_cases = [
        _task_case(row, index, domain, files) for index, row in enumerate(tasks)
    ]
    by_id = {case["tau"]["upstream_task_id"]: case for case in all_cases}
    if len(by_id) != len(all_cases):
        raise BenchmarkError("tau tasks contain duplicate IDs")
    if set(splits["base"]) != set(by_id):
        raise BenchmarkError("tau base split must cover tasks exactly")
    selected_ids = splits[upstream_split]
    if not selected_ids or not set(selected_ids) <= set(by_id):
        raise BenchmarkError("tau selected split empty or references unknown tasks")
    if "train" in splits and "test" in splits:
        if set(splits["train"]) & set(splits["test"]):
            raise BenchmarkError("tau train and test splits overlap")
    runtimes: Dict[str, Any] | None = None
    registry_sha: str | None = None
    if runtime_registry_path is not None or runtime_registry_sha256 is not None:
        registry_pin = _pinned_file(
            runtime_registry_path, runtime_registry_sha256, "runtime registry"
        )
        registry_sha = registry_pin["sha256"]
        registry = _json_file(registry_pin["path"], "runtime registry")
        if (
            not isinstance(registry, dict)
            or registry.get("schema_version") != "benchmark-runtime-registry-v1"
            or not isinstance(registry.get("cases"), dict)
        ):
            raise BenchmarkError("invalid tau runtime registry schema")
        runtimes = registry["cases"]
        if set(runtimes) != set(selected_ids):
            raise BenchmarkError("tau runtime registry must match selected IDs")
    selected: List[Dict[str, Any]] = []
    for task_id in selected_ids:
        case = dict(by_id[task_id])
        if runtimes is not None:
            runtime = lock_runtime(runtimes[task_id])
            by_path = {asset["path"]: asset for asset in runtime["assets"]}
            for pin in files.values():
                existing = by_path.get(pin["path"])
                if existing is not None and existing != pin:
                    raise BenchmarkError("tau runtime source asset SHA-256 conflict")
                by_path[pin["path"]] = pin
            runtime["assets"] = list(by_path.values())
            case["runtime"] = runtime
        selected.append(case)
    return {
        "schema_version": "benchmark-source-v1",
        "family": "tau",
        "release": release,
        "split": split,
        "license": dict(license_record),
        "source_snapshot": {
            **files["tasks"],
            "format": "tau-text-tasks-v1",
            "domain": domain,
            "upstream_split": upstream_split,
            "files": files,
        },
        "runtime_registry_sha256": registry_sha,
        "cases": selected,
    }


def verify_tau_source_manifest(source: Dict[str, Any]) -> None:
    """Recreate the selected task metadata from the pinned upstream files."""
    snapshot = source.get("source_snapshot")
    if (
        not isinstance(snapshot, dict)
        or snapshot.get("format") != "tau-text-tasks-v1"
        or not isinstance(snapshot.get("files"), dict)
    ):
        raise BenchmarkError("tau text source needs pinned source files")
    files = snapshot["files"]
    if set(files) not in (
        {"tasks", "split_tasks", "policy", "db"},
        {"tasks", "split_tasks", "policy", "db", "user_db"},
    ):
        raise BenchmarkError("tau source files incomplete")
    if {key: snapshot.get(key) for key in ("path", "sha256")} != files["tasks"]:
        raise BenchmarkError("tau primary task snapshot pin mismatch")
    for name, pin in files.items():
        if not isinstance(pin, dict):
            raise BenchmarkError(f"tau {name} pin invalid")
        _pinned_file(pin.get("path"), pin.get("sha256"), name)
    expected = import_tau_snapshot(
        files["tasks"]["path"], files["tasks"]["sha256"],
        split_path=files["split_tasks"]["path"],
        split_sha256=files["split_tasks"]["sha256"],
        policy_path=files["policy"]["path"],
        policy_sha256=files["policy"]["sha256"],
        db_path=files["db"]["path"],
        db_sha256=files["db"]["sha256"],
        user_db_path=files.get("user_db", {}).get("path"),
        user_db_sha256=files.get("user_db", {}).get("sha256"),
        domain=snapshot.get("domain"),
        release=source.get("release"),
        split=source.get("split"),
        upstream_split=snapshot.get("upstream_split"),
        license_record=source.get("license"),
    )
    if snapshot != expected["source_snapshot"]:
        raise BenchmarkError("tau source snapshot metadata drift")
    cases = source.get("cases")
    if not isinstance(cases, list) or len(cases) != len(expected["cases"]):
        raise BenchmarkError("tau source cases incomplete")
    for actual, fresh in zip(cases, expected["cases"]):
        if not isinstance(actual, dict):
            raise BenchmarkError("tau source case invalid")
        for key, value in fresh.items():
            if actual.get(key) != value:
                raise BenchmarkError(f"tau source metadata drift: {key}")
        if "runtime" in actual:
            assert_tau_runtime_assets(actual["runtime"], snapshot)


def _clean_relative(relative: str) -> str:
    parts = Path(relative).parts
    if (
        not relative.strip()
        or Path(relative).is_absolute()
        or "\\" in relative
        or ".." in parts
        or any(part in ("", ".") for part in parts)
    ):
        raise BenchmarkError(f"must be a clean relative path: {relative!r}")
    return relative


def _pinned_db_sha256(source: Dict[str, Any]) -> str:
    snapshot = source.get("source_snapshot")
    files = snapshot.get("files") if isinstance(snapshot, dict) else None
    pin = files.get("db") if isinstance(files, dict) else None
    if not isinstance(pin, dict) or not pin.get("path") or not pin.get("sha256"):
        raise BenchmarkError("tau offline derivation needs a pinned db file")
    path = Path(pin["path"])
    if not path.is_absolute() or not path.is_file() or file_sha256(path) != pin["sha256"]:
        raise BenchmarkError("tau pinned db file missing or drifted")
    return pin["sha256"]


def _expected_db_state(source: Dict[str, Any], value: Any) -> Dict[str, Any]:
    """TAU4 criterion 2: explicit expected object or JSON patch on the pinned db."""
    if isinstance(value, dict) and set(value) == {"patch"}:
        snapshot = source.get("source_snapshot")
        files = snapshot.get("files") if isinstance(snapshot, dict) else None
        pin = files.get("db") if isinstance(files, dict) else None
        if not isinstance(pin, dict):
            raise BenchmarkError("tau offline patch mode needs a pinned db file")
        initial = _json_file(pin["path"], "pinned db")
        return _apply_patch(initial, value["patch"], "tau_offline expected")
    if not isinstance(value, dict) or not value:
        raise BenchmarkError(
            "tau_offline expected_state must be a non-empty object or {patch: [...]}"
        )
    return value


def derive_tau_offline_spec(
    source: Dict[str, Any], source_case: Dict[str, Any], directive: Any
) -> Dict[str, Any]:
    """TAU5 pregeneration: build the offline verifier_spec from the directive.

    The trusted directive supplies the expected terminal database state — an
    explicit object or a JSON patch against the pinned db (TAU4 criterion 2) —
    plus, for COMMUNICATE reward bases, the required message strings (TAU4
    criterion 1). Only the derived verifier_spec travels into the trusted
    manifest; official tool semantics are never re-implemented.
    """
    if not isinstance(directive, dict):
        raise BenchmarkError("tau_offline directive must be an object")
    allowed = {
        "artifact", "expected_state", "messages", "gold", "fields", "timeout_s",
    }
    if set(directive) - allowed:
        raise BenchmarkError(f"tau_offline directive accepts only {sorted(allowed)}")
    if ("expected_state" in directive) == ("gold" in directive):
        raise BenchmarkError(
            "tau_offline directive needs exactly one of expected_state or gold"
        )
    artifact = directive.get("artifact")
    if artifact is None:
        artifact = {"path": "tau_offline_artifact.json"}
    if (
        not isinstance(artifact, dict)
        or set(artifact) - {"path"}
        or not isinstance(artifact.get("path"), str)
    ):
        raise BenchmarkError("tau_offline artifact must be {path: relative}")
    relative = _clean_relative(artifact["path"])
    timeout = directive.get("timeout_s", 30.0)
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or timeout <= 0
    ):
        raise BenchmarkError("tau_offline timeout_s must be a positive number")
    metadata = source_case.get("tau")
    offline = metadata.get("offline") if isinstance(metadata, dict) else None
    task_id = source_case.get("source_task_id")
    if not isinstance(metadata, dict) or not isinstance(offline, dict) or (
        offline.get("admission") != "expressible"
    ):
        reason = offline.get("reason") if isinstance(offline, dict) else "unknown"
        raise BenchmarkError(
            f"tau task {task_id!r} is not admitted for offline derivation "
            f"({reason})"
        )
    basis = metadata.get("reward_basis")
    communicate = isinstance(basis, list) and "COMMUNICATE" in basis
    messages = directive.get("messages")
    if "messages" in directive and not communicate:
        raise BenchmarkError(
            "tau_offline messages assertions need a COMMUNICATE reward basis"
        )
    if communicate and (
        not isinstance(messages, list)
        or not messages
        or not all(isinstance(item, str) and item.strip() for item in messages)
    ):
        raise BenchmarkError("tau_offline needs assertable required message strings")
    release = require_nonempty(source.get("release"), "release")
    if "gold" in directive:
        if communicate:
            raise BenchmarkError(
                "tau_offline artifact_compare cannot assert messages; "
                "provide expected_state for COMMUNICATE reward bases"
            )
        gold = directive["gold"]
        if not isinstance(gold, dict) or not gold:
            raise BenchmarkError("tau_offline gold must be a non-empty object")
        fields = directive.get("fields") or {}
        if not isinstance(fields, dict):
            raise BenchmarkError("tau_offline fields must be an object")
        spec = {
            "kind": "artifact_compare",
            "generator_ref": f"tau-offline-state@{release}",
            "payload": {
                "artifact": {"path": relative},
                "gold": gold,
                "fields": fields,
            },
        }
        expected_input: Dict[str, Any] = {"gold": gold, "fields": fields}
        initial_input: Dict[str, Any] = {"gold_mode": True}
    else:
        expected = _expected_db_state(source, directive["expected_state"])
        literal = json.dumps(json.dumps(expected, sort_keys=True), ensure_ascii=False)
        script = (
            "import json\n"
            f"EXPECTED = json.loads({literal})\n"
            "DB = json.load(open('db_state.json', encoding='utf-8'))\n"
            "assert DB == EXPECTED, 'db_state mismatch: %r' % (DB,)\n"
        )
        if communicate:
            required = json.dumps(
                json.dumps(messages, ensure_ascii=False), ensure_ascii=False
            )
            script += (
                f"REQUIRED = json.loads({required})\n"
                "RECORD = json.dumps(json.load(open('messages.json', "
                "encoding='utf-8')), ensure_ascii=False)\n"
                "missing = [item for item in REQUIRED if item not in RECORD]\n"
                "assert not missing, ('missing required messages', missing)\n"
            )
        spec = {
            "kind": "unit_test",
            "generator_ref": f"tau-offline-state@{release}",
            "payload": {
                "artifact": {"path": relative},
                "test_script": script,
                "timeout_s": float(timeout),
            },
        }
        expected_input = {"expected_state": expected, "messages": messages or []}
        initial_input = {"patch": directive["expected_state"].get("patch")}
    return {
        "verifier_spec": spec,
        "initial_state_sha256": canonical_sha256(
            {"db_sha256": _pinned_db_sha256(source), "directive": initial_input}
        ),
        "expected_state_sha256": canonical_sha256(expected_input),
    }


def assert_tau_offline_build(
    runtime: Dict[str, Any], private: Dict[str, Any], source_case: Dict[str, Any]
) -> None:
    """M2 gate: every new tau taskset pins the offline runner and a real spec."""
    metadata = source_case.get("tau")
    offline = metadata.get("offline") if isinstance(metadata, dict) else None
    task_id = source_case.get("source_task_id")
    if not isinstance(offline, dict) or offline.get("admission") != "expressible":
        reason = offline.get("reason") if isinstance(offline, dict) else "unknown"
        raise BenchmarkError(
            f"tau task {task_id!r} is excluded from offline derivation ({reason})"
        )
    verifier = runtime.get("verifier")
    if (
        not isinstance(verifier, dict)
        or verifier.get("version") != OFFLINE_RUNNER_VERSION
    ):
        raise BenchmarkError(
            "tau derivation must pin the offline runner "
            f"({OFFLINE_RUNNER_VERSION}); the repo-external action-replay "
            "runner is retired for new tasksets"
        )
    spec = private.get("verifier_spec") if isinstance(private, dict) else None
    if not isinstance(spec, dict) or spec.get("kind") not in OFFLINE_SPEC_KINDS:
        raise BenchmarkError(
            "tau trusted case lacks a unit_test/artifact_compare "
            f"verifier_spec: {task_id}"
        )
