"""Assemble, fingerprint, write and reload Gym-native environment bundles.

The package root follows Gym's ``package_loader`` schema.  DataForge
audit material (rich draft tasks, anchors, gates, source provenance) lives under
``audit/`` and is ignored by Gym but covered by the Gym fingerprint.

Determinism rules:
  * ``manifest.fingerprint`` is Gym's 16-hex canonical-json fingerprint.
  * ``asset_sha256`` is the SHA-256 of the deterministic zip (including manifest).
  * Absolute user-data paths never enter the package; only doc ids and SHAs do.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import shutil
import tempfile
import zipfile
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from data_designer.pipelines.envsynth import grounding
from data_designer.pipelines.envsynth import gym_contract
from data_designer.pipelines.envsynth.domain_adapters import (
    DomainAdapterError,
    adapt_environment,
    resolve_domain_adapter,
)
from data_designer.pipelines.envsynth.environment_model import (
    METADATA_DANGEROUS_ACTIONS,
    METADATA_DOMAIN_SCHEMA_VERSION,
    METADATA_GLOSSARY,
    EnvironmentModel,
)
from data_designer.pipelines.envsynth.package_identity import (
    DEFAULT_MAX_STEPS,
    compute_content_digest,
    compute_fingerprint,
    project_manifest,
)
from data_designer.pipelines.envsynth.profile_compilers import (
    CompileContext,
    ProfileCompileError,
    ProfileCompiler,
    get_profile_compiler,
    supported_profiles,
)
from data_designer.pipelines.envsynth.redaction import (
    redact,
    redact_text,
    sanitize_record,
)
from data_designer.pipelines.envsynth.serialization import (
    json_bytes as _json_bytes,
    jsonl_bytes as _jsonl_bytes,
    yaml_bytes as _yaml_bytes,
)
PACKAGE_SCHEMA_VERSION = "gym-env-package-v1"
ENV_SPEC_SCHEMA_VERSION = "env-spec-v2"
RULES_SCHEMA_VERSION = "rules-v1"
EVENT_MODEL_SCHEMA_VERSION = "event-model-v1"
TASK_SCHEMA_VERSION = "env-task-v1"
RELEASE_SCHEMA_VERSION = "env-release-v1"
GENERATOR_NAME = "envsynth"
ZIP_NAME = "gym-env-package-v1.zip"
_SAFE_COMPONENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_AUDIT_PREFIX = "audit/"
_RESERVED_COMPILER_FILES = frozenset({"manifest.json", "tasks.jsonl"})


class EnvPackageError(RuntimeError):
    """Package build, validation or write failure."""


def _validated_package_path(path: Any) -> str:
    if not isinstance(path, str) or not path:
        raise EnvPackageError("package file path must be a non-empty string")
    if "\x00" in path or "\\" in path:
        raise EnvPackageError(f"package file path is invalid: {path!r}")
    if path.startswith("/") or re.match(r"^[A-Za-z]:", path):
        raise EnvPackageError(f"package file path must be relative: {path!r}")
    parts = path.split("/")
    if any(part in ("", ".", "..") for part in parts):
        raise EnvPackageError(f"package file path escapes its root: {path!r}")
    return path


def _validated_compiler_files(files: Any) -> Dict[str, bytes]:
    if not isinstance(files, dict):
        raise EnvPackageError("ProfileCompiler.build_files() must return an object")
    result: Dict[str, bytes] = {}
    for path, data in files.items():
        relative = _validated_package_path(path)
        if relative in _RESERVED_COMPILER_FILES or relative.startswith(_AUDIT_PREFIX):
            raise EnvPackageError(
                f"ProfileCompiler must not write reserved package path: {relative!r}"
            )
        if not isinstance(data, bytes):
            raise EnvPackageError(f"package file {relative!r} content must be bytes")
        result[relative] = data
    return result


def _redact_package_text(
    files: Dict[str, bytes], allowed_paths: Sequence[str]
) -> Dict[str, bytes]:
    result: Dict[str, bytes] = {}
    for path, data in files.items():
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError as error:
            raise EnvPackageError(
                f"package file {path!r} must be UTF-8 text"
            ) from error
        result[path] = redact_text(text, allowed_paths=allowed_paths).encode("utf-8")
    return result


def _validated_bundle_files(files: Dict[str, bytes]) -> None:
    for path, data in files.items():
        _validated_package_path(path)
        if not isinstance(data, bytes):
            raise EnvPackageError(f"package file {path!r} content must be bytes")


def _allowed_virtual_paths(model: EnvironmentModel) -> tuple[str, ...]:
    allowed = {item.path for item in model.workspace}
    for task in model.tasks:
        allowed.update(task.expected_files)
    for entry in model.replay:
        allowed.add(entry.cwd)
        cwd_to = entry.outcome.get("cwd_to")
        if isinstance(cwd_to, str) and cwd_to.startswith("/"):
            allowed.add(cwd_to)
        writes = entry.outcome.get("fs_writes")
        if isinstance(writes, dict):
            allowed.update(path for path in writes if isinstance(path, str))
        deletes = entry.outcome.get("fs_deletes")
        if isinstance(deletes, list):
            allowed.update(path for path in deletes if isinstance(path, str))
    return tuple(
        sorted(path for path in allowed if path.startswith("/") and path != "/")
    )


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@lru_cache(maxsize=1)
def generator_code_hash() -> str:
    package_dir = Path(__file__).resolve().parent
    repo_root = package_dir.parents[2]
    files = sorted(package_dir.rglob("*.py")) + [
        repo_root / "scripts" / "gym_publish_env.py",
    ]
    digest = hashlib.sha256()
    for path in files:
        if not path.is_file():
            continue
        try:
            relative = path.relative_to(repo_root)
        except ValueError:
            relative = path
        digest.update(str(relative).replace("\\", "/").encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\n")
    return digest.hexdigest()


def generator_ref(seed: int) -> str:
    return "{}@{}#seed={}".format(
        GENERATOR_NAME, generator_code_hash()[:16], int(seed)
    )


def _audit_env_spec(model: EnvironmentModel, profile: str) -> Dict[str, Any]:
    """Profile-neutral DD audit view; Gym never reads this file."""
    state_schema = {
        collection.audit_name or collection.name: dict(collection.fields)
        for collection in model.state_collections
    }
    initial_state = {
        collection.name: [sanitize_record(dict(row)) for row in collection.rows]
        for collection in model.state_collections
    }
    actions = [
        {
            "name": action.name,
            "description": action.description,
            "parameters": dict(action.parameters),
            "effect": dict(action.effect),
            "transitions": [
                {
                    "from": transition.from_state,
                    "to": transition.to_state,
                    "by_role": transition.role,
                }
                for transition in action.transitions
            ],
        }
        for action in model.actions
    ]
    return {
        "schemaVersion": ENV_SPEC_SCHEMA_VERSION,
        "profile": profile,
        "family": model.family,
        "domainSchemaVersion": model.metadata.get(
            METADATA_DOMAIN_SCHEMA_VERSION, "domain-unknown-v1"
        ),
        "stateSchema": state_schema,
        "glossary": list(model.metadata.get(METADATA_GLOSSARY) or []),
        "actions": actions,
        "dangerousActions": list(
            model.metadata.get(METADATA_DANGEROUS_ACTIONS) or []
        ),
        "initialState": initial_state,
    }


@dataclass
class EnvPackageBundle:
    manifest: Dict[str, Any]
    files: Dict[str, bytes]
    tasks: List[Dict[str, Any]]
    anchors: List[Dict[str, Any]]
    content_fingerprint: str
    content_digest: str = ""
    gym_tasks: List[Dict[str, Any]] = field(default_factory=list)
    audit_manifest: Dict[str, Any] = field(default_factory=dict)
    seed: int = 42
    default_max_steps: int = DEFAULT_MAX_STEPS
    gate_report: Dict[str, Any] = field(default_factory=dict)
    zip_bytes: bytes = b""
    asset_sha256: str = ""
    model: Optional[EnvironmentModel] = None
    compiler: Optional[ProfileCompiler] = None
    compile_context: Optional[CompileContext] = None


def _manifest(
    *,
    draft: Dict[str, Any],
    family: str,
    domain_schema_version: str,
    glossary: Sequence[str],
    ref_id: str,
    version: int,
    seed: int,
    mode: str,
    max_rounds: int,
    anchor_ratio: float,
    synthesis_model: Optional[str],
    simulator_model: Optional[str],
    source_files: Sequence[Dict[str, Any]],
    seed_manifest_sha256: str,
    curation_resolved: bool,
    expert_review: Dict[str, Any],
    grounding_index: Dict[str, Any],
    assumptions: Sequence[str],
) -> Dict[str, Any]:
    source_inventory = [
        {
            "sha256": item.get("sha256"),
            "mediaType": item.get("mediaType"),
            "role": item.get("role"),
            "docId": item.get("docId"),
            "byteSize": item.get("byteSize"),
        }
        for item in source_files
    ]
    return {
        "schema_version": PACKAGE_SCHEMA_VERSION,
        "ref_id": ref_id,
        "version": int(version),
        "fingerprint": "",
        "generator": {
            "name": GENERATOR_NAME,
            "code_hash": generator_code_hash(),
            "synthesis_model": synthesis_model,
            "simulator_model": simulator_model,
            "seed": int(seed),
            "max_rounds": int(max_rounds),
            "mode": mode,
        },
        "source": {
            "seed_manifest_sha256": seed_manifest_sha256,
            "anchor_holdout_ratio": float(anchor_ratio),
            "curator_schema": "parsed-doc-v2",
            "files": source_inventory,
        },
        "gates": {
            "anchor_replay": "pending",
            "distribution": "pending",
            "expert_review": (
                "waived" if expert_review.get("mode") == "waived" else "pending"
            ),
        },
        "curation_questions_resolved": bool(curation_resolved),
        "grounding": {
            "embedding_model": grounding_index.get("embeddingModel"),
            "dim": grounding_index.get("dim"),
            "chunk_count": grounding_index.get("chunkCount"),
        },
        "schema_versions": {
            "domain": domain_schema_version,
            "env_spec": ENV_SPEC_SCHEMA_VERSION,
            "rules": RULES_SCHEMA_VERSION,
            "event_model": EVENT_MODEL_SCHEMA_VERSION,
            "task": TASK_SCHEMA_VERSION,
            "grounding": grounding.GROUNDING_SCHEMA_VERSION,
        },
        "family": family,
        "task_count": len(draft.get("tasks") or []),
        "anchor_count": len(draft.get("anchors") or []),
        "glossary": list(glossary),
        "assumptions": list(assumptions),
        "expert_review": dict(expert_review),
    }


def _validated_ref_id(ref_id: str) -> str:
    if not isinstance(ref_id, str) or not _SAFE_COMPONENT_RE.fullmatch(ref_id):
        raise EnvPackageError("ref_id must be a safe file-path component")
    return ref_id


def _resolve_compiler(profile: str) -> ProfileCompiler:
    try:
        return get_profile_compiler(profile)
    except ProfileCompileError as error:
        raise EnvPackageError(
            "Gym profile {} is not registered; registered profiles: {}".format(
                profile, supported_profiles()
            )
        ) from error


def build_env_package(
    domain: Dict[str, Any],
    draft: Dict[str, Any],
    *,
    ref_id: str,
    version: int,
    profile: str = "office",
    seed: int = 42,
    mode: str = "template",
    max_rounds: int = 3,
    anchor_ratio: float = 0.3,
    synthesis_model: Optional[str] = None,
    simulator_model: Optional[str] = None,
    backend: Optional[grounding.EmbeddingBackend] = None,
    source_files: Optional[Sequence[Dict[str, Any]]] = None,
    seed_manifest_sha256: str = "",
    curation_resolved: bool = True,
    expert_review: Optional[Dict[str, Any]] = None,
) -> EnvPackageBundle:
    _validated_ref_id(ref_id)
    if int(version) < 1:
        raise EnvPackageError("version must be >= 1")
    compiler = _resolve_compiler(profile)

    raw_tasks = draft.get("tasks")
    raw_anchors = draft.get("anchors")
    if not isinstance(raw_tasks, list) or not raw_tasks:
        raise EnvPackageError("env package requires non-empty tasks")
    if not isinstance(raw_anchors, list) or not raw_anchors:
        raise EnvPackageError("env package requires non-empty anchors")
    if any(not isinstance(item, Mapping) for item in raw_anchors):
        raise EnvPackageError("env package anchors must all be objects")

    try:
        selected_adapter = resolve_domain_adapter(domain, draft)
    except DomainAdapterError as error:
        raise EnvPackageError("Domain semantic extraction failed: {}".format(error)) from error
    renderer = getattr(selected_adapter, "render_seed_text", None)
    grounding_chunks = grounding.chunks_from_domain(
        domain, renderer=renderer if callable(renderer) else None
    )
    index = grounding.build_index(grounding_chunks, backend=backend)
    chunks = [
        {key: value for key, value in chunk.items() if key != "vector"}
        for chunk in index.get("chunks") or []
    ]

    try:
        model = adapt_environment(
            domain,
            draft,
            grounding_chunks=chunks,
            adapter=selected_adapter,
        )
    except DomainAdapterError as error:
        raise EnvPackageError("Domain semantic extraction failed: {}".format(error)) from error
    allowed_paths = _allowed_virtual_paths(model)
    tasks = [
        sanitize_record(task, allowed_paths=allowed_paths) for task in raw_tasks
    ]
    anchors = [
        sanitize_record(anchor, allowed_paths=allowed_paths)
        for anchor in raw_anchors
    ]
    context = CompileContext(
        ref_id=ref_id,
        version=int(version),
        seed=int(seed),
        max_steps=DEFAULT_MAX_STEPS,
    )
    try:
        files = _validated_compiler_files(compiler.build_files(model, context))
    except ProfileCompileError as error:
        raise EnvPackageError("Gym package projection failed: {}".format(error)) from error
    files = _redact_package_text(files, allowed_paths)

    # DataForge audit view: rich draft tasks, anchors and original assets are
    # retained for gate/audit consumers, but Gym never reads this subtree.
    audit_chunks = [
        sanitize_record(chunk, allowed_paths=allowed_paths) for chunk in chunks
    ]
    audit_rules = sanitize_record(
        {
            "schemaVersion": RULES_SCHEMA_VERSION,
            "rules": draft.get("rules") or [],
        },
        allowed_paths=allowed_paths,
    )
    simulator_prompt = redact_text(
        str(draft.get("simulator_prompt") or ""), allowed_paths=allowed_paths
    )
    audit_files: Dict[str, bytes] = {
        "audit/env_spec.yaml": _yaml_bytes(_audit_env_spec(model, profile)),
        "audit/rules.yaml": _yaml_bytes(audit_rules),
        "audit/simulator_prompt.md": simulator_prompt.encode("utf-8"),
        "audit/event_model.yaml": _yaml_bytes(
            sanitize_record(
                draft.get("event_model") or {}, allowed_paths=allowed_paths
            )
        ),
        "audit/grounding/chunks.jsonl": _jsonl_bytes(audit_chunks),
        "audit/grounding/index.json": _json_bytes(
            sanitize_record(index, allowed_paths=allowed_paths)
        ),
        "audit/anchors/holdout.jsonl": _jsonl_bytes(anchors),
        "audit/tasks.jsonl": _jsonl_bytes(tasks),
        "audit/gym_projection.json": _json_bytes(
            sanitize_record(
                compiler.projection_report(model), allowed_paths=allowed_paths
            )
        ),
    }
    for actor in model.actors:
        role = str(actor.role)
        if not _SAFE_COMPONENT_RE.fullmatch(role):
            raise EnvPackageError("npc card role must be a safe file-path component")
        audit_files["audit/npc_cards/{}.yaml".format(role)] = _yaml_bytes(
            sanitize_record(dict(actor.card), allowed_paths=allowed_paths)
        )
    raw_replay = domain.get("replay", draft.get("replay"))
    terminal_config = domain.get("terminal")
    if raw_replay is None and isinstance(terminal_config, Mapping):
        raw_replay = terminal_config.get("replay")
    if raw_replay:
        audit_files["audit/replay/trie.json"] = _json_bytes(
            sanitize_record(raw_replay, allowed_paths=allowed_paths)
        )
    conflicts = sorted(set(files) & set(audit_files))
    if conflicts:
        raise EnvPackageError(
            "ProfileCompiler wrote audit/reserved paths: {}".format(conflicts)
        )
    files.update(_redact_package_text(audit_files, allowed_paths))

    expert = dict(expert_review or {"mode": "waived", "sampleRate": 0.0})
    audit_manifest = _manifest(
        draft=draft,
        family=model.family,
        domain_schema_version=str(
            model.metadata.get(METADATA_DOMAIN_SCHEMA_VERSION, "domain-unknown-v1")
        ),
        glossary=list(model.metadata.get(METADATA_GLOSSARY) or []),
        ref_id=ref_id,
        version=version,
        seed=seed,
        mode=mode,
        max_rounds=max_rounds,
        anchor_ratio=anchor_ratio,
        synthesis_model=synthesis_model,
        simulator_model=simulator_model,
        source_files=list(source_files or []),
        seed_manifest_sha256=seed_manifest_sha256,
        curation_resolved=curation_resolved,
        expert_review=expert,
        grounding_index=index,
        assumptions=draft.get("assumptions") or domain.get("assumptions") or [],
    )
    # The audit manifest is package content and must not encode the release
    # version: versions are allocated from a version-independent content digest.
    audit_manifest.pop("version", None)
    audit_manifest.pop("fingerprint", None)
    audit_manifest = sanitize_record(audit_manifest, allowed_paths=allowed_paths)
    files["audit/manifest.json"] = _json_bytes(audit_manifest)
    manifest = project_manifest(
        ref_id=ref_id,
        version=int(version),
        profile=profile,
        fingerprint="",
        simulator_model=simulator_model,
        description="Gym-native synthetic environment for {}".format(ref_id),
    )
    bundle = EnvPackageBundle(
        manifest=manifest,
        files=files,
        tasks=tasks,
        anchors=anchors,
        content_fingerprint="",
        audit_manifest=audit_manifest,
        seed=int(seed),
        default_max_steps=DEFAULT_MAX_STEPS,
        model=model,
        compiler=compiler,
        compile_context=context,
    )
    bundle.content_digest = compute_content_digest(ref_id, profile, files)
    _refresh_identity(bundle)
    return bundle


def _refresh_identity(bundle: EnvPackageBundle) -> None:
    """Rebuild Gym fingerprint + task pins after content or version changes."""
    if bundle.model is None or bundle.compiler is None or bundle.compile_context is None:
        raise EnvPackageError("bundle is missing the EnvironmentModel/ProfileCompiler context")
    files = bundle.files
    files.pop("manifest.json", None)
    files.pop("tasks.jsonl", None)
    manifest = bundle.manifest
    fingerprint = compute_fingerprint(
        str(manifest["ref_id"]),
        manifest["version"],
        str(manifest["profile"]),
        files,
    )
    try:
        gym_tasks = bundle.compiler.build_tasks(
            bundle.model,
            bundle.compile_context,
            fingerprint=fingerprint,
        )
        gym_tasks = redact(
            gym_tasks, allowed_paths=_allowed_virtual_paths(bundle.model)
        )
        gym_contract.validate_tasks(gym_tasks)
    except ProfileCompileError as error:
        raise EnvPackageError("Gym package projection failed: {}".format(error)) from error
    except gym_contract.GymContractError as error:
        raise EnvPackageError("Gym task row validation failed: {}".format(error)) from error
    files["tasks.jsonl"] = _jsonl_bytes(gym_tasks)
    manifest["fingerprint"] = fingerprint
    files["manifest.json"] = _json_bytes(manifest)
    _validated_bundle_files(files)
    bundle.gym_tasks = gym_tasks
    bundle.content_fingerprint = fingerprint


def deterministic_zip_bytes(files: Dict[str, bytes]) -> bytes:
    _validated_bundle_files(files)
    buffer = io.BytesIO()
    with zipfile.ZipFile(
        buffer, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True
    ) as archive:
        for path in sorted(files):
            info = zipfile.ZipInfo(filename=path, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = (0o644 & 0xFFFF) << 16
            archive.writestr(info, files[path])
    return buffer.getvalue()


def finalize_bundle(
    bundle: EnvPackageBundle, gate_report: Dict[str, Any]
) -> EnvPackageBundle:
    """Attach gate results and refresh Gym identity for the allocated version."""
    current = compute_content_digest(
        str(bundle.manifest["ref_id"]),
        str(bundle.manifest["profile"]),
        bundle.files,
    )
    if bundle.content_digest and current != bundle.content_digest:
        raise EnvPackageError(
            "Gate results changed the content digest; inspect the inputs to finalize_bundle"
        )
    bundle.gate_report = gate_report
    _refresh_identity(bundle)
    bundle.zip_bytes = deterministic_zip_bytes(bundle.files)
    bundle.asset_sha256 = _sha256(bundle.zip_bytes)
    return bundle


def _release_record(bundle: EnvPackageBundle) -> Dict[str, Any]:
    return {
        "schemaVersion": RELEASE_SCHEMA_VERSION,
        "refId": bundle.manifest["ref_id"],
        "version": bundle.manifest["version"],
        "fingerprint": bundle.content_fingerprint,
        "assetSha256": bundle.asset_sha256,
        "assetDir": "package",
        "manifest": bundle.manifest,
        "auditManifest": bundle.audit_manifest,
        "gates": bundle.gate_report,
    }


def write_release(release_dir: str, bundle: EnvPackageBundle) -> Dict[str, Any]:
    """Write ``release_dir/{package,gym-env-package-v1.zip,release.json}`` atomically."""
    if not bundle.zip_bytes or not bundle.asset_sha256:
        raise EnvPackageError("bundle is not finalized (missing zip/assetSha256)")
    _validated_bundle_files(bundle.files)
    target = Path(release_dir)
    if target.exists():
        record_path = target / "release.json"
        if not record_path.is_file():
            raise EnvPackageError(f"release directory already exists but is missing release.json: {target}")
        try:
            existing = json.loads(record_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise EnvPackageError(
                "release directory release.json is not readable: {}".format(error)
            ) from error
        if not isinstance(existing, dict):
            raise EnvPackageError("release directory release.json must be an object")
        if existing.get("assetSha256") != bundle.asset_sha256:
            raise EnvPackageError(
                "release directory already exists with different content; refusing to overwrite (F1 immutability): {}".format(target)
            )
        try:
            existing_manifest, existing_files = load_env_package(
                str(target / "package")
            )
            actual_zip_sha = zip_sha256(str(target / "package"))
        except EnvPackageError as error:
            raise EnvPackageError(
                "release directory already exists but cannot be fully revalidated: {}".format(error)
            ) from error
        if existing_manifest != bundle.manifest:
            raise EnvPackageError(
                "release directory manifest does not match the pending content: {}".format(target)
            )
        expected_files = {
            path: data
            for path, data in bundle.files.items()
            if path != "manifest.json"
        }
        if existing_files != expected_files:
            raise EnvPackageError(
                "release directory package file has different content: {}".format(target)
            )
        if actual_zip_sha != bundle.asset_sha256:
            raise EnvPackageError(
                "release directory package and zip SHA-256 do not match: {}".format(target)
            )
        zip_path = target / ZIP_NAME
        if not zip_path.is_file() or _sha256(zip_path.read_bytes()) != bundle.asset_sha256:
            raise EnvPackageError(
                "release directory zip file does not match assetSha256: {}".format(target)
            )
        record = dict(existing)
        record["assetDir"] = str(target / "package")
        return record

    target.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(dir=str(target.parent), prefix=".envrel-"))
    try:
        package_dir = temp_dir / "package"
        for relative, data in sorted(bundle.files.items()):
            path = package_dir / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        (temp_dir / ZIP_NAME).write_bytes(bundle.zip_bytes)
        (temp_dir / "release.json").write_bytes(_json_bytes(_release_record(bundle)))
        os.replace(str(temp_dir), str(target))
    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)
    record = _release_record(bundle)
    record["assetDir"] = str(target / "package")
    return record


def load_env_package(package_dir: str) -> Tuple[Dict[str, Any], Dict[str, bytes]]:
    root = Path(package_dir)
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise EnvPackageError(f"env package is missing manifest.json: {root}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise EnvPackageError(f"env package manifest is not readable: {error}") from error
    for key in ("ref_id", "version", "fingerprint", "profile"):
        if manifest.get(key) in (None, ""):
            raise EnvPackageError("env package manifest is missing field {}".format(key))
    if manifest.get("profile") not in supported_profiles():
        raise EnvPackageError(
            "env package profile is not registered: {}".format(manifest.get("profile"))
        )
    files: Dict[str, bytes] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path == manifest_path:
            continue
        relative = str(path.relative_to(root)).replace("\\", "/")
        files[relative] = path.read_bytes()
    if "tasks.jsonl" not in files:
        raise EnvPackageError("env package is missing tasks.jsonl")
    if "env_spec.json" not in files:
        raise EnvPackageError("env package is missing env_spec.json")
    if "fs/snapshot.jsonl" not in files:
        raise EnvPackageError("env package is missing fs/snapshot.jsonl")
    fingerprint = compute_fingerprint(
        str(manifest["ref_id"]),
        manifest["version"],
        str(manifest["profile"]),
        files,
    )
    if manifest.get("fingerprint") != fingerprint:
        raise EnvPackageError(
            "env package content fingerprint mismatch: manifest={} actual={}".format(
                manifest.get("fingerprint"), fingerprint
            )
        )
    return manifest, files


def zip_sha256(package_dir: str) -> str:
    root = Path(package_dir).resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise EnvPackageError(f"env package is missing manifest.json: {root}")
    files = {"manifest.json": manifest_path.read_bytes()}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path == manifest_path:
            continue
        relative = str(path.relative_to(root)).replace("\\", "/")
        files[relative] = path.read_bytes()
    return _sha256(deterministic_zip_bytes(files))


__all__ = [
    "ENV_SPEC_SCHEMA_VERSION",
    "EnvPackageBundle",
    "EnvPackageError",
    "PACKAGE_SCHEMA_VERSION",
    "RELEASE_SCHEMA_VERSION",
    "TASK_SCHEMA_VERSION",
    "ZIP_NAME",
    "build_env_package",
    "deterministic_zip_bytes",
    "finalize_bundle",
    "generator_code_hash",
    "generator_ref",
    "load_env_package",
    "write_release",
]