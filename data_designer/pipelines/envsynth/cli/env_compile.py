"""C-WK-1 file-protocol worker: seed import → gates → frozen env package.

Primary v0.1 entry point::

    python -m data_designer.pipelines.envsynth.cli.synthesis_worker <input.json> <snapshot.json>

Internal alias (no artifact-store persistence)::

    python -m data_designer.pipelines.envsynth.cli.env_compile <input.json> <compile-result.json>

Exit codes (design §3.6): 0 success, 1 build/gate/curation failure, 2 invalid
envelope.  The snapshot is written only on exit 0.  On failure, diagnostics and
curation questions are written next to the requested snapshot path with
``.<suffix>.curation_questions.json`` / ``.<suffix>.diagnostics.json``.
"""

from __future__ import annotations

import json
import copy
import os
import re
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from data_designer.pipelines.envsynth import artifacts, gym_contract, interview
from data_designer.pipelines.envsynth.cli.common import (
    EXIT_FAILED,
    EXIT_OK,
    EXIT_USAGE,
    SCHEMA_VERSION,
    WorkerError,
    absolute_path,
    load_json,
)
from data_designer.pipelines.envsynth.curator_contract import CuratorContractError
from data_designer.pipelines.envsynth.gates import GateError, run_gates
from data_designer.pipelines.envsynth.grounding import CharNgramEmbeddingBackend
from data_designer.pipelines.envsynth.package import (
    EnvPackageError,
    build_env_package,
    finalize_bundle,
    write_release,
)
from data_designer.pipelines.envsynth.redaction import (
    sanitize_draft,
    sanitize_record,
)
from data_designer.pipelines.envsynth.serialization import json_bytes
from data_designer.pipelines.envsynth.seed_import import (
    SeedBundle,
    SeedImportError,
    build_seed_import_from_source,
    load_seed_bundle,
    sha256_file,
)
from data_designer.pipelines.envsynth.synthesizer import (
    MAX_AGENT_ROUNDS,
    OpenAICompatibleChatClient,
    SynthesisError,
    synthesize,
    validate_draft,
)
from data_designer.pipelines.envsynth.domains.user_examples_seed_adapter import (
    FAMILY,
    compute_anchor_split,
    domain_schema_guess,
    validate_domain,
)


_REF_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class BuildFailure(RuntimeError):
    """Build/gate/curation failure (exit 1), with diagnostics for the host."""

    def __init__(
        self,
        message: str,
        *,
        stage: str = "build",
        questions: Optional[List[Dict[str, Any]]] = None,
        gate_report: Optional[Dict[str, Any]] = None,
        diagnostics: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.stage = stage
        self.questions = list(questions or [])
        self.gate_report = gate_report
        self.diagnostics = dict(diagnostics or {})


@dataclass
class SynthesisModel:
    base_url: str
    api_key_env: str
    model: str
    temperature: float


@dataclass
class ExpertReview:
    mode: str
    sample_rate: float
    evidence_path: Optional[str]


@dataclass
class SynthesisInput:
    asset_root: str
    store_path: Optional[str]
    ref_id: str
    family: str
    split: str
    seed_manifest_path: Optional[str]
    seed_manifest_sha256: Optional[str]
    source_paths: List[str]
    regulation_paths: List[str]
    source_manifest_out_dir: Optional[str]
    max_questions: int
    parse_options: Dict[str, Any]
    env_profile: str
    mode: str
    max_rounds: int
    model: Optional[SynthesisModel]
    simulator_model: Optional[str]
    embed_model: str
    top_k: int
    curation_answers_path: Optional[str]
    quality_threshold: float
    anchor_holdout_ratio: float
    expert_review: ExpertReview
    seed: int


def _absolute(value: Any, field: str) -> str:
    return absolute_path(value, field)


def _number(value: Any, field: str, default: float) -> float:
    if value is None:
        return default
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise WorkerError(f"{field} must be numeric")
    return float(value)


def _expert_review(request: Dict[str, Any]) -> ExpertReview:
    raw = request.get("expertReview") or {"mode": "waived", "sampleRate": 0.0}
    if not isinstance(raw, dict):
        raise WorkerError("request.expertReview must be an object")
    mode = raw.get("mode", "waived")
    if mode not in ("waived", "required"):
        raise WorkerError("request.expertReview.mode must be required or waived")
    sample_rate = _number(raw.get("sampleRate"), "expertReview.sampleRate", 0.0)
    if not 0.0 <= sample_rate <= 1.0:
        raise WorkerError("expertReview.sampleRate must be in [0,1]")
    evidence_path = raw.get("evidencePath")
    if evidence_path is not None:
        evidence_path = _absolute(evidence_path, "expertReview.evidencePath")
    if mode == "required" and not evidence_path:
        raise WorkerError("evidencePath is required when expertReview.mode=required")
    return ExpertReview(mode=mode, sample_rate=sample_rate, evidence_path=evidence_path)


def parse_input(path: str, *, require_store: bool = False) -> SynthesisInput:
    raw = load_json(path)
    if not isinstance(raw, dict):
        raise WorkerError("Input must be a JSON object")
    if raw.get("schemaVersion") != SCHEMA_VERSION:
        raise WorkerError(f"schemaVersion must be {SCHEMA_VERSION}")
    request = raw.get("request")
    if not isinstance(request, dict):
        raise WorkerError("request must be an object")
    asset_root = _absolute(request.get("assetRoot"), "request.assetRoot")
    raw_store_path = request.get("storePath")
    store_path: Optional[str] = None
    if raw_store_path is not None or require_store:
        store_path = _absolute(raw_store_path, "request.storePath")
    ref_id = request.get("refId")
    if not isinstance(ref_id, str) or not _REF_ID_RE.fullmatch(ref_id):
        raise WorkerError("request.refId must match ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
    family = request.get("family", FAMILY)
    if not isinstance(family, str) or not family:
        raise WorkerError("request.family must be a non-empty string")
    split = request.get("split", "train")
    if not isinstance(split, str) or not split:
        raise WorkerError("request.split must be a non-empty string")

    seed_manifest_path = request.get("seedManifestPath")
    seed_manifest_sha256 = request.get("seedManifestSha256")
    source = request.get("source")
    source_paths: List[str] = []
    regulation_paths: List[str] = []
    source_manifest_out_dir: Optional[str] = None
    max_questions = 20
    parse_options: Dict[str, Any] = {}
    if seed_manifest_path is not None:
        seed_manifest_path = _absolute(seed_manifest_path, "request.seedManifestPath")
        if not isinstance(seed_manifest_sha256, str) or not re.fullmatch(
            r"[0-9a-f]{64}", seed_manifest_sha256
        ):
            raise WorkerError("request.seedManifestSha256 must be 64 lowercase hex characters")
        if source is not None:
            raise WorkerError("seedManifestPath and source are mutually exclusive")
    else:
        if not isinstance(source, dict):
            raise WorkerError("request requires seedManifestPath or source")
        raw_paths = source.get("paths")
        if not isinstance(raw_paths, list) or not raw_paths:
            raise WorkerError("request.source.paths must be a non-empty array")
        source_paths = [_absolute(item, "source.paths[]") for item in raw_paths]
        raw_reg = source.get("regulationPaths") or []
        if not isinstance(raw_reg, list):
            raise WorkerError("request.source.regulationPaths must be an array")
        regulation_paths = [
            _absolute(item, "source.regulationPaths[]") for item in raw_reg
        ]
        source_manifest_out_dir = _absolute(
            source.get("manifestOutDir"), "source.manifestOutDir"
        )
        raw_max_questions = source.get("maxQuestions", 20)
        if (
            not isinstance(raw_max_questions, int)
            or isinstance(raw_max_questions, bool)
            or raw_max_questions < 1
        ):
            raise WorkerError("source.maxQuestions must be a positive integer")
        max_questions = raw_max_questions
        parse_options = source.get("parseOptions") or {}
        if not isinstance(parse_options, dict):
            raise WorkerError("source.parseOptions must be an object")

    env_profile = request.get("envProfile", "office")
    if not isinstance(env_profile, str) or not env_profile:
        raise WorkerError("request.envProfile must be a non-empty string")
    synthesis = request.get("synthesis") or {}
    if not isinstance(synthesis, dict):
        raise WorkerError("request.synthesis must be an object")
    mode = synthesis.get("mode", "template")
    if mode not in ("template", "agent"):
        raise WorkerError("synthesis.mode must be template or agent")
    max_rounds = synthesis.get("maxRounds", 3)
    if (
        not isinstance(max_rounds, int)
        or isinstance(max_rounds, bool)
        or not 1 <= max_rounds <= MAX_AGENT_ROUNDS
    ):
        raise WorkerError("synthesis.maxRounds must be an integer in 1..{}".format(MAX_AGENT_ROUNDS))
    model: Optional[SynthesisModel] = None
    if mode == "agent":
        raw_model = synthesis.get("model")
        if not isinstance(raw_model, dict):
            raise WorkerError("synthesis.model is required when synthesis.mode=agent")
        for key in ("baseUrl", "apiKeyEnv", "model"):
            if not isinstance(raw_model.get(key), str) or not raw_model[key].strip():
                raise WorkerError(f"synthesis.model.{key} must be a non-empty string")
        model = SynthesisModel(
            base_url=raw_model["baseUrl"],
            api_key_env=raw_model["apiKeyEnv"],
            model=raw_model["model"],
            temperature=_number(raw_model.get("temperature"), "synthesis.model.temperature", 0.2),
        )
    simulator_model = synthesis.get("simulatorModel")
    if simulator_model is not None and not isinstance(simulator_model, str):
        raise WorkerError("synthesis.simulatorModel must be a string")
    grounding_cfg = synthesis.get("grounding") or {}
    if not isinstance(grounding_cfg, dict):
        raise WorkerError("synthesis.grounding must be an object")
    embed_model = grounding_cfg.get("embedModel", "char-ngram-v1")
    if embed_model not in ("char-ngram-v1", "auto"):
        raise WorkerError(
            "The current DataForge implementation supports only embedModel=char-ngram-v1/auto;"
            "the bge backend must be injected in a future release"
        )
    top_k = grounding_cfg.get("topK", 4)
    if not isinstance(top_k, int) or isinstance(top_k, bool) or not 1 <= top_k <= 8:
        raise WorkerError("synthesis.grounding.topK must be an integer in 1..8")
    curation_answers_path = request.get("curationAnswersPath")
    if curation_answers_path is not None:
        curation_answers_path = _absolute(
            curation_answers_path, "request.curationAnswersPath"
        )
    quality_threshold = _number(
        request.get("qualityThreshold"), "request.qualityThreshold", 0.6
    )
    if not 0.0 < quality_threshold <= 1.0:
        raise WorkerError("request.qualityThreshold must be in (0,1]")
    anchor_ratio = _number(
        request.get("anchorHoldoutRatio"), "request.anchorHoldoutRatio", 0.3
    )
    if not 0.0 < anchor_ratio < 1.0:
        raise WorkerError("request.anchorHoldoutRatio must be in (0,1)")
    seed = request.get("seed", 42)
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise WorkerError("request.seed must be an integer")
    return SynthesisInput(
        asset_root=asset_root,
        store_path=store_path,
        ref_id=ref_id,
        family=family,
        split=split,
        seed_manifest_path=seed_manifest_path,
        seed_manifest_sha256=seed_manifest_sha256,
        source_paths=source_paths,
        regulation_paths=regulation_paths,
        source_manifest_out_dir=source_manifest_out_dir,
        max_questions=max_questions,
        parse_options=parse_options,
        env_profile=env_profile,
        mode=mode,
        max_rounds=max_rounds,
        model=model,
        simulator_model=simulator_model,
        embed_model=embed_model,
        top_k=top_k,
        curation_answers_path=curation_answers_path,
        quality_threshold=quality_threshold,
        anchor_holdout_ratio=anchor_ratio,
        expert_review=_expert_review(request),
        seed=seed,
    )


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(path.name + ".tmp")
    try:
        temp_path.write_bytes(json_bytes(payload))
        os.replace(str(temp_path), str(path))
    finally:
        if temp_path.exists():
            temp_path.unlink()


def _diagnostics_path(snapshot_path: Path) -> Path:
    return snapshot_path.with_name(snapshot_path.stem + ".diagnostics.json")


def _curation_path(snapshot_path: Path) -> Path:
    return snapshot_path.with_name(snapshot_path.stem + ".curation_questions.json")


def _seed_audit_summary(bundle: SeedBundle, *, mode: str) -> Dict[str, Any]:
    files = list(bundle.files or [])
    return {
        "mode": mode,
        "seedManifestSha256": bundle.manifest_sha256,
        "fileCount": len(files),
        "seedFileCount": len([item for item in files if item.get("role") == "seed"]),
        "regulationFileCount": len(
            [item for item in files if item.get("role") == "regulation"]
        ),
        "files": [
            {
                "sha256": item.get("sha256"),
                "mediaType": item.get("mediaType"),
                "role": item.get("role"),
                "docId": item.get("docId"),
                "byteSize": item.get("byteSize"),
            }
            for item in files
        ],
    }


def _load_seed(data: SynthesisInput) -> Tuple[SeedBundle, Dict[str, Any]]:
    if data.seed_manifest_path:
        try:
            bundle = load_seed_bundle(
                data.seed_manifest_path,
                expected_sha256=data.seed_manifest_sha256,
            )
        except (SeedImportError, CuratorContractError) as error:
            raise BuildFailure(
                f"seed manifest validation failed: {error}",
                stage="seed_import",
                diagnostics={"errorType": type(error).__name__},
            ) from error
        return bundle, _seed_audit_summary(bundle, mode="manifest")

    try:
        build_seed_import_from_source(
            data.source_paths,
            str(data.source_manifest_out_dir),
            regulation_paths=data.regulation_paths,
            family=data.family,
            anchor_holdout_ratio=data.anchor_holdout_ratio,
            max_questions=data.max_questions,
            parse_options=data.parse_options,
        )
        manifest_path = Path(str(data.source_manifest_out_dir)) / "seed_manifest.json"
        bundle = load_seed_bundle(str(manifest_path))
    except (SeedImportError, CuratorContractError) as error:
        raise BuildFailure(
            f"source import failed: {error}",
            stage="seed_import",
            diagnostics={"errorType": type(error).__name__},
        ) from error
    return bundle, _seed_audit_summary(bundle, mode="source")


def _recompute_split_if_needed(
    domain: Dict[str, Any], ratio: float
) -> Tuple[Dict[str, Any], Dict[str, Any], bool]:
    tickets = list(domain.get("tickets") or [])
    computed = compute_anchor_split(tickets, ratio)
    existing = domain.get("anchorSplit") or {}
    changed = not (
        abs(float(existing.get("ratio", -1.0)) - float(ratio)) < 1e-9
        and list(existing.get("anchorTicketIds") or [])
        == list(computed.get("anchorTicketIds") or [])
        and list(existing.get("groundingTicketIds") or [])
        == list(computed.get("groundingTicketIds") or [])
    )
    split = computed if changed else dict(existing)
    if changed:
        domain["anchorSplit"] = split
    anchor_ids = {str(ticket_id) for ticket_id in split.get("anchorTicketIds") or []}
    for ticket in domain.get("tickets") or []:
        ticket["split"] = (
            "anchor" if str(ticket.get("ticket_id")) in anchor_ids else "grounding"
        )
    return domain, split, changed


def _prepare_domain(
    bundle: SeedBundle, data: SynthesisInput
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    domain = copy.deepcopy(bundle.domain)
    applied: List[str] = []
    if data.curation_answers_path:
        try:
            domain, unresolved, applied = interview.apply_answers_file(
                domain, data.curation_answers_path
            )
        except interview.CurationError as error:
            raise BuildFailure(
                f"Unable to apply curation answers: {error}",
                stage="curation",
                diagnostics={"seedManifestSha256": bundle.manifest_sha256},
            ) from error

    blocking = interview.blocking_questions(domain)
    if blocking:
        raise BuildFailure(
            "Blocking under-specified issues require human clarification",
            stage="curation",
            questions=list(domain.get("questions") or []),
            diagnostics={
                "seedManifestSha256": bundle.manifest_sha256,
                "blockingQuestionCount": len(blocking),
                "curationAnswersApplied": applied,
            },
        )

    domain, split, split_changed = _recompute_split_if_needed(
        domain, data.anchor_holdout_ratio
    )
    try:
        validate_domain(domain)
    except ValueError as error:
        raise BuildFailure(
            f"domain dataset validation failed: {error}",
            stage="domain_validation",
            diagnostics={"seedManifestSha256": bundle.manifest_sha256},
        ) from error
    domain["schemaGuess"] = domain_schema_guess(domain)
    coverage = dict(domain.get("coverage") or {})
    coverage["ticketCount"] = len(domain.get("tickets") or [])
    coverage["questionCount"] = len(domain.get("questions") or [])
    coverage["documentCount"] = len(domain.get("documents") or [])
    domain["coverage"] = coverage
    curation = {
        "applied": applied,
        "remainingQuestions": list(domain.get("questions") or []),
        "blockingQuestionCount": 0,
        "splitRecomputed": split_changed,
        "anchorSplit": split,
        "curationResolved": True,
        "seedManifestSha256": bundle.manifest_sha256,
    }
    return domain, curation


def _build_fingerprint(content_digest: str, data: SynthesisInput) -> str:
    evidence_sha: Optional[str] = None
    if data.expert_review.evidence_path:
        try:
            evidence_sha = sha256_file(Path(data.expert_review.evidence_path))
        except OSError:
            evidence_sha = "missing"
    return artifacts.fingerprint_of(
        {
            "contentFingerprint": content_digest,
            "qualityThreshold": round(float(data.quality_threshold), 9),
            "anchorHoldoutRatio": round(float(data.anchor_holdout_ratio), 9),
            "expertReview": {
                "mode": data.expert_review.mode,
                "sampleRate": round(float(data.expert_review.sample_rate), 9),
                "evidenceSha256": evidence_sha,
            },
            "mode": data.mode,
            "maxRounds": int(data.max_rounds),
            "seed": int(data.seed),
            "embedModel": data.embed_model,
            "topK": int(data.top_k),
            "synthesisModel": data.model.model if data.model else None,
            "simulatorModel": data.simulator_model,
            "envProfile": data.env_profile,
            "family": data.family,
            "split": data.split,
            "tasksetSchema": "env-task-v1",
        }
    )


def _assemble_package(
    data: SynthesisInput,
    domain: Dict[str, Any],
    draft: Dict[str, Any],
    version: int,
    bundle: SeedBundle,
    backend: CharNgramEmbeddingBackend,
    curation_resolved: bool,
) -> Any:
    try:
        return build_env_package(
            domain,
            draft,
            ref_id=data.ref_id,
            version=version,
            profile=data.env_profile,
            seed=data.seed,
            mode=data.mode,
            max_rounds=data.max_rounds,
            anchor_ratio=data.anchor_holdout_ratio,
            synthesis_model=data.model.model if data.model else None,
            simulator_model=data.simulator_model,
            backend=backend,
            source_files=bundle.files,
            seed_manifest_sha256=bundle.manifest_sha256,
            curation_resolved=curation_resolved,
            expert_review={
                "mode": data.expert_review.mode,
                "sampleRate": float(data.expert_review.sample_rate),
            },
        )
    except EnvPackageError as error:
        raise BuildFailure(
            f"env package assembly failed: {error}",
            stage="package",
            diagnostics={"seedManifestSha256": bundle.manifest_sha256},
        ) from error


def _taskset_payload(
    data: SynthesisInput,
    draft: Dict[str, Any],
    bundle: SeedBundle,
    release_info: Dict[str, Any],
    gym_tasks: List[Dict[str, Any]],
) -> Dict[str, Any]:
    tasks = copy.deepcopy(gym_tasks)
    return {
        "schemaVersion": "env-task-v1",
        "gymContract": {
            "design": gym_contract.GYM_CONTRACT_DESIGN,
            "sections": list(gym_contract.GYM_CONTRACT_SECTIONS),
            "taskSchema": gym_contract.GYM_TASK_SCHEMA,
        },
        "refId": "{}-env".format(data.ref_id),
        "family": data.family,
        "profile": data.env_profile,
        "split": data.split,
        "envRelease": {
            "refId": release_info["refId"],
            "version": release_info["version"],
            "fingerprint": release_info["fingerprint"],
            "assetSha256": release_info["assetSha256"],
        },
        "source": {
            "seedManifestSha256": bundle.manifest_sha256,
            "curatorSchema": "parsed-doc-v2",
        },
        "tasks": tasks,
        "itemCount": len(tasks),
    }


def _build_report(
    data: SynthesisInput,
    seed_info: Dict[str, Any],
    curation: Dict[str, Any],
    synthesis_info: Dict[str, Any],
    gate_report: Dict[str, Any],
    release_info: Dict[str, Any],
    taskset_info: Dict[str, Any],
    draft: Dict[str, Any],
) -> Dict[str, Any]:
    split = curation.get("anchorSplit") or {}
    return {
        "schemaVersion": "env-build-report-v1",
        "refId": data.ref_id,
        "family": data.family,
        "split": data.split,
        "profile": data.env_profile,
        "seedImport": seed_info,
        "params": {
            "mode": data.mode,
            "maxRounds": data.max_rounds,
            "seed": data.seed,
            "qualityThreshold": data.quality_threshold,
            "anchorHoldoutRatio": data.anchor_holdout_ratio,
            "embedModel": data.embed_model,
            "topK": data.top_k,
            "expertReview": {
                "mode": data.expert_review.mode,
                "sampleRate": data.expert_review.sample_rate,
            },
        },
        "curation": curation,
        "synthesis": synthesis_info,
        "gates": gate_report,
        "anchors": {
            "anchorCount": len(draft.get("anchors") or []),
            "taskCount": len(draft.get("tasks") or []),
            "anchorTicketIds": list(split.get("anchorTicketIds") or []),
            "groundingTicketIds": list(split.get("groundingTicketIds") or []),
            "overlap": bool(split.get("overlap")),
        },
        "distribution": gate_report.get("distribution"),
        "envRelease": {
            key: release_info.get(key)
            for key in ("refId", "version", "fingerprint", "assetSha256")
        },
        "taskset": taskset_info,
    }


def _run_build(
    data: SynthesisInput,
    *,
    find_version: Optional[Callable[[str, str], Optional[int]]] = None,
    next_version: Optional[Callable[[str], int]] = None,
) -> Dict[str, Any]:
    bundle, seed_info = _load_seed(data)
    domain, curation = _prepare_domain(bundle, data)
    backend = CharNgramEmbeddingBackend()
    client = None
    if data.mode == "agent":
        if data.model is None:
            raise BuildFailure("mode=agent is missing model configuration", stage="synthesis")
        client = OpenAICompatibleChatClient(
            base_url=data.model.base_url,
            model=data.model.model,
            api_key_env=data.model.api_key_env,
            temperature=data.model.temperature,
        )

    try:
        result = synthesize(
            domain,
            mode=data.mode,
            max_rounds=data.max_rounds,
            client=client,
            seed=data.seed,
            anchor_ratio=data.anchor_holdout_ratio,
            backend=backend,
            profile=data.env_profile,
            quality_threshold=data.quality_threshold,
            expert_review={
                "mode": data.expert_review.mode,
                "sampleRate": float(data.expert_review.sample_rate),
            },
            expert_evidence_path=data.expert_review.evidence_path,
        )
    except SynthesisError as error:
        raise BuildFailure(
            f"Synthesis failed: {error}",
            stage="synthesis",
            diagnostics={"seedManifestSha256": bundle.manifest_sha256},
        ) from error

    synthesis_info = {
        "status": result.status,
        "rounds": list(result.rounds),
        "failedChecks": list(result.failed_checks),
        "toolCalls": list(result.tool_calls),
        "warnings": list(result.warnings),
        "questionCount": len(result.questions),
    }
    if result.status == "needs_curation":
        raise BuildFailure(
            "The synthesizer reported blocking under-specified issues that require human clarification",
            stage="curation",
            questions=list(result.questions),
            diagnostics={
                "seedManifestSha256": bundle.manifest_sha256,
                "synthesis": synthesis_info,
                "curation": curation,
            },
        )
    if (
        result.status != "synthesized"
        or result.draft is None
        or result.failed_checks
    ):
        raise BuildFailure(
            "Synthesis failed (synthesis_failed)",
            stage="synthesis",
            questions=list(result.questions),
            diagnostics={
                "seedManifestSha256": bundle.manifest_sha256,
                "synthesis": synthesis_info,
                "curation": curation,
            },
        )

    draft = sanitize_draft(result.draft)
    try:
        validate_draft(draft)
    except SynthesisError as error:
        raise BuildFailure(
            f"Draft validation failed: {error}",
            stage="synthesis",
            diagnostics={
                "seedManifestSha256": bundle.manifest_sha256,
                "synthesis": synthesis_info,
            },
        ) from error

    curation_resolved = not interview.blocking_questions(domain)
    prelim = _assemble_package(
        data, domain, draft, 1, bundle, backend, curation_resolved
    )

    try:
        gate_report = run_gates(
            prelim,
            quality_threshold=data.quality_threshold,
            expert_review={
                "mode": data.expert_review.mode,
                "sampleRate": float(data.expert_review.sample_rate),
            },
            expert_evidence_path=data.expert_review.evidence_path,
        )
    except GateError as error:
        raise BuildFailure(
            f"Invalid gate input: {error}",
            stage="gates",
            diagnostics={"seedManifestSha256": bundle.manifest_sha256},
        ) from error
    if not gate_report.get("passed"):
        raise BuildFailure(
            "Gate check failed: {}".format(", ".join(gate_report.get("failedChecks") or [])),
            stage="gates",
            gate_report=gate_report,
            diagnostics={"seedManifestSha256": bundle.manifest_sha256},
        )
    with artifacts.release_lock(data.asset_root, data.ref_id):
        build_fingerprint = _build_fingerprint(prelim.content_digest, data)
        existing_version = (
            find_version(data.ref_id, build_fingerprint)
            if find_version is not None
            else artifacts.find_release_version(
                data.asset_root, data.ref_id, build_fingerprint
            )
        )
        version = (
            existing_version
            if existing_version is not None
            else (
                next_version(data.ref_id)
                if next_version is not None
                else artifacts.next_release_version(data.asset_root, data.ref_id)
            )
        )
        package = copy.deepcopy(prelim)
        package.manifest["version"] = int(version)
        try:
            package = finalize_bundle(package, gate_report)
        except EnvPackageError as error:
            raise BuildFailure(
                f"Freeze failed: {error}",
                stage="package",
                diagnostics={"seedManifestSha256": bundle.manifest_sha256},
            ) from error

        release_root = Path(data.asset_root)
        release_dir = release_root / data.ref_id / "v{}".format(version)
        try:
            release = write_release(str(release_dir), package)
        except EnvPackageError as error:
            raise BuildFailure(
                f"release write failed: {error}",
                stage="release",
                diagnostics={"seedManifestSha256": bundle.manifest_sha256},
            ) from error
        release_info = {
            "refId": data.ref_id,
            "version": int(version),
            "fingerprint": package.content_fingerprint,
            "assetDir": release["assetDir"],
            "assetSha256": package.asset_sha256,
        }

        taskset_payload = _taskset_payload(
            data, draft, bundle, release_info, package.gym_tasks
        )
        taskset_fingerprint = artifacts.fingerprint_of(taskset_payload)
        taskset_info = {
            "refId": "{}-env".format(data.ref_id),
            "version": int(version),
            "fingerprint": taskset_fingerprint,
            "itemCount": len(taskset_payload["tasks"]),
        }

        env_payload = {
            "schemaVersion": "env-release-v1",
            "refId": data.ref_id,
            "version": int(version),
            "fingerprint": package.content_fingerprint,
            "assetSha256": package.asset_sha256,
            "buildFingerprint": build_fingerprint,
            "source": {
                "seedManifestSha256": bundle.manifest_sha256,
                "curatorSchema": "parsed-doc-v2",
            },
            "params": {
                "mode": data.mode,
                "maxRounds": data.max_rounds,
                "seed": data.seed,
                "qualityThreshold": data.quality_threshold,
                "anchorHoldoutRatio": data.anchor_holdout_ratio,
                "expertReview": {
                    "mode": data.expert_review.mode,
                    "sampleRate": data.expert_review.sample_rate,
                },
            },
            "taskset": taskset_info,
            "manifest": package.manifest,
            "auditManifest": package.audit_manifest,
            "gates": gate_report,
        }
        try:
            artifacts.write_release_metadata(release_dir, env_payload, taskset_payload)
        except artifacts.ArtifactConflictError as error:
            raise BuildFailure(
                f"release metadata conflict (F1 immutability): {error}",
                stage="freeze",
                diagnostics={"seedManifestSha256": bundle.manifest_sha256},
            ) from error

    report = _build_report(
        data,
        seed_info,
        curation,
        synthesis_info,
        gate_report,
        release_info,
        taskset_info,
        draft,
    )
    return {
        "schemaVersion": artifacts.COMPILE_RESULT_SCHEMA,
        "envRelease": release_info,
        "taskset": taskset_info,
        "buildReport": report,
    }


def _write_failure_files(
    snapshot_path: Path, data: SynthesisInput, failure: BuildFailure
) -> None:
    diagnostics: Dict[str, Any] = {
        "schemaVersion": "env-diagnostics-v1",
        "ok": False,
        "refId": data.ref_id,
        "stage": failure.stage,
        "message": str(failure),
        "params": {
            "mode": data.mode,
            "maxRounds": data.max_rounds,
            "seed": data.seed,
            "qualityThreshold": data.quality_threshold,
            "anchorHoldoutRatio": data.anchor_holdout_ratio,
            "expertReview": {
                "mode": data.expert_review.mode,
                "sampleRate": data.expert_review.sample_rate,
            },
        },
    }
    diagnostics.update(failure.diagnostics)
    if failure.gate_report is not None:
        diagnostics["gateReport"] = failure.gate_report
    try:
        _write_json(_diagnostics_path(snapshot_path), diagnostics)
    except OSError as error:
        print(
            "[env-synthesis-worker] Failed to write diagnostics file: {}".format(error),
            file=sys.stderr,
        )
    if failure.questions:
        try:
            interview.write_curation_questions(
                str(_curation_path(snapshot_path)),
                {"questions": failure.questions},
                seed_manifest_sha256=str(
                    failure.diagnostics.get("seedManifestSha256") or ""
                ),
            )
        except OSError as error:
            print(
                "[env-synthesis-worker] Failed to write curation file: {}".format(error),
                file=sys.stderr,
            )


def run_worker(
    argv: List[str],
    *,
    data: Optional[SynthesisInput] = None,
    require_store: bool = False,
    find_version: Optional[Callable[[str, str], Optional[int]]] = None,
    next_version: Optional[Callable[[str], int]] = None,
    after_build: Optional[Callable[[SynthesisInput, Dict[str, Any]], None]] = None,
) -> int:
    if len(argv) != 2:
        print(
            "usage: python -m data_designer.pipelines.envsynth.cli.env_compile "
            "<input.json> <compile-result.json>",
            file=sys.stderr,
        )
        return EXIT_USAGE
    snapshot_path = Path(argv[1])
    if not snapshot_path.is_absolute() or snapshot_path.exists():
        print(
            "[env-synthesis-worker] snapshot must be an absolute path that does not yet exist",
            file=sys.stderr,
        )
        return EXIT_USAGE
    if data is None:
        try:
            data = parse_input(argv[0], require_store=require_store)
        except WorkerError as error:
            print(
                f"[env-synthesis-worker] Invalid input (build not started): {error}",
                file=sys.stderr,
            )
            return EXIT_USAGE

    try:
        snapshot = _run_build(
            data, find_version=find_version, next_version=next_version
        )
        if after_build is not None:
            after_build(data, snapshot)
    except BuildFailure as error:
        _write_failure_files(snapshot_path, data, error)
        print(f"[env-synthesis-worker] Build failed [{error.stage}]: {error}", file=sys.stderr)
        return EXIT_FAILED
    except Exception as error:  # never emit a usable-looking result on surprise
        failure = BuildFailure(str(error), stage="unexpected")
        try:
            _write_failure_files(snapshot_path, data, failure)
        except Exception:
            pass
        print(f"[env-synthesis-worker] Unexpected exception: {error}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return EXIT_FAILED
    try:
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = snapshot_path.with_name(snapshot_path.name + ".tmp")
        try:
            temp_path.write_bytes(json_bytes(snapshot))
            os.replace(str(temp_path), str(snapshot_path))
        finally:
            if temp_path.exists():
                temp_path.unlink()
    except OSError as error:
        print(f"[env-synthesis-worker] Failed to write snapshot: {error}", file=sys.stderr)
        return EXIT_FAILED

    release = snapshot["envRelease"]
    print(
        "[env-synthesis-worker] ok: env_release/{}/v{} fingerprint={} "
        "taskset=v{}".format(
            release["refId"],
            release["version"],
            release["fingerprint"],
            snapshot["taskset"]["version"],
        )
    )
    return EXIT_OK


def main(argv: Optional[List[str]] = None) -> int:
    return run_worker(sys.argv[1:] if argv is None else list(argv))


if __name__ == "__main__":
    sys.exit(main())
