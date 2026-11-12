"""HF single-turn seed → benchmark-source-v1 importer / scoring re-expression / augment adapter.

Input is the trusted seed material produced by ``HfSingleTurnSeedAdapter``
(either the in-repo synthetic fixture or a locally-held authorized HF snapshot
export — never redistributed). The importer freezes one family's rows into a
``benchmark-source-v1`` manifest; the D9 admission marker travels per case in
the family metadata block and gates every derivation entry point.

Re-expression (R1): the deterministic ``score.ts`` matchers module's source
text is embedded verbatim into the frozen ``unit_test`` ``test_script`` (the
scoring subprocess must stay import-free); per-benchmark assertion bodies are
composed from the trusted gold read back from the pinned snapshot. simpleqa /
healthbench freeze as ``llm_rubric`` with the full rubric text and
``pass_threshold`` in the payload (D7) — the judge endpoint is never part of
the published task.

License (T3 precedent): upstream licensing is archived, not a gate — internal
evaluation use; the license decision itself must be explicit. Real-subset
releases must stay non-redistributable.

Contamination red line: every derived ``task_text`` is screened against the
fingerprint set of the seed texts (exact ``normalized_text_sha256`` match) so
seed text never re-enters a published item unchanged.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..contracts import (
    BenchmarkError,
    canonical_sha256,
    file_sha256,
    normalized_text_sha256,
    require_nonempty,
    require_sha,
    require_split,
)
from ..runners._single_turn_common import (
    SUPPORTED_SPEC_KINDS,
    VERSION as RUNNER_VERSION,
)
from ..runtime import lock_runtime

# D9 per-operator admission criteria are finalized in the plan M2 enhanced operator table (§2 M2).
FAMILY_OPERATIONS: Dict[str, tuple] = {
    "reasoning_qa": (
        "numeric_perturbation", "constraint_tightening", "language_transfer",
        "distractor_injection",
    ),
    "knowledge_qa": (
        "option_permutation", "distractor_replacement", "paraphrase",
        "alias_expansion",
    ),
    "code_gen": (
        "signature_variation", "edge_case_tightening", "docstring_perturbation",
    ),
    "health_qa": ("rubric_perturbation", "scenario_variation"),
}
SINGLE_TURN_FAMILIES = tuple(FAMILY_OPERATIONS)

SUPPORTED_SEED_SOURCES = ("synthetic-fixture", "upstream-checkout")
_SNAPSHOT_FORMAT = "hf-single-turn-snapshot-v1"
_JUDGE_BENCHMARKS = frozenset({"simpleqa", "healthbench"})
_DERIVED_KEYS = (
    "source_task_id", "task_text", "tools_available", "success_criteria",
    "difficulty", "verifier_spec",
)

# Scoring re-expression material: the matchers module source text is embedded verbatim in the frozen test_script.
_MATCHERS_SOURCE = (
    Path(__file__).with_name("hf_single_turn_matchers.py")
    .read_text(encoding="utf-8")
)


def derive_release(
    seed_source: str,
    seed_rows: List[Dict[str, Any]],
    dataset_id: Optional[str] = None,
    snapshot_sha256: Optional[str] = None,
) -> str:
    """D2 release semantics: fixture SHA-8 or ``hf-<flattened-name>-<snapshot SHA-8>``."""
    if seed_source == "synthetic-fixture":
        return "synthetic-fixture:" + canonical_sha256(list(seed_rows))[:8]
    if seed_source == "upstream-checkout":
        flat = re.sub(r"[^0-9a-z]+", "-", str(dataset_id or "").lower()).strip("-")
        if not flat or not snapshot_sha256:
            raise BenchmarkError(
                "upstream-checkout release needs dataset_id and snapshot SHA-256"
            )
        return "hf-" + flat + "-" + require_sha(snapshot_sha256, "snapshot")[:8]
    raise BenchmarkError(f"unknown hf_single_turn seed_source: {seed_source!r}")


def _seed_case(row: Dict[str, Any], family: str) -> Dict[str, Any]:
    if not isinstance(row, dict):
        raise BenchmarkError("seed row must be an object")
    if row.get("family") != family:
        raise BenchmarkError(
            f"seed row family mismatch: {row.get('seed_id')!r} is "
            f"{row.get('family')!r}, not {family!r}"
        )
    admission = row.get("admission")
    if admission not in ("expressible", "excluded"):
        raise BenchmarkError("seed row lacks a valid admission marker")
    exclusion_reason = row.get("exclusion_reason")
    if admission == "excluded":
        require_nonempty(exclusion_reason, "exclusion_reason")
    elif exclusion_reason not in (None, ""):
        raise BenchmarkError("expressible rows must not carry an exclusion reason")
    source_task_id = require_nonempty(row.get("seed_id"), "seed seed_id")
    materialization = row.get("materialization")
    if not isinstance(materialization, dict):
        raise BenchmarkError(
            f"seed row lacks a materialization record: {source_task_id}"
        )
    task_text = str(row.get("task_text") or "")
    if admission == "expressible":
        task_text = require_nonempty(task_text, "seed task_text")
    tools = row.get("tools_available")
    if not isinstance(tools, list) or not all(isinstance(t, str) for t in tools):
        raise BenchmarkError("seed tools_available must be a string list")
    return {
        "source_task_id": source_task_id,
        "task_text": task_text,
        "tools_available": list(tools),
        "success_criteria": require_nonempty(
            row.get("success_criteria"), "seed success_criteria"
        ),
        "difficulty": require_nonempty(row.get("difficulty"), "seed difficulty"),
        "initial_state_sha256": canonical_sha256(
            {
                "stage": "seed-prompt",
                "benchmark": row.get("benchmark"),
                "task_text": task_text,
                "materialization": materialization,
            }
        ),
        "expected_state_sha256": canonical_sha256(
            {
                "success_criteria": row.get("success_criteria"),
                "admission": admission,
            }
        ),
        "hf_single_turn": {
            "benchmark": require_nonempty(row.get("benchmark"), "seed benchmark"),
            "admission": admission,
            "exclusion_reason": exclusion_reason,
            "materialization": dict(materialization),
        },
    }


def import_hf_single_turn_source(
    seed_rows: List[Dict[str, Any]],
    *,
    family: str,
    release: str,
    split: str,
    license_record: Dict[str, Any],
    seed_source: str,
    dataset_id: str,
    snapshot_sha256: Optional[str] = None,
    archive_path: Optional[str] = None,
    archive_sha256: Optional[str] = None,
) -> Dict[str, Any]:
    """Freeze one family's trusted seed rows into a benchmark-source-v1 manifest."""
    if family not in FAMILY_OPERATIONS:
        raise BenchmarkError(f"unknown single-turn family: {family!r}")
    if seed_source not in SUPPORTED_SEED_SOURCES:
        raise BenchmarkError(f"unknown hf_single_turn seed_source: {seed_source!r}")
    split = require_split(split)
    release = require_nonempty(release, "release")
    if not isinstance(seed_rows, list) or not seed_rows:
        raise BenchmarkError(f"{family} source needs non-empty seed rows")
    if not isinstance(license_record, dict) or (
        not require_nonempty(license_record.get("id"), "license.id")
        or type(license_record.get("derive_allowed")) is not bool
    ):
        raise BenchmarkError("hf_single_turn license decision required (T3)")
    expected_release = derive_release(
        seed_source, seed_rows, dataset_id, snapshot_sha256
    )
    if release != expected_release:
        raise BenchmarkError(
            f"hf_single_turn release pin mismatch: {release!r} != "
            f"{expected_release!r}"
        )
    cases = [_seed_case(row, family) for row in seed_rows]
    if len({case["source_task_id"] for case in cases}) != len(cases):
        raise BenchmarkError("duplicate hf_single_turn seed IDs")
    if not any(
        case["hf_single_turn"]["admission"] == "expressible" for case in cases
    ):
        raise BenchmarkError(f"{family} source has no expressible cases")
    source: Dict[str, Any] = {
        "schema_version": "benchmark-source-v1",
        "family": family,
        "release": release,
        "split": split,
        "license": dict(license_record),
        "seed_source": seed_source,
        "dataset_id": require_nonempty(dataset_id, "dataset_id"),
        "exclusion_text_sha256": sorted(
            {
                normalized_text_sha256(row.get("task_text"))
                for row in seed_rows
                if str(row.get("task_text") or "").strip()
            }
        ),
        "cases": cases,
    }
    if archive_path is not None or archive_sha256 is not None:
        if archive_path is None or archive_sha256 is None:
            raise BenchmarkError("archive pin needs path and SHA-256 together")
        source["source_snapshot"] = {
            "path": str(archive_path),
            "sha256": require_sha(archive_sha256, "source_snapshot.sha256"),
            "format": _SNAPSHOT_FORMAT,
            "dataset_id": source["dataset_id"],
            "seed_source": seed_source,
        }
    return source


def verify_hf_single_turn_source_manifest(source: Dict[str, Any]) -> None:
    """Re-derive the manifest from the pinned snapshot archive and compare pins."""
    from ....sources.hf_single_turn_seed import HfSingleTurnSeedAdapter

    snapshot = source.get("source_snapshot")
    if (
        not isinstance(snapshot, dict)
        or snapshot.get("format") != _SNAPSHOT_FORMAT
        or not isinstance(snapshot.get("path"), str)
    ):
        raise BenchmarkError("hf_single_turn source needs a pinned seed archive")
    adapter = HfSingleTurnSeedAdapter.from_archive(
        snapshot["path"], seed_source=snapshot.get("seed_source")
    )
    if not isinstance(snapshot.get("sha256"), str) or not snapshot["sha256"]:
        raise BenchmarkError("hf_single_turn source snapshot sha256 pin missing")
    if adapter.archive_sha256 != snapshot["sha256"]:
        raise BenchmarkError("hf_single_turn seed archive SHA-256 pin mismatch")
    if adapter.dataset_id != snapshot.get("dataset_id"):
        raise BenchmarkError("hf_single_turn seed archive dataset ID drift")
    rows = [
        row for row in adapter.load_seed_tasks()
        if row["family"] == source.get("family")
    ]
    expected = import_hf_single_turn_source(
        rows,
        family=source.get("family"),
        release=source.get("release"),
        split=source.get("split"),
        license_record=source.get("license"),
        seed_source=snapshot.get("seed_source"),
        dataset_id=snapshot.get("dataset_id"),
        snapshot_sha256=snapshot.get("sha256"),
        archive_path=snapshot.get("path"),
        archive_sha256=snapshot.get("sha256"),
    )
    if (
        source.get("release") != expected["release"]
        or source.get("license") != expected["license"]
        or source.get("exclusion_text_sha256") != expected["exclusion_text_sha256"]
        or source.get("cases") != expected["cases"]
    ):
        raise BenchmarkError("hf_single_turn source metadata drift")


def screen_hf_single_turn_contamination(
    derived_rows: List[Dict[str, Any]], seed_rows: List[Dict[str, Any]]
) -> None:
    """Reject derived rows whose task text exactly matches any seed text."""
    seed_fingerprints = {
        normalized_text_sha256(row.get("task_text")) for row in seed_rows
    }
    for row in derived_rows:
        if normalized_text_sha256(row.get("task_text")) in seed_fingerprints:
            raise BenchmarkError(
                "derived task text matches a seed text; seed text must never "
                "re-enter a published item"
            )


def _pin(value: Any, name: str) -> Dict[str, str]:
    if not isinstance(value, dict):
        raise BenchmarkError(f"{name} must be an object")
    path = Path(require_nonempty(value.get("path"), f"{name}.path"))
    sha = require_sha(value.get("sha256"), f"{name}.sha256")
    if not path.is_absolute() or not path.is_file() or file_sha256(path) != sha:
        raise BenchmarkError(f"hf_single_turn {name} missing or SHA-256 drift")
    return {"path": str(path), "sha256": sha}


def _clean_relative(value: Any) -> str:
    relative = str(value)
    parts = Path(relative).parts
    if (
        Path(relative).is_absolute()
        or ".." in parts
        or not all(part not in ("", ".", "..") for part in parts)
    ):
        raise BenchmarkError(
            f"deliverable path must be a clean relative path: {relative!r}"
        )
    return relative


def _require_admitted(source: Dict[str, Any], source_case: Dict[str, Any]) -> None:
    metadata = source_case.get("hf_single_turn")
    task_id = source_case.get("source_task_id")
    if not isinstance(metadata, dict):
        raise BenchmarkError(
            f"hf_single_turn case {task_id!r} lacks family metadata; "
            "import with the HF single-turn seed adapter"
        )
    if metadata.get("admission") != "expressible":
        raise BenchmarkError(
            f"hf_single_turn case {task_id!r} is excluded from derivation "
            f"({metadata.get('exclusion_reason', 'unknown')})"
        )


def _pinned_adapter(source: Dict[str, Any]):
    from ....sources.hf_single_turn_seed import HfSingleTurnSeedAdapter

    snapshot = source.get("source_snapshot")
    if (
        not isinstance(snapshot, dict)
        or not isinstance(snapshot.get("path"), str)
    ):
        raise BenchmarkError("hf_single_turn derive needs a pinned seed archive")
    adapter = HfSingleTurnSeedAdapter.from_archive(
        snapshot["path"], seed_source=snapshot.get("seed_source")
    )
    if not isinstance(snapshot.get("sha256"), str) or not snapshot["sha256"]:
        raise BenchmarkError("hf_single_turn derive snapshot sha256 pin missing")
    if adapter.archive_sha256 != snapshot["sha256"]:
        raise BenchmarkError("hf_single_turn seed archive SHA-256 pin mismatch")
    if adapter.dataset_id != snapshot.get("dataset_id"):
        raise BenchmarkError("hf_single_turn seed archive dataset ID drift")
    return adapter


def _unit_test_script(
    benchmark: str, gold: Dict[str, Any], deliverable: str
) -> str:
    """frozen test_script = matchers source + per-benchmark assertion body.

    ``deliverable`` is a per-task relative path (D6); the assertion body reads it from the workspace
    (run_unit_tests writes the artifact file dictionary into the temporary workspace under the original keys).
    """
    path_literal = repr(deliverable)
    if benchmark == "humaneval":
        body = (
            f"code = open({path_literal}, encoding=\"utf-8\").read()\n"
# exec into globals(): test_code assertions reference candidate function names at module scope.
            "exec(compile(code, \"solution.py\", \"exec\"), globals())\n"
            + str(gold.get("test_code") or "").strip()
            + "\n"
        )
    elif benchmark in ("gsm8k", "mgsm"):
        body = (
            f"text = open({path_literal}, encoding=\"utf-8\").read()\n"
            "assert numeric_answer_equal(\n"
            '    extract_after_marker(text, "the answer is"), '
            f"{str(gold.get('answer') or '')!r}\n"
            '), "answer mismatch"\n'
        )
    elif benchmark == "math500":
        answer = str(gold.get("answer") or "")
        body = (
            f"text = open({path_literal}, encoding=\"utf-8\").read()\n"
            "answer = extract_boxed(text)\n"
            f"assert answer == {answer!r} or normalize_numeric(answer) == "
            f"normalize_numeric({answer!r}), \"answer mismatch\"\n"
        )
    elif benchmark == "drop":
        spans = [str(span) for span in (gold.get("answer_spans") or [])]
        if not spans:
            spans = [str(gold.get("answer") or "")]
        body = (
            f"text = open({path_literal}, encoding=\"utf-8\").read()\n"
            f"assert fuzzy_span_correct(text, {spans!r}), \"answer mismatch\"\n"
        )
    elif benchmark in ("mmlu", "mmlu_pro", "gpqa"):
# mmlu gold key is choices; mmlu_pro/gpqa use options (adapter gold_fields).
        options = gold.get("options") or gold.get("choices") or []
        allowed = "ABCDEFGHIJ"[: len(options)]
        body = (
            f"text = open({path_literal}, encoding=\"utf-8\").read()\n"
            "assert multichoice_correct(\n"
            f"    text, {str(gold.get('answer_letter') or '')!r}, {allowed!r}\n"
            '), "wrong option"\n'
        )
    elif benchmark == "triviaqa":
        aliases = [str(alias) for alias in (gold.get("aliases") or [])]
        body = (
            f"text = open({path_literal}, encoding=\"utf-8\").read()\n"
            f"assert trivia_answer_correct(text, {aliases!r}), \"wrong answer\"\n"
        )
    else:
        raise BenchmarkError(f"benchmark {benchmark!r} is not a unit_test member")
    return _MATCHERS_SOURCE + "\n\n" + body


def _gold_deliverable(
    benchmark: str, gold: Dict[str, Any], deliverable: str
) -> Dict[str, str]:
    """Reference audit deliverable: the expected-pass response text assembled from trusted gold."""
    if benchmark == "humaneval":
        return {
            deliverable: (
                str(gold.get("prompt") or "")
                + str(gold.get("canonical_solution") or "")
                + "\n"
            )
        }
    if benchmark in ("gsm8k", "mgsm", "simpleqa"):
        return {deliverable: f"The answer is {gold.get('answer')}."}
    if benchmark == "math500":
        return {
            deliverable: f"The answer is \\boxed{{{gold.get('answer')}}}."
        }
    if benchmark in ("mmlu", "mmlu_pro", "gpqa"):
        return {deliverable: f"The answer is ({gold.get('answer_letter')})."}
    if benchmark == "triviaqa":
        return {deliverable: str((gold.get("aliases") or [""])[0])}
    if benchmark == "drop":
        spans = [str(span) for span in (gold.get("answer_spans") or [])]
        text = spans[0] if spans else str(gold.get("answer") or "")
        return {deliverable: text}
    if benchmark == "healthbench":
        return {deliverable: str(gold.get("reference_response") or "")}
    raise BenchmarkError(f"benchmark {benchmark!r} lacks a gold deliverable rule")


def _negative_deliverable(
    benchmark: str, gold: Dict[str, Any], deliverable: str
) -> Dict[str, str]:
    """Negative audit deliverable: deterministically scores 0 (empty response / NotImplemented solution)."""
    if benchmark == "humaneval":
        return {
            deliverable: (
                str(gold.get("prompt") or "") + "    raise NotImplementedError\n"
            )
        }
    return {deliverable: ""}


def candidate_deliverable_path(seed_id: str, benchmark: str) -> str:
    """D6 per-task deliverable relative path (the public side can derive it from benchmark.source_task_id).

    When multiple tasks share the same candidateRoot evaluation, a shared ``response.txt`` would overwrite other tasks;
    the per-task path ``deliverables/<source_task_id>/…`` is deterministically
    derived from source_task_id (':' → '-'), so the candidate side and trusted side need no extra negotiation.
    """
    slug = require_nonempty(seed_id, "seed_id").replace(":", "-")
    name = "solution.py" if benchmark == "humaneval" else "response.txt"
    return f"deliverables/{slug}/{name}"


def derive_hf_single_turn_spec(
    source: Dict[str, Any], source_case: Dict[str, Any], directive: Any
) -> Dict[str, Any]:
    """Trusted-side re-expression: HF snapshot gold → frozen verifier_spec.

    ``directive`` accepts ``timeout_s`` (unit_test execution timeout, pinned)
    and ``pass_threshold`` (llm_rubric judge threshold, default 0.5). The
    candidate deliverable path follows the family convention (D6):
    ``response.txt``, or ``solution.py`` for code_gen. Gold is read from the
    pinned snapshot via the adapter and never travels beyond the trusted case
    manifest.
    """
    if not isinstance(directive, dict) or set(directive) - {
        "timeout_s", "pass_threshold",
    }:
        raise BenchmarkError(
            "hf_single_turn directive accepts only timeout_s and pass_threshold"
        )
    timeout = directive.get("timeout_s", 30.0)
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or timeout <= 0
    ):
        raise BenchmarkError("hf_single_turn timeout_s must be a positive number")
    threshold = directive.get("pass_threshold", 0.5)
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not 0.0 <= float(threshold) <= 1.0
    ):
        raise BenchmarkError("hf_single_turn pass_threshold must be in [0,1]")
    _require_admitted(source, source_case)
    adapter = _pinned_adapter(source)
    seed_id = source_case["source_task_id"]
    benchmark = source_case["hf_single_turn"]["benchmark"]
    gold = adapter.gold_fields(seed_id)
    release = require_nonempty(source.get("release"), "release")
    if benchmark in _JUDGE_BENCHMARKS:
        if benchmark == "simpleqa":
            rubric = (
                "Does the response state the correct answer to the question? "
                f"The verified reference answer is: {gold.get('answer')}"
            )
        else:
            rubric = str(gold.get("rubric") or "")
        deliverable = candidate_deliverable_path(seed_id, benchmark)
        spec = {
            "kind": "llm_rubric",
            "generator_ref": f"hf-single-turn@{release}",
            "payload": {
                "artifact": {"path": deliverable, "format": "text"},
                "rubric": rubric,
                "pass_threshold": float(threshold),
                "score_scale": {"min": 1, "max": 5},
            },
        }
    else:
        deliverable = candidate_deliverable_path(seed_id, benchmark)
        spec = {
            "kind": "unit_test",
            "generator_ref": f"hf-single-turn@{release}",
            "payload": {
                "artifact": {"path": deliverable, "format": "text"},
                "test_script": _unit_test_script(benchmark, gold, deliverable),
                "timeout_s": float(timeout),
            },
        }
    gold_deliverable = _gold_deliverable(benchmark, gold, deliverable)
    return {
        "verifier_spec": spec,
        "gold_deliverable": gold_deliverable,
        "negative_deliverable": _negative_deliverable(
            benchmark, gold, deliverable
        ),
# build.py's derivation special case replaces the entire private block with this bundle, so initial must
# differ from the source case's seed initial (same shape as deepswe: augment must change state).
        "initial_state_sha256": canonical_sha256(
            {
                "stage": "single-turn-reexpression",
                "seed_initial": source_case["initial_state_sha256"],
                "verifier_kind": spec["kind"],
            }
        ),
        "expected_state_sha256": canonical_sha256(
            {"verifier_spec": spec, "gold_deliverable": gold_deliverable}
        ),
    }


def _hf_derived_variant(
    row: Dict[str, Any], index: int, strategy: str,
    environment: Dict[str, str], verifier: Dict[str, Any],
    audit: Dict[str, Any],
) -> Dict[str, Any]:
    if not isinstance(row, dict):
        raise BenchmarkError("derived row must be an object")
    missing = [key for key in _DERIVED_KEYS if row.get(key) in (None, "")]
    if missing:
        raise BenchmarkError(f"derived row missing fields: {missing}")
    spec = row["verifier_spec"]
    if not isinstance(spec, dict) or spec.get("kind") not in SUPPORTED_SPEC_KINDS:
        raise BenchmarkError(
            f"derived verifier_spec kind must be one of {SUPPORTED_SPEC_KINDS}"
        )
    if not isinstance(spec.get("payload"), dict):
        raise BenchmarkError("derived verifier_spec invalid")
    artifact = spec["payload"].get("artifact")
    if not isinstance(artifact, dict) or not str(artifact.get("path") or "").strip():
        raise BenchmarkError("derived verifier_spec needs a deliverable path")
    deliverable = _clean_relative(artifact["path"])
    source_task_id = require_nonempty(row["source_task_id"], "source_task_id")
    variant_id = require_nonempty(
        row.get("item_fingerprint") or f"{strategy}-{index:04d}", "variant_id"
    )
    tools = row["tools_available"]
    if not isinstance(tools, list) or not all(isinstance(t, str) for t in tools):
        raise BenchmarkError("derived tools_available must be a string list")
    return {
        "source_task_id": source_task_id,
        "variant_id": variant_id,
        "operation": strategy,
        "public": {
            "task_text": require_nonempty(row["task_text"], "task_text"),
            "tools_available": list(tools),
            "success_criteria": require_nonempty(
                row["success_criteria"], "success_criteria"
            ),
            "difficulty": require_nonempty(row["difficulty"], "difficulty"),
        },
        "private": {
            "initial_state_sha256": canonical_sha256(
                {
                    "environment_sha256": environment["sha256"],
                    "task_text": row["task_text"],
                }
            ),
            "expected_state_sha256": canonical_sha256(
                {"verifier_spec": spec, "gold_deliverable": row["gold_deliverable"]}
            ),
            "verifier_spec": spec,
            "deliverable_path": deliverable,
            "strategy": strategy,
        },
        "runtime": {
            "environment": dict(environment),
            "verifier": dict(verifier),
            "assets": [],
        },
        "audit": dict(audit),
    }


def build_hf_single_turn_augment_config(
    derived_rows: List[Dict[str, Any]],
    *,
    family: str,
    release: str,
    split: str,
    strategy: str,
    prompt_archive: Any,
    verifier_path: str,
    verifier_version: str = RUNNER_VERSION,
    verifier_command: Optional[List[str]] = None,
    audit_roots: Any,
) -> Dict[str, Any]:
    """Wrap re-expressed single-turn rows into a benchmark-augment-v1 config.

    ``prompt_archive`` is the pinned prompt-only tar.gz exported by
    ``HfSingleTurnSeedAdapter.export_prompt_archive`` (D5: gold never enters
    the archive) — the per-case environment pin. The verifier pin locks the
    family's native runner shim and its command. The trusted-side
    ``verifier_spec`` (including the re-expressed assertions / rubric) travels
    in each variant's ``private`` block only.
    """
    if family not in FAMILY_OPERATIONS:
        raise BenchmarkError(f"unknown single-turn family: {family!r}")
    if strategy not in FAMILY_OPERATIONS[family]:
        raise BenchmarkError(
            f"unsupported hf_single_turn operation {strategy!r} for {family}"
        )
    split = require_split(split)
    release = require_nonempty(release, "release")
    if not isinstance(derived_rows, list) or not derived_rows:
        raise BenchmarkError(f"{family} augment needs non-empty derived rows")
    environment_pin = _pin(prompt_archive, "environment")
    runner_path = Path(require_nonempty(verifier_path, "verifier_path"))
    if not runner_path.is_absolute() or not runner_path.is_file():
        raise BenchmarkError(f"{family} runner shim unavailable")
    command = list(
        verifier_command
        if verifier_command is not None
        else [sys.executable, str(runner_path)]
    )
    verifier = {
        "path": str(runner_path),
        "sha256": file_sha256(runner_path),
        "version": require_nonempty(verifier_version, "verifier_version"),
        "command": command,
    }
    if isinstance(audit_roots, dict):
        audits = [audit_roots] * len(derived_rows)
    elif isinstance(audit_roots, list) and len(audit_roots) == len(derived_rows):
        audits = audit_roots
    else:
        raise BenchmarkError("audit_roots must be one record or one per row")
    variants = [
        _hf_derived_variant(row, index, strategy, environment_pin, verifier, audit)
        for index, (row, audit) in enumerate(zip(derived_rows, audits))
    ]
    for variant in variants:
        lock_runtime(variant["runtime"])
    return {
        "schema_version": "benchmark-augment-v1",
        "family": family,
        "release": release,
        "split": split,
        "variants": variants,
    }


def assert_hf_single_turn_build(
    runtime: Dict[str, Any], private: Dict[str, Any], source_case: Dict[str, Any]
) -> None:
    """M2 gate: every new single-turn taskset pins the shim runner and a real spec."""
    metadata = source_case.get("hf_single_turn")
    task_id = source_case.get("source_task_id")
    if not isinstance(metadata, dict):
        raise BenchmarkError(
            f"hf_single_turn case {task_id!r} lacks family metadata; "
            "import with the HF single-turn seed adapter"
        )
    if metadata.get("admission") != "expressible":
        raise BenchmarkError(
            f"hf_single_turn case {task_id!r} is excluded from derivation "
            f"({metadata.get('exclusion_reason', 'unknown')})"
        )
    verifier = runtime.get("verifier")
    if (
        not isinstance(verifier, dict)
        or verifier.get("version") != RUNNER_VERSION
    ):
        raise BenchmarkError(
            "hf_single_turn derivation must pin the offline runner shim "
            f"({RUNNER_VERSION})"
        )
    spec = private.get("verifier_spec") if isinstance(private, dict) else None
    if not isinstance(spec, dict) or spec.get("kind") not in SUPPORTED_SPEC_KINDS:
        raise BenchmarkError(
            f"hf_single_turn trusted case lacks a {SUPPORTED_SPEC_KINDS} "
            f"verifier_spec: {task_id}"
        )
    benchmark = metadata.get("benchmark")
    expected_kind = "llm_rubric" if benchmark in _JUDGE_BENCHMARKS else "unit_test"
    if spec.get("kind") != expected_kind:
        raise BenchmarkError(
            f"hf_single_turn benchmark {benchmark!r} must freeze as "
            f"{expected_kind}, got {spec.get('kind')!r}: {task_id}"
        )


__all__ = [
    "FAMILY_OPERATIONS",
    "SINGLE_TURN_FAMILIES",
    "SUPPORTED_SEED_SOURCES",
    "SUPPORTED_SPEC_KINDS",
    "candidate_deliverable_path",
    "derive_release",
    "import_hf_single_turn_source",
    "verify_hf_single_turn_source_manifest",
    "screen_hf_single_turn_contamination",
    "derive_hf_single_turn_spec",
    "build_hf_single_turn_augment_config",
    "assert_hf_single_turn_build",
]
