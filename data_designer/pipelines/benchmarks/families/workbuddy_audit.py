"""workbuddy native-ingest audit (M1 ingest side; M3 extends reference/negative synthesis).

M1 provides the two ingest components from design §6.2/§9:

- ``materialize_public_inputs``: unpack each task's ``environment/workspace.tar.gz``,
  exclude ``tests/**`` and PRIVATE_MARKERS hits, compute per-file sha256, and produce
  the ``public_inputs_by_task`` manifest (path+sha references, no embedded content, D5);
- ``build_native_ingest_report``: per-task ingest report--native-ingest evidence
  (task_text fingerprint == seed fingerprint, replacing contamination screening, design §5), workspace pins,
  delivery type, and excluded decisions. After M2/M3/M4 scoring is implemented, add reference/negative/
  judge_calibration fields; the whole report is pinned by sha256 into the taskset payload
  (``ingest_report_sha256``).

M3 adds the two-arm synthesis and oracle loop from design §7.1–7.3:

- ``synthesize_reference_artifacts``: dispatch by delivery type (json=gold reserialization;
  xlsx=drug-inventory-style gold table member filtering; otherwise -> curation path);
- ``synthesize_negative_artifacts``: type-correct empty/garbage artifacts (§7.3 negative defense);
- ``iterate_reference_synthesis``: iterate-generate-verify with eval_core as the
  oracle (path-rewritten host-side execution, trusted audit only; authoritative scoring still uses the real
  runner end to end), <=3 rounds, still failing -> needs_curation / excluded, never silently;
- ``iterate_negative_check``: negative oracle scoring and assert score < reference
  score (conditional checks for false passes -> explicitly flagged, §7.3).
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import tarfile
import types
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..contracts import (
    PRIVATE_MARKERS,
    BenchmarkError,
    file_sha256,
    normalized_text_sha256,
    require_nonempty,
    require_sha,
)

INGEST_REPORT_SCHEMA = "workbuddy-ingest-report-v1"

SYNTH_OUTCOME_SYNTHESIZED = "synthesized"
SYNTH_OUTCOME_NEEDS_CURATION = "needs_curation"
SYNTH_OUTCOME_FAILED = "synthesis_failed"

_RECORD_ID_COLUMNS = ("RECORD_ID", "记录编号", "记录ID")
_SHEET_KEY_RE = re.compile(r"^sheet\d+_record_ids$")


class SynthesisUnavailable(Exception):
    """Reference artifact cannot be synthesized mechanically (use §7.2 LLM-assisted curation or the exclusion path)."""


def _is_private_path(relative: str) -> bool:
    parts = Path(relative).parts
    if parts and parts[0] == "tests":
        return True
    return any(
        marker in part.lower() for part in parts for marker in PRIVATE_MARKERS
    )


def materialize_public_inputs(
    records: List[Dict[str, Any]], out_root: str
) -> Dict[str, Any]:
    """Unpack each task workspace into a public input tree and produce a reference manifest.

    Returns ``{"dir": <out_root>, "by_task": {seed_id: [{"path", "sha256"}, ...]}}``.
    Output tree layout: ``<out_root>/<seed_id>/<workspace relative path>``. Workspace input
    is the task's public surface (the agent already has it in a real bench) and contains no gold/verifier.
    """
    root = Path(out_root)
    if not root.is_absolute():
        raise BenchmarkError("public inputs root must be absolute")
    by_task: Dict[str, List[Dict[str, str]]] = {}
    for record in records:
        if not isinstance(record, dict):
            raise BenchmarkError("native record must be an object")
        task_id = require_nonempty(record.get("seed_id"), "native seed_id")
        workspace = record.get("workspace")
        if not isinstance(workspace, dict):
            raise BenchmarkError(f"native record needs a workspace pin: {task_id}")
        archive = Path(require_nonempty(workspace.get("path"), "workspace.path"))
        sha = require_sha(workspace.get("sha256"), "workspace.sha256")
        if not archive.is_file() or file_sha256(archive) != sha:
            raise BenchmarkError(f"workspace missing or SHA-256 drift: {task_id}")
        task_dir = root / task_id
        task_dir.mkdir(parents=True, exist_ok=True)
        resolved_root = task_dir.resolve()
        entries: List[Dict[str, str]] = []
        with tarfile.open(archive, "r:gz") as tar:
            for member in tar.getmembers():
                if not member.isfile():
                    continue
                relative = Path(member.name)
                if relative.is_absolute() or ".." in relative.parts:
                    raise BenchmarkError(
                        f"workspace member escapes layout: {task_id}/{member.name}"
                    )
                name = relative.as_posix()
                if _is_private_path(name):
                    continue
                target = task_dir / relative
                if not target.resolve().is_relative_to(resolved_root):
                    raise BenchmarkError(
                        f"workspace member escapes root: {task_id}/{member.name}"
                    )
                stream = tar.extractfile(member)
                if stream is None:
                    raise BenchmarkError(
                        f"workspace member unreadable: {task_id}/{member.name}"
                    )
                data = stream.read()
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
                entries.append(
                    {
                        "path": name,
                        "sha256": hashlib.sha256(data).hexdigest(),
                    }
                )
        if not entries:
            raise BenchmarkError(f"workspace has no public inputs: {task_id}")
        by_task[task_id] = entries
    return {"dir": str(root), "by_task": by_task}


def build_native_ingest_report(
    records: List[Dict[str, Any]],
    *,
    release: str,
    public_inputs: Optional[Dict[str, Any]] = None,
    foreign_text_sha256: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Per-task ingest report (M1 field set; M2+ scoring fields are added by later milestones).

    ``foreign_text_sha256`` is the text fingerprint of items already ingested by other releases (cross-release collision
    check, design §5); items with a collision are marked excluded and never silently admitted.
    """
    foreign = {
        require_sha(value, "foreign text sha256")
        for value in (foreign_text_sha256 or [])
    }
    by_task_counts = {
        task_id: len(entries)
        for task_id, entries in (public_inputs or {}).get("by_task", {}).items()
    }
    tasks: List[Dict[str, Any]] = []
    for record in records:
        if not isinstance(record, dict):
            raise BenchmarkError("native record must be an object")
        task_id = require_nonempty(record.get("seed_id"), "native seed_id")
        text_sha = normalized_text_sha256(record.get("task_text"))
        collisions = sorted(foreign & {text_sha})
        excluded = bool(collisions)
        tasks.append(
            {
                "task_id": task_id,
                "deliverable_types": list(record.get("deliverable_types") or []),
                "native_evidence": {
                    "task_text_sha256": text_sha,
                    "seed_text_verbatim": True,
                    "foreign_release_collision": collisions,
                },
                "workspace_inputs": {
                    "file_count": by_task_counts.get(task_id),
                    "sha_pinned": True,
                },
                "excluded": excluded,
                "exclude_reason": (
                    "task text collides with another release's ingested text"
                    if excluded
                    else None
                ),
            }
        )
    return {
        "schema_version": INGEST_REPORT_SCHEMA,
        "release": require_nonempty(release, "release"),
        "tasks": tasks,
    }


# ── M3: reference/negative artifact synthesis (design §7.1–7.3) ──────────────────────────────────


def _required_artifacts(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    judge = record.get("judge")
    if not isinstance(judge, dict):
        raise BenchmarkError("native record needs a parsed judge document")
    artifacts = [
        artifact
        for artifact in (judge.get("artifacts") or [])
        if isinstance(artifact, dict) and artifact.get("required")
    ]
    if not artifacts:
        raise BenchmarkError("native record has no required artifacts")
    return artifacts


def _artifact_target(out_root: Path, relative: str) -> Path:
    target = out_root / relative
    if not target.resolve().is_relative_to(out_root.resolve()):
        raise BenchmarkError(f"deliverable path escapes root: {relative}")
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def _json_synth(gold: Dict[str, Any], target: Path) -> None:
    """json-type reference artifact: normalized reserialization of gold (§7.2 gold-mirror scoring family)."""
    target.write_text(
        json.dumps(gold, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _xlsx_input_rows(
    workspace_dir: Path, wanted_ids: set
) -> Tuple[List[str], Dict[str, tuple]]:
    """Locate the xlsx in the workspace input tree that contains all gold records, and read all rows.

    Returns ``(input column name list, {record_id: original row tuple})``; if no input workbook covers all
    gold ids -> ``SynthesisUnavailable`` (no guessing, no silence).
    """
    import openpyxl

    candidates = [
        path for path in sorted(workspace_dir.rglob("*.xlsx"))
        if not path.is_relative_to(workspace_dir / "output")
    ]
    best: Optional[Tuple[float, Path, List[str], Dict[str, tuple]]] = None
    for path in candidates:
        try:
            workbook = openpyxl.load_workbook(str(path), data_only=True)
        except Exception:
            continue
        for sheet_name in workbook.sheetnames:
            sheet = workbook[sheet_name]
            rows = list(sheet.iter_rows(values_only=True))
            if not rows:
                continue
            headers = [str(h).strip() if h is not None else "" for h in rows[0]]
            id_col = next(
                (
                    index
                    for index, header in enumerate(headers)
                    if header.upper() in _RECORD_ID_COLUMNS
                    or header in _RECORD_ID_COLUMNS
                ),
                None,
            )
            if id_col is None:
                continue
            by_id = {
                str(row[id_col]).strip(): row
                for row in rows[1:]
                if row and len(row) > id_col and row[id_col] is not None
            }
            coverage = len(wanted_ids & set(by_id)) / len(wanted_ids)
            if best is None or coverage > best[0]:
                best = (coverage, path, headers, by_id)
    if best is None or best[0] < 1.0:
        coverage = best[0] if best else 0.0
        raise SynthesisUnavailable(
            f"no input workbook covers all gold record ids "
            f"(best coverage {coverage:.4f} across {len(candidates)} candidates)"
        )
    return best[2], best[3]


_TABLE_CONTAINER_KEYS = ("output_sheets", "outputs", "sheets")


def _xlsx_tables_from_gold(gold: Dict[str, Any]) -> List[Tuple[str, Any]]:
    """Second xlsx gold shape: named-sheet container / record list -> (sheet name, table data).

    - Names under container keys (``outputs`` / ``output_sheets`` / ``sheets``) are authoritative
      sheet names;
    - Other top-level list-of-dict keys are fallback tables (one table per key).
    """
    tables: List[Tuple[str, Any]] = []
    for container in _TABLE_CONTAINER_KEYS:
        value = gold.get(container)
        if isinstance(value, dict):
            for name, table in value.items():
                tables.append((str(name), table))
    for key, value in gold.items():
        if key in _TABLE_CONTAINER_KEYS or _SHEET_KEY_RE.match(key):
            continue
        if (
            isinstance(value, list)
            and value
            and all(isinstance(row, dict) for row in value)
        ):
            tables.append((str(key), value))
    return tables


def _xlsx_cell(value: Any) -> Any:
    """openpyxl accepts only scalars; nested list/dict values are written as canonical JSON strings."""
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def _xlsx_write_table(sheet: Any, table: Any) -> bool:
    """Write table data to a worksheet; if it cannot be expressed -> False (the caller accumulates Curation)."""
    if isinstance(table, list):
        rows = [row for row in table if isinstance(row, dict)]
        if not rows:
            return False
        headers: List[str] = []
        for row in rows:
            for key in row:
                if key not in headers:
                    headers.append(key)
        sheet.append(headers)
        for row in rows:
            sheet.append([_xlsx_cell(row.get(header)) for header in headers])
        return True
    if isinstance(table, dict):
        if table and all(isinstance(value, list) for value in table.values()):
            sub_tables = list(table.items())
            width = 0
            materialized = []
            for name, sub in sub_tables:
                rows = [row for row in sub if isinstance(row, dict)]
                if not rows:
                    return False
                headers = []
                for row in rows:
                    for key in row:
                        if key not in headers:
                            headers.append(key)
                materialized.append((name, headers, rows))
                width = max(width, len(headers) + 1)
            sheet.append([name for name, _, _ in materialized])
            depth = max(len(rows) for _, _, rows in materialized)
            for index in range(depth):
                row: List[Any] = []
                for _, headers, rows in materialized:
                    if index < len(rows):
                        row.extend(
                            [_xlsx_cell(rows[index].get(h)) for h in headers]
                        )
                    row.extend([None] * (width - len(headers)))
                sheet.append(row)
            return True
        sheet.append(["key", "value"])
        for key, value in table.items():
            sheet.append([key, _xlsx_cell(value)])
        return True
    return False


def _xlsx_synth_table_projection(
    gold: Dict[str, Any], target: Path
) -> None:
    """Table-projection xlsx reference artifact: gold named table -> workbook (decided by oracle rounds)."""
    import openpyxl

    tables = _xlsx_tables_from_gold(gold)
    if not tables:
        raise SynthesisUnavailable(
            "xlsx gold has neither headers/sheetN_record_ids nor "
            "named table containers (long-tail shape needs curation)"
        )
    workbook = openpyxl.Workbook()
    workbook.remove(workbook.active)
    written_any = False
    for name, table in tables:
        safe_name = re.sub(r"[\\/*?:\[\]]", "_", name)[:31] or "sheet"
        sheet = workbook.create_sheet(safe_name)
        if _xlsx_write_table(sheet, table):
            written_any = True
    if not written_any:
        raise SynthesisUnavailable(
            "xlsx gold tables not expressible as sheets (needs curation)"
        )
    workbook.save(str(target))


def _xlsx_synth(
    gold: Dict[str, Any], workspace_dir: Path, target: Path
) -> None:
    """xlsx reference artifact: try drug-inventory table-member filtering first, then table projection."""
    try:
        _xlsx_synth_sheet_split(gold, workspace_dir, target)
        return
    except SynthesisUnavailable:
        pass
    _xlsx_synth_table_projection(gold, target)


def _xlsx_synth_sheet_split(
    gold: Dict[str, Any], workspace_dir: Path, target: Path
) -> None:
    """drug-inventory-style xlsx reference artifact: filter input rows by gold table membership (§7.1)."""
    import openpyxl

    sheet_keys = sorted(
        (key for key in gold if _SHEET_KEY_RE.match(key)),
        key=lambda key: int(re.search(r"\d+", key).group()),
    )
    if not sheet_keys or not isinstance(gold.get("headers"), list):
        raise SynthesisUnavailable(
            "xlsx gold lacks headers/sheetN_record_ids structure "
            "(long-tail shape needs curation)"
        )
    wanted = set()
    for key in sheet_keys:
        wanted |= {str(value) for value in gold[key]}
    input_headers, rows_by_id = _xlsx_input_rows(workspace_dir, wanted)
    missing = sorted(wanted - set(rows_by_id))
    if missing:
        raise SynthesisUnavailable(f"gold ids absent from input: {missing[:5]}")

    workbook = openpyxl.Workbook()
    workbook.remove(workbook.active)
    for key in sheet_keys:
        sheet = workbook.create_sheet(key.replace("_record_ids", ""))
        sheet.append(gold["headers"])
        for value in gold[key]:
            sheet.append(rows_by_id[str(value)])
    workbook.save(str(target))


def synthesize_reference_artifacts(
    record: Dict[str, Any], out_root: str, *, workspace_dir: Optional[str] = None
) -> Dict[str, Any]:
    """Dispatch reference synthesis by delivery type (§7.1); if not synthesizable -> needs_curation (never silent)."""
    task_id = require_nonempty(record.get("seed_id"), "native seed_id")
    gold = record.get("gold")
    if not isinstance(gold, dict) or not gold:
        raise BenchmarkError(f"native record needs frozen gold: {task_id}")
    root = Path(out_root) / task_id / "reference"
    artifacts = _required_artifacts(record)
    unavailable: List[str] = []
    for artifact in artifacts:
        artifact_type = str(artifact.get("type") or "").lower()
        relative = str(artifact.get("path") or "").strip().strip("/")
        if not relative:
            raise BenchmarkError(f"artifact path missing: {task_id}")
        target = _artifact_target(root, relative)
        try:
            if artifact_type == "json":
                _json_synth(gold, target)
            elif artifact_type == "xlsx":
                if not workspace_dir:
                    raise SynthesisUnavailable(
                        "xlsx synthesis requires an extracted workspace"
                    )
                _xlsx_synth(gold, Path(workspace_dir), target)
            else:
                raise SynthesisUnavailable(
                    f"type {artifact_type!r} not machine-synthesizable "
                    "(§7.2 curation path)"
                )
        except SynthesisUnavailable as error:
            unavailable.append({"path": relative, "reason": str(error)})
    if unavailable:
        return {
            "outcome": SYNTH_OUTCOME_NEEDS_CURATION,
            "root": None,
            "unavailable": unavailable,
        }
    return {"outcome": SYNTH_OUTCOME_SYNTHESIZED, "root": str(root)}


def synthesize_negative_artifacts(
    record: Dict[str, Any], out_root: str
) -> Dict[str, Any]:
    """Type-correct empty/garbage artifact (§7.3): exists and parses, but fails contract checks."""
    task_id = require_nonempty(record.get("seed_id"), "native seed_id")
    root = Path(out_root) / task_id / "negative"
    written: List[Dict[str, Any]] = []
    for artifact in _required_artifacts(record):
        artifact_type = str(artifact.get("type") or "").lower()
        relative = str(artifact.get("path") or "").strip().strip("/")
        target = _artifact_target(root, relative)
        if artifact_type == "json":
            target.write_text("{}\n", encoding="utf-8")
        elif artifact_type == "xlsx":
            import openpyxl

            openpyxl.Workbook().save(str(target))
        else:
            target.write_text("", encoding="utf-8")
        written.append(
            {
                "path": relative,
                "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            }
        )
    return {"root": str(root), "artifacts": written}


def extract_seed_tests(archive_path: str, member_prefix: str, dest: str) -> str:
    """Extract ``<member_prefix>/`` from the frozen seed archive (for the trusted-side oracle)."""
    archive = Path(archive_path)
    prefix = member_prefix.strip("/")
    with tarfile.open(archive, "r:gz") as tar:
        members = tar.getmembers()
        tops = {
            Path(member.name).parts[0]
            for member in members
            if member.name.strip()
        }
        if len(tops) != 1:
            raise BenchmarkError("seed archive must have a single top-level root")
        archive_prefix = f"{tops.pop()}/{prefix}/"
        selected = [
            member
            for member in members
            if member.name.startswith(archive_prefix) and member.isfile()
        ]
        if not selected:
            raise BenchmarkError(f"seed archive members absent: {prefix}")
        root = Path(dest)
        root.mkdir(parents=True, exist_ok=True)
        for member in selected:
            relative = Path(member.name).relative_to(archive_prefix)
            target = root / relative
            if not target.resolve().is_relative_to(root.resolve()):
                raise BenchmarkError(f"member escapes tests root: {member.name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            stream = tar.extractfile(member)
            if stream is None:
                raise BenchmarkError(f"member unreadable: {member.name}")
            target.write_bytes(stream.read())
    return str(root)


# Path rewrite table for the host-side oracle: longest first. eval_core freezes absolute paths
# (/tests/gold, /workspace, /logs) that map to temporary directories during trusted audit.
_ORACLE_REWRITES = (
    "/tests/gold/gold_answer.json",
    "/tests/gold/fixtures",
    "/tests/gold",
    "/logs/verifier",
    "/workspace/output",
    "/workspace",
)


def load_eval_oracle(
    grading_dir: str, gold_path: str, workspace_dir: str
) -> types.ModuleType:
    """Load eval_core as a host-side oracle module (path rewrite, eval semantics unchanged).

    For trusted-side audit iterations only; authoritative scoring must use the real runner end to end (M2 sandbox).
    """
    source_path = Path(grading_dir) / "eval_core.py"
    source = source_path.read_text(encoding="utf-8")
    rewrites = {
        "/tests/gold/gold_answer.json": str(gold_path),
        "/tests/gold/fixtures": str(Path(workspace_dir) / "_fixtures"),
        "/tests/gold": str(Path(gold_path).parent),
        "/logs/verifier": str(Path(workspace_dir) / "_logs"),
        "/workspace/output": str(Path(workspace_dir) / "output"),
        "/workspace": str(workspace_dir),
    }
# Single-pass regex replacement (longest-first alternation): sequential str.replace would repeatedly match already replaced
# host path suffixes (such as .../tests/gold), producing doubled paths.
    pattern = re.compile(
        "|".join(re.escape(frozen) for frozen in rewrites)
    )
    source = pattern.sub(lambda match: rewrites[match.group(0)], source)
    module_name = f"wb_eval_core_oracle_{abs(hash(source_path)) % 10**10}"
    module = types.ModuleType(module_name)
    module.__file__ = str(source_path)
# dataclass/@wrapper resolution looks up sys.modules via __module__, so it must be registered first.
    sys.modules[module_name] = module
    exec(compile(source, str(source_path), "exec"), module.__dict__)  # noqa: S102
    if not hasattr(module, "evaluate"):
        raise BenchmarkError("eval_core exposes no evaluate()")
    return module


def oracle_evaluate(module: types.ModuleType, candidate_path: str) -> Dict[str, Any]:
    """Call evaluate using grader semantics (file style: one argument; dir style: two arguments compatible)."""
    evaluate = module.evaluate
    try:
        result = evaluate(candidate_path)
    except TypeError:
        result = evaluate(candidate_path, None)
    if not isinstance(result, dict):
        raise BenchmarkError("oracle evaluate() returned a non-dict")
    return result


def _oracle_result_summary(result: Dict[str, Any]) -> Dict[str, Any]:
    checks = result.get("checks") or []
    failed = [check for check in checks if check.get("passed") is not True]
    return {
        "passed_count": result.get("passed_count"),
        "total_count": result.get("total_count", len(checks)),
        "score": result.get("pass_rate", result.get("score")),
        "failed_checks": [
            {"name": check.get("name"), "detail": str(check.get("detail"))[:200]}
            for check in failed[:20]
        ],
        "failed_count": len(failed),
    }


def _extract_workspace_pin(record: Dict[str, Any], dest: Path) -> str:
    """Unpack the task workspace pin into dest (trusted-side oracle input tree)."""
    workspace = record.get("workspace")
    if not isinstance(workspace, dict):
        raise BenchmarkError("native record needs a workspace pin")
    archive = Path(require_nonempty(workspace.get("path"), "workspace.path"))
    sha = require_sha(workspace.get("sha256"), "workspace.sha256")
    if not archive.is_file() or file_sha256(archive) != sha:
        raise BenchmarkError(f"workspace missing or SHA-256 drift: {archive}")
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar.getmembers():
            if not member.isfile():
                continue
            relative = Path(member.name)
            if relative.is_absolute() or ".." in relative.parts:
                raise BenchmarkError(
                    f"workspace member escapes layout: {member.name}"
                )
            target = dest / relative
            if not target.resolve().is_relative_to(dest.resolve()):
                raise BenchmarkError(
                    f"workspace member escapes root: {member.name}"
                )
            target.parent.mkdir(parents=True, exist_ok=True)
            stream = tar.extractfile(member)
            if stream is None:
                raise BenchmarkError(f"workspace member unreadable: {member.name}")
            target.write_bytes(stream.read())
    return str(dest)


def _oracle_candidate_path(
    root: Path, artifacts: List[Dict[str, Any]]
) -> str:
    """Select candidate paths using grader semantics: single-file deliverable -> file; multiple deliverables -> common parent directory.

    Consistent with each grader's ``DEFAULT_OUTPUT_PATH`` (file vs directory semantics).
    """
    if len(artifacts) == 1:
        relative = str(artifacts[0].get("path") or "").strip().strip("/")
        if Path(relative).suffix:
            return str(root / relative)
    parents = {
        str(Path(str(artifact.get("path") or "").strip().strip("/")).parent)
        for artifact in artifacts
    }
    common = parents.pop() if len(parents) == 1 else "output"
    return str(root / common)


def _prepare_oracle(
    record: Dict[str, Any], base: Path, archive_path: str
) -> Tuple[types.ModuleType, Path, List[Dict[str, Any]]]:
    task_id = require_nonempty(record.get("seed_id"), "native seed_id")
    workspace_dir = _extract_workspace_pin(record, base / "workspace")
    tests_dir = Path(
        extract_seed_tests(
            archive_path, f"tasks/{task_id}/tests", str(base / "tests")
        )
    )
    gold_path = str(tests_dir / "gold" / "gold_answer.json")
    oracle = load_eval_oracle(
        str(tests_dir / "grading"), gold_path, workspace_dir
    )
    return oracle, Path(workspace_dir), _required_artifacts(record)


def iterate_reference_synthesis(
    record: Dict[str, Any],
    out_root: str,
    *,
    archive_path: str,
    max_rounds: int = 3,
) -> Dict[str, Any]:
    """iterate-generate-verify (§7.2): synthesise -> oracle scoring -> <=3 rounds.

    Returns outcome: ``synthesized`` (all checks pass) / ``needs_curation`` (type
    not synthesizable or rounds exhausted but structure is curatable) / ``synthesis_failed``. No path is
    silent; the caller records excluded decisions together with the ingest report.
    """
    task_id = require_nonempty(record.get("seed_id"), "native seed_id")
    base = Path(out_root) / task_id
    oracle, _workspace_dir, artifacts = _prepare_oracle(record, base, archive_path)
    rounds: List[Dict[str, Any]] = []
    for round_index in range(1, max_rounds + 1):
        synthesis = synthesize_reference_artifacts(
            record, out_root, workspace_dir=str(base / "workspace")
        )
        if synthesis["outcome"] == SYNTH_OUTCOME_NEEDS_CURATION:
            return {
                "task_id": task_id,
                "outcome": SYNTH_OUTCOME_NEEDS_CURATION,
                "rounds": rounds,
                "unavailable": synthesis["unavailable"],
            }
        candidate = _oracle_candidate_path(Path(synthesis["root"]), artifacts)
        result = oracle_evaluate(oracle, candidate)
        summary = _oracle_result_summary(result)
        rounds.append({"round": round_index, **summary})
        if summary["failed_count"] == 0:
            return {
                "task_id": task_id,
                "outcome": SYNTH_OUTCOME_SYNTHESIZED,
                "rounds": rounds,
                "reference_root": synthesis["root"],
            }
    return {
        "task_id": task_id,
        "outcome": SYNTH_OUTCOME_NEEDS_CURATION,
        "rounds": rounds,
        "unavailable": [
            {"path": "*", "reason": "oracle checks still failing after "
             f"{max_rounds} rounds"}
        ],
    }


def iterate_negative_check(
    record: Dict[str, Any],
    out_root: str,
    *,
    archive_path: str,
    reference_score: float,
) -> Dict[str, Any]:
    """Negative oracle scoring + false-pass defense (§7.3): assert negative < reference."""
    task_id = require_nonempty(record.get("seed_id"), "native seed_id")
    base = Path(out_root) / task_id
    oracle, _workspace_dir, artifacts = _prepare_oracle(record, base, archive_path)
    negative = synthesize_negative_artifacts(record, out_root)
    candidate = _oracle_candidate_path(Path(negative["root"]), artifacts)
    result = oracle_evaluate(oracle, candidate)
    summary = _oracle_result_summary(result)
    score = summary["score"]
    if score is None:
        raise BenchmarkError(f"negative oracle produced no score: {task_id}")
    return {
        "task_id": task_id,
        "negative_root": negative["root"],
        "score": float(score),
        "reference_score": float(reference_score),
        "false_pass": not float(score) < float(reference_score),
        **summary,
    }


__all__ = [
    "INGEST_REPORT_SCHEMA",
    "SYNTH_OUTCOME_FAILED",
    "SYNTH_OUTCOME_NEEDS_CURATION",
    "SYNTH_OUTCOME_SYNTHESIZED",
    "SynthesisUnavailable",
    "build_native_ingest_report",
    "extract_seed_tests",
    "iterate_negative_check",
    "iterate_reference_synthesis",
    "load_eval_oracle",
    "materialize_public_inputs",
    "oracle_evaluate",
    "synthesize_negative_artifacts",
    "synthesize_reference_artifacts",
]
