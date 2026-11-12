"""WBBenchSeedAdapter: workbuddy-bench subset archive -> seed_tasks rows (Phase 0 / M0).

Schema follows ``docs/wbbench_schema_notes.md`` (2026-09-20 Office subset reverse engineering).
Input is a dataset root directory unpacked in the Harbor native layout (containing ``dataset.toml``) or
an unextracted ``wb-bench-*-v1.0.tar.gz`` archive; output rows align with the ``agent_task`` recipe
input fields and add metadata columns needed for expansion:

    subset / role / category / deliverable_type / verifier_form / wb_keywords

License red line: workbuddy-bench data is for local use only; this adapter reads only local archives,
does not copy, and does not redistribute any original task text.

seed_source semantics (to prevent "pretending to be based on workbuddy-bench"):
  - ``real-hf-subset``       real HF subset unpacked directory (dataset id is wb-bench-*)
  - ``synthetic-fixture``    sanitized synthetic fixture (dataset id contains fixture)
Explicit parameters may override this, but the manifest must carry the declaration.
"""

from __future__ import annotations

import hashlib
import json
import re
import tarfile
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

try:  # Python 3.11+
    import tomllib as _toml
except ModuleNotFoundError:  # pragma: no cover - legacy interpreter fallback
    _toml = None


class WBBenchSeedAdapterError(RuntimeError):
    """Invalid seed archive structure / missing required files."""


# ── Lightweight TOML parser (covers the subset used by dataset.toml / task.toml) ──────────────

_TOML_INT = re.compile(r"^[-+]?\d+$")
_TOML_FLOAT = re.compile(r"^[-+]?\d+\.\d+$")


def _parse_toml_value(raw: str) -> Any:
    raw = raw.strip()
    if raw.startswith('"""') or raw.startswith("'''"):
        marker = raw[:3]
        if raw.endswith(marker) and len(raw) >= 6:
            return raw[3:-3]
        raise WBBenchSeedAdapterError(f"Unterminated multi-line TOML string: {raw[:60]!r}")
    if raw.startswith('"') and raw.endswith('"') and len(raw) >= 2:
        return raw[1:-1]
    if raw.startswith("[") and raw.endswith("]"):
        inner = raw[1:-1].strip()
        if not inner:
            return []
        return [_parse_toml_value(part) for part in _split_toml_items(inner)]
    if raw.startswith("{") and raw.endswith("}"):
        inner = raw[1:-1].strip()
        table: Dict[str, Any] = {}
        if not inner:
            return table
        for part in _split_toml_items(inner):
            key, _, val = part.partition("=")
            table[key.strip().strip('"')] = _parse_toml_value(val)
        return table
    if raw in ("true", "false"):
        return raw == "true"
    if _TOML_INT.match(raw):
        return int(raw)
    if _TOML_FLOAT.match(raw):
        return float(raw)
    return raw


def _split_toml_items(inner: str) -> List[str]:
    parts: List[str] = []
    depth = 0
    quote = ""
    buf = ""
    for ch in inner:
        if quote:
            buf += ch
            if ch == quote:
                quote = ""
            continue
        if ch in ('"', "'"):
            quote = ch
            buf += ch
        elif ch in "[{":
            depth += 1
            buf += ch
        elif ch in "]}":
            depth -= 1
            buf += ch
        elif ch == "," and depth == 0:
            parts.append(buf)
            buf = ""
        else:
            buf += ch
    if buf.strip():
        parts.append(buf)
    return parts


def _load_toml(path: Path) -> Dict[str, Any]:
    if _toml is not None:
        with path.open("rb") as fh:
            return _toml.load(fh)
    data: Dict[str, Any] = {}
    section: Dict[str, Any] = data
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("[") and stripped.endswith("]") and "=" not in stripped:
            section = data
            for part in stripped[1:-1].split("."):
                section = section.setdefault(part.strip().strip('"'), {})
            continue
        key, sep, val = stripped.partition("=")
        if not sep:
            continue
        section[key.strip().strip('"')] = _parse_toml_value(val)
    return data


# ── Minimal judge.yaml parser (extracts only artifacts / score_merge weights; no pyyaml dependency) ────

_JUDGE_ARTIFACT_LINE = re.compile(r"^\s*-\s*id:\s*(\S+)")
_JUDGE_FIELD = re.compile(r"^\s*(type|required|path):\s*(\S+)")
_JUDGE_WEIGHT = re.compile(r"^\s*(rule_weight|llm_weight):\s*([0-9.]+)")
_JUDGE_SECTION = re.compile(r"^([a-z_]+):\s*$")


def _parse_judge_yaml(text: str) -> Dict[str, Any]:
    artifacts: List[Dict[str, Any]] = []
    current: Optional[Dict[str, Any]] = None
    weights: Dict[str, float] = {}
    section = "artifacts"
    for line in text.splitlines():
        sec = _JUDGE_SECTION.match(line)
        if sec:
            section = sec.group(1)
            current = None
            continue
        art = _JUDGE_ARTIFACT_LINE.match(line)
        if art and section == "artifacts":
            current = {"id": art.group(1)}
            artifacts.append(current)
            continue
        fld = _JUDGE_FIELD.match(line)
        if fld and current is not None and section == "artifacts":
            current[fld.group(1)] = fld.group(2)
            continue
        wgt = _JUDGE_WEIGHT.match(line)
        if wgt:
            weights[wgt.group(1)] = float(wgt.group(2))
    return {"artifacts": artifacts, "score_merge": weights}


# ── Full block-level judge.yaml parser (native ingestion M1; no pyyaml dependency, covers only this repo's YAML subset) ─

def _yaml_scalar(raw: str) -> Any:
    raw = raw.strip()
    if raw in ("", "~", "null"):
        return None
    if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in ("'", '"'):
        return raw[1:-1]
    if raw in ("true", "True"):
        return True
    if raw in ("false", "False"):
        return False
    if _TOML_INT.match(raw):
        return int(raw)
    if _TOML_FLOAT.match(raw):
        return float(raw)
    if raw.startswith("[") and raw.endswith("]"):
        inner = raw[1:-1].strip()
        if not inner:
            return []
        return [_yaml_scalar(part) for part in _split_toml_items(inner)]
    return raw


def _yaml_indent(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def _yaml_skip(lines: List[str], index: int) -> int:
    while index < len(lines):
        stripped = lines[index].strip()
        if stripped and not stripped.startswith("#"):
            return index
        index += 1
    return index


def _yaml_fold_scalar(
    lines: List[str], index: int, key_indent: int, first: str
) -> tuple:
    """Pure scalar multiline folding: a continuation line indented deeper than the key line and without ": " is merged into the same scalar."""
    parts = [first] if first else []
    while True:
        probe = _yaml_skip(lines, index)
        if probe >= len(lines):
            break
        line = lines[probe]
        if _yaml_indent(line) <= key_indent:
            break
        text = line.strip()
        if text.startswith("-") or ": " in text or text.endswith(":"):
            break
        parts.append(text)
        index = probe + 1
    return " ".join(parts), index


def _parse_yaml_map(lines: List[str], index: int, indent: int):
    result: Dict[str, Any] = {}
    while True:
        index = _yaml_skip(lines, index)
        if index >= len(lines):
            break
        line = lines[index]
        current = _yaml_indent(line)
        if current < indent:
            break
        stripped = line.strip()
        if stripped.startswith("-"):
            break
        if current > indent:
            raise WBBenchSeedAdapterError(f"YAML indentation error: {line!r}")
        key, sep, rest = stripped.partition(":")
        if not sep:
            raise WBBenchSeedAdapterError(f"YAML line is missing a colon: {line!r}")
        key = key.strip().strip("'\"")
        rest = rest.strip()
        index += 1
        if rest:
            value, index = _yaml_fold_scalar(lines, index, indent, rest)
            result[key] = _yaml_scalar(value)
            continue
        probe = _yaml_skip(lines, index)
        if probe < len(lines):
            probe_indent = _yaml_indent(lines[probe])
            if probe_indent > indent:
                result[key], index = _parse_yaml_block(lines, probe, indent + 1)
                continue
            if probe_indent == indent and lines[probe].strip().startswith("-"):
                result[key], index = _parse_yaml_list(lines, probe, indent)
                continue
        result[key] = None
    return result, index


def _parse_yaml_list(lines: List[str], index: int, indent: int):
    result: List[Any] = []
    while True:
        index = _yaml_skip(lines, index)
        if index >= len(lines):
            break
        line = lines[index]
        current = _yaml_indent(line)
        if current < indent:
            break
        stripped = line.strip()
        if not stripped.startswith("-"):
            break
        if current > indent:
            raise WBBenchSeedAdapterError(f"YAML list indentation error: {line!r}")
        item = stripped[1:].strip()
        index += 1
        if not item:
            value, index = _parse_yaml_block(lines, index, indent + 1)
            result.append(value)
            continue
        key, sep, rest = item.partition(":")
        if sep and " " not in key.strip() and not item.startswith(("'", '"')):
            name = key.strip().strip("'\"")
            rest = rest.strip()
            if rest:
                value, index = _yaml_fold_scalar(lines, index, indent + 2, rest)
                entry: Dict[str, Any] = {name: _yaml_scalar(value)}
                tail, index = _parse_yaml_map(lines, index, indent + 2)
                entry.update(tail)
            else:
                entry_value, index = _parse_yaml_block(lines, index, indent + 1)
                entry = {name: entry_value}
            result.append(entry)
        else:
            value, index = _yaml_fold_scalar(lines, index, indent, item)
            result.append(_yaml_scalar(value))
    return result, index


def _parse_yaml_block(lines: List[str], index: int, min_indent: int):
    index = _yaml_skip(lines, index)
    if index >= len(lines):
        return None, index
    indent = _yaml_indent(lines[index])
    if indent < min_indent:
        return None, index
    if lines[index].strip().startswith("-"):
        return _parse_yaml_list(lines, index, indent)
    return _parse_yaml_map(lines, index, indent)


def _parse_yaml_subset(text: str) -> Dict[str, Any]:
    """Block-level YAML subset parser (mapping/list/scalar; covers generator output from fixture judge.yaml)."""
    lines = text.splitlines()
    value, _ = _parse_yaml_map(lines, 0, 0)
    if not isinstance(value, dict):
        raise WBBenchSeedAdapterError("judge.yaml top level must be a mapping")
    return value


def _parse_judge_document(text: str) -> Dict[str, Any]:
    """Full judge.yaml parsing: prefer pyyaml (real corpora contain >- folded blocks and multiline scalars);
    when pyyaml is unavailable, fall back to ``_parse_yaml_subset`` (covers only the simple subset)."""
    try:
        import yaml
    except ModuleNotFoundError:  # pragma: no cover - fallback path
        return _parse_yaml_subset(text)
    value = yaml.safe_load(text)
    if not isinstance(value, dict):
        raise WBBenchSeedAdapterError("judge.yaml top level must be a mapping")
    return value


def _dir_content_sha256(root: Path) -> str:
    """Directory content pin: sort the (relative path, file sha256) list and hash the whole list."""
    entries = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        entries.append(
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    payload = json.dumps(
        entries, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load_gold_json(path: Path) -> Dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except (OSError, ValueError) as error:
        raise WBBenchSeedAdapterError(
            f"gold_answer.json is invalid: {path}: {error}"
        ) from error
    if not isinstance(value, dict) or not value:
        raise WBBenchSeedAdapterError(f"gold_answer.json must be a non-empty object: {path}")
    return value


def _validate_native_judge(judge: Dict[str, Any], task_dir: Path) -> Dict[str, Any]:
    artifacts = judge.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise WBBenchSeedAdapterError(f"judge.yaml artifacts are missing: {task_dir}")
    seen_ids: set = set()
    required_count = 0
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            raise WBBenchSeedAdapterError(f"judge.yaml artifacts item is invalid: {task_dir}")
        for field_name in ("id", "path", "type"):
            if not str(artifact.get(field_name) or "").strip():
                raise WBBenchSeedAdapterError(
                    f"judge.yaml artifact is missing {field_name}: {task_dir}"
                )
        relative = str(artifact["path"])
        parts = Path(relative).parts
        if (
            Path(relative).is_absolute()
            or ".." in parts
            or not all(part not in ("", ".", "..") for part in parts)
        ):
            raise WBBenchSeedAdapterError(
                f"judge.yaml artifact path must be a clean relative path: {relative!r}"
            )
        if artifact["id"] in seen_ids:
            raise WBBenchSeedAdapterError(f"judge.yaml artifact id is duplicated: {task_dir}")
        seen_ids.add(artifact["id"])
        if artifact.get("required") is True:
            required_count += 1
    if required_count < 1:
        raise WBBenchSeedAdapterError(f"judge.yaml has no required artifact: {task_dir}")
    llm_judge = judge.get("llm_judge")
    if llm_judge is None:
        llm_judge = {}
    if not isinstance(llm_judge, dict):
        raise WBBenchSeedAdapterError(f"judge.yaml llm_judge is invalid: {task_dir}")
    rubrics = llm_judge.get("rubrics")
    if rubrics is None:
        rubrics = []
    if not isinstance(rubrics, list) or not all(isinstance(r, dict) for r in rubrics):
        raise WBBenchSeedAdapterError(f"judge.yaml rubrics are invalid: {task_dir}")
    score_merge = judge.get("score_merge")
    if not isinstance(score_merge, dict):
        raise WBBenchSeedAdapterError(f"judge.yaml score_merge is missing: {task_dir}")
    for weight in ("rule_weight", "llm_weight"):
        value = score_merge.get(weight)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise WBBenchSeedAdapterError(
                f"judge.yaml score_merge.{weight} is invalid: {task_dir}"
            )
    return {
        "artifacts": artifacts,
        "llm_judge": {**llm_judge, "rubrics": rubrics},
        "score_merge": score_merge,
    }


# ── Metadata heuristics ─────────────────────────────────────────────────────────────
_ROLE_KEYWORDS = (
    (("finance-quant", "aggregation-metric-reasoning"), "algo"),
    (("automation-workdir", "mock-app", "state-diff", "skill-mcp-harness"), "developer"),
    (("governance", "incident", "sop", "document-control", "control-state"), "ops"),
)

SUBSET_RE = re.compile(r"wb-bench-([a-z]+)")


def _role_from_keywords(keywords: List[str]) -> str:
    for keys, role in _ROLE_KEYWORDS:
        if any(key in kw for kw in keywords for key in keys):
            return role
    return "unspecified"


def _verifier_form_from_judge(judge: Optional[Dict[str, Any]]) -> str:
    if not judge:
        return "rule"
    weights = judge.get("score_merge") or {}
    llm_w = float(weights.get("llm_weight") or 0.0)
    rule_w = float(weights.get("rule_weight") or 0.0)
    if llm_w <= 0.0:
        return "rule"
    if rule_w <= 0.0:
        return "llm_rubric"
    return "composite"


# ── Main adapter ─────────────────────────────────────────────────────────────────

@dataclass
class WBBenchSeedManifest:
    """Seed ingestion manifest (seed_source declaration is mandatory; see module docstring)."""

    seed_source: str
    dataset_id: str
    subset: str
    schema: str
    task_count: int
    archive_sha256: Optional[str] = None
    categories: Dict[str, int] = field(default_factory=dict)
    difficulties: Dict[str, int] = field(default_factory=dict)
    verifier_forms: Dict[str, int] = field(default_factory=dict)
    roles: Dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


class WBBenchSeedAdapter:
    """workbuddy-bench subset -> seed_tasks rows + taxonomy grid + manifest.

    Usage:
        adapter = WBBenchSeedAdapter(dataset_dir)          # already-unpacked root directory
        adapter = WBBenchSeedAdapter.from_archive(path)    # automatic tar.gz unpacking
        rows = adapter.load_seed_tasks()
    """

    def __init__(
        self,
        dataset_dir: "str | Path",
        seed_source: Optional[str] = None,
        archive_sha256: Optional[str] = None,
    ):
        self.dataset_dir = Path(dataset_dir)
        self.archive_sha256 = archive_sha256
        dataset_toml = self.dataset_dir / "dataset.toml"
        if not dataset_toml.is_file():
            raise WBBenchSeedAdapterError(
                f"dataset.toml does not exist: {dataset_toml} (an extracted dataset root is required)")
        meta = _load_toml(dataset_toml)
        dataset = meta.get("dataset") or {}
        self.dataset_id = str(dataset.get("id") or self.dataset_dir.name)
        self.schema_name = str(dataset.get("schema") or "")
        self.declared_task_count = int(dataset.get("task_count") or 0)
        m = SUBSET_RE.search(self.dataset_id)
        self.subset = m.group(1) if m else "unknown"
        if seed_source is not None:
            self.seed_source = str(seed_source)
        elif "fixture" in self.dataset_id:
            self.seed_source = "synthetic-fixture"
        else:
            self.seed_source = "real-hf-subset"

    @classmethod
    def from_archive(
        cls,
        archive_path: "str | Path",
        extract_dir: Optional["str | Path"] = None,
        seed_source: Optional[str] = None,
    ) -> "WBBenchSeedAdapter":
        """Unpack tar.gz (default temporary directory; validate a single top-level root directory) and return the adapter."""
        archive_path = Path(archive_path)
        if not archive_path.is_file():
            raise WBBenchSeedAdapterError(f"Archive does not exist: {archive_path}")
        data = archive_path.read_bytes()
        archive_sha256 = hashlib.sha256(data).hexdigest()
        target = Path(extract_dir) if extract_dir else Path(
            tempfile.mkdtemp(prefix="wbbench-seeds-"))
        target.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(target)
        roots = [p for p in target.iterdir() if (p / "dataset.toml").is_file()]
        if len(roots) != 1:
            raise WBBenchSeedAdapterError(
                f"Archive top level must contain exactly one root with dataset.toml; found {len(roots)}")
        return cls(roots[0], seed_source=seed_source, archive_sha256=archive_sha256)

# ── Seed rows ───────────────────────────────────────────────────────────

    def task_dirs(self) -> List[Path]:
        root = self.dataset_dir / "tasks"
        if not root.is_dir():
            raise WBBenchSeedAdapterError(f"tasks/ directory does not exist: {root}")
        return sorted(p for p in root.iterdir() if p.is_dir())

    def load_seed_tasks(self) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for task_dir in self.task_dirs():
            row = self._load_task(task_dir)
            if row is not None:
                rows.append(row)
        if self.declared_task_count and len(rows) != self.declared_task_count:
            raise WBBenchSeedAdapterError(
                f"task_count mismatch: dataset.toml declares {self.declared_task_count}, "
                f"but parsed {len(rows)}")
        return rows

    def _load_task(self, task_dir: Path) -> Optional[Dict[str, Any]]:
        task_toml_path = task_dir / "task.toml"
        instruction_path = task_dir / "instruction.md"
        if not task_toml_path.is_file() or not instruction_path.is_file():
            raise WBBenchSeedAdapterError(
                f"Task directory is missing task.toml or instruction.md: {task_dir}")
        task_meta = _load_toml(task_toml_path)
        task_sec = task_meta.get("task") or {}
        meta_sec = task_meta.get("metadata") or {}
        env_sec = task_meta.get("environment") or {}
        instruction = instruction_path.read_text(encoding="utf-8").strip()
        if not instruction:
            raise WBBenchSeedAdapterError(f"instruction.md is empty: {task_dir}")

        judge_path = task_dir / "tests" / "judge.yaml"
        judge = _parse_judge_yaml(judge_path.read_text(encoding="utf-8")) \
            if judge_path.is_file() else None
        verifier_form = _verifier_form_from_judge(judge)

        artifacts = (judge or {}).get("artifacts") or []
        deliverable_types = sorted({str(a["type"]) for a in artifacts if a.get("type")})
        artifact_paths = [str(a.get("path") or a.get("id")) for a in artifacts]
        weights = (judge or {}).get("score_merge") or {}
        rule_w = float(weights.get("rule_weight") or 0.0)
        llm_w = float(weights.get("llm_weight") or 0.0)

        keywords = [str(k) for k in (task_sec.get("keywords") or [])]
        raw_name = str(task_sec.get("name") or task_dir.name)
        seed_id = raw_name.split("/", 1)[-1]

        criteria = (
            f"Produce required artifact(s) {artifact_paths} that pass the "
            f"frozen gold grading checks "
            f"(score_merge=weighted_sum rule={rule_w:g}, llm={llm_w:g})."
        )
        rubric = (
            f"verifier_form={verifier_form}; artifacts={artifact_paths}; "
            f"gold=tests/gold/gold_answer.json"
        )
        return {
            "seed_id": seed_id,
            "task_text": instruction,
            "tools_available": [str(t) for t in (env_sec.get("mcp_servers") or [])],
            "success_criteria": criteria,
            "ground_truth_action_sequence": [],
            "scoring_rubric": rubric,
            "difficulty": str(meta_sec.get("difficulty") or "unspecified"),
            "subset": self.subset,
            "role": _role_from_keywords(keywords),
            "category": str(meta_sec.get("category") or "unspecified"),
            "deliverable_type": deliverable_types,
            "verifier_form": verifier_form,
            "wb_keywords": keywords,
        }

# ── Full native-ingest record (M1: full judge + gold + grading/workspace pins) ──

    def load_native_task_records(self) -> List[Dict[str, Any]]:
        """Per-task full record: original instruction + full judge.yaml + gold_answer.json.

        Unlike the regex summary in ``load_seed_tasks``, this method preserves the complete artifacts/rubrics/
        score_merge structure, the original gold property assertions, and the tests/grading directory content sha256,
        for freezing into ``wb_composite`` verifier_spec (design §3/§5). The workspace is fixed to
        ``environment/workspace.tar.gz`` (where 49/50 tasks' data lives; the task text contains no inline data).
        """
        rows: List[Dict[str, Any]] = []
        for task_dir in self.task_dirs():
            rows.append(self._load_native_task(task_dir))
        if self.declared_task_count and len(rows) != self.declared_task_count:
            raise WBBenchSeedAdapterError(
                f"task_count mismatch: dataset.toml declares {self.declared_task_count}, "
                f"but parsed {len(rows)}")
        return rows

    def _load_native_task(self, task_dir: Path) -> Dict[str, Any]:
        task_toml_path = task_dir / "task.toml"
        instruction_path = task_dir / "instruction.md"
        if not task_toml_path.is_file() or not instruction_path.is_file():
            raise WBBenchSeedAdapterError(
                f"Task directory is missing task.toml or instruction.md: {task_dir}")
        task_meta = _load_toml(task_toml_path)
        task_sec = task_meta.get("task") or {}
        meta_sec = task_meta.get("metadata") or {}
        env_sec = task_meta.get("environment") or {}
        verifier_sec = task_meta.get("verifier") or {}
        instruction = instruction_path.read_text(encoding="utf-8").strip()
        if not instruction:
            raise WBBenchSeedAdapterError(f"instruction.md is empty: {task_dir}")

        judge_path = task_dir / "tests" / "judge.yaml"
        if not judge_path.is_file():
            raise WBBenchSeedAdapterError(f"judge.yaml does not exist: {task_dir}")
        judge = _validate_native_judge(
            _parse_judge_document(judge_path.read_text(encoding="utf-8")), task_dir
        )

        artifacts = judge["artifacts"]
        deliverable_types = sorted({str(a["type"]) for a in artifacts})
        artifact_paths = [str(a["path"]) for a in artifacts]
        weights = judge["score_merge"]
        rule_w = float(weights.get("rule_weight") or 0.0)
        llm_w = float(weights.get("llm_weight") or 0.0)

        gold_path = task_dir / "tests" / "gold" / "gold_answer.json"
        if not gold_path.is_file():
            raise WBBenchSeedAdapterError(f"gold_answer.json does not exist: {task_dir}")
        grading_dir = task_dir / "tests" / "grading"
        if not grading_dir.is_dir():
            raise WBBenchSeedAdapterError(f"tests/grading directory does not exist: {task_dir}")
        workspace_path = task_dir / "environment" / "workspace.tar.gz"
        if not workspace_path.is_file():
            raise WBBenchSeedAdapterError(
                f"environment/workspace.tar.gz does not exist: {task_dir}")

        timeout = verifier_sec.get("timeout_sec")
        if timeout is not None and (
            isinstance(timeout, bool) or not isinstance(timeout, (int, float))
        ):
            raise WBBenchSeedAdapterError(f"[verifier] timeout_sec is invalid: {task_dir}")

        keywords = [str(k) for k in (task_sec.get("keywords") or [])]
        raw_name = str(task_sec.get("name") or task_dir.name)
        seed_id = raw_name.split("/", 1)[-1]
        criteria = (
            f"Produce required artifact(s) {artifact_paths} that pass the "
            f"frozen gold grading checks "
            f"(score_merge=weighted_sum rule={rule_w:g}, llm={llm_w:g})."
        )
        return {
            "seed_id": seed_id,
            "task_text": instruction,
            "tools_available": [str(t) for t in (env_sec.get("mcp_servers") or [])],
            "success_criteria": criteria,
            "difficulty": str(meta_sec.get("difficulty") or "unspecified"),
            "subset": self.subset,
            "category": str(meta_sec.get("category") or "unspecified"),
            "deliverable_types": deliverable_types,
            "judge": judge,
            "gold": _load_gold_json(gold_path),
            "grading_dir_sha256": _dir_content_sha256(grading_dir),
            "verifier_timeout_sec": timeout,
            "workspace": {
                "path": str(workspace_path.resolve()),
                "sha256": hashlib.sha256(workspace_path.read_bytes()).hexdigest(),
            },
            "wb_keywords": keywords,
        }

# ── Taxonomy and manifest ─────────────────────────────────────────────

    def taxonomy_grid(self) -> Dict[str, Dict[str, int]]:
        """category x difficulty coverage grid (S2's input for finding gaps)."""
        grid: Dict[str, Dict[str, int]] = {}
        for row in self.load_seed_tasks():
            cell = grid.setdefault(row["category"], {})
            cell[row["difficulty"]] = cell.get(row["difficulty"], 0) + 1
        return grid

    def build_manifest(self) -> Dict[str, Any]:
        rows = self.load_seed_tasks()
        manifest = WBBenchSeedManifest(
            seed_source=self.seed_source,
            dataset_id=self.dataset_id,
            subset=self.subset,
            schema=self.schema_name,
            task_count=len(rows),
            archive_sha256=self.archive_sha256,
        )
        counter_by_key = {
            "category": manifest.categories,
            "difficulty": manifest.difficulties,
            "verifier_form": manifest.verifier_forms,
            "role": manifest.roles,
        }
        for row in rows:
            for key, counter in counter_by_key.items():
                counter[row[key]] = counter.get(row[key], 0) + 1
        return manifest.as_dict()


__all__ = ["WBBenchSeedAdapter", "WBBenchSeedAdapterError", "WBBenchSeedManifest"]
