"""HfSingleTurnSeedAdapter: HF single-turn snapshot directory -> seed_tasks rows (NeMo/HF single-turn evaluation ingestion M2).

Input is a snapshot directory in ``hf-single-turn-snapshot-v1`` form (row-count pin in ``dataset.json`` +
``rows/<benchmark>.jsonl``), producing seed rows for 11 upstream benchmarks (gsm8k / math500 / mgsm / drop /
mmlu / mmlu_pro / gpqa / triviaqa / humaneval / simpleqa / healthbench), mapped into 4 families
(reasoning_qa / knowledge_qa / code_gen / health_qa).

D9 admission: only rows with offline-recomputable answers are admitted; rows missing an answer/alias/option/test/rubric get
``admission = "excluded"`` and ``exclusion_reason = "answer_not_recomputable:<detail>"``.

R2 randomized-decision materialization: gpqa option shuffle (upstream ``PythonRandom(42)``; replay its permutation
sequence per row with the same seed; the materialized option order and answer letter go only to the trusted side), drop few-shot 3-example
combinations (upstream ``PythonRandom(42+index)``; materialize the selected row numbers and freeze them into task_text).

D5 boundary: gold (answer / aliases / canonical_solution / test_code / rubric /
reference_response) never enters the seed row; it is read on demand from the trusted side via ``gold_fields(seed_id)``;
``export_prompt_archive`` packages only prompt text.
"""

from __future__ import annotations

import hashlib
import io
import json
import random
import re
import tarfile
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional


class HfSingleTurnSeedAdapterError(RuntimeError):
    """Invalid snapshot structure / missing required files / row-count pin mismatch."""


SNAPSHOT_SCHEMA = "hf-single-turn-snapshot-v1"
EXCLUDED_REASON_ANSWER = "answer_not_recomputable"

BENCHMARK_FAMILIES: Dict[str, str] = {
    "gsm8k": "reasoning_qa",
    "math500": "reasoning_qa",
    "mgsm": "reasoning_qa",
    "drop": "reasoning_qa",
    "mmlu": "knowledge_qa",
    "mmlu_pro": "knowledge_qa",
    "gpqa": "knowledge_qa",
    "triviaqa": "knowledge_qa",
    "simpleqa": "knowledge_qa",
    "humaneval": "code_gen",
    "healthbench": "health_qa",
}

UNIT_TEST_BENCHMARKS = frozenset(
    {"gsm8k", "math500", "mgsm", "drop", "mmlu", "mmlu_pro", "gpqa",
     "triviaqa", "humaneval"}
)
LLM_RUBRIC_BENCHMARKS = frozenset({"simpleqa", "healthbench"})

# The 11 language slices of upstream MGSM.
_MGSM_LANGUAGES = (
    "bn", "de", "en", "es", "fr", "ja", "ru", "sw", "te", "th", "zh",
)
_NUMERIC_RE = re.compile(r"-?\d")
_LETTERS = "ABCDEFGHIJ"

_SUCCESS_CRITERIA = (
    "Deliver the final answer text as the candidate deliverable response "
    "file; it must satisfy the frozen offline assertion re-expressed from the "
    "upstream deterministic scoring semantics."
)


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise HfSingleTurnSeedAdapterError(
                f"{path.name} line {line_number} is not valid JSON: {error}"
            ) from error
        if not isinstance(row, dict):
            raise HfSingleTurnSeedAdapterError(
                f"{path.name} line {line_number} line must be a JSON object")
        rows.append(row)
    return rows


def _gpqa_order(count: int) -> List[int]:
    """Per-row replay of the upstream ``PythonRandom(42)`` permutation sequence (R2)."""
    order = list(range(count))
    random.Random(42).shuffle(order)
    return order


class HfSingleTurnSeedAdapter:
    """HF single-turn snapshot directory -> seed rows + trusted-side gold accessor.

    Usage:
        adapter = HfSingleTurnSeedAdapter(snapshot_dir)     # already-unpacked snapshot root directory
        adapter = HfSingleTurnSeedAdapter.from_archive(path)  # automatic tar.gz unpacking
        rows = adapter.load_seed_tasks()
        gold = adapter.gold_fields("gsm8k:row-0")
    """

    def __init__(
        self,
        snapshot_dir: "str | Path",
        seed_source: Optional[str] = None,
        archive_sha256: Optional[str] = None,
    ):
        self.snapshot_dir = Path(snapshot_dir)
        self.archive_sha256 = archive_sha256
        dataset_json = self.snapshot_dir / "dataset.json"
        if not dataset_json.is_file():
            raise HfSingleTurnSeedAdapterError(
                f"dataset.json does not exist: {dataset_json} (an extracted snapshot root is required)")
        meta = json.loads(dataset_json.read_text(encoding="utf-8"))
        self.dataset_id = str(meta.get("dataset_id") or self.snapshot_dir.name)
        self.schema_name = str(meta.get("schema") or "")
        if self.schema_name != SNAPSHOT_SCHEMA:
            raise HfSingleTurnSeedAdapterError(
                f"schema must be {SNAPSHOT_SCHEMA}; actual value: {self.schema_name!r}")
        self.row_counts = {
            str(name): int(count)
            for name, count in (meta.get("row_counts") or {}).items()
        }
        if not self.row_counts:
            raise HfSingleTurnSeedAdapterError("dataset.json is missing the row_counts pin")
        missing = sorted(set(BENCHMARK_FAMILIES) - set(self.row_counts))
        if missing:
            raise HfSingleTurnSeedAdapterError(
                f"row_counts is missing the benchmark row-count pin: {missing}")
        if seed_source is not None:
            self.seed_source = str(seed_source)
        elif "fixture" in self.dataset_id:
            self.seed_source = "synthetic-fixture"
        else:
            self.seed_source = "upstream-checkout"

    @classmethod
    def from_archive(
        cls,
        archive_path: "str | Path",
        extract_dir: Optional["str | Path"] = None,
        seed_source: Optional[str] = None,
    ) -> "HfSingleTurnSeedAdapter":
        """Unpack tar.gz (default temporary directory; validate a single top-level root directory) and return the adapter."""
        archive_path = Path(archive_path)
        if not archive_path.is_file():
            raise HfSingleTurnSeedAdapterError(f"Archive does not exist: {archive_path}")
        data = archive_path.read_bytes()
        archive_sha256 = hashlib.sha256(data).hexdigest()
        target = Path(extract_dir) if extract_dir else Path(
            tempfile.mkdtemp(prefix="hf-single-turn-seeds-"))
        target.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(target)
        roots = [
            path for path in target.iterdir()
            if (path / "dataset.json").is_file()
        ]
        if len(roots) != 1:
            raise HfSingleTurnSeedAdapterError(
                f"Archive top level must contain exactly one root with dataset.json; found {len(roots)}")
        return cls(roots[0], seed_source=seed_source, archive_sha256=archive_sha256)

# ── Snapshot pins and digest ──────────────────────────────────────────────────

    def rows_file(self, benchmark: str) -> Path:
        if benchmark not in BENCHMARK_FAMILIES:
            raise HfSingleTurnSeedAdapterError(f"Unknown benchmark: {benchmark}")
        return self.snapshot_dir / "rows" / f"{benchmark}.jsonl"

    def _load_rows(self, benchmark: str) -> List[Dict[str, Any]]:
        path = self.rows_file(benchmark)
        if not path.is_file():
            raise HfSingleTurnSeedAdapterError(f"Snapshot is missing row file: {path}")
        rows = _read_jsonl(path)
        pinned = self.row_counts.get(benchmark, len(rows))
        if len(rows) != pinned:
            raise HfSingleTurnSeedAdapterError(
                f"{benchmark} row count does not match the dataset.json pin: "
                f"pin {pinned}; actual value: {len(rows)}")
        return rows

    def _load_fewshot_pool(self) -> List[Dict[str, Any]]:
        path = self.snapshot_dir / "rows" / "drop_fewshot.jsonl"
        if not path.is_file():
            raise HfSingleTurnSeedAdapterError(f"drop few-shot pool is missing file: {path}")
        pool = _read_jsonl(path)
        pinned = self.row_counts.get("drop_fewshot", len(pool))
        if len(pool) != pinned:
            raise HfSingleTurnSeedAdapterError(
                f"drop_fewshot row count does not match the pin: pin {pinned}; actual value: {len(pool)}")
        return pool

    def snapshot_digest(self) -> str:
        """Snapshot content SHA-256 (sorted relative paths + byte streams), material for the D2 release suffix."""
        digest = hashlib.sha256()
        for path in sorted(self.snapshot_dir.rglob("*")):
            if not path.is_file() or path.is_symlink():
                continue
            relative = path.relative_to(self.snapshot_dir).as_posix()
            digest.update(relative.encode("utf-8"))
            digest.update(b"\0")
            digest.update(path.read_bytes())
        return digest.hexdigest()

# ── Seed rows ──────────────────────────────────────────────────────────

    def load_seed_tasks(self) -> List[Dict[str, Any]]:
        """Expand 11 benchmarks into seed rows in BENCHMARK_FAMILIES order (without gold)."""
        rows: List[Dict[str, Any]] = []
        for benchmark in BENCHMARK_FAMILIES:
            for index, raw in enumerate(self._load_rows(benchmark)):
                rows.append(self._load_row(benchmark, index, raw))
        return rows

    def _load_row(
        self, benchmark: str, index: int, raw: Dict[str, Any]
    ) -> Dict[str, Any]:
        family = BENCHMARK_FAMILIES[benchmark]
        if benchmark == "mgsm":
            language = str(raw.get("language") or "")
            if language not in _MGSM_LANGUAGES:
                raise HfSingleTurnSeedAdapterError(
                    f"mgsm line {index} has an invalid language: {language!r}")
            selection = language
        else:
            selection = "row"
        seed_id = f"{benchmark}:{selection}:row-{index}"

        materialization: Dict[str, Any] = {}
        exclusion: Optional[str] = self._exclusion_reason(benchmark, raw)
        if exclusion is None and benchmark == "gpqa":
            materialization = {
                "mode": "option_shuffle",
                "seed": 42,
            }
        elif exclusion is None and benchmark == "drop":
            materialization = {
                "mode": "few_shot_selection",
                "seed": 42 + index,
                "selected_train_rows": self._drop_selected_rows(index),
            }

        task_text = (
            self._compose_task_text(benchmark, index, raw)
            if exclusion is None else str(raw.get("question") or raw.get("problem") or "")
        )
        return {
            "seed_id": seed_id,
            "benchmark": benchmark,
            "family": family,
            "task_text": task_text,
            "tools_available": [],
            "success_criteria": _SUCCESS_CRITERIA,
            "difficulty": str(raw.get("difficulty") or "unspecified"),
            "admission": "excluded" if exclusion else "expressible",
            "exclusion_reason": exclusion,
            "materialization": materialization,
        }

# ── D9 admission pre-screen (missing/non-recomputable answer -> excluded) ─────────────────────

    def _exclusion_reason(
        self, benchmark: str, raw: Dict[str, Any]
    ) -> Optional[str]:
        def missing(detail: str) -> str:
            return f"{EXCLUDED_REASON_ANSWER}:{detail}"

        if benchmark in ("gsm8k", "math500", "mgsm"):
            answer = str(raw.get("answer") or "").strip()
            if not answer:
                return missing("missing answer")
            if benchmark == "gsm8k" and not _NUMERIC_RE.search(answer):
                return missing("non-numeric answer")
            return None
        if benchmark == "drop":
            answer = raw.get("answer")
            spans = raw.get("answer_spans")
            if (answer is None or not str(answer).strip()) and not spans:
                return missing("missing answer and spans")
            return None
        if benchmark in ("mmlu", "mmlu_pro", "gpqa"):
            options = raw.get("options") if benchmark == "mmlu_pro" else (
                raw.get("choices") if benchmark == "mmlu" else None)
            if benchmark == "gpqa":
                correct = str(raw.get("correct_answer") or "").strip()
                distractors = [
                    str(raw.get(f"distractor_{i}") or "").strip()
                    for i in (1, 2, 3)
                ]
                if not correct or any(not d for d in distractors):
                    return missing("missing gpqa options")
                return None
            answer_index = raw.get("answer")
            if not options or not isinstance(answer_index, int) or not (
                0 <= answer_index < len(options)
            ):
                return missing("missing or out-of-range option answer")
            return None
        if benchmark == "triviaqa":
            aliases = raw.get("aliases")
            if not aliases or not isinstance(aliases, list):
                return missing("missing aliases")
            return None
        if benchmark == "humaneval":
            if not str(raw.get("canonical_solution") or "").strip():
                return missing("missing canonical solution")
            if not str(raw.get("test_code") or "").strip():
                return missing("missing test code")
            return None
        if benchmark == "simpleqa":
            if not str(raw.get("answer") or "").strip():
                return missing("missing answer")
            return None
        if benchmark == "healthbench":
            if not str(raw.get("reference_response") or "").strip():
                return missing("missing reference response")
            if not str(raw.get("rubric") or "").strip():
                return missing("missing rubric")
            return None
        raise HfSingleTurnSeedAdapterError(f"Unknown benchmark admission branch: {benchmark}")

# ── Public task_text composition (few-shot / option blocks enter the prompt; gold does not) ────

    def _compose_task_text(
        self, benchmark: str, index: int, raw: Dict[str, Any]
    ) -> str:
        if benchmark == "gsm8k":
            return str(raw.get("question") or "").strip()
        if benchmark == "math500":
            return str(raw.get("problem") or "").strip()
        if benchmark == "mgsm":
            language = str(raw.get("language") or "")
            return f"[{language}] {str(raw.get('question') or '').strip()}"
        if benchmark == "drop":
            selected = self._drop_selected_rows(index)
            pool = self._load_fewshot_pool()
            exemplars = [
                "Passage: {passage}\nQuestion: {question}\nAnswer: {answer}".format(
                    passage=str(pool[row].get("passage") or "").strip(),
                    question=str(pool[row].get("question") or "").strip(),
                    answer=str(pool[row].get("answer") or "").strip(),
                )
                for row in selected
            ]
            return "\n\n".join(exemplars + [
                "Passage: {passage}\nQuestion: {question}\nAnswer:".format(
                    passage=str(raw.get("passage") or "").strip(),
                    question=str(raw.get("question") or "").strip(),
                )
            ])
        if benchmark in ("mmlu", "mmlu_pro", "gpqa"):
            if benchmark == "gpqa":
                options = [str(raw.get("correct_answer") or "")] + [
                    str(raw.get(f"distractor_{i}") or "") for i in (1, 2, 3)
                ]
                order = _gpqa_order(len(options))
                options = [options[position] for position in order]
            else:
                source = raw.get("choices") if benchmark == "mmlu" else (
                    raw.get("options"))
                options = [str(option) for option in (source or [])]
            option_block = "\n".join(
                f"{_LETTERS[position]}. {text}"
                for position, text in enumerate(options)
            )
            return (
                f"{str(raw.get('question') or '').strip()}\n\n{option_block}\n\n"
                "Answer with a single option letter."
            )
        if benchmark == "triviaqa":
            return str(raw.get("question") or "").strip()
        if benchmark == "humaneval":
            prompt = str(raw.get("prompt") or "").strip()
            return (
                "Complete the following Python function. Deliver the full "
                "solution (signature and body) as solution.py.\n\n" + prompt
            )
        if benchmark == "simpleqa":
            return str(raw.get("question") or "").strip()
        if benchmark == "healthbench":
            return str(raw.get("question") or "").strip()
        raise HfSingleTurnSeedAdapterError(f"Unknown task_text branch: {benchmark}")

    def _drop_selected_rows(self, index: int) -> List[int]:
        pool = self._load_fewshot_pool()
        if len(pool) < 3:
            raise HfSingleTurnSeedAdapterError(
                f"drop few-shot pool has fewer than 3 rows: {len(pool)}")
        return sorted(random.Random(42 + index).sample(range(len(pool)), 3))

# ── Trusted-side gold accessor (D5: never enters seed rows / public items) ───────────

    def seed_ids(self, benchmark: Optional[str] = None) -> List[str]:
        return [
            row["seed_id"] for row in self.load_seed_tasks()
            if benchmark is None or row["benchmark"] == benchmark
        ]

    def row(self, seed_id: str) -> Dict[str, Any]:
        for row in self.load_seed_tasks():
            if row["seed_id"] == seed_id:
                return row
        raise HfSingleTurnSeedAdapterError(f"seed row does not exist: {seed_id}")

    def gold_fields(self, seed_id: str) -> Dict[str, Any]:
        """Trusted-side gold: answer / aliases / options / test_code / rubric, etc.

        gpqa rows additionally materialize ``option_order`` and ``answer_letter``; mmlu / mmlu_pro
        materialize ``answer_letter``; drop rows materialize ``few_shot_texts``.
        """
        benchmark, _, remainder = seed_id.partition(":")
        if benchmark not in BENCHMARK_FAMILIES:
            raise HfSingleTurnSeedAdapterError(f"Unknown benchmark: {benchmark}")
        index = int(remainder.rsplit("-", 1)[-1])
        raw = self._load_rows(benchmark)[index]
        if self._exclusion_reason(benchmark, raw) is not None:
            raise HfSingleTurnSeedAdapterError(
                f"excluded row has no gold: {seed_id}")

        if benchmark in ("gsm8k", "math500", "mgsm", "simpleqa"):
            return {"answer": str(raw.get("answer") or "").strip()}
        if benchmark == "drop":
            return {
                "answer": str(raw.get("answer") or "").strip(),
                "answer_spans": [
                    str(span) for span in (raw.get("answer_spans") or [])
                ],
            }
        if benchmark == "mmlu":
            choices = [str(choice) for choice in (raw.get("choices") or [])]
            letter = _LETTERS[int(raw.get("answer"))]
            return {"choices": choices, "answer_letter": letter}
        if benchmark == "mmlu_pro":
            options = [str(option) for option in (raw.get("options") or [])]
            letter = _LETTERS[int(raw.get("answer"))]
            return {"options": options, "answer_letter": letter}
        if benchmark == "gpqa":
            options = [str(raw.get("correct_answer") or "")] + [
                str(raw.get(f"distractor_{i}") or "") for i in (1, 2, 3)
            ]
            order = _gpqa_order(len(options))
            shuffled = [options[position] for position in order]
            return {
                "options": shuffled,
                "answer_letter": _LETTERS[order.index(0)],
                "option_order": order,
            }
        if benchmark == "triviaqa":
            return {
                "aliases": [str(alias) for alias in (raw.get("aliases") or [])],
            }
        if benchmark == "humaneval":
# The prompt ends with a signature line; the canonical solution relies on trailing-newline concatenation,
# which must be restored after strip (losing it produces a SyntaxError solution).
            prompt = str(raw.get("prompt") or "").strip()
            if not prompt.endswith("\n"):
                prompt += "\n"
            return {
                "prompt": prompt,
                "canonical_solution": str(
                    raw.get("canonical_solution") or ""),
                "test_code": str(raw.get("test_code") or "").strip(),
            }
        if benchmark == "healthbench":
            return {
                "reference_response": str(
                    raw.get("reference_response") or "").strip(),
                "rubric": str(raw.get("rubric") or "").strip(),
            }
        raise HfSingleTurnSeedAdapterError(f"Unknown gold branch: {benchmark}")

# ── Prompt archive (D5: package only task_text; gold never enters the archive) ───────────────

    def export_prompt_archive(
        self, archive_path: "str | Path", seed_id: Optional[str] = None
    ) -> str:
        """Freeze all (or one task's) prompts as tar.gz and return SHA-256."""
        archive_path = Path(archive_path)
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive_path, "w:gz") as tar:
            for row in self.load_seed_tasks():
                if seed_id is not None and row["seed_id"] != seed_id:
                    continue
                if not row["task_text"]:
                    raise HfSingleTurnSeedAdapterError(
                        f"excluded row has no prompt to archive: {row['seed_id']}")
                payload = row["task_text"].encode("utf-8")
                info = tarfile.TarInfo(
                    name=f"{row['seed_id'].replace(':', '_')}/prompt.txt")
                info.size = len(payload)
                tar.addfile(info, fileobj=io.BytesIO(payload))
        return hashlib.sha256(archive_path.read_bytes()).hexdigest()


__all__ = [
    "HfSingleTurnSeedAdapter",
    "HfSingleTurnSeedAdapterError",
    "BENCHMARK_FAMILIES",
    "UNIT_TEST_BENCHMARKS",
    "LLM_RUBRIC_BENCHMARKS",
    "SNAPSHOT_SCHEMA",
    "EXCLUDED_REASON_ANSWER",
]
