"""S1 parameterized mutation line (mutation twins, design §6): wb seed profile -> controlled mutation tasks.

Discipline (aligned with deterministic synthesis in GymBenchmarkSynthesisOperation):
  - Pure Python, no LLM calls (LLM polishing is an optional recipe-layer step, disabled by default);
  - Each item = seed profile x mutation operator combination x random parameters, reproducible from the global seed (RSI L3);
  - ground truth and verifier are recomputed programmatically in sync--correctness is guaranteed by construction;
  - Seed profiles provide only metadata such as category/difficulty; all item content is freshly
    generated synthetic values, with zero text overlap with original workbuddy-bench items (contamination red line).

The three task template families align with Office subset categories (docs/wbbench_schema_notes.md §1):
  data-file-ops       → SalesRollupFamily     (aggregation + filtering + polarity/boundary mutations)
  doc-ops             → MinutesExtractFamily  (structured extraction + normalization + missing-value semantics)
  automation-workdir  → WorkdirReorgFamily    (rule execution + threshold/polarity mutations)
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

GENERATOR_NAME = "workbuddy_mutation"
GENERATOR_CODE_HASH = hashlib.sha256(
    Path(__file__).read_bytes()).hexdigest()


def generator_ref(seed: int) -> str:
    """verifier_spec.generator_ref: fingerprint of the code/parameters that produced this verifier (§7.1)."""
    return f"{GENERATOR_NAME}@{GENERATOR_CODE_HASH[:16]}#seed={seed}"


@dataclass
class MutationTask:
    """A mutation candidate item (includes constructed positive/negative solutions for dual validation during trial runs)."""

    seed_id: str
    task_text: str
    success_criteria: str
    difficulty: str
    category: str
    family: str
    verifier_spec: Dict[str, Any]
    gold_solution: Dict[str, Any]
    negative_solution: Dict[str, Any]
    provenance_extra: Dict[str, Any] = field(default_factory=dict)


# ── Synthetic material pool (all fictional, avoiding overlap with real task text) ────────────────────────────

_REGION_POOL = [
    ("north", ["N", "北区", "NO"]), ("south", ["S", "南区", "SO"]),
    ("east", ["E", "东区", "EA"]), ("west", ["W", "西区", "WE"]),
    ("central", ["C", "中区", "CE"]),
]
_PRODUCT_POOL = ["alpha-kit", "beta-kit", "gamma-kit", "delta-kit"]

_NAME_POOL = [
    ("陈立", ["LC", "小陈"]), ("李霜", ["LS", "小李"]),
    ("赵嵘", ["ZR", "老赵"]), ("王芷", ["WZ", "小王"]),
]
_ACTION_POOL = [
    "提交联调计划", "整理测试用例清单", "确认验收环境", "更新风险登记表",
    "复核容量评估", "同步进度看板", "整理复盘纪要", "校准监控阈值",
]

_EXT_POOL = ["csv", "md", "json", "log", "xlsx"]


def _fmt_date(d: date) -> str:
    return d.isoformat()


def _round1(x: float) -> float:
    return round(x + 1e-9, 1)


# ── Family A: data-file-ops sales/data aggregation ────────────────────────────────────

class SalesRollupFamily:
    """CSV aggregation: aggregation operators x filter thresholds x polarity x missing-value semantics; gold is recomputed programmatically."""

    name = "sales_rollup"
    category = "data-file-ops"

    def build(self, rng: random.Random, profile: Dict[str, Any], index: int) -> MutationTask:
        n_regions = rng.randint(3, 5)
        regions = rng.sample(_REGION_POOL, n_regions)
        n_rows = rng.randint(8, 18)
        n_missing = rng.randint(1, 3)
        missing_idx = set(rng.sample(range(n_rows), n_missing))

        agg_op = rng.choice(["sum", "max", "min"])
        polarity = rng.choice(["highest", "lowest"])
        threshold_enabled = rng.random() < 0.5
        threshold = rng.randint(5, 30) if threshold_enabled else 0

        rows = []
        for i in range(n_rows):
            region_code = regions[rng.randrange(n_regions)][0][0].upper()
            rows.append({
                "row": i + 1, "region": region_code,
                "product": rng.choice(_PRODUCT_POOL),
                "units": rng.randint(1, 60),
                "revenue_cny": None if i in missing_idx else _round1(rng.uniform(50, 2000)),
            })

        codebook = {r[0][0].upper(): r[0] for r in regions}

# ── Programmatic gold recomputation (single source of truth) ──
        included = [r for r in rows
                    if r["revenue_cny"] is not None and r["units"] >= threshold]
        excluded_rows = n_rows - len(included)
        by_region: Dict[str, float] = {}
        for r in included:
            name = codebook[r["region"]]
            val = r["revenue_cny"]
            if agg_op == "sum":
                by_region[name] = _round1(by_region.get(name, 0.0) + val)
            elif agg_op == "max":
                by_region[name] = max(by_region.get(name, float("-inf")), val)
            else:
                by_region[name] = min(by_region.get(name, float("inf")), val)
        by_region = {k: _round1(v) for k, v in by_region.items()}
        focus = (max if polarity == "highest" else min)(by_region, key=by_region.get)
        revenues = [r["revenue_cny"] for r in included]
        overall = (sum(revenues) if agg_op == "sum"
                   else (max(revenues) if agg_op == "max" else min(revenues)))
        metric_name = {
            "sum": "total_revenue_cny", "max": "max_single_revenue_cny",
            "min": "min_single_revenue_cny",
        }[agg_op]
        gold = {
            "metric": metric_name,
            "metric_value": _round1(overall),
            "by_region": by_region,
            "focus_region": focus,
            "excluded_rows": excluded_rows,
        }

        csv_lines = ["region,product,units,revenue_cny"] + [
            f"{r['region']},{r['product']},{r['units']},"
            f"{'' if r['revenue_cny'] is None else r['revenue_cny']}" for r in rows]
        codebook_text = ", ".join(f'"{k}": "{v}"' for k, v in codebook.items())

        ops = ["agg_variant:" + agg_op, "polarity_flip:" + polarity]
        if threshold_enabled:
            ops.append(f"boundary_threshold:units>={threshold}")
        ops.append(f"missing_policy:exclude={n_missing}")

        threshold_text = (
            f"只统计 units ≥ {threshold} 的行；" if threshold_enabled
            else "所有 revenue_cny 非空的行都参与统计；")
        metric_text = {
            "sum": "revenue 合计（total_revenue_cny）",
            "max": "最大单行 revenue（max_single_revenue_cny）",
            "min": "最小单行 revenue（min_single_revenue_cny）",
        }[agg_op]
        task_text = (
            "你是业务数据汇总助理。下面给出一份合成销售明细与地区代码表，"
            "请只依据给定数据完成统计，不联网、不补猜缺失值。\n\n"
            "```csv sales.csv\n" + "\n".join(csv_lines) + "\n```\n\n"
            "```json region_codebook.json\n{" + codebook_text + "}\n```\n\n"
            "请生成 `output/summary.json`，字段与口径如下：\n"
            f"- `metric`：固定字符串 \"{metric_name}\"；\n"
            f"- `metric_value`：全表{metric_text}，保留一位小数；\n"
            "- `by_region`：按标准地区名（用代码表归一）分组给出同样的"
            "聚合指标，保留一位小数；\n"
            f"- `focus_region`：by_region 中指标值"
            f"{('最高' if polarity == 'highest' else '最低')}的标准地区名；\n"
            "- `excluded_rows`：未参与统计的行数（revenue_cny 为空"
            + (f"或 units < {threshold}" if threshold_enabled else "") + "）。\n"
            f"{threshold_text}"
            "revenue_cny 为空的行不得计入聚合，也不得补猜。\n\n"
            "最终答案只输出一个 JSON 代码块，内容为 `output/summary.json` 的完整内容。"
        )
        criteria = (
            f"summary.json 的 {len(gold)} 个字段与程序化金标一致"
            f"（metric={metric_name}, focus={polarity}, threshold={threshold}）"
        )
        artifact_payload = {
            "artifact": {"path": "output/summary.json", "format": "json"},
            "gold": gold,
            "fields": {
                "metric": {"type": "string"},
                "metric_value": {"type": "number", "tolerance": 0.01},
                "by_region": {"type": "number_map", "tolerance": 0.01},
                "focus_region": {"type": "string"},
                "excluded_rows": {"type": "integer"},
            },
        }

        return MutationTask(
            seed_id=f"{profile.get('seed_id', 'profile')}-mut{index:04d}",
            task_text=task_text,
            success_criteria=criteria,
            difficulty=profile.get("difficulty", "medium"),
            category=self.category,
            family=self.name,
            verifier_spec={
                "kind": "artifact_compare",
                "payload": artifact_payload,
                "generator_ref": generator_ref(-1),  # Overwritten by the engine with the seeded version
            },
            gold_solution=gold,
            negative_solution={},
            provenance_extra={
                "seed_task_id": profile.get("seed_id"),
                "strategy": "mutation",
                "mutation_ops": ops,
                "derivative": False,
            },
        )


# ── Family B: doc-ops action-item extraction ─────────────────────────────────────────────

class MinutesExtractFamily:
    """Meeting minutes -> action-item JSON: alias normalization + due-date handling + completion-state semantics."""

    name = "minutes_extract"
    category = "doc-ops"

    def build(self, rng: random.Random, profile: Dict[str, Any], index: int) -> MutationTask:
        members = rng.sample(_NAME_POOL, rng.randint(3, 4))
        roster = {name: aliases for name, aliases in members}
        due_rule = rng.choice(["friday_same_week", "wednesday_next_week"])
        status_default = rng.choice(["open", "pending"])
        n_items = rng.randint(3, 6)
        n_done = rng.randint(1, max(1, n_items - 2))
        n_distractors = rng.randint(1, 2)

        meet1 = date(2026, 1, 5) + timedelta(days=7 * rng.randint(0, 6))
        meet2 = meet1 + timedelta(days=7)
        due_offset = {"friday_same_week": 4 - meet1.weekday(),
                      "wednesday_next_week": 9 - meet1.weekday()}[due_rule]

        actions, done_flags = [], []
        for i in range(n_items):
            name, _aliases = members[rng.randrange(len(members))]
            actions.append((name, rng.choice(_ACTION_POOL)))
            done_flags.append(i < n_done)
        distractors = [rng.choice(_ACTION_POOL) + "（讨论中）" for _ in range(n_distractors)]

# ── Programmatic gold recomputation ──
        gold_actions = [
            {
                "owner": name,
                "due": _fmt_date(meet1 + timedelta(days=due_offset)),
                "action": verb_obj,
                "source_file": "input/minutes_m1.md",
                "status": "done" if done else status_default,
            }
            for (name, verb_obj), done in zip(actions, done_flags)
        ]
        gold_actions.sort(key=lambda r: (r["owner"], r["due"], r["action"]))
        gold = {"actions": gold_actions}

        roster_text = ", ".join(
            f'"{name}"（别名: {"/".join(aliases)}）' for name, aliases in roster.items())
        item_lines = [
            f"- 行动项：{verb_obj}（负责人用别名标注：{rng.choice(roster[name])}）"
            f"{'；已在 m2 纪要确认完成' if done else ''}"
            for (name, verb_obj), done in zip(actions, done_flags)
        ]
        distractor_lines = [f"- 讨论记录：{d}" for d in distractors]

        due_text = {
            "friday_same_week": "行动项截止日为该会议当周的周五",
            "wednesday_next_week": "行动项截止日为该会议下周的周三",
        }[due_rule]
        task_text = (
            "你是会议纪要整理助理。下面给出两份合成例会纪要与成员花名册，"
            "请只依据给定文本抽取行动项，不联网、不臆造。\n\n"
            "```md input/minutes_m1.md\n"
            f"# 例会纪要 {_fmt_date(meet1)}\n" + "\n".join(item_lines) + "\n```\n\n"
            "```md input/minutes_m2.md\n"
            f"# 例会纪要 {_fmt_date(meet2)}\n"
            "本次会议确认部分上次行动项的完成情况（见各条目标注）；"
            "其余条目维持未完成状态。\n```\n\n"
            f"```json input/owner_roster.json\n{{{roster_text}}}\n```\n\n"
            "请生成 `output/actions.json`，顶层为对象 {\"actions\": [...]}，"
            "每条行动项包含：\n"
            "- `owner`：负责人标准姓名（用花名册归一别名）；\n"
            f"- `due`：日期字符串（{due_text}，格式 YYYY-MM-DD）；\n"
            "- `action`：行动项一句话描述（讨论记录不是行动项，不得收录）；\n"
            "- `source_file`：\"input/minutes_m1.md\"；\n"
            f"- `status`：纪要标注完成的写 \"done\"，其余写 \"{status_default}\"。\n\n"
            "讨论记录（仅供背景，不得作为行动项输出）：\n" + "\n".join(distractor_lines)
            + "\n\n最终答案只输出一个 JSON 代码块，"
            "内容为 `output/actions.json` 的完整内容。"
        )
        criteria = (
            f"actions.json 的 {n_items} 条行动项与程序化金标一致"
            f"（owner 归一 + due={due_rule} + status 语义，排除 {n_distractors} 条讨论项）"
        )
        artifact_payload = {
            "artifact": {"path": "output/actions.json", "format": "json"},
            "gold": gold,
            "fields": {
                "actions": {"type": "record_list", "key_fields": ["owner", "due", "action"]},
            },
        }

        ops = [
            f"date_rule:{due_rule}",
            f"status_default:{status_default}",
            "owner_alias:sample",
        ]
        return MutationTask(
            seed_id=f"{profile.get('seed_id', 'profile')}-mut{index:04d}",
            task_text=task_text,
            success_criteria=criteria,
            difficulty=profile.get("difficulty", "medium"),
            category=self.category,
            family=self.name,
            verifier_spec={
                "kind": "artifact_compare",
                "payload": artifact_payload,
                "generator_ref": generator_ref(-1),
            },
            gold_solution=gold,
            negative_solution={"actions": []},
            provenance_extra={
                "seed_task_id": profile.get("seed_id"),
                "strategy": "mutation",
                "mutation_ops": ops,
                "derivative": False,
            },
        )


# ── Family C: automation-workdir workspace reorganization ─────────────────────────────────

class WorkdirReorgFamily:
    """File manifest -> archive plan JSON: date threshold + extension filtering + polarity mutation."""

    name = "workdir_reorg"
    category = "automation-workdir"

    def build(self, rng: random.Random, profile: Dict[str, Any], index: int) -> MutationTask:
        n_files = rng.randint(8, 14)
        cutoff = date(2026, 3, 1) + timedelta(days=30 * rng.randint(0, 5))
        polarity = rng.choice(["older", "newer"])
        exts = rng.sample(_EXT_POOL, rng.randint(2, 3))
        files = []
        for i in range(n_files):
            ext = rng.choice(exts)
            f_date = cutoff + timedelta(days=rng.randint(-40, 40))
            files.append({
                "name": f"artifact_{i:02d}.{ext}", "ext": ext,
                "size_kb": rng.randint(1, 900), "mtime": _fmt_date(f_date),
            })

# ── Programmatic gold recomputation (files exactly at cutoff stay untouched) ──
        selected_files = [
            f for f in files
            if date.fromisoformat(f["mtime"]) != cutoff
            and (date.fromisoformat(f["mtime"]) < cutoff) == (polarity == "older")
        ]
        moves = [
            {"file": f["name"], "target": f"archive/{f['ext']}/{f['name']}"}
            for f in selected_files
        ]
        moves.sort(key=lambda m: m["file"])
        gold = {
            "cutoff": _fmt_date(cutoff),
            "moves": moves,
            "archive_count": len(moves),
            "remaining_count": n_files - len(moves),
        }

        file_lines = [
            f"{f['name']}  size_kb={f['size_kb']}  mtime={f['mtime']}" for f in files]
        polarity_text = "早于" if polarity == "older" else "晚于"
        task_text = (
            "你是工作区治理助理。下面给出一个合成工作区的文件清单（名称/大小/修改日期），"
            "请按规则产出归档计划，只依据给定信息，不联网。\n\n"
            "```text workspace_manifest.txt\n" + "\n".join(file_lines) + "\n```\n\n"
            f"归档规则：把修改日期{polarity_text} {gold['cutoff']} 的文件移动到 "
            "`archive/<扩展名>/<原文件名>`（按扩展名分子目录）；"
            f"恰好等于 {gold['cutoff']} 的文件不动。\n\n"
            "请生成 `output/reorg_plan.json`：\n"
            "- `cutoff`：规则截止日期（YYYY-MM-DD）；\n"
            "- `moves`：数组，每项 {\"file\": 原名, \"target\": 目标路径}，按 file 升序；\n"
            "- `archive_count`：移动文件数；\n"
            "- `remaining_count`：留在原地的文件数。\n\n"
            "最终答案只输出一个 JSON 代码块，内容为 `output/reorg_plan.json` 的完整内容。"
        )
        criteria = (
            f"reorg_plan.json 与程序化金标一致（{polarity} cutoff {gold['cutoff']}，"
            f"{len(moves)}/{n_files} 个文件归档）"
        )
        artifact_payload = {
            "artifact": {"path": "output/reorg_plan.json", "format": "json"},
            "gold": gold,
            "fields": {
                "cutoff": {"type": "string"},
                "moves": {"type": "record_list", "key_fields": ["file"]},
                "archive_count": {"type": "integer"},
                "remaining_count": {"type": "integer"},
            },
        }

        ops = [f"threshold_date:{gold['cutoff']}", f"polarity_flip:{polarity}",
               f"ext_filter:{'+'.join(exts)}"]
        return MutationTask(
            seed_id=f"{profile.get('seed_id', 'profile')}-mut{index:04d}",
            task_text=task_text,
            success_criteria=criteria,
            difficulty=profile.get("difficulty", "medium"),
            category=self.category,
            family=self.name,
            verifier_spec={
                "kind": "artifact_compare",
                "payload": artifact_payload,
                "generator_ref": generator_ref(-1),
            },
            gold_solution=gold,
            negative_solution={},
            provenance_extra={
                "seed_task_id": profile.get("seed_id"),
                "strategy": "mutation",
                "mutation_ops": ops,
                "derivative": False,
            },
        )


# ── Engine ─────────────────────────────────────────────────────────────────────

FAMILIES = {
    "data-file-ops": SalesRollupFamily(),
    "doc-ops": MinutesExtractFamily(),
    "automation-workdir": WorkdirReorgFamily(),
}
_FAMILY_CYCLE = [SalesRollupFamily, MinutesExtractFamily, WorkdirReorgFamily]

_DIFFICULTY_CYCLE = ["easy", "medium", "hard"]

#: Built-in synthetic profiles when no real seed profiles are available (category/difficulty metadata only)
DEFAULT_PROFILES = [
    {"seed_id": "profile-data-file-ops", "category": "data-file-ops"},
    {"seed_id": "profile-doc-ops", "category": "doc-ops"},
    {"seed_id": "profile-automation-workdir", "category": "automation-workdir"},
]


class WorkbuddyMutationEngine:
    """Seed profile x mutation operator x seeded RNG -> MutationTask list (S1 production-line core)."""

    name = GENERATOR_NAME

    def __init__(self, seed: int = 0):
        self.seed = int(seed)

    def generate(
        self,
        profiles: Optional[List[Dict[str, Any]]] = None,
        count: int = 10,
    ) -> List[MutationTask]:
        profiles = list(profiles) if profiles else list(DEFAULT_PROFILES)
        if not profiles:
            raise ValueError("profiles is empty and no built-in default profile is available: S1 cannot generate tasks")
        tasks: List[MutationTask] = []
        for i in range(count):
            profile = profiles[i % len(profiles)]
            family_cls = self._family_for(profile, i)
            rng = random.Random(f"{self.seed}:{i}:{family_cls.name}")
            difficulty = profile.get("difficulty") or \
                _DIFFICULTY_CYCLE[i % len(_DIFFICULTY_CYCLE)]
            prof = dict(profile, difficulty=difficulty)
            task = family_cls.build(rng, prof, i)
            task.verifier_spec["generator_ref"] = generator_ref(self.seed)
            tasks.append(task)
        return tasks

    @staticmethod
    def _family_for(profile: Dict[str, Any], index: int):
        family = FAMILIES.get(str(profile.get("category") or ""))
        if family is not None:
            return family
        return _FAMILY_CYCLE[index % len(_FAMILY_CYCLE)]()


__all__ = [
    "MutationTask",
    "WorkbuddyMutationEngine",
    "FAMILIES",
    "DEFAULT_PROFILES",
    "generator_ref",
    "GENERATOR_CODE_HASH",
]
