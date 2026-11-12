"""S2 taxonomy-driven template generation (design §6).

Complements S1 (workbuddy_mutation, mutating existing profiles): S2 finds gaps in seed coverage on the taxonomy grid
(category x difficulty) and uses **new template families** to fill blind spots:

  - InvoiceTotalsFamily     → (data-file-ops, easy/medium) invoice totals and taxes
  - ScheduleConflictsFamily → (doc-ops, medium/hard)        meeting-room conflict arbitration
  - AccessAuditFamily       → (automation-workdir, hard)    permission-matrix violation audit

The correctness red line is the same as S1: gold and verifier are recomputed programmatically (correctness guaranteed by construction),
and the LLM does not participate in judgement (``llm_polish`` is an optional recipe-layer step).
GAP allocation: cells with low seed-grid coverage receive generation quota first.
"""

from __future__ import annotations

import hashlib
import json
import random
from typing import Any, Dict, List, Optional, Tuple

from data_designer.generation.algorithms.builtin.workbuddy_mutation import (
    GENERATOR_CODE_HASH,
    MutationTask,
)

TAXONOMY_GENERATOR_REF_BASE = f"workbuddy_taxonomy@{GENERATOR_CODE_HASH[:16]}"


# ── Template families ───────────────────────────────────────────────────────────────────


class InvoiceTotalsFamily:
    """Invoice line items -> totals including tax/discount (data-file-ops)."""

    name = "invoice_totals"
    cells = (("data-file-ops", "easy"), ("data-file-ops", "medium"))

    def build(self, rng: random.Random, difficulty: str) -> MutationTask:
        n_lines = rng.randint(3, 5) if difficulty == "easy" else rng.randint(6, 9)
        lines = []
        for i in range(n_lines):
            lines.append({
                "item": f"item-{i + 1:02d}",
                "qty": rng.randint(1, 12),
                "unit_price": round(rng.uniform(10, 200), 2),
            })
        tax_rate = rng.choice([0.06, 0.09, 0.13])
        has_discount = difficulty == "medium"
        discount_rate = round(rng.uniform(0.02, 0.10), 2) if has_discount else 0.0

        subtotal = round(sum(l["qty"] * l["unit_price"] for l in lines), 2)
        discount_amount = round(subtotal * discount_rate, 2)
        taxable = round(subtotal - discount_amount, 2)
        tax_amount = round(taxable * tax_rate, 2)
        total = round(taxable + tax_amount, 2)

        lines_text = "\n".join(
            f"- {l['item']}: qty={l['qty']}, unit_price={l['unit_price']:.2f} CNY"
            for l in lines)
        discount_line = (f"\n- 一口价折扣：{discount_rate:.0%}（先折后税）"
                         if has_discount else "")
        task_text = (
            f"财务需要核对发票 [inv-{rng.randint(1000, 9999)}]。"
            f"发票包含以下 {n_lines} 行项目：\n{lines_text}\n"
            f"- 增值税率：{tax_rate:.0%}{discount_line}\n\n"
            "请计算小计（quantity × unit_price 之和）、折扣金额（如有）、"
            "税额（对折后金额计税）与价税合计，把结果写入 output/invoice.json。"
            "必须包含键：line_count, subtotal, discount_amount, tax_amount, total。")
        gold = {
            "line_count": n_lines,
            "subtotal": subtotal,
            "discount_amount": discount_amount,
            "tax_amount": tax_amount,
            "total": total,
        }
        spec = {
            "kind": "artifact_compare",
            "payload": {
                "artifact": {"path": "output/invoice.json", "format": "json"},
                "gold": gold,
                "fields": {
                    "subtotal": {"type": "number", "tolerance": 0.01},
                    "discount_amount": {"type": "number", "tolerance": 0.01},
                    "tax_amount": {"type": "number", "tolerance": 0.01},
                    "total": {"type": "number", "tolerance": 0.01},
                },
            },
            "generator_ref": f"{TAXONOMY_GENERATOR_REF_BASE}#{self.name}",
        }
        return MutationTask(
            seed_id="taxonomy-gap",
            task_text=task_text,
            success_criteria=(
                "output/invoice.json 的 5 个字段全部与独立重算一致"
                f"（税率 {tax_rate:.0%}，折扣率 {discount_rate:.0%}）。"),
            difficulty=difficulty,
            category="data-file-ops",
            family=self.name,
            verifier_spec=spec,
            gold_solution=gold,
            negative_solution={},
            provenance_extra={
                "strategy": "taxonomy",
                "template_slots": {
                    "n_lines": n_lines, "tax_rate": tax_rate,
                    "discount_rate": discount_rate,
                },
            },
        )


class ScheduleConflictsFamily:
    """Meeting-room booking conflict arbitration (doc-ops): first-come, first-served; reject conflicts."""

    name = "schedule_conflicts"
    cells = (("doc-ops", "medium"), ("doc-ops", "hard"))

    def build(self, rng: random.Random, difficulty: str) -> MutationTask:
        rooms = ["room-a", "room-b"]
        n_requests = rng.randint(4, 5) if difficulty == "medium" else rng.randint(6, 7)
        base_hour = 9
        requests = []
        occupied: Dict[str, List[Tuple[int, int]]] = {r: [] for r in rooms}
        for i in range(n_requests):
            if difficulty == "hard" and any(occupied.values()) \
                    and rng.random() < 0.4:
    # hard: force a conflict (reuse the interval of an occupied room)
                room = rng.choice([r for r in rooms if occupied[r]])
                start, end = rng.choice(occupied[room])
            else:
                room = rng.choice(rooms)
                start = base_hour + rng.randint(0, 6)
                end = start + rng.randint(1, 2)
                while any(s < end and start < e for s, e in occupied[room]):
                    start = base_hour + rng.randint(0, 6)
                    end = start + rng.randint(1, 2)
            requests.append({"id": f"mtg-{i + 1:02d}", "room": room,
                             "start": start, "end": end})
            occupied[room].append((start, end))
        requests_shuffled = requests[:]
        rng.shuffle(requests_shuffled)

    # First-come, first-served simulation: the shuffled list is submission order; touching endpoints (s==e) are not a conflict
        accepted: List[Dict[str, Any]] = []
        rejected: List[Dict[str, Any]] = []
        for r in requests_shuffled:
            clash = next((a for a in accepted
                          if a["room"] == r["room"]
                          and a["start"] < r["end"] and r["start"] < a["end"]), None)
            if clash:
                rejected.append({"id": r["id"], "conflicts_with": clash["id"]})
            else:
                accepted.append(dict(r))
        gold = {
            "accepted": sorted(accepted, key=lambda a: (a["start"], a["id"])),
            "rejected": sorted(rejected, key=lambda x: x["id"]),
        }
        req_text = "\n".join(
            f"- {r['id']}: {r['room']} {r['start']:02d}:00-{r['end']:02d}:00"
            for r in requests_shuffled)
        task_text = (
            "行政助理收到以下会议室预约申请（按列出顺序处理，先到先得；"
            "时间区间重叠即冲突，端点相接不算）：\n"
            f"{req_text}\n\n"
            "请生成最终排期 output/schedule.json，包含键："
            "accepted（按开始时间排序的全部通过预约对象数组，元素含 "
            "id/room/start/end）、"
            "rejected（数组，元素 {id, conflicts_with}，conflicts_with 为"
            "与其冲突的已通过预约 id）。")
        spec = {
            "kind": "artifact_compare",
            "payload": {
                "artifact": {"path": "output/schedule.json", "format": "json"},
                "gold": gold,
                "fields": {
                    "accepted": {"type": "record_list", "key_fields": ["id"]},
                    "rejected": {"type": "record_list", "key_fields": ["id"]},
                },
            },
            "generator_ref": f"{TAXONOMY_GENERATOR_REF_BASE}#{self.name}",
        }
        return MutationTask(
            seed_id="taxonomy-gap",
            task_text=task_text,
            success_criteria=(
                f"output/schedule.json：{len(accepted)} 个预约通过、"
                f"{len(rejected)} 个因冲突被拒，逐条与先到先得重算一致。"),
            difficulty=difficulty,
            category="doc-ops",
            family=self.name,
            verifier_spec=spec,
            gold_solution=gold,
            negative_solution={"accepted": [], "rejected": []},
            provenance_extra={
                "strategy": "taxonomy",
                "template_slots": {"n_requests": n_requests},
            },
        )


class AccessAuditFamily:
    """Permission-matrix violation audit (automation-workdir): expired/over-privileged entries."""

    name = "access_audit"
    cells = (("automation-workdir", "hard"),)

    def build(self, rng: random.Random, difficulty: str) -> MutationTask:
        n_users = rng.randint(5, 7)
        users = [f"user-{i:02d}" for i in range(1, n_users + 1)]
        cutoff = f"2026-{rng.randint(1, 9):02d}-01"
        roles = ("reader", "writer", "admin")
        entries = []
        violations = []
        for u in users:
            role = rng.choice(roles)
            expired = rng.random() < 0.4
            entries.append({"user": u, "role": role,
                            "expires": None if not expired else cutoff})
            if expired:
                violations.append({"user": u, "reason": "expired"})
            elif role == "admin" and rng.random() < 0.5:
                violations.append({"user": u, "reason": "excessive"})
        rng.shuffle(entries)
        gold = {
            "violations": sorted(violations, key=lambda v: v["user"]),
            "clean_count": len(users) - len(violations),
        }
        entries_text = json.dumps(entries, ensure_ascii=False, indent=1)
        task_text = (
            f"安全审计（基准日 {cutoff} 之后）：权限条目如下\n"
            f"```json\n{entries_text}\n```\n"
            "违规规则：① expires 非空且早于等于基准日的条目（reason=expired）；"
            "② 未过期但 role=admin 的条目（reason=excessive）。"
            "请输出审计报告 output/access_report.json，键："
            "violations（数组，元素 {user, reason}，按 user 排序）、"
            "clean_count（无违规条目数）。")
        spec = {
            "kind": "artifact_compare",
            "payload": {
                "artifact": {"path": "output/access_report.json",
                             "format": "json"},
                "gold": gold,
                "fields": {
                    "violations": {"type": "record_list",
                                   "key_fields": ["user"]},
                    "clean_count": {"type": "integer"},
                },
            },
            "generator_ref": f"{TAXONOMY_GENERATOR_REF_BASE}#{self.name}",
        }
        return MutationTask(
            seed_id="taxonomy-gap",
            task_text=task_text,
            success_criteria=(
                f"output/access_report.json：{len(violations)} 条违规逐条命中规则，"
                f"clean_count={gold['clean_count']}。"),
            difficulty=difficulty,
            category="automation-workdir",
            family=self.name,
            verifier_spec=spec,
            gold_solution=gold,
            negative_solution={"violations": [], "clean_count": 0},
            provenance_extra={
                "strategy": "taxonomy",
                "template_slots": {"n_users": n_users, "cutoff": cutoff},
            },
        )


_TAXONOMY_FAMILIES = [InvoiceTotalsFamily(), ScheduleConflictsFamily(),
                      AccessAuditFamily()]


# ── Coverage gaps and engine ───────────────────────────────────────────────────────────


def taxonomy_gaps(
    seed_grid: Dict[str, Dict[str, int]],
) -> List[Tuple[str, str]]:
    """Seed coverage grid -> gap cells that templates can fill (ascending coverage; zero coverage first)."""
    cells: List[Tuple[str, str, int]] = []
    supported = {(cat, diff) for fam in _TAXONOMY_FAMILIES
                 for cat, diff in fam.cells}
    for cat, diffs in (seed_grid or {}).items():
        for diff, n in diffs.items():
            if (cat, diff) in supported:
                cells.append((cat, diff, int(n)))
    for fam in _TAXONOMY_FAMILIES:
        for cat, diff in fam.cells:
            if all(c != (cat, diff) for c, _, _ in cells):
                cells.append((cat, diff, 0))
    cells.sort(key=lambda c: c[2])
    return [(cat, diff) for cat, diff, _ in cells]


class WorkbuddyTaxonomyEngine:
    """S2 engine: gap cells -> template-family generation (deterministic, reproducible from the global seed)."""

    def __init__(self, seed: int = 0):
        self.seed = int(seed)

    def generate(
        self,
        seed_grid: Optional[Dict[str, Dict[str, int]]],
        count: int,
        exclude_fingerprints: Optional[List[str]] = None,
    ) -> List[MutationTask]:
        exclude = set(exclude_fingerprints or [])
        gaps = taxonomy_gaps(seed_grid or {})
        if not gaps:
            raise ValueError("taxonomy_gaps is empty: template families and seed grid have no intersection")
        tasks: List[MutationTask] = []
        i = 0
        attempts = 0
        max_attempts = max(count * 8, 64)
        while len(tasks) < count and attempts < max_attempts:
            attempts += 1
            cat, diff = gaps[i % len(gaps)]
            fam = next(f for f in _TAXONOMY_FAMILIES if (cat, diff) in f.cells)
            rng = random.Random(f"{self.seed}:{i}:{fam.name}")
            task = fam.build(rng, diff)
            i += 1
            if _fingerprint_of(task) in exclude:
                continue
            tasks.append(task)
        if len(tasks) < count:
            raise ValueError(
                f"taxonomy engine generated only {len(tasks)}/{count} tasks "
                f"within {max_attempts} attempts (exclusion set too large or templates exhausted)")
        return tasks


def _fingerprint_of(task: MutationTask) -> str:
    """Content deduplication key before constructing a release row (SHA truncation isomorphic to wb_item_fingerprint)."""
    probe = {
        "seed_id": task.seed_id,
        "task_text": task.task_text,
        "tools_available": [],
        "success_criteria": task.success_criteria,
        "ground_truth_action_sequence": [],
        "scoring_rubric": "",
        "difficulty": task.difficulty,
        "verifier_spec": task.verifier_spec,
    }
    return hashlib.sha256(json.dumps(
        probe, sort_keys=True, ensure_ascii=False,
        separators=(",", ":")).encode("utf-8")).hexdigest()[:16]


__all__ = [
    "WorkbuddyTaxonomyEngine",
    "taxonomy_gaps",
    "TAXONOMY_GENERATOR_REF_BASE",
]
