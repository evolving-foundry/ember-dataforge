"""S3 targeted failure-cluster probes (design §6).

Merges with the trajectory synthesis line (``rsi/ops/benchmark_synthesis.py``, trajectories -> general exams) but keeps artifacts separate:
this line generates **minimal probe tasks in which each failure mode must occur**,
used after a skill release to verify whether the failure cluster disappears (disappearance-curve resolution).

Probe families are dispatched by error_category (programmatic gold, correctness guaranteed by construction):
  - assertion     BoundaryProbeFamily      threshold boundary (>= vs >) trap
  - missing_data  MissingFieldProbeFamily  empty rows must be removed before aggregation
  - duplicate     DuplicateRowProbeFamily  duplicate records must be deduplicated before aggregation
  - unit          UnitMismatchProbeFamily  mixed currencies converted at exchange rates
  - other         RoundingProbeFamily      precision/rounding semantics
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional

from data_designer.generation.algorithms.builtin.workbuddy_mutation import (
    GENERATOR_CODE_HASH,
    MutationTask,
)

PROBE_GENERATOR_REF_BASE = f"workbuddy_probe@{GENERATOR_CODE_HASH[:16]}"

DEFAULT_CLUSTER_ERROR_CATEGORY = "assertion"


class BoundaryProbeFamily:
    """Threshold boundary trap: whether the target line includes the boundary (>=), and it must."""

    name = "probe_boundary"

    def build(self, rng: random.Random, cluster: Dict[str, Any]) -> MutationTask:
        n = rng.randint(4, 6)
        threshold = rng.randint(60, 90)
        rows = []
        for i in range(n):
            rows.append({"region": f"r{i + 1}", "score": rng.randint(50, 100)})
        # Ensure the boundary value occurs (a must-occur trap)
        rows[rng.randrange(n)]["score"] = threshold
        gold_rows = sorted((r for r in rows if r["score"] >= threshold),
                           key=lambda r: r["region"])
        gold = {"qualified": gold_rows, "qualified_count": len(gold_rows)}
        rows_text = "\n".join(f"- {r['region']}: {r['score']}" for r in rows)
        task_text = (
            "质量看板需要筛选达标区域。达标线（含边界，即恰好等于达标线也算达标）："
            f"score >= {threshold}。各区域分数：\n{rows_text}\n\n"
            "请输出 output/qualified.json，键：qualified（数组，元素 {region, score}，"
            "按 region 排序）、qualified_count（整数）。")
        spec = {
            "kind": "artifact_compare",
            "payload": {
                "artifact": {"path": "output/qualified.json", "format": "json"},
                "gold": gold,
                "fields": {
                    "qualified": {"type": "record_list", "key_fields": ["region"]},
                    "qualified_count": {"type": "integer"},
                },
            },
            "generator_ref": f"{PROBE_GENERATOR_REF_BASE}#{self.name}",
        }
        return MutationTask(
            seed_id=str(cluster.get("cluster_id") or "probe"),
            task_text=task_text,
            success_criteria=(
                f"边界值 {threshold} 必须计入达标（>= 语义），与独立重算逐条一致。"),
            difficulty="easy",
            category="automation-workdir",
            family=self.name,
            verifier_spec=spec,
            gold_solution=gold,
            negative_solution={"qualified": [], "qualified_count": 0},
            provenance_extra={
                "strategy": "failure_probe",
                "failure_cluster_id": cluster.get("cluster_id"),
                "error_category": cluster.get("error_category"),
            },
        )


class MissingFieldProbeFamily:
    """Remove empty rows before aggregation (missing_data failure cluster)."""

    name = "probe_missing_data"

    def build(self, rng: random.Random, cluster: Dict[str, Any]) -> MutationTask:
        n = rng.randint(5, 7)
        rows = []
        for i in range(n):
            missing = rng.random() < 0.35
            rows.append({
                "store": f"s{i + 1:02d}",
                "revenue": None if missing else round(rng.uniform(80, 900), 1),
            })
        valid = [r for r in rows if r["revenue"] is not None]
        gold = {
            "total_revenue": round(sum(r["revenue"] for r in valid), 2),
            "reported_stores": len(valid),
        }
        rows_text = "\n".join(
            f"- {r['store']}: "
            f"revenue={r['revenue'] if r['revenue'] is not None else 'null'}"
            for r in rows)
        task_text = (
            "部分门店未上报营收（revenue=null），这些店**不参与**合计。"
            f"数据：\n{rows_text}\n\n"
            "请输出 output/revenue_summary.json，键：total_revenue（有效营收之和）、"
            "reported_stores（实际上报的门店数）。")
        spec = {
            "kind": "artifact_compare",
            "payload": {
                "artifact": {"path": "output/revenue_summary.json",
                             "format": "json"},
                "gold": gold,
                "fields": {
                    "total_revenue": {"type": "number", "tolerance": 0.01},
                },
            },
            "generator_ref": f"{PROBE_GENERATOR_REF_BASE}#{self.name}",
        }
        return MutationTask(
            seed_id=str(cluster.get("cluster_id") or "probe"),
            task_text=task_text,
            success_criteria=(
                f"{len(rows) - len(valid)} 个空值行被剔除，合计只含有效营收。"),
            difficulty="medium",
            category="data-file-ops",
            family=self.name,
            verifier_spec=spec,
            gold_solution=gold,
            negative_solution={"total_revenue": 0, "reported_stores": 0},
            provenance_extra={
                "strategy": "failure_probe",
                "failure_cluster_id": cluster.get("cluster_id"),
                "error_category": cluster.get("error_category"),
            },
        )


class DuplicateRowProbeFamily:
    """Deduplicate records before aggregation (duplicate failure cluster)."""

    name = "probe_duplicate"

    def build(self, rng: random.Random, cluster: Dict[str, Any]) -> MutationTask:
        n = rng.randint(4, 6)
        rows = [{"order_id": f"o{i + 1:03d}",
                 "amount": round(rng.uniform(20, 500), 2)} for i in range(n)]
        dup_source = rng.choice(rows)
        rows.append(dict(dup_source))  # One fully duplicated row
        rng.shuffle(rows)
        deduped = {(r["order_id"], r["amount"]): r for r in rows}.values()
        gold = {
            "net_total": round(sum(r["amount"] for r in deduped), 2),
            "order_count": len(deduped),
        }
        rows_text = "\n".join(f"- {r['order_id']}: {r['amount']}" for r in rows)
        task_text = (
            "导入日志出现重复行（order_id 与 amount 完全相同的记录只计一次）。"
            f"原始记录：\n{rows_text}\n\n"
            "请输出 output/net_orders.json，键：net_total（去重后金额合计）、"
            "order_count（去重后订单数）。")
        spec = {
            "kind": "artifact_compare",
            "payload": {
                "artifact": {"path": "output/net_orders.json", "format": "json"},
                "gold": gold,
                "fields": {
                    "net_total": {"type": "number", "tolerance": 0.01},
                    "order_count": {"type": "integer"},
                },
            },
            "generator_ref": f"{PROBE_GENERATOR_REF_BASE}#{self.name}",
        }
        return MutationTask(
            seed_id=str(cluster.get("cluster_id") or "probe"),
            task_text=task_text,
            success_criteria=(
                "完全重复的记录只计一次，合计与计数基于去重后的订单集。"),
            difficulty="medium",
            category="data-file-ops",
            family=self.name,
            verifier_spec=spec,
            gold_solution=gold,
            negative_solution={"net_total": 0, "order_count": 0},
            provenance_extra={
                "strategy": "failure_probe",
                "failure_cluster_id": cluster.get("cluster_id"),
                "error_category": cluster.get("error_category"),
            },
        )


class UnitMismatchProbeFamily:
    """Convert mixed currencies at the given exchange rates (unit failure cluster)."""

    name = "probe_unit"

    def build(self, rng: random.Random, cluster: Dict[str, Any]) -> MutationTask:
        rate = round(rng.uniform(6.8, 7.4), 2)
        rows = []
        for i in range(rng.randint(3, 5)):
            rows.append({
                "market": f"m{i + 1}",
                "amount": round(rng.uniform(50, 400), 2),
                "currency": rng.choice(["CNY", "USD"]),
            })
        gold = {
            "total_cny": round(sum(
                r["amount"] if r["currency"] == "CNY" else r["amount"] * rate
                for r in rows), 2),
        }
        rows_text = "\n".join(
            f"- {r['market']}: {r['amount']} {r['currency']}" for r in rows)
        task_text = (
            f"汇率固定为 1 USD = {rate} CNY。各地销售额：\n{rows_text}\n\n"
            "请输出 output/consolidated.json，键：total_cny（全部折算为 CNY 后的"
            "合计，保留两位小数）。")
        spec = {
            "kind": "artifact_compare",
            "payload": {
                "artifact": {"path": "output/consolidated.json",
                             "format": "json"},
                "gold": gold,
                "fields": {"total_cny": {"type": "number", "tolerance": 0.01}},
            },
            "generator_ref": f"{PROBE_GENERATOR_REF_BASE}#{self.name}",
        }
        return MutationTask(
            seed_id=str(cluster.get("cluster_id") or "probe"),
            task_text=task_text,
            success_criteria=(
                f"USD 金额按 1:{rate} 折算后与 CNY 金额求和，容差 0.01。"),
            difficulty="medium",
            category="data-file-ops",
            family=self.name,
            verifier_spec=spec,
            gold_solution=gold,
            negative_solution={"total_cny": 0},
            provenance_extra={
                "strategy": "failure_probe",
                "failure_cluster_id": cluster.get("cluster_id"),
                "error_category": cluster.get("error_category"),
            },
        )


class RoundingProbeFamily:
    """Precision semantics: two decimal places, round half up (fallback probe family)."""

    name = "probe_rounding"

    def build(self, rng: random.Random, cluster: Dict[str, Any]) -> MutationTask:
        rows = [{"dept": f"d{i + 1}", "ratio": round(rng.uniform(0.1, 0.9), 4)}
                for i in range(rng.randint(3, 5))]
        gold = {
            "avg_ratio": round(sum(r["ratio"] for r in rows) / len(rows), 2),
        }
        rows_text = "\n".join(f"- {r['dept']}: {r['ratio']}" for r in rows)
        task_text = (
            f"各部门转化率：\n{rows_text}\n\n"
            "请输出 output/avg_ratio.json，键：avg_ratio（算术平均，"
            "四舍五入保留两位小数）。")
        spec = {
            "kind": "artifact_compare",
            "payload": {
                "artifact": {"path": "output/avg_ratio.json", "format": "json"},
                "gold": gold,
                "fields": {"avg_ratio": {"type": "number", "tolerance": 0.005}},
            },
            "generator_ref": f"{PROBE_GENERATOR_REF_BASE}#{self.name}",
        }
        return MutationTask(
            seed_id=str(cluster.get("cluster_id") or "probe"),
            task_text=task_text,
            success_criteria="算术平均并按四舍五入保留两位小数。",
            difficulty="easy",
            category="doc-ops",
            family=self.name,
            verifier_spec=spec,
            gold_solution=gold,
            negative_solution={"avg_ratio": 0},
            provenance_extra={
                "strategy": "failure_probe",
                "failure_cluster_id": cluster.get("cluster_id"),
                "error_category": cluster.get("error_category"),
            },
        )


_PROBE_FAMILIES = {
    "assertion": BoundaryProbeFamily,
    "missing_data": MissingFieldProbeFamily,
    "duplicate": DuplicateRowProbeFamily,
    "unit": UnitMismatchProbeFamily,
}
_PROBE_DEFAULT_FAMILY = RoundingProbeFamily


class WorkbuddyFailureProbeEngine:
    """S3 engine: failure clusters -> minimal probe item set per cluster (deterministic, reproducible from seed)."""

    def __init__(self, seed: int = 0):
        self.seed = int(seed)

    def generate(
        self,
        clusters: List[Dict[str, Any]],
        per_cluster: int = 1,
        exclude_fingerprints: Optional[List[str]] = None,
    ) -> List[MutationTask]:
        from data_designer.generation.algorithms.builtin.workbuddy_taxonomy import _fingerprint_of

        exclude = set(exclude_fingerprints or [])
        tasks: List[MutationTask] = []
        for idx, cluster in enumerate(clusters or []):
            if not isinstance(cluster, dict) or not cluster.get("cluster_id"):
                raise ValueError(
                    f"failure_clusters[{idx}] is missing cluster_id: probes must be attributable")
            category = str(cluster.get("error_category")
                           or DEFAULT_CLUSTER_ERROR_CATEGORY)
            fam = _PROBE_FAMILIES.get(category, _PROBE_DEFAULT_FAMILY)()
            made = 0
            attempt = 0
            need = int(cluster.get("count") or per_cluster)
            while made < need and attempt < need * 8 + 8:
                attempt += 1
                rng = random.Random(
                    f"{self.seed}:{cluster['cluster_id']}:{fam.name}:{attempt}")
                task = fam.build(rng, cluster)
                if _fingerprint_of(task) in exclude:
                    continue
                tasks.append(task)
                made += 1
            if made < need:
                raise ValueError(
                    f"Probe cluster {cluster['cluster_id']} generated only {made}/{need} tasks")
        return tasks


__all__ = ["WorkbuddyFailureProbeEngine", "PROBE_GENERATOR_REF_BASE"]
