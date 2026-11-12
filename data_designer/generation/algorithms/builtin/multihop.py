"""multihop_dag_fusion recipe: deterministic multi-hop DAG fusion.

Ported from DeepFusion (data_generation/step*_iter*_Fusion/). Fuses verified
multi-hop DAGs into longer-hop DAGs via two strategies:

- ``bridge`` (safe_bridge_union): two DAGs sharing a non-terminal entity
  (matched by ``entity_meta.normalized_name``) are composed on that pivot;
  the pivot becomes a ``Shared_Pivot`` intersection node.
- ``chain`` (strict_answer_to_start): DAG B's Start entity equals DAG A's
  terminal Answer entity; B is composed onto A with the Answer node
  becoming a ``Chain_Pivot`` link.

Pure networkx, no LLM/network dependency. Output rows carry ``fusion_meta``
provenance and an incremented ``hop_count`` (DeepFusion's N+1 convention:
fused hop = max(parent hops, default 2) + 1).
"""

from __future__ import annotations

import copy
import random
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx

from data_designer.contracts.algorithms import AlgorithmResult
from data_designer.contracts.datasets import RecordEnvelope
from .descriptors import DAG_FUSION

MULTIHOP_FUSION_SCHEMA_VERSION = "multihop-fusion-v1"
DEFAULT_PARENT_HOP_COUNT = 2

FUSION_STRATEGIES = ("bridge", "chain", "both")

BRIDGE_FUSION_TYPE = "safe_bridge_union"
CHAIN_FUSION_TYPE = "strict_answer_to_start"

BRIDGE_PIVOT_TYPE = "Shared_Pivot"
CHAIN_PIVOT_TYPE = "Chain_Pivot"

_OUTPUT_KEYS = ("graph_id", "dag", "fusion_meta", "hop_count", "source_graph_ids")


def _normalized_name(node: Dict[str, Any]) -> Optional[str]:
    meta = node.get("entity_meta")
    if not isinstance(meta, dict):
        return None
    name = meta.get("normalized_name")
    return str(name) if name else None


def _valid_dag_item(item: Any) -> bool:
    if not isinstance(item, dict):
        return False
    nodes = (item.get("dag") or {}).get("nodes")
    if not isinstance(nodes, list):
        return False
    return all(isinstance(node, dict) and "id" in node for node in nodes)


def _dag_to_nx(dag_dict: Optional[Dict[str, Any]]) -> nx.DiGraph:
    graph = nx.DiGraph()
    for node in (dag_dict or {}).get("nodes", []):
        if not isinstance(node, dict):
            continue
        nid = node.get("id")
        if nid is None:
            continue
        graph.add_node(nid, **copy.deepcopy(node))
    for edge in (dag_dict or {}).get("edges", []):
        if not isinstance(edge, dict):
            continue
        source = edge.get("source")
        target = edge.get("target")
        if source is None or target is None:
            continue
        graph.add_edge(source, target, relation=edge.get("relation", "connected_to"))
    return graph


def _nx_to_dag(graph: nx.DiGraph, new_gid: str, meta: Dict[str, Any]) -> Dict[str, Any]:
    nodes = []
    for nid, attr in graph.nodes(data=True):
        node_data = dict(attr)
        node_data["id"] = nid
        nodes.append(node_data)
    edges = [
        {"source": u, "target": v, "relation": attr.get("relation", "connected_to")}
        for u, v, attr in graph.edges(data=True)
    ]
    return {"graph_id": new_gid, "dag": {"nodes": nodes, "edges": edges}, "fusion_meta": meta}


def _child_hop_count(item_a: Dict[str, Any], item_b: Dict[str, Any]) -> int:
    hop_a = item_a.get("hop_count") or DEFAULT_PARENT_HOP_COUNT
    hop_b = item_b.get("hop_count") or DEFAULT_PARENT_HOP_COUNT
    return max(int(hop_a), int(hop_b)) + 1


class DagFusionAlgorithm:
    """Whole-dataset DeepFusion kernel; no Curator/Ray or operator dependency."""

    def describe(self):
        return DAG_FUSION

    def run(self, request, context):
        config = request.config
        if len(request.records) > config["max_input_records"]:
            raise ValueError("DAG fusion input limit exceeded")
        pool = []
        graph_ids = {}
        rejected = 0
        allowed = set(config["gold_graph_ids"])
        for record in request.records:
            row = copy.deepcopy(record.payload)
            gid = row.get("graph_id")
            if not gid or not _valid_dag_item(row):
                rejected += 1
                continue
            if allowed and str(gid) not in allowed:
                continue
            graph = _dag_to_nx(row["dag"])
            if not nx.is_directed_acyclic_graph(graph):
                rejected += 1
                continue
            if str(gid) in graph_ids:
                raise ValueError("duplicate graph_id in fusion input")
            graph_ids[str(gid)] = record.id
            pool.append(row)
        self._context = context
        self._remaining_attempts = request.max_attempts
        rng = context.random
        fused = []
        strategy = config["strategy"]
        if strategy in ("bridge", "both"):
            fused.extend(self._fuse_bridge(pool, rng, config["max_pair_occurrences"]))
        if strategy in ("chain", "both"):
            fused.extend(self._fuse_chain(pool, rng, config["sample_b_per_answer"],
                                          config["require_answer_terminal"]))
        records = []
        for row in fused:
            parents = tuple(graph_ids[str(gid)] for gid in row["source_graph_ids"])
            records.append(RecordEnvelope(str(row["graph_id"]), row,
                                          DAG_FUSION.output_schema, parents))
        return AlgorithmResult(records, {"invalid_dag": rejected} if rejected else {},
                               {"pair_attempts": request.max_attempts - self._remaining_attempts,
                                "full_input_memory": True})

    def _attempt(self):
        self._context.check_cancelled()
        if self._remaining_attempts <= 0:
            return False
        self._remaining_attempts -= 1
        return True

    # ------------------------------------------------------------------
    # bridge fusion (safe_bridge_union)
    # ------------------------------------------------------------------

    def _fuse_bridge(
        self,
        pool: List[Dict[str, Any]],
        rng: random.Random,
        max_pair_occurrences: int,
    ) -> List[Dict[str, Any]]:
        entity_index: Dict[str, List[Dict[str, Any]]] = {}
        for item in pool:
            for node in item["dag"]["nodes"]:
                name = _normalized_name(node)
                if not name:
                    continue
                entity_index.setdefault(name, []).append(
                    {"gid": item.get("graph_id"), "nid": node["id"], "type": node.get("type"), "item": item}
                )

        fused: List[Dict[str, Any]] = []
        seen_pairs = set()
        for pivot_name, occurrences in entity_index.items():
            if len(occurrences) < 2:
                continue
            current = (
                rng.sample(occurrences, max_pair_occurrences)
                if len(occurrences) > max_pair_occurrences
                else occurrences
            )
            for i in range(len(current)):
                for j in range(len(current)):
                    if i == j:
                        continue
                    occ_a, occ_b = current[i], current[j]
                    gid_a, gid_b = occ_a["gid"], occ_b["gid"]
                    if gid_a == gid_b:
                        continue
                    pair_signature = tuple(sorted((str(gid_a), str(gid_b))))
                    if pair_signature in seen_pairs:
                        continue
                    is_chain_ab = occ_a["type"] == "Answer" and occ_b["type"] == "Start"
                    is_chain_ba = occ_a["type"] == "Start" and occ_b["type"] == "Answer"
                    if is_chain_ab or is_chain_ba:
                        continue
                    if not self._attempt():
                        return fused
                    result = self._fuse_bridge_pair(occ_a, occ_b, pivot_name)
                    if result:
                        fused.append(result)
                        seen_pairs.add(pair_signature)
        return fused

    def _fuse_bridge_pair(
        self,
        occ_a: Dict[str, Any],
        occ_b: Dict[str, Any],
        pivot_name: str,
    ) -> Optional[Dict[str, Any]]:
        graph_a = _dag_to_nx(occ_a["item"]["dag"])
        graph_b = _dag_to_nx(occ_b["item"]["dag"])
        pivot_id_a = occ_a["nid"]
        pivot_id_b = occ_b["nid"]
        suffix = f"_from_{occ_b['gid']}"
        mapping = {
            node: (pivot_id_a if node == pivot_id_b else f"{node}{suffix}")
            for node in graph_b.nodes()
        }
        try:
            graph_b_relabeled = nx.relabel_nodes(graph_b, mapping)
            fused_graph = nx.compose(graph_a, graph_b_relabeled)
            if pivot_id_a in fused_graph:
                fused_graph.nodes[pivot_id_a]["type"] = BRIDGE_PIVOT_TYPE
                fused_graph.nodes[pivot_id_a]["description"] = (
                    "Intersection point of two distinct knowledge paths"
                )
                meta_dict = fused_graph.nodes[pivot_id_a].get("entity_meta")
                if isinstance(meta_dict, dict):
                    meta_dict["role"] = "intersection"
                    fused_graph.nodes[pivot_id_a]["entity_meta"] = meta_dict
            if not nx.is_directed_acyclic_graph(fused_graph):
                return None
            new_gid = f"FUSED_{occ_a['gid']}_{occ_b['gid']}"
            meta = {
                "pivot": pivot_name,
                "source_a": occ_a["gid"],
                "source_b": occ_b["gid"],
                "type": BRIDGE_FUSION_TYPE,
                "schema_version": MULTIHOP_FUSION_SCHEMA_VERSION,
            }
            result = _nx_to_dag(fused_graph, new_gid, meta)
            result["hop_count"] = _child_hop_count(occ_a["item"], occ_b["item"])
            result["source_graph_ids"] = [occ_a["gid"], occ_b["gid"]]
            result["sample_id"] = new_gid
            return result
        except nx.NetworkXException:
            return None

    # ------------------------------------------------------------------
    # chain fusion (strict_answer_to_start)
    # ------------------------------------------------------------------

    def _fuse_chain(
        self,
        pool: List[Dict[str, Any]],
        rng: random.Random,
        sample_b_per_answer: int,
        require_answer_terminal: bool,
    ) -> List[Dict[str, Any]]:
        start_index: Dict[str, List[Dict[str, Any]]] = {}
        for item in pool:
            for node in item["dag"]["nodes"]:
                if node.get("type") != "Start":
                    continue
                name = _normalized_name(node)
                if not name:
                    continue
                start_index.setdefault(name, []).append(
                    {"gid": item.get("graph_id"), "start_nid": node["id"], "item": item}
                )

        fused: List[Dict[str, Any]] = []
        seen_pairs = set()
        for item_a in pool:
            gid_a = item_a.get("graph_id")
            pivot_name, answer_id_a = self._pick_answer_pivot(item_a)
            if not pivot_name or not answer_id_a:
                continue
            candidates_b = start_index.get(pivot_name)
            if not candidates_b:
                continue
            current_bs = (
                rng.sample(candidates_b, sample_b_per_answer)
                if len(candidates_b) > sample_b_per_answer
                else candidates_b
            )
            for occ_b in current_bs:
                gid_b = occ_b["gid"]
                if gid_a == gid_b:
                    continue
                signature = (str(gid_a), str(gid_b), pivot_name)
                if signature in seen_pairs:
                    continue
                if not self._attempt():
                    return fused
                result = self._fuse_chain_pair(
                    item_a, occ_b, pivot_name, answer_id_a, require_answer_terminal
                )
                if result:
                    fused.append(result)
                    seen_pairs.add(signature)
        return fused

    @staticmethod
    def _pick_answer_pivot(item: Dict[str, Any]) -> Tuple[Optional[str], Optional[Any]]:
        for node in (item.get("dag") or {}).get("nodes", []):
            if isinstance(node, dict) and node.get("type") == "Answer":
                name = _normalized_name(node)
                if name:
                    return name, node.get("id")
        return None, None

    def _fuse_chain_pair(
        self,
        item_a: Dict[str, Any],
        occ_b: Dict[str, Any],
        pivot_name: str,
        answer_id_a: Any,
        require_answer_terminal: bool,
    ) -> Optional[Dict[str, Any]]:
        gid_a = item_a.get("graph_id")
        gid_b = occ_b["gid"]
        graph_a = _dag_to_nx(item_a.get("dag"))
        graph_b = _dag_to_nx(occ_b["item"].get("dag"))
        if answer_id_a not in graph_a:
            return None
        if require_answer_terminal and graph_a.out_degree(answer_id_a) != 0:
            return None
        start_id_b = occ_b.get("start_nid")
        if start_id_b is None or start_id_b not in graph_b:
            return None
        suffix = f"_from_{gid_b}"
        mapping = {
            node: (answer_id_a if node == start_id_b else f"{node}{suffix}")
            for node in graph_b.nodes()
        }
        try:
            graph_b_relabeled = nx.relabel_nodes(graph_b, mapping)
            fused_graph = nx.compose(graph_b_relabeled, graph_a)
            if answer_id_a in fused_graph:
                fused_graph.nodes[answer_id_a]["type"] = CHAIN_PIVOT_TYPE
                fused_graph.nodes[answer_id_a]["description"] = (
                    "Merged intermediate pivot node connecting two DAGs"
                )
                meta_dict = fused_graph.nodes[answer_id_a].get("entity_meta")
                if isinstance(meta_dict, dict):
                    meta_dict["role"] = "chain_link"
                    fused_graph.nodes[answer_id_a]["entity_meta"] = meta_dict
            if not nx.is_directed_acyclic_graph(fused_graph):
                return None
            new_gid = f"CHAIN_{gid_a}_{gid_b}"
            meta = {
                "pivot": pivot_name,
                "source_head": gid_a,
                "source_tail": gid_b,
                "type": CHAIN_FUSION_TYPE,
                "schema_version": MULTIHOP_FUSION_SCHEMA_VERSION,
            }
            result = _nx_to_dag(fused_graph, new_gid, meta)
            result["hop_count"] = _child_hop_count(item_a, occ_b["item"])
            result["source_graph_ids"] = [gid_a, gid_b]
            result["sample_id"] = new_gid
            return result
        except nx.NetworkXException:
            return None
