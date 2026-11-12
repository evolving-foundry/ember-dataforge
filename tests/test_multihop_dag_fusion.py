from __future__ import annotations

import json

import pytest

from data_designer.api import generate


def _node(nid, ntype, name=None):
    node = {"id": nid, "type": ntype, "description": f"desc {nid}", "content": f"content {nid}"}
    if name:
        node["entity_meta"] = {
            "normalized_name": name,
            "category": "entity",
            "role": "primary",
        }
    return node


def _chain_dag(graph_id, start_name, mid_name, answer_name, hop_count=None, answer_out_edge=False):
    nodes = [
        _node(f"{graph_id}_n1", "Start", start_name),
        _node(f"{graph_id}_n2", "Retrieval", mid_name),
        _node(f"{graph_id}_n3", "Answer", answer_name),
    ]
    edges = [
        {"source": f"{graph_id}_n1", "target": f"{graph_id}_n2", "relation": "leads_to"},
        {"source": f"{graph_id}_n2", "target": f"{graph_id}_n3", "relation": "resolves_to"},
    ]
    if answer_out_edge:
        nodes.append(_node(f"{graph_id}_n4", "Reasoning", "post-answer reasoning"))
        edges.append(
            {"source": f"{graph_id}_n3", "target": f"{graph_id}_n4", "relation": "extends"}
        )
    row = {"graph_id": graph_id, "dag": {"nodes": nodes, "edges": edges}}
    if hop_count is not None:
        row["hop_count"] = hop_count
    return row


def _bridge_pool():
    dag_a = _chain_dag("dag_a", "Albert Einstein", "Nobel Prize in Physics", "1921")
    dag_b = _chain_dag("dag_b", "Nobel Prize in Physics", "Nobel Committee", "Stockholm")
    return [dag_a, dag_b]


def _run(rows, recipe_config=None):
    params = dict(recipe_config or {})
    seed = params.pop("seed", 42)
    result = generate(rows, {"recipe": "multihop_fusion", "input": {"schema": "multihop-dag-v1"},
                             "execution": {"seed": seed},
                             "recipe_config": {"algorithms": {"fusion": {"params": params}}}})
    return [{**record.payload, "provenance": {"parents": record.parents}} for record in result.records]


def _by_graph_id(rows):
    return {row["graph_id"]: row for row in rows}


def test_bridge_fusion_composes_shared_pivot():
    out = _run(_bridge_pool(), {"strategy": "bridge"})

    fused = _by_graph_id(out)
    assert set(fused) == {"FUSED_dag_a_dag_b"}
    row = fused["FUSED_dag_a_dag_b"]

    meta = row["fusion_meta"]
    assert meta["type"] == "safe_bridge_union"
    assert meta["pivot"] == "Nobel Prize in Physics"
    assert meta["source_a"] == "dag_a"
    assert meta["source_b"] == "dag_b"

    dag = row["dag"]
    node_ids = {node["id"] for node in dag["nodes"]}
    assert node_ids == {
        "dag_a_n1",
        "dag_a_n2",
        "dag_a_n3",
        "dag_b_n2_from_dag_b",
        "dag_b_n3_from_dag_b",
    }
    assert len(dag["edges"]) == 4
    assert {("dag_a_n2", "dag_b_n2_from_dag_b")} <= {
        (e["source"], e["target"]) for e in dag["edges"]
    }

    pivot = next(node for node in dag["nodes"] if node["id"] == "dag_a_n2")
    assert pivot["type"] == "Shared_Pivot"
    assert pivot["entity_meta"]["role"] == "intersection"

    assert row["hop_count"] == 3
    assert row["source_graph_ids"] == ["dag_a", "dag_b"]


def test_chain_fusion_links_terminal_answer_to_start():
    dag_a = _chain_dag("dag_a", "Albert Einstein", "Nobel Prize in Physics", "1921")
    dag_c = _chain_dag("dag_c", "1921", "Nobel Committee 1921", "Peace Prize")
    out = _run([dag_a, dag_c], {"strategy": "chain"})

    fused = _by_graph_id(out)
    assert set(fused) == {"CHAIN_dag_a_dag_c"}
    row = fused["CHAIN_dag_a_dag_c"]

    meta = row["fusion_meta"]
    assert meta["type"] == "strict_answer_to_start"
    assert meta["pivot"] == "1921"
    assert meta["source_head"] == "dag_a"
    assert meta["source_tail"] == "dag_c"

    dag = row["dag"]
    node_ids = {node["id"] for node in dag["nodes"]}
    assert node_ids == {
        "dag_a_n1",
        "dag_a_n2",
        "dag_a_n3",
        "dag_c_n2_from_dag_c",
        "dag_c_n3_from_dag_c",
    }
    pivot = next(node for node in dag["nodes"] if node["id"] == "dag_a_n3")
    assert pivot["type"] == "Chain_Pivot"
    assert pivot["entity_meta"]["role"] == "chain_link"
    assert row["hop_count"] == 3


def test_both_strategy_produces_bridge_and_chain_outputs():
    dag_a = _chain_dag("dag_a", "Albert Einstein", "Nobel Prize in Physics", "1921")
    dag_b = _chain_dag("dag_b", "Nobel Prize in Physics", "Nobel Committee", "Stockholm")
    dag_c = _chain_dag("dag_c", "1921", "Nobel Committee 1921", "Peace Prize")
    out = _run([dag_a, dag_b, dag_c], {"strategy": "both"})

    assert set(_by_graph_id(out)) == {"FUSED_dag_a_dag_b", "CHAIN_dag_a_dag_c"}


def test_bridge_skips_answer_start_pivot_pairs():
    dag_a = _chain_dag("dag_a", "Albert Einstein", "Prize Nomination", "Nobel Prize in Physics")
    dag_b = _chain_dag("dag_b", "Nobel Prize in Physics", "Nobel Committee", "Stockholm")
    out = _run([dag_a, dag_b], {"strategy": "bridge"})

    assert all("fusion_meta" not in row for row in out)


def test_chain_requires_terminal_answer():
    dag_a = _chain_dag("dag_a", "Albert Einstein", "Nobel Prize in Physics", "1921", answer_out_edge=True)
    dag_c = _chain_dag("dag_c", "1921", "Nobel Committee 1921", "Peace Prize")
    out = _run([dag_a, dag_c], {"strategy": "chain"})

    assert all("fusion_meta" not in row for row in out)


def test_gold_graph_ids_filters_pool():
    dag_a = _chain_dag("dag_a", "Albert Einstein", "Shared Entity", "1921")
    dag_b = _chain_dag("dag_b", "Shared Entity", "Nobel Committee", "Stockholm")
    dag_d = _chain_dag("dag_d", "Shared Entity", "Other Committee", "Oslo")
    out = _run(
        [dag_a, dag_b, dag_d],
        {"strategy": "bridge", "gold_graph_ids": ["dag_a", "dag_b"]},
    )

    assert set(_by_graph_id(out)) == {"FUSED_dag_a_dag_b"}


def test_invalid_rows_are_skipped():
    dag_a = _chain_dag("dag_a", "Albert Einstein", "Nobel Prize in Physics", "1921")
    dag_b = _chain_dag("dag_b", "Nobel Prize in Physics", "Nobel Committee", "Stockholm")
    bad = {"graph_id": "dag_bad", "dag": {"nodes": [{"type": "Start"}], "edges": []}}
    out = _run([dag_a, bad, dag_b], {"strategy": "bridge"})

    assert set(_by_graph_id(out)) == {"FUSED_dag_a_dag_b"}


def test_hop_count_increments_from_parents():
    dag_a = _chain_dag("dag_a", "Albert Einstein", "Nobel Prize in Physics", "1921", hop_count=3)
    dag_b = _chain_dag("dag_b", "Nobel Prize in Physics", "Nobel Committee", "Stockholm", hop_count=4)
    out = _run([dag_a, dag_b], {"strategy": "bridge"})

    row = _by_graph_id(out)["FUSED_dag_a_dag_b"]
    assert row["hop_count"] == 5


def test_invalid_strategy_raises():
    with pytest.raises(ValueError, match="strategy"):
        _run(_bridge_pool(), {"strategy": "unsupported"})


def test_fusion_is_deterministic_for_same_seed():
    def canonical(rows):
        return sorted(json.dumps(row, sort_keys=True, default=str) for row in rows)

    first = canonical(_run(_bridge_pool(), {"strategy": "bridge", "seed": 42}))
    second = canonical(_run(_bridge_pool(), {"strategy": "bridge", "seed": 42}))

    assert first == second


def test_fused_rows_carry_provenance():
    out = _run(_bridge_pool(), {"strategy": "bridge"})

    row = _by_graph_id(out)["FUSED_dag_a_dag_b"]
    assert row["provenance"]["parents"] == ("dag_a", "dag_b")
