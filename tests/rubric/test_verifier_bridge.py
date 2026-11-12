"""rubric_critic verifier integration tests: registry routing, spec validation, and
credential isolation.
"""

from __future__ import annotations

import json

import pytest

from data_designer.rubric.verifier_bridge import run_rubric_critic
from data_designer.verifiers.dispatch import SUPPORTED_VERIFIER_KINDS, run_verifier
from data_designer.verifiers.registry import IMPLEMENTATIONS

RUBRIC_LIBRARY = [
    {"rubric_key": "r0", "title": "t0", "description": "mentions apples", "weight": 0.5},
    {"rubric_key": "r1", "title": "t1", "description": "structured sections", "weight": 0.5},
]

RESPONSES = [
    {"response_id": f"resp-{i}", "rollout_index": i, "text": f"answer {i}"}
    for i in range(8)
]


def _artifact():
    return {"query": "write about apples", "query_id": "q-1", "responses": RESPONSES}


@pytest.fixture()
def stubbed_gate(monkeypatch):
    """No network: replace gate_group scoring with a deterministic grid."""
    from data_designer.rubric.orchestrate import pipeline

    def fake_gate_group(self, *, query_id, query, responses):
        ordered = sorted(responses, key=lambda row: int(row["rollout_index"]))
        decisions = []
        for index, row in enumerate(ordered):
            scores = [1, 0] if index % 2 == 0 else [1, 1]
            from data_designer.rubric.gate.acceptance import decide

            decision = decide(
                sample_id=str(row.get("response_id")),
                query_id=query_id,
                rubric_ids=("r0", "r1"),
                rubric_scores=scores,
                rubric_weights=(0.5, 0.5),
                group_pass_counts=[8, 4],
                threshold=self.threshold,
                rubric_library_sha256=self.library_sha256,
                judge_route={"model": "stub-judge"},
            )
            decisions.append(decision.as_dict())
        return decisions

    monkeypatch.setattr(pipeline.RubricGate, "gate_group", fake_gate_group)


def test_registry_contains_rubric_critic():
    assert "rubric_critic" in IMPLEMENTATIONS
    assert "rubric_critic" in SUPPORTED_VERIFIER_KINDS


def test_run_rubric_critic_result_shape(stubbed_gate):
    result = run_rubric_critic(
        _artifact(),
        {"rubric_library": RUBRIC_LIBRARY, "group_size": 8, "threshold": 0.6},
    )
    assert result["status"] == "completed"
    assert result["passed"] is False  # Half the samples fail r1; scores mix 0.75 and
                                      # 1.0.
    assert 0.0 <= result["score"] <= 1.0
    detail = json.loads(result["detail"])
    assert detail["group_size"] == 8
    assert detail["rubric_library_sha256"].startswith("sha256:")
    assert len(detail["decisions"]) == 8


def test_run_verifier_dispatch_routing(stubbed_gate):
    result = run_verifier(
        {
            "kind": "rubric_critic",
            "payload": {"rubric_library": RUBRIC_LIBRARY, "group_size": 8},
            "generator_ref": "gen-1",
        },
        _artifact(),
    )
    assert result["status"] == "completed"
    assert result["verifier"] == "rubric_critic"


def test_run_rubric_critic_rejects_incomplete_group(stubbed_gate):
    with pytest.raises(Exception):
        run_rubric_critic(
            {**_artifact(), "responses": RESPONSES[:4]},
            {"rubric_library": RUBRIC_LIBRARY, "group_size": 8},
        )


def test_run_rubric_critic_rejects_bad_payload(stubbed_gate):
    with pytest.raises(Exception):
        run_rubric_critic(_artifact(), {"group_size": 8})
    with pytest.raises(Exception):
        run_rubric_critic("not a mapping", {"rubric_library": RUBRIC_LIBRARY})


def test_credential_isolation(stubbed_gate):
    """Credential discipline: credential-like fields in the payload never enter result
    detail.
    """
    payload = {
        "rubric_library": RUBRIC_LIBRARY,
        "group_size": 8,
        "api_key": "sk-secret",
    }
    result = run_rubric_critic(_artifact(), payload)
    assert "sk-secret" not in result["detail"]
