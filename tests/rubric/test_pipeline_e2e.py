"""Scenario A gate closed-loop e2e drill (P2 subset of P2 acceptance criteria 2/5):
under a mock Judge endpoint: group -> 8xN grid scoring -> aggregate score + per-item
evidence -> section 6.1 decision JSON atomically persisted; circuit-break drill:
Judge endpoint interruption -> explicit failure, no all-zero fallback, no partial
persistence.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from data_designer.rubric.judge.transport import (
    configure_judge_endpoints,
    reset_judge_endpoints,
)
from data_designer.rubric.orchestrate.pipeline import RubricGate, persist_decisions

RUBRIC_LIBRARY = [
    {"rubric_key": "r0", "title": "t0", "description": "criterion A", "weight": 0.5},
    {"rubric_key": "r1", "title": "t1", "description": "criterion B", "weight": 0.25},
    {"rubric_key": "r2", "title": "t2", "description": "criterion C", "weight": 0.25},
]


class _JudgeHandler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length) or b"{}")
        server = self.server
        server.requests.append(body)
        if server.mode == "down":
            self.send_response(503)
            self.end_headers()
            return
        # Deterministic grid: criterion indices come from the response_format schema;
        # labels alternate in order
        properties = body["response_format"]["json_schema"]["schema"]["properties"]
        rubric_count = len(properties)
        labels = ["PRESENT" if i % 2 == 0 else "NOT_PRESENT" for i in range(rubric_count)]
        payload = json.dumps({str(i + 1): labels[i] for i in range(rubric_count)})
        self.send_response(200)
        self.end_headers()
        self.wfile.write(
            json.dumps(
                {"choices": [{"finish_reason": "stop", "message": {"content": payload}}]}
            ).encode()
        )


@pytest.fixture()
def judge_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _JudgeHandler)
    server.requests = []
    server.mode = "up"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.shutdown()
    server.server_close()
    reset_judge_endpoints()


def _responses(group_size=8):
    return [
        {
            "response_id": f"resp-{i}",
            "rollout_index": i,
            "text": f"sample answer {i}",
        }
        for i in range(group_size)
    ]


def test_gate_pipeline_end_to_end(judge_server, tmp_path, monkeypatch):
    monkeypatch.setenv("HEALTHBENCH_JUDGE_RETRY_BACKOFF_MAX_SECONDS", "0.1")
    transport = configure_judge_endpoints(
        api_base=f"http://127.0.0.1:{judge_server.server_port}",
        api_key="e2e-key",
        model="e2e-judge",
    )
    gate = RubricGate(
        rubric_library=RUBRIC_LIBRARY,
        threshold=0.6,
        transport=transport,
        judge_workers=4,
    )
    assert gate.filter_bounds() == (2, 6)
    decisions = gate.gate_group(
        query_id="q-1", query="produce a report", responses=_responses()
    )
    assert len(decisions) == 8
    for record in decisions:
        assert record["decision"] in {"accept", "reject", "holdout"}
        assert record["difficulty_bucket"] in {"mid", "saturated", "unreachable"}
        evidence = record["rubric_evidence"]
        assert len(evidence["rubric_scores"]) == 3
        assert len(evidence["group_pass_counts"]) == 3
        assert record["audit"]["rubric_library_sha256"].startswith("sha256:")
        assert record["audit"]["judge_route"]["model"] == "e2e-judge"
    written = persist_decisions(decisions, tmp_path / "decisions")
    assert len(written) == 8
    stored = json.loads(written[0].read_text(encoding="utf-8"))
    assert stored["sample_id"] == decisions[0]["sample_id"]
    # Decision file is not overwritten: repeated persistence fails explicitly
    with pytest.raises(FileExistsError):
        persist_decisions(decisions, tmp_path / "decisions")


def test_gate_pipeline_fails_closed_when_judge_down(judge_server, tmp_path, monkeypatch):
    monkeypatch.setenv("HEALTHBENCH_JUDGE_RETRY_BACKOFF_MAX_SECONDS", "0.1")
    monkeypatch.setenv("HEALTHBENCH_JUDGE_MAX_ATTEMPTS", "2")
    transport = configure_judge_endpoints(
        api_base=f"http://127.0.0.1:{judge_server.server_port}",
        api_key="e2e-key",
        model="e2e-judge",
    )
    gate = RubricGate(
        rubric_library=RUBRIC_LIBRARY,
        threshold=0.6,
        transport=transport,
        judge_workers=4,
    )
    judge_server.mode = "down"
    from data_designer.rubric.judge.prompt import JudgeGiveUpError

    with pytest.raises(JudgeGiveUpError):
        gate.gate_group(query_id="q-1", query="produce a report", responses=_responses())
    # Explicit failure: no decision persisted
    assert not (tmp_path / "decisions").exists() or not list(
        (tmp_path / "decisions").iterdir()
    )


def test_gate_requires_complete_group(judge_server):
    transport = configure_judge_endpoints(
        api_base=f"http://127.0.0.1:{judge_server.server_port}",
        api_key="e2e-key",
        model="e2e-judge",
    )
    gate = RubricGate(rubric_library=RUBRIC_LIBRARY, threshold=0.6, transport=transport)
    with pytest.raises(ValueError):
        gate.gate_group(
            query_id="q-1", query="q", responses=_responses()[:4]
        )
    with pytest.raises(ValueError):
        RubricGate(rubric_library=RUBRIC_LIBRARY, threshold=0.6, anchor_scores=[0.5])
