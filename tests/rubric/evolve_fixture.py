"""P3 evolution test shared fixtures: grouped-observation rows, valid decomposition, and
mock endpoints.
"""
from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

RUBRICS = [
    {"rubric_id": "r0", "title": "title r0", "description": "criterion zero", "weight": 0.6},
    {"rubric_id": "r1", "title": "title r1", "description": "criterion one", "weight": 0.4},
]

QUERY = "How should an operator escalate a suspected outage?"


def group_rows(
    *,
    group_id: str = "g1",
    query_id: str = "q1",
    query: str = QUERY,
    version: int = 0,
    rubrics=None,
    scores_matrix=None,
    run_id: str | None = None,
):
    """8-row grouped observation: r0 saturated (all 1), r1 mixed -> non-zero group std."""
    rubrics = rubrics if rubrics is not None else RUBRICS
    if scores_matrix is None:
        scores_matrix = [[1, i % 2] for i in range(8)]
    rows = []
    for i in range(8):
        rows.append(
            {
                "group_id": group_id,
                "query_id": query_id,
                "query": query,
                "rubric_query_version": version,
                "response_id": f"{group_id}-resp-{i}",
                "rollout_index": i,
                "original_index": i,
                "response_text": f"answer text {i} for {query_id}",
                "rubric_ids": [r["rubric_id"] for r in rubrics],
                "rubric_titles": [r["title"] for r in rubrics],
                "rubric_descriptions": [r["description"] for r in rubrics],
                "rubric_weights": [float(r["weight"]) for r in rubrics],
                "rubric_scores": list(scores_matrix[i]),
            }
        )
    if run_id:
        for row in rows:
            row["run_id"] = run_id
    return rows


def decomposition_for(
    *,
    parent: dict,
    children: list[dict],
    response_ids: list[str],
    blend_lambda: float = 0.5,
) -> dict:
    """Build a decomposition in the shape produced by single_pass (submittable to
    registry).
    """
    return {
        "parent_rubric_id": parent["rubric_id"],
        "parent_fingerprint": parent["parent_fingerprint"],
        "parent_title": parent["title"],
        "parent_description": parent["description"],
        "parent_weight": float(parent["weight"]),
        "blend_lambda": blend_lambda,
        "children": children,
    }


def child_for(
    *,
    index: int,
    response_ids: list[str],
    pass_pattern: list[int] | None = None,
    source: str = "online_generation",
) -> dict:
    pattern = pass_pattern if pass_pattern is not None else [1] * 3 + [0] * 5
    point = f"completion point {index}: names the concrete escalation detail {index}."
    return {
        "title": f"child {index}",
        "description": point,
        "completion_point": point,
        "necessity": "optional",
        "semantic_family": point,
        "meta_verification": {
            "decision": "not_performed",
            "validation_scope": "format_and_deduplication",
        },
        "judge_scores": [
            {"response_id": response_id, "score": pattern[position]}
            for position, response_id in enumerate(response_ids)
        ],
        "pass_count": sum(pattern),
        "generator_candidate_index": index,
        "candidate_index": index,
        "candidate_source": source,
        "source_candidate_index": index,
    }


class MockChatHandler(BaseHTTPRequestHandler):
    """OpenAI-compatible /chat/completions mock: behavior is driven by server.mode.
    mode='ok': return deterministic candidate JSON matching the schema's minItems;
    mode='invalid_json': return non-JSON text (triggers the repair path);
    mode='down': HTTP 503.
    """

    def log_message(self, *args):
        pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length) or b"{}")
        self.server.requests.append(body)
        mode = self.server.mode
        if mode == "down":
            self.send_response(503)
            self.end_headers()
            return
        if mode == "invalid_json":
            content = "sorry, I cannot produce JSON today"
        else:
            schema = body["response_format"]["json_schema"]["schema"]
            count = schema["properties"]["candidates"]["minItems"]
            items_property = schema["properties"]["candidates"]["items"]
            required = list(items_property.get("required", ["title", "completion_point"]))
            candidates = []
            for i in range(count):
                candidate = {}
                for field in required:
                    candidate[field] = (
                        f"demo candidate {i}"
                        if field == "title"
                        else f"demo completion point {i}: concrete detail {i}."
                    )
                candidates.append(candidate)
            content = json.dumps({"candidates": candidates})
        payload = json.dumps(
            {
                "choices": [
                    {
                        "message": {"content": content},
                        "finish_reason": "stop",
                    }
                ]
            }
        )
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(payload.encode("utf-8"))


def mock_generator_server(mode: str = "ok"):
    server = ThreadingHTTPServer(("127.0.0.1", 0), MockChatHandler)
    server.mode = mode
    server.requests = []
    return server
