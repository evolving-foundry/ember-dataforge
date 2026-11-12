"""The GAIA adapter calls a SHA-pinned scorer and keeps answers private."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from data_designer.pipelines.benchmarks.runners.gaia import VERSION, score_request


def _hash(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def test_gaia_pinned_scorer_positive_negative_and_drift(tmp_path: Path):
    scorer = tmp_path / "scorer.py"
    scorer.write_text(
        "def question_scorer(model_answer, ground_truth):\n"
        "    return model_answer.strip().lower() == ground_truth.strip().lower()\n",
        encoding="utf-8",
    )
    expected = "42"
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "task_id": "gaia:owned:source:v1",
                "private": {
                    "expected_answer": expected,
                    "expected_answer_sha256": _hash(expected.encode()),
                    "gaia_scorer": {
                        "path": str(scorer),
                        "sha256": _hash(scorer.read_bytes()),
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    root = tmp_path / "candidate"
    root.mkdir()
    request = {
        "caseManifestPath": str(manifest),
        "candidateRoot": str(root),
        "taskId": "gaia:owned:source:v1",
        "attempt": {"index": 0, "seed": 37},
    }
    (root / "answer.txt").write_text("42", encoding="utf-8")
    positive = score_request(request)
    assert positive["passed"] is True
    assert positive["verifier_version"] == VERSION
    assert (positive["attempt_index"], positive["seed"]) == (0, 37)
    (root / "answer.txt").write_text("43", encoding="utf-8")
    assert score_request(request)["passed"] is False
    scorer.write_text("def question_scorer(*args): return True\n", encoding="utf-8")
    with pytest.raises(ValueError, match="drifted"):
        score_request(request)


def test_gaia_answer_cannot_be_changed_without_manifest_hash(tmp_path: Path):
    scorer = tmp_path / "scorer.py"
    scorer.write_text("def question_scorer(a, b): return a == b\n", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "task_id": "case",
                "private": {
                    "expected_answer": "secret",
                    "expected_answer_sha256": "0" * 64,
                    "gaia_scorer": {
                        "path": str(scorer),
                        "sha256": _hash(scorer.read_bytes()),
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    (candidate / "answer.txt").write_text("secret", encoding="utf-8")
    with pytest.raises(ValueError, match="answer SHA-256 drifted"):
        score_request(
            {
                "caseManifestPath": str(manifest),
                "candidateRoot": str(candidate),
                "taskId": "case",
            }
        )
