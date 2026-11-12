"""P5 semantic review layer: verdict validation, weighted sampling strategy, apply
write-back, and version write-back chain.
"""
from __future__ import annotations

import random

import pytest

from data_designer.rubric.generator.client import WorkerConfig
from data_designer.rubric.protocol.schemas import PositiveRefinementProtocolError
from data_designer.rubric.review import meta_review

LIBRARY = [
    {"rubric_key": "r0", "title": "t0", "description": "d0", "weight": 0.5},
    {"rubric_key": "r1", "title": "t1", "description": "d1", "weight": 0.5},
]


def _fake_generate(monkeypatch, value, *, capture=None):
    def fake_generate_json(prompt, schema, *, config, deadline=None):
        if capture is not None:
            capture.append((prompt, schema, config))
        return value, {"model": config.model, "attempts": 1}

    monkeypatch.setattr(meta_review, "generate_json", fake_generate_json)


def test_review_criteria_returns_verdicts_with_meta_audit(monkeypatch):
    capture = []
    _fake_generate(
        monkeypatch,
        {"action": "approve", "reason": "sound criterion"},
        capture=capture,
    )
    config = WorkerConfig(
        base_url="http://gen", model="main-model", meta_model="meta-model"
    )
    verdicts = meta_review.review_criteria(LIBRARY, config=config)
    assert [verdict["rubric_key"] for verdict in verdicts] == ["r0", "r1"]
    assert all(verdict["action"] == "approve" for verdict in verdicts)
    # meta-judge decoupling: the actual call uses for_meta()'s meta_model, which is
    # recorded in the audit
    assert all(call[2].model == "meta-model" for call in capture)
    assert all(verdict["review_audit"]["meta_model"] == "meta-model" for verdict in verdicts)
    # Each prompt contains only the text of the criterion under review, with no
    # generated samples/rollout statistics
    for (prompt, _, _), rubric in zip(capture, LIBRARY):
        assert rubric["title"] in prompt and rubric["description"] in prompt
        lowered = prompt.lower()
        assert "rollout" not in lowered and "sample_id" not in lowered


def test_meta_review_prompt_includes_query_context_only_when_present():
    prompt = meta_review.meta_review_prompt(
        {"rubric_key": "r0", "title": "t0", "description": "d0", "query": "Q"}
    )
    assert "Target query (context only)" in prompt and "Q" in prompt
    bare = meta_review.meta_review_prompt(
        {"rubric_key": "r0", "title": "t0", "description": "d0"}
    )
    assert "Target query" not in bare


def test_review_criteria_rejects_bad_verdicts(monkeypatch):
    config = WorkerConfig(base_url="http://gen", model="m")

    def fake(value):
        return lambda prompt, schema, *, config, deadline=None: (value, {"model": config.model})

    # Unknown action fails explicitly
    _fake_generate(monkeypatch, {"action": "maybe", "reason": "r"})
    with pytest.raises(PositiveRefinementProtocolError):
        meta_review.review_criteria(LIBRARY, config=config)
    # Empty reason fails explicitly
    _fake_generate(monkeypatch, {"action": "approve", "reason": "   "})
    with pytest.raises(PositiveRefinementProtocolError):
        meta_review.review_criteria(LIBRARY, config=config)
    # rewrite missing replacement text fails explicitly
    _fake_generate(monkeypatch, {"action": "rewrite", "reason": "vague"})
    with pytest.raises(PositiveRefinementProtocolError):
        meta_review.review_criteria(LIBRARY, config=config)
    # Candidate missing rubric_key fails explicitly
    _fake_generate(monkeypatch, {"action": "approve", "reason": "ok"})
    with pytest.raises(PositiveRefinementProtocolError):
        meta_review.review_criteria(
            [{"title": "x", "description": "y", "weight": 1.0}], config=config
        )


def test_sample_for_review_rate_force_keys_and_order():
    candidates = [{"rubric_key": f"k{i:02d}", "weight": 1.0} for i in range(20)]
    sampled = meta_review.sample_for_review(candidates, rate=0.2, rng=random.Random(7))
    assert len(sampled) == 4  # ceil(20 * 0.2)
    keys = [c["rubric_key"] for c in sampled]
    assert keys == sorted(keys)  # Preserve original library order (stable audit)

    forced = meta_review.sample_for_review(
        candidates, rate=0.1, force_keys=["k19"], rng=random.Random(7)
    )
    forced_keys = {c["rubric_key"] for c in forced}
    assert "k19" in forced_keys
    assert len(forced) == 2  # force items count toward the quota: ceil(20*0.1)=2

    assert meta_review.sample_for_review(candidates[:1], rate=0.05)  # At least 1 for a non-empty library
    assert meta_review.sample_for_review([], rate=0.5) == []
    with pytest.raises(PositiveRefinementProtocolError):
        meta_review.sample_for_review(candidates, rate=0.0)
    with pytest.raises(PositiveRefinementProtocolError):
        meta_review.sample_for_review([{"weight": 1.0}], rate=0.5)


def test_apply_reviews_rewrite_retire_version_chain():
    library = [
        {"rubric_key": "r0", "title": "t0", "description": "d0", "weight": 0.5},
        {"rubric_key": "r1", "title": "t1", "description": "d1", "weight": 0.5},
        {"rubric_key": "r2", "title": "t2", "description": "d2", "weight": 1.0},
    ]
    verdicts = [
        {"rubric_key": "r0", "action": "approve", "reason": "ok"},
        {
            "rubric_key": "r1",
            "action": "rewrite",
            "reason": "vague wording",
            "rewritten_title": "t1'",
            "rewritten_description": "d1'",
        },
        {"rubric_key": "r2", "action": "retire", "reason": "unjudgeable"},
    ]
    new_library, audit = meta_review.apply_reviews(library, verdicts)
    assert [rubric["rubric_key"] for rubric in new_library] == ["r0", "r1"]
    assert new_library[0]["semantic_review"] == {"action": "approve", "reason": "ok"}
    assert new_library[1]["title"] == "t1'"
    assert new_library[1]["description"] == "d1'"
    assert new_library[1]["semantic_review"]["action"] == "rewrite"
    # Version write-back chain: audit carries old/new library fingerprints and they
    # differ
    assert audit["previous_rubric_library_sha256"]
    assert audit["previous_rubric_library_sha256"] != audit["rubric_library_sha256"]
    assert audit["approved_count"] == 1
    assert audit["rewritten_count"] == 1
    assert audit["retired_count"] == 1
    retired = next(v for v in audit["verdicts"] if v["action"] == "retire")
    assert retired["retired_weight"] == 1.0


def test_apply_reviews_rejects_unknown_duplicate_or_bad_verdicts():
    with pytest.raises(PositiveRefinementProtocolError):
        meta_review.apply_reviews(
            LIBRARY, [{"rubric_key": "rX", "action": "approve", "reason": "ok"}]
        )
    with pytest.raises(PositiveRefinementProtocolError):
        meta_review.apply_reviews(
            LIBRARY,
            [
                {"rubric_key": "r0", "action": "approve", "reason": "ok"},
                {"rubric_key": "r0", "action": "retire", "reason": "again"},
            ],
        )
    with pytest.raises(PositiveRefinementProtocolError):
        meta_review.apply_reviews(
            LIBRARY, [{"rubric_key": "r0", "action": "mutate", "reason": "ok"}]
        )
    # Empty verdicts = no-op; library unchanged, audit all zero
    new_library, audit = meta_review.apply_reviews(LIBRARY, [])
    assert new_library == [dict(rubric) for rubric in LIBRARY]
    assert audit["reviewed_count"] == 0


def test_apply_reviews_retiring_every_criterion_fails_validation():
    with pytest.raises(PositiveRefinementProtocolError):
        meta_review.apply_reviews(
            LIBRARY[:1],
            [{"rubric_key": "r0", "action": "retire", "reason": "only one"}],
        )
