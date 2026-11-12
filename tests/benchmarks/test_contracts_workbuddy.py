"""Contract tests for the workbuddy family (M1): public item round-trip, family prefixes, and private-field blocking.

Covers implementation plan §2 M1 acceptance:
  - ``workbuddy`` public items pass ``make_public_item`` / ``assert_public_item``;
  - task_ids without the ``workbuddy:`` prefix are rejected;
  - PRIVATE_MARKERS fields such as ``verifier_spec`` are rejected as soon as they enter a public item;
  - tampered public item fingerprints are rejected.
"""

import pytest

from data_designer.pipelines.benchmarks.contracts import (
    FAMILIES,
    BenchmarkError,
    assert_public_item,
    make_public_item,
    require_family,
)

PUBLIC_FIELDS = {
    "task_id": "workbuddy:wb-fixture-1:v1",
    "task_text": "Summarize the approved quarterly revenue workbook.",
    "tools_available": ["workdir"],
    "success_criteria": "Produce output/summary.json matching the frozen gold.",
    "difficulty": "medium",
    "family": "workbuddy",
    "source_release": "synthetic-fixture:abcd1234",
    "source_task_id": "seed-1",
    "variant_id": "v1",
}


def test_workbuddy_is_a_registered_family():
    assert "workbuddy" in FAMILIES
    assert require_family("workbuddy") == "workbuddy"


def test_workbuddy_public_item_roundtrip():
    item = make_public_item(PUBLIC_FIELDS, "a" * 64, "b" * 64, "c" * 64)
    assert set(item) == {
        "task_id", "task_text", "tools_available", "success_criteria",
        "difficulty", "item_fingerprint", "benchmark",
    }
    assert item["benchmark"]["family"] == "workbuddy"
    assert item["task_id"].startswith("workbuddy:")
    assert "verifier_spec" not in item
    assert not ({"gold", "verifier_path"} & set(item))
    assert_public_item(item)


def test_non_workbuddy_task_id_prefix_is_rejected():
    with pytest.raises(BenchmarkError, match="family namespace"):
        make_public_item(
            {**PUBLIC_FIELDS, "task_id": "tau:wb-fixture-1:v1"},
            "a" * 64, "b" * 64, "c" * 64,
        )


def test_private_marker_key_is_rejected():
    item = make_public_item(PUBLIC_FIELDS, "a" * 64, "b" * 64, "c" * 64)
    with pytest.raises(BenchmarkError):
        assert_public_item({**item, "verifier_spec": {"kind": "artifact_compare"}})
    with pytest.raises(BenchmarkError):
        assert_public_item({**item, "gold_path": "/tmp/gold.json"})


def test_tampered_fingerprint_is_rejected():
    item = make_public_item(PUBLIC_FIELDS, "a" * 64, "b" * 64, "c" * 64)
    tampered = dict(item, item_fingerprint="d" * 64)
    with pytest.raises(BenchmarkError, match="fingerprint"):
        assert_public_item(tampered)
