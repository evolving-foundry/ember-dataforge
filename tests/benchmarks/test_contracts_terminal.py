"""Contract tests for the terminal family (M1): public item round-trip, family prefixes, and private-field blocking.

Covers the terminal onboarding implementation plan §2 M1 acceptance:
  - ``terminal`` public items pass ``make_public_item`` / ``assert_public_item``;
  - task_ids without the ``terminal:`` prefix are rejected;
  - PRIVATE_MARKERS fields such as ``patch``/``test_code`` are rejected as soon as they enter a public item
    (upstream tests/solution stay only on the trusted side; public item leaks are rejected);
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
    "task_id": "terminal:tb21-fixture-1:v1",
    "task_text": "Fix the failing assertion in scripts/report.py and deliver "
                 "the corrected file.",
    "tools_available": ["shell", "python"],
    "success_criteria": "Deliver the corrected script file that passes the "
                        "frozen file assertions.",
    "difficulty": "medium",
    "family": "terminal",
    "source_release": "synthetic-fixture:abcd1234",
    "source_task_id": "tb-fixture-001",
    "variant_id": "v1",
}


def test_terminal_is_a_registered_family():
    assert "terminal" in FAMILIES
    assert require_family("terminal") == "terminal"


def test_deepswe_is_the_ninth_registered_family():
    assert "deepswe" in FAMILIES
    assert FAMILIES.index("deepswe") == 8
    assert require_family("deepswe") == "deepswe"


def test_terminal_public_item_roundtrip():
    item = make_public_item(PUBLIC_FIELDS, "a" * 64, "b" * 64, "c" * 64)
    assert set(item) == {
        "task_id", "task_text", "tools_available", "success_criteria",
        "difficulty", "item_fingerprint", "benchmark",
    }
    assert item["benchmark"]["family"] == "terminal"
    assert item["task_id"].startswith("terminal:")
    assert "verifier_spec" not in item
    assert not ({"patch", "test_code", "gold"} & set(item))
    assert_public_item(item)


def test_non_terminal_task_id_prefix_is_rejected():
    with pytest.raises(BenchmarkError, match="family namespace"):
        make_public_item(
            {**PUBLIC_FIELDS, "task_id": "workbuddy:tb21-fixture-1:v1"},
            "a" * 64, "b" * 64, "c" * 64,
        )


def test_upstream_test_and_solution_leak_markers_are_rejected():
    item = make_public_item(PUBLIC_FIELDS, "a" * 64, "b" * 64, "c" * 64)
    with pytest.raises(BenchmarkError):
        assert_public_item({**item, "patch": "diff --git a/x b/x"})
    with pytest.raises(BenchmarkError):
        assert_public_item({**item, "test_code": "assert result == gold"})


def test_tampered_fingerprint_is_rejected():
    item = make_public_item(PUBLIC_FIELDS, "a" * 64, "b" * 64, "c" * 64)
    tampered = dict(item, item_fingerprint="d" * 64)
    with pytest.raises(BenchmarkError, match="fingerprint"):
        assert_public_item(tampered)
