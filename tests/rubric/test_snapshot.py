"""P5 frozen_seed_only snapshot (section 4.5): library-wide review gate (section
6.5(d)), noreplace freeze semantics, dual-sha256 integrity validation.
"""
from __future__ import annotations

import json

import pytest

from data_designer.rubric.evolve.snapshot import (
    full_review_coverage,
    export_frozen_snapshot,
    load_frozen_snapshot,
)
from data_designer.rubric.protocol.schemas import PositiveRefinementProtocolError

LIBRARY = [
    {"rubric_key": "r0", "title": "t0", "description": "d0", "weight": 0.5},
    {"rubric_key": "r1", "title": "t1", "description": "d1", "weight": 0.5},
]


def _audit(keys, *, action="approve"):
    return {
        "verdicts": [
            {"rubric_key": key, "action": action, "reason": "ok"} for key in keys
        ]
    }


def test_full_review_coverage_counts_missing_and_treats_retire_as_reviewed():
    coverage = full_review_coverage(LIBRARY, _audit(["r0"]))
    assert coverage["missing_count"] == 1
    assert coverage["missing"] == [{"rubric_key": "r1", "observed_action": "none"}]
    # retire counts as reviewed: the retired key is absent from the new library and does
    # not enter the coverage gap
    retire_audit = {
        "verdicts": [
            {"rubric_key": "r0", "action": "approve", "reason": "ok"},
            {"rubric_key": "r1", "action": "retire", "reason": "dup"},
        ]
    }
    assert full_review_coverage([LIBRARY[0]], retire_audit)["missing_count"] == 0


def test_export_and_load_roundtrip(tmp_path):
    path = tmp_path / "snapshot.json"
    result = export_frozen_snapshot(
        LIBRARY,
        output_path=path,
        review_audit=_audit(["r0", "r1"]),
        exported_step=3,
        demonstrations_sha256="a" * 64,
    )
    assert result["evidence_scope"] == "frozen_seed_only"
    payload = load_frozen_snapshot(path)
    assert payload["exported_step"] == 3
    assert payload["evidence_scope"] == "frozen_seed_only"
    assert payload["demonstrations_sha256"] == "a" * 64
    assert [r["rubric_key"] for r in payload["rubric_library"]] == ["r0", "r1"]
    assert payload["snapshot_sha256"] == result["snapshot_sha256"]
    assert payload["review"]["verdict_count"] == 2


def test_export_requires_full_review_unless_explicitly_waived(tmp_path):
    with pytest.raises(PositiveRefinementProtocolError):
        export_frozen_snapshot(
            LIBRARY, output_path=tmp_path / "s.json", review_audit=_audit(["r0"])
        )
    # Explicit escape hatch: gaps are recorded in the audit persisted with the snapshot
    export_frozen_snapshot(
        LIBRARY,
        output_path=tmp_path / "waived.json",
        review_audit=_audit(["r0"]),
        require_full_review=False,
    )
    payload = load_frozen_snapshot(tmp_path / "waived.json")
    assert payload["review"]["full_review_coverage"]["missing_count"] == 1
    assert payload["review"]["full_review_coverage"]["missing"] == [
        {"rubric_key": "r1", "observed_action": "none"}
    ]


def test_export_refuses_overwrite_and_preserves_old_snapshot(tmp_path):
    path = tmp_path / "snapshot.json"
    export_frozen_snapshot(LIBRARY, output_path=path, review_audit=_audit(["r0", "r1"]))
    before = path.read_bytes()
    evolved = [dict(LIBRARY[0]), {**LIBRARY[1], "description": "evolved d1"}]
    with pytest.raises(FileExistsError):
        export_frozen_snapshot(evolved, output_path=path, review_audit=_audit(["r0", "r1"]))
    # Freeze semantics: further evolution/export after export does not affect the old
    # snapshot content or sha256
    assert path.read_bytes() == before
    payload = load_frozen_snapshot(path)
    assert payload["rubric_library"][1]["description"] == "d1"


def test_tampered_snapshot_rejected(tmp_path):
    path = tmp_path / "snapshot.json"
    export_frozen_snapshot(LIBRARY, output_path=path, review_audit=_audit(["r0", "r1"]))
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["rubric_library"][0]["description"] = "tampered"
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(PositiveRefinementProtocolError):
        load_frozen_snapshot(path)


def test_snapshot_schema_and_evidence_scope_enforced(tmp_path):
    path = tmp_path / "snapshot.json"
    export_frozen_snapshot(LIBRARY, output_path=path, review_audit=_audit(["r0", "r1"]))
    payload = json.loads(path.read_text(encoding="utf-8"))

    wrong_schema = dict(payload, snapshot_schema="other", snapshot_sha256="")
    bad_path = tmp_path / "schema.json"
    bad_path.write_text(json.dumps(wrong_schema), encoding="utf-8")
    with pytest.raises(PositiveRefinementProtocolError):
        load_frozen_snapshot(bad_path)

    # evidence_scope is checked before integrity validation: anything other than
    # frozen_seed_only is rejected immediately
    wrong_scope = dict(payload, evidence_scope="live_library")
    scope_path = tmp_path / "scope.json"
    scope_path.write_text(json.dumps(wrong_scope), encoding="utf-8")
    with pytest.raises(PositiveRefinementProtocolError):
        load_frozen_snapshot(scope_path)

    # Invalid exported_step fails explicitly
    with pytest.raises(PositiveRefinementProtocolError):
        export_frozen_snapshot(
            LIBRARY,
            output_path=tmp_path / "step.json",
            review_audit=_audit(["r0", "r1"]),
            exported_step=-1,
        )
    with pytest.raises(PositiveRefinementProtocolError):
        export_frozen_snapshot(
            LIBRARY,
            output_path=tmp_path / "step2.json",
            review_audit=_audit(["r0", "r1"]),
            exported_step=True,
        )
