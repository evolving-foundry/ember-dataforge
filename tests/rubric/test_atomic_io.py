"""Atomic disk-write primitive tests (matching source atomic_io.py behavior)."""

from __future__ import annotations

import json

import pytest

from data_designer.rubric.io.atomic import (
    atomic_rename_dir_noreplace,
    atomic_replace_bytes,
    atomic_write_bytes_noreplace,
    atomic_write_json_noreplace,
)


def test_atomic_write_json_noreplace_sorted_keys(tmp_path):
    target = tmp_path / "audit" / "result.json"
    atomic_write_json_noreplace(target, {"b": 1, "a": {"d": 2, "c": 3}})
    content = json.loads(target.read_text(encoding="utf-8"))
    assert content == {"b": 1, "a": {"d": 2, "c": 3}}
    raw = target.read_text(encoding="utf-8")
    assert raw.index('"a"') < raw.index('"b"')


def test_atomic_write_noreplace_refuses_overwrite(tmp_path):
    target = tmp_path / "artifact.bin"
    atomic_write_bytes_noreplace(target, b"first")
    with pytest.raises(FileExistsError):
        atomic_write_bytes_noreplace(target, b"second")
    assert target.read_bytes() == b"first"
    leftovers = [p.name for p in target.parent.iterdir() if p.name != "artifact.bin"]
    assert leftovers == []


def test_atomic_replace_bytes_overwrites(tmp_path):
    target = tmp_path / "state.json"
    atomic_replace_bytes(target, b"v1")
    atomic_replace_bytes(target, b"v2")
    assert target.read_bytes() == b"v2"


def test_atomic_rename_dir_noreplace(tmp_path):
    source = tmp_path / "staging"
    source.mkdir()
    (source / "payload.txt").write_text("x", encoding="utf-8")
    target = tmp_path / "committed"
    atomic_rename_dir_noreplace(source, target)
    assert (target / "payload.txt").read_text(encoding="utf-8") == "x"

    source2 = tmp_path / "staging-2"
    source2.mkdir()
    with pytest.raises(FileExistsError):
        atomic_rename_dir_noreplace(source2, target)
    assert (source2 / ".").is_dir()
