"""Canonical deterministic JSON/YAML serialization for environment synthesis."""

from __future__ import annotations

import json
from typing import Any, Iterable, Mapping


def json_bytes(value: Any, *, pretty: bool = True) -> bytes:
    if pretty:
        text = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    else:
        text = json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ) + "\n"
    return text.encode("utf-8")


def jsonl_bytes(rows: Iterable[Mapping[str, Any]]) -> bytes:
    lines = [
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        for row in rows
    ]
    return ("\n".join(lines) + "\n").encode("utf-8") if lines else b""


def yaml_bytes(value: Any) -> bytes:
    try:
        import yaml
    except ModuleNotFoundError:
        return json_bytes(value)
    return yaml.safe_dump(
        value, allow_unicode=True, sort_keys=False, default_flow_style=False
    ).encode("utf-8")


__all__ = ["json_bytes", "jsonl_bytes", "yaml_bytes"]
