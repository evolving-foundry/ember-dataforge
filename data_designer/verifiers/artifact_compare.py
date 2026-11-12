"""artifact_compare verifier: structured artifact read-back vs frozen gold programmatic comparison (§7.2).

payload structure (wb-task-v1):
    {
      "artifact": {"path": "output/summary.json", "format": "json"},
      "gold":     {"total_revenue_cny": 1500.0, ...},        # frozen gold label (synthetic value)
      "fields":   {                                          # optional per-field specification
        "total_revenue_cny": {"type": "number", "tolerance": 0.01},
        "by_region":         {"type": "number_map", "tolerance": 0.01},
        "top_region":        {"type": "string"},
        "missing_rows":      {"type": "integer"},
        "rows":              {"type": "record_list", "key_fields": ["institution", "report_date"]}
      }
    }

Default inference when a field is not declared in ``fields``: int->integer, float->number (tolerance 1e-6),
str->string, bool->boolean, dict[str,num]->number_map, list[dict]->record_list.
Comparison uses exact/tolerance semantics (Office "exactly-correct artifact") and does not make natural-language judgments.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

DEFAULT_TOLERANCE = 1e-6


def _infer_field_spec(name: str, gold_value: Any) -> Dict[str, Any]:
    if isinstance(gold_value, bool):
        return {"type": "boolean"}
    if isinstance(gold_value, int):
        return {"type": "integer"}
    if isinstance(gold_value, float):
        return {"type": "number", "tolerance": DEFAULT_TOLERANCE}
    if isinstance(gold_value, str):
        return {"type": "string"}
    if isinstance(gold_value, dict):
        if all(isinstance(v, (int, float)) and not isinstance(v, bool)
               for v in gold_value.values()):
            return {"type": "number_map", "tolerance": DEFAULT_TOLERANCE}
        return {"type": "json"}
    if isinstance(gold_value, list):
        if gold_value and all(isinstance(v, dict) for v in gold_value):
            return {"type": "record_list"}
        return {"type": "json"}
    return {"type": "json"}


def _num_eq(a: Any, b: Any, tolerance: float) -> bool:
    try:
        fa, fb = float(a), float(b)
    except (TypeError, ValueError):
        return False
    if math.isnan(fa) or math.isnan(fb):
        return False
    return abs(fa - fb) <= tolerance + 1e-12


def _json_sort_key(row: Dict[str, Any], key_fields: List[str]) -> Tuple:
    if key_fields:
        return tuple(str(row.get(k, "")) for k in key_fields)
    return tuple(f"{k}={row[k]!r}" for k in sorted(row))


def _compare_field(
    name: str,
    spec: Dict[str, Any],
    gold_value: Any,
    artifact_value: Any,
) -> Tuple[bool, str]:
    ftype = spec.get("type") or "json"
    if ftype == "number":
        ok = _num_eq(artifact_value, gold_value, float(spec.get("tolerance", DEFAULT_TOLERANCE)))
        return ok, f"{name}: got {artifact_value!r}, want {gold_value!r}"
    if ftype == "integer":
        ok = isinstance(artifact_value, int) and not isinstance(artifact_value, bool) \
            and int(artifact_value) == int(gold_value)
        return ok, f"{name}: got {artifact_value!r}, want int {gold_value!r}"
    if ftype == "boolean":
        ok = isinstance(artifact_value, bool) and artifact_value == gold_value
        return ok, f"{name}: got {artifact_value!r}, want {gold_value!r}"
    if ftype == "string":
        ok = isinstance(artifact_value, str) and artifact_value.strip() == str(gold_value).strip()
        return ok, f"{name}: got {artifact_value!r}, want {gold_value!r}"
    if ftype == "number_map":
        tol = float(spec.get("tolerance", DEFAULT_TOLERANCE))
        if not isinstance(artifact_value, dict) or set(artifact_value) != set(gold_value):
            got = sorted(artifact_value) if isinstance(artifact_value, dict) else artifact_value
            return False, f"{name}: keys mismatch got {got!r}, want {sorted(gold_value)}"
        bad = [k for k in gold_value if not _num_eq(artifact_value[k], gold_value[k], tol)]
        if bad:
            return False, f"{name}: value mismatch at {bad}"
        return True, f"{name}: ok"
    if ftype == "record_list":
        key_fields = list(spec.get("key_fields") or [])
        if not isinstance(artifact_value, list) or len(artifact_value) != len(gold_value):
            n = len(artifact_value) if isinstance(artifact_value, list) else None
            return False, f"{name}: row count got {n}, want {len(gold_value)}"
        want = sorted(gold_value, key=lambda r: _json_sort_key(r, key_fields))
        got = sorted(artifact_value, key=lambda r: _json_sort_key(r, key_fields)
                     if isinstance(r, dict) else ())
        for idx, (g_row, a_row) in enumerate(zip(want, got)):
            if not isinstance(a_row, dict):
                return False, f"{name}[{idx}]: not an object"
            for k, g_val in g_row.items():
                a_val = a_row.get(k)
                if isinstance(g_val, (int, float)) and not isinstance(g_val, bool):
                    if not _num_eq(a_val, g_val, DEFAULT_TOLERANCE):
                        return False, f"{name}[{idx}].{k}: got {a_val!r}, want {g_val!r}"
                elif a_val != g_val:
                    return False, f"{name}[{idx}].{k}: got {a_val!r}, want {g_val!r}"
        return True, f"{name}: {len(gold_value)} rows match"
# json / default: deep structural comparison (relax tolerance at numeric leaves)
    return _deep_eq(artifact_value, gold_value, name)


def _deep_eq(artifact_value: Any, gold_value: Any, path: str) -> Tuple[bool, str]:
    if isinstance(gold_value, (int, float)) and not isinstance(gold_value, bool) \
            and isinstance(artifact_value, (int, float)) and not isinstance(artifact_value, bool):
        return _num_eq(artifact_value, gold_value, DEFAULT_TOLERANCE), \
            f"{path}: got {artifact_value!r}, want {gold_value!r}"
    if type(artifact_value) is not type(gold_value) and not (
            isinstance(artifact_value, (list, tuple)) and isinstance(gold_value, (list, tuple))):
        return False, f"{path}: type {type(artifact_value).__name__}, want {type(gold_value).__name__}"
    if isinstance(gold_value, dict):
        if not isinstance(artifact_value, dict) or set(artifact_value) != set(gold_value):
            return False, f"{path}: keys mismatch"
        for k in gold_value:
            ok, detail = _deep_eq(artifact_value[k], gold_value[k], f"{path}.{k}")
            if not ok:
                return False, detail
        return True, f"{path}: ok"
    if isinstance(gold_value, list):
        if not isinstance(artifact_value, (list, tuple)) or \
                len(artifact_value) != len(gold_value):
            return False, f"{path}: length mismatch"
        for idx, (g, a) in enumerate(zip(gold_value, artifact_value)):
            ok, detail = _deep_eq(a, g, f"{path}[{idx}]")
            if not ok:
                return False, detail
        return True, f"{path}: ok"
    ok = artifact_value == gold_value
    return ok, f"{path}: got {artifact_value!r}, want {gold_value!r}"


def compare_artifact(artifact: Any, payload: Dict[str, Any]) -> Dict[str, Any]:
    """verifier main entry: artifact (already-parsed JSON value) vs payload.gold.

    Returns {"passed": bool, "score": float, "detail": str} (aligned with
    ExternalEvalHarness result rows; score = passed fields / total fields).
    """
    gold = payload.get("gold")
    if not isinstance(gold, dict) or not gold:
        return {"passed": False, "score": 0.0,
                "detail": "verifier payload is missing gold (configuration error; fail explicitly)"}
    if not isinstance(artifact, dict):
        return {"passed": False, "score": 0.0,
                "detail": f"artifact is not a JSON object: {type(artifact).__name__}"}

    field_specs: Dict[str, Dict[str, Any]] = dict(payload.get("fields") or {})
    results: List[Tuple[bool, str]] = []
    for name, gold_value in gold.items():
        spec = field_specs.get(name) or _infer_field_spec(name, gold_value)
        results.append(_compare_field(name, spec, gold_value, artifact.get(name)))
    passed_count = sum(1 for ok, _ in results if ok)
    total = len(results)
    failed = [detail for ok, detail in results if not ok]
    return {
        "passed": passed_count == total,
        "score": round(passed_count / total, 4) if total else 0.0,
        "detail": "; ".join(failed) if failed else f"all {total} fields match",
    }


__all__ = ["compare_artifact", "DEFAULT_TOLERANCE"]
