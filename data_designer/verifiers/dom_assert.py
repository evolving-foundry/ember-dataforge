"""dom_assert verifier: static HTML/DOM parsing assertions (§7.2, browser-free).

payload structure (wb-task-v1):
    {
      "assertions": [
        {"type": "element_exists", "tag": "table"},
        {"type": "element_count", "tag": "tr", "op": ">=", "value": 5},
        {"type": "text_contains", "value": "Q3 revenue"},
        {"type": "attr_equals", "tag": "a", "attr": "href",
         "value": "report.pdf", "match": "exact|contains"}
      ]
    }

Uses stdlib ``html.parser`` (zero new dependencies); passed only when all assertions pass,
score = passed assertions / total assertions (aligned with artifact_compare score semantics).
"""

from __future__ import annotations

from html.parser import HTMLParser
from typing import Any, Dict, List, Tuple


class _DomIndex(HTMLParser):
    """Lightweight DOM index: tag counts (including nesting), full text, attribute samples."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tag_counts: Dict[str, int] = {}
        self.attrs: Dict[str, List[Dict[str, str]]] = {}
        self.text_parts: List[str] = []

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Any]]) -> None:
        self.tag_counts[tag] = self.tag_counts.get(tag, 0) + 1
        self.attrs.setdefault(tag, []).append(
            {k: (v if isinstance(v, str) else "") for k, v in attrs})

    def handle_startendtag(self, tag: str, attrs: List[Tuple[str, Any]]) -> None:
        self.handle_starttag(tag, attrs)

    def handle_data(self, data: str) -> None:
        if data.strip():
            self.text_parts.append(data.strip())

    @property
    def text(self) -> str:
        return "\n".join(self.text_parts)


def _parse_op(actual: int, op: str, want: int) -> bool:
    ops = {
        "==": actual == want, ">=": actual >= want,
        "<=": actual <= want, ">": actual > want, "<": actual < want,
    }
    return ops.get(op, False)


def _run_assertion(dom: _DomIndex, assertion: Dict[str, Any]) -> Tuple[bool, str]:
    atype = str(assertion.get("type") or "")
    if atype == "element_exists":
        tag = str(assertion.get("tag") or "")
        ok = dom.tag_counts.get(tag, 0) > 0
        return ok, f"element_exists <{tag}>: count={dom.tag_counts.get(tag, 0)}"
    if atype == "element_count":
        tag = str(assertion.get("tag") or "")
        want = int(assertion.get("value") or 0)
        op = str(assertion.get("op") or "==")
        actual = dom.tag_counts.get(tag, 0)
        return _parse_op(actual, op, want), \
            f"element_count <{tag}> {op} {want}: actual={actual}"
    if atype == "text_contains":
        value = str(assertion.get("value") or "")
        ok = value in dom.text
        return ok, f"text_contains {value!r}: {'hit' if ok else 'miss'}"
    if atype == "attr_equals":
        tag = str(assertion.get("tag") or "")
        attr = str(assertion.get("attr") or "")
        want = str(assertion.get("value") or "")
        mode = str(assertion.get("match") or "exact")
        candidates = [a.get(attr, "") for a in dom.attrs.get(tag, [])]
        if mode == "contains":
            ok = any(want in c for c in candidates)
        else:
            ok = want in candidates
        return ok, f"attr_equals <{tag} {attr}> {mode} {want!r}: " \
                   f"candidates={candidates[:5]}"
    return False, f"Unknown assertion type: {atype!r}"


def assert_dom(artifact: Any, payload: Dict[str, Any]) -> Dict[str, Any]:
    """verifier main entry: artifact is the HTML text produced by the agent."""
    assertions = payload.get("assertions")
    if not isinstance(assertions, list) or not assertions:
        return {"passed": False, "score": 0.0,
                "detail": "verifier payload is missing assertions (configuration error; fail explicitly)"}
    if not isinstance(artifact, str) or not artifact.strip():
        return {"passed": False, "score": 0.0,
                "detail": f"artifact is not HTML text: {type(artifact).__name__}"}
    dom = _DomIndex()
    try:
        dom.feed(artifact)
        dom.close()
    except Exception as exc:  # noqa: BLE001 - parse failure is a fail decision; detail carries the reason
        return {"passed": False, "score": 0.0,
                "detail": f"HTML parsing failed: {exc}"}
    results = [_run_assertion(dom, a if isinstance(a, dict) else {})
               for a in assertions]
    passed_count = sum(1 for ok, _ in results if ok)
    total = len(results)
    failed = [d for ok, d in results if not ok]
    return {
        "passed": passed_count == total,
        "score": round(passed_count / total, 4) if total else 0.0,
        "detail": "; ".join(failed) if failed else f"all {total} assertions pass",
    }


__all__ = ["assert_dom"]
