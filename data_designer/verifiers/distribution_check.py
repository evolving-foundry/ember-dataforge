"""Distribution-overlap verifier used by environment distribution gates."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence


class DistributionCheckError(ValueError):
    """The distribution payload is malformed."""


def _distribution(values: Any, name: str) -> List[float]:
    if not isinstance(values, list) or not values:
        raise DistributionCheckError("{} must be a non-empty numeric array".format(name))
    result: List[float] = []
    for value in values:
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise DistributionCheckError("{} may contain only numeric values".format(name))
        number = float(value)
        if number < 0:
            raise DistributionCheckError("{} must not contain negative numbers".format(name))
        result.append(number)
    total = sum(result)
    if total <= 0:
        raise DistributionCheckError("{} sum must be greater than 0".format(name))
    return [value / total for value in result]


def check_distribution(artifact: Any, payload: Dict[str, Any]) -> Dict[str, Any]:
    reference = _distribution(payload.get("reference"), "payload.reference")
    candidate = _distribution(payload.get("candidate"), "payload.candidate")
    if len(reference) != len(candidate):
        raise DistributionCheckError("reference/candidate lengths must match")
    threshold = payload.get("threshold", 0.6)
    if not isinstance(threshold, (int, float)) or isinstance(threshold, bool):
        raise DistributionCheckError("payload.threshold must be numeric")
    threshold = float(threshold)
    if not 0.0 <= threshold <= 1.0:
        raise DistributionCheckError("payload.threshold must be in [0,1]")
    overlap = 1.0 - 0.5 * sum(
        abs(left - right) for left, right in zip(reference, candidate)
    )
    score = round(max(0.0, min(1.0, overlap)), 6)
    passed = score >= threshold
    return {
        "passed": passed,
        "score": score,
        "detail": {
            "threshold": threshold,
            "bins": len(reference),
            "method": "normalized-l1-overlap",
        },
    }


__all__ = ["DistributionCheckError", "check_distribution"]
