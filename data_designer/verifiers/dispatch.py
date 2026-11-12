from __future__ import annotations

from data_designer.contracts.verification import VerificationResult
from .registry import IMPLEMENTATIONS, load_verifier


class VerifierDispatchError(RuntimeError):
    pass


SUPPORTED_VERIFIER_KINDS = tuple(IMPLEMENTATIONS)


def validate_spec(spec):
    if not isinstance(spec, dict) or spec.get("kind") not in IMPLEMENTATIONS:
        raise VerifierDispatchError("unknown or missing verifier kind")
    if not isinstance(spec.get("payload"), dict) or not spec["payload"] or not spec.get("generator_ref"):
        raise VerifierDispatchError("verifier payload and generator_ref are required")
    return spec


def run_verifier(spec, artifact, *, judge=None):
    validate_spec(spec)
    kind = spec["kind"]
    try:
        implementation = load_verifier(kind)
    except ImportError:
        return VerificationResult("unavailable", kind, detail="verifier dependency is unavailable").as_dict()
# Verifier execution exceptions are raised explicitly (except missing dependencies degraded to unavailable),
# not swallowed as a 0 score--callers decide how to present failures according to their own protocol.
    result = (
        implementation(artifact, spec["payload"], judge=judge)
        if kind == "llm_rubric"
        else implementation(artifact, spec["payload"])
    )
    status = result.get("status", "completed")
    return VerificationResult(status, kind, passed=result.get("passed") if status == "completed" else None,
                              score=result.get("score") if status == "completed" else None,
                              detail=result.get("detail", "")).as_dict()
