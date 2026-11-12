"""Rubric capability layer: RuACT inference-side migration -- binary rubric protocol,
Judge scoring path, and sample gate. Capability-layer discipline: depend only on
`data_designer.contracts` and the standard library; never import pipelines /
integrations / cli, and never import AscendAgent-Gym. Public integration surfaces: -
verifier kind `rubric_critic` (routed through `data_designer.verifiers.dispatch`); -
Python API: `RubricGate` (score -> gate -> decision), `configure_judge_endpoints`
(explicit credential injection; never published with task payloads); - evolution
engine (P3): `build_refinement_jobs`, `process_refinement_job`,
`PositiveRefinementRegistry`, `PositiveRefinementSeedLibrary`,
`overlay_decompositions` / `apply_decompositions_to_scores`; - scenario B extension
(P4): `extract_error_patterns`, `classify_expansion_samples` /
`split_expansion_sets`, `negative_criterion_review`.
"""

from .evolve.decompose import (
    apply_decompositions_to_scores,
    build_refinement_jobs,
    overlay_decompositions,
)
from .evolve.registry import PositiveRefinementRegistry
from .evolve.seed_library import PositiveRefinementSeedLibrary
from .gate.acceptance import FamilyThresholdAnchor, decide, threshold_from_scores
from .gate.decisions import GateDecision, RubricEvidence
from .gate.triage import (
    classify_expansion_samples,
    negative_criterion_review,
    split_expansion_sets,
)
from .generator.extract import extract_error_patterns
from .generator.single_pass import process_refinement_job
from .judge.score import aggregate_weighted_reward, rubric_passed
from .judge.transport import (
    JudgeGiveUpError,
    JudgeProtocolError,
    JudgeResult,
    JudgeTransport,
    configure_judge_endpoints,
    reset_judge_endpoints,
)
from .orchestrate.group import ALLOWED_GROUP_SIZES, GroupCollector, GroupTimeoutError
from .orchestrate.pipeline import RubricGate, persist_decisions
from .protocol.schemas import PositiveRefinementProtocolError, validate_rubrics

__all__ = [
    "ALLOWED_GROUP_SIZES",
    "FamilyThresholdAnchor",
    "GateDecision",
    "GroupCollector",
    "GroupTimeoutError",
    "JudgeGiveUpError",
    "JudgeProtocolError",
    "JudgeResult",
    "JudgeTransport",
    "PositiveRefinementProtocolError",
    "PositiveRefinementRegistry",
    "PositiveRefinementSeedLibrary",
    "RubricEvidence",
    "RubricGate",
    "aggregate_weighted_reward",
    "apply_decompositions_to_scores",
    "build_refinement_jobs",
    "classify_expansion_samples",
    "configure_judge_endpoints",
    "decide",
    "extract_error_patterns",
    "negative_criterion_review",
    "overlay_decompositions",
    "persist_decisions",
    "process_refinement_job",
    "reset_judge_endpoints",
    "rubric_passed",
    "split_expansion_sets",
    "threshold_from_scores",
    "validate_rubrics",
]
