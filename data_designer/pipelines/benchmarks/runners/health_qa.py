"""health_qa native runner: healthbench offline scoring (judge threshold semantics).

Thin entry over ``_single_turn_common`` — family pin plus the shared JSON file
protocol. healthbench is an ``llm_rubric`` member: the rubric text and
``pass_threshold`` are frozen in the trusted payload; the judge endpoint is
injected by the marking side (``configure_llm_judge``) and an unconfigured
judge is an explicit failure, never a fabricated score (D7).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from data_designer.pipelines.benchmarks.runners._single_turn_common import (  # noqa: E402
    SUPPORTED_SPEC_KINDS,
    VERSION,
    score_request,
)

FAMILY = "health_qa"


def main() -> None:
    if len(sys.argv) != 3:
        raise ValueError("usage: health_qa.py <request.json> <result.json>")
    request = json.loads(
        Path(sys.argv[1]).read_text(encoding="utf-8"),
        parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
    )
    result = score_request(request, FAMILY)
    Path(sys.argv[2]).write_text(
        json.dumps(result, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()


__all__ = ["FAMILY", "SUPPORTED_SPEC_KINDS", "VERSION"]
