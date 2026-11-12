"""code_gen native runner: humaneval offline scoring (D8 execution boundary).

Thin entry over ``_single_turn_common`` — family pin plus the shared JSON file
protocol. The candidate delivers ``solution.py``; the frozen ``unit_test``
assertion script assembles signature + candidate code + tests and executes
them inside the trusted whitelisted-interpreter boundary with an enforced
timeout pinned in the verifier_spec — the terminal M3 security model, no new
mechanism.
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

FAMILY = "code_gen"


def main() -> None:
    if len(sys.argv) != 3:
        raise ValueError("usage: code_gen.py <request.json> <result.json>")
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
