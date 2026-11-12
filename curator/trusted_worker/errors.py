"""Shared error types and exit-code semantics for trusted_worker.

Exit-code conventions (unified across the three hosts; see RSI Runtime integration design v0.2, §3.9.2 and
Evaluator integration design v0.1, §4.3/§4.5):

  * 0   Success
  * 1   Known failure (op not in registry / data unreadable / operator raised an error / non-finite score)
  * 2   Invalid input (envelope or request structure error; semantically 'evaluation did not start')

Red line: failures do not write results, do not produce stdout, and do not fabricate scores; diagnostics go only to stderr
(a single line ``{'code': '...', 'message': '...'}``) and never into the result file.
"""
from __future__ import annotations

EXIT_OK = 0
EXIT_KNOWN_FAILURE = 1
EXIT_INVALID_INPUT = 2


class WorkerError(Exception):
    """Worker failure with structured diagnostics.

    ``code`` goes into the stderr diagnostic line; ``exit_code`` determines the process exit code.
    """

    def __init__(self, code: str, message: str, exit_code: int = EXIT_KNOWN_FAILURE):
        super().__init__(message)
        self.code = code
        self.message = message
        self.exit_code = exit_code


def diagnostic_line(error: WorkerError) -> str:
    """Render as a single-line JSON diagnostic on stderr."""
    import json

    return json.dumps(
        {"code": error.code, "message": error.message},
        ensure_ascii=False,
    )
