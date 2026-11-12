"""op mode: operator invocation entry point using a single JSON message over stdin/stdout (Evaluator §4.3)."""
from curator.trusted_worker.op.protocol import handle, main

__all__ = ["handle", "main"]
