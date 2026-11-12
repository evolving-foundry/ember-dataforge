"""trusted_worker production package: trusted subprocess entry point for Curator operators.

Both entry points share the ``ops/`` operator implementations (one infrastructure, three hosts: RSI / DataForge / Evaluator):

* **evaluation mode** (RSI frozen; RSI Runtime integration design v0.2, §3.9.2, v0.2.1):
  ``python -m curator.trusted_worker <input.json> <result.json>``,
  two-file envelope/result, ``schemaVersion: 2``.
* **op mode** (Evaluator integration design v0.1, §4.3):
  ``python -m curator.trusted_worker.op``, a single JSON message over stdin/stdout,
  ``schemaVersion: 1``.
"""
from curator.trusted_worker.errors import (
    EXIT_INVALID_INPUT,
    EXIT_KNOWN_FAILURE,
    EXIT_OK,
    WorkerError,
)
from curator.trusted_worker.registry import OP_REGISTRY
from curator.trusted_worker.spawn import run

__all__ = [
    "OP_REGISTRY",
    "run",
    "WorkerError",
    "EXIT_OK",
    "EXIT_KNOWN_FAILURE",
    "EXIT_INVALID_INPUT",
]
