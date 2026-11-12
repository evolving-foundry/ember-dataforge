"""``python -m curator.trusted_worker.op`` (op mode CLI, Evaluator §4.3)."""
import sys

from curator.trusted_worker._bootstrap import ensure_torch_stub

ensure_torch_stub()

from curator.trusted_worker.op.protocol import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main(sys.stdin, sys.stdout, sys.stderr))
