"""``python -m curator.trusted_worker <input.json> <result.json>`` (evaluation mode CLI)."""
import sys

from curator.trusted_worker.spawn import run

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(
            "usage: python -m curator.trusted_worker <input.json> <result.json>",
            file=sys.stderr,
        )
        sys.exit(2)
    sys.exit(run(sys.argv[1], sys.argv[2]))
