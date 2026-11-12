"""``python -m curator.runtime`` entry point."""

from .zmq_server import main

if __name__ == "__main__":
    raise SystemExit(main())
