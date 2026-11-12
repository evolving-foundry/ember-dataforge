"""Standalone stdlib-only worker. No DataForge or Curator import."""
import json
import sys
from pathlib import Path


def main():
    request = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    if request["protocol_version"] != "dataforge-subprocess-v1":
        raise ValueError("unsupported request protocol")
    response = {key: request[key] for key in (
        "protocol_version", "run_id", "algorithm", "implementation_version", "request_digest")}
    response["status"] = "completed"
    response["records"] = [{"id": "demo:" + row["id"], "payload": {**row["payload"], "expanded": True},
                            "schema": "record-v1", "parents": [row["id"]]}
                           for row in request["records"]]
    Path(sys.argv[2]).write_text(json.dumps(response, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
