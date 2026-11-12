from __future__ import annotations

import json
from pathlib import Path


def read_rows(uri, format="jsonl"):
    path = Path(uri)
    if format == "parquet":
        try:
            import pyarrow.parquet as pq
        except ImportError as error:
            raise RuntimeError("Parquet requires data-forge[parquet]") from error
        for batch in pq.ParquetFile(path).iter_batches():
            yield from batch.to_pylist()
    elif format == "jsonl":
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if line.strip():
                    try:
                        row = json.loads(line, parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
                    except ValueError as error:
                        raise ValueError(f"invalid JSONL record at line {line_number}") from error
                    if not isinstance(row, dict):
                        raise ValueError(f"JSONL line {line_number} must be an object")
                    yield row
    else:
        raise ValueError(f"unsupported input format: {format}")
