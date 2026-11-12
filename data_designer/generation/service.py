"""Shared generation use case. Curator and hosts remain outside this module."""
from __future__ import annotations

import math
import uuid
from collections import Counter
from dataclasses import asdict

from data_designer.config import load_config, validate_config
from data_designer.contracts.algorithms import AlgorithmRequest, AlgorithmResult
from data_designer.contracts.datasets import DatasetRef, RecordEnvelope
from .result import GenerationResult
from data_designer.contracts.provenance import content_digest
from data_designer.sources.local import read_rows
from .algorithms.registry import AlgorithmRegistry
from .context import AlgorithmContext
from .registry import RecipeRegistry


class GenerationIncompleteError(RuntimeError):
    def __init__(self, result):
        super().__init__(f"generation quota unmet: {result.report['shortfall']} records short")
        self.result = result


def _records(rows, schema, limit):
    result, ids = [], set()
    for index, row in enumerate(rows):
        if index >= limit:
            raise ValueError(f"input resource limit exceeded ({limit} records); no truncation applied")
        if isinstance(row, RecordEnvelope):
            record = row
        elif isinstance(row, dict):
            record_id = row.get("id") or row.get("sample_id") or row.get("graph_id") or row.get("seed_id")
            record = RecordEnvelope(str(record_id) if record_id is not None else f"input:{index}", dict(row), schema)
        else:
            raise ValueError("input records must be objects or RecordEnvelope instances")
        if record.id in ids:
            raise ValueError(f"duplicate input sample id: {record.id}")
        if record.schema != schema:
            raise ValueError(f"input schema mismatch: {record.schema} != {schema}")
        ids.add(record.id)
        result.append(record)
    return tuple(result)


def generate(input=None, config=None, context=None, *, base_directory=None):
    job = load_config(config, base_directory=base_directory)
    if input is not None and "uri" in job["input"]:
        raise ValueError("input must have one source: explicit data or input.uri, not both")
    output, execution = job["output"], job["execution"]
    context = context or AlgorithmContext(seed=execution["seed"])
    context.check_cancelled()
    if context.seed != execution["seed"]:
        raise ValueError("context seed must match execution.seed")
    registry = context.registry or AlgorithmRegistry()
    selection = RecipeRegistry().resolve(job["recipe"], job["recipe_config"])
    descriptor = registry.describe(selection["id"])
    spec = descriptor.spec
    if selection.get("implementation_version", spec.implementation_version) != spec.implementation_version:
        raise ValueError("algorithm implementation version mismatch")
    if selection["adapter"] not in spec.execution:
        raise ValueError("unsupported algorithm execution adapter")
    params = validate_config(selection["params"], spec.config_schema)
    if output["format"] != "jsonl":
        raise ValueError("dataset package output currently supports JSONL; Parquet is an input capability")
    if "uri" in output:
        from pathlib import Path
        if Path(output["uri"]).exists() and not output["overwrite"]:
            raise FileExistsError("output already exists")
        if "uri" in job["input"]:
            src, dst = Path(job["input"]["uri"]), Path(output["uri"])
            if src == dst or dst in src.parents:
                raise ValueError("output cannot overwrite the input source")
    schema = job["input"]["schema"]
    if spec.input_schema != "record-v1" and schema != spec.input_schema:
        raise ValueError(f"algorithm requires input schema {spec.input_schema}")
    if input is None:
        if "uri" not in job["input"]:
            raise ValueError("input data or input.uri is required")
        rows = read_rows(job["input"]["uri"], job["input"]["format"])
    elif isinstance(input, DatasetRef):
        if input.schema != schema:
            raise ValueError("DatasetRef schema mismatch")
        rows = read_rows(input.uri, input.format)
    else:
        rows = input
    limit = min(execution["max_input_records"], params.get("max_input_records", execution["max_input_records"]),
                spec.resources.get("max_input_records", execution["max_input_records"]))
    records = _records(rows, schema, limit)
    input_ids = {r.id for r in records}
    target = output.get("target_new_records")
    if "expansion_ratio" in output:
        target = math.ceil(len(records) * output["expansion_ratio"])
    if target is not None and "quota" not in spec.capabilities:
        raise ValueError("selected algorithm does not support output quotas")
    if output["policy"] == "append" and "expand" not in spec.capabilities:
        raise ValueError("append requires an expand-capable algorithm")
    run_id = str(uuid.uuid4())
    source_refs = ("config:" + content_digest({"algorithm": spec.id, "params": params}),)
    algorithm = registry.create(spec.id)
    rejected = Counter()
    generated, generated_ids, diagnostics = [], set(), []
    provenance = []
    attempts = 0
    if spec.scope == "dataset":
        chunks = (records,)
    else:
        size = 1 if spec.scope == "record" else execution["batch_size"]
        chunks = (records[i:i + size] for i in range(0, len(records), size))
    for chunk in chunks:
        if (target is not None and len(generated) >= target) or attempts >= execution["max_attempts"]:
            break
        context.check_cancelled()
        attempts += 1
        request = AlgorithmRequest(tuple(chunk), params, run_id,
                                   None if target is None else target - len(generated),
                                   execution["max_attempts"] - attempts + 1, source_refs)
        try:
            response = algorithm.run(request, context)
        except Exception:
            # Dataset failures and cancellation never masquerade as rejected records.
            if spec.scope != "record" or execution["on_record_error"] != "reject":
                raise
            context.check_cancelled()
            rejected["record_execution_error"] += 1
            continue
        if not isinstance(response, AlgorithmResult):
            raise TypeError("algorithm must return AlgorithmResult")
        rejected.update(response.rejected)
        diagnostics.append(response.diagnostics)
        for record in response.records:
            context.check_cancelled()
            if not isinstance(record, RecordEnvelope):
                raise TypeError("algorithm outputs must be RecordEnvelope instances")
            if record.schema != spec.output_schema:
                raise ValueError("algorithm output schema mismatch")
            if record.id in generated_ids or (output["policy"] == "append" and record.id in input_ids):
                raise ValueError(f"output sample id collision: {record.id}")
            if not record.parents or not set(record.parents) <= input_ids | set(source_refs):
                raise ValueError(f"output parent provenance invalid: {record.id}")
            generated.append(record)
            generated_ids.add(record.id)
            provenance.append({"sample_id": record.id, "parents": list(record.parents),
                               "content_digest": content_digest(record.payload),
                               "algorithm": spec.id, "implementation_version": spec.implementation_version})
            if target is not None and len(generated) >= target:
                break
    shortfall = max(0, (target or (1 if output["require_new_records"] else 0)) - len(generated))
    # A bounded execution that skipped inputs is incomplete even without a quota.
    if spec.scope in {"record", "batch"} and target is None:
        size = 1 if spec.scope == "record" else execution["batch_size"]
        shortfall = max(shortfall, len(records) - attempts * size)
    status = "partial" if shortfall else "completed"
    report = {"run_id": run_id, "status": status, "recipe": job["recipe"],
              "algorithm": asdict(spec), "plugin_version": descriptor.package_version,
              "resolved_config": params, "seed": execution["seed"],
              "source_refs": source_refs,
              "input_digest": content_digest([asdict(r) for r in records]),
              "input_count": len(records), "generated_count": len(generated),
              "output_schema": spec.output_schema, "output_policy": output["policy"],
              "target_new_records": target, "shortfall": shortfall, "attempts": attempts,
              "rejected": dict(rejected), "diagnostics": diagnostics,
              "reproducibility": spec.determinism, "service_buffers_input": True,
              "max_input_records": limit}
    final = list(records) + generated if output["policy"] == "append" else generated
    result = GenerationResult(final, len(generated), len(records), status, report, provenance)
    if shortfall and not execution["allow_partial"]:
        result.status = "failed"
        result.report["status"] = "failed"
        raise GenerationIncompleteError(result)
    if "uri" in output:
        result.save(output["uri"], overwrite=output["overwrite"])
    return result
