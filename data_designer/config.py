"""One strict configuration model shared by SDK, CLI and Curator."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Mapping

JOB_VERSION = "dataforge-job-v1"


def object_schema(properties, required=()):
    return {"type": "object", "properties": properties,
            "required": list(required), "additionalProperties": False}


JOB_SCHEMA = object_schema({
    "schema_version": {"const": JOB_VERSION, "default": JOB_VERSION},
    "recipe": {"type": "string", "minLength": 1},
    "input": {**object_schema({
        "uri": {"type": "string", "minLength": 1},
        "format": {"enum": ["jsonl", "parquet"], "default": "jsonl"},
        "schema": {"type": "string", "default": "record-v1"},
    }), "default": {}},
    "output": {**object_schema({
        "uri": {"type": "string", "minLength": 1},
        "format": {"enum": ["jsonl", "parquet"], "default": "jsonl"},
        "policy": {"enum": ["generated_only", "append", "replace"], "default": "generated_only"},
        "overwrite": {"type": "boolean", "default": False},
        "target_new_records": {"type": "integer", "minimum": 0},
        "expansion_ratio": {"type": "number", "minimum": 0},
        "require_new_records": {"type": "boolean", "default": False},
    }), "default": {}},
    "recipe_config": {"type": "object", "default": {}},
    "execution": {**object_schema({
        "seed": {"type": "integer", "default": 42},
        "max_attempts": {"type": "integer", "minimum": 1, "default": 10000},
        "max_input_records": {"type": "integer", "minimum": 1, "default": 100000},
        "batch_size": {"type": "integer", "minimum": 1, "default": 32},
        "on_record_error": {"enum": ["raise", "reject"], "default": "raise"},
        "allow_partial": {"type": "boolean", "default": False},
    }), "default": {}},
}, required=("recipe",))


def validate_config(value: Any, schema: Mapping[str, Any]) -> dict:
    from jsonschema import Draft202012Validator
    Draft202012Validator.check_schema(schema)
    result = copy.deepcopy(value)

    def defaults(node, spec):
        if isinstance(node, dict):
            for key, child in spec.get("properties", {}).items():
                if key not in node and "default" in child:
                    node[key] = copy.deepcopy(child["default"])
                if key in node:
                    defaults(node[key], child)
    defaults(result, schema)
    errors = sorted(Draft202012Validator(schema).iter_errors(result), key=lambda e: str(e.path))
    if errors:
        error = errors[0]
        raise ValueError(f"configuration {'.'.join(map(str, error.path)) or '<root>'}: {error.message}")
    return result


def load_config(config, *, base_directory=None) -> dict:
    if isinstance(config, (str, Path)):
        path = Path(config).resolve()
        base = path.parent
        text = path.read_text(encoding="utf-8")
        if path.suffix.lower() == ".json":
            value = json.loads(text)
        else:
            try:
                import yaml
            except ImportError as error:
                raise RuntimeError("YAML jobs require data-forge[yaml]") from error
            value = yaml.safe_load(text)
    else:
        value = config
        base = Path(base_directory or Path.cwd()).resolve()
    result = validate_config(value, JOB_SCHEMA)
    out = result["output"]
    if "target_new_records" in out and "expansion_ratio" in out:
        raise ValueError("target_new_records and expansion_ratio are mutually exclusive")
    for section in (result["input"], out):
        if "uri" in section:
            path = Path(section["uri"])
            if "://" in section["uri"]:
                raise ValueError("local jobs require filesystem paths, not remote URIs")
            section["uri"] = str((base / path).resolve())
    return result


def operator_schema():
    return {"category": "generation", "description": "Shared DataForge generation job",
            "params": {key: {"required": key in JOB_SCHEMA["required"], **spec}
                       for key, spec in JOB_SCHEMA["properties"].items()}}
