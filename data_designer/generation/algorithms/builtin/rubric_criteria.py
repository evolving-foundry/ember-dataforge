"""Rubric criteria generation algorithm family (design document section 3.2, integration
surface 2). - `rubric.criteria_extract`: reviewed good cases -> demonstrations
payload (verbatim evidence_quote citation validation). -
`rubric.criteria_extract_error`: reviewed error cases -> negative rubric library
(prompt-layer variant; payload marks library_polarity=negative). -
`rubric.criteria_generate`: demonstration library + target query/parent ->
`generated_unscored` rubric candidates (not stored before Judge scoring and
discriminative filtering). Generator endpoint credentials are injected explicitly per
design document section 6.4: when config
`generator_base_url`/`generator_api_key`/`generator_model` are absent, fall back to
`HEALTHBENCH_GENERATOR_API_BASE/KEY/MODEL` environment variables -- credentials never
enter configuration files.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from data_designer.config import object_schema
from data_designer.contracts.algorithms import AlgorithmResult, AlgorithmSpec
from data_designer.contracts.datasets import RecordEnvelope

from data_designer.rubric.generator.client import WorkerConfig
from data_designer.rubric.generator.demonstrations import DemonstrationLibrary
from data_designer.rubric.generator.extract import (
    extract_demonstrations,
    extract_error_patterns,
    generate_criteria,
)

_EXTRACT_CONFIG = object_schema(
    {
        "generator_base_url": {"type": "string"},
        "generator_api_key": {"type": "string"},
        "generator_model": {"type": "string"},
        "max_tokens": {"type": "integer", "minimum": 1, "default": 4096},
        "max_input_characters": {"type": "integer", "minimum": 1000, "default": 48000},
        "exclude_queries": {"type": "array", "items": {"type": "string"}, "default": []},
        "output_path": {"type": "string"},
    }
)

_GENERATE_CONFIG = object_schema(
    {
        "generator_base_url": {"type": "string"},
        "generator_api_key": {"type": "string"},
        "generator_model": {"type": "string"},
        "max_tokens": {"type": "integer", "minimum": 1, "default": 4096},
        "max_input_characters": {"type": "integer", "minimum": 1000, "default": 48000},
        "demonstrations_path": {"type": "string"},
    }
)

CRITERIA_EXTRACT = AlgorithmSpec(
    id="rubric.criteria_extract",
    implementation_version="1",
    config_schema=_EXTRACT_CONFIG,
    output_schema="rubric-demonstrations-v1",
    scope="dataset",
    capabilities=("transform",),
    resources={"full_input_memory": True, "dependencies": ["network"]},
    determinism="external",
    upstream={
        "method": "RuACT criteria.py extract — reviewed excellent cases to demonstration library",
        "differences": "Sync urllib generator client; atomic noreplace output; verbatim evidence quotes enforced",
    },
)

CRITERIA_EXTRACT_ERROR = AlgorithmSpec(
    id="rubric.criteria_extract_error",
    implementation_version="1",
    config_schema=_EXTRACT_CONFIG,
    output_schema="rubric-demonstrations-v1",
    scope="dataset",
    capabilities=("transform",),
    resources={"full_input_memory": True, "dependencies": ["network"]},
    determinism="external",
    upstream={
        "method": "Design doc §5.4 — reviewed error scenes to negative-criterion library (prompt-level variant)",
        "differences": "Prompt pivots to violation points; evidence_quote must be a verbatim error-scene substring; payload marked library_polarity=negative",
    },
)

CRITERIA_GENERATE = AlgorithmSpec(
    id="rubric.criteria_generate",
    implementation_version="1",
    config_schema=_GENERATE_CONFIG,
    output_schema="rubric-criteria-generated-v1",
    scope="dataset",
    capabilities=("transform",),
    resources={"full_input_memory": True, "dependencies": ["network"]},
    determinism="external",
    upstream={
        "method": "RuACT criteria.py generate — demonstration-anchored criteria for target queries",
        "differences": "Outputs stay generated_unscored until Judge scoring plus discriminative filtering",
    },
)


def generator_config(config) -> WorkerConfig:
    worker = WorkerConfig(
        base_url=str(
            config.get("generator_base_url")
            or os.environ.get("HEALTHBENCH_GENERATOR_API_BASE")
            or ""
        ),
        api_key=str(
            config.get("generator_api_key")
            or os.environ.get("HEALTHBENCH_GENERATOR_API_KEY")
            or ""
        ),
        model=str(
            config.get("generator_model")
            or os.environ.get("HEALTHBENCH_GENERATOR_MODEL")
            or ""
        ),
        max_tokens=int(config.get("max_tokens", 4096)),
        max_attempts=3,
    )
    worker.validate()
    return worker


class CriteriaExtractAlgorithm:
    def describe(self):
        return CRITERIA_EXTRACT

    def run(self, request, context):
        rows = [item.payload for item in request.records]
        payload = extract_demonstrations(
            rows,
            config=generator_config(request.config),
            exclude_queries=tuple(request.config.get("exclude_queries") or ()),
            max_input_characters=int(
                request.config.get("max_input_characters", 48000)
            ),
        )
        output_path = request.config.get("output_path")
        if output_path:
            from data_designer.rubric.generator.extract import write_new

            write_new(output_path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        record = RecordEnvelope(
            f"demonstrations:{request.run_id}",
            payload,
            CRITERIA_EXTRACT.output_schema,
            tuple((item.id for item in request.records)),
        )
        return AlgorithmResult(
            (record,),
            diagnostics={"example_count": len(payload["examples"])},
        )


class CriteriaExtractErrorAlgorithm:
    def describe(self):
        return CRITERIA_EXTRACT_ERROR

    def run(self, request, context):
        rows = [item.payload for item in request.records]
        payload = extract_error_patterns(
            rows,
            config=generator_config(request.config),
            exclude_queries=tuple(request.config.get("exclude_queries") or ()),
            max_input_characters=int(
                request.config.get("max_input_characters", 48000)
            ),
        )
        output_path = request.config.get("output_path")
        if output_path:
            from data_designer.rubric.generator.extract import write_new

            write_new(output_path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        record = RecordEnvelope(
            f"error-patterns:{request.run_id}",
            payload,
            CRITERIA_EXTRACT_ERROR.output_schema,
            tuple((item.id for item in request.records)),
        )
        return AlgorithmResult(
            (record,),
            diagnostics={
                "example_count": len(payload["examples"]),
                "library_polarity": payload["library_polarity"],
            },
        )


class CriteriaGenerateAlgorithm:
    def describe(self):
        return CRITERIA_GENERATE

    def run(self, request, context):
        demonstrations_path = request.config.get("demonstrations_path")
        if not demonstrations_path:
            raise ValueError("criteria_generate requires demonstrations_path")
        library = DemonstrationLibrary(demonstrations_path)
        rows = [item.payload for item in request.records]
        generated = generate_criteria(
            rows,
            library,
            config=generator_config(request.config),
            max_input_characters=int(
                request.config.get("max_input_characters", 48000)
            ),
        )
        records = tuple(
            RecordEnvelope(
                f"criteria:{request.run_id}:{index}",
                row,
                CRITERIA_GENERATE.output_schema,
                tuple((item.id for item in request.records)),
            )
            for index, row in enumerate(generated)
        )
        return AlgorithmResult(
            records,
            diagnostics={
                "target_count": len(generated),
                "demonstrations_sha256": library.sha256,
            },
        )
