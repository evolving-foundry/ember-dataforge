"""A lightweight descriptor; the worker is a separate isolated interpreter."""
from dataclasses import replace
from pathlib import Path
import sys

from data_designer.contracts.algorithms import AlgorithmDescriptor, AlgorithmSpec

SPEC = AlgorithmSpec(
    id="demo.independent", implementation_version="1",
    config_schema={"type": "object", "properties": {}, "additionalProperties": False},
    capabilities=("expand",), execution=("subprocess",), determinism="deterministic",
    upstream={"method": "Adapter demonstration only; no research-quality claim"})


def describe():
    return AlgorithmDescriptor(SPEC, "independent_demo:create", "0.1.0")


def create():
    from data_designer.generation.algorithms.adapters.subprocess import SubprocessAlgorithm
    return SubprocessAlgorithm(SPEC, [sys.executable, "-I", str(Path(__file__).with_name("worker.py"))])
