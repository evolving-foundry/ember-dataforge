"""Lightweight, host-independent data and algorithm contracts."""

from .datasets import DatasetPackage, DatasetRef, RecordEnvelope
from .algorithms import AlgorithmRequest, AlgorithmResult, AlgorithmSpec, AlgorithmDescriptor

__all__ = ["DatasetPackage", "DatasetRef", "RecordEnvelope", "AlgorithmRequest",
           "AlgorithmResult", "AlgorithmSpec", "AlgorithmDescriptor"]
