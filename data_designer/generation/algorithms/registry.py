from __future__ import annotations

from dataclasses import replace
from importlib import import_module, metadata

from data_designer.contracts.algorithms import AlgorithmDescriptor


def load_symbol(reference):
    module, name = reference.split(":", 1)
    return getattr(import_module(module), name)


class AlgorithmRegistry:
    def __init__(self, *, discover=True):
        self._descriptors = {}
        from .builtin.descriptors import descriptors
        for descriptor in descriptors():
            self.register(descriptor)
        if discover:
            points = metadata.entry_points()
            points = points.select(group="data_designer.algorithms") if hasattr(points, "select") else points.get("data_designer.algorithms", ())
            for point in sorted(points, key=lambda p: (p.name, p.value)):
                descriptor = point.load()()
                if not isinstance(descriptor, AlgorithmDescriptor):
                    raise TypeError(f"algorithm descriptor required: {point.name}")
                if descriptor.spec.id != point.name:
                    raise ValueError(f"algorithm entry point id mismatch: {point.name}")
                distribution = getattr(point, "dist", None)
                if distribution is not None:
                    descriptor = replace(descriptor, package_version=distribution.version)
                self.register(descriptor)

    def register(self, descriptor):
        if not isinstance(descriptor, AlgorithmDescriptor):
            raise TypeError("AlgorithmDescriptor required")
        key = descriptor.spec.id
        if key in self._descriptors:
            raise ValueError(f"duplicate algorithm id: {key}")
        self._descriptors[key] = descriptor

    def describe(self, key):
        try:
            return self._descriptors[key]
        except KeyError as error:
            raise ValueError(f"unknown algorithm: {key}") from error

    def list(self):
        return tuple(self._descriptors[key] for key in sorted(self._descriptors))

    def create(self, key):
        descriptor = self.describe(key)
        try:
            algorithm = load_symbol(descriptor.factory)()
        except ImportError as error:
            raise RuntimeError(f"algorithm {key} dependency unavailable: {error.name}") from error
        if algorithm.describe() != descriptor.spec:
            raise ValueError(f"algorithm descriptor/implementation mismatch: {key}")
        return algorithm
