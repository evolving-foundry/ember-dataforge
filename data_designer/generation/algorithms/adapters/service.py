"""Service boundary: explicit identity, version and idempotency; no automatic retry."""
from __future__ import annotations

from dataclasses import asdict

from data_designer.contracts.algorithms import AlgorithmResult
from data_designer.contracts.datasets import RecordEnvelope
from data_designer.contracts.provenance import content_digest


class ServiceAlgorithm:
    def __init__(self, spec, provider, *, timeout_seconds=60):
        if timeout_seconds <= 0:
            raise ValueError("service timeout must be positive")
        self.spec, self.provider, self.timeout_seconds = spec, provider, timeout_seconds

    def describe(self):
        return self.spec

    def run(self, request, context):
        context.check_cancelled()
        provider = context.provider(self.provider)
        identity = provider.describe()
        if identity.get("algorithm") != self.spec.id:
            raise ValueError("service algorithm identity mismatch")
        version = identity.get("implementation_version")
        if version is not None and version != self.spec.implementation_version:
            raise ValueError("service implementation version mismatch")
        body = {"protocol_version": "dataforge-service-v1", "run_id": request.run_id,
                "algorithm": self.spec.id, "config": dict(request.config), "seed": context.seed,
                "records": [asdict(record) for record in request.records],
                "target_new_records": request.target_new_records, "max_attempts": request.max_attempts}
        body["idempotency_key"] = content_digest(body)
        # Provider owns transport timeout/cancellation. A timeout may leave execution unknown.
        response = provider.generate(body, timeout_seconds=self.timeout_seconds, cancelled=context.cancelled)
        context.check_cancelled()
        if response.get("protocol_version") != body["protocol_version"] or response.get("run_id") != request.run_id:
            raise ValueError("service response identity mismatch")
        if response.get("status") != "completed":
            raise RuntimeError("service generation did not complete")
        records = [RecordEnvelope(**{**row, "parents": tuple(row.get("parents", ()))})
                   for row in response["records"]]
        return AlgorithmResult(records, response.get("rejected", {}), {
            "adapter": "service", "provider": self.provider,
            "provider_version": version or "unknown", "cache_reusable": version is not None,
            "determinism": "external", "automatic_retries": 0})
