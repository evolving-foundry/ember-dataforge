"""DatasetRegistry tests — single namespace contract (§14/§41.3)."""

import pytest

from curator.runtime.dataset_registry import DatasetRegistry


class FakeDataset:
    def __init__(self, rows=3, modalities=None):
        self.dataset = type("Inner", (), {"count": staticmethod(lambda: rows)})()
        self.modalities = modalities or ["text"]

    def get_stats(self):
        return {"fake": True}


def test_register_assigns_reference_id():
    created = []
    reg = DatasetRegistry(on_created=lambda i, b: created.append((i, b)))
    ds_id = reg.register(FakeDataset(), backend="ray")
    assert ds_id.startswith("ds-")
    assert created == [(ds_id, "ray")]
    assert reg.exists(ds_id)


def test_register_honors_explicit_id_for_shared_namespace():
    """§41.3 Constraint 1: native path and ZMQ path share one namespace."""
    reg = DatasetRegistry()
    reg.register(FakeDataset(), dataset_id="ds-native-001", backend="parquet")
    assert reg.exists("ds-native-001")


def test_inspect_returns_metadata():
    reg = DatasetRegistry()
    ds_id = reg.register(FakeDataset(rows=7, modalities=["text", "image"]))
    meta = reg.inspect(ds_id)
    assert meta["datasetId"] == ds_id
    assert meta["backend"] == "ray"
    assert meta["numRows"] == 7
    assert meta["modalities"] == ["text", "image"]
    assert meta["extra"] == {"fake": True}


def test_inspect_unknown_dataset_raises():
    reg = DatasetRegistry()
    with pytest.raises(KeyError):
        reg.inspect("ds-missing")


def test_release_fires_event_and_drops_reference():
    released = []
    reg = DatasetRegistry(on_released=lambda i, b, r: released.append((i, b, r)))
    ds_id = reg.register(FakeDataset())
    assert reg.release(ds_id, reason="test") is True
    assert not reg.exists(ds_id)
    assert released == [(ds_id, "ray", "test")]
    assert reg.release(ds_id) is False
