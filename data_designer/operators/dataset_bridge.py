"""Curator dataset boundary for the generation operator."""

from __future__ import annotations


def rows_from_dataset(dataset):
    """Iterate without a second full take_all() copy in the operator."""
    return dataset.dataset.iter_rows()


def wrap_result(result, source):
    import ray
    from base.dataset import MultimodalDataset

    rows = [dict(record.payload) for record in result.records]
    return MultimodalDataset(
        ray.data.from_items(rows),
        modalities=source.modalities,
        validation_config=source.validation_config,
        **source.kwargs,
    )
