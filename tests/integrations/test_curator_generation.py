from __future__ import annotations

import pytest

ray = pytest.importorskip("ray")


def test_operator_shares_sdk_and_retains_metadata():
    from base.dataset import MultimodalDataset
    from data_designer.api import generate
    from data_designer.operators import SyntheticDataGenerationOperation
    rows = [{"id": "a", "task_text": "Normalize task"}]
    dataset = MultimodalDataset(ray.data.from_items(rows), modalities=["text"], custom="metadata")
    op = SyntheticDataGenerationOperation({"recipe": "agent_task"})
    result = op.execute(dataset)
    assert result.modalities == dataset.modalities
    assert result.validation_config is dataset.validation_config
    assert result.kwargs == dataset.kwargs
    assert result.dataset.take_all() == [r.payload for r in generate(rows, {"recipe": "agent_task"}).records]
    with pytest.raises(ValueError, match="Additional properties"):
        SyntheticDataGenerationOperation({"recipe": "agent_task", "typo": 1})
