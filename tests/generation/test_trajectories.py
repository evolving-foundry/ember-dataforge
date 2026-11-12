from data_designer.api import generate
from data_designer.generation.algorithms.registry import AlgorithmRegistry


def config(algorithm, params=None):
    return {"recipe": "algorithm", "recipe_config": {"algorithms": {"main": {
        "id": algorithm, "params": params or {}}}}}


def test_cleaning_omissions_and_grounded_failure_lesson():
    row = {"trajectory_id": "t1", "task": "Contact admin@example.test",
           "steps": [{"index": 0, "action": "run", "observation": "<think>hidden</think>AssertionError admin@example.test", "error": True}],
           "outcome": {"status": "failure"}, "meta": {"agent_version": "fixture", "model_version": "fixture"}}
    cleaned = generate([row], config("native.trajectory_clean"))
    record = cleaned.records[0]
    assert "hidden" not in str(record.payload)
    assert "admin@example.test" not in str(record.payload)
    assert record.metadata["omissions"]["total_omissions"] >= 2
    lesson = generate([record], {**config("native.trajectory_lesson"), "input": {"schema": "trajectory-v1"}}).records[0]
    evidence = lesson.payload["evidence"]
    assert evidence["quote"] in record.payload["steps"][0]["observation"]
    assert lesson.parents == (record.id,)
