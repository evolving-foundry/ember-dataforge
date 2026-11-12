"""
Test configuration file - pytest configuration
"""
import pytest
import sys
from pathlib import Path

# Add the merged project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Try to import ray; skip tests that require ray if it fails
try:
    import ray
    HAS_RAY = True
except ImportError:
    HAS_RAY = False


def pytest_configure(config):
    """Pytest configuration hook"""
    if not HAS_RAY:
        config.addinivalue_line(
            "markers", "requires_ray: tests that require Ray"
        )


@pytest.fixture(scope="session")
def require_ray():
    """Skip tests if ray is unavailable"""
    if not HAS_RAY:
        pytest.skip("Ray is not installed", allow_module_level=True)
    return True


@pytest.fixture(scope="session", autouse=True)
def init_ray():
    """Initialize Ray (if not already initialized)"""
    if not HAS_RAY:
        yield
        return
    
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, num_cpus=2)
    yield
    # Shut down Ray after tests
    if ray.is_initialized():
        ray.shutdown()


@pytest.fixture
def sample_json_path(tmp_path):
    """Create a sample JSON file"""
    import json
    data = [
        {"id": 1, "text": "Hello world", "category": "A"},
        {"id": 2, "text": "Test data", "category": "B"},
        {"id": 3, "text": "Sample text", "category": "A"}
    ]
    file_path = tmp_path / "sample.json"
    with open(file_path, 'w') as f:
        json.dump(data, f)
    return str(file_path)


@pytest.fixture
def sample_jsonl_path(tmp_path):
    """Create a sample JSONL file"""
    import json
    data = [
        {"id": 1, "text": "Line 1"},
        {"id": 2, "text": "Line 2"},
        {"id": 3, "text": "Line 3"}
    ]
    file_path = tmp_path / "sample.jsonl"
    with open(file_path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')
    return str(file_path)


@pytest.fixture
def sample_csv_path(tmp_path):
    """Create a sample CSV file"""
    import csv
    data = [
        {"id": 1, "name": "Alice", "age": "25"},
        {"id": 2, "name": "Bob", "age": "30"},
        {"id": 3, "name": "Charlie", "age": "35"}
    ]
    file_path = tmp_path / "sample.csv"
    with open(file_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["id", "name", "age"])
        writer.writeheader()
        writer.writerows(data)
    return str(file_path)
