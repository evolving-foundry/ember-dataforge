"""
Test migrated text filter operators
"""
import pytest
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from base.dataset import MultimodalDataset
from curator.tracking.tracking import OperationType

# Try to import ray
try:
    import ray
    HAS_RAY = True
except ImportError:
    HAS_RAY = False
    pytestmark = pytest.mark.skipif(not HAS_RAY, reason="Ray is not installed")


@pytest.fixture
def sample_text_dataset():
    """Create a sample text dataset"""
    if not HAS_RAY:
        pytest.skip("Ray is not installed")
    
    data = [
        {"id": 1, "text": "Hello world, this is a test.", "input_text": "Hello world", "reference_text": "Hello world"},
        {"id": 2, "text": "Test <script>alert('xss')</script>", "input_text": "Test data", "reference_text": "Test example"},
        {"id": 3, "text": "Another sentence here.", "input_text": "", "reference_text": "Reference"},
        {"id": 4, "text": "Short", "input_text": "Short text", "reference_text": "Short"},
    ]
    ds = ray.data.from_items(data)
    return MultimodalDataset(dataset=ds)


class TestBleuSampleFilterOperation:
    """Test BleuSampleFilterOperation"""

    def test_operation_type(self):
        """Test operation type"""
        from curator.operations.filtering import BleuSampleFilterOperation
        assert BleuSampleFilterOperation.operation_type == OperationType.FILTERING

    def test_params_schema(self):
        """Test parameter schema"""
        from curator.operations.filtering import BleuSampleFilterOperation
        assert "params" in BleuSampleFilterOperation.PARAMS_SCHEMA
        assert "n_grams" in BleuSampleFilterOperation.PARAMS_SCHEMA["params"]
        assert "threshold" in BleuSampleFilterOperation.PARAMS_SCHEMA["params"]

    def test_default_params(self):
        """Test default parameters"""
        from curator.operations.filtering import BleuSampleFilterOperation
        op = BleuSampleFilterOperation({})
        assert op.params.get("n_grams") == 1
        assert op.params.get("threshold") == 0.5
        assert op.params.get("reference_key") == "reference_text"
        assert op.params.get("text_key") == "input_text"


class TestBlocklistFilterOperation:
    """Test BlocklistFilterOperation"""

    def test_operation_type(self):
        """Test operation type"""
        from curator.operations.filtering import BlocklistFilterOperation
        assert BlocklistFilterOperation.operation_type == OperationType.FILTERING

    def test_params_schema(self):
        """Test parameter schema"""
        from curator.operations.filtering import BlocklistFilterOperation
        assert "params" in BlocklistFilterOperation.PARAMS_SCHEMA
        assert "blocklist" in BlocklistFilterOperation.PARAMS_SCHEMA["params"]
        assert "max_hits" in BlocklistFilterOperation.PARAMS_SCHEMA["params"]

    def test_filter_with_blocklist(self, sample_text_dataset):
        """Test blocklist filtering"""
        from curator.operations.filtering import BlocklistFilterOperation
        
        op = BlocklistFilterOperation({
            "blocklist": ["script", "xss"],
            "text_key": "text",
            "max_hits": 0
        })
        
        result = op.execute(sample_text_dataset)
        assert isinstance(result, MultimodalDataset)
        # Records containing script and xss should be filtered out

    def test_case_insensitive(self):
        """Test case insensitivity"""
        from curator.operations.filtering import BlocklistFilterOperation
        
        op = BlocklistFilterOperation({
            "blocklist": ["TEST"],
            "case_insensitive": True,
            "text_key": "text"
        })
        
        assert op.params.get("case_insensitive") == True


class TestCiderSampleFilterOperation:
    """Test CiderSampleFilterOperation"""

    def test_operation_type(self):
        """Test operation type"""
        from curator.operations.filtering import CiderSampleFilterOperation
        assert CiderSampleFilterOperation.operation_type == OperationType.FILTERING

    def test_params_schema(self):
        """Test parameter schema"""
        from curator.operations.filtering import CiderSampleFilterOperation
        assert "params" in CiderSampleFilterOperation.PARAMS_SCHEMA
        assert "threshold" in CiderSampleFilterOperation.PARAMS_SCHEMA["params"]
        assert "n" in CiderSampleFilterOperation.PARAMS_SCHEMA["params"]

    def test_default_params(self):
        """Test default parameters"""
        from curator.operations.filtering import CiderSampleFilterOperation
        op = CiderSampleFilterOperation({})
        assert op.params.get("threshold") == 0.5
        assert op.params.get("n") == 4


class TestMeanWordLengthFilterOperation:
    """Test MeanWordLengthFilterOperation"""

    def test_operation_type(self):
        """Test operation type"""
        from curator.operations.filtering import MeanWordLengthFilterOperation
        assert MeanWordLengthFilterOperation.operation_type == OperationType.FILTERING

    def test_params_schema(self):
        """Test parameter schema"""
        from curator.operations.filtering import MeanWordLengthFilterOperation
        assert "params" in MeanWordLengthFilterOperation.PARAMS_SCHEMA
        assert "min_len" in MeanWordLengthFilterOperation.PARAMS_SCHEMA["params"]
        assert "max_len" in MeanWordLengthFilterOperation.PARAMS_SCHEMA["params"]
        assert "lang" in MeanWordLengthFilterOperation.PARAMS_SCHEMA["params"]

    def test_default_params(self):
        """Test default parameters"""
        from curator.operations.filtering import MeanWordLengthFilterOperation
        op = MeanWordLengthFilterOperation({})
        assert op.params.get("min_len") == 3
        assert op.params.get("max_len") == 10
        assert op.params.get("lang") == "en"


class TestNgramSampleFilterOperation:
    """Test NgramSampleFilterOperation"""

    def test_operation_type(self):
        """Test operation type"""
        from curator.operations.filtering import NgramSampleFilterOperation
        assert NgramSampleFilterOperation.operation_type == OperationType.FILTERING

    def test_params_schema(self):
        """Test parameter schema"""
        from curator.operations.filtering import NgramSampleFilterOperation
        assert "params" in NgramSampleFilterOperation.PARAMS_SCHEMA
        assert "n_gram" in NgramSampleFilterOperation.PARAMS_SCHEMA["params"]
        assert "threshold" in NgramSampleFilterOperation.PARAMS_SCHEMA["params"]

    def test_default_params(self):
        """Test default parameters"""
        from curator.operations.filtering import NgramSampleFilterOperation
        op = NgramSampleFilterOperation({})
        assert op.params.get("n_gram") == 2
        assert op.params.get("threshold") == 0.5


class TestSpecialCharFilterOperation:
    """Test SpecialCharFilterOperation"""

    def test_operation_type(self):
        """Test operation type"""
        from curator.operations.filtering import SpecialCharFilterOperation
        assert SpecialCharFilterOperation.operation_type == OperationType.FILTERING

    def test_params_schema(self):
        """Test parameter schema"""
        from curator.operations.filtering import SpecialCharFilterOperation
        assert "params" in SpecialCharFilterOperation.PARAMS_SCHEMA
        assert "special_chars" in SpecialCharFilterOperation.PARAMS_SCHEMA["params"]
        assert "text_key" in SpecialCharFilterOperation.PARAMS_SCHEMA["params"]

    def test_default_params(self):
        """Test default parameters"""
        from curator.operations.filtering import SpecialCharFilterOperation
        op = SpecialCharFilterOperation({})
        assert op.params.get("text_key") == "text"
        assert "<" in op.params.get("special_chars", [])

    def test_filter_with_special_chars(self, sample_text_dataset):
        """Test special character filtering"""
        from curator.operations.filtering import SpecialCharFilterOperation
        
        op = SpecialCharFilterOperation({
            "special_chars": ["<", ">"],
            "text_key": "text"
        })
        
        result = op.execute(sample_text_dataset)
        assert isinstance(result, MultimodalDataset)
