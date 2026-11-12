"""
Test migrated text augmentation operators
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
        {"id": 1, "text": "Hello world, this is a test.", "output_text": ""},
        {"id": 2, "text": "The quick brown fox jumps!!! Over the lazy dog...", "output_text": ""},
        {"id": 3, "text": "I am running and testing.", "output_text": ""},
    ]
    ds = ray.data.from_items(data)
    return MultimodalDataset(dataset=ds)


class TestRandomNounAugmentationOperation:
    """Test RandomNounAugmentationOperation"""

    def test_operation_type(self):
        """Test operation type"""
        from curator.operations.augmentation import RandomNounAugmentationOperation
        assert RandomNounAugmentationOperation.operation_type == OperationType.AUGMENTATION

    def test_params_schema(self):
        """Test parameter schema"""
        from curator.operations.augmentation import RandomNounAugmentationOperation
        assert "params" in RandomNounAugmentationOperation.PARAMS_SCHEMA
        assert "text_key" in RandomNounAugmentationOperation.PARAMS_SCHEMA["params"]
        assert "replacement_prob" in RandomNounAugmentationOperation.PARAMS_SCHEMA["params"]

    def test_default_params(self):
        """Test default parameters"""
        from curator.operations.augmentation import RandomNounAugmentationOperation
        op = RandomNounAugmentationOperation({})
        assert op.params.get("text_key") == "text"
        assert op.params.get("replacement_prob") == 0.5


class TestRemoveRepetitionsPunctuationOperation:
    """Test RemoveRepetitionsPunctuationOperation"""

    def test_operation_type(self):
        """Test operation type"""
        from curator.operations.augmentation import RemoveRepetitionsPunctuationOperation
        assert RemoveRepetitionsPunctuationOperation.operation_type == OperationType.AUGMENTATION

    def test_params_schema(self):
        """Test parameter schema"""
        from curator.operations.augmentation import RemoveRepetitionsPunctuationOperation
        assert "params" in RemoveRepetitionsPunctuationOperation.PARAMS_SCHEMA
        assert "text_key" in RemoveRepetitionsPunctuationOperation.PARAMS_SCHEMA["params"]
        assert "keep_repeat_char" in RemoveRepetitionsPunctuationOperation.PARAMS_SCHEMA["params"]

    def test_default_params(self):
        """Test default parameters"""
        from curator.operations.augmentation import RemoveRepetitionsPunctuationOperation
        op = RemoveRepetitionsPunctuationOperation({})
        assert op.params.get("text_key") == "text"
        assert op.params.get("keep_repeat_char") == 2

    def test_remove_repetitions(self, sample_text_dataset):
        """Test removal of repeated punctuation"""
        from curator.operations.augmentation import RemoveRepetitionsPunctuationOperation
        
        op = RemoveRepetitionsPunctuationOperation({
            "text_key": "text"
        })
        
        result = op.execute(sample_text_dataset)
        assert isinstance(result, MultimodalDataset)


class TestRemoveStopwordsOperation:
    """Test RemoveStopwordsOperation"""

    def test_operation_type(self):
        """Test operation type"""
        from curator.operations.augmentation import RemoveStopwordsOperation
        assert RemoveStopwordsOperation.operation_type == OperationType.AUGMENTATION

    def test_params_schema(self):
        """Test parameter schema"""
        from curator.operations.augmentation import RemoveStopwordsOperation
        assert "params" in RemoveStopwordsOperation.PARAMS_SCHEMA
        assert "text_key" in RemoveStopwordsOperation.PARAMS_SCHEMA["params"]
        assert "custom_stopwords" in RemoveStopwordsOperation.PARAMS_SCHEMA["params"]

    def test_default_params(self):
        """Test default parameters"""
        from curator.operations.augmentation import RemoveStopwordsOperation
        op = RemoveStopwordsOperation({})
        assert op.params.get("text_key") == "text"
        assert op.params.get("custom_stopwords") == []


class TestSpellingCorrectionOperation:
    """Test SpellingCorrectionOperation"""

    def test_operation_type(self):
        """Test operation type"""
        from curator.operations.augmentation import SpellingCorrectionOperation
        assert SpellingCorrectionOperation.operation_type == OperationType.AUGMENTATION

    def test_params_schema(self):
        """Test parameter schema"""
        from curator.operations.augmentation import SpellingCorrectionOperation
        assert "params" in SpellingCorrectionOperation.PARAMS_SCHEMA
        assert "text_key" in SpellingCorrectionOperation.PARAMS_SCHEMA["params"]
        assert "max_edit_distance" in SpellingCorrectionOperation.PARAMS_SCHEMA["params"]

    def test_default_params(self):
        """Test default parameters"""
        from curator.operations.augmentation import SpellingCorrectionOperation
        op = SpellingCorrectionOperation({})
        assert op.params.get("text_key") == "text"
        assert op.params.get("max_edit_distance") == 2


class TestStemmingLemmatizationOperation:
    """Test StemmingLemmatizationOperation"""

    def test_operation_type(self):
        """Test operation type"""
        from curator.operations.augmentation import StemmingLemmatizationOperation
        assert StemmingLemmatizationOperation.operation_type == OperationType.AUGMENTATION

    def test_params_schema(self):
        """Test parameter schema"""
        from curator.operations.augmentation import StemmingLemmatizationOperation
        assert "params" in StemmingLemmatizationOperation.PARAMS_SCHEMA
        assert "text_key" in StemmingLemmatizationOperation.PARAMS_SCHEMA["params"]
        assert "mode" in StemmingLemmatizationOperation.PARAMS_SCHEMA["params"]

    def test_default_params(self):
        """Test default parameters"""
        from curator.operations.augmentation import StemmingLemmatizationOperation
        op = StemmingLemmatizationOperation({})
        assert op.params.get("text_key") == "text"
        assert op.params.get("mode") == "stem"

    def test_stem_mode(self, sample_text_dataset):
        """Test stemming mode"""
        from curator.operations.augmentation import StemmingLemmatizationOperation
        
        op = StemmingLemmatizationOperation({
            "text_key": "text",
            "mode": "stem"
        })
        
        result = op.execute(sample_text_dataset)
        assert isinstance(result, MultimodalDataset)


class TestTextNormalizationOperation:
    """Test TextNormalizationOperation"""

    def test_operation_type(self):
        """Test operation type"""
        from curator.operations.augmentation import TextNormalizationOperation
        assert TextNormalizationOperation.operation_type == OperationType.AUGMENTATION

    def test_params_schema(self):
        """Test parameter schema"""
        from curator.operations.augmentation import TextNormalizationOperation
        assert "params" in TextNormalizationOperation.PARAMS_SCHEMA
        assert "text_key" in TextNormalizationOperation.PARAMS_SCHEMA["params"]

    def test_default_params(self):
        """Test default parameters"""
        from curator.operations.augmentation import TextNormalizationOperation
        op = TextNormalizationOperation({})
        assert op.params.get("text_key") == "text"

    def test_normalize_date(self):
        """Test date normalization"""
        from curator.operations.augmentation import TextNormalizationOperation
        
        if not HAS_RAY:
            pytest.skip("Ray is not installed")
        
        data = [
            {"id": 1, "text": "Meeting on 12/25/2023 and January 15, 2024. Price: $50"},
        ]
        ds = ray.data.from_items(data)
        dataset = MultimodalDataset(dataset=ds)
        
        op = TextNormalizationOperation({"text_key": "text"})
        result = op.execute(dataset)
        
        assert isinstance(result, MultimodalDataset)
