"""
Migrated operator integration tests
Test the migrated operators in Pipeline usage
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))

# Try to import ray
try:
    import ray
    HAS_RAY = True
except ImportError:
    HAS_RAY = False
    pytestmark = pytest.mark.skipif(not HAS_RAY, reason="Ray is not installed")


@pytest.fixture
def sample_dataset():
    """Create a sample dataset"""
    if not HAS_RAY:
        pytest.skip("Ray is not installed")
    
    from base.dataset import MultimodalDataset
    
    data = [
        {"id": 1, "text": "Hello world, this is a test sentence.", "category": "A"},
        {"id": 2, "text": "Another example with special chars: <>", "category": "B"},
        {"id": 3, "text": "Short text.", "category": "A"},
        {"id": 4, "text": "A very long sentence with many words to test filtering.", "category": "C"},
    ]
    ds = ray.data.from_items(data)
    return MultimodalDataset(dataset=ds)


class TestMigratedOperatorsInPipeline:
    """Test integration of migrated operators in Pipeline"""

    def test_pipeline_can_create_operations(self):
        """Test that Pipeline can create all migrated operations"""
        from curator.core.pipeline import OperationRegistry
        
        operations_to_test = [
            ('blocklist_filter', {'blocklist': ['test'], 'text_key': 'text'}),
            ('special_char_filter', {'special_chars': ['<', '>'], 'text_key': 'text'}),
            ('text_normalization', {'text_key': 'text'}),
            ('remove_stopwords', {'text_key': 'text'}),
            ('image_aspect_ratio_filter', {'image_key': 'image_path'}),
            ('image_size_filter', {'image_key': 'image_path'}),
            ('video_qae_materialization', {'materialization_mode': 'JOINT_SYNTHESIS', 'use_llm_client': False}),
            ('video_consistency_revision', {'revision_mode': 'FIELD_ALIGN'}),
            ('video_quality_screening', {'quality_mode': 'RULE_FILTER'}),
        ]
        
        for op_name, params in operations_to_test:
            op = OperationRegistry.create_operation(op_name, params)
            assert op is not None, f"Unable to create operation: {op_name}"
            print(f"  ✓ {op_name} created successfully")

    def test_filter_operation_execution(self, sample_dataset):
        """Test Filter operation execution"""
        from curator.operations.filtering import SpecialCharFilterOperation
        
        op = SpecialCharFilterOperation({
            'special_chars': ['<', '>'],
            'text_key': 'text'
        })
        
        result = op.execute(sample_dataset)
        from base.dataset import MultimodalDataset
        assert isinstance(result, MultimodalDataset)
        # Records containing special characters should be filtered

    def test_augmentation_operation_execution(self, sample_dataset):
        """Test Augmentation operation execution"""
        from curator.operations.augmentation import TextNormalizationOperation
        
        op = TextNormalizationOperation({
            'text_key': 'text'
        })
        
        result = op.execute(sample_dataset)
        from base.dataset import MultimodalDataset
        assert isinstance(result, MultimodalDataset)

    def test_operation_params_accessible(self):
        """Test that operator parameters are accessible"""
        from curator.operations.filtering import BlocklistFilterOperation
        
        op = BlocklistFilterOperation({
            'blocklist': ['spam', 'test'],
            'text_key': 'content',
            'max_hits': 1
        })
        
        params = op.get_params()
        assert params['blocklist'] == ['spam', 'test']
        assert params['text_key'] == 'content'
        assert params['max_hits'] == 1


class TestMigratedOperatorCompatibility:
    """Test compatibility of migrated operators"""

    def test_all_migrated_ops_have_operation_type(self):
        """Test that all migrated operators have operation_type"""
        from curator.core.pipeline import OperationRegistry
        from curator.tracking.tracking import OperationType
        
        migrated_ops = [
            'bleu_sample_filter', 'blocklist_filter', 'cider_sample_filter',
            'mean_word_length_filter', 'ngram_sample_filter', 'special_char_filter',
            'image_aspect_ratio_filter', 'image_nsfw_filter', 'image_size_filter',
            'random_noun', 'remove_repetitions_punctuation', 'remove_stopwords',
            'spelling_correction', 'stemming_lemmatization', 'text_normalization',
            'image_canny', 'image_captioning', 'image_random_crop',
            'image_random_flip', 'video_resize_aspect_ratio'
        ]
        
        for op_name in migrated_ops:
            op_class = OperationRegistry.get_operation_class(op_name)
            assert op_class is not None, f"{op_name} is not registered"
            assert op_class.operation_type is not None, f"{op_name} has no operation_type"
            assert isinstance(op_class.operation_type, OperationType), f"{op_name} has an invalid operation_type"

    def test_all_migrated_ops_have_params_schema(self):
        """Test that all migrated operators have PARAMS_SCHEMA"""
        from curator.core.pipeline import OperationRegistry
        
        migrated_ops = [
            'bleu_sample_filter', 'blocklist_filter', 'cider_sample_filter',
            'mean_word_length_filter', 'ngram_sample_filter', 'special_char_filter',
            'image_aspect_ratio_filter', 'image_nsfw_filter', 'image_size_filter',
            'random_noun', 'remove_repetitions_punctuation', 'remove_stopwords',
            'spelling_correction', 'stemming_lemmatization', 'text_normalization',
            'image_canny', 'image_captioning', 'image_random_crop',
            'image_random_flip', 'video_resize_aspect_ratio'
        ]
        
        for op_name in migrated_ops:
            op_class = OperationRegistry.get_operation_class(op_name)
            assert hasattr(op_class, 'PARAMS_SCHEMA'), f"{op_name} has no PARAMS_SCHEMA"
            assert 'params' in op_class.PARAMS_SCHEMA, f"{op_name}.PARAMS_SCHEMA has an invalid format"
