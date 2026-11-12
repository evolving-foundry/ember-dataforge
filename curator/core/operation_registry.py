"""
Operation registry module.

Provides the OperationRegistry class and the register_operation decorator,
used to manage and register Pipeline operation classes.
"""
from typing import Any, Dict, Optional
import threading

from .operation import Operation


class OperationRegistry:
    """Operation registry — lazily initialized, distinguishing implemented capabilities from placeholder operators."""

    _operations: Dict[str, type] = {}
    _initialized: bool = False
    _init_lock = threading.Lock()

    @classmethod
    def register(cls, name: str, operation_class: type) -> None:
        cls._operations[name] = operation_class

    @classmethod
    def _ensure_initialized(cls) -> None:
        """Lazy loading: register all default operators on first call and ensure initialization runs only once."""
        if cls._initialized:
            return
        with cls._init_lock:
            if cls._initialized:
                return
        from ..operations.loading import (
            TextLoadingOperation, ImageLoadingOperation, VideoLoadingOperation,
            AudioLoadingOperation, MultiModalLoadingOperation,
            DocumentParsingOperation, ImageOcrOperation,
        )
        from ..operations.saving import (
            TextSavingOperation, ImageSavingOperation, VideoSavingOperation,
            AudioSavingOperation, MultiModalSavingOperation,
        )
        from ..operations.cleaning import (
            InvalidDataRemovalOperation, CorruptedDataFixOperation,
            FormatStandardizationOperation,
        )
        from ..operations.filtering import (
            QualityBasedFilterOperation, ModelBasedFilterOperation, RuleBasedFilterOperation,
            BleuSampleFilterOperation, BlocklistFilterOperation, CiderSampleFilterOperation,
            MeanWordLengthFilterOperation, NgramSampleFilterOperation,
            SpecialCharFilterOperation, ImageAspectRatioFilterOperation,
            ImageNsfwFilterOperation, ImageSizeFilterOperation, VideoContextOrchestrationOperation,
            VideoQualityScreeningOperation,
        )
        from ..operations.formatting import (
            UniformFormatConversionOperation, SchemaAlignmentOperation,
            MultiSourceNormalizationOperation,
        )
        from ..operations.deduplication import (
            ExactDuplicateRemovalOperation, SimilarityBasedDedupOperation,
            FuzzyMatchingDedupOperation,
        )
        from ..operations.selection import (
            FeatureBasedSelectionOperation, SemanticSimilaritySelectionOperation,
            MetadataBasedSelectionOperation,
        )
        from ..operations.aggregation.nested_aggregation_operator import NestedAggregationOperation
        from ..operations.aggregation.entity_attribute_aggregation_operator import (
            EntityAttributeAggregationOperation,
        )
        from ..operations.aggregation.meta_tags_aggregation_operator import MetaTagsAggregationOperation
        from ..operations.augmentation.image_random_flip_operation import ImageRandomFlipOperation
        from ..operations.augmentation.image_augmentation_operator import ImageAugmentationOperation
        from ..operations.augmentation.audio_augmentation_operator import AudioAugmentationOperation
        from ..operations.augmentation.video_augmentation_operator import VideoAugmentationOperation
        from ..operations.augmentation.random_noun_operation import RandomNounAugmentationOperation
        from ..operations.augmentation.remove_repetitions_punctuation_operation import (
            RemoveRepetitionsPunctuationOperation,
        )
        from ..operations.augmentation.remove_stopwords_operation import RemoveStopwordsOperation
        from ..operations.augmentation.spelling_correction_operation import SpellingCorrectionOperation
        from ..operations.augmentation.stemming_lemmatization_operation import (
            StemmingLemmatizationOperation,
        )
        from ..operations.augmentation.text_normalization_operation import TextNormalizationOperation
        from ..operations.augmentation.image_canny_operation import ImageCannyAugmentationOperation
        from ..operations.augmentation.image_captioning_operation import ImageCaptioningOperation
        from ..operations.augmentation.image_random_crop_operation import ImageRandomCropOperation
        from ..operations.augmentation.video_partition_operation import VideoPartitionOperation
        from ..operations.augmentation.video_sampling_operation import VideoSamplingOperation
        from ..operations.augmentation.video_multigranularity_description_operation import (
            VideoMultigranularityDescriptionOperation,
        )
        from ..operations.augmentation.video_temporal_structure_operation import (
            VideoTemporalStructureOperation,
        )
        from ..operations.augmentation.video_scaffold_construction_operation import (
            VideoScaffoldConstructionOperation,
        )
        from ..operations.augmentation.video_text_evidence_operation import (
            VideoTextEvidenceOperation,
        )
        from ..operations.augmentation.video_qae_materialization_operation import (
            VideoQaeMaterializationOperation,
        )
        from ..operations.augmentation.video_consistency_revision_operation import (
            VideoConsistencyRevisionOperation,
        )
        from ..operations.augmentation.video_resize_aspect_ratio_operation import VideoResizeAspectRatioOperation
        from ..operations.augmentation.text_augmentation_operator import TextAugmentationOperation
        from ..operations.common.fusion_operator import FusionOperation
        from ..operations.common.helper_function_operator import HelperFunctionOperation
        from ..operations.common.special_characters_operator import SpecialCharactersOperation
        from ..operations.grouping.key_value_grouping_operator import KeyValueGroupingOperation
        from ..operations.grouping.naive_grouping_operator import NaiveGroupingOperation
        from ..operations.grouping.naive_reverse_grouping_operator import NaiveReverseGroupingOperation
        from ..operations.mapping.text_chunk_mapping_operator import TextChunkMappingOperation
        from ..operations.mapping.remove_header_mapping_operator import RemoveHeaderMappingOperation
        from ..operations.mapping.punctuation_normalization_mapping_operator import (
            PunctuationNormalizationMappingOperation,
        )
        from ..operations.mapping.whitespace_normalization_mapping_operator import (
            WhitespaceNormalizationMappingOperation,
        )
        from ..operations.mapping.token_count_mapping_operator import TokenCountMappingOperation

        operations_map = {
            "text_loading": TextLoadingOperation,
            "image_loading": ImageLoadingOperation,
            "video_loading": VideoLoadingOperation,
            "audio_loading": AudioLoadingOperation,
            "multimodal_loading": MultiModalLoadingOperation,
            "document_parsing": DocumentParsingOperation,
            "image_ocr": ImageOcrOperation,
            "text_saving": TextSavingOperation,
            "image_saving": ImageSavingOperation,
            "video_saving": VideoSavingOperation,
            "audio_saving": AudioSavingOperation,
            "multimodal_saving": MultiModalSavingOperation,
            "invalid_data_removal": InvalidDataRemovalOperation,
            "corrupted_data_fix": CorruptedDataFixOperation,
            "format_standardization": FormatStandardizationOperation,
            "quality_based_filter": QualityBasedFilterOperation,
            "model_based_filter": ModelBasedFilterOperation,
            "rule_based_filter": RuleBasedFilterOperation,
            "bleu_sample_filter": BleuSampleFilterOperation,
            "blocklist_filter": BlocklistFilterOperation,
            "cider_sample_filter": CiderSampleFilterOperation,
            "mean_word_length_filter": MeanWordLengthFilterOperation,
            "ngram_sample_filter": NgramSampleFilterOperation,
            "special_char_filter": SpecialCharFilterOperation,
            "image_aspect_ratio_filter": ImageAspectRatioFilterOperation,
            "image_nsfw_filter": ImageNsfwFilterOperation,
            "image_size_filter": ImageSizeFilterOperation,
            "video_context_orchestration": VideoContextOrchestrationOperation,
            "video_quality_screening": VideoQualityScreeningOperation,
            "text_augmentation": TextAugmentationOperation,
            "image_augmentation": ImageAugmentationOperation,
            "audio_augmentation": AudioAugmentationOperation,
            "video_augmentation": VideoAugmentationOperation,
            "random_noun": RandomNounAugmentationOperation,
            "remove_repetitions_punctuation": RemoveRepetitionsPunctuationOperation,
            "remove_stopwords": RemoveStopwordsOperation,
            "spelling_correction": SpellingCorrectionOperation,
            "stemming_lemmatization": StemmingLemmatizationOperation,
            "text_normalization": TextNormalizationOperation,
            "image_canny": ImageCannyAugmentationOperation,
            "image_captioning": ImageCaptioningOperation,
            "image_random_crop": ImageRandomCropOperation,
            "image_random_flip": ImageRandomFlipOperation,
            "video_partition": VideoPartitionOperation,
            "video_sampling": VideoSamplingOperation,
            "video_multigranularity_description": VideoMultigranularityDescriptionOperation,
            "video_temporal_structure": VideoTemporalStructureOperation,
            "video_scaffold_construction": VideoScaffoldConstructionOperation,
            "video_text_evidence": VideoTextEvidenceOperation,
            "video_qae_materialization": VideoQaeMaterializationOperation,
            "video_consistency_revision": VideoConsistencyRevisionOperation,
            "video_resize_aspect_ratio": VideoResizeAspectRatioOperation,
            "uniform_format_conversion": UniformFormatConversionOperation,
            "schema_alignment": SchemaAlignmentOperation,
            "multi_source_normalization": MultiSourceNormalizationOperation,
            "exact_duplicate_removal": ExactDuplicateRemovalOperation,
            "similarity_based_dedup": SimilarityBasedDedupOperation,
            "fuzzy_matching_dedup": FuzzyMatchingDedupOperation,
            "feature_based_selection": FeatureBasedSelectionOperation,
            "semantic_similarity_selection": SemanticSimilaritySelectionOperation,
            "metadata_based_selection": MetadataBasedSelectionOperation,
            "nested_aggregation": NestedAggregationOperation,
            "entity_attribute_aggregation": EntityAttributeAggregationOperation,
            "meta_tags_aggregation": MetaTagsAggregationOperation,
            "key_value_grouping": KeyValueGroupingOperation,
            "naive_grouping": NaiveGroupingOperation,
            "naive_reverse_grouping": NaiveReverseGroupingOperation,
            "text_chunk_mapping": TextChunkMappingOperation,
            "remove_header_mapping": RemoveHeaderMappingOperation,
            "punctuation_normalization_mapping": PunctuationNormalizationMappingOperation,
            "whitespace_normalization_mapping": WhitespaceNormalizationMappingOperation,
            "token_count_mapping": TokenCountMappingOperation,
            "helper_function": HelperFunctionOperation,
            "special_characters": SpecialCharactersOperation,
            "fusion": FusionOperation,
        }

        for name_, op_cls_ in operations_map.items():
            cls.register(name_, op_cls_)

        # Automatically discover third-party plugins
        try:
            from ..plugins.plugin_registry import PluginRegistry
            PluginRegistry().auto_discover()
        except Exception:
            pass

        cls._initialized = True

    _generation_operator_names = frozenset({"synthetic_data_generation"})

    @classmethod
    def create_operation(cls, name: str, params: Optional[Dict[str, Any]] = None) -> Operation:
        cls._ensure_initialized()
        if name not in cls._operations:
            if name in cls._generation_operator_names:
                raise ValueError(
                    f"Unknown operation: {name}. "
                    "It is provided by the optional data-forge package. "
                    "Install it with: pip install data-forge"
                )
            raise ValueError(
                f"Unknown operation: {name}. "
                "Please register the operation with @register_operation decorator "
                "or add it to register_default_operations."
            )
        op_class = cls._operations[name]
        return op_class(params or {})

    @classmethod
    def get_operation_class(cls, name: str):
        cls._ensure_initialized()
        return cls._operations.get(name)


def register_operation(name: str):
    """Decorator for registering an operation class."""
    def decorator(cls):
        OperationRegistry.register(name, cls)
        return cls
    return decorator
