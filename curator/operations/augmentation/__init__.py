from __future__ import annotations

import importlib
from typing import Any

_OPERATIONS = {
    'AugmentationOperation': 'augmentation_operator',
    'TextAugmentationOperation': 'text_augmentation_operator',
    'ImageAugmentationOperation': 'image_augmentation_operator',
    'AudioAugmentationOperation': 'audio_augmentation_operator',
    'VideoAugmentationOperation': 'video_augmentation_operator',
    'RandomNounAugmentationOperation': 'random_noun_operation',
    'RemoveRepetitionsPunctuationOperation': 'remove_repetitions_punctuation_operation',
    'RemoveStopwordsOperation': 'remove_stopwords_operation',
    'SpellingCorrectionOperation': 'spelling_correction_operation',
    'StemmingLemmatizationOperation': 'stemming_lemmatization_operation',
    'TextNormalizationOperation': 'text_normalization_operation',
    'ImageCannyAugmentationOperation': 'image_canny_operation',
    'ImageCaptioningOperation': 'image_captioning_operation',
    'ImageRandomCropOperation': 'image_random_crop_operation',
    'ImageRandomFlipOperation': 'image_random_flip_operation',
    'VideoResizeAspectRatioOperation': 'video_resize_aspect_ratio_operation',
    'VideoPartitionOperation': 'video_partition_operation',
    'VideoSamplingOperation': 'video_sampling_operation',
    'VideoMultigranularityDescriptionOperation': 'video_multigranularity_description_operation',
    'VideoTemporalStructureOperation': 'video_temporal_structure_operation',
    'VideoScaffoldConstructionOperation': 'video_scaffold_construction_operation',
    'VideoTextEvidenceOperation': 'video_text_evidence_operation',
    'VideoQaeMaterializationOperation': 'video_qae_materialization_operation',
    'VideoConsistencyRevisionOperation': 'video_consistency_revision_operation',
}


def __getattr__(name: str) -> Any:
    module_name = _OPERATIONS.get(name)
    if module_name is None:
        raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
    module = importlib.import_module(f'{__name__}.{module_name}')
    value = getattr(module, name)
    globals()[name] = value
    return value


__all__ = list(_OPERATIONS)
