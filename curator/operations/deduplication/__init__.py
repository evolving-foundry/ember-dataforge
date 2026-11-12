from .deduplication_operator import DeduplicationOperation
from .exact_duplicate_removal_operator import ExactDuplicateRemovalOperation
from .similarity_based_dedup_operator import SimilarityBasedDedupOperation
from .fuzzy_matching_dedup_operator import FuzzyMatchingDedupOperation

__all__ = [
    'DeduplicationOperation',
    'ExactDuplicateRemovalOperation',
    'SimilarityBasedDedupOperation',
    'FuzzyMatchingDedupOperation'
]