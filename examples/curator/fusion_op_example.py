"""Fusion OP feature demonstration example"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from curator.core.pipeline import MultimodalPipeline
from curator.operations.cleaning import InvalidDataRemovalOperation
from curator.operations.filtering import QualityBasedFilterOperation
from curator.operations.formatting import UniformFormatConversionOperation
from curator.operations.common.fusion_operator import FusionOperation


def demonstrate_fusion_op():
    """Demonstrates how to use Fusion OP"""
    
    print("=== Fusion OP Feature Demo ===")
    
    # Create some basic operations
    clean_op = InvalidDataRemovalOperation({"threshold": 0.8})
    filter_op = QualityBasedFilterOperation({"min_quality_score": 0.7})
    format_op = UniformFormatConversionOperation({"target_format": "json"})
    
    # Create a fused operation - merge three operations into one atomic operation
    fused_operation = FusionOperation([
        clean_op,
        filter_op,
        format_op
    ])
    
    print(f"Fused operation contains: {fused_operation.get_operation_names()}")
    
    # Create a pipeline and add the fused operation
    pipeline = MultimodalPipeline()
    pipeline.add_operation(fused_operation)
    
    print(f"Operations in pipeline: {[op.name for op in pipeline.operations]}")
    
    # Compare individual operations with the fused operation
    print("\n--- Comparison: separate operations vs. fused operation ---")
    
    # Individual operation approach
    separate_pipeline = MultimodalPipeline()
    separate_pipeline.add_operation('invalid_data_removal', {"threshold": 0.8})
    separate_pipeline.add_operation('quality_based_filter', {"min_quality_score": 0.7})
    separate_pipeline.add_operation('uniform_format_conversion', {"target_format": "json"})
    
    print(f"Separate operation pipeline: {[op.name for op in separate_pipeline.operations]}")
    
    # Fused operation approach
    fused_pipeline = MultimodalPipeline()
    fused_op = FusionOperation([
        InvalidDataRemovalOperation({"threshold": 0.8}),
        QualityBasedFilterOperation({"min_quality_score": 0.7}),
        UniformFormatConversionOperation({"target_format": "json"})
    ])
    fused_pipeline.add_operation(fused_op)
    
    print(f"Fused operation pipeline: {[op.name for op in fused_pipeline.operations]}")
    
    print("\nKey benefits of Fusion OP:")
    print("1. Performance: reduces intermediate data transfer and storage overhead")
    print("2. Atomicity: executes multiple operations as a single unit")
    print("3. Simpler pipelines: reduces the number of operations in a pipeline")
    print("4. Reusability: predefined operation combinations can be reused")


if __name__ == "__main__":
    demonstrate_fusion_op()
