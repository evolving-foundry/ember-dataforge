"""
Validate the basic structure and parameters of migrated operators
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))


def test_operation_structure():
    """Test the basic structure of all migrated operators"""
    print("=" * 60)
    print("Validating the basic structure of migrated operators...")
    print("=" * 60)
    
    errors = []
    success_count = 0
    
    # Test Filtering Operations
    filtering_ops = [
        ("BleuSampleFilterOperation", "filtering"),
        ("BlocklistFilterOperation", "filtering"),
        ("CiderSampleFilterOperation", "filtering"),
        ("MeanWordLengthFilterOperation", "filtering"),
        ("NgramSampleFilterOperation", "filtering"),
        ("SpecialCharFilterOperation", "filtering"),
        ("ImageAspectRatioFilterOperation", "filtering"),
        ("ImageNsfwFilterOperation", "filtering"),
        ("ImageSizeFilterOperation", "filtering"),
    ]
    
    # Test Augmentation Operations
    augmentation_ops = [
        ("RandomNounAugmentationOperation", "augmentation"),
        ("RemoveRepetitionsPunctuationOperation", "augmentation"),
        ("RemoveStopwordsOperation", "augmentation"),
        ("SpellingCorrectionOperation", "augmentation"),
        ("StemmingLemmatizationOperation", "augmentation"),
        ("TextNormalizationOperation", "augmentation"),
        ("ImageCannyAugmentationOperation", "augmentation"),
        ("ImageCaptioningOperation", "augmentation"),
        ("ImageRandomCropOperation", "augmentation"),
        ("ImageRandomFlipOperation", "augmentation"),
        ("VideoResizeAspectRatioOperation", "augmentation"),
    ]
    
    all_ops = filtering_ops + augmentation_ops
    
    for op_name, module in all_ops:
        try:
            if module == "filtering":
                exec(f"from curator.operations.filtering import {op_name}")
            else:
                exec(f"from curator.operations.augmentation import {op_name}")
            
            # Validate operation_type
            exec(f"assert {op_name}.operation_type is not None, '{op_name} has no operation_type'")
            
            # Validate PARAMS_SCHEMA
            exec(f"assert hasattr({op_name}, 'PARAMS_SCHEMA'), '{op_name} has no PARAMS_SCHEMA'")
            exec(f"assert 'params' in {op_name}.PARAMS_SCHEMA, '{op_name}.PARAMS_SCHEMA is missing the params key'")
            
            print(f"  ✓ {op_name} - structure validation passed")
            success_count += 1
            
        except Exception as e:
            error_msg = f"  ✗ {op_name} - validation failed: {str(e)}"
            print(error_msg)
            errors.append(error_msg)
    
    print("\n" + "=" * 60)
    print(f"Validation result: {success_count}/{len(all_ops)} operators passed validation")
    print("=" * 60)
    
    if errors:
        print("\nError details:")
        for err in errors:
            print(err)
        return False
    
    return True


def test_pipeline_registration():
    """Test Pipeline registration"""
    print("\n" + "=" * 60)
    print("Validating Pipeline registration...")
    print("=" * 60)
    
    try:
        from curator.core.pipeline import register_default_operations
        print("  ✓ register_default_operations imported successfully")
        
        # Check whether all operations are registered
        from curator.core.pipeline import OperationRegistry
        
        expected_ops = [
            'bleu_sample_filter', 'blocklist_filter', 'cider_sample_filter',
            'mean_word_length_filter', 'ngram_sample_filter', 'special_char_filter',
            'image_aspect_ratio_filter', 'image_nsfw_filter', 'image_size_filter',
            'random_noun', 'remove_repetitions_punctuation', 'remove_stopwords',
            'spelling_correction', 'stemming_lemmatization', 'text_normalization',
            'image_canny', 'image_captioning', 'image_random_crop',
            'image_random_flip', 'video_resize_aspect_ratio'
        ]
        
        missing_ops = []
        for op_name in expected_ops:
            op_class = OperationRegistry.get_operation_class(op_name)
            if op_class is None:
                missing_ops.append(op_name)
                print(f"  ✗ {op_name} - not registered in Pipeline")
            else:
                print(f"  ✓ {op_name} - registered: {op_class.__name__}")
        
        if missing_ops:
            print(f"\n  Warning: {len(missing_ops)} operators are not registered")
            return False
        
        print(f"\n  All {len(expected_ops)} operators were registered successfully in Pipeline")
        return True
        
    except Exception as e:
        print(f"  ✗ Pipeline validation failed: {str(e)}")
        return False


def test_params_schema():
    """Test parameter Schema definitions"""
    print("\n" + "=" * 60)
    print("Validating parameter Schema definitions...")
    print("=" * 60)
    
    test_cases = [
        ("BleuSampleFilterOperation", "filtering", ["n_grams", "reference_key", "text_key", "threshold"]),
        ("BlocklistFilterOperation", "filtering", ["blocklist", "max_hits", "match_mode", "text_key"]),
        ("TextNormalizationOperation", "augmentation", ["text_key"]),
        ("ImageAspectRatioFilterOperation", "filtering", ["image_key", "min_ratio", "max_ratio"]),
        ("VideoResizeAspectRatioOperation", "augmentation", ["video_key", "target_aspect_ratio_range"]),
    ]
    
    success_count = 0
    
    for op_name, module, expected_params in test_cases:
        try:
            if module == "filtering":
                exec(f"from curator.operations.filtering import {op_name}")
            else:
                exec(f"from curator.operations.augmentation import {op_name}")
            
            # Check whether parameters are defined
            for param in expected_params:
                exec(f"assert param in {op_name}.PARAMS_SCHEMA['params'], f'{op_name}missing parameter {{param}}'")
            
            print(f"  ✓ {op_name} - parameter Schema validation passed")
            success_count += 1
            
        except Exception as e:
            print(f"  ✗ {op_name} - validation failed: {str(e)}")
    
    print(f"\n  {success_count}/{len(test_cases)} operators passed parameter Schema validation")
    return success_count == len(test_cases)


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Migrated operator validation tests")
    print("=" * 60 + "\n")
    
    results = []
    
    # Test 1: basic structure
    results.append(("Basic structure validation", test_operation_structure()))
    
    # Test 2: Pipeline registration
    results.append(("Pipeline registration validation", test_pipeline_registration()))
    
    # Test 3: parameter Schema
    results.append(("Parameter Schema validation", test_params_schema()))
    
    # Summary
    print("\n" + "=" * 60)
    print("Validation summary")
    print("=" * 60)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status} - {name}")
    
    all_passed = all(r[1] for r in results)
    
    if all_passed:
        print("\n🎉 All validation tests passed!")
        sys.exit(0)
    else:
        print("\n⚠️  Some validation tests failed; check the error messages")
        sys.exit(1)
