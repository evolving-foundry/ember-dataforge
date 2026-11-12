"""
Test public lazy-loading modules
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))


def test_lazy_imports():
    """Test public lazy-loading modules."""
    print("=" * 60)
    print("Testing the shared lazy-loading module")
    print("=" * 60)
    
    # Test 1: import the module
    print("\n1. Test module import:")
    try:
        from curator.utils import get_nlp_backend, ensure_nltk, ensure_jieba
        print("  ✓ Shared methods imported successfully")
    except Exception as e:
        print(f"  ✗ Import failed: {e}")
        return False
    
    # Test 2: get the NLP backend
    print("\n2. Test NLP backend acquisition:")
    try:
        nlp = get_nlp_backend()
        print(f"  ✓ NLP backend instance acquired successfully")
        print(f"    Singleton check: {nlp is get_nlp_backend()}")
    except Exception as e:
        print(f"  ✗ Acquisition failed: {e}")
        return False
    
    # Test 3: test ensure_nltk (independent of whether nltk is installed)
    print("\n3. Test ensure_nltk:")
    try:
        result = nlp.ensure_nltk()
        print(f"  ✓ ensure_nltk() returned: {result}")
        if result:
            print("    nltk is installed")
        else:
            print("    nltk is not installed; using fallback")
    except Exception as e:
        print(f"  ✗ Call failed: {e}")
        return False
    
    # Test 4: test ensure_jieba (independent of whether jieba is installed)
    print("\n4. Test ensure_jieba:")
    try:
        result = nlp.ensure_jieba()
        print(f"  ✓ ensure_jieba() returned: {result}")
        if result:
            print("    jieba is installed")
        else:
            print("    jieba is not installed; using fallback")
    except Exception as e:
        print(f"  ✗ Call failed: {e}")
        return False
    
    # Test 5: verify operators use the public methods
    print("\n5. Verify operators use shared methods:")
    try:
        from curator.operations.filtering import BleuSampleFilterOperation
        from curator.operations.filtering import MeanWordLengthFilterOperation
        from curator.operations.augmentation import RandomNounAugmentationOperation
        
        # Check whether public methods are imported in the code
        import inspect
        
        bleu_source = inspect.getsource(BleuSampleFilterOperation)
        mean_source = inspect.getsource(MeanWordLengthFilterOperation)
        random_noun_source = inspect.getsource(RandomNounAugmentationOperation)
        
        checks = []
        if 'from ...utils.lazy_imports import' in bleu_source:
            checks.append("BleuSampleFilterOperation ✓")
        if 'from ...utils.lazy_imports import' in mean_source:
            checks.append("MeanWordLengthFilterOperation ✓")
        if 'from ...utils.lazy_imports import' in random_noun_source:
            checks.append("RandomNounAugmentationOperation ✓")
        
        if checks:
            print("  ✓ The following operators use the shared lazy-loading methods:")
            for check in checks:
                print(f"    - {check}")
        else:
            print("  ✗ Operators do not use the shared methods")
            return False
            
    except Exception as e:
        print(f"  ✗ Validation failed: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
    return True


def test_no_module_level_import():
    """Test that modules do not import nltk or jieba at module level."""
    print("\n" + "=" * 60)
    print("Testing module-level import cleanup")
    print("=" * 60)
    
    files_to_check = [
        "curator/operations/filtering/bleu_sample_filter_operation.py",
        "curator/operations/filtering/cider_sample_filter_operation.py",
        "curator/operations/filtering/mean_word_length_filter_operation.py",
        "curator/operations/filtering/ngram_sample_filter_operation.py",
        "curator/operations/augmentation/random_noun_operation.py",
        "curator/operations/augmentation/remove_stopwords_operation.py",
        "curator/operations/augmentation/stemming_lemmatization_operation.py",
    ]
    
    all_passed = True
    
    for filepath in files_to_check:
        with open(filepath, 'r') as f:
            content = f.read()
        
        # Check whether any module-level nltk/jieba imports remain
        lines = content.split('\n')
        has_module_import = False
        
        for i, line in enumerate(lines[:30]):  # Check the first 30 lines
            stripped = line.strip()
            if stripped.startswith('import nltk') or stripped.startswith('import jieba'):
                # Exclude imports inside methods (judged by indentation)
                if not line.startswith(' ') and not line.startswith('\t'):
                    has_module_import = True
                    print(f"  ✗ {filepath}: line {i+1} still has a module-level import: {line.strip()}")
                    all_passed = False
                    break
        
        if not has_module_import:
            filename = os.path.basename(filepath)
            print(f"  ✓ {filename}: no module-level nltk/jieba imports")
    
    if all_passed:
        print("\n✅ All files have module-level imports removed")
    else:
        print("\n⚠️ Some files still have module-level imports")
    
    return all_passed


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Shared lazy-loading module tests")
    print("=" * 60 + "\n")
    
    results = []
    results.append(("Shared module functionality test", test_lazy_imports()))
    results.append(("Module-level import cleanup test", test_no_module_level_import()))
    
    print("\n" + "=" * 60)
    print("Test summary")
    print("=" * 60)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status} - {name}")
    
    if all(r[1] for r in results):
        print("\n🎉 All tests passed!")
        sys.exit(0)
    else:
        print("\n⚠️  Some tests failed")
        sys.exit(1)
