"""
Validate the basic structure and parameters of migrated operators (without external library dependencies)
"""
import sys
import os
import ast

# Add the project root to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))


def parse_operation_file(filepath):
    """Parse an operator file and extract key information"""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    try:
        tree = ast.parse(content)
    except SyntaxError as e:
        return {"error": f"Syntax error: {e}"}
    
    result = {
        "class_name": None,
        "operation_type": None,
        "has_params_schema": False,
        "params_schema_keys": [],
        "has_execute": False,
        "has_filter": False,
        "base_classes": []
    }
    
    for node in ast.walk(tree):
        # Find class definitions
        if isinstance(node, ast.ClassDef):
            # Skip stub classes (names starting with an underscore)
            if node.name.startswith('_'):
                continue
            
            result["class_name"] = node.name
            result["class_name"] = node.name
            # Get base classes
            for base in node.bases:
                if isinstance(base, ast.Name):
                    result["base_classes"].append(base.id)
                elif isinstance(base, ast.Attribute):
                    result["base_classes"].append(base.attr)
            
            # Find class attributes and methods
            for item in node.body:
                # Check operation_type
                if isinstance(item, ast.Assign):
                    for target in item.targets:
                        if isinstance(target, ast.Name) and target.id == "operation_type":
                            if isinstance(item.value, ast.Attribute):
                                result["operation_type"] = item.value.attr
                
                # Check PARAMS_SCHEMA
                if isinstance(item, ast.Assign):
                    for target in item.targets:
                        if isinstance(target, ast.Name) and target.id == "PARAMS_SCHEMA":
                            result["has_params_schema"] = True
                            # Try to extract parameter keys
                            if isinstance(item.value, ast.Dict):
                                for key in item.value.keys:
                                    if isinstance(key, ast.Constant):
                                        result["params_schema_keys"].append(key.value)
                
                # Check methods
                if isinstance(item, ast.FunctionDef):
                    if item.name == "execute":
                        result["has_execute"] = True
                    if item.name == "filter":
                        result["has_filter"] = True
    
    return result


def validate_all_operations():
    """Validate all migrated operations"""
    print("=" * 70)
    print("Migrated-operator structural validation (static analysis)")
    print("=" * 70)
    
    base_path = "curator/operations"
    
    operations = {
        "filtering": [
            "bleu_sample_filter_operation.py",
            "blocklist_filter_operation.py",
            "cider_sample_filter_operation.py",
            "mean_word_length_filter_operation.py",
            "ngram_sample_filter_operation.py",
            "special_char_filter_operation.py",
            "image_aspect_ratio_filter_operation.py",
            "image_nsfw_filter_operation.py",
            "image_size_filter_operation.py",
        ],
        "augmentation": [
            "random_noun_operation.py",
            "remove_repetitions_punctuation_operation.py",
            "remove_stopwords_operation.py",
            "spelling_correction_operation.py",
            "stemming_lemmatization_operation.py",
            "text_normalization_operation.py",
            "image_canny_operation.py",
            "image_captioning_operation.py",
            "image_random_crop_operation.py",
            "image_random_flip_operation.py",
            "video_resize_aspect_ratio_operation.py",
        ]
    }
    
    total = 0
    passed = 0
    errors = []
    
    for module, files in operations.items():
        print(f"\n📁 {module.upper()} module:")
        print("-" * 70)
        
        for filename in files:
            total += 1
            filepath = os.path.join(base_path, module, filename)
            
            if not os.path.exists(filepath):
                print(f"  ✗ {filename} - file does not exist")
                errors.append(f"{filename}: file does not exist")
                continue
            
            result = parse_operation_file(filepath)
            
            if "error" in result:
                print(f"  ✗ {filename} - {result['error']}")
                errors.append(f"{filename}: {result['error']}")
                continue
            
            # Validation rules
            checks = []
            failures = []

            # 1. Class names should end with Operation
            if result["class_name"] and result["class_name"].endswith("Operation"):
                checks.append("Class name is correct")
            else:
                checks.append(f"Class name is incorrect: {result['class_name']}")
                failures.append("class name")

            # 2. operation_type should be set
            if result["operation_type"]:
                checks.append(f"operation_type={result['operation_type']}")
            else:
                checks.append("Missing operation_type")
                failures.append("operation_type")

            # 3. PARAMS_SCHEMA should be present
            if result["has_params_schema"]:
                checks.append("Has PARAMS_SCHEMA")
            else:
                checks.append("Missing PARAMS_SCHEMA")
                failures.append("PARAMS_SCHEMA")

            # 4. execute method should be present
            if result["has_execute"]:
                checks.append("Has execute method")
            else:
                checks.append("Missing execute method")
                failures.append("execute method")

            # 5. FilteringOperation should have a filter method
            if "FilteringOperation" in result["base_classes"]:
                if result["has_filter"]:
                    checks.append("Has filter method")
                else:
                    checks.append("Missing filter method")
                    failures.append("filter method")

            # Check whether all key items pass (failures tracked by field, not string matching)
            all_passed = not failures
            
            if all_passed:
                print(f"  ✓ {result['class_name']}")
                print(f"    {', '.join(checks)}")
                passed += 1
            else:
                print(f"  ✗ {result['class_name']}")
                print(f"    {', '.join(checks)}")
                errors.append(f"{filename}: {'; '.join(checks)}")
    
    print("\n" + "=" * 70)
    print(f"Validation result: {passed}/{total} operators passed validation")
    print("=" * 70)
    
    if errors:
        print("\n❌ Error details:")
        for err in errors:
            print(f"  - {err}")
    else:
        print("\n🎉 All operator structure validations passed!")
    
    return passed == total


def check_pipeline_registration():
    """Check Pipeline registration"""
    print("\n" + "=" * 70)
    print("Pipeline registration check")
    print("=" * 70)
    
    pipeline_file = "curator/core/pipeline.py"
    
    if not os.path.exists(pipeline_file):
        print(f"  ✗ Pipeline file does not exist: {pipeline_file}")
        return False
    
    with open(pipeline_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check import statements
    filtering_imports = [
        "BleuSampleFilterOperation",
        "BlocklistFilterOperation",
        "CiderSampleFilterOperation",
        "MeanWordLengthFilterOperation",
        "NgramSampleFilterOperation",
        "SpecialCharFilterOperation",
        "ImageAspectRatioFilterOperation",
        "ImageNsfwFilterOperation",
        "ImageSizeFilterOperation",
    ]
    
    augmentation_imports = [
        "RandomNounAugmentationOperation",
        "RemoveRepetitionsPunctuationOperation",
        "RemoveStopwordsOperation",
        "SpellingCorrectionOperation",
        "StemmingLemmatizationOperation",
        "TextNormalizationOperation",
        "ImageCannyAugmentationOperation",
        "ImageCaptioningOperation",
        "ImageRandomCropOperation",
        "ImageRandomFlipOperation",
        "VideoResizeAspectRatioOperation",
    ]
    
    registered_ops = [
        "'bleu_sample_filter'",
        "'blocklist_filter'",
        "'cider_sample_filter'",
        "'mean_word_length_filter'",
        "'ngram_sample_filter'",
        "'special_char_filter'",
        "'image_aspect_ratio_filter'",
        "'image_nsfw_filter'",
        "'image_size_filter'",
        "'random_noun'",
        "'remove_repetitions_punctuation'",
        "'remove_stopwords'",
        "'spelling_correction'",
        "'stemming_lemmatization'",
        "'text_normalization'",
        "'image_canny'",
        "'image_captioning'",
        "'image_random_crop'",
        "'image_random_flip'",
        "'video_resize_aspect_ratio'",
    ]
    
    print("\n  Import check:")
    all_imports = filtering_imports + augmentation_imports
    missing_imports = [op for op in all_imports if op not in content]
    
    if missing_imports:
        print(f"    ✗ Missing {len(missing_imports)} imports")
        for op in missing_imports[:5]:
            print(f"      - {op}")
        if len(missing_imports) > 5:
            print(f"      ... and {len(missing_imports) - 5} more")
    else:
        print(f"    ✓ All {len(all_imports)} operator classes imported")
    
    print("\n  Registration check:")
    missing_regs = [op for op in registered_ops if op not in content]
    
    if missing_regs:
        print(f"    ✗ Missing {len(missing_regs)} registrations")
        for op in missing_regs[:5]:
            print(f"      - {op}")
        if len(missing_regs) > 5:
            print(f"      ... and {len(missing_regs) - 5} more")
    else:
        print(f"    ✓ All {len(registered_ops)} operators are registered in Pipeline")
    
    return len(missing_imports) == 0 and len(missing_regs) == 0


def check_init_exports():
    """Check __init__.py exports"""
    print("\n" + "=" * 70)
    print("__init__.py export check")
    print("=" * 70)
    
    modules = ["filtering", "augmentation"]
    all_ok = True
    
    for module in modules:
        init_file = f"curator/operations/{module}/__init__.py"
        
        if not os.path.exists(init_file):
            print(f"\n  ✗ {init_file} does not exist")
            all_ok = False
            continue
        
        with open(init_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check whether __all__ is present
        has_all = "__all__" in content
        has_from_imports = "from ." in content
        
        if has_all and has_from_imports:
            print(f"\n  ✓ {module}/__init__.py exports are correct")
        else:
            print(f"\n  ✗ {module}/__init__.py exports are incorrect")
            if not has_all:
                print("    - missing __all__ definition")
            if not has_from_imports:
                print("    - missing from . imports")
            all_ok = False
    
    return all_ok


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("Migrated operator validation tests (no external dependencies)")
    print("=" * 70)
    
    results = []
    
    # Test 1: operator structure
    results.append(("Operator structure validation", validate_all_operations()))
    
    # Test 2: Pipeline registration
    results.append(("Pipeline registration check", check_pipeline_registration()))
    
    # Test 3: __init__.py exports
    results.append(("__init__.py export check", check_init_exports()))
    
    # Summary
    print("\n" + "=" * 70)
    print("Validation summary")
    print("=" * 70)
    
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
