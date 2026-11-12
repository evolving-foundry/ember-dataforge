"""
NPU operator adaptation end-to-end test script

Tests the following operators adapted for NPU:
1. _resolve_torch_device() — public device resolution function
2. ImageNsfwFilterOperation — NSFW image filtering
3. VideoTextEvidenceOperation (OCR) — video text recognition
4. ImageCaptioningOperation — image caption generation
5. VideoMultigranularityDescriptionOperation — multigranularity description
6. VideoTemporalStructureOperation (EVENT_CHAIN) — event chain generation
7. load_action_scene_pipeline — action scene classification
"""
import sys
import os
import argparse
import traceback

import pytest

torch = pytest.importorskip("torch")
torch_npu = pytest.importorskip("torch_npu")
if not torch_npu.npu.is_available():
    pytest.skip("NPU is not available", allow_module_level=True)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

os.chdir(PROJECT_ROOT)

# Model directory, passed via the --model_dir argument
MODEL_DIR = ""


def _mp(name):
    """Return the local model path"""
    return os.path.join(MODEL_DIR, name)


def test_resolve_torch_device():
    """Test that _resolve_torch_device returns npu"""
    print("=" * 60)
    print("[TEST 1] _resolve_torch_device()")
    print("=" * 60)
    from curator.utils.video_ops_utils import _resolve_torch_device

    device = _resolve_torch_device()
    print(f"  Auto-detected device: {device}")
    assert device == "npu", f"Expected 'npu', got '{device}'"

    device_explicit = _resolve_torch_device("cpu")
    print(f"  Explicitly specified cpu: {device_explicit}")
    assert device_explicit == "cpu"

    print("  [PASS]\n")


def test_nsfw_filter_npu():
    """Test the NSFW filter running on NPU"""
    print("=" * 60)
    print("[TEST 2] ImageNsfwFilterOperation on NPU")
    print("=" * 60)

    import torch
    import torch_npu
    import numpy as np
    from PIL import Image
    import torch.nn.functional as F

    from curator.operations.filtering.image_nsfw_filter_operation import ImageNsfwFilterOperation

    op = ImageNsfwFilterOperation(params={
        "device_preference": "npu:0",
        "threshold": 0.5,
        "model_path": _mp("nsfw_image_detection"),
    })
    device = op._get_available_device()
    print(f"  Device selection: {device}")
    assert device == "npu:0"

    op._load_model()
    model_device = op._model.device
    print(f"  Model device: {model_device}")
    assert "npu" in str(model_device), f"Model not on NPU: {model_device}"

    test_image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    with torch.no_grad():
        inputs = op._processor(images=test_image, return_tensors="pt")
        inputs = {k: v.to(op._model.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
        outputs = op._model(**inputs)
        probs = F.softmax(outputs.logits[0], dim=0)
    print(f"  NSFW probability: {probs[1].item():.4f}")
    print("  [PASS]\n")


def test_ocr_npu():
    """Test the OCR model running on NPU"""
    print("=" * 60)
    print("[TEST 3] VideoTextEvidenceOperation (OCR) on NPU")
    print("=" * 60)

    import torch
    import torch_npu
    from PIL import Image, ImageDraw

    from curator.operations.augmentation.video_text_evidence_operation import VideoTextEvidenceOperation

    op = VideoTextEvidenceOperation(params={
        "text_evidence_mode": "OCR",
        "model_name": "trocr_small_printed",
        "device_preference": "npu:0",
    })
    # Override the model path with a local directory
    op.model_path = _mp("trocr-small-printed")
    device = op._get_available_device()
    print(f"  Device selection: {device}")
    assert device == "npu:0"

    model, processor = op._load_ocr_components()
    model_device = model.device
    print(f"  Model device: {model_device}")
    assert "npu" in str(model_device), f"Model not on NPU: {model_device}"

    img = Image.new('RGB', (384, 64), color='white')
    draw = ImageDraw.Draw(img)
    draw.text((10, 20), "Hello World", fill='black')

    with torch.no_grad():
        pixel_values = processor(images=img, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(model.device)
        generated_ids = model.generate(pixel_values)
        result = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    print(f"  OCR result: '{result}'")
    print("  [PASS]\n")


def test_image_captioning_npu():
    """Test image caption generation running on NPU"""
    print("=" * 60)
    print("[TEST 4] ImageCaptioningOperation on NPU")
    print("=" * 60)

    import torch
    import torch_npu
    import numpy as np
    from PIL import Image

    from curator.operations.augmentation.image_captioning_operation import ImageCaptioningOperation

    op = ImageCaptioningOperation(params={
        "device_preference": "npu:0",
        "model_path": _mp("blip-image-captioning-base"),
        "max_new_tokens": 16,
    })
    device = op._get_available_device()
    print(f"  Device selection: {device}")
    assert device == "npu:0"

    op._load_model()
    model_device = op._model.device
    print(f"  Model device: {model_device}")
    assert "npu" in str(model_device), f"Model not on NPU: {model_device}"

    test_image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    caption = op._caption_image(test_image, "a photography of")
    print(f"  Generated caption: '{caption}'")
    print("  [PASS]\n")


def test_multigranularity_description_npu():
    """Test multigranularity description running on NPU"""
    print("=" * 60)
    print("[TEST 5] VideoMultigranularityDescriptionOperation on NPU")
    print("=" * 60)

    import torch
    import torch_npu
    import numpy as np
    from PIL import Image
    import tempfile

    from curator.operations.augmentation.video_multigranularity_description_operation import (
        VideoMultigranularityDescriptionOperation,
    )

    op = VideoMultigranularityDescriptionOperation(params={
        "model_name": "blip_base",
        "model_path": _mp("blip-image-captioning-base"),
        "device_preference": "npu:0",
        "max_new_tokens": 16,
    })
    device = op._get_available_device()
    print(f"  Device selection: {device}")
    assert device == "npu:0"

    model, processor, torch_mod = op._load_caption_components()
    model_device = model.device
    print(f"  Model device: {model_device}")
    assert "npu" in str(model_device), f"Model not on NPU: {model_device}"

    test_image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        test_image.save(f.name)
        caption = op._caption_frame(f.name, "a picture of", model, processor, torch_mod, {})
        os.unlink(f.name)

    print(f"  Generated caption: '{caption}'")
    print("  [PASS]\n")


def test_temporal_structure_npu():
    """Test temporal structure (EVENT_CHAIN) running on NPU"""
    print("=" * 60)
    print("[TEST 6] VideoTemporalStructureOperation (EVENT_CHAIN) on NPU")
    print("=" * 60)

    import torch
    import torch_npu

    from curator.operations.augmentation.video_temporal_structure_operation import (
        VideoTemporalStructureOperation,
    )

    op = VideoTemporalStructureOperation(params={
        "temporal_mode": "EVENT_CHAIN",
        "model_path": _mp("Qwen2.5-0.5B-Instruct"),
        "device_preference": "npu:0",
        "max_new_tokens": 32,
    })
    device = op._get_available_device()
    print(f"  Device selection: {device}")
    assert device == "npu:0"

    model, tokenizer = op._load_chain_components()
    model_device = model.device
    print(f"  Model device: {model_device}")
    assert "npu" in str(model_device), f"Model not on NPU: {model_device}"

    prompt = "Summarize: A person walks into a room and sits down."
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=32, pad_token_id=tokenizer.pad_token_id)
    result = tokenizer.batch_decode(
        output_ids[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True,
    )[0].strip()
    print(f"  Generated result: '{result}'")
    print("  [PASS]\n")


def test_action_scene_pipeline_npu():
    """Test the action scene classification pipeline running on NPU"""
    print("=" * 60)
    print("[TEST 7] load_action_scene_pipeline on NPU")
    print("=" * 60)

    import torch
    import torch_npu
    import numpy as np
    from PIL import Image
    import tempfile

    from curator.utils.video_ops_utils import load_action_scene_pipeline

    model_path = _mp("Human-Action-Recognition-VIT-Base-patch16-224")
    classifier = load_action_scene_pipeline(model_path, None)
    print(f"  Pipeline device: {classifier.device}")

    test_image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        test_image.save(f.name)
        predictions = classifier(f.name, top_k=1)
        os.unlink(f.name)

    print(f"  Prediction result: {predictions}")
    print("  [PASS]\n")


def test_video_partition_device_npu():
    """Test VideoPartitionOperation device resolution + model loading onto NPU"""
    print("=" * 60)
    print("[TEST 8] VideoPartitionOperation (SEMANTIC) on NPU")
    print("=" * 60)

    from curator.operations.augmentation.video_partition_operation import VideoPartitionOperation
    from curator.utils.video_ops_utils import _load_semantic_encoder

    op = VideoPartitionOperation(params={
        "partition_mode": "SEMANTIC",
        "device_preference": "npu:0",
        "semantic_window_sec": 4.0,
    })
    device = op._resolve_device()
    print(f"  Device resolution: {device}")
    assert device == "npu:0", f"Expected 'npu:0', got '{device}'"

    # Test model loading onto NPU
    print("  Loading Cosmos-Embed1 model...")
    processor, model, torch_mod = _load_semantic_encoder("nvidia/Cosmos-Embed1-224p", device)
    model_device = model.device
    print(f"  Model device: {model_device}")
    assert "npu" in str(model_device), f"Model not on NPU: {model_device}"
    print("  [PASS]\n")


def test_scaffold_construction_device_npu():
    """Test VideoScaffoldConstructionOperation device resolution"""
    print("=" * 60)
    print("[TEST 9] VideoScaffoldConstructionOperation device resolution")
    print("=" * 60)

    from curator.operations.augmentation.video_scaffold_construction_operation import (
        VideoScaffoldConstructionOperation,
    )
    from curator.utils.video_ops_utils import _resolve_torch_device

    # Test default None (automatic detection)
    op_auto = VideoScaffoldConstructionOperation(params={
        "scaffold_mode": "ACTION_GRAPH",
        "action_scene_model_path": _mp("Human-Action-Recognition-VIT-Base-patch16-224"),
    })
    resolved = _resolve_torch_device(op_auto.action_scene_device)
    print(f"  Default device resolution (None): {resolved}")
    assert resolved == "npu", f"Expected 'npu', got '{resolved}'"

    # Test explicit npu:0
    op_npu = VideoScaffoldConstructionOperation(params={
        "scaffold_mode": "ACTION_GRAPH",
        "action_scene_device": "npu:0",
        "action_scene_model_path": _mp("Human-Action-Recognition-VIT-Base-patch16-224"),
    })
    print(f"  Explicitly specified device: {op_npu.action_scene_device}")
    assert op_npu.action_scene_device == "npu:0"
    print("  [PASS]\n")


def main():
    global MODEL_DIR

    parser = argparse.ArgumentParser(description="NPU operator adaptation end-to-end tests")
    parser.add_argument("--model_dir", type=str, required=True,
                        help="Local model directory (required)")
    parser.add_argument("--test", type=int, nargs="*", default=None,
                        help="Run only the specified test numbers (for example --test 1 2 3)")
    args = parser.parse_args()

    MODEL_DIR = args.model_dir
    print(f"Model directory: {MODEL_DIR}\n")

    tests = [
        ("_resolve_torch_device", test_resolve_torch_device),
        ("NSFW Filter NPU", test_nsfw_filter_npu),
        ("OCR NPU", test_ocr_npu),
        ("Image Captioning NPU", test_image_captioning_npu),
        ("Multigranularity Description NPU", test_multigranularity_description_npu),
        ("Temporal Structure (EVENT_CHAIN) NPU", test_temporal_structure_npu),
        ("Action Scene Pipeline NPU", test_action_scene_pipeline_npu),
        ("Video Partition (SEMANTIC) NPU", test_video_partition_device_npu),
        ("Scaffold Construction device", test_scaffold_construction_device_npu),
    ]

    results = []
    for idx, (name, test_fn) in enumerate(tests, start=1):
        if args.test and idx not in args.test:
            continue
        try:
            test_fn()
            results.append((name, "PASS"))
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")
            traceback.print_exc()
            results.append((name, f"FAIL: {e}"))
            print()

    print("\n" + "=" * 60)
    print("Test result summary")
    print("=" * 60)
    for name, status in results:
        icon = "+" if status == "PASS" else "x"
        print(f"  {icon} {name}: {status}")

    passed = sum(1 for _, s in results if s == "PASS")
    total = len(results)
    print(f"\n  Total: {passed}/{total} passed")


if __name__ == "__main__":
    main()
