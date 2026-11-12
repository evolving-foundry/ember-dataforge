#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ImageOcrOperation example script

Function:
    Generate an OCR sample image and output parsed-doc-v2 through ImageOcrOperation.
    You can also pass in your own png/jpg image.

End-to-end examples (with image and complete JSONL output):
    MindOCR:  examples/curator/single_ops/loading/image_ocr_cpu_example.md
    PaddleOCR: examples/curator/single_ops/loading/image_ocr_paddleocr_cpu_example.md
    Backend quality ranking: examples/curator/single_ops/loading/ocr_backend_ranking_example.md

Preflight (does not load an OCR backend):
    python examples/curator/single_ops/loading/image_ocr_example.py --dry-run

MindOCR + CPU actual run (English):
    python examples/curator/single_ops/loading/image_ocr_example.py --backend mindocr --device cpu --lang en --allow-model-download

MindOCR + CPU actual run (Chinese):
    python examples/curator/single_ops/loading/image_ocr_example.py --backend mindocr --device cpu --lang zh --allow-model-download

MindOCR + NPU actual run:
    python examples/curator/single_ops/loading/image_ocr_example.py \
        --backend mindocr --device npu --allow-model-download

Use local models:
    python examples/curator/single_ops/loading/image_ocr_example.py \
        --backend mindocr --device npu \
        --model-det /path/to/det --model-rec /path/to/rec

Use a local Chinese dictionary:
    python examples/curator/single_ops/loading/image_ocr_example.py --backend mindocr --device cpu --lang zh --model-rec-dict /path/to/ppocr_keys_v1.txt --allow-model-download

Dependencies:
    pip install -e ".[ocr-mindocr]"
    pip install -e ".[ocr-paddleocr]"
    The MindOCR NPU image requires the vendor MindSpore NPU/CANN runtime;
    PaddleOCR NPU requires a PaddlePaddle runtime that supports NPU.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "examples" / "curator"))

from curator.checkpoint.checkpoint_models import CheckpointConfig
from curator.core.pipeline import MultimodalPipeline
from curator.operations.loading import ImageOcrOperation
from common import init_ray, shutdown_ray

_NO_CHECKPOINT = CheckpointConfig(enabled=False)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "output" / "image_ocr"


def _load_font(size: int):
    from PIL import ImageFont

    for name in ("DejaVuSans.ttf", "arial.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _create_sample_image(path: Path) -> None:
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (1400, 720), "white")
    draw = ImageDraw.Draw(image)
    title_font = _load_font(42)
    text_font = _load_font(30)

    draw.text((60, 45), "DataForge Curator OCR Sample", fill="black", font=title_font)
    draw.text((60, 115), "Invoice No: OCR-2026-001", fill="black", font=text_font)
    draw.text((60, 165), "Supplier: Beijing Xinghe Technology", fill="black", font=text_font)
    draw.text((60, 215), "Date: 2026-10-30", fill="black", font=text_font)

    draw.rectangle((60, 285, 1340, 650), outline="black", width=3)
    draw.line((60, 355, 1340, 355), fill="black", width=2)
    draw.line((720, 355, 720, 650), fill="black", width=2)
    draw.text((85, 305), "Item", fill="black", font=text_font)
    draw.text((745, 305), "Amount", fill="black", font=text_font)
    draw.text((85, 390), "AI workstation", fill="black", font=text_font)
    draw.text((745, 390), "128560.00", fill="black", font=text_font)
    draw.text((85, 475), "Support service", fill="black", font=text_font)
    draw.text((745, 475), "86400.00", fill="black", font=text_font)
    draw.text((85, 575), "Total: 214960.00 CNY", fill="black", font=text_font)

    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(str(path), format="PNG")


def _jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json", exclude_none=False)
    if isinstance(value, Mapping):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(_jsonable(row), ensure_ascii=False, separators=(",", ":"))
            )
            handle.write("\n")


def _print_rows(rows: Iterable[Mapping[str, Any]]) -> None:
    for index, row in enumerate(rows, start=1):
        error = row.get("error")
        if error:
            print(f"  Record {index}: failed {error.get('code')} ({error.get('stage')})")
            continue
        document = row.get("parsed_doc") or {}
        blocks = document.get("blocks", [])
        texts = [block.get("text", "") for block in blocks if block.get("text")]
        first_text = texts[0][:80] if texts else ""
        print(
            f"  Record {index}: language={document.get('language')}, "
            f"blocks={len(blocks)}, first_text={first_text!r}"
        )


def _build_model_paths(args: argparse.Namespace) -> Dict[str, str]:
    result = {}
    if args.model_det:
        result["det"] = str(Path(args.model_det).resolve())
    if args.model_rec:
        result["rec"] = str(Path(args.model_rec).resolve())
    if args.model_layout:
        result["layout"] = str(Path(args.model_layout).resolve())
    if args.model_rec_dict:
        result["rec_dict"] = str(Path(args.model_rec_dict).resolve())
    if result and args.backend == "auto":
        raise SystemExit("model_paths requires explicitly specifying --backend mindocr or paddleocr")
    return result


def _build_params(args: argparse.Namespace, image_path: Path, model_paths: Mapping[str, str]) -> Dict[str, Any]:
    params: Dict[str, Any] = {
        "data_paths": [str(image_path)],
        "allowed_roots": [str(image_path.parent)],
        "output_key": args.output_key,
        "backend": args.backend,
        "device": args.device,
        "lang": args.lang,
        "layout_analysis": not args.no_layout_analysis,
        "on_error": args.on_error,
        "allow_model_download": args.allow_model_download,
    }
    if model_paths:
        params["model_paths"] = dict(model_paths)
    if args.ranking_path:
        params["backend_ranking_path"] = str(Path(args.ranking_path).resolve())
    return params


def _print_dry_run(args: argparse.Namespace, image_path: Path, params: Mapping[str, Any]) -> None:
    print("OCR dry-run complete; no backend was loaded or invoked")
    print(f"  Sample image: {image_path}")
    print(f"  backend: {params['backend']}")
    print(f"  device: {params['device']}")
    print(f"  lang: {params['lang']}")
    for module in ("mindocr", "mindspore", "paddleocr"):
        installed = importlib.util.find_spec(module) is not None
        print(f"  {module} installed: {installed}")
    print("\nMindOCR and PaddleOCR are supported; auto prefers MindOCR when no ranking table is available.")
    print("MindOCR CPU command example (Chinese):")
    print(
        "  python examples/curator/single_ops/loading/image_ocr_example.py"
        f" --image {image_path} --backend mindocr --device cpu --lang zh --allow-model-download"
    )
    print("PaddleOCR CPU command example:")
    print(
        "  python examples/curator/single_ops/loading/image_ocr_example.py"
        f" --image {image_path} --backend paddleocr --device cpu --lang zh --allow-model-download"
    )
    print("MindOCR NPU command example:")
    print(
        "  python examples/curator/single_ops/loading/image_ocr_example.py"
        f" --image {image_path} --backend mindocr --device npu --allow-model-download"
    )


def run_example(args: argparse.Namespace) -> int:
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.image:
        image_path = Path(args.image).resolve()
        if not image_path.is_file():
            raise SystemExit(f"Image does not exist or is not a file: {image_path}")
    else:
        image_path = output_dir / "inputs" / "ocr_sample.png"
        _create_sample_image(image_path)
        print(f"Generated OCR sample image: {image_path}")

    model_paths = _build_model_paths(args)
    params = _build_params(args, image_path, model_paths)
    if args.dry_run:
        _print_dry_run(args, image_path, params)
        return 0

    if not init_ray():
        raise SystemExit("Ray initialization failed; cannot run the example")

    output_file = output_dir / "ocr-documents.jsonl"
    pipeline = MultimodalPipeline(enable_tracking=False, checkpoint_config=_NO_CHECKPOINT)
    try:
        pipeline.add_operation(ImageOcrOperation(params))
        result = pipeline.run()
        rows = [_jsonable(row) for row in result.dataset.iter_rows()]
        _write_jsonl(output_file, rows)
        print(f"OCR complete: {len(rows)} records")
        _print_rows(rows)
        print(f"Output file: {output_file}")
        return 0
    except Exception as exc:
        code = getattr(exc, "code", None)
        message = getattr(exc, "message", str(exc))
        print(f"Execution failed: {code or type(exc).__name__}: {message}")
        if code == "DEPENDENCY_MISSING":
            print(
                'Install example: pip install -e ".[ocr-mindocr]" '
                'or pip install -e ".[ocr-paddleocr]"'
            )
        return 1
    finally:
        pipeline.cleanup()
        shutdown_ray()


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="ImageOcrOperation example")
    parser.add_argument("--image", type=Path, default=None, help="Image to recognize; generates a PNG automatically when omitted")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="JSONL output directory")
    parser.add_argument("--backend", choices=["auto", "mindocr", "paddleocr"], default="auto")
    parser.add_argument("--device", choices=["auto", "npu", "cuda", "cpu"], default="auto")
    parser.add_argument("--lang", choices=["zh", "en", "mixed", "ch"], default="en")
    parser.add_argument("--output-key", default="parsed_doc")
    parser.add_argument("--no-layout-analysis", action="store_true")
    parser.add_argument("--on-error", choices=["raise", "skip"], default="raise")
    parser.add_argument("--ranking-path", default=None)
    parser.add_argument("--model-det", default=None)
    parser.add_argument("--model-rec", default=None)
    parser.add_argument("--model-layout", default=None)
    parser.add_argument(
        "--model-rec-dict",
        default=None,
        help="MindOCR Chinese/mixed recognition dictionary (ppocr_keys_v1.txt); PaddleOCR does not support this argument",
    )
    parser.add_argument("--allow-model-download", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Only generate the image and print arguments; do not invoke OCR")
    args = parser.parse_args(argv)
    return run_example(args)


if __name__ == "__main__":
    raise SystemExit(main())
