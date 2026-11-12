#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Run a real OCR backend ranking example end to end.

The example does three things:

1. Runs an annotated image manifest through one or more real OCR backends.
2. Writes and validates an ``ocr-ranking-v1`` YAML table.
3. Runs ``ImageOcrOperation`` with ``backend=auto`` and the generated table,
   then verifies that provenance records ``routing_reason="ranking"``.

``--route-only`` reuses an existing ranking table without rerunning OCR. This
is useful when MindOCR and PaddleOCR run in separate virtual environments.

The bundled corpus is intentionally small. It is a reproducible integration
example, not a production quality benchmark or a replacement for the package
default ranking table.

Dependencies:
    pip install -e ".[ocr-mindocr]"   # MindOCR CPU
    pip install -e ".[ocr-paddleocr]" # PaddleOCR CPU

The supplied lock files pin incompatible NumPy versions, so each backend must
run in its own environment and write to the shared ranking file with ``--merge``.

Run one backend per environment:
    python examples/curator/single_ops/loading/ocr_backend_ranking_example.py \
        --backend paddleocr --device cpu --lang en --allow-model-download

    python examples/curator/single_ops/loading/ocr_backend_ranking_example.py \
        --backend mindocr --device cpu --lang en --allow-model-download --merge

"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'examples' / 'curator'))

from curator.checkpoint.checkpoint_models import CheckpointConfig
from curator.core.pipeline import MultimodalPipeline
from curator.operations.loading import ImageOcrOperation
from curator.operations.loading.ocr_benchmark import (
    load_benchmark_manifest,
    run_benchmark,
    write_ranking_table,
)
from curator.operations.loading.ocr_ranking import parse_ranking_document
from common import init_ray, shutdown_ray

import yaml

_NO_CHECKPOINT = CheckpointConfig(enabled=False)
ASSET_DIR = SCRIPT_DIR / 'assets' / 'ocr_ranking'
DEFAULT_MANIFEST = ASSET_DIR / 'manifest.jsonl'
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / 'data' / 'output' / 'ocr_backend_ranking'
DEFAULT_RANKING_VERSION = 'ocr-ranking-demo-v1'
DEFAULT_EVALUATION_SET_VERSION = 'ocr-ranking-demo-en-v1'
_BACKENDS = ('mindocr', 'paddleocr')
_SAMPLE_IMAGES = (
    (
        'invoice.png',
        (1400, 900),
        (
            'NORTHWIND SUPPLY INVOICE',
            'Invoice No: INV-2026-0042',
            'Supplier: Northwind Trading Ltd',
            'Date: 2026-10-31',
            'Item: AI workstation',
            'Quantity: 2',
            'Unit Price: 64280.00 CNY',
            'Total: 214960.00 CNY',
        ),
    ),
    (
        'receipt.png',
        (1000, 700),
        (
            'CITY MART RECEIPT',
            'Order: R-1088',
            'Coffee 32.50',
            'Sandwich 48.00',
            'Total 80.50 CNY',
        ),
    ),
    (
        'report.png',
        (1200, 700),
        (
            'QUARTERLY OPERATIONS REPORT',
            'Region: East China',
            'Revenue: 1285600.00 CNY',
            'Growth: 12.5 percent',
            'Owner: Li Ming',
        ),
    ),
)


def _load_font(size: int):
    from PIL import ImageFont

    for name in ('DejaVuSans.ttf', 'arial.ttf', 'Arial.ttf'):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _normalized_box(box: Sequence[float], width: int, height: int) -> List[float]:
    x0, y0, x1, y1 = (float(value) for value in box)
    return [
        round(max(0.0, min(1.0, x0 / width)), 6),
        round(max(0.0, min(1.0, y0 / height)), 6),
        round(max(0.0, min(1.0, x1 / width)), 6),
        round(max(0.0, min(1.0, y1 / height)), 6),
    ]


def create_sample_assets(asset_dir: Path = ASSET_DIR) -> Path:
    '''Create a deterministic, annotated OCR ranking corpus.'''
    from PIL import Image, ImageDraw

    asset_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = asset_dir / 'manifest.jsonl'
    records: List[Dict[str, Any]] = []
    font = _load_font(34)

    for name, (width, height), texts in _SAMPLE_IMAGES:
        image = Image.new('RGB', (width, height), 'white')
        draw = ImageDraw.Draw(image)
        boxes: List[List[float]] = []
        y = 60
        for index, text in enumerate(texts):
            draw.text((70, y), text, fill='black', font=font)
            box = draw.textbbox((70, y), text, font=font)
            boxes.append(_normalized_box(box, width, height))
            y += 78 if index else 96
        path = asset_dir / name
        image.save(path, format='PNG')
        records.append(
            {
                'imagePath': name,
                'texts': list(texts),
                'boxes': boxes,
                'blockTypes': ['paragraph'] * len(texts),
            }
        )

    with manifest_path.open('w', encoding='utf-8', newline='\n') as handle:
        for record in records:
            handle.write(
                json.dumps(record, ensure_ascii=False, separators=(',', ':'))
            )
            handle.write('\n')
    return manifest_path


def _jsonable(value: Any) -> Any:
    if hasattr(value, 'model_dump'):
        return value.model_dump(mode='json', exclude_none=False)
    if isinstance(value, Mapping):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='\n') as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    _jsonable(row), ensure_ascii=False, separators=(',', ':')
                )
            )
            handle.write('\n')


def _installed_backends() -> List[str]:
    from curator.operations.loading.ocr_adapters import (
        MindOcrAdapter,
        PaddleOcrAdapter,
    )

    result: List[str] = []
    for backend, adapter_type in (
        ('mindocr', MindOcrAdapter),
        ('paddleocr', PaddleOcrAdapter),
    ):
        if adapter_type.dependency_installed():
            result.append(backend)
    return result


def _backend_list(raw: Optional[Sequence[str]]) -> List[str]:
    if raw:
        return list(dict.fromkeys(raw))
    installed = _installed_backends()
    if not installed:
        raise SystemExit(
            'no OCR backend is installed; install '
            'pip install -e ".[ocr-mindocr]" or '
            'pip install -e ".[ocr-paddleocr]"'
        )
    return installed


def _ranking_rows(path: Path):
    return parse_ranking_document(
        yaml.safe_load(path.read_text(encoding='utf-8'))
    )


def _print_ranking(table) -> None:
    print('\nOCR backend ranking:')
    for row in table.rows:
        print(
            f'  {row.backend}: cer={row.cer:.6f}, '
            f'line_accuracy={row.line_accuracy:.6f}, '
            f'layout_f1={row.layout_f1:.6f}, '
            f'throughput_pages_s={row.throughput_pages_s:.3f}, '
            f'model_id={row.model_id}, device={row.device}'
        )


def _run_auto_route(
    *,
    image_path: Path,
    ranking_path: Path,
    table,
    output_dir: Path,
    device: str,
    lang: str,
    layout_analysis: bool,
    allow_model_download: bool,
) -> Dict[str, Any]:
    if not init_ray():
        raise SystemExit('Ray initialization failed')
    params = {
        'data_paths': [str(image_path)],
        'allowed_roots': [str(image_path.parent)],
        'backend': 'auto',
        'device': device,
        'lang': lang,
        'layout_analysis': layout_analysis,
        'backend_ranking_path': str(ranking_path),
        'allow_model_download': allow_model_download,
    }
    output_file = output_dir / 'ocr-auto-route.jsonl'
    pipeline = MultimodalPipeline(
        enable_tracking=False, checkpoint_config=_NO_CHECKPOINT
    )
    rows: List[Dict[str, Any]] = []
    try:
        pipeline.add_operation(ImageOcrOperation(params))
        result = pipeline.run()
        rows = [_jsonable(row) for row in result.dataset.iter_rows()]
        _write_jsonl(output_file, rows)
    finally:
        pipeline.cleanup()
        shutdown_ray()
    if not rows:
        raise SystemExit('auto OCR route produced no rows')
    row = rows[0]
    error = row.get('error')
    if error is not None:
        raise SystemExit('auto OCR route failed: ' + str(error))
    document = row.get('parsed_doc') or {}
    provenance = document.get('provenance') or {}
    reason = provenance.get('routing_reason')
    if reason != 'ranking':
        raise SystemExit('auto route did not use ranking: ' + str(reason))
    if provenance.get('ranking_version') != table.ranking_version:
        raise SystemExit('auto route ranking_version does not match the table')
    selected = provenance.get('backend')
    matching = next(
        (item for item in table.rows if item.backend == selected), None
    )
    if matching is None:
        raise SystemExit(
            'auto route selected a backend absent from the ranking table'
        )
    if provenance.get('model_id') != matching.model_id:
        raise SystemExit('auto route model_id does not match the ranking row')
    installed = set(_installed_backends())
    expected_rows = [
        item
        for item in table.rows
        if item.device == provenance.get('device')
        and item.eligible
        and item.backend in installed
    ]
    if expected_rows:
        expected = min(
            expected_rows,
            key=lambda item: (
                item.cer,
                -item.line_accuracy,
                -item.layout_f1,
                -item.throughput_pages_s,
                item.backend,
            ),
        )
        if selected != expected.backend:
            raise SystemExit(
                'auto route did not select the best installed ranked backend: '
                'selected=' + str(selected)
                + ', expected=' + expected.backend
            )
    print(
        'auto route: backend=' + str(selected)
        + ', device=' + str(provenance.get('device'))
        + ', routing_reason=' + str(reason)
        + ', ranking_version=' + str(provenance.get('ranking_version'))
        + ', evaluation_set=' + str(provenance.get('ranking_evaluation_set'))
    )
    print('auto route output: ' + str(output_file))
    return row


def run_example(args: argparse.Namespace) -> int:
    manifest_path = Path(args.manifest).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not manifest_path.exists():
        if manifest_path != DEFAULT_MANIFEST.resolve():
            raise SystemExit('manifest does not exist: ' + str(manifest_path))
        manifest_path = create_sample_assets(manifest_path.parent)
        print('created sample ranking corpus: ' + str(manifest_path))

    cases = load_benchmark_manifest(
        manifest_path, allowed_roots=[str(manifest_path.parent)]
    )
    backends = _backend_list(args.backend)
    ranking_path = Path(
        args.ranking_output or output_dir / 'ocr_backend_ranking.yaml'
    ).resolve()
    layout_analysis = not args.no_layout_analysis

    print('manifest: ' + str(manifest_path))
    print('cases: ' + str(len(cases)))
    print('backends: ' + ', '.join(backends))
    print('ranking output: ' + str(ranking_path))
    if args.dry_run:
        for case in cases:
            print(
                '  case: '
                + Path(case.image_path).name
                + ' ('
                + str(len(case.texts))
                + ' lines)'
            )
        return 0

    if args.route_only:
        if not ranking_path.is_file():
            raise SystemExit('ranking file does not exist: ' + str(ranking_path))
        table = _ranking_rows(ranking_path)
        _print_ranking(table)
        if args.no_route:
            return 0
        route_image = (
            Path(args.route_image).resolve()
            if args.route_image
            else Path(cases[0].image_path)
        )
        _run_auto_route(
            image_path=route_image,
            ranking_path=ranking_path,
            table=table,
            output_dir=output_dir,
            device=args.device,
            lang=args.lang,
            layout_analysis=layout_analysis,
            allow_model_download=args.allow_model_download,
        )
        return 0

    table = None
    for index, backend in enumerate(backends):
        payload = run_benchmark(
            manifest_path,
            backend=backend,
            device=args.device,
            lang=args.lang,
            layout_analysis=layout_analysis,
            ranking_version=args.ranking_version,
            evaluation_set_version=args.evaluation_set_version,
            allow_model_download=args.allow_model_download,
            allowed_roots=[str(manifest_path.parent)],
        )
        write_ranking_table(
            payload,
            ranking_path,
            merge=args.merge or index > 0,
        )
        table = _ranking_rows(ranking_path)
        row = payload['rows'][0]
        print(
            'measured ' + backend + ': '
            + 'cer=' + format(row['cer'], '.6f')
            + ', line_accuracy=' + format(row['line_accuracy'], '.6f')
            + ', layout_f1=' + format(row['layout_f1'], '.6f')
            + ', throughput_pages_s='
            + format(row['throughput_pages_s'], '.3f')
        )

    if table is None:
        raise SystemExit('no backend was measured')
    _print_ranking(table)
    print('ranking file: ' + str(ranking_path))

    if args.no_route:
        return 0
    route_image = (
        Path(args.route_image).resolve()
        if args.route_image
        else Path(cases[0].image_path)
    )
    if not route_image.is_file():
        raise SystemExit('route image does not exist: ' + str(route_image))
    _run_auto_route(
        image_path=route_image,
        ranking_path=ranking_path,
        table=table,
        output_dir=output_dir,
        device=args.device,
        lang=args.lang,
        layout_analysis=layout_analysis,
        allow_model_download=args.allow_model_download,
    )
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description='OCR backend ranking example')
    parser.add_argument(
        '--manifest',
        type=Path,
        default=DEFAULT_MANIFEST,
        help='annotated JSONL manifest; defaults to the bundled demo corpus',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help='directory for ranking and auto-route JSONL output',
    )
    parser.add_argument(
        '--ranking-output',
        type=Path,
        default=None,
        help='ranking YAML path; default is output-dir/ocr_backend_ranking.yaml',
    )
    parser.add_argument(
        '--backend',
        action='append',
        choices=_BACKENDS,
        default=None,
        help='backend to benchmark; repeat for multiple backends',
    )
    parser.add_argument(
        '--device', choices=('auto', 'npu', 'cuda', 'cpu'), default='cpu'
    )
    parser.add_argument('--lang', choices=('zh', 'en', 'mixed'), default='en')
    parser.add_argument('--no-layout-analysis', action='store_true')
    parser.add_argument('--allow-model-download', action='store_true')
    parser.add_argument(
        '--ranking-version',
        default=DEFAULT_RANKING_VERSION,
        help='ranking_version written to the table',
    )
    parser.add_argument(
        '--evaluation-set-version',
        default=DEFAULT_EVALUATION_SET_VERSION,
        help='evaluation_set_version written to the table',
    )
    parser.add_argument(
        '--merge',
        action='store_true',
        help='merge into an existing ranking YAML instead of replacing it',
    )
    parser.add_argument(
        '--route-image',
        type=Path,
        default=None,
        help='image used for backend=auto verification; defaults to first case',
    )
    parser.add_argument(
        '--no-route', action='store_true', help='skip auto-route verification'
    )
    parser.add_argument(
        '--route-only',
        action='store_true',
        help='use an existing ranking table without rerunning benchmarks',
    )
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    try:
        return run_example(args)
    except SystemExit:
        raise
    except Exception as exc:
        code = getattr(exc, 'code', None)
        message = getattr(exc, 'message', str(exc))
        print(
            'OCR ranking example failed: '
            + str(code or type(exc).__name__)
            + ': '
            + message
        )
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
