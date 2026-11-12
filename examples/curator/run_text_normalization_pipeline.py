#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Text regex normalization pipeline
- Input: data/texts/test_normalization.txt (15 lines, containing MM/DD/YYYY dates + Month DD, YYYY dates + $XX dollars)
- Pipeline: text_loading → text_normalization → text_saving
- Output: data/output/text_normalization/normalized.jsonl + normalized.txt

The text_normalization operator is already implemented (curator/operations/augmentation/text_normalization_operation.py),
using three regexes:
  1. _RE_MDYYYY       — MM/DD/YYYY → YYYY-MM-DD
  2. _RE_MONTH_DD_YYYY — Month DD, YYYY → YYYY-MM-DD
  3. _RE_USD          — $X(.XX) → X(.XX) USD
"""

import os
import sys

# Allow the script to run directly as `python3 examples/curator/run_text_normalization_pipeline.py`
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'examples', 'curator'))

from common import init_ray, shutdown_ray
from curator import MultimodalPipeline


CONFIG_PATH = os.path.join(PROJECT_ROOT, "examples", "yaml_configs", "text_normalization_pipeline.yaml")
INPUT_PATH  = os.path.join(PROJECT_ROOT, "data", "texts", "test_normalization.txt")
OUTPUT_DIR  = os.path.join(PROJECT_ROOT, "data", "output", "text_normalization")


def print_input_preview():
    print(f"[input] {INPUT_PATH}")
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            print(f"  {i:>2}: {line.rstrip()}")


def print_output_preview(rows):
    print(f"\n[output] records after normalization: {len(rows)}")
    for i, row in enumerate(rows, 1):
        print(f"  {i:>2}: {row.get('text', '')}")


def dump_plain_text(rows, out_path):
    """(Side path) Save the normalized result as a single .txt file for direct comparison with the original .txt"""
    with open(out_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(str(row.get("text", "")) + "\n")
    print(f"[dump ] saved plain text to {out_path}")


def print_side_by_side(src_path, rows):
    print(f"\n[compare] original vs normalized (line by line)")
    with open(src_path, "r", encoding="utf-8") as f:
        src_lines = [ln.rstrip("\n") for ln in f if ln.strip()]
    for i, (a, b) in enumerate(zip(src_lines, rows), 1):
        b_text = str(b.get("text", ""))
        marker = "  " if a == b_text else "→ "
        print(f"  {i:>2} {marker} {a}")
        if a != b_text:
            print(f"     {'':>2}  {b_text}")


def main():
    print("=" * 70)
    print("Text regex normalization pipeline")
    print("=" * 70)
    print_input_preview()

    init_ray(num_cpus=2)

    try:
        pipeline = MultimodalPipeline(config=CONFIG_PATH)

        print(f"\n[pipeline] operator chain ({len(pipeline.operations)} steps):")
        for i, op in enumerate(pipeline.operations, 1):
            print(f"  [{i}] {op.name}  params={op.params}")

        result = pipeline.run()
        rows = result.dataset.take_all() if hasattr(result.dataset, "take_all") else list(result.dataset.iter_rows())

        print_output_preview(rows)
        print_side_by_side(INPUT_PATH, rows)

        # The text_saving operator already persists output in pipeline.run()
        out_csv = os.path.join(OUTPUT_DIR, "normalized.csv")
        if os.path.exists(out_csv):
            if os.path.isfile(out_csv):
                print(f"[save ] CSV written: {out_csv} ({os.path.getsize(out_csv)} bytes)")
            elif os.path.isdir(out_csv):
                # Ray write_csv also creates a sharded directory in some versions
                files = [f for f in os.listdir(out_csv) if f.endswith(".csv")]
                if files:
                    print(f"[save ] CSV written: {out_csv}/ (with {len(files)} shards: {', '.join(files)})")

        # Save an additional plain-text copy for comparison with the original .txt
        dump_plain_text(rows, os.path.join(OUTPUT_DIR, "normalized.txt"))

        pipeline.cleanup()
        print("\n[done ] pipeline execution complete")
    finally:
        shutdown_ray()


if __name__ == "__main__":
    main()
