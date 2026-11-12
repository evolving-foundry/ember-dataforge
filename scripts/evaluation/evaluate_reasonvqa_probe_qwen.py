from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from curator.utils.jsonl import read_jsonl, write_jsonl  # noqa: E402


MC_MODES = {"mc", "blank_mc"}
ROOT_ORACLE_OPEN_MODES = {"open_root_oracle"}

CATEGORY_NAMES = {
    0: "location",
    1: "person",
    3: "country",
    5: "birthplace",
    6: "capital",
    8: "knowledge",
}

ANSWER_ALIASES = {
    "france": {"france", "french republic", "the french republic"},
    "germany": {"germany", "federal republic of germany"},
    "italy": {"italy", "italian republic"},
    "spain": {"spain", "kingdom of spain"},
    "paris": {"paris"},
    "berlin": {"berlin"},
    "rome": {"rome", "roma"},
    "madrid": {"madrid"},
    "amiens": {"amiens"},
    "osnabruck": {"osnabruck", "osnabrück"},
    "naples": {"naples", "napoli"},
    "valladolid": {"valladolid"},
}


@dataclass(frozen=True)
class Score:
    exact: bool
    substring: bool
    semantic_proxy: bool
    parsed_choice: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Qwen2.5-VL on the 100-row ReasonVQA-style probe."
    )
    parser.add_argument("--model", type=Path, default=os.environ.get("REASONVQA_MODEL_PATH"))
    parser.add_argument("--data-dir", type=Path, default=os.environ.get("REASONVQA_DATA_DIR"))
    parser.add_argument("--output-dir", type=Path, default=os.environ.get("REASONVQA_OUTPUT_DIR"))
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=("open", "open_root_oracle", "mc", "blank_mc"),
        default=["open", "mc"],
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens-open", type=int, default=32)
    parser.add_argument("--max-new-tokens-mc", type=int, default=8)
    parser.add_argument("--min-visual-tokens", type=int, default=256)
    parser.add_argument("--max-visual-tokens", type=int, default=768)
    parser.add_argument(
        "--shuffle-choices",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Deterministically shuffle multiple-choice option order to avoid position leakage.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--local-files-only", action="store_true", default=True)
    args = parser.parse_args()
    if not args.model or not args.data_dir:
        parser.error("--model/REASONVQA_MODEL_PATH and --data-dir/REASONVQA_DATA_DIR are required")
    args.model = Path(args.model)
    args.data_dir = Path(args.data_dir)
    args.output_dir = Path(args.output_dir) if args.output_dir else None
    return args


def normalize_text(value: str) -> str:
    value = value.lower().strip()
    value = value.replace("osnabrück", "osnabruck")
    value = re.sub(r"[\(\)\[\]\{\}\"'`]", " ", value)
    value = re.sub(r"[^a-z0-9]+", " ", value)
    value = re.sub(r"\b(the|a|an)\b", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def answer_aliases(answer: str) -> set[str]:
    normalized = normalize_text(answer)
    return {normalized, *ANSWER_ALIASES.get(normalized, set())}


def contains_answer(prediction: str, answer: str) -> bool:
    pred = f" {normalize_text(prediction)} "
    return any(f" {normalize_text(alias)} " in pred for alias in answer_aliases(answer))


def fuzzy_answer_match(prediction: str, answer: str) -> bool:
    pred_norm = normalize_text(prediction)
    if not pred_norm:
        return False
    aliases = answer_aliases(answer)
    if any(alias in pred_norm.split() for alias in aliases):
        return True
    return max(
        SequenceMatcher(None, pred_norm, normalize_text(alias)).ratio()
        for alias in aliases
    ) >= 0.86


def score_open_prediction(prediction: str, answer: str) -> Score:
    pred_norm = normalize_text(prediction)
    answer_norm = normalize_text(answer)
    exact = pred_norm == answer_norm
    substring = contains_answer(prediction, answer)
    semantic_proxy = substring or fuzzy_answer_match(prediction, answer)
    return Score(exact=exact, substring=substring, semantic_proxy=semantic_proxy)


def parse_mc_prediction(prediction: str, choices: list[str]) -> str | None:
    text = prediction.strip()
    upper = text.upper()
    letter_match = re.search(
        r"(?:^|\b)(?:ANSWER\s*(?:IS|:)?\s*)?([A-D])(?:\b|[\.\)])",
        upper,
    )
    if letter_match:
        return letter_match.group(1)

    normalized_prediction = normalize_text(text)
    matched = [
        index
        for index, choice in enumerate(choices)
        if normalize_text(choice) and normalize_text(choice) in normalized_prediction
    ]
    if len(matched) == 1:
        return chr(65 + matched[0])
    return None


def score_mc_prediction(prediction: str, row: dict[str, Any]) -> Score:
    parsed_choice = parse_mc_prediction(prediction, row["answers"])
    gold_index = row["correct"].index(1)
    gold_letter = chr(65 + gold_index)
    exact = parsed_choice == gold_letter
    gold_answer = row["answers"][gold_index]
    open_score = score_open_prediction(prediction, gold_answer)
    return Score(
        exact=exact,
        substring=open_score.substring,
        semantic_proxy=exact or open_score.semantic_proxy,
        parsed_choice=parsed_choice,
    )


def choice_view(row: dict[str, Any], shuffle_choices: bool) -> dict[str, Any]:
    if not shuffle_choices:
        return row
    gold_answer = correct_answer(row)
    rng = random.Random(int(row["question_id"]) + 104729)
    negatives = [answer for answer in row["answers"] if answer != gold_answer]
    rng.shuffle(negatives)
    gold_index = int(row["question_id"]) % len(row["answers"])
    answers = negatives[:gold_index] + [gold_answer] + negatives[gold_index:]
    correct = [1 if answer == gold_answer else 0 for answer in answers]
    if sum(correct) != 1:
        raise ValueError(f"Shuffled choices lost unique answer for qid={row['question_id']}")
    view = dict(row)
    view["answers"] = answers
    view["correct"] = correct
    return view


def correct_answer(row: dict[str, Any]) -> str:
    return row["answers"][row["correct"].index(1)]


def is_mc_mode(mode: str) -> bool:
    return mode in MC_MODES


def root_oracle_context(row: dict[str, Any]) -> str:
    root_concept = row.get("root_concept") or {}
    inaturalist = row.get("inaturalist") or {}
    scientific_name = (
        root_concept.get("taxon_name")
        or inaturalist.get("scientific_name")
        or row.get("entity_name")
        or "unknown"
    )
    taxon_rank = root_concept.get("taxon_rank") or "unknown"
    qid = (
        root_concept.get("wikidata_qid")
        or (row.get("root") or {}).get("qid")
        or row.get("entity_id")
        or "unknown"
    )
    return (
        "Oracle root concept: the organism in the image has already been "
        f"identified as the taxon with scientific name '{scientific_name}' "
        f"(taxon rank: {taxon_rank}; Wikidata QID: {qid}). Use this as the "
        "starting Wikidata entity for the question."
    )


def build_prompt(row: dict[str, Any], mode: str) -> str:
    if is_mc_mode(mode):
        choices = "\n".join(
            f"{chr(65 + index)}. {answer}" for index, answer in enumerate(row["answers"])
        )
        return (
            "Answer the visual multiple-choice question. Use the image and any "
            "required factual knowledge. Choose one option.\n"
            f"Question: {row['question']}\n"
            f"Choices:\n{choices}\n"
            "Respond with only the option letter (A, B, C, or D)."
        )
    if mode in ROOT_ORACLE_OPEN_MODES:
        return (
            "Answer the visual knowledge question using the supplied root-concept "
            "oracle and any required factual knowledge. Give only the short answer, "
            "with no explanation.\n"
            f"{root_oracle_context(row)}\n"
            f"Question: {row['question']}"
        )
    return (
        "Answer the visual question using the image and any required factual "
        "knowledge. Give only the short answer, with no explanation.\n"
        f"Question: {row['question']}"
    )


def load_rows(data_dir: Path, limit: int | None) -> list[dict[str, Any]]:
    qa_rows = {row["question_id"]: row for row in read_jsonl(data_dir / "train.jsonl")}
    ann_rows = {row["question_id"]: row for row in read_jsonl(data_dir / "train_ann.jsonl")}
    merged_path = data_dir / "knowledge_reasoning_vqa_merged.jsonl"
    if merged_path.exists():
        merged_rows = {row["question_id"]: row for row in read_jsonl(merged_path)}
    else:
        merged_rows = {}

    rows = []
    for question_id in sorted(qa_rows):
        row = dict(merged_rows.get(question_id, {}))
        row.update(ann_rows[question_id])
        row.update(qa_rows[question_id])
        rows.append(row)
    if limit is not None:
        rows = rows[:limit]
    return rows


def load_done_predictions(path: Path) -> dict[tuple[int, str], dict[str, Any]]:
    if not path.exists():
        return {}
    done = {}
    for row in read_jsonl(path):
        done[(int(row["question_id"]), str(row["mode"]))] = row
    return done


def iter_batches(items: list[tuple[dict[str, Any], str]], batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def generate_batch(
    model: Any,
    processor: Any,
    rows_and_modes: list[tuple[dict[str, Any], str]],
    device: str,
    max_new_tokens_open: int,
    max_new_tokens_mc: int,
    shuffle_choices: bool,
) -> list[dict[str, Any]]:
    import torch
    from PIL import Image

    texts = []
    images = []
    prompt_rows = []
    for row, mode in rows_and_modes:
        prompt_row = choice_view(row, shuffle_choices) if is_mc_mode(mode) else row
        prompt_rows.append(prompt_row)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": build_prompt(prompt_row, mode)},
                ],
            }
        ]
        texts.append(
            processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        )
        if mode == "blank_mc":
            images.append(Image.new("RGB", (448, 448), "white"))
        else:
            images.append(Image.open(row["image_path"]).convert("RGB"))

    inputs = processor(
        text=texts,
        images=images,
        padding=True,
        return_tensors="pt",
    ).to(device)

    max_new_tokens = (
        max_new_tokens_mc
        if all(is_mc_mode(mode) for _, mode in rows_and_modes)
        else max_new_tokens_open
    )
    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=processor.tokenizer.pad_token_id,
        )
    trimmed_ids = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
    ]
    predictions = processor.batch_decode(
        trimmed_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    results = []
    for (row, mode), prompt_row, prediction in zip(
        rows_and_modes, prompt_rows, predictions
    ):
        answer = correct_answer(row)
        score = (
            score_mc_prediction(prediction, prompt_row)
            if is_mc_mode(mode)
            else score_open_prediction(
                prediction,
                answer,
            )
        )
        gold_letter = chr(65 + prompt_row["correct"].index(1))
        results.append(
            {
                "question_id": row["question_id"],
                "mode": mode,
                "question": row["question"],
                "image_id": row["image_id"],
                "image_path": row["image_path"],
                "entity_name": row["entity_name"],
                "hop": row["hop"],
                "property_id": row["property_id"],
                "property_label": row["property_label"],
                "categories": row["categories"],
                "category_names": [CATEGORY_NAMES.get(cat, str(cat)) for cat in row["categories"]],
                "has_scene_graph": row["has_scene_graph"],
                "route": row["route"],
                "answers": prompt_row["answers"],
                "original_answers": row["answers"],
                "choices_shuffled": int(is_mc_mode(mode) and prompt_row["answers"] != row["answers"]),
                "gold_answer": answer,
                "gold_letter": gold_letter,
                "raw_prediction": prediction.strip(),
                "parsed_choice": score.parsed_choice,
                "exact": int(score.exact),
                "substring": int(score.substring),
                "semantic_proxy": int(score.semantic_proxy),
            }
        )
    return results


def accuracy(values: list[int]) -> float:
    return float(sum(values) / len(values)) if values else math.nan


def summarize_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n": len(rows),
        "exact_acc": accuracy([row["exact"] for row in rows]),
        "substring_acc": accuracy([row["substring"] for row in rows]),
        "semantic_proxy_acc": accuracy([row["semantic_proxy"] for row in rows]),
        "parse_rate": accuracy([int(bool(row.get("parsed_choice"))) for row in rows])
        if rows and is_mc_mode(rows[0]["mode"])
        else None,
    }


def summarize(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "overall": {},
        "by_hop": {},
        "by_entity": {},
        "by_property": {},
        "by_category": {},
        "by_scene_graph": {},
    }
    for mode in sorted({row["mode"] for row in predictions}):
        mode_rows = [row for row in predictions if row["mode"] == mode]
        metrics["overall"][mode] = summarize_group(mode_rows)

        def add_bucket(section: str, key: str, bucket_rows: list[dict[str, Any]]) -> None:
            metrics[section].setdefault(mode, {})[str(key)] = summarize_group(bucket_rows)

        for hop in sorted({row["hop"] for row in mode_rows}):
            add_bucket("by_hop", f"{hop}-hop", [row for row in mode_rows if row["hop"] == hop])
        for entity in sorted({row["entity_name"] for row in mode_rows}):
            add_bucket(
                "by_entity",
                entity,
                [row for row in mode_rows if row["entity_name"] == entity],
            )
        for prop in sorted({row["property_label"] for row in mode_rows}):
            add_bucket(
                "by_property",
                prop,
                [row for row in mode_rows if row["property_label"] == prop],
            )
        categories = sorted({cat for row in mode_rows for cat in row["category_names"]})
        for category in categories:
            add_bucket(
                "by_category",
                category,
                [row for row in mode_rows if category in row["category_names"]],
            )
        for scene_graph in sorted({row["has_scene_graph"] for row in mode_rows}):
            add_bucket(
                "by_scene_graph",
                f"has_scene_graph={scene_graph}",
                [row for row in mode_rows if row["has_scene_graph"] == scene_graph],
            )
    return metrics


def write_report(output_dir: Path, metrics: dict[str, Any], predictions: list[dict[str, Any]]) -> None:
    run_config = metrics.get("_run_config", {})
    lines = [
        "# Qwen2.5-VL ReasonVQA Probe Evaluation",
        "",
        f"- Samples: {len({row['question_id'] for row in predictions})}",
        f"- Prediction rows: {len(predictions)}",
        f"- Multiple-choice choices shuffled: {run_config.get('shuffle_choices', 'unknown')}",
        "- Open-ended metrics: normalized exact, substring, semantic_proxy.",
        "- Multiple-choice metric: exact option-letter accuracy; semantic_proxy also credits answer-text matches.",
        "",
        "## Overall",
        "",
        "| Mode | N | Exact/MC Acc | Substring Acc | Semantic Proxy Acc | Parse Rate |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for mode, values in metrics["overall"].items():
        parse_rate = values["parse_rate"]
        lines.append(
            "| {mode} | {n} | {exact:.3f} | {substring:.3f} | {semantic:.3f} | {parse_rate} |".format(
                mode=mode,
                n=values["n"],
                exact=values["exact_acc"],
                substring=values["substring_acc"],
                semantic=values["semantic_proxy_acc"],
                parse_rate="-" if parse_rate is None else f"{parse_rate:.3f}",
            )
        )

    for section, title in [
        ("by_hop", "By Hop"),
        ("by_property", "By Property"),
        ("by_entity", "By Entity"),
        ("by_category", "By Category"),
        ("by_scene_graph", "By Scene Graph"),
    ]:
        lines += [
            "",
            f"## {title}",
            "",
            "| Mode | Bucket | N | Exact/MC Acc | Substring Acc | Semantic Proxy Acc |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
        for mode, buckets in metrics[section].items():
            for bucket, values in buckets.items():
                lines.append(
                    "| {mode} | {bucket} | {n} | {exact:.3f} | {substring:.3f} | {semantic:.3f} |".format(
                        mode=mode,
                        bucket=bucket,
                        n=values["n"],
                        exact=values["exact_acc"],
                        substring=values["substring_acc"],
                        semantic=values["semantic_proxy_acc"],
                    )
                )

    errors = [
        row
        for row in predictions
        if is_mc_mode(row["mode"]) and not row["exact"]
    ][:20]
    if errors:
        lines += ["", "## First Multiple-Choice Errors", ""]
        for row in errors:
            lines.append(
                "- q{qid:03d} hop={hop}: gold={gold} pred={pred} raw={raw!r} question={question}".format(
                    qid=row["question_id"],
                    hop=row["hop"],
                    gold=row["gold_letter"],
                    pred=row["parsed_choice"],
                    raw=row["raw_prediction"],
                    question=row["question"],
                )
            )

    (output_dir / "metrics.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or (args.data_dir / "eval_qwen2_5_vl_7b" / timestamp)
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / "predictions.jsonl"

    rows = load_rows(args.data_dir, args.limit)
    done = load_done_predictions(prediction_path) if args.resume else {}
    work_items = [
        (row, mode)
        for mode in args.modes
        for row in rows
        if (int(row["question_id"]), mode) not in done
    ]

    processor = AutoProcessor.from_pretrained(
        args.model,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
        min_pixels=args.min_visual_tokens * 28 * 28,
        max_pixels=args.max_visual_tokens * 28 * 28,
    )
    processor.tokenizer.padding_side = "left"
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype="auto",
        device_map="auto",
        local_files_only=args.local_files_only,
    )
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    predictions = list(done.values())
    for batch_index, batch in enumerate(iter_batches(work_items, args.batch_size), start=1):
        batch_results = generate_batch(
            model=model,
            processor=processor,
            rows_and_modes=batch,
            device=device,
            max_new_tokens_open=args.max_new_tokens_open,
            max_new_tokens_mc=args.max_new_tokens_mc,
            shuffle_choices=args.shuffle_choices,
        )
        predictions.extend(batch_results)
        write_jsonl(prediction_path, predictions)
        print(
            json.dumps(
                {
                    "batch": batch_index,
                    "completed": len(predictions),
                    "total": len(done) + len(work_items),
                    "last_question_id": batch_results[-1]["question_id"],
                    "last_mode": batch_results[-1]["mode"],
                    "last_prediction": batch_results[-1]["raw_prediction"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    predictions = sorted(predictions, key=lambda row: (row["mode"], int(row["question_id"])))
    write_jsonl(prediction_path, predictions)
    metrics = summarize(predictions)
    metrics["_run_config"] = {
        "model": str(args.model),
        "data_dir": str(args.data_dir),
        "modes": args.modes,
        "limit": args.limit,
        "batch_size": args.batch_size,
        "shuffle_choices": args.shuffle_choices,
        "min_visual_tokens": args.min_visual_tokens,
        "max_visual_tokens": args.max_visual_tokens,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_report(output_dir, metrics, predictions)
    print(json.dumps({"output_dir": str(output_dir), "metrics": metrics["overall"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
