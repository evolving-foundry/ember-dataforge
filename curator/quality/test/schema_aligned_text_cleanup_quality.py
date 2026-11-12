#!/usr/bin/env python3
"""Run real text-cleaning operators and visualize aligned quality outputs.

This file intentionally reports only score fields that are declared by the
real ops-quality assessor implementations.  It does not invent additional
summary metrics such as evidence_score or coverage_rate, because those fields
would not align with downstream systems that consume assessor output schemas.

Run from the repository root:

    python curator/quality/test/schema_aligned_text_cleanup_quality.py

Outputs are written to:

    curator/quality/test/outputs/
"""

from __future__ import annotations

import copy
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import ray


SCRIPT_PATH = Path(__file__).resolve()
OUTPUT_DIR = SCRIPT_PATH.parent / "outputs"
OUTPUT_PREFIX = "schema-aligned-text-cleanup-quality"
SUMMARY_CSV = OUTPUT_DIR / f"{OUTPUT_PREFIX}.csv"
SUMMARY_SVG = OUTPUT_DIR / f"{OUTPUT_PREFIX}.svg"

OPERATOR_SEQUENCE = [
    "TextNormalizationOperation(text_key='text')",
    "RemoveRepetitionsPunctuationOperation(text_key='text', keep_repeat_char=2)",
]

TEXT_NOISE_METRIC = "text_noise_contamination"
TEXT_NORMALIZATION_METRIC = "text_normalization_validity"
SUPPORTED_SCORE_FIELDS = {
    TEXT_NOISE_METRIC: ("ngram_repetition_rate_mean",),
    TEXT_NORMALIZATION_METRIC: (
        "residual_unstandardized_pattern_rate",
    ),
}


@dataclass(frozen=True)
class MetricDefinition:
    metric_name: str
    score_name: str
    label: str
    direction: str
    description: str

    @property
    def key(self) -> str:
        return f"{self.metric_name}.{self.score_name}"


METRICS = [
    MetricDefinition(
        TEXT_NOISE_METRIC,
        "ngram_repetition_rate_mean",
        "N-gram repetition rate",
        "lower",
        "Declared output score from TextNoiseContaminationAssessor.",
    ),
    MetricDefinition(
        TEXT_NORMALIZATION_METRIC,
        "residual_unstandardized_pattern_rate",
        "Residual unstandardized pattern rate",
        "lower",
        "Declared output score from TextNormalizationValidityAssessor.",
    ),
]

EPSILON = 1.0e-12


def find_repo_root(start: Path = SCRIPT_PATH) -> Path:
    for candidate in [start.parent, *start.parents]:
        if (candidate / "curator").is_dir():
            return candidate
    raise RuntimeError(f"Could not locate repository root from {start}")


REPO_ROOT = find_repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from base.dataset import MultimodalDataset  # noqa: E402
from curator.operations.augmentation.remove_repetitions_punctuation_operation import (  # noqa: E402
    RemoveRepetitionsPunctuationOperation,
)
from curator.operations.augmentation.text_normalization_operation import (  # noqa: E402
    TextNormalizationOperation,
)
from curator.quality.ops_quality_assessor.config import (  # noqa: E402
    OpsQualityAssessmentConfig,
)
from curator.quality.ops_quality_assessor.context import (  # noqa: E402
    EvaluationContextBuilder,
)
from curator.quality.ops_quality_assessor.metrics.text_noise_contamination import (  # noqa: E402
    TextNoiseContaminationAssessor,
)
from curator.quality.ops_quality_assessor.metrics.text_normalization_validity import (  # noqa: E402
    TextNormalizationValidityAssessor,
)
from curator.quality.ops_quality_assessor.results import MetricResult  # noqa: E402


def build_raw_records() -> list[dict[str, Any]]:
    """Only raw inputs are authored here; after text is produced by operators."""
    specs = [
        (
            "butterfly_leaf_caption",
            "On 04/12/2025 a black butterfly butterfly with white spots rests rests on a green green leaf!!!",
        ),
        (
            "fruits_caption",
            "A plate plate of fruit includes apples apples oranges and bananas on a table!!!",
        ),
        (
            "soccer_player_caption",
            "A soccer soccer player in a striped striped uniform runs on the field near the ball!!!",
        ),
        (
            "portrait_hat_caption",
            "A woman woman wearing a hat hat looks toward the camera....",
        ),
        (
            "cooking_price_caption",
            "A cooking cooking scene includes ingredients beside a note recording $12,,,",
        ),
    ]

    records = []
    for item_id, before_text in specs:
        records.append(
            {
                "item_id": item_id,
                "before_text": before_text,
                "text": before_text,
            }
        )
    return records


def make_dataset(rows: list[dict[str, Any]]) -> MultimodalDataset:
    return MultimodalDataset(dataset=ray.data.from_items(rows), modalities=["text"])


def run_real_operator_pipeline(raw_records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    before_rows = copy.deepcopy(raw_records)
    after_dataset = make_dataset(copy.deepcopy(raw_records))
    after_dataset = TextNormalizationOperation({"text_key": "text"}).execute(after_dataset)
    after_dataset = RemoveRepetitionsPunctuationOperation({"text_key": "text", "keep_repeat_char": 2}).execute(after_dataset)
    after_rows = after_dataset.dataset.take_all()
    return before_rows, after_rows


def build_context(rows: list[dict[str, Any]], metric_configs: dict[str, dict[str, Any]]) :
    config = OpsQualityAssessmentConfig(
        sample_size=len(rows),
        enabled_metrics=list(metric_configs),
        executed_operations=OPERATOR_SEQUENCE,
        metric_configs=metric_configs,
    )
    return EvaluationContextBuilder(config).build_from_rows(make_dataset(rows), rows)


def assess_noise(rows: list[dict[str, Any]], text_key: str) -> MetricResult:
    context_rows = [{**row, "text": row[text_key]} for row in rows]
    context = build_context(
        context_rows,
        {
            TEXT_NOISE_METRIC: {
                "text_field": "text",
                "n_gram": 1,
            }
        },
    )
    return TextNoiseContaminationAssessor().assess(context)


def assess_normalization(rows: list[dict[str, Any]], after_key: str) -> MetricResult:
    context_rows = [
        {
            **row,
            "before_text": row["before_text"],
            "after_text": row[after_key],
        }
        for row in rows
    ]
    context = build_context(
        context_rows,
        {
            TEXT_NORMALIZATION_METRIC: {
                "before_text_key": "before_text",
                "after_text_key": "after_text",
                "use_default_rules": True,
            }
        },
    )
    return TextNormalizationValidityAssessor().assess(context)


def collect_metric_scores(before_rows: list[dict[str, Any]], after_rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    before_noise = assess_noise(before_rows, "before_text")
    after_noise = assess_noise(after_rows, "text")

    before_baseline_rows = [{**row, "text": row["before_text"]} for row in before_rows]
    before_by_id = {row["item_id"]: row for row in before_rows}
    after_comparison_rows = [
        {
            **after_row,
            "before_text": before_by_id[after_row["item_id"]]["before_text"],
        }
        for after_row in after_rows
    ]
    before_norm = assess_normalization(before_baseline_rows, "before_text")
    after_norm = assess_normalization(after_comparison_rows, "text")

    return {
        "before": {
            f"{before_noise.metric_name}.{score_name}": score
            for score_name, score in before_noise.scores.items()
        }
        | {
            f"{before_norm.metric_name}.{score_name}": score
            for score_name, score in before_norm.scores.items()
        },
        "after": {
            f"{after_noise.metric_name}.{score_name}": score
            for score_name, score in after_noise.scores.items()
        }
        | {
            f"{after_norm.metric_name}.{score_name}": score
            for score_name, score in after_norm.scores.items()
        },
        "metric_results": {
            "before_noise": before_noise.model_dump(),
            "after_noise": after_noise.model_dump(),
            "before_normalization": before_norm.model_dump(),
            "after_normalization": after_norm.model_dump(),
        },
    }


def build_summary(scores: dict[str, dict[str, float]]) -> list[dict[str, Any]]:
    rows = []
    for metric in METRICS:
        before_value = float(scores["before"].get(metric.key, 0.0))
        after_value = float(scores["after"].get(metric.key, 0.0))
        raw_delta = after_value - before_value
        improvement = raw_delta if metric.direction == "higher" else before_value - after_value
        rows.append(
            {
                "metric": metric.key,
                "metric_name": metric.metric_name,
                "score_name": metric.score_name,
                "label": metric.label,
                "direction": metric.direction,
                "before": before_value,
                "after": after_value,
                "raw_delta": raw_delta,
                "quality_improvement": improvement,
                "relative_improvement": improvement / max(abs(before_value), EPSILON),
                "description": metric.description,
            }
        )
    return rows


def assert_metric_schema_aligned(summary_rows: list[dict[str, Any]]) -> None:
    unsupported = []
    for row in summary_rows:
        if row["score_name"] not in SUPPORTED_SCORE_FIELDS.get(row["metric_name"], ()):
            unsupported.append(str(row["metric"]))
    if unsupported:
        raise AssertionError("summary contains unsupported score fields: " + ", ".join(unsupported))


def assert_no_metric_regression(summary_rows: list[dict[str, Any]]) -> None:
    failures = []
    for row in summary_rows:
        if row["quality_improvement"] < -1.0e-9:
            failures.append(
                f"{row['metric']} regressed: before={row['before']:.6f}, after={row['after']:.6f}"
            )
    if failures:
        raise AssertionError("; ".join(failures))


def assert_operator_generated_after(before_rows: list[dict[str, Any]], after_rows: list[dict[str, Any]]) -> None:
    unchanged = []
    after_by_id = {row["item_id"]: row for row in after_rows}
    for before_row in before_rows:
        after_row = after_by_id[before_row["item_id"]]
        if before_row["before_text"] == after_row["text"]:
            unchanged.append(before_row["item_id"])
    if unchanged:
        raise AssertionError("operator pipeline did not change text for: " + ", ".join(unchanged))


def format_metric_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        **row,
        "before": f"{row['before']:.6f}",
        "after": f"{row['after']:.6f}",
        "raw_delta": f"{row['raw_delta']:.6f}",
        "quality_improvement": f"{row['quality_improvement']:.6f}",
        "relative_improvement": f"{row['relative_improvement']:.6f}",
    }


def write_summary_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "metric",
        "metric_name",
        "score_name",
        "label",
        "direction",
        "before",
        "after",
        "raw_delta",
        "quality_improvement",
        "relative_improvement",
        "description",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(format_metric_row(row))


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def try_write_matplotlib_svg(summary_rows: list[dict[str, Any]], path: Path) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return False

    before_color = "#64748b"
    after_color = "#10b981"
    grid_color = "#e2e8f0"
    text_color = "#0f172a"
    secondary_text_color = "#64748b"

    labels = [str(row["score_name"]) for row in summary_rows]
    metric_details = [
        f"{row['metric_name']}; lower is better; improvement={float(row['quality_improvement']):.3f}"
        for row in summary_rows
    ]
    before_values = [clamp01(float(row["before"])) for row in summary_rows]
    after_values = [clamp01(float(row["after"])) for row in summary_rows]
    y_positions = list(range(len(summary_rows)))
    bar_height = 0.28

    fig_height = max(3.8, 1.15 * len(summary_rows) + 1.9)
    fig, ax = plt.subplots(figsize=(11.8, fig_height))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    before_bars = ax.barh(
        [position - bar_height / 2 for position in y_positions],
        before_values,
        height=bar_height,
        color=before_color,
        label="before",
    )
    after_bars = ax.barh(
        [position + bar_height / 2 for position in y_positions],
        after_values,
        height=bar_height,
        color=after_color,
        label="after from real ops",
    )

    ax.set_xlim(0.0, 1.04)
    ax.set_xticks([tick / 10.0 for tick in range(0, 11, 2)])
    ax.xaxis.grid(True, color=grid_color, linewidth=1.0)
    ax.set_axisbelow(True)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, color=text_color, fontsize=10, fontweight="bold")
    ax.invert_yaxis()
    ax.tick_params(axis="x", colors=secondary_text_color)
    ax.tick_params(axis="y", length=0, pad=16)

    for spine in ax.spines.values():
        spine.set_visible(False)

    for index, detail in enumerate(metric_details):
        ax.text(
            -0.01,
            index + 0.42,
            detail,
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="center",
            fontsize=8.5,
            color=secondary_text_color,
            clip_on=False,
        )

    for bars, values in ((before_bars, before_values), (after_bars, after_values)):
        for bar, value in zip(bars, values):
            label_x = min(value + 0.014, 1.012)
            ax.text(
                label_x,
                bar.get_y() + bar.get_height() / 2,
                f"{value:.3f}",
                va="center",
                ha="left",
                fontsize=9,
                color=text_color,
            )

    ax.set_title(
        "Schema-aligned real operator quality scores: before vs after",
        loc="left",
        fontsize=18,
        fontweight="bold",
        color=text_color,
        pad=22,
    )
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.08),
        ncol=2,
        frameon=False,
        fontsize=10,
    )
    fig.subplots_adjust(left=0.42, right=0.97, top=0.78, bottom=0.12)
    fig.savefig(path, format="svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return True


def write_svg(summary_rows: list[dict[str, Any]], path: Path) -> None:
    if try_write_matplotlib_svg(summary_rows, path):
        return
    raise RuntimeError("matplotlib is required to write the schema-aligned quality SVG")


def run_evaluation(*, write_outputs: bool = True) -> dict[str, Any]:
    raw_records = build_raw_records()
    before_rows, after_rows = run_real_operator_pipeline(raw_records)
    metric_scores = collect_metric_scores(before_rows, after_rows)
    summary_rows = build_summary(metric_scores)
    assert_metric_schema_aligned(summary_rows)
    assert_no_metric_regression(summary_rows)
    assert_operator_generated_after(before_rows, after_rows)

    if write_outputs:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        write_summary_csv(summary_rows, SUMMARY_CSV)
        write_svg(summary_rows, SUMMARY_SVG)

    return {
        "before_rows": before_rows,
        "after_rows": after_rows,
        "metric_scores": metric_scores,
        "summary": summary_rows,
    }


def main() -> None:
    result = run_evaluation(write_outputs=True)
    print("Generated schema-aligned real-operator quality evaluation outputs:")
    print(f"  summary csv : {SUMMARY_CSV}")
    print(f"  summary svg : {SUMMARY_SVG}")
    print()
    print("Real operator sequence:")
    for operation in OPERATOR_SEQUENCE:
        print(f"  - {operation}")
    print()
    print("Declared output score fields only:")
    for metric_name, score_names in SUPPORTED_SCORE_FIELDS.items():
        for score_name in score_names:
            print(f"  - {metric_name}.{score_name}")
    print()
    print("Metric summary:")
    for row in result["summary"]:
        print(
            f"  {row['metric']:<72} before={row['before']:.4f} "
            f"after={row['after']:.4f} improvement(down)={row['quality_improvement']:.4f}"
        )


if __name__ == "__main__":
    main()
