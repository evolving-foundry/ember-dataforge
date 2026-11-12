from __future__ import annotations

import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any


def export_video_qa_showcase_dataset(
    input_jsonl: str | Path,
    output_dir: str | Path,
    *,
    split: str = "train",
    copy_media: bool = True,
) -> dict[str, Any]:
    input_path = Path(input_jsonl)
    output_path = Path(output_dir)
    rows = _read_jsonl(input_path)
    if not rows:
        raise ValueError(f"No rows found in {input_path}")

    data_dir = output_path / "data"
    metadata_dir = output_path / "metadata"
    videos_dir = output_path / "videos"
    frames_dir = output_path / "frames"
    data_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    if copy_media:
        videos_dir.mkdir(parents=True, exist_ok=True)
        frames_dir.mkdir(parents=True, exist_ok=True)

    video_records = []
    qa_records = []
    copied_videos: dict[str, str] = {}
    copied_frames: dict[str, str] = {}

    for video_index, row in enumerate(rows, start=1):
        video_id = str(row.get("sample_id") or row.get("id") or f"video_{video_index:06d}")
        video_rel_path = _materialize_media(
            src=row.get("video_path"),
            dst_dir=videos_dir,
            output_root=output_path,
            default_name=f"{video_id}.mp4",
            cache=copied_videos,
            copy_media=copy_media,
        )
        descriptions_by_id = {
            str(unit.get("description_id") or unit.get("id")): unit
            for unit in row.get("description_units") or []
            if unit.get("description_id") or unit.get("id")
        }
        captions_by_granularity = _captions_by_granularity(row.get("description_units") or [])
        video_records.append(
            {
                "video_id": video_id,
                "video": video_rel_path,
                "source_video": row.get("video_path"),
                "source_repo": row.get("source_repo"),
                "source_repo_url": row.get("source_repo_url"),
                "source_path": row.get("source_path"),
                "domain": row.get("domain"),
                "task_family": row.get("task_family"),
                "duration_sec": row.get("video_duration_sec"),
                "fps": row.get("video_fps"),
                "width": row.get("width"),
                "height": row.get("height"),
                "sha256": row.get("sha256"),
                "captions": captions_by_granularity,
                "segments": [_public_segment(segment) for segment in row.get("segments") or []],
            }
        )

        for qae_index, triplet in enumerate(row.get("qae_triplets") or [], start=1):
            record_id = f"{video_id}__qa_{qae_index:04d}"
            evidence_refs = triplet.get("evidence_refs") or {}
            frame_paths = []
            for frame_path in ((evidence_refs.get("image") or {}).get("frame_paths") or []):
                rel_path = _materialize_media(
                    src=frame_path,
                    dst_dir=frames_dir / video_id,
                    output_root=output_path,
                    default_name=Path(str(frame_path)).name,
                    cache=copied_frames,
                    copy_media=copy_media,
                )
                if rel_path:
                    frame_paths.append(rel_path)
            description_ids = list((evidence_refs.get("text") or {}).get("description_ids") or [])
            evidence_captions = [
                _public_description(descriptions_by_id[description_id])
                for description_id in description_ids
                if description_id in descriptions_by_id
            ]
            temporal_unit_ids = list((evidence_refs.get("temporal") or {}).get("temporal_unit_ids") or [])
            qa_records.append(
                {
                    "id": record_id,
                    "split": split,
                    "video_id": video_id,
                    "video": video_rel_path,
                    "question": triplet.get("question"),
                    "answer": triplet.get("answer"),
                    "question_type": triplet.get("question_type"),
                    "triplet_type": triplet.get("triplet_type"),
                    "grounding_level": triplet.get("grounding_level"),
                    "evidence": {
                        "frames": frame_paths,
                        "sample_ids": list((evidence_refs.get("image") or {}).get("sample_ids") or []),
                        "description_ids": description_ids,
                        "temporal_unit_ids": temporal_unit_ids,
                        "captions": evidence_captions,
                    },
                    "analysis": triplet.get("llm_materialization_analysis"),
                    "quality": _quality_for_triplet(row.get("quality_records") or [], triplet)
                    or _video_quality_summary(row),
                    "provenance": {
                        "source_repo": row.get("source_repo"),
                        "source_repo_url": row.get("source_repo_url"),
                        "source_path": row.get("source_path"),
                        "generator_model": row.get("generator_model")
                        or triplet.get("materializer_model_path"),
                        "generator_prompt_version": row.get("generator_prompt_version"),
                        "schema_version": row.get("schema_version"),
                        "recipe": row.get("recipe"),
                    },
                }
            )

    _write_jsonl(data_dir / f"{split}.jsonl", qa_records)
    _write_jsonl(data_dir / "videos.jsonl", video_records)
    summary = _build_summary(
        input_path=input_path,
        output_path=output_path,
        split=split,
        rows=rows,
        video_records=video_records,
        qa_records=qa_records,
        copy_media=copy_media,
    )
    _write_json(metadata_dir / "dataset_summary.json", summary)
    _write_readme(output_path / "README.md", summary, qa_records[:3])
    return summary


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
        encoding="utf-8",
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _materialize_media(
    *,
    src: Any,
    dst_dir: Path,
    output_root: Path,
    default_name: str,
    cache: dict[str, str],
    copy_media: bool,
) -> str | None:
    if not src:
        return None
    src_path = Path(str(src))
    cache_key = str(src_path)
    if cache_key in cache:
        return cache[cache_key]
    if copy_media and src_path.exists():
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst_path = dst_dir / (src_path.name or default_name)
        shutil.copy2(src_path, dst_path)
        rel_path = dst_path.relative_to(output_root).as_posix()
    else:
        rel_path = str(src)
    cache[cache_key] = rel_path
    return rel_path


def _captions_by_granularity(description_units: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {
        "frame": [],
        "event": [],
        "segment": [],
        "context": [],
    }
    for unit in description_units:
        granularity = str(unit.get("granularity") or "").strip().lower()
        if granularity not in result:
            continue
        result[granularity].append(_public_description(unit))
    return result


def _public_description(unit: dict[str, Any]) -> dict[str, Any]:
    anchors = unit.get("anchor_refs") or {}
    return {
        "id": unit.get("description_id") or unit.get("id"),
        "granularity": unit.get("granularity"),
        "text": unit.get("description_text"),
        "scope": unit.get("description_scope"),
        "segment_id": unit.get("segment_id"),
        "sample_ids": anchors.get("sample_ids") or [],
        "segment_ids": anchors.get("segment_ids") or [],
    }


def _public_segment(segment: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": segment.get("segment_id") or segment.get("id"),
        "timestamp_range": segment.get("timestamp_range"),
        "frame_range": segment.get("frame_range"),
        "partition_mode": segment.get("partition_mode"),
        "boundary_type": segment.get("boundary_type"),
    }


def _quality_for_triplet(
    quality_records: list[dict[str, Any]],
    triplet: dict[str, Any],
) -> dict[str, Any] | None:
    triplet_id = str(triplet.get("triplet_id") or triplet.get("id") or "")
    scaffold_id = str(triplet.get("scaffold_id") or "")
    for record in quality_records:
        record_triplet_id = str(
            record.get("triplet_id")
            or record.get("target_triplet_id")
            or record.get("id")
            or ""
        )
        record_scaffold_id = str(record.get("scaffold_id") or "")
        if triplet_id and record_triplet_id == triplet_id:
            return _public_quality(record)
        if scaffold_id and record_scaffold_id == scaffold_id:
            return _public_quality(record)
    return None


def _public_quality(record: dict[str, Any]) -> dict[str, Any]:
    return {
        key: record.get(key)
        for key in (
            "quality_id",
            "id",
            "target_triplet_id",
            "quality_mode",
            "scores",
            "decision",
            "reason",
            "rank_in_video",
            "assigned_dataset_tier",
        )
        if key in record
    }


def _video_quality_summary(row: dict[str, Any]) -> dict[str, Any] | None:
    quality_scores = row.get("quality_scores") or {}
    audit_status = row.get("audit_status")
    if not quality_scores and not audit_status:
        return None
    return {
        "audit_status": audit_status,
        "scores": quality_scores,
    }


def _build_summary(
    *,
    input_path: Path,
    output_path: Path,
    split: str,
    rows: list[dict[str, Any]],
    video_records: list[dict[str, Any]],
    qa_records: list[dict[str, Any]],
    copy_media: bool,
) -> dict[str, Any]:
    question_type_counts = Counter(record.get("question_type") for record in qa_records)
    caption_counts = Counter()
    for video_record in video_records:
        for granularity, captions in (video_record.get("captions") or {}).items():
            caption_counts[granularity] += len(captions or [])
    return {
        "dataset_name": output_path.name,
        "format_version": "video-qa-showcase-v1",
        "source_internal_jsonl": str(input_path),
        "output_dir": str(output_path),
        "split": split,
        "num_videos": len(video_records),
        "num_qa_records": len(qa_records),
        "question_type_counts": dict(question_type_counts),
        "caption_counts": dict(caption_counts),
        "copy_media": copy_media,
        "source_repos": sorted(
            {
                str(row.get("source_repo"))
                for row in rows
                if row.get("source_repo")
            }
        ),
        "generator_models": sorted(
            {
                str(row.get("generator_model"))
                for row in rows
                if row.get("generator_model")
            }
        ),
        "files": {
            "qa_jsonl": f"data/{split}.jsonl",
            "videos_jsonl": "data/videos.jsonl",
            "summary": "metadata/dataset_summary.json",
            "readme": "README.md",
        },
    }


def _write_readme(path: Path, summary: dict[str, Any], examples: list[dict[str, Any]]) -> None:
    lines = [
        f"# {summary['dataset_name']}",
        "",
        "Small video temporal-QA showcase export generated from DataForge internal synthetic output.",
        "",
        "## Contents",
        "",
        f"- QA records: `{summary['files']['qa_jsonl']}`",
        f"- Video-level metadata: `{summary['files']['videos_jsonl']}`",
        f"- Summary: `{summary['files']['summary']}`",
        "- Media paths are relative to this dataset directory when `copy_media=true`.",
        "",
        "## Statistics",
        "",
        f"- Videos: {summary['num_videos']}",
        f"- QA records: {summary['num_qa_records']}",
        f"- Question types: `{json.dumps(summary['question_type_counts'], ensure_ascii=False)}`",
        f"- Caption counts: `{json.dumps(summary['caption_counts'], ensure_ascii=False)}`",
        "",
        "## Record Schema",
        "",
        "`data/train.jsonl` is one QAE record per line with fields: `id`, `video_id`, `video`, `question`, `answer`, `question_type`, `evidence`, `analysis`, `quality`, and `provenance`.",
        "",
        "`data/videos.jsonl` is one video record per line with source metadata, temporal segments, and frame/event/segment/context captions.",
        "",
        "## Examples",
        "",
    ]
    for example in examples:
        lines.extend(
            [
                f"### {example['id']}",
                "",
                f"- Video: `{example.get('video')}`",
                f"- Question type: `{example.get('question_type')}`",
                f"- Question: {example.get('question')}",
                f"- Answer: {example.get('answer')}",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")
