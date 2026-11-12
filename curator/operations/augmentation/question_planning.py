from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from ...utils.video_ops_utils import unique_preserve_order


QUESTION_PLAN_PROFILE_DIR = Path(__file__).with_name("question_plan_profiles")
DEFAULT_QUESTION_PLAN_PROFILE = "generic"

def choose_evidence_aware_question_plan(
    scaffold_summary,
    evidence_refs=None,
    index=0,
    question_plans=None,
    evidence_index=None,
):
    plans = normalize_question_plans(question_plans)
    features = extract_evidence_features(evidence_refs or {}, evidence_index or {})
    eligible, rejected = [], []
    for plan in plans:
        matched, reasons = question_plan_applicability(plan, features)
        if matched:
            eligible.append({"plan": plan, "matched_conditions": reasons})
        else:
            rejected.append({"type": plan["type"], "reasons": reasons})

    candidate_plans = [item["plan"] for item in eligible] or plans
    selected_plan = choose_question_plan(
        scaffold_summary=scaffold_summary,
        evidence_refs=evidence_refs,
        index=index,
        question_plans=candidate_plans,
    )
    selected_match = next(
        (item for item in eligible if item["plan"]["type"] == selected_plan["type"]),
        None,
    )
    selected_plan = dict(selected_plan)
    selected_plan["selected_by"] = (
        "evidence_aware_rule_selector"
        if eligible
        else "evidence_aware_rule_selector_fallback"
    )
    selected_plan["matched_conditions"] = (
        selected_match["matched_conditions"] if selected_match else []
    )
    selected_plan["eligible"] = bool(selected_match)
    return {
        "question_plan": selected_plan,
        "planning_trace": {
            "selected_by": selected_plan["selected_by"],
            "profile_features": features,
            "eligible_plan_types": [item["plan"]["type"] for item in eligible],
            "rejected_candidates": rejected,
            "selected_plan_type": selected_plan["type"],
            "matched_conditions": selected_plan["matched_conditions"],
        },
    }


def extract_evidence_features(evidence_refs, evidence_index):
    image_refs = (evidence_refs or {}).get("image") or {}
    text_refs = (evidence_refs or {}).get("text") or {}
    sample_ids = value_list(image_refs, "sample_ids")
    frame_paths = value_list(image_refs, "frame_paths")
    frame_refs = value_list(image_refs, "frame_refs")
    description_ids = value_list(text_refs, "description_ids")
    text_signal_ids = value_list(text_refs, "text_signal_ids")
    temporal_ids = evidence_temporal_ref_ids(evidence_refs)

    descriptions = [
        evidence_index[item_id]["item"]
        for item_id in description_ids
        if evidence_index.get(item_id, {}).get("item")
    ]
    temporal_items = [
        evidence_index[item_id]["item"]
        for item_id in temporal_ids
        if evidence_index.get(item_id, {}).get("item")
    ]
    sample_items = [
        evidence_index[item_id]["item"]
        for item_id in sample_ids + frame_refs
        if evidence_index.get(item_id, {}).get("item")
    ]

    granularities = unique_preserve_order(
        str(
            item.get("granularity")
            or description_mode_granularity(item.get("description_mode"))
            or ""
        )
        .strip()
        .lower()
        for item in descriptions
        if str(item.get("granularity") or item.get("description_mode") or "").strip()
    )
    frame_texts = [
        item_text(item)
        for item in descriptions
        if str(
            item.get("granularity")
            or description_mode_granularity(item.get("description_mode"))
            or ""
        )
        .strip()
        .lower()
        == "frame"
        and item_text(item)
    ]
    timestamps = []
    for item in sample_items:
        timestamp = item.get("timestamp_sec", item.get("timestamp"))
        if timestamp is not None:
            try:
                timestamps.append(float(timestamp))
            except (TypeError, ValueError):
                pass
    visual_time_span_sec = max(timestamps) - min(timestamps) if len(timestamps) >= 2 else 0.0
    temporal_span_sec = max(
        (time_range_duration(get_time_range(item)) for item in temporal_items),
        default=0.0,
    )
    frame_caption_jaccard = average_pairwise_jaccard(frame_texts)
    num_frames = max(len(frame_paths), len(sample_ids), len(frame_refs))
    return {
        "num_frames": num_frames,
        "num_temporal_units": len(temporal_ids),
        "num_descriptions": len(description_ids),
        "num_text_signals": len(text_signal_ids),
        "caption_granularities": granularities,
        "has_frame_caption": "frame" in granularities,
        "has_event_caption": "event" in granularities,
        "has_segment_caption": "segment" in granularities,
        "has_context_caption": "context" in granularities,
        "has_temporal_span": temporal_span_sec > 0,
        "visual_time_span_sec": round(visual_time_span_sec, 4),
        "temporal_span_sec": round(temporal_span_sec, 4),
        "frame_caption_jaccard": (
            round(frame_caption_jaccard, 4)
            if frame_caption_jaccard is not None
            else None
        ),
        "has_text_signal": bool(text_signal_ids),
        "sample_ids": sample_ids,
        "description_ids": description_ids,
        "temporal_unit_ids": temporal_ids,
    }


def question_plan_applicability(plan, features):
    cond = plan.get("applicable_when") or {}
    matched, rejected = [], []

    def require(ok, label):
        (matched if ok else rejected).append(label)

    min_frames = cond.get("min_evidence_frames")
    if min_frames is not None:
        require(
            features["num_frames"] >= int(min_frames),
            f"min_evidence_frames>={int(min_frames)}",
        )
    max_frames = cond.get("max_evidence_frames")
    if max_frames is not None:
        require(
            features["num_frames"] <= int(max_frames),
            f"max_evidence_frames<={int(max_frames)}",
        )
    if cond.get("requires_temporal_span"):
        require(features["has_temporal_span"], "requires_temporal_span")
    min_visual_span = cond.get("min_visual_time_span_sec")
    if min_visual_span is not None:
        require(
            float(features.get("visual_time_span_sec") or 0.0) >= float(min_visual_span),
            f"min_visual_time_span_sec>={float(min_visual_span)}",
        )
    required_granularities = set(value_list(cond, "required_granularities"))
    if required_granularities:
        available = set(features.get("caption_granularities") or [])
        require(
            required_granularities <= available,
            "required_granularities=" + ",".join(sorted(required_granularities)),
        )
    any_granularities = set(value_list(cond, "any_granularities"))
    if any_granularities:
        available = set(features.get("caption_granularities") or [])
        require(
            bool(any_granularities & available),
            "any_granularities=" + ",".join(sorted(any_granularities)),
        )
    if cond.get("requires_text_signal"):
        require(features["has_text_signal"], "requires_text_signal")
    max_jaccard = cond.get("max_frame_caption_jaccard")
    if max_jaccard is not None:
        observed = features.get("frame_caption_jaccard")
        require(
            observed is not None and observed <= float(max_jaccard),
            f"max_frame_caption_jaccard<={float(max_jaccard)}",
        )

    return not rejected, matched if not rejected else rejected


def resolve_question_plan_config(params):
    params = params or {}
    profile_name = (
        str(params.get("question_plan_profile") or DEFAULT_QUESTION_PLAN_PROFILE).strip()
        or DEFAULT_QUESTION_PLAN_PROFILE
    )
    merged = load_question_plan_profile(profile_name)

    path_config = load_question_plan_config_file(params.get("question_plan_config_path"))
    if path_config:
        merged = merge_question_plan_config(merged, path_config)
    inline_config = params.get("question_plan_config")
    if inline_config:
        merged = merge_question_plan_config(
            merged,
            parse_inline_question_plan_config(inline_config),
        )
    if params.get("question_plans"):
        merged["question_plans"] = params["question_plans"]
    if params.get("generic_question_prefixes"):
        merged["generic_question_prefixes"] = params["generic_question_prefixes"]

    merged["question_plans"] = normalize_question_plans(merged.get("question_plans"))
    merged["generic_question_prefixes"] = tuple(
        str(prefix).strip().lower()
        for prefix in value_list(merged, "generic_question_prefixes")
        if str(prefix).strip()
    )
    return merged


def load_question_plan_profile(profile_name):
    profile_name = str(profile_name or DEFAULT_QUESTION_PLAN_PROFILE).strip()
    if not profile_name:
        profile_name = DEFAULT_QUESTION_PLAN_PROFILE
    if "/" in profile_name or "\\" in profile_name or profile_name.startswith("."):
        raise ValueError(f"Invalid question_plan_profile: {profile_name}")
    profile_path = QUESTION_PLAN_PROFILE_DIR / f"{profile_name}.json"
    if not profile_path.exists():
        known = ", ".join(available_question_plan_profiles())
        raise ValueError(
            f"Unsupported question_plan_profile: {profile_name}. "
            f"Available profiles: {known}."
        )
    profile = load_question_plan_config_file(profile_path)
    if not isinstance(profile, dict):
        raise ValueError(f"question plan profile must be a JSON object: {profile_path}")
    merged = dict(profile)
    merged["profile"] = str(merged.get("profile") or profile_name)
    return merged


def available_question_plan_profiles():
    if not QUESTION_PLAN_PROFILE_DIR.exists():
        return []
    return sorted(path.stem for path in QUESTION_PLAN_PROFILE_DIR.glob("*.json"))


def load_question_plan_config_file(config_path):
    if not config_path:
        return None
    path = Path(config_path).expanduser()
    if not path.exists():
        raise ValueError(f"question_plan_config_path does not exist: {path}")
    raw_text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        return json.loads(raw_text)
    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:
            raise ValueError("YAML question_plan_config_path requires PyYAML.") from exc
        return yaml.safe_load(raw_text)
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        try:
            import yaml
        except ImportError as exc:
            raise ValueError(f"Unsupported question plan config format: {path}") from exc
        return yaml.safe_load(raw_text)


def parse_inline_question_plan_config(config):
    if isinstance(config, str):
        text = config.strip()
        if not text:
            return {}
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            try:
                import yaml
            except ImportError as exc:
                raise ValueError(
                    "String question_plan_config must be JSON unless PyYAML is installed."
                ) from exc
            return yaml.safe_load(text)
    return config


def merge_question_plan_config(base_config, override_config):
    if override_config is None:
        return base_config
    if isinstance(override_config, list):
        override_config = {"question_plans": override_config}
    if not isinstance(override_config, dict):
        raise ValueError("question plan config must be a dict or list of question plans.")
    merged = dict(base_config)
    if override_config.get("profile"):
        merged["profile"] = str(override_config["profile"])
    if override_config.get("question_plans") is not None:
        merged["question_plans"] = override_config["question_plans"]
    if override_config.get("generic_question_prefixes") is not None:
        merged["generic_question_prefixes"] = override_config["generic_question_prefixes"]
    return merged


def normalize_question_plans(question_plans):
    plans = []
    if question_plans is None:
        question_plans = load_question_plan_profile(DEFAULT_QUESTION_PLAN_PROFILE).get(
            "question_plans"
        )
    for index, raw_plan in enumerate(question_plans or [], start=1):
        if not isinstance(raw_plan, dict):
            raise ValueError(f"question plan #{index} must be a dict.")
        plan = {
            "type": str(raw_plan.get("type") or "").strip(),
            "instruction": str(raw_plan.get("instruction") or "").strip(),
            "template": str(raw_plan.get("template") or "").strip(),
        }
        if not plan["type"] or not plan["instruction"] or not plan["template"]:
            raise ValueError(
                f"question plan #{index} must include type, instruction, and template."
            )
        try:
            plan["weight"] = max(1, int(raw_plan.get("weight", 1)))
        except (TypeError, ValueError):
            raise ValueError(f"question plan #{index} weight must be an integer.")
        for optional_key in (
            "applicable_when",
            "evidence_policy",
        ):
            if optional_key in raw_plan:
                plan[optional_key] = raw_plan[optional_key]
        plans.append(plan)
    if not plans:
        raise ValueError("At least one question plan is required.")
    return tuple(plans)


def weighted_question_plan_pool(question_plans):
    pool = []
    for plan in normalize_question_plans(question_plans):
        pool.extend(dict(plan) for _ in range(int(plan.get("weight", 1))))
    return pool


def choose_question_plan(scaffold_summary, evidence_refs=None, index=0, question_plans=None):
    plan_pool = weighted_question_plan_pool(question_plans)
    seed_text = json.dumps(
        {
            "summary": str(scaffold_summary or ""),
            "evidence_refs": evidence_refs or {},
            "index": int(index),
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    digest = hashlib.sha1(seed_text.encode("utf-8")).hexdigest()
    plan_index = (int(digest[:8], 16) + int(index)) % len(plan_pool)
    return dict(plan_pool[plan_index])


def is_generic_repeated_question(question, generic_prefixes=None):
    normalized = " ".join(str(question or "").strip().lower().split())
    if generic_prefixes is None:
        generic_prefixes = (
            load_question_plan_profile(DEFAULT_QUESTION_PLAN_PROFILE).get(
                "generic_question_prefixes"
            )
            or ()
        )
    return any(normalized.startswith(prefix) for prefix in generic_prefixes)


def build_rule_question_from_scaffold(
    scaffold_unit,
    question_plan=None,
    question_plans=None,
):
    focus = str(
        scaffold_unit.get("question_focus")
        or scaffold_unit.get("scaffold_summary")
        or "this video"
    ).strip()
    question_plan = question_plan or choose_question_plan(
        scaffold_summary=focus,
        evidence_refs=build_evidence_refs_by_modality(scaffold_unit),
        index=0,
        question_plans=question_plans,
    )
    if question_plan["type"] == "visible_action":
        return f"What specific action is visible in the evidence for {focus}?"
    if question_plan["type"] == "object_state":
        return f"What object state is directly visible in the evidence for {focus}?"
    if question_plan["type"] == "spatial_relation":
        return (
            "Where are the main visible elements positioned in relation to each "
            f"other for {focus}?"
        )
    if question_plan["type"] == "temporal_change":
        return f"What visible change occurs across the referenced evidence for {focus}?"
    if question_plan["type"] == "scene_context":
        return f"What scene context is visible around the evidence for {focus}?"
    return f"What immediate outcome is directly supported by the evidence for {focus}?"


def scaffold_image_refs(scaffold):
    frame_refs, sample_ids, frame_paths = [], [], []
    visual_items = list(scaffold.get("visual_evidence") or [])
    for node in scaffold.get("action_nodes") or []:
        if not isinstance(node, dict):
            continue
        if node.get("node_type") == "frame" and node.get("node_ref"):
            frame_refs.append(str(node["node_ref"]))
        sample_ids.extend(value_list(node, "sample_id", "sample_ids"))
        frame_paths.extend(value_list(node, "frame_path", "frame_paths"))
        visual_items.extend(node.get("visual_evidence") or [])
    for item in visual_items:
        if isinstance(item, dict):
            sample_ids.extend(value_list(item, "sample_id", "sample_ids"))
            frame_paths.extend(value_list(item, "frame_path", "frame_paths"))
    return {
        "frame_refs": unique_preserve_order(frame_refs),
        "sample_ids": unique_preserve_order(sample_ids),
        "frame_paths": unique_preserve_order(frame_paths),
    }


def build_evidence_refs_by_modality(scaffold):
    return {
        "text": {
            "description_ids": value_list(scaffold, "source_description_ids"),
            "text_signal_ids": value_list(scaffold, "source_text_signal_ids"),
            "evidence_unit_ids": value_list(scaffold, "source_evidence_unit_ids"),
        },
        "image": scaffold_image_refs(scaffold),
        "temporal": {
            "temporal_unit_ids": value_list(
                scaffold,
                "source_temporal_unit_ids",
                "temporal_unit_id",
            ),
        },
    }


def value_list(item, *keys):
    values = []
    for key in keys:
        value = (item or {}).get(key)
        if value is None:
            continue
        if isinstance(value, (list, tuple, set)):
            values.extend(str(part).strip() for part in value if str(part).strip())
        elif str(value).strip():
            values.append(str(value).strip())
    return unique_preserve_order(values)


def item_text(item):
    return " ".join(
        value_list(
            item,
            "evidence_summary",
            "scaffold_summary",
            "description_text",
            "text",
            "content",
            "summary",
            "temporal_label",
        )
    )


def get_time_range(item):
    raw_range = (item or {}).get("timestamp_range") or {}
    start = raw_range.get("start_sec", raw_range.get("start"))
    end = raw_range.get("end_sec", raw_range.get("end"))
    if start is None:
        start = (item or {}).get("start_sec", (item or {}).get("start"))
    if end is None:
        end = (item or {}).get("end_sec", (item or {}).get("end"))
    if start is None or end is None:
        return None
    start, end = float(start), float(end)
    if end < start:
        start, end = end, start
    return {"start_sec": start, "end_sec": end}


def evidence_temporal_ref_ids(evidence_refs):
    return value_list((evidence_refs or {}).get("temporal") or {}, "temporal_unit_ids")


def description_mode_granularity(mode):
    value = str(mode or "").strip().lower()
    if value.endswith("_description"):
        return value.replace("_description", "")
    return value


def time_range_duration(time_range):
    if not time_range:
        return 0.0
    return max(0.0, float(time_range["end_sec"]) - float(time_range["start_sec"]))


def average_pairwise_jaccard(texts):
    texts = [text for text in texts or [] if text]
    if len(texts) < 2:
        return None
    scores = []
    for left_index in range(len(texts)):
        for right_index in range(left_index + 1, len(texts)):
            scores.append(token_jaccard(texts[left_index], texts[right_index]))
    return sum(scores) / len(scores) if scores else None


def token_jaccard(left, right):
    left_tokens = set(re.findall(r"[A-Za-z]+", str(left or "").lower()))
    right_tokens = set(re.findall(r"[A-Za-z]+", str(right or "").lower()))
    if not left_tokens and not right_tokens:
        return 1.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)
