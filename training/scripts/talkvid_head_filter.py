#!/usr/bin/env python3
"""
TalkVid head-pose filter presets for dataset generation.
"""

import json
import os
from pathlib import Path


HEAD_FILTER_PRESETS = {
    "soft": {
        "min_avg_orientation": 90.0,
        "min_min_orientation": 82.0,
    },
    "medium": {
        "min_avg_orientation": 92.0,
        "min_min_orientation": 84.0,
    },
    "strict": {
        "min_avg_orientation": 94.0,
        "min_min_orientation": 88.0,
    },
}

HEAD_FILTER_MODE_CHOICES = ["off", "soft", "medium", "strict"]


def describe_head_filter_mode(mode):
    mode = (mode or "off").strip().lower()
    if mode == "off":
        return "off"
    cfg = HEAD_FILTER_PRESETS[mode]
    return (
        f"{mode}: avg_orientation>={cfg['min_avg_orientation']:.0f}, "
        f"min_orientation>={cfg['min_min_orientation']:.0f}"
    )


def head_scores_from_item(item):
    return ((item or {}).get("head_detail") or {}).get("scores") or {}


def evaluate_head_filter_scores(scores, mode):
    mode = (mode or "off").strip().lower()
    if mode == "off":
        return True, "mode=off"

    cfg = HEAD_FILTER_PRESETS[mode]
    avg_orientation = scores.get("avg_orientation")
    min_orientation = scores.get("min_orientation")
    if avg_orientation is None or min_orientation is None:
        return False, "missing_orientation_scores"

    if float(avg_orientation) < cfg["min_avg_orientation"]:
        return (
            False,
            f"avg_orientation={float(avg_orientation):.2f} < {cfg['min_avg_orientation']:.2f}",
        )
    if float(min_orientation) < cfg["min_min_orientation"]:
        return (
            False,
            f"min_orientation={float(min_orientation):.2f} < {cfg['min_min_orientation']:.2f}",
        )
    return (
        True,
        f"avg_orientation={float(avg_orientation):.2f}, "
        f"min_orientation={float(min_orientation):.2f}",
    )


def evaluate_head_filter_item(item, mode):
    return evaluate_head_filter_scores(head_scores_from_item(item), mode)


def raw_sidecar_path(raw_path):
    return str(Path(raw_path).with_suffix(".json"))


def load_raw_sidecar_item(raw_path):
    sidecar_path = raw_sidecar_path(raw_path)
    if not os.path.exists(sidecar_path):
        return None, "missing_sidecar"
    try:
        with open(sidecar_path) as f:
            return json.load(f), None
    except Exception as exc:
        return None, f"sidecar_error:{type(exc).__name__}"


def evaluate_head_filter_raw_clip(raw_path, mode):
    mode = (mode or "off").strip().lower()
    if mode == "off":
        return True, "mode=off"
    item, load_error = load_raw_sidecar_item(raw_path)
    if item is None:
        return False, load_error or "missing_sidecar"
    return evaluate_head_filter_item(item, mode)
