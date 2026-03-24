#!/usr/bin/env python3
"""
Assign quality tiers to already processed TalkVid samples and write allowlists.
"""

import argparse
import json
import os
from collections import Counter
from pathlib import Path


QUALITY_LEVELS = ["confident", "medium", "unconfident"]
MIN_QUALITY_NAMES = QUALITY_LEVELS


def load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def classify_sample(processed_meta, raw_meta):
    reasons = []
    if processed_meta.get("bad_sample", False):
        reasons.append("bad_sample")
        if processed_meta.get("bad_reasons"):
            reasons.extend([f"bad_reason:{x}" for x in processed_meta.get("bad_reasons", [])])
        return "rejected", reasons

    quality = processed_meta.get("quality") or {}
    scores = ((raw_meta or {}).get("head_detail") or {}).get("scores") or {}

    coverage = float(quality.get("detection_coverage") or 0.0)
    kept_ratio = float(quality.get("kept_ratio") or 0.0)
    min_edge = float(quality.get("min_edge_margin_ratio") or 0.0)
    max_center = float(quality.get("max_center_jump_ratio") or 999.0)
    max_size = float(quality.get("max_size_jump_ratio") or 999.0)
    avg_orientation = scores.get("avg_orientation")
    min_orientation = scores.get("min_orientation")
    avg_rotation = scores.get("avg_rotation")
    min_rotation = scores.get("min_rotation")

    if avg_orientation is None or min_orientation is None:
        reasons.append("missing_head_scores")
        avg_orientation = 0.0
        min_orientation = 0.0
    else:
        avg_orientation = float(avg_orientation)
        min_orientation = float(min_orientation)

    if avg_rotation is None or min_rotation is None:
        avg_rotation = 0.0
        min_rotation = 0.0
        reasons.append("missing_rotation_scores")
    else:
        avg_rotation = float(avg_rotation)
        min_rotation = float(min_rotation)

    confident_ok = (
        avg_orientation >= 94.0
        and min_orientation >= 88.0
        and avg_rotation >= 90.0
        and min_rotation >= 80.0
        and coverage >= 0.68
        and kept_ratio >= 0.95
        and min_edge >= 0.04
        and max_center <= 0.09
        and max_size <= 0.10
    )
    if confident_ok:
        return "confident", reasons

    medium_ok = (
        avg_orientation >= 90.0
        and min_orientation >= 82.0
        and avg_rotation >= 86.0
        and min_rotation >= 72.0
        and coverage >= 0.55
        and kept_ratio >= 0.80
        and min_edge >= 0.02
        and max_center <= 0.18
        and max_size <= 0.18
    )
    if medium_ok:
        return "medium", reasons

    return "unconfident", reasons


def selected_for_min_quality(tier, min_quality):
    if tier == "rejected":
        return False
    if min_quality == "confident":
        return tier == "confident"
    if min_quality == "medium":
        return tier in {"confident", "medium"}
    if min_quality == "unconfident":
        return tier in {"confident", "medium", "unconfident"}
    raise ValueError(f"unsupported min_quality={min_quality}")


def write_list(path, names):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for name in sorted(names):
            f.write(name + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", required=True)
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    processed_dir = Path(args.processed_dir)
    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = output_dir / "quality_manifest.jsonl"
    tier_names = {tier: [] for tier in ["confident", "medium", "unconfident", "rejected"]}
    stats = Counter()

    with manifest_path.open("w") as out:
        for name in sorted(os.listdir(processed_dir)):
            speaker_dir = processed_dir / name
            if not speaker_dir.is_dir():
                continue

            frames_path = speaker_dir / "frames.npy"
            mel_path = speaker_dir / "mel.npy"
            bbox_path = speaker_dir / "bbox.json"
            if not (frames_path.exists() and mel_path.exists() and bbox_path.exists()):
                continue

            processed_meta = load_json(bbox_path)
            if processed_meta is None:
                tier = "rejected"
                reasons = ["invalid_bbox_json"]
                raw_meta = None
            else:
                raw_meta = load_json(raw_dir / f"{name}.json")
                tier, reasons = classify_sample(processed_meta, raw_meta)

            tier_names[tier].append(name)
            stats[tier] += 1
            record = {
                "name": name,
                "tier": tier,
                "usable": tier != "rejected",
                "reasons": reasons,
                "processed_dir": str(speaker_dir),
                "raw_sidecar": str(raw_dir / f"{name}.json"),
                "bbox_bad_sample": None if processed_meta is None else bool(processed_meta.get("bad_sample", False)),
                "bbox_bad_reasons": None if processed_meta is None else processed_meta.get("bad_reasons", []),
                "quality": None if processed_meta is None else processed_meta.get("quality", {}),
                "head_scores": ((raw_meta or {}).get("head_detail") or {}).get("scores") or {},
                "dover_scores": (raw_meta or {}).get("dover_scores"),
                "cotracker_ratio": (raw_meta or {}).get("cotracker_ratio"),
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")

    write_list(str(output_dir / "tier_confident.txt"), tier_names["confident"])
    write_list(str(output_dir / "tier_medium.txt"), tier_names["medium"])
    write_list(str(output_dir / "tier_unconfident.txt"), tier_names["unconfident"])
    write_list(str(output_dir / "tier_rejected.txt"), tier_names["rejected"])

    usable = []
    for tier in QUALITY_LEVELS:
        usable.extend(tier_names[tier])
    write_list(str(output_dir / "min_unconfident.txt"), usable)

    medium_plus = tier_names["confident"] + tier_names["medium"]
    write_list(str(output_dir / "min_medium.txt"), medium_plus)
    write_list(str(output_dir / "min_confident.txt"), tier_names["confident"])

    summary = {
        "processed_dir": str(processed_dir),
        "raw_dir": str(raw_dir),
        "manifest": str(manifest_path),
        "counts": {
            "confident": stats["confident"],
            "medium": stats["medium"],
            "unconfident": stats["unconfident"],
            "rejected": stats["rejected"],
            "usable_total": stats["confident"] + stats["medium"] + stats["unconfident"],
        },
        "lists": {
            "tier_confident": str(output_dir / "tier_confident.txt"),
            "tier_medium": str(output_dir / "tier_medium.txt"),
            "tier_unconfident": str(output_dir / "tier_unconfident.txt"),
            "tier_rejected": str(output_dir / "tier_rejected.txt"),
            "min_confident": str(output_dir / "min_confident.txt"),
            "min_medium": str(output_dir / "min_medium.txt"),
            "min_unconfident": str(output_dir / "min_unconfident.txt"),
        },
    }
    with (output_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"processed_dir={processed_dir}")
    print(f"raw_dir={raw_dir}")
    for tier in ["confident", "medium", "unconfident", "rejected"]:
        print(f"{tier}={stats[tier]}")
    print(f"usable_total={stats['confident'] + stats['medium'] + stats['unconfident']}")
    print(f"saved_manifest={manifest_path}")
    print(f"saved_summary={output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
