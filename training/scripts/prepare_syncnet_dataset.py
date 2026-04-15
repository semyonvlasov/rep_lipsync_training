#!/usr/bin/env python3
"""
Precompute sync alignment for selected lazy entries and freeze the eligible
SyncNet dataset manifest before building train/val splits.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

TRAINING_ROOT = Path(__file__).resolve().parents[1]
if str(TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINING_ROOT))

from config_loader import load_config
from data import LipSyncDataset
from data.sync_alignment import load_sync_alignment


def timestamp() -> str:
    return time.strftime("%H:%M:%S")


def log(message: str) -> None:
    print(f"[{timestamp()}] {message}", flush=True)


def parse_tiers(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def selected_roots(training_root: Path, hdtf_tiers: list[str], talkvid_tiers: list[str]) -> list[tuple[str, str, Path]]:
    roots: list[tuple[str, str, Path]] = []
    for tier in hdtf_tiers:
        roots.append(("hdtf", tier, training_root / "data" / "hdtf" / "processed" / "_lazy_imports" / tier))
    for tier in talkvid_tiers:
        roots.append(("talkvid", tier, training_root / "data" / "talkvid" / "processed" / "_lazy_imports" / tier))
    return roots


def build_sync_alignment_kwargs(cfg: dict) -> dict:
    sync_cfg = cfg.get("data", {}).get("sync_alignment", {})
    return {
        "sync_alignment_enabled": sync_cfg.get("enabled", True),
        "sync_alignment_compute_if_missing": sync_cfg.get("compute_if_missing", True),
        "sync_alignment_guard_mel_ticks": sync_cfg.get("guard_mel_ticks", 10),
        "sync_alignment_search_mel_ticks": sync_cfg.get("search_mel_ticks", 10),
        "sync_alignment_samples": sync_cfg.get("samples", 0),
        "sync_alignment_sample_density_per_5s": sync_cfg.get("sample_density_per_5s", 10.0),
        "sync_alignment_seed": sync_cfg.get("seed", 20260403),
        "sync_alignment_min_start_gap_ratio": sync_cfg.get("min_start_gap_ratio", 0.0),
        "sync_alignment_start_gap_multiple": sync_cfg.get("start_gap_multiple", 0),
        "sync_alignment_device": sync_cfg.get("device", "auto"),
        "sync_alignment_batch_size": sync_cfg.get("batch_size", 640),
        "sync_alignment_outlier_trim_ratio": sync_cfg.get("outlier_trim_ratio", 0.2),
        "sync_alignment_min_consensus_ratio": sync_cfg.get("min_consensus_ratio"),
        "sync_alignment_max_shift_mad": sync_cfg.get("max_shift_mad"),
        "sync_alignment_syncnet_checkpoint": sync_cfg.get("syncnet_checkpoint"),
        "sync_alignment_write_manifest": sync_cfg.get("write_manifest", True),
    }


def collect_lazy_entries(helper: LipSyncDataset, roots: list[tuple[str, str, Path]]) -> tuple[list[dict], dict]:
    entries_by_name: dict[str, dict] = {}
    stats = {
        "roots_seen": 0,
        "json_candidates": 0,
        "duplicates_replaced": 0,
        "duplicates_skipped": 0,
        "per_root_candidates": {},
    }

    for source_name, tier_name, root in roots:
        if not root.exists():
            continue
        stats["roots_seen"] += 1
        per_root_key = f"{source_name}:{tier_name}"
        root_candidates = 0

        for json_path in sorted(root.rglob("*.json")):
            if json_path.name.endswith(".detections.json") or json_path.name == "summary.json":
                continue
            mp4_path = json_path.with_suffix(".mp4")
            if not mp4_path.exists():
                continue

            root_candidates += 1
            stats["json_candidates"] += 1
            name = json_path.stem
            meta = helper._load_meta({"meta_path": str(json_path), "meta": None})
            if helper.skip_bad_samples and meta.get("bad_sample", False):
                continue

            detections_path = mp4_path.with_suffix(".detections.json")
            entry = {
                "key": str(mp4_path),
                "type": "lazy",
                "name": name,
                "root": str(root),
                "video_path": str(mp4_path),
                "meta_path": str(json_path),
                "detections_path": str(detections_path) if detections_path.exists() else None,
                "meta": meta,
                "_sort_mtime": float(json_path.stat().st_mtime),
                "_source_dataset": source_name,
                "_tier": tier_name,
            }
            height_stats = helper._lazy_entry_materialize_height_stats(entry, meta)
            if height_stats is not None and not height_stats["passes"]:
                continue

            cache_dir = helper._lazy_cache_dir(str(root), str(mp4_path), name)
            entry["cache_dir"] = cache_dir
            entry["frames_path"] = os.path.join(cache_dir, helper._frames_cache_name())
            entry["mel_path"] = os.path.join(cache_dir, helper._mel_cache_name)

            existing = entries_by_name.get(name)
            if existing is None or (entry["_sort_mtime"], name) >= (existing["_sort_mtime"], existing["name"]):
                if existing is not None:
                    stats["duplicates_replaced"] += 1
                entries_by_name[name] = entry
            else:
                stats["duplicates_skipped"] += 1

        stats["per_root_candidates"][per_root_key] = root_candidates

    ordered_entries = sorted(entries_by_name.values(), key=lambda item: (item["_sort_mtime"], item["name"]))
    return ordered_entries, stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--prepared-dir", required=True)
    parser.add_argument("--hdtf-tiers", default="confident")
    parser.add_argument("--talkvid-tiers", default="confident,medium")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--target-eligible-total", type=int, default=0)
    args = parser.parse_args()

    cfg = load_config(args.config)
    hdtf_tiers = parse_tiers(args.hdtf_tiers)
    talkvid_tiers = parse_tiers(args.talkvid_tiers)
    roots = selected_roots(TRAINING_ROOT, hdtf_tiers, talkvid_tiers)
    existing_root_paths = [str(root) for _, _, root in roots if root.exists()]
    if not existing_root_paths:
        raise SystemExit("No selected lazy roots exist for SyncNet dataset preparation")

    helper = LipSyncDataset(
        roots=existing_root_paths,
        img_size=cfg["model"]["img_size"],
        mel_step_size=cfg["model"]["mel_steps"],
        fps=cfg["data"]["fps"],
        audio_cfg=cfg["audio"],
        syncnet_T=cfg["syncnet"]["T"],
        mode="syncnet",
        syncnet_style=cfg["syncnet"].get("model_type", "local"),
        cache_size=1,
        skip_bad_samples=cfg["data"].get("skip_bad_samples", True),
        lazy_cache_root=cfg["data"].get("lazy_cache_root"),
        ffmpeg_bin=cfg["data"].get("ffmpeg_bin", "ffmpeg"),
        materialize_timeout=cfg["data"].get("materialize_timeout", 600),
        materialize_frames_size=cfg["data"].get("materialize_frames_size", cfg["model"]["img_size"]),
        **build_sync_alignment_kwargs(cfg),
    )

    entries, scan_stats = collect_lazy_entries(helper, roots)
    min_frames = (3 * int(cfg["syncnet"]["T"])) + 1 if cfg["syncnet"].get("model_type") == "mirror" else 1
    log(f"Selected lazy roots: {[str(root) for _, _, root in roots if root.exists()]}")
    log(f"Collected {len(entries)} unique lazy entries from {scan_stats['json_candidates']} json candidates")

    eligible_manifest: list[dict] = []
    failed_manifest: list[dict] = []
    short_manifest: list[dict] = []
    source_counts: dict[str, int] = {}
    effective_frames_total = 0
    started = time.time()
    target_eligible_total = max(0, int(args.target_eligible_total or 0))

    for idx, entry in enumerate(entries, start=1):
        meta = helper._ensure_entry_sync_alignment(entry)
        sync_alignment = meta.get("sync_alignment") if isinstance(meta, dict) else None
        record = load_sync_alignment(meta)
        frame_count = int(helper._entry_frame_count(entry))

        base_item = {
            "name": entry["name"],
            "source_dataset": entry["_source_dataset"],
            "tier": entry["_tier"],
            "meta_path": entry["meta_path"],
            "video_path": entry["video_path"],
            "sort_mtime": entry["_sort_mtime"],
            "frame_count": frame_count,
        }

        if record is None:
            failed_manifest.append(
                {
                    **base_item,
                    "status": "failed",
                    "reason": None if not isinstance(sync_alignment, dict) else sync_alignment.get("reason"),
                    "error": None if not isinstance(sync_alignment, dict) else sync_alignment.get("error"),
                }
            )
        elif frame_count < min_frames:
            short_manifest.append({**base_item, "status": "too_short"})
        else:
            eligible_manifest.append({**base_item, "status": "eligible"})
            effective_frames_total += frame_count
            key = f"{entry['_source_dataset']}:{entry['_tier']}"
            source_counts[key] = source_counts.get(key, 0) + 1

        if idx == len(entries) or idx % max(1, args.log_every) == 0:
            elapsed = time.time() - started
            rate = idx / elapsed if elapsed > 0 else 0.0
            log(
                f"Prepare {idx}/{len(entries)} eligible={len(eligible_manifest)} "
                f"failed={len(failed_manifest)} short={len(short_manifest)} rate={rate:.2f} clips/s"
            )

        if target_eligible_total > 0 and len(eligible_manifest) >= target_eligible_total:
            log(
                f"Reached target eligible_total={target_eligible_total}; "
                f"stopping early at {idx}/{len(entries)} scanned clips"
            )
            break

    prepared_dir = Path(args.prepared_dir)
    prepared_dir.mkdir(parents=True, exist_ok=True)
    (prepared_dir / "eligible_snapshot.txt").write_text(
        "".join(f"{item['name']}\n" for item in eligible_manifest),
        encoding="utf-8",
    )
    (prepared_dir / "eligible_manifest.json").write_text(
        json.dumps(eligible_manifest, indent=2),
        encoding="utf-8",
    )
    (prepared_dir / "failed_manifest.json").write_text(
        json.dumps(failed_manifest, indent=2),
        encoding="utf-8",
    )
    (prepared_dir / "short_manifest.json").write_text(
        json.dumps(short_manifest, indent=2),
        encoding="utf-8",
    )
    (prepared_dir / "summary.json").write_text(
        json.dumps(
            {
                "total_candidates": len(entries),
                "eligible_total": len(eligible_manifest),
                "failed_sync_alignment": len(failed_manifest),
                "too_short_after_alignment": len(short_manifest),
                "effective_frames_total": effective_frames_total,
                "target_eligible_total": target_eligible_total,
                "min_frames_required": min_frames,
                "hdtf_tiers": hdtf_tiers,
                "talkvid_tiers": talkvid_tiers,
                "source_counts": source_counts,
                "scan_stats": scan_stats,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    log(
        f"Prepared dataset eligible_total={len(eligible_manifest)} failed={len(failed_manifest)} "
        f"short={len(short_manifest)} effective_frames={effective_frames_total}"
    )


if __name__ == "__main__":
    main()
