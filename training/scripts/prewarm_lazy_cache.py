#!/usr/bin/env python3
"""
Pre-materialize lazy faceclip samples into the configured cache root.

This is useful after a Drive merge so the next SyncNet or generator run can
start from a warm `frames_s*.npy + mel_*.npy` cache instead of spending the
first batches on ffmpeg decode and mel extraction.
"""

import argparse
import json
import os
import sys
import time

import yaml

TRAINING_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, TRAINING_ROOT)
from data import LipSyncDataset
from scripts.dataset_roots import get_dataset_roots


def load_json(path):
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


def load_allowlist(path):
    if not path:
        return None
    with open(path) as f:
        return {line.strip() for line in f if line.strip()}


def collect_entries(helper, roots, allowlist=None, skip_bad_samples=True):
    entries = []
    selected_names = set()
    stats = {
        "processed_selected": 0,
        "lazy_selected": 0,
        "skipped_allowlist": 0,
        "skipped_bad": 0,
        "skipped_duplicate": 0,
    }

    for root in roots:
        if not root or not os.path.isdir(root):
            continue

        for name in sorted(os.listdir(root)):
            speaker_dir = os.path.join(root, name)
            if not os.path.isdir(speaker_dir):
                continue
            frames_path = os.path.join(speaker_dir, "frames.npy")
            mel_path = os.path.join(speaker_dir, "mel.npy")
            if not (os.path.exists(frames_path) and os.path.exists(mel_path)):
                continue

            if allowlist is not None and name not in allowlist:
                stats["skipped_allowlist"] += 1
                continue

            meta = load_json(os.path.join(speaker_dir, "bbox.json"))
            if skip_bad_samples and meta.get("bad_sample", False):
                stats["skipped_bad"] += 1
                continue

            selected_names.add(name)
            stats["processed_selected"] += 1

        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in {"__pycache__", "_lazy_cache"}]
            for filename in sorted(filenames):
                if not filename.endswith(".mp4"):
                    continue
                mp4_path = os.path.join(dirpath, filename)
                json_path = os.path.splitext(mp4_path)[0] + ".json"
                if not os.path.exists(json_path):
                    continue

                name = os.path.splitext(filename)[0]
                if allowlist is not None and name not in allowlist:
                    stats["skipped_allowlist"] += 1
                    continue
                if name in selected_names:
                    stats["skipped_duplicate"] += 1
                    continue

                meta = load_json(json_path)
                if skip_bad_samples and meta.get("bad_sample", False):
                    stats["skipped_bad"] += 1
                    continue

                cache_dir = helper._lazy_cache_dir(root, mp4_path, name)
                entry = {
                    "key": mp4_path,
                    "type": "lazy",
                    "name": name,
                    "root": root,
                    "video_path": mp4_path,
                    "meta_path": json_path,
                    "meta": meta,
                    "cache_dir": cache_dir,
                    "frames_path": os.path.join(cache_dir, helper._frames_cache_name()),
                    "mel_path": os.path.join(cache_dir, helper._mel_cache_name),
                }
                entries.append(entry)
                selected_names.add(name)
                stats["lazy_selected"] += 1

    return entries, stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--speaker-list", default=None, help="Optional newline-separated allowlist")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--max-items", type=int, default=0, help="Limit number of lazy entries for testing")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    def log(msg):
        import datetime
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        print(f"[{ts}] {msg}", flush=True)

    allowlist = load_allowlist(args.speaker_list)
    roots = get_dataset_roots(cfg)
    helper = LipSyncDataset(
        roots=[],
        img_size=cfg["model"]["img_size"],
        mel_step_size=cfg["model"]["mel_steps"],
        fps=cfg["data"]["fps"],
        audio_cfg=cfg["audio"],
        syncnet_T=cfg["syncnet"]["T"],
        mode="syncnet",
        cache_size=1,
        skip_bad_samples=cfg["data"].get("skip_bad_samples", True),
        lazy_cache_root=cfg["data"].get("lazy_cache_root"),
        ffmpeg_bin=cfg["data"].get("ffmpeg_bin", "ffmpeg"),
        materialize_timeout=cfg["data"].get("materialize_timeout", 600),
        materialize_frames_size=cfg["data"].get("materialize_frames_size", cfg["model"]["img_size"]),
    )

    entries, stats = collect_entries(
        helper,
        roots=roots,
        allowlist=allowlist,
        skip_bad_samples=cfg["data"].get("skip_bad_samples", True),
    )
    if args.max_items > 0:
        entries = entries[: args.max_items]

    log(f"Prewarm roots: {roots}")
    log(
        "Selected entries: "
        f"processed={stats['processed_selected']} lazy={stats['lazy_selected']} "
        f"skipped_allowlist={stats['skipped_allowlist']} skipped_bad={stats['skipped_bad']} "
        f"skipped_duplicate={stats['skipped_duplicate']}"
    )
    log(
        "Cache target: "
        f"{cfg['data'].get('lazy_cache_root', '<root>/_lazy_cache')} "
        f"frames={helper._frames_cache_name()} mel={helper._mel_cache_name}"
    )
    log(f"Lazy entries to prewarm: {len(entries)}")

    started = time.time()
    hits = 0
    misses = 0
    failures = 0

    for idx, entry in enumerate(entries, start=1):
        had_frames = os.path.exists(entry["frames_path"])
        had_mel = os.path.exists(entry["mel_path"])
        if had_frames and had_mel:
            hits += 1
        else:
            misses += 1
        try:
            helper._materialize_lazy_entry(entry)
        except Exception as exc:
            failures += 1
            log(f"ERROR {idx}/{len(entries)} {entry['name']}: {exc}")

        if idx == len(entries) or idx % max(1, args.log_every) == 0:
            elapsed = time.time() - started
            rate = idx / elapsed if elapsed > 0 else 0.0
            log(
                f"Progress {idx}/{len(entries)} "
                f"hits={hits} misses={misses} failures={failures} "
                f"rate={rate:.2f} clips/s elapsed={elapsed:.1f}s"
            )

    log(
        f"Done: lazy={len(entries)} hits={hits} misses={misses} "
        f"failures={failures} elapsed={time.time() - started:.1f}s"
    )


if __name__ == "__main__":
    main()
