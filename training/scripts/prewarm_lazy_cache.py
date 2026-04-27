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
import shutil
import subprocess
import sys
import time
import tempfile
from pathlib import Path

import yaml

TRAINING_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, TRAINING_ROOT)
from data import LipSyncDataset
from data.sync_alignment import (
    is_failed_sync_alignment,
    load_sync_alignment_registry,
    write_sync_alignment_registry,
)
from config_loader import load_config
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


def resolve_training_path(value):
    if not value:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return Path(TRAINING_ROOT) / path


def run_rclone(cmd, *, log_prefix):
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()
        raise RuntimeError(f"{log_prefix} failed rc={proc.returncode}: {detail}")
    return proc


def rclone_file_exists(remote, folder_id, remote_name):
    proc = subprocess.run(
        [
            "rclone",
            "lsf",
            "--files-only",
            "--drive-root-folder-id",
            folder_id,
            remote,
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()
        raise RuntimeError(f"rclone lsf failed rc={proc.returncode}: {detail}")
    return remote_name in {line.strip() for line in proc.stdout.splitlines() if line.strip()}


def download_sync_alignment_registry(args, log):
    if args.no_sync_alignment_download or not args.sync_alignment_folder_id:
        return
    local_path = resolve_training_path(args.sync_alignment_registry_path)
    if local_path is None:
        return
    local_path.parent.mkdir(parents=True, exist_ok=True)
    if not rclone_file_exists(
        args.sync_alignment_remote,
        args.sync_alignment_folder_id,
        args.sync_alignment_remote_name,
    ):
        log(
            "Sync registry: remote file is absent; will create "
            f"{args.sync_alignment_remote_name} on upload"
        )
        return
    run_rclone(
        [
            "rclone",
            "copyto",
            "--drive-root-folder-id",
            args.sync_alignment_folder_id,
            f"{args.sync_alignment_remote}{args.sync_alignment_remote_name}",
            str(local_path),
        ],
        log_prefix="sync registry download",
    )
    log(f"Sync registry: downloaded {args.sync_alignment_remote_name} -> {local_path}")


def upload_sync_alignment_registry(args, log):
    if args.no_sync_alignment_upload or not args.sync_alignment_folder_id:
        return
    local_path = resolve_training_path(args.sync_alignment_registry_path)
    if local_path is None or not local_path.exists():
        log("Sync registry: no local file to upload")
        return

    with tempfile.TemporaryDirectory(prefix="sync_registry_merge_") as tmp_dir:
        remote_path = Path(tmp_dir) / "remote.jsonl"
        merged_path = Path(tmp_dir) / "merged.jsonl"
        registries = []
        if rclone_file_exists(
            args.sync_alignment_remote,
            args.sync_alignment_folder_id,
            args.sync_alignment_remote_name,
        ):
            run_rclone(
                [
                    "rclone",
                    "copyto",
                    "--drive-root-folder-id",
                    args.sync_alignment_folder_id,
                    f"{args.sync_alignment_remote}{args.sync_alignment_remote_name}",
                    str(remote_path),
                ],
                log_prefix="sync registry merge download",
            )
            registries.append(load_sync_alignment_registry(remote_path))
        registries.append(load_sync_alignment_registry(local_path))

        merged = {}
        for registry in registries:
            merged.update(registry)
        write_sync_alignment_registry(merged_path, merged)
        shutil.copy2(merged_path, local_path)
        run_rclone(
            [
                "rclone",
                "copyto",
                "--drive-root-folder-id",
                args.sync_alignment_folder_id,
                str(merged_path),
                f"{args.sync_alignment_remote}{args.sync_alignment_remote_name}",
            ],
            log_prefix="sync registry upload",
        )
        log(
            f"Sync registry: uploaded {len(merged)} records -> "
            f"{args.sync_alignment_remote_name}"
        )


def build_sync_alignment_kwargs(cfg, registry_path):
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
        "sync_alignment_registry_path": str(registry_path) if registry_path else sync_cfg.get("registry_path"),
    }


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
            meta = helper._apply_sync_alignment_registry_to_meta(
                meta,
                meta_path=os.path.join(speaker_dir, "bbox.json"),
                name=name,
                root=root,
            )
            if skip_bad_samples and meta.get("bad_sample", False):
                stats["skipped_bad"] += 1
                continue
            if skip_bad_samples and is_failed_sync_alignment(meta):
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
                meta = helper._apply_sync_alignment_registry_to_meta(
                    meta,
                    meta_path=json_path,
                    name=name,
                    root=root,
                )
                if skip_bad_samples and meta.get("bad_sample", False):
                    stats["skipped_bad"] += 1
                    continue
                if skip_bad_samples and is_failed_sync_alignment(meta):
                    stats["skipped_bad"] += 1
                    cache_dir = helper._lazy_cache_dir(root, mp4_path, name)
                    helper._cleanup_lazy_materialization({"cache_dir": cache_dir})
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
    parser.add_argument("--sync-alignment-registry-path", default="output/sync_alignment/sync_alignment_manifest.jsonl")
    parser.add_argument("--sync-alignment-remote", default="gdrive:")
    parser.add_argument("--sync-alignment-folder-id", default=None)
    parser.add_argument("--sync-alignment-remote-name", default="sync_alignment_manifest.jsonl")
    parser.add_argument("--no-sync-alignment-download", action="store_true")
    parser.add_argument("--no-sync-alignment-upload", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)

    def log(msg):
        import datetime
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        print(f"[{ts}] {msg}", flush=True)

    download_sync_alignment_registry(args, log)

    allowlist = load_allowlist(args.speaker_list)
    roots = get_dataset_roots(cfg)
    registry_path = resolve_training_path(args.sync_alignment_registry_path)
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
        **build_sync_alignment_kwargs(cfg, registry_path),
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
    sync_bad = 0

    for idx, entry in enumerate(entries, start=1):
        had_frames = os.path.exists(entry["frames_path"])
        had_mel = os.path.exists(entry["mel_path"])
        if had_frames and had_mel:
            hits += 1
        else:
            misses += 1
        try:
            if helper.sync_alignment_enabled:
                meta = helper._ensure_entry_sync_alignment(entry)
                if is_failed_sync_alignment(meta):
                    sync_bad += 1
                    helper._cleanup_lazy_materialization(entry)
                    continue
            helper._materialize_lazy_entry(entry)
        except Exception as exc:
            failures += 1
            log(f"ERROR {idx}/{len(entries)} {entry['name']}: {exc}")

        if idx == len(entries) or idx % max(1, args.log_every) == 0:
            elapsed = time.time() - started
            rate = idx / elapsed if elapsed > 0 else 0.0
            log(
                f"Progress {idx}/{len(entries)} "
                f"hits={hits} misses={misses} sync_bad={sync_bad} failures={failures} "
                f"rate={rate:.2f} clips/s elapsed={elapsed:.1f}s"
            )

    log(
        f"Done: lazy={len(entries)} hits={hits} misses={misses} "
        f"sync_bad={sync_bad} failures={failures} elapsed={time.time() - started:.1f}s"
    )
    upload_sync_alignment_registry(args, log)


if __name__ == "__main__":
    main()
