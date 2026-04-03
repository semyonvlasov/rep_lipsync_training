#!/usr/bin/env python3
"""
Render a compact JSON status snapshot for the raw->lazy faceclip archive cycle.

This script is intended to run on the remote box. A local monitor can call it
over SSH and then aggregate several remotes into one short combined status log.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


STATE_MANIFEST_BASENAME = "active_archive_state.json"


def load_json(path: Path):
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def read_last_jsonl_record(path: Path):
    if not path.exists():
        return None
    last = None
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    last = json.loads(line)
                except Exception:
                    continue
    except Exception:
        return None
    return last


def count_mp4_files(path: Path) -> int:
    if not path.exists() or not path.is_dir():
        return 0
    total = 0
    try:
        for child in path.iterdir():
            if child.is_file() and child.suffix.lower() == ".mp4":
                total += 1
    except Exception:
        return 0
    return total


def count_completed_videos(export_manifest_path: Path) -> int:
    if not export_manifest_path.exists():
        return 0
    completed = set()
    try:
        with open(export_manifest_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except Exception:
                    continue
                source_video = payload.get("source_video")
                if isinstance(source_video, str) and source_video:
                    completed.add(source_video)
    except Exception:
        return 0
    return len(completed)


def pid_alive(pid_path: Path) -> bool:
    if not pid_path.exists():
        return False
    try:
        raw = pid_path.read_text().strip()
        pid = int(raw)
    except Exception:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def build_status_text(status: str, archive_name: str | None, progress_done: int | None, progress_total: int | None, *, last_stage: str | None = None) -> str:
    archive_name = archive_name or "<none>"
    if status == "download":
        return f"download ({archive_name})"
    if status == "process":
        if progress_done is not None and progress_total is not None and progress_total > 0:
            return f"process ({archive_name}) ({progress_done}/{progress_total})"
        return f"process ({archive_name})"
    if status == "upload":
        return f"upload ({archive_name})"
    if status == "failed":
        return f"failed ({archive_name})"
    if status == "idle":
        if last_stage:
            return f"idle, last {last_stage} ({archive_name})"
        return "idle"
    return f"unknown ({archive_name})"


def payload_from_active_state(state: dict, cycle_alive: bool) -> dict:
    stage = str(state.get("stage", "claimed"))
    source_archive = state.get("source_archive")
    processed_archive = state.get("processed_archive")
    updated_ts = state.get("updated_ts")

    if stage in {"claimed", "download_started"}:
        status = "download"
        archive_name = source_archive or processed_archive
        progress_done = None
        progress_total = None
    elif stage in {"downloaded", "extracted", "process_started", "processed"}:
        status = "process"
        archive_name = source_archive
        extract_root = Path(str(state.get("extract_root", "")))
        export_manifest_path = Path(str(state.get("export_root", ""))) / "export_manifest.jsonl"
        progress_total = count_mp4_files(extract_root)
        progress_done = count_completed_videos(export_manifest_path)
    elif stage in {"packaged", "upload_started", "uploaded"}:
        status = "upload"
        archive_name = processed_archive or source_archive
        progress_done = None
        progress_total = None
    else:
        status = "unknown"
        archive_name = source_archive or processed_archive
        progress_done = None
        progress_total = None

    if not cycle_alive and status in {"download", "process", "upload"}:
        status = "failed"

    status_text = build_status_text(
        status,
        archive_name,
        progress_done,
        progress_total,
        last_stage=stage,
    )
    return {
        "status": status,
        "archive_name": archive_name,
        "progress_done": progress_done,
        "progress_total": progress_total,
        "last_stage": stage,
        "updated_ts": updated_ts,
        "cycle_alive": cycle_alive,
        "status_text": status_text,
    }


def payload_from_last_record(record, cycle_alive: bool) -> dict:
    if not isinstance(record, dict):
        status = "idle"
        archive_name = None
        last_stage = None
        updated_ts = None
    else:
        stage = str(record.get("stage", "idle"))
        archive_name = record.get("source_archive") or record.get("processed_archive")
        updated_ts = record.get("ts")
        if stage in {"failed", "resume_failed", "claim_failed"}:
            status = "failed"
            last_stage = stage
        else:
            status = "idle"
            last_stage = stage

    status_text = build_status_text(
        status,
        archive_name,
        None,
        None,
        last_stage=last_stage,
    )
    return {
        "status": status,
        "archive_name": archive_name,
        "progress_done": None,
        "progress_total": None,
        "last_stage": last_stage,
        "updated_ts": updated_ts,
        "cycle_alive": cycle_alive,
        "status_text": status_text,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-path", required=True)
    parser.add_argument("--cycle-log", default="")
    parser.add_argument("--cycle-pid-path", default="")
    args = parser.parse_args()

    manifest_path = Path(args.manifest_path)
    state_path = manifest_path.with_name(STATE_MANIFEST_BASENAME)
    cycle_pid_path = Path(args.cycle_pid_path) if args.cycle_pid_path else Path()
    cycle_alive = pid_alive(cycle_pid_path) if args.cycle_pid_path else False

    state = load_json(state_path)
    if isinstance(state, dict):
        payload = payload_from_active_state(state, cycle_alive)
    else:
        record = read_last_jsonl_record(manifest_path)
        payload = payload_from_last_record(record, cycle_alive)

    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
