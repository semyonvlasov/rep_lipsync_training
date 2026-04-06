#!/usr/bin/env python3
"""Shared helpers for process-stage logging, state manifests, and export commands."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tarfile
import time
from pathlib import Path


def timestamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S %Z")


def log(message: str) -> None:
    print(f"{timestamp()} {message}", flush=True)


def append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def load_json(path: Path):
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def load_summary(path: Path) -> dict:
    payload = load_json(path)
    return payload if isinstance(payload, dict) else {}


def load_latest_state(path: Path, key_field: str) -> dict[str, dict]:
    latest: dict[str, dict] = {}
    if not path.exists():
        return latest
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = obj.get(key_field)
            if key:
                latest[str(key)] = obj
    return latest


def rclone_lsf(remote: str, folder_id: str) -> list[str]:
    cmd = [
        "rclone",
        "lsf",
        "--files-only",
        "--drive-root-folder-id",
        folder_id,
        remote,
    ]
    proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return sorted([line.strip() for line in proc.stdout.splitlines() if line.strip()])


def run_logged(cmd: list[str], prefix: str) -> None:
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        line = line.rstrip()
        if line:
            log(f"{prefix} {line}")
    rc = process.wait()
    if rc != 0:
        raise subprocess.CalledProcessError(rc, cmd)


def cleanup_paths(paths: list[Path]) -> None:
    for path in paths:
        if not path.exists():
            continue
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        else:
            try:
                path.unlink()
            except OSError:
                pass


def pack_dir_to_tar(input_dir: Path, tar_path: Path) -> None:
    tar_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "w") as tar:
        tar.add(input_dir, arcname=input_dir.name)


def count_exported_samples(export_dir: Path) -> int:
    total = 0
    for tier in ("confident", "medium", "unconfident"):
        tier_dir = export_dir / tier
        if tier_dir.exists():
            total += len(list(tier_dir.glob("*.mp4")))
    return total


def load_state_manifest(path: Path) -> dict | None:
    payload = load_json(path)
    return payload if isinstance(payload, dict) else None


def write_state_manifest(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def remove_state_manifest(path: Path) -> None:
    try:
        path.unlink()
    except OSError:
        pass


def update_state_manifest(path: Path, state: dict, stage: str, **extra) -> dict:
    state = dict(state)
    state["stage"] = stage
    state["updated_ts"] = timestamp()
    state.update(extra)
    write_state_manifest(path, state)
    return state


def append_state_event(manifest_path: Path, state: dict, stage: str, **extra) -> None:
    payload = {"ts": timestamp(), "stage": stage}
    for key in (
        "batch_name",
        "source_archive",
        "claimed_archive",
        "processed_archive",
        "dataset_kind",
    ):
        if key in state and state.get(key) is not None:
            payload[key] = state[key]
    payload.update(extra)
    append_jsonl(manifest_path, payload)


def append_failure_event(
    manifest_path: Path,
    state: dict | None,
    stage: str,
    exc: Exception,
    **extra,
) -> None:
    payload = {
        "ts": timestamp(),
        "stage": stage,
        "error_type": type(exc).__name__,
        "error": str(exc),
    }
    if state:
        for key in (
            "batch_name",
            "source_archive",
            "claimed_archive",
            "processed_archive",
            "dataset_kind",
        ):
            if key in state and state.get(key) is not None:
                payload[key] = state[key]
    payload.update(extra)
    append_jsonl(manifest_path, payload)


def build_faceclip_export_cmd(
    *,
    python_bin: str,
    export_script: Path,
    input_dir: Path,
    output_dir: Path,
    normalized_dir: Path,
    source_archive: str,
    dataset_kind: str,
    size: int,
    fps: int,
    max_frames: int,
    detect_every: int,
    smooth_window: int,
    detector_backend: str,
    detector_device: str,
    detector_batch_size: int,
    resize_device: str,
    ffmpeg_bin: str,
    ffmpeg_threads: int,
    ffmpeg_timeout: int,
    video_encoder: str,
    normalized_video_bitrate: str,
    video_bitrate: str,
    smoothing_style: str | None = None,
    framing_style: str | None = None,
    min_detector_score: float | None = None,
    input_is_normalized: bool = False,
) -> list[str]:
    cmd = [
        python_bin,
        str(export_script),
        "--input-dir",
        str(input_dir),
        "--output-dir",
        str(output_dir),
        "--normalized-dir",
        str(normalized_dir),
        "--source-archive",
        source_archive,
        "--dataset-kind",
        dataset_kind,
        "--size",
        str(size),
        "--fps",
        str(fps),
        "--max-frames",
        str(max_frames),
        "--detect-every",
        str(detect_every),
        "--smooth-window",
        str(smooth_window),
        "--detector-backend",
        detector_backend,
        "--detector-device",
        detector_device,
        "--detector-batch-size",
        str(detector_batch_size),
        "--resize-device",
        resize_device,
        "--ffmpeg-bin",
        ffmpeg_bin or "",
        "--ffmpeg-threads",
        str(ffmpeg_threads),
        "--ffmpeg-timeout",
        str(ffmpeg_timeout),
        "--video-encoder",
        video_encoder,
        "--normalized-video-bitrate",
        normalized_video_bitrate,
        "--video-bitrate",
        video_bitrate,
    ]
    if smoothing_style is not None:
        cmd.extend(["--smoothing-style", smoothing_style])
    if framing_style is not None:
        cmd.extend(["--framing-style", framing_style])
    if min_detector_score is not None:
        cmd.extend(["--min-detector-score", str(min_detector_score)])
    if input_is_normalized:
        cmd.append("--input-is-normalized")
    return cmd
