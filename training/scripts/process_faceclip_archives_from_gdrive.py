#!/usr/bin/env python3
"""
Sequentially download source archives from Google Drive, export faceclip videos,
pack processed outputs, upload them to a destination folder, and clean up local
staging after successful upload.
"""

import argparse
import fnmatch
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


def load_latest_state(path: Path) -> dict[str, dict]:
    latest = {}
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
            archive_name = obj.get("source_archive")
            if archive_name:
                latest[str(archive_name)] = obj
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


def guess_archive_kind(name: str) -> tuple[str, bool]:
    lower = name.lower()
    if "talkvid" in lower and "raw" in lower:
        return "talkvid", False
    if "hdtf" in lower and "clips" in lower:
        return "hdtf", True
    if "hdtf" in lower and "raw" in lower:
        return "hdtf", False
    if "clips" in lower and "raw" not in lower:
        return "hdtf", True
    return "talkvid", False


def processed_archive_name(source_archive: str) -> str:
    name = source_archive
    if name.endswith(".tar"):
        stem = name[:-4]
    else:
        stem = name
    stem = stem.replace("talkvid_raw_", "talkvid_faceclips_")
    stem = stem.replace("hdtf_clips_", "hdtf_faceclips_")
    stem = stem.replace("hdtf_raw_", "hdtf_faceclips_")
    if stem == name[:-4]:
        stem = stem + "_faceclips"
    return stem + ".tar"


def extract_tar(tar_path: Path, extract_dir: Path) -> None:
    extract_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "r") as tar:
        tar.extractall(extract_dir)


def pack_dir_to_tar(input_dir: Path, tar_path: Path) -> None:
    tar_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "w") as tar:
        tar.add(input_dir, arcname=input_dir.name)


def count_exported_samples(export_dir: Path) -> int:
    total = 0
    for tier in ("confident", "medium", "unconfident"):
        tier_dir = export_dir / tier
        if not tier_dir.exists():
            continue
        total += len(list(tier_dir.glob("*.mp4")))
    return total


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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-folder-id", required=True)
    parser.add_argument("--dest-folder-id", required=True)
    parser.add_argument("--remote", default="gdrive:")
    parser.add_argument("--data-root", default="training/data/faceclip_local")
    parser.add_argument("--manifest-path", default="training/output/faceclip_local_cycle/archive_manifest.jsonl")
    parser.add_argument("--max-archives", type=int, default=0, help="0=all pending")
    parser.add_argument("--archive-glob", default="*.tar")
    parser.add_argument("--python-bin", default="python3")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--max-frames", type=int, default=750)
    parser.add_argument("--detect-every", type=int, default=10)
    parser.add_argument("--smooth-window", type=int, default=9)
    parser.add_argument("--detector-backend", choices=["opencv", "sfd"], default="sfd")
    parser.add_argument("--detector-device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--detector-batch-size", type=int, default=4)
    parser.add_argument("--resize-device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--ffmpeg-bin", default=None)
    parser.add_argument("--ffmpeg-threads", type=int, default=1)
    parser.add_argument("--ffmpeg-timeout", type=int, default=180)
    parser.add_argument("--video-encoder", default="auto")
    parser.add_argument("--video-bitrate", default="2200k")
    args = parser.parse_args()

    root = Path(args.data_root)
    downloads_dir = root / "downloads"
    extracted_dir = root / "extracted"
    normalized_dir = root / "normalized"
    processed_dir = root / "processed_work"
    archives_dir = root / "archives"
    manifest_path = Path(args.manifest_path)
    downloads_dir.mkdir(parents=True, exist_ok=True)
    extracted_dir.mkdir(parents=True, exist_ok=True)
    normalized_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    archives_dir.mkdir(parents=True, exist_ok=True)

    latest_state = load_latest_state(manifest_path)

    source_archives = [
        name
        for name in rclone_lsf(args.remote, args.source_folder_id)
        if name.endswith(".tar") and fnmatch.fnmatch(name, args.archive_glob)
    ]
    try:
        dest_archives = set(rclone_lsf(args.remote, args.dest_folder_id))
    except Exception:
        dest_archives = set()

    log(f"[FaceclipCycle] source_archives={len(source_archives)}")
    log(f"[FaceclipCycle] dest_archives={len(dest_archives)}")
    processed_count = 0

    for archive_name in source_archives:
        state = latest_state.get(archive_name, {})
        output_name = processed_archive_name(archive_name)
        final_stage = str(state.get("stage") or "")
        if output_name in dest_archives or final_stage in {"uploaded_cleaned", "cleaned_no_output"}:
            continue

        dataset_kind, input_is_normalized = guess_archive_kind(archive_name)
        source_tar = downloads_dir / archive_name
        archive_stem = archive_name[:-4] if archive_name.endswith(".tar") else archive_name
        extract_root = extracted_dir / archive_stem
        export_root = processed_dir / archive_stem
        normalize_root = normalized_dir / archive_stem
        processed_tar = archives_dir / output_name
        summary_path = export_root / "summary.json"

        append_jsonl(
            manifest_path,
            {
                "ts": timestamp(),
                "source_archive": archive_name,
                "processed_archive": output_name,
                "dataset_kind": dataset_kind,
                "stage": "start",
            },
        )

        if not source_tar.exists():
            log(f"[FaceclipCycle] downloading {archive_name}")
            append_jsonl(
                manifest_path,
                {
                    "ts": timestamp(),
                    "source_archive": archive_name,
                    "processed_archive": output_name,
                    "dataset_kind": dataset_kind,
                    "stage": "download_started",
                },
            )
            run_logged(
                [
                    "rclone",
                    "copyto",
                    "--drive-root-folder-id",
                    args.source_folder_id,
                    f"{args.remote}{archive_name}",
                    str(source_tar),
                ],
                prefix="[FaceclipCycle:rclone-download]",
            )
            append_jsonl(
                manifest_path,
                {
                    "ts": timestamp(),
                    "source_archive": archive_name,
                    "processed_archive": output_name,
                    "dataset_kind": dataset_kind,
                    "stage": "downloaded",
                    "local_tar": str(source_tar),
                    "bytes": source_tar.stat().st_size,
                },
            )

        if not extract_root.exists():
            log(f"[FaceclipCycle] extracting {archive_name}")
            extract_tar(source_tar, extract_root)
            append_jsonl(
                manifest_path,
                {
                    "ts": timestamp(),
                    "source_archive": archive_name,
                    "processed_archive": output_name,
                    "dataset_kind": dataset_kind,
                    "stage": "extracted",
                    "extract_root": str(extract_root),
                },
            )

        log(f"[FaceclipCycle] exporting {archive_name} -> {output_name}")
        run_logged(
            [
                args.python_bin,
                "training/scripts/export_faceclip_batch.py",
                "--input-dir",
                str(extract_root),
                "--output-dir",
                str(export_root),
                "--normalized-dir",
                str(normalize_root),
                "--source-archive",
                archive_name,
                "--dataset-kind",
                dataset_kind,
                *(["--input-is-normalized"] if input_is_normalized else []),
                "--size",
                str(args.size),
                "--fps",
                str(args.fps),
                "--max-frames",
                str(args.max_frames),
                "--detect-every",
                str(args.detect_every),
                "--smooth-window",
                str(args.smooth_window),
                "--detector-backend",
                args.detector_backend,
                "--detector-device",
                args.detector_device,
                "--detector-batch-size",
                str(args.detector_batch_size),
                "--resize-device",
                args.resize_device,
                "--ffmpeg-bin",
                args.ffmpeg_bin or "",
                "--ffmpeg-threads",
                str(args.ffmpeg_threads),
                "--ffmpeg-timeout",
                str(args.ffmpeg_timeout),
                "--video-encoder",
                args.video_encoder,
                "--video-bitrate",
                args.video_bitrate,
            ],
            prefix="[FaceclipCycle:export]",
        )

        summary = {}
        if summary_path.exists():
            with open(summary_path) as f:
                summary = json.load(f)
        exported_samples = count_exported_samples(export_root)
        append_jsonl(
            manifest_path,
            {
                "ts": timestamp(),
                "source_archive": archive_name,
                "processed_archive": output_name,
                "dataset_kind": dataset_kind,
                "stage": "processed",
                "export_root": str(export_root),
                "normalized_root": str(normalize_root),
                "summary": summary,
                "exported_samples": exported_samples,
            },
        )

        if exported_samples == 0:
            log(f"[FaceclipCycle] no usable samples in {archive_name}; cleaning local staging")
            cleanup_paths([source_tar, extract_root, normalize_root, export_root, processed_tar])
            append_jsonl(
                manifest_path,
                {
                    "ts": timestamp(),
                    "source_archive": archive_name,
                    "processed_archive": output_name,
                    "dataset_kind": dataset_kind,
                    "stage": "cleaned_no_output",
                    "summary": summary,
                },
            )
            processed_count += 1
            if args.max_archives > 0 and processed_count >= args.max_archives:
                break
            continue

        if not processed_tar.exists():
            log(f"[FaceclipCycle] packaging {output_name}")
            pack_dir_to_tar(export_root, processed_tar)
            append_jsonl(
                manifest_path,
                {
                    "ts": timestamp(),
                    "source_archive": archive_name,
                    "processed_archive": output_name,
                    "dataset_kind": dataset_kind,
                    "stage": "packaged",
                    "processed_tar": str(processed_tar),
                    "bytes": processed_tar.stat().st_size,
                },
            )

        log(f"[FaceclipCycle] uploading {output_name}")
        run_logged(
            [
                "rclone",
                "copyto",
                "--drive-root-folder-id",
                args.dest_folder_id,
                str(processed_tar),
                f"{args.remote}{output_name}",
            ],
            prefix="[FaceclipCycle:rclone-upload]",
        )
        append_jsonl(
            manifest_path,
            {
                "ts": timestamp(),
                "source_archive": archive_name,
                "processed_archive": output_name,
                "dataset_kind": dataset_kind,
                "stage": "uploaded",
                "remote_name": output_name,
                "summary": summary,
            },
        )

        cleanup_paths([source_tar, extract_root, normalize_root, export_root, processed_tar])
        append_jsonl(
            manifest_path,
            {
                "ts": timestamp(),
                "source_archive": archive_name,
                "processed_archive": output_name,
                "dataset_kind": dataset_kind,
                "stage": "uploaded_cleaned",
                "remote_name": output_name,
                "summary": summary,
            },
        )
        processed_count += 1
        if args.max_archives > 0 and processed_count >= args.max_archives:
            break

    log(f"[FaceclipCycle] done processed_archives={processed_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
