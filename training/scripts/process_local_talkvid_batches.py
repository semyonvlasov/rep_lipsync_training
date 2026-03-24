#!/usr/bin/env python3
"""
Process completed local TalkVid fetch batches into faceclip archives, upload the
processed batches to Google Drive, and clean local staging after success.
"""

import argparse
import json
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
            batch_name = obj.get("batch_name")
            if batch_name:
                latest[str(batch_name)] = obj
    return latest


def load_json(path: Path):
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


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


def processed_archive_name(batch_name: str) -> str:
    return f"talkvid_faceclips_{batch_name}.tar"


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


def pack_dir_to_tar(input_dir: Path, tar_path: Path) -> None:
    tar_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "w") as tar:
        tar.add(input_dir, arcname=input_dir.name)


def iter_completed_batches(batches_dir: Path):
    for batch_root in sorted(batches_dir.glob("batch_*")):
        if not batch_root.is_dir():
            continue
        if not (batch_root / "fetch_complete.json").exists():
            continue
        yield batch_root


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batches-dir", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--manifest-path", required=True)
    parser.add_argument("--dest-folder-id", required=True)
    parser.add_argument("--remote", default="gdrive:")
    parser.add_argument("--python-bin", default="python3")
    parser.add_argument("--follow", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=20)
    parser.add_argument("--producer-done-flag", default=None)
    parser.add_argument("--max-batches", type=int, default=0, help="0=all available")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--max-frames", type=int, default=750)
    parser.add_argument("--detect-every", type=int, default=10)
    parser.add_argument("--smooth-window", type=int, default=9)
    parser.add_argument("--detector-backend", choices=["opencv", "sfd"], default="opencv")
    parser.add_argument("--detector-device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--detector-batch-size", type=int, default=4)
    parser.add_argument("--resize-device", choices=["auto", "cpu", "cuda", "mps"], default="cpu")
    parser.add_argument("--ffmpeg-bin", default=None)
    parser.add_argument("--ffmpeg-threads", type=int, default=1)
    parser.add_argument("--ffmpeg-timeout", type=int, default=180)
    parser.add_argument("--video-encoder", default="auto")
    parser.add_argument("--normalized-video-bitrate", default="")
    parser.add_argument("--video-bitrate", default="2200k")
    args = parser.parse_args()
    args.normalized_video_bitrate = args.normalized_video_bitrate or args.video_bitrate

    batches_dir = Path(args.batches_dir)
    data_root = Path(args.data_root)
    manifest_path = Path(args.manifest_path)
    producer_done_flag = Path(args.producer_done_flag) if args.producer_done_flag else None

    normalized_root = data_root / "normalized"
    export_root = data_root / "processed_work"
    archives_root = data_root / "archives"
    normalized_root.mkdir(parents=True, exist_ok=True)
    export_root.mkdir(parents=True, exist_ok=True)
    archives_root.mkdir(parents=True, exist_ok=True)

    try:
        dest_archives = set(rclone_lsf(args.remote, args.dest_folder_id))
    except Exception:
        dest_archives = set()

    log(f"[TalkVidFaceclip] batches_dir={batches_dir}")
    log(f"[TalkVidFaceclip] data_root={data_root}")
    log(f"[TalkVidFaceclip] dest_archives={len(dest_archives)}")

    processed_batches = 0
    while True:
        latest_state = load_latest_state(manifest_path)
        pending = []
        for batch_root in iter_completed_batches(batches_dir):
            batch_name = batch_root.name
            output_name = processed_archive_name(batch_name)
            state = latest_state.get(batch_name, {})
            final_stage = str(state.get("stage") or "")
            if output_name in dest_archives or final_stage in {"uploaded_cleaned", "cleaned_no_output"}:
                continue
            pending.append((batch_root, batch_name, output_name))

        if not pending:
            if args.follow and not (producer_done_flag and producer_done_flag.exists()):
                time.sleep(max(1, args.poll_seconds))
                continue
            break

        batch_root, batch_name, output_name = pending[0]
        raw_dir = batch_root / "raw"
        fetch_meta = load_json(batch_root / "fetch_complete.json") or {}
        source_archive = f"talkvid_raw_{batch_name}.tar"
        work_name = output_name[:-4]
        batch_export_root = export_root / work_name
        batch_normalized_root = normalized_root / work_name
        processed_tar = archives_root / output_name
        summary_path = batch_export_root / "summary.json"

        append_jsonl(
            manifest_path,
            {
                "ts": timestamp(),
                "batch_name": batch_name,
                "source_archive": source_archive,
                "processed_archive": output_name,
                "stage": "start",
                "fetch_meta": fetch_meta,
            },
        )

        if output_name in dest_archives:
            cleanup_paths([batch_root, batch_export_root, batch_normalized_root, processed_tar])
            append_jsonl(
                manifest_path,
                {
                    "ts": timestamp(),
                    "batch_name": batch_name,
                    "source_archive": source_archive,
                    "processed_archive": output_name,
                    "stage": "uploaded_cleaned",
                    "remote_name": output_name,
                    "note": "remote_already_present",
                },
            )
            processed_batches += 1
            if args.max_batches > 0 and processed_batches >= args.max_batches:
                break
            continue

        log(f"[TalkVidFaceclip] exporting {batch_name} -> {output_name}")
        run_logged(
            [
                args.python_bin,
                "scripts/export_faceclip_batch.py",
                "--input-dir",
                str(raw_dir),
                "--output-dir",
                str(batch_export_root),
                "--normalized-dir",
                str(batch_normalized_root),
                "--source-archive",
                source_archive,
                "--dataset-kind",
                "talkvid",
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
                "--normalized-video-bitrate",
                args.normalized_video_bitrate,
                "--video-bitrate",
                args.video_bitrate,
            ],
            prefix="[TalkVidFaceclip:export]",
        )

        summary = {}
        if summary_path.exists():
            with open(summary_path) as f:
                summary = json.load(f)
        exported_samples = count_exported_samples(batch_export_root)
        append_jsonl(
            manifest_path,
            {
                "ts": timestamp(),
                "batch_name": batch_name,
                "source_archive": source_archive,
                "processed_archive": output_name,
                "stage": "processed",
                "batch_root": str(batch_root),
                "export_root": str(batch_export_root),
                "normalized_root": str(batch_normalized_root),
                "summary": summary,
                "exported_samples": exported_samples,
            },
        )

        if exported_samples == 0:
            log(f"[TalkVidFaceclip] no usable samples in {batch_name}; cleaning local staging")
            cleanup_paths([batch_root, batch_export_root, batch_normalized_root, processed_tar])
            append_jsonl(
                manifest_path,
                {
                    "ts": timestamp(),
                    "batch_name": batch_name,
                    "source_archive": source_archive,
                    "processed_archive": output_name,
                    "stage": "cleaned_no_output",
                    "summary": summary,
                },
            )
            processed_batches += 1
            if args.max_batches > 0 and processed_batches >= args.max_batches:
                break
            continue

        if not processed_tar.exists():
            log(f"[TalkVidFaceclip] packaging {output_name}")
            pack_dir_to_tar(batch_export_root, processed_tar)
            append_jsonl(
                manifest_path,
                {
                    "ts": timestamp(),
                    "batch_name": batch_name,
                    "source_archive": source_archive,
                    "processed_archive": output_name,
                    "stage": "packaged",
                    "processed_tar": str(processed_tar),
                    "bytes": processed_tar.stat().st_size,
                    "summary": summary,
                },
            )

        log(f"[TalkVidFaceclip] uploading {output_name}")
        run_logged(
            [
                "rclone",
                "copyto",
                "--drive-root-folder-id",
                args.dest_folder_id,
                str(processed_tar),
                f"{args.remote}{output_name}",
            ],
            prefix="[TalkVidFaceclip:rclone-upload]",
        )
        dest_archives.add(output_name)
        append_jsonl(
            manifest_path,
            {
                "ts": timestamp(),
                "batch_name": batch_name,
                "source_archive": source_archive,
                "processed_archive": output_name,
                "stage": "uploaded",
                "remote_name": output_name,
                "summary": summary,
            },
        )

        cleanup_paths([batch_root, batch_export_root, batch_normalized_root, processed_tar])
        append_jsonl(
            manifest_path,
            {
                "ts": timestamp(),
                "batch_name": batch_name,
                "source_archive": source_archive,
                "processed_archive": output_name,
                "stage": "uploaded_cleaned",
                "remote_name": output_name,
                "summary": summary,
            },
        )
        processed_batches += 1
        if args.max_batches > 0 and processed_batches >= args.max_batches:
            break

    log(f"[TalkVidFaceclip] done processed_batches={processed_batches}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
