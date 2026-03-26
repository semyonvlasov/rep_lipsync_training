#!/usr/bin/env python3
"""
Publish a generator checkpoint to Drive and run the report benchmark on-server.

Workflow:
1. Take a training checkpoint or an infer-only checkpoint.
2. Save/refresh an infer-only checkpoint next to it when needed.
3. Upload the checkpoint artifact(s) to the shared Drive.
4. Run the official Wav2Lip-style benchmark remotely against one or more face
   samples using the infer-only checkpoint.
5. Write a small manifest that records uploaded paths, links, and benchmark
   outputs so the tiny MP4 results can be copied back locally.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import torch


TRAINING_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = TRAINING_ROOT.parent
DEFAULT_DRIVE_ROOT_ID = "1y1P-LI3YTPV65zpHXMSrNYsykN2-s5Pv"


def timestamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S %Z")


def log(message: str) -> None:
    print(f"{timestamp()} {message}", flush=True)


def run_logged(cmd: list[str], prefix: str) -> None:
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.rstrip()
        if line:
            log(f"{prefix} {line}")
    rc = proc.wait()
    if rc != 0:
        raise subprocess.CalledProcessError(rc, cmd)


def run_capture(cmd: list[str]) -> str:
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return proc.stdout.strip()


def rclone_copy_file(local_path: Path, remote: str, drive_root_folder_id: str, remote_path: str) -> None:
    cmd = [
        "rclone",
        "copyto",
        "--drive-root-folder-id",
        drive_root_folder_id,
        str(local_path),
        f"{remote}{remote_path}",
    ]
    run_logged(cmd, prefix="[CheckpointPublish:rclone-copy]")


def rclone_link(remote: str, drive_root_folder_id: str, remote_path: str) -> str | None:
    cmd = [
        "rclone",
        "link",
        "--drive-root-folder-id",
        drive_root_folder_id,
        f"{remote}{remote_path}",
    ]
    try:
        return run_capture(cmd)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def resolve_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (TRAINING_ROOT / path).resolve()


def default_run_name(checkpoint_path: Path) -> str:
    parent = checkpoint_path.parent
    if parent.name == "generator":
        return parent.parent.name
    return checkpoint_path.stem


def ensure_infer_only_checkpoint(checkpoint_path: Path, infer_only_out: Path | None) -> tuple[Path, list[Path]]:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict) or "generator" not in ckpt or "config" not in ckpt:
        raise RuntimeError(
            f"Checkpoint does not look like a local generator checkpoint: {checkpoint_path}"
        )

    keep_keys = {"generator", "config", "epoch"}
    if set(ckpt.keys()).issubset(keep_keys):
        log(f"Infer-only checkpoint already provided: {checkpoint_path}")
        return checkpoint_path, [checkpoint_path]

    if infer_only_out is None:
        infer_only_out = checkpoint_path.with_name(f"{checkpoint_path.stem}_infer_only.pth")
    infer_only_out.parent.mkdir(parents=True, exist_ok=True)

    slim = {key: ckpt[key] for key in ("generator", "config", "epoch") if key in ckpt}
    torch.save(slim, infer_only_out)
    log(f"Saved infer-only checkpoint: {infer_only_out}")
    return infer_only_out, [checkpoint_path, infer_only_out]


def benchmark_output_name(face_path: Path, audio_path: Path, checkpoint_path: Path) -> str:
    return f"{face_path.stem}_{audio_path.stem}_{checkpoint_path.stem}.mp4"


def build_remote_path(run_name: str, local_path: Path) -> str:
    return f"checkpoints/{run_name}/{local_path.name}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload checkpoint artifacts and run remote benchmark")
    parser.add_argument("--checkpoint", required=True, help="Local generator checkpoint (.pth)")
    parser.add_argument(
        "--infer-only-out",
        default=None,
        help="Optional output path for the slim infer-only checkpoint",
    )
    parser.add_argument("--run-name", default=None, help="Drive subdirectory name")
    parser.add_argument("--remote", default="gdrive:", help="rclone remote name")
    parser.add_argument(
        "--drive-root-folder-id",
        default=DEFAULT_DRIVE_ROOT_ID,
        help="Drive root folder id passed to rclone",
    )
    parser.add_argument("--skip-upload", action="store_true", help="Skip Drive upload")
    parser.add_argument("--face", action="append", default=[], help="Benchmark face video/image; repeatable")
    parser.add_argument("--audio", default=None, help="Benchmark audio file shared by all faces")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for benchmark outputs and manifest (defaults under training/output)",
    )
    parser.add_argument("--device", default="cuda", choices=("auto", "cpu", "mps", "cuda"))
    parser.add_argument(
        "--detector-device",
        default="cuda",
        choices=("auto", "cpu", "mps", "cuda"),
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--face-det-batch-size", type=int, default=4)
    parser.add_argument("--s3fd-path", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    checkpoint_path = resolve_path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)

    infer_only_out = resolve_path(args.infer_only_out) if args.infer_only_out else None
    infer_checkpoint, upload_candidates = ensure_infer_only_checkpoint(checkpoint_path, infer_only_out)

    run_name = args.run_name or default_run_name(checkpoint_path)
    output_dir = resolve_path(args.output_dir) if args.output_dir else (
        TRAINING_ROOT / "output" / "checkpoint_benchmarks" / run_name
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "created_at": timestamp(),
        "run_name": run_name,
        "checkpoint": str(checkpoint_path),
        "infer_only_checkpoint": str(infer_checkpoint),
        "uploads": [],
        "benchmarks": [],
    }

    if not args.skip_upload:
        for path in upload_candidates:
            remote_path = build_remote_path(run_name, path)
            log(f"Uploading {path.name} -> {args.remote}{remote_path}")
            rclone_copy_file(path, args.remote, args.drive_root_folder_id, remote_path)
            manifest["uploads"].append(
                {
                    "local_path": str(path),
                    "remote_path": remote_path,
                    "link": rclone_link(args.remote, args.drive_root_folder_id, remote_path),
                }
            )

    if args.face:
        if not args.audio:
            raise RuntimeError("--audio is required when benchmarking faces")

        audio_path = resolve_path(args.audio)
        if not audio_path.exists():
            raise FileNotFoundError(audio_path)

        benchmark_script = TRAINING_ROOT / "scripts" / "run_official_wav2lip_benchmark.py"
        for face_raw in args.face:
            face_path = resolve_path(face_raw)
            if not face_path.exists():
                raise FileNotFoundError(face_path)
            out_path = output_dir / benchmark_output_name(face_path, audio_path, infer_checkpoint)
            cmd = [
                sys.executable,
                str(benchmark_script),
                "--face",
                str(face_path),
                "--audio",
                str(audio_path),
                "--checkpoint",
                str(infer_checkpoint),
                "--outfile",
                str(out_path),
                "--device",
                args.device,
                "--detector_device",
                args.detector_device,
                "--batch_size",
                str(args.batch_size),
                "--face_det_batch_size",
                str(args.face_det_batch_size),
            ]
            if args.s3fd_path:
                cmd.extend(["--s3fd_path", str(resolve_path(args.s3fd_path))])

            log(f"Benchmarking {face_path.name} -> {out_path.name}")
            run_logged(cmd, prefix="[CheckpointPublish:bench]")
            manifest["benchmarks"].append(
                {
                    "face": str(face_path),
                    "audio": str(audio_path),
                    "output": str(out_path),
                    "size_bytes": out_path.stat().st_size,
                }
            )

    manifest_path = output_dir / "checkpoint_publish_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    log(f"Wrote manifest: {manifest_path}")
    log(f"Benchmark outputs dir: {output_dir}")
    log("Recommended local copy command:")
    log(
        f"  scp -P <PORT> root@<HOST>:{output_dir}/*.mp4 "
        f"{REPO_ROOT / 'training' / 'output' / output_dir.relative_to(TRAINING_ROOT / 'output')}"
    )


if __name__ == "__main__":
    main()
