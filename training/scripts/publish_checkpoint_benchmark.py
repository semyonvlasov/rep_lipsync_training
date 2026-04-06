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
OFFICIAL_WAV2LIP_ROOT = REPO_ROOT / "models" / "wav2lip"
OFFICIAL_SYNCNET_ROOT = REPO_ROOT / "models" / "official_syncnet"


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


def is_local_generator_checkpoint(ckpt: object) -> bool:
    return isinstance(ckpt, dict) and "generator" in ckpt and "config" in ckpt


def is_plain_state_dict_checkpoint(ckpt: object) -> bool:
    return isinstance(ckpt, dict) and bool(ckpt) and all(isinstance(v, torch.Tensor) for v in ckpt.values())


def load_official_wav2lip_class():
    candidate_roots = [
        OFFICIAL_SYNCNET_ROOT,
        OFFICIAL_WAV2LIP_ROOT,
        REPO_ROOT.parent / "models" / "wav2lip",
    ]
    for root in candidate_roots:
        if not (root / "models").exists():
            continue
        root_str = str(root)
        if root_str not in sys.path:
            sys.path.insert(0, root_str)
        if "models" in sys.modules:
            del sys.modules["models"]
        from models import Wav2Lip

        return Wav2Lip
    raise RuntimeError("Could not locate an official Wav2Lip package root for checkpoint adaptation")


def convert_local_generator_state_to_official(local_state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    converted: dict[str, torch.Tensor] = {}
    for key, value in local_state.items():
        if key.startswith("output_alpha."):
            continue
        mapped = key.replace(".conv.", ".conv_block.").replace(".deconv.", ".conv_block.")
        if mapped.startswith("output_face."):
            mapped = "output_block." + mapped[len("output_face."):]
        converted[mapped] = value
    return converted


def convert_local_generator_checkpoint_to_official(
    checkpoint_path: Path,
    ckpt: dict,
    official_out: Path | None,
) -> tuple[Path, list[Path]]:
    cfg = ckpt["config"]
    model_cfg = cfg.get("model", {})
    img_size = int(model_cfg.get("img_size", 0))
    base_channels = int(model_cfg.get("base_channels", 0))
    predict_alpha = bool(model_cfg.get("predict_alpha", False))

    if img_size != 96 or base_channels != 32 or predict_alpha:
        raise RuntimeError(
            "Official-style adapter only supports the 96x96 no-alpha mirror generator "
            f"(got img_size={img_size}, base_channels={base_channels}, predict_alpha={predict_alpha})"
        )

    if official_out is None:
        official_out = checkpoint_path.with_name(f"{checkpoint_path.stem}_official_wav2lip.pth")
    official_out.parent.mkdir(parents=True, exist_ok=True)

    Wav2Lip = load_official_wav2lip_class()
    official_model = Wav2Lip()
    official_state = convert_local_generator_state_to_official(ckpt["generator"])
    official_model.load_state_dict(official_state, strict=True)
    payload = {
        "state_dict": official_model.state_dict(),
        "checkpoint_kind": "official_wav2lip_adapter",
        "source_checkpoint": str(checkpoint_path),
        "source_epoch": ckpt.get("epoch"),
        "source_global_step": ckpt.get("global_step"),
        "source_best_off_eval_step": ckpt.get("best_off_eval_step"),
    }
    torch.save(payload, official_out)
    log(f"Saved official-style checkpoint: {official_out}")
    return official_out, [checkpoint_path, official_out]


def prepare_benchmark_checkpoint(
    checkpoint_path: Path,
    prepared_out: Path | None,
    checkpoint_adapter: str,
) -> tuple[Path, list[Path], dict]:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, torch.jit.RecursiveScriptModule):
        return checkpoint_path, [checkpoint_path], {
            "checkpoint_format": "official_torchscript",
            "benchmark_format": "official",
            "adapter": "native",
        }

    if is_local_generator_checkpoint(ckpt):
        if checkpoint_adapter == "official_wav2lip":
            prepared_checkpoint, upload_candidates = convert_local_generator_checkpoint_to_official(
                checkpoint_path=checkpoint_path,
                ckpt=ckpt,
                official_out=prepared_out,
            )
            return prepared_checkpoint, upload_candidates, {
                "checkpoint_format": "local_generator",
                "benchmark_format": "official",
                "adapter": "official_wav2lip",
            }

        keep_keys = {"generator", "config", "epoch"}
        if set(ckpt.keys()).issubset(keep_keys):
            log(f"Infer-only checkpoint already provided: {checkpoint_path}")
            return checkpoint_path, [checkpoint_path], {
                "checkpoint_format": "local_generator_infer_only",
                "benchmark_format": "custom",
                "adapter": "native",
            }

        if prepared_out is None:
            prepared_out = checkpoint_path.with_name(f"{checkpoint_path.stem}_infer_only.pth")
        prepared_out.parent.mkdir(parents=True, exist_ok=True)

        slim = {key: ckpt[key] for key in ("generator", "config", "epoch") if key in ckpt}
        torch.save(slim, prepared_out)
        log(f"Saved infer-only checkpoint: {prepared_out}")
        return prepared_out, [checkpoint_path, prepared_out], {
            "checkpoint_format": "local_generator_training",
            "benchmark_format": "custom",
            "adapter": "native",
        }

    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        return checkpoint_path, [checkpoint_path], {
            "checkpoint_format": "official_state_dict_wrapper",
            "benchmark_format": "official",
            "adapter": "native",
        }

    if is_plain_state_dict_checkpoint(ckpt):
        return checkpoint_path, [checkpoint_path], {
            "checkpoint_format": "official_state_dict",
            "benchmark_format": "official",
            "adapter": "native",
        }

    raise RuntimeError(f"Unsupported checkpoint format for benchmark publish: {checkpoint_path}")


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
        help="Optional output path for the prepared benchmark checkpoint",
    )
    parser.add_argument(
        "--checkpoint-adapter",
        default="native",
        choices=("native", "official_wav2lip"),
        help="How to adapt a local generator checkpoint before benchmarking",
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

    prepared_out = resolve_path(args.infer_only_out) if args.infer_only_out else None
    benchmark_checkpoint, upload_candidates, checkpoint_info = prepare_benchmark_checkpoint(
        checkpoint_path,
        prepared_out,
        args.checkpoint_adapter,
    )

    run_name = args.run_name or default_run_name(checkpoint_path)
    output_dir = resolve_path(args.output_dir) if args.output_dir else (
        TRAINING_ROOT / "output" / "checkpoint_benchmarks" / run_name
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "created_at": timestamp(),
        "run_name": run_name,
        "checkpoint": str(checkpoint_path),
        "benchmark_checkpoint": str(benchmark_checkpoint),
        "infer_only_checkpoint": str(benchmark_checkpoint),
        "checkpoint_info": checkpoint_info,
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
            out_path = output_dir / benchmark_output_name(face_path, audio_path, benchmark_checkpoint)
            cmd = [
                sys.executable,
                str(benchmark_script),
                "--face",
                str(face_path),
                "--audio",
                str(audio_path),
                "--checkpoint",
                str(benchmark_checkpoint),
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
