#!/usr/bin/env python3
"""
Run a fixed-weight SyncNet ablation from a shared generator checkpoint.

Each arm:
1. Resumes from the same generator checkpoint.
2. Trains for exactly one epoch, capped by max_batches_per_epoch.
3. Runs the official benchmark on one or more face samples.

The goal is to compare teacher identity and sync-loss aggressiveness without
changing the starting point between runs.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import torch
import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
TRAINING_ROOT = SCRIPT_DIR.parent
REPO_ROOT = TRAINING_ROOT.parent


ABlation_ARMS = (
    ("official_w0p010", "official", 0.010),
    ("local_w0p000", "local", 0.000),
    ("local_w0p003", "local", 0.003),
    ("local_w0p006", "local", 0.006),
    ("local_w0p010", "local", 0.010),
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run fixed-weight generator SyncNet ablation")
    parser.add_argument("--base-config", required=True, help="YAML config to clone for all ablation arms")
    parser.add_argument("--resume", required=True, help="Shared generator checkpoint to resume from")
    parser.add_argument("--speaker-list", required=True, help="Train speaker snapshot")
    parser.add_argument("--val-speaker-list", required=True, help="Validation speaker snapshot")
    parser.add_argument("--official-syncnet", required=True, help="Official SyncNet checkpoint path")
    parser.add_argument("--local-syncnet", required=True, help="Local SyncNet checkpoint path")
    parser.add_argument(
        "--faces",
        nargs="+",
        required=True,
        help="One or more face sample videos/images for post-train benchmarking",
    )
    parser.add_argument("--audio", required=True, help="Audio sample for post-train benchmarking")
    parser.add_argument("--s3fd-path", required=True, help="S3FD detector checkpoint for benchmarking")
    parser.add_argument(
        "--output-root",
        default=None,
        help="Directory for configs, manifests, run folders, and benchmark outputs "
        "(defaults under training/output)",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=25000,
        help="Maximum batches to process in the single ablation epoch",
    )
    parser.add_argument(
        "--eval-interval-steps",
        type=int,
        default=3000,
        help="How often to log val_sync in monitor-only mode",
    )
    parser.add_argument(
        "--eval-batches",
        type=int,
        default=700,
        help="How many validation batches to average per val_sync evaluation",
    )
    return parser.parse_args()


def run_cmd(cmd: list[str], cwd: Path | None = None):
    print(f"[cmd] {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, cwd=str(cwd) if cwd else None)


def load_yaml(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def save_yaml(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def latest_generator_ckpt(generator_dir: Path) -> Path:
    checkpoints = sorted(generator_dir.glob("generator_epoch*.pth"))
    if not checkpoints:
        raise RuntimeError(f"No generator checkpoints found in {generator_dir}")
    return checkpoints[-1]


def main():
    args = parse_args()
    base_config_path = Path(args.base_config).resolve()
    resume_path = Path(args.resume).resolve()
    speaker_list = Path(args.speaker_list).resolve()
    val_speaker_list = Path(args.val_speaker_list).resolve()
    official_syncnet = Path(args.official_syncnet).resolve()
    local_syncnet = Path(args.local_syncnet).resolve()
    faces = [Path(face).resolve() for face in args.faces]
    audio = Path(args.audio).resolve()
    s3fd_path = Path(args.s3fd_path).resolve()

    if args.output_root:
        output_root = Path(args.output_root).resolve()
    else:
        output_root = TRAINING_ROOT / "output" / "generator_sync_weight_ablation_e1_20260328"
    output_root.mkdir(parents=True, exist_ok=True)
    configs_dir = output_root / "configs"
    manifests_dir = output_root / "manifests"
    benchmarks_root = output_root / "benchmarks"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    benchmarks_root.mkdir(parents=True, exist_ok=True)

    resume_ck = torch.load(resume_path, map_location="cpu", weights_only=False)
    resume_epoch = int(resume_ck["epoch"])
    next_epoch = resume_epoch + 1
    total_epochs = next_epoch + 1

    base_cfg = load_yaml(base_config_path)
    manifest = {
        "base_config": str(base_config_path),
        "resume": str(resume_path),
        "resume_epoch": resume_epoch,
        "max_batches_per_epoch": args.max_batches,
        "arms": [],
    }

    for arm_name, teacher_kind, sync_weight in ABlation_ARMS:
        arm_run_dir = output_root / arm_name
        arm_generator_dir = arm_run_dir / "generator"
        arm_bench_dir = benchmarks_root / arm_name
        arm_bench_dir.mkdir(parents=True, exist_ok=True)
        cfg = deepcopy(base_cfg)
        cfg["generator"]["epochs"] = total_epochs
        cfg["generator"]["eval_interval_steps"] = args.eval_interval_steps
        cfg["generator"]["eval_batches"] = args.eval_batches
        cfg.setdefault("training", {})
        cfg["training"]["max_batches_per_epoch"] = args.max_batches
        cfg["training"]["output_dir"] = str(arm_run_dir.relative_to(TRAINING_ROOT))

        loss_cfg = cfg["generator"]["loss"]
        loss_cfg["sync"] = float(sync_weight)
        loss_cfg["sync_initial"] = float(sync_weight)
        loss_cfg["sync_official_schedule"] = False
        loss_cfg["sync_adaptive_schedule"] = False
        loss_cfg["sync_eval_monitor_only"] = True
        loss_cfg["sync_warmup_epochs"] = 0

        cfg_path = configs_dir / f"{arm_name}.yaml"
        save_yaml(cfg_path, cfg)

        teacher_path = official_syncnet if teacher_kind == "official" else local_syncnet
        train_cmd = [
            sys.executable,
            "scripts/train_generator.py",
            "--config",
            str(cfg_path),
            "--syncnet",
            str(teacher_path),
            "--resume",
            str(resume_path),
            "--speaker-list",
            str(speaker_list),
            "--val-speaker-list",
            str(val_speaker_list),
        ]
        run_cmd(train_cmd, cwd=TRAINING_ROOT)

        ckpt_path = latest_generator_ckpt(arm_generator_dir)
        benchmark_outputs = []
        for face_path in faces:
            outfile = arm_bench_dir / f"{face_path.stem}_{audio.stem}_{arm_name}.mp4"
            bench_cmd = [
                sys.executable,
                str(TRAINING_ROOT / "scripts" / "run_official_wav2lip_benchmark.py"),
                "--face",
                str(face_path),
                "--audio",
                str(audio),
                "--checkpoint",
                str(ckpt_path),
                "--outfile",
                str(outfile),
                "--device",
                "cuda",
                "--detector_device",
                "cuda",
                "--s3fd_path",
                str(s3fd_path),
            ]
            run_cmd(bench_cmd, cwd=REPO_ROOT)
            benchmark_outputs.append(str(outfile))

        arm_manifest = {
            "name": arm_name,
            "teacher_kind": teacher_kind,
            "sync_weight": sync_weight,
            "teacher_path": str(teacher_path),
            "config_path": str(cfg_path),
            "run_dir": str(arm_run_dir),
            "checkpoint": str(ckpt_path),
            "benchmarks": benchmark_outputs,
        }
        manifest["arms"].append(arm_manifest)
        with (manifests_dir / f"{arm_name}.json").open("w") as f:
            json.dump(arm_manifest, f, indent=2)

    manifest_path = manifests_dir / "ablation_manifest.json"
    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[done] wrote {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
