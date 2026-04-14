#!/usr/bin/env python3
"""
Launch SyncNet training and automatically resume from the latest available checkpoint.

Priority order:
1. syncnet/syncnet_latest.pth in the configured output directory
2. latest syncnet_epochXXX.pth in the configured output directory
3. --initial-resume (if provided)
4. fresh start

This script is intended to be safe for manual launches and reboot hooks.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from config_loader import load_config


SCRIPT_DIR = Path(__file__).resolve().parent
TRAINING_ROOT = SCRIPT_DIR.parent


def log(msg: str):
    print(f"[syncnet-launcher] {msg}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Run SyncNet train with resume-from-latest behavior")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", default=None, help="Configured training.output_dir (optional override)")
    parser.add_argument("--initial-resume", default=None)
    parser.add_argument("--speaker-list", default=None)
    parser.add_argument("--val-speaker-list", default=None)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def syncnet_output_dir(config_path: Path, output_dir: str | None) -> Path:
    if output_dir:
        return (TRAINING_ROOT / output_dir / "syncnet").resolve()
    cfg = load_config(config_path)
    return (TRAINING_ROOT / cfg["training"]["output_dir"] / "syncnet").resolve()


def find_resume_checkpoint(output_dir: Path, initial_resume: Path | None) -> tuple[Path | None, str]:
    latest_path = output_dir / "syncnet_latest.pth"
    if latest_path.exists():
        return latest_path, "latest"

    epoch_ckpts = sorted(output_dir.glob("syncnet_epoch*.pth"))
    if epoch_ckpts:
        return epoch_ckpts[-1], "epoch"

    if initial_resume and initial_resume.exists():
        return initial_resume, "initial"

    return None, "fresh"


def train_already_running(config_path: Path) -> bool:
    target = str(config_path.resolve())
    try:
        output = subprocess.check_output(["ps", "-ef"], text=True)
    except Exception:
        return False
    for line in output.splitlines():
        if "train_syncnet.py" in line and target in line:
            return True
    return False


def main():
    args = parse_args()
    config_path = Path(args.config).resolve()
    initial_resume = Path(args.initial_resume).resolve() if args.initial_resume else None
    output_dir = syncnet_output_dir(config_path, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    resume_path, resume_source = find_resume_checkpoint(output_dir, initial_resume)
    cmd = [
        args.python,
        "-u",
        str(TRAINING_ROOT / "scripts" / "train_syncnet.py"),
        "--config",
        str(config_path),
    ]
    if resume_path is not None:
        cmd.extend(["--resume", str(resume_path)])
    if args.speaker_list:
        cmd.extend(["--speaker-list", args.speaker_list])
    if args.val_speaker_list:
        cmd.extend(["--val-speaker-list", args.val_speaker_list])

    log(f"output_dir={output_dir}")
    log(f"resume_source={resume_source}")
    if resume_path is not None:
        log(f"resume_path={resume_path}")
    else:
        log("resume_path=<fresh>")
    log(f"cmd={' '.join(cmd)}")

    if args.dry_run:
        return

    if train_already_running(config_path):
        log(f"train_syncnet already running for config {config_path}; exiting")
        return

    subprocess.run(cmd, cwd=str(TRAINING_ROOT), check=True)


if __name__ == "__main__":
    main()
