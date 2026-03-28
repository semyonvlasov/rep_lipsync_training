#!/usr/bin/env python3
"""
Launch generator training and automatically resume from the latest available checkpoint.

Priority order:
1. generator/generator_latest.pth in the configured output directory
2. latest generator_epochXXX.pth in the configured output directory
3. --initial-resume (if provided)
4. fresh start

This script is intended to be safe for manual launches and reboot hooks.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

try:
    import yaml
except ModuleNotFoundError:
    yaml = None


SCRIPT_DIR = Path(__file__).resolve().parent
TRAINING_ROOT = SCRIPT_DIR.parent


def log(msg: str):
    print(f"[generator-launcher] {msg}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Run generator train with resume-from-latest behavior")
    parser.add_argument("--config", required=True)
    parser.add_argument("--syncnet", required=True)
    parser.add_argument("--output-dir", default=None, help="Configured training.output_dir (optional override)")
    parser.add_argument("--initial-resume", default=None)
    parser.add_argument("--speaker-list", default=None)
    parser.add_argument("--val-speaker-list", default=None)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_yaml(path: Path) -> dict:
    if yaml is None:
        raise RuntimeError("PyYAML is required unless --output-dir is provided explicitly")
    with path.open() as f:
        return yaml.safe_load(f)


def generator_output_dir(config_path: Path, output_dir: str | None) -> Path:
    if output_dir:
        return (TRAINING_ROOT / output_dir / "generator").resolve()
    cfg = load_yaml(config_path)
    return (TRAINING_ROOT / cfg["training"]["output_dir"] / "generator").resolve()


def find_resume_checkpoint(output_dir: Path, initial_resume: Path | None) -> tuple[Path | None, str]:
    latest_path = output_dir / "generator_latest.pth"
    if latest_path.exists():
        return latest_path, "latest"

    epoch_ckpts = sorted(output_dir.glob("generator_epoch*.pth"))
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
    return ("train_generator.py" in output) and (target in output)


def main():
    args = parse_args()
    config_path = Path(args.config).resolve()
    syncnet_path = Path(args.syncnet).resolve()
    initial_resume = Path(args.initial_resume).resolve() if args.initial_resume else None
    output_dir = generator_output_dir(config_path, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    resume_path, resume_source = find_resume_checkpoint(output_dir, initial_resume)
    cmd = [
        args.python,
        "-u",
        str(TRAINING_ROOT / "scripts" / "train_generator.py"),
        "--config",
        str(config_path),
        "--syncnet",
        str(syncnet_path),
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
        log(f"train_generator already running for config {config_path}; exiting")
        return

    subprocess.run(cmd, cwd=str(TRAINING_ROOT), check=True)


if __name__ == "__main__":
    main()
