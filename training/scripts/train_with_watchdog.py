#!/usr/bin/env python3
"""
Run generator training epoch-by-epoch and evaluate quality after each checkpoint.

The watchdog monitors a composite quality score built from:
  - train sync reward from the epoch log
  - same-face different-mel margin
  - GT correct-audio margin
  - same-mel different-reference stability margin

If the score degrades beyond a threshold for a configurable number of epochs,
training stops early.
"""

import argparse
import copy
import datetime
import json
import os
import re
import subprocess
import sys
import tempfile

import yaml


TRAINING_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(TRAINING_ROOT, "scripts"))
import check_audio_sensitivity as cas
from config_loader import load_config


EPOCH_RE = re.compile(
    r"Epoch\s+(?P<epoch>\d+)/(?P<epochs>\d+):.*?reward=(?P<reward>-?\d+(?:\.\d+)?)"
)


def write_yaml(path, data):
    with open(path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def find_checkpoint(output_dir, epoch):
    return os.path.join(output_dir, "generator", f"generator_epoch{epoch:03d}.pth")


def load_speaker_allowlist(path):
    if not path:
        return None
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def run_train_epoch(python_exe, config_path, syncnet_path, resume_path, cwd, speaker_list_path=None):
    cmd = [
        python_exe,
        "-u",
        os.path.join("scripts", "train_generator.py"),
        "--config",
        config_path,
        "--syncnet",
        syncnet_path,
    ]
    if resume_path:
        cmd.extend(["--resume", resume_path])
    if speaker_list_path:
        cmd.extend(["--speaker-list", speaker_list_path])

    print(f"[watchdog] Running: {' '.join(cmd)}", flush=True)
    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    reward = None
    for line in proc.stdout:
        print(line, end="", flush=True)
        match = EPOCH_RE.search(line)
        if match:
            reward = float(match.group("reward"))

    ret = proc.wait()
    if ret != 0:
        raise RuntimeError(f"train_generator.py exited with code {ret}")
    return reward


def run_sanity_suite(checkpoint_path, cfg, syncnet_path, samples, output_dir, speaker_allowlist=None):
    device = cfg["training"]["device"]
    generator = cas.load_generator(checkpoint_path, cfg, device)
    syncnet = cas.load_syncnet(syncnet_path, device)
    dataset = cas.build_dataset(cfg, speaker_allowlist=speaker_allowlist)

    results = [
        cas.check_same_face_different_mel(generator, syncnet, dataset, device, samples, output_dir),
        cas.check_gt_correct_audio(syncnet, dataset, device, samples),
        cas.check_same_mel_different_ref(generator, dataset, device, samples, output_dir),
    ]
    return results


def build_watchdog_summary(results, train_reward):
    by_name = {item["name"]: item for item in results}
    same_face_margin = (
        by_name["same_face_different_mel"]["correct_sync"] -
        by_name["same_face_different_mel"]["swapped_under_original_audio"]
    )
    gt_margin = by_name["gt_correct_audio_beats_wrong"]["margin"]
    ref_margin = (
        by_name["same_mel_different_ref"]["upper_delta"] -
        by_name["same_mel_different_ref"]["mouth_delta"]
    )
    pass_count = sum(1 for item in results if item["pass"])

    score = train_reward + same_face_margin + gt_margin + ref_margin + 0.02 * pass_count
    return {
        "train_reward": train_reward,
        "same_face_margin": same_face_margin,
        "gt_margin": gt_margin,
        "ref_margin": ref_margin,
        "pass_count": pass_count,
        "score": score,
    }


def append_history(path, record):
    def to_builtin(value):
        if isinstance(value, dict):
            return {k: to_builtin(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [to_builtin(v) for v in value]
        if hasattr(value, "item") and callable(value.item):
            try:
                return value.item()
            except Exception:
                pass
        return value

    with open(path, "a") as f:
        f.write(json.dumps(to_builtin(record), ensure_ascii=True) + "\n")


def write_status(path, lines):
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def load_existing_history(path, start_epoch):
    if start_epoch <= 0 or not os.path.exists(path):
        return None, None

    best_score = None
    best_epoch = None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            epoch = int(record.get("epoch", -1))
            if epoch >= start_epoch:
                continue
            score = float(record.get("watchdog", {}).get("score", float("-inf")))
            if best_score is None or score > best_score:
                best_score = score
                best_epoch = epoch

    return best_score, best_epoch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--syncnet", required=True)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--degrade-threshold", type=float, default=0.03)
    parser.add_argument("--min-watchdog-epoch", type=int, default=2)
    parser.add_argument("--start-epoch", type=int, default=0)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--speaker-list", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    total_epochs = int(cfg["generator"]["epochs"])
    output_dir = cfg["training"]["output_dir"]
    abs_output_dir = os.path.join(TRAINING_ROOT, output_dir)
    ensure_dir(abs_output_dir)

    history_path = os.path.join(abs_output_dir, "watchdog_history.jsonl")
    status_path = os.path.join(abs_output_dir, "watchdog_status.txt")
    speaker_allowlist = load_speaker_allowlist(args.speaker_list)
    best_score, best_epoch = load_existing_history(history_path, args.start_epoch)
    degrade_streak = 0

    if args.start_epoch > 0:
        print(f"[watchdog] Resuming watchdog from epoch {args.start_epoch}", flush=True)
        if args.resume:
            print(f"[watchdog] Initial resume checkpoint: {args.resume}", flush=True)
        if best_score is not None:
            print(f"[watchdog] Existing best score from prior epochs: "
                  f"{best_score:.4f} at epoch {best_epoch}", flush=True)

    for epoch in range(args.start_epoch, total_epochs):
        epoch_cfg = copy.deepcopy(cfg)
        epoch_cfg["generator"]["epochs"] = epoch + 1

        ensure_dir(os.path.join(abs_output_dir, "tmp"))
        tmp_cfg_path = os.path.join(abs_output_dir, "tmp", f"epoch_{epoch:03d}.yaml")
        write_yaml(tmp_cfg_path, epoch_cfg)

        if epoch == args.start_epoch and args.resume:
            resume_path = args.resume
        else:
            resume_path = find_checkpoint(output_dir, epoch - 1) if epoch > 0 else None
        train_reward = run_train_epoch(
            args.python,
            tmp_cfg_path,
            args.syncnet,
            resume_path,
            TRAINING_ROOT,
            speaker_list_path=args.speaker_list,
        )

        checkpoint_path = find_checkpoint(output_dir, epoch)
        sanity_dir = os.path.join(abs_output_dir, f"sanity_epoch{epoch:03d}")
        ensure_dir(sanity_dir)
        results = run_sanity_suite(
            checkpoint_path,
            cfg,
            args.syncnet,
            args.samples,
            sanity_dir,
            speaker_allowlist=speaker_allowlist,
        )
        watchdog = build_watchdog_summary(results, train_reward or 0.0)

        record = {
            "timestamp": datetime.datetime.now().isoformat(),
            "epoch": epoch,
            "checkpoint": checkpoint_path,
            "watchdog": watchdog,
            "results": results,
        }
        append_history(history_path, record)

        status_lines = [
            f"epoch={epoch}",
            f"checkpoint={checkpoint_path}",
            f"train_reward={watchdog['train_reward']:.4f}",
            f"same_face_margin={watchdog['same_face_margin']:.4f}",
            f"gt_margin={watchdog['gt_margin']:.4f}",
            f"ref_margin={watchdog['ref_margin']:.4f}",
            f"pass_count={watchdog['pass_count']}",
            f"score={watchdog['score']:.4f}",
        ]
        write_status(status_path, status_lines)

        print(f"[watchdog] Epoch {epoch}: score={watchdog['score']:.4f}, "
              f"reward={watchdog['train_reward']:.4f}, pass_count={watchdog['pass_count']}", flush=True)
        for item in results:
            print(cas.format_result(item), flush=True)

        if best_score is None or watchdog["score"] > best_score:
            best_score = watchdog["score"]
            best_epoch = epoch
            degrade_streak = 0
        elif epoch + 1 >= args.min_watchdog_epoch and watchdog["score"] < (best_score - args.degrade_threshold):
            degrade_streak += 1
            print(f"[watchdog] Degradation detected: score {watchdog['score']:.4f} "
                  f"< best {best_score:.4f} - {args.degrade_threshold:.4f} "
                  f"(streak {degrade_streak}/{args.patience})", flush=True)
            if degrade_streak >= args.patience:
                print(f"[watchdog] Early stopping at epoch {epoch}. Best epoch: {best_epoch}", flush=True)
                break
        else:
            degrade_streak = 0

    print("[watchdog] Training loop finished", flush=True)


if __name__ == "__main__":
    main()
