#!/usr/bin/env python3
"""
Wait for a running SyncNet job to finish, benchmark all saved epochs against the
official Wav2Lip teacher, select the winner, and launch generator training with
the selected teacher.
"""

import argparse
import copy
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import yaml
from config_loader import load_config


TRAINING_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = TRAINING_ROOT.parent


def log(message):
    import datetime

    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S %Z")
    print(f"[watcher] {ts} {message}", flush=True)


def save_yaml(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def read_text(path):
    if not path.exists():
        return ""
    return path.read_text(errors="ignore")


def pid_is_running(pid):
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def resolve_training_path(raw_path):
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return TRAINING_ROOT / path


def resolve_output_rel(path):
    try:
        return str(path.relative_to(TRAINING_ROOT))
    except ValueError:
        return str(path)


def run_cmd(cmd, cwd=TRAINING_ROOT):
    pretty = " ".join(str(part) for part in cmd)
    log(f"Running: {pretty}")
    subprocess.run(cmd, cwd=cwd, check=True)


def unique_paths(paths):
    seen = set()
    result = []
    for path in paths:
        resolved = Path(path).resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        result.append(Path(path))
    return result


def derive_generator_output_dir(syncnet_output_dir, explicit_output_dir):
    if explicit_output_dir:
        return explicit_output_dir
    sync_run_name = Path(syncnet_output_dir).name
    return Path("output") / f"generator_medium_x96_b8_from_{sync_run_name}"


def derive_generator_lazy_cache_root(sync_cfg, syncnet_output_dir, explicit_lazy_root):
    if explicit_lazy_root:
        return explicit_lazy_root
    data_cfg = sync_cfg.get("data", {})
    sync_lazy = data_cfg.get("lazy_cache_root")
    if sync_lazy:
        return str(sync_lazy)
    return None


def derive_generator_val_speaker_list(template_config_path, explicit_val_speaker_list):
    if explicit_val_speaker_list:
        return explicit_val_speaker_list
    template_dir = resolve_training_path(template_config_path).parent
    candidate = template_dir / "val_confident_medium_allowlist_officialish.txt"
    if candidate.exists():
        return str(candidate)
    return None


def build_generator_config(sync_cfg, template_cfg, args):
    cfg = copy.deepcopy(template_cfg)

    cfg["model"] = copy.deepcopy(sync_cfg.get("model", {}))
    cfg["audio"] = copy.deepcopy(sync_cfg.get("audio", {}))

    cfg.setdefault("syncnet", {})
    cfg["syncnet"].update(copy.deepcopy(sync_cfg.get("syncnet", {})))
    cfg["syncnet"]["epochs"] = 0

    cfg.setdefault("generator", {})
    if args.generator_batch_size is not None:
        cfg["generator"]["batch_size"] = args.generator_batch_size
    if args.generator_epochs is not None:
        cfg["generator"]["epochs"] = args.generator_epochs

    cfg.setdefault("data", {})
    sync_data = sync_cfg.get("data", {})
    for key in (
        "hdtf_root",
        "talkvid_root",
        "fps",
        "num_workers",
        "persistent_workers",
        "prefetch_factor",
        "cache_size",
        "skip_bad_samples",
        "ffmpeg_bin",
        "materialize_timeout",
        "min_samples_per_speaker",
    ):
        if key in sync_data:
            cfg["data"][key] = copy.deepcopy(sync_data[key])
    cfg["data"]["crop_size"] = cfg["data"].get("crop_size", 256)
    cfg["data"]["materialize_frames_size"] = cfg["model"]["img_size"]
    cfg["data"]["lazy_cache_root"] = derive_generator_lazy_cache_root(
        sync_cfg, args.syncnet_output_dir, args.generator_lazy_cache_root
    )

    cfg.setdefault("training", {})
    sync_training = sync_cfg.get("training", {})
    for key in (
        "device",
        "mixed_precision",
        "allow_tf32",
        "cudnn_benchmark",
        "gradient_clip",
        "save_every",
        "sample_every",
        "float32_matmul_precision",
    ):
        if key in sync_training:
            cfg["training"][key] = copy.deepcopy(sync_training[key])
    cfg["training"]["output_dir"] = str(
        derive_generator_output_dir(args.syncnet_output_dir, args.generator_output_dir)
    )

    return cfg


def wait_for_syncnet(syncnet_dir, expected_epochs, syncnet_pid, poll_seconds):
    pipeline_log = syncnet_dir / "pipeline.log"
    syncnet_ckpt_dir = syncnet_dir / "syncnet"

    while True:
        ckpts = sorted(syncnet_ckpt_dir.glob("syncnet_epoch*.pth"))
        pipeline_text = read_text(pipeline_log)
        complete = "SyncNet training complete!" in pipeline_text
        pid_running = pid_is_running(syncnet_pid) if syncnet_pid else None

        if complete:
            log(f"Found SyncNet completion marker with {len(ckpts)} checkpoints")
            return

        if syncnet_pid and not pid_running:
            if ckpts:
                log(
                    f"SyncNet PID {syncnet_pid} exited without explicit completion marker; "
                    f"proceeding with {len(ckpts)} checkpoints"
                )
                return
            raise RuntimeError(
                f"SyncNet PID {syncnet_pid} exited before any checkpoints appeared"
            )

        last_epoch = None
        if ckpts:
            last_name = ckpts[-1].stem
            if "epoch" in last_name:
                try:
                    last_epoch = int(last_name.split("epoch", 1)[1])
                except ValueError:
                    last_epoch = None

        status_bits = [f"checkpoints={len(ckpts)}"]
        if last_epoch is not None:
            status_bits.append(f"last_epoch={last_epoch}")
        if expected_epochs is not None:
            status_bits.append(f"target_epochs={expected_epochs}")
        if syncnet_pid:
            status_bits.append(f"pid_running={int(pid_running)}")
        log("Waiting for SyncNet: " + " ".join(status_bits))
        time.sleep(poll_seconds)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--syncnet-config", required=True)
    parser.add_argument("--syncnet-output-dir", required=True)
    parser.add_argument("--speaker-snapshot", required=True)
    parser.add_argument("--official-syncnet", required=True)
    parser.add_argument("--generator-template-config", default="configs/lipsync_cuda3090_hdtf_talkvid.yaml")
    parser.add_argument("--syncnet-pid", type=int, default=0)
    parser.add_argument("--compare-samples", type=int, default=200)
    parser.add_argument("--compare-seed", type=int, default=123)
    parser.add_argument("--compare-device", default="cuda")
    parser.add_argument("--generator-batch-size", type=int, default=8)
    parser.add_argument("--generator-epochs", type=int, default=None)
    parser.add_argument("--generator-output-dir", default=None)
    parser.add_argument("--generator-lazy-cache-root", default=None)
    parser.add_argument("--generator-val-speaker-list", default=None)
    parser.add_argument("--generator-resume", default=None)
    parser.add_argument(
        "--extra-checkpoint",
        action="append",
        default=[],
        help="Additional SyncNet checkpoint(s) to include in compare/select",
    )
    parser.add_argument("--poll-seconds", type=int, default=60)
    args = parser.parse_args()

    syncnet_config_path = resolve_training_path(args.syncnet_config)
    syncnet_output_dir = resolve_training_path(args.syncnet_output_dir)
    speaker_snapshot = resolve_training_path(args.speaker_snapshot)
    official_syncnet = resolve_training_path(args.official_syncnet)
    generator_template_config = resolve_training_path(args.generator_template_config)
    generator_resume = resolve_training_path(args.generator_resume) if args.generator_resume else None
    generator_val_speaker_list = derive_generator_val_speaker_list(
        args.generator_template_config,
        args.generator_val_speaker_list,
    )

    sync_cfg = load_config(syncnet_config_path)
    template_cfg = load_config(generator_template_config)

    sync_epochs = sync_cfg.get("syncnet", {}).get("epochs")
    wait_for_syncnet(
        syncnet_output_dir,
        expected_epochs=sync_epochs,
        syncnet_pid=args.syncnet_pid or None,
        poll_seconds=args.poll_seconds,
    )

    syncnet_ckpt_dir = syncnet_output_dir / "syncnet"
    checkpoints = sorted(syncnet_ckpt_dir.glob("syncnet_epoch*.pth"))
    extra_checkpoints = [resolve_training_path(path) for path in args.extra_checkpoint]
    for extra_ckpt in extra_checkpoints:
        if not extra_ckpt.exists():
            raise RuntimeError(f"Extra checkpoint not found: {extra_ckpt}")
    all_checkpoints = unique_paths([*checkpoints, *extra_checkpoints])
    if not all_checkpoints:
        raise RuntimeError(f"No SyncNet checkpoints found in {syncnet_ckpt_dir}")

    compare_json = syncnet_output_dir / "syncnet_teacher_compare_all_epochs_vs_official.json"
    selected_json = syncnet_output_dir / "syncnet_selected_teacher_all_epochs_vs_official.json"

    if not compare_json.exists():
        compare_cmd = [
            sys.executable,
            "scripts/compare_syncnet_teachers.py",
            "--speaker-snapshot",
            str(speaker_snapshot),
            "--official-checkpoint",
            str(official_syncnet),
            "--output",
            str(compare_json),
            "--samples",
            str(args.compare_samples),
            "--seed",
            str(args.compare_seed),
            "--device",
            args.compare_device,
            "--fps",
            str(sync_cfg["data"]["fps"]),
            "--T",
            str(sync_cfg["syncnet"]["T"]),
            "--img-size",
            str(sync_cfg["model"]["img_size"]),
            "--mel-step-size",
            str(sync_cfg["model"]["mel_steps"]),
            "--sample-rate",
            str(sync_cfg["audio"]["sample_rate"]),
            "--hop-size",
            str(sync_cfg["audio"]["hop_size"]),
            "--n-fft",
            str(sync_cfg["audio"]["n_fft"]),
            "--win-size",
            str(sync_cfg["audio"]["win_size"]),
            "--n-mels",
            str(sync_cfg["audio"]["n_mels"]),
            "--fmin",
            str(sync_cfg["audio"]["fmin"]),
            "--fmax",
            str(sync_cfg["audio"]["fmax"]),
            "--preemphasis",
            str(sync_cfg["audio"]["preemphasis"]),
            "--ffmpeg-bin",
            str(sync_cfg["data"].get("ffmpeg_bin", "ffmpeg")),
            "--materialize-timeout",
            str(sync_cfg["data"].get("materialize_timeout", 600)),
            "--cache-size",
            str(sync_cfg["data"].get("cache_size", 16)),
        ]
        lazy_cache_root = sync_cfg.get("data", {}).get("lazy_cache_root")
        if lazy_cache_root:
            compare_cmd.extend(["--lazy-cache-root", str(lazy_cache_root)])
        for root_key in ("hdtf_root", "talkvid_root"):
            root = sync_cfg.get("data", {}).get(root_key)
            if root:
                compare_cmd.extend(["--processed-root", root])
        compare_cmd.extend(["--checkpoints", *[str(path) for path in all_checkpoints]])
        run_cmd(compare_cmd)
    else:
        log(f"Reusing existing compare json: {compare_json}")

    if not selected_json.exists():
        select_cmd = [
            sys.executable,
            "scripts/select_best_syncnet_teacher.py",
            "--compare-json",
            str(compare_json),
            "--official-checkpoint",
            str(official_syncnet),
            "--output",
            str(selected_json),
            "--checkpoints",
            *[str(path) for path in all_checkpoints],
        ]
        run_cmd(select_cmd)
    else:
        log(f"Reusing existing selected-teacher json: {selected_json}")

    with open(selected_json) as f:
        selected = json.load(f)
    selected_teacher = selected["winner_path"]
    log(
        "Selected teacher: "
        f"{selected['winner_name']} kind={selected['winner_kind']} path={selected_teacher}"
    )

    generator_cfg = build_generator_config(sync_cfg, template_cfg, args)
    generator_output_dir = resolve_training_path(generator_cfg["training"]["output_dir"])
    effective_generator_cfg = syncnet_output_dir / "generator_selected_teacher_config.yaml"
    save_yaml(effective_generator_cfg, generator_cfg)
    log(f"Saved generator config: {effective_generator_cfg}")

    existing_generator_ckpts = sorted((generator_output_dir / "generator").glob("generator_epoch*.pth"))
    if existing_generator_ckpts:
        log(
            f"Generator output already contains {len(existing_generator_ckpts)} checkpoints; "
            "skipping generator launch"
        )
        return

    generator_cmd = [
        sys.executable,
        "scripts/train_generator.py",
        "--config",
        str(effective_generator_cfg),
        "--syncnet",
        str(selected_teacher),
        "--speaker-list",
        str(speaker_snapshot),
    ]
    if generator_val_speaker_list:
        generator_cmd.extend(["--val-speaker-list", str(resolve_training_path(generator_val_speaker_list))])
    if generator_resume:
        if not generator_resume.exists():
            raise RuntimeError(f"Generator resume checkpoint not found: {generator_resume}")
        generator_cmd.extend(["--resume", str(generator_resume)])
    run_cmd(generator_cmd)


if __name__ == "__main__":
    main()
