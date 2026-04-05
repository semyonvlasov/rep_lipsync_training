#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise SystemExit("PyYAML is required for upload_syncnet_results_to_gdrive.py") from exc


def timestamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S %Z")


def log(message: str) -> None:
    print(f"{timestamp()} [SyncnetUpload] {message}", flush=True)


def run_logged(cmd: list[str]) -> None:
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
            log(line)
    rc = proc.wait()
    if rc != 0:
        raise subprocess.CalledProcessError(rc, cmd)


OFFICIAL_BASELINE_RE = re.compile(
    r"Official eval baseline: loss=(?P<loss>[-+]?\d*\.?\d+) acc=(?P<acc>[-+]?\d*\.?\d+) batches=(?P<batches>\d+)"
)
OFFICIAL_OUR_BASELINE_RE = re.compile(
    r"Official our-eval baseline: pairwise_acc=(?P<pairwise>[-+]?\d*\.?\d+) margin=(?P<margin>[-+]?\d*\.?\d+) "
    r"shift_acc=(?P<shift>[-+]?\d*\.?\d+) foreign_acc=(?P<foreign>[-+]?\d*\.?\d+) samples=(?P<samples>\d+)"
)
OFFICIAL_STEP_RE = re.compile(
    r"Official eval step (?P<step>\d+): loss=(?P<loss>[-+]?\d*\.?\d+) acc=(?P<acc>[-+]?\d*\.?\d+) "
    r"vs_official loss_delta=(?P<loss_delta>[-+]?\d*\.?\d+) acc_delta=(?P<acc_delta>[-+]?\d*\.?\d+)"
)
OUR_STEP_RE = re.compile(
    r"Our eval step (?P<step>\d+): pairwise_acc=(?P<pairwise>[-+]?\d*\.?\d+) margin=(?P<margin>[-+]?\d*\.?\d+) "
    r"shift_acc=(?P<shift>[-+]?\d*\.?\d+) foreign_acc=(?P<foreign>[-+]?\d*\.?\d+) "
    r"vs_official acc_mean_delta=(?P<acc_delta>[-+]?\d*\.?\d+) margin_delta=(?P<margin_delta>[-+]?\d*\.?\d+)"
)
BEST_OFF_RE = re.compile(
    r"New best off-eval: loss=(?P<loss>[-+]?\d*\.?\d+) at step (?P<step>\d+) -> (?P<path>\S+)"
)
BEST_OUR_RE = re.compile(
    r"New best our-eval: pairwise_acc=(?P<pairwise>[-+]?\d*\.?\d+) margin=(?P<margin>[-+]?\d*\.?\d+) "
    r"at step (?P<step>\d+) -> (?P<path>\S+)"
)
BEST_OFF_PATH_RE = re.compile(r"Best off-eval checkpoint path: (?P<path>\S+)")
BEST_OUR_PATH_RE = re.compile(r"Best our-eval checkpoint path: (?P<path>\S+)")


def _f(match: re.Match[str], key: str) -> float:
    return float(match.group(key))


def _i(match: re.Match[str], key: str) -> int:
    return int(match.group(key))


def parse_launcher_log(log_path: Path) -> dict:
    official_baseline: dict | None = None
    official_our_baseline: dict | None = None
    official_steps: dict[int, dict] = {}
    our_steps: dict[int, dict] = {}
    best_off_step: int | None = None
    best_our_step: int | None = None
    best_off_path: str | None = None
    best_our_path: str | None = None

    for raw_line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if match := OFFICIAL_BASELINE_RE.search(line):
            official_baseline = {
                "loss": _f(match, "loss"),
                "acc": _f(match, "acc"),
                "batches": _i(match, "batches"),
            }
            continue
        if match := OFFICIAL_OUR_BASELINE_RE.search(line):
            official_our_baseline = {
                "pairwise_acc": _f(match, "pairwise"),
                "margin": _f(match, "margin"),
                "shift_acc": _f(match, "shift"),
                "foreign_acc": _f(match, "foreign"),
                "samples": _i(match, "samples"),
            }
            continue
        if match := OFFICIAL_STEP_RE.search(line):
            step = _i(match, "step")
            official_steps[step] = {
                "step": step,
                "loss": _f(match, "loss"),
                "acc": _f(match, "acc"),
                "loss_delta_vs_official": _f(match, "loss_delta"),
                "acc_delta_vs_official": _f(match, "acc_delta"),
            }
            continue
        if match := OUR_STEP_RE.search(line):
            step = _i(match, "step")
            our_steps[step] = {
                "step": step,
                "pairwise_acc": _f(match, "pairwise"),
                "margin": _f(match, "margin"),
                "shift_acc": _f(match, "shift"),
                "foreign_acc": _f(match, "foreign"),
                "acc_mean_delta_vs_official": _f(match, "acc_delta"),
                "margin_delta_vs_official": _f(match, "margin_delta"),
            }
            continue
        if match := BEST_OFF_RE.search(line):
            best_off_step = _i(match, "step")
            best_off_path = match.group("path")
            continue
        if match := BEST_OUR_RE.search(line):
            best_our_step = _i(match, "step")
            best_our_path = match.group("path")
            continue
        if match := BEST_OFF_PATH_RE.search(line):
            best_off_path = match.group("path")
            continue
        if match := BEST_OUR_PATH_RE.search(line):
            best_our_path = match.group("path")
            continue

    return {
        "official_baseline": official_baseline,
        "official_our_eval_baseline": official_our_baseline,
        "best_off_eval": {
            "checkpoint_path": best_off_path,
            "step": best_off_step,
            "metrics": official_steps.get(best_off_step) if best_off_step is not None else None,
        },
        "best_our_eval": {
            "checkpoint_path": best_our_path,
            "step": best_our_step,
            "metrics": our_steps.get(best_our_step) if best_our_step is not None else None,
        },
    }


def infer_folder_name(run_dir: Path) -> str:
    name = run_dir.name
    for prefix in ("training_cuda3090_", "training_"):
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


def infer_split_paths(run_dir: Path) -> tuple[Path | None, Path | None, Path | None]:
    parent = run_dir.parent
    folder_name = infer_folder_name(run_dir)
    split_dir = parent / f"{folder_name}_split"
    summary = split_dir / "summary.json"
    train_snapshot = split_dir / "train_snapshot.txt"
    val_snapshot = split_dir / "val_snapshot.txt"
    if summary.exists() or train_snapshot.exists() or val_snapshot.exists():
        return summary if summary.exists() else None, train_snapshot if train_snapshot.exists() else None, val_snapshot if val_snapshot.exists() else None
    return None, None, None


def upload_file(local_path: Path, remote: str, drive_root_folder_id: str, remote_path: str) -> None:
    cmd = [
        "rclone",
        "copyto",
        "--drive-root-folder-id",
        drive_root_folder_id,
        str(local_path),
        f"{remote}{remote_path}",
    ]
    run_logged(cmd)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--drive-root-folder-id", required=True)
    parser.add_argument("--remote", default="gdrive:")
    parser.add_argument("--folder-name", default=None)
    parser.add_argument("--config-path", default=None)
    parser.add_argument("--launcher-log", default=None)
    parser.add_argument("--split-summary", default=None)
    parser.add_argument("--train-snapshot", default=None)
    parser.add_argument("--val-snapshot", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"run_dir not found: {run_dir}")

    folder_name = args.folder_name or infer_folder_name(run_dir)
    syncnet_dir = run_dir / "syncnet"
    required = {
        "syncnet_best_off_eval.pth": syncnet_dir / "syncnet_best_off_eval.pth",
        "syncnet_best_our_eval.pth": syncnet_dir / "syncnet_best_our_eval.pth",
        "syncnet_latest.pth": syncnet_dir / "syncnet_latest.pth",
    }
    for name, path in required.items():
        if not path.exists():
            raise FileNotFoundError(f"missing required checkpoint {name}: {path}")

    config_path = Path(args.config_path).resolve() if args.config_path else None
    if config_path is None:
        yaml_candidates = sorted(run_dir.glob("*.yaml"))
        if len(yaml_candidates) != 1:
            raise SystemExit(f"expected exactly one yaml in {run_dir}, found {len(yaml_candidates)}")
        config_path = yaml_candidates[0]

    launcher_log = Path(args.launcher_log).resolve() if args.launcher_log else (run_dir / "launcher.log")
    if not launcher_log.exists():
        raise FileNotFoundError(f"launcher_log not found: {launcher_log}")

    split_summary, train_snapshot, val_snapshot = infer_split_paths(run_dir)
    if args.split_summary:
        split_summary = Path(args.split_summary).resolve()
    if args.train_snapshot:
        train_snapshot = Path(args.train_snapshot).resolve()
    if args.val_snapshot:
        val_snapshot = Path(args.val_snapshot).resolve()

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    log_summary = parse_launcher_log(launcher_log)
    split_summary_json = None
    if split_summary and split_summary.exists():
        split_summary_json = json.loads(split_summary.read_text(encoding="utf-8"))

    with tempfile.TemporaryDirectory(prefix="syncnet_upload_bundle_") as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        launch_yaml = tmp_dir / "launch_config.yaml"
        shutil.copy2(config_path, launch_yaml)

        if split_summary and split_summary.exists():
            shutil.copy2(split_summary, tmp_dir / "split_summary.json")

        launch_summary = {
            "generated_at": timestamp(),
            "run_dir": str(run_dir),
            "folder_name": folder_name,
            "config_path": str(config_path),
            "launcher_log": str(launcher_log),
            "train_snapshot": str(train_snapshot) if train_snapshot and train_snapshot.exists() else None,
            "val_snapshot": str(val_snapshot) if val_snapshot and val_snapshot.exists() else None,
            "config": config,
            "split_summary": split_summary_json,
            "eval_summary": log_summary,
            "checkpoints": {name: name for name in required},
        }
        launch_json = tmp_dir / "launch_config.json"
        launch_json.write_text(json.dumps(launch_summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

        upload_items: list[tuple[Path, str]] = []
        for name, path in required.items():
            upload_items.append((path, name))
        upload_items.append((launch_yaml, "launch_config.yaml"))
        upload_items.append((launch_json, "launch_config.json"))
        if (tmp_dir / "split_summary.json").exists():
            upload_items.append((tmp_dir / "split_summary.json", "split_summary.json"))

        for local_path, filename in upload_items:
            remote_path = f"{folder_name}/{filename}"
            log(f"{'[dry-run] ' if args.dry_run else ''}uploading {local_path.name} -> {remote_path}")
            if not args.dry_run:
                upload_file(local_path, remote=args.remote, drive_root_folder_id=args.drive_root_folder_id, remote_path=remote_path)

    log(f"done folder={folder_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
