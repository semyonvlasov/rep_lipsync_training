#!/usr/bin/env python3
"""
Refresh the faceclip remote monitor and destroy finished Vast instances.

Policy:
- never touch explicitly preserved remotes (default: remote_24_3090)
- destroy remotes whose status is `idle, last uploaded_cleaned (...)`
- if SSH is already unreachable, assume the instance was manually killed and skip
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
MONITOR_DIR = REPO_ROOT / "training" / "output" / "faceclip_remote_monitor"
DEFAULT_REGISTRY = MONITOR_DIR / "targets.json"
DEFAULT_CACHE = MONITOR_DIR / "status_cache.json"
DEFAULT_OUTPUT = MONITOR_DIR / "combined_status.log"
DEFAULT_LOG = MONITOR_DIR / "idle_reaper.log"
DEFAULT_MONITOR_SCRIPT = REPO_ROOT / "training" / "scripts" / "monitor_faceclip_remotes.py"


def timestamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S %Z")


def log(message: str, log_path: Path) -> None:
    line = f"{timestamp()} [idle-reaper] {message}"
    print(line, flush=True)
    with open(log_path, "a") as f:
        f.write(line + "\n")


def log_once(seen_messages: set[str], key: str, message: str, log_path: Path) -> None:
    if key in seen_messages:
        return
    seen_messages.add(key)
    log(message, log_path)


def run(cmd: list[str], timeout: int | None = None, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        check=check,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def load_registry(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path) as f:
        payload = json.load(f)
    return payload if isinstance(payload, list) else []


def parse_status_lines(path: Path) -> dict[str, str]:
    pattern = re.compile(
        r"^(?P<name>[^()]+?) \((?P<endpoint>[^)]+)\) \| (?P<status>.*?)(?: @ \d{2}:\d{2}:\d{2})?$"
    )
    statuses: dict[str, str] = {}
    if not path.exists():
        return statuses
    with open(path) as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            match = pattern.match(line)
            if not match:
                continue
            statuses[match.group("name").strip()] = match.group("status").strip()
    return statuses


def canonical_status(status_text: str) -> str:
    prefix = "no response, last status: "
    if status_text.startswith(prefix):
        return status_text[len(prefix) :].strip()
    return status_text.strip()


def should_destroy(status_text: str) -> bool:
    return canonical_status(status_text).startswith("idle, last uploaded_cleaned ")


def ssh_alive(entry: dict, timeout: int) -> bool:
    cmd = [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=5",
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "UserKnownHostsFile=/dev/null",
        "-p",
        str(entry["port"]),
        str(entry["remote"]),
        "exit 0",
    ]
    proc = run(cmd, timeout=timeout, check=False)
    return proc.returncode == 0


def refresh_monitor(args: argparse.Namespace, log_path: Path) -> None:
    cmd = [
        "python3",
        str(args.monitor_script),
        "refresh",
        "--registry",
        str(args.registry),
        "--cache",
        str(args.cache),
        "--output",
        str(args.output),
        "--ssh-timeout",
        str(args.monitor_ssh_timeout),
    ]
    try:
        run(cmd, timeout=args.refresh_timeout)
    except subprocess.TimeoutExpired:
        log(
            f"refresh timed out after {args.refresh_timeout}s; using current combined_status snapshot",
            log_path,
        )
    except subprocess.CalledProcessError as exc:
        log(f"refresh failed rc={exc.returncode}; using current combined_status snapshot", log_path)


def load_vast_label_map() -> dict[str, str]:
    proc = run(["vastai", "show", "instances"], timeout=30)
    pattern = re.compile(r"^\s*(\d+)\s+.*?\s+(faceclip-[^\s]+)\s+\S+\s+\S+\s*$", re.MULTILINE)
    mapping: dict[str, str] = {}
    for match in pattern.finditer(proc.stdout):
        mapping[match.group(2)] = match.group(1)
    return mapping


def unregister_entry(args: argparse.Namespace, entry: dict) -> None:
    cmd = [
        "python3",
        str(args.monitor_script),
        "unregister",
        "--registry",
        str(args.registry),
        "--remote",
        str(entry["remote"]),
        "--port",
        str(entry["port"]),
    ]
    run(cmd, timeout=15)


def maybe_destroy_entry(
    args: argparse.Namespace,
    entry: dict,
    status_text: str,
    label_map: dict[str, str],
    log_path: Path,
) -> None:
    name = str(entry.get("name", ""))
    if name in args.preserve_name:
        return
    if not should_destroy(status_text):
        return

    if not ssh_alive(entry, args.ssh_timeout):
        log_once(
            args.seen_skip_messages,
            f"{name}:ssh_unavailable",
            f"skip {name}: ssh unavailable; assuming already manually killed",
            log_path,
        )
        return

    vast_label = f"faceclip-{name}"
    instance_id = label_map.get(vast_label)
    if not instance_id:
        log_once(
            args.seen_skip_messages,
            f"{name}:missing_label",
            f"skip {name}: no active Vast instance found for label {vast_label}",
            log_path,
        )
        return

    if args.dry_run:
        log(f"dry-run destroy {name}: instance_id={instance_id} status='{status_text}'", log_path)
        return

    log(f"destroy {name}: instance_id={instance_id} status='{status_text}'", log_path)
    run(["vastai", "destroy", "instance", instance_id], timeout=60)
    unregister_entry(args, entry)
    log(f"destroyed and unregistered {name}: instance_id={instance_id}", log_path)


def run_once(args: argparse.Namespace) -> None:
    log_path = args.log_file
    refresh_monitor(args, log_path)
    statuses = parse_status_lines(args.output)
    registry = load_registry(args.registry)
    label_map = load_vast_label_map()

    for entry in registry:
        name = str(entry.get("name", ""))
        status_text = statuses.get(name)
        if not status_text:
            continue
        maybe_destroy_entry(args, entry, status_text, label_map, log_path)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--monitor-script", type=Path, default=DEFAULT_MONITOR_SCRIPT)
    parser.add_argument("--log-file", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--interval", type=int, default=60)
    parser.add_argument("--monitor-ssh-timeout", type=int, default=20)
    parser.add_argument("--refresh-timeout", type=int, default=300)
    parser.add_argument("--ssh-timeout", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--once", action="store_true")
    parser.add_argument(
        "--preserve-name",
        action="append",
        default=["remote_24_3090"],
        help="Remote names that must never be destroyed",
    )
    return parser


def main() -> int:
    args = build_argparser().parse_args()
    args.preserve_name = set(str(item) for item in args.preserve_name)
    args.seen_skip_messages = set()
    args.log_file.parent.mkdir(parents=True, exist_ok=True)

    if args.once:
        run_once(args)
        return 0

    log(
        "start "
        f"interval={args.interval}s preserve={sorted(args.preserve_name)} "
        f"registry={args.registry}",
        args.log_file,
    )
    while True:
        try:
            run_once(args)
        except Exception as exc:
            log(f"loop error: {exc}", args.log_file)
        time.sleep(max(5, args.interval))


if __name__ == "__main__":
    raise SystemExit(main())
