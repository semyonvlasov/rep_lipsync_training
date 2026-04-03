#!/usr/bin/env python3
"""
Maintain a local registry of remote faceclip-processing machines and render a
combined one-line-per-machine status snapshot for `watch`.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import time
from pathlib import Path


DEFAULT_MANIFEST_REL = "training/output/raw_faceclips_x288_cycle/archive_manifest.jsonl"
DEFAULT_CYCLE_LOG_REL = "training/output/raw_faceclips_x288_cycle/cycle.log"
DEFAULT_CYCLE_PID_REL = "training/output/raw_faceclips_x288_cycle/cycle.pid"


def timestamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S %Z")


def load_json(path: Path, default):
    if not path.exists():
        return default
    try:
        with open(path) as f:
            payload = json.load(f)
    except Exception:
        return default
    return payload


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def load_registry(path: Path) -> list[dict]:
    payload = load_json(path, [])
    return payload if isinstance(payload, list) else []


def upsert_registry_entry(registry: list[dict], entry: dict) -> list[dict]:
    entry_id = entry["id"]
    updated = [item for item in registry if item.get("id") != entry_id]
    updated.append(entry)
    updated.sort(key=lambda item: (str(item.get("name", "")), str(item.get("remote", "")), int(item.get("port", 0))))
    return updated


def remove_registry_entry(registry: list[dict], entry_id: str) -> list[dict]:
    return [item for item in registry if item.get("id") != entry_id]


def entry_id(remote: str, port: int) -> str:
    return f"{remote}:{int(port)}"


def entry_label(entry: dict) -> str:
    name = str(entry.get("name", "remote") or "remote")
    remote = str(entry["remote"])
    port = int(entry["port"])
    return f"{name} ({remote}:{port})"


def remote_abs_path(remote_root: str, rel_path: str) -> str:
    root = remote_root.rstrip("/")
    rel = rel_path.lstrip("/")
    return f"{root}/{rel}"


def build_remote_status_command(entry: dict) -> str:
    remote_root = str(entry["remote_root"])
    remote_python = str(entry.get("remote_python", "python3"))
    script_path = remote_abs_path(remote_root, "training/scripts/faceclip_cycle_status.py")
    manifest_path = remote_abs_path(remote_root, str(entry.get("manifest_rel", DEFAULT_MANIFEST_REL)))
    cycle_log_path = remote_abs_path(remote_root, str(entry.get("cycle_log_rel", DEFAULT_CYCLE_LOG_REL)))
    cycle_pid_path = remote_abs_path(remote_root, str(entry.get("cycle_pid_rel", DEFAULT_CYCLE_PID_REL)))
    parts = [
        shlex.quote(remote_python),
        shlex.quote(script_path),
        "--manifest-path",
        shlex.quote(manifest_path),
        "--cycle-log",
        shlex.quote(cycle_log_path),
        "--cycle-pid-path",
        shlex.quote(cycle_pid_path),
    ]
    return " ".join(parts)


def fetch_remote_status(entry: dict, timeout: int) -> dict:
    ssh_cmd = (
        "ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "
        f"-p {shlex.quote(str(entry['port']))} "
        f"{shlex.quote(str(entry['remote']))} "
        f"{shlex.quote(build_remote_status_command(entry))}"
    )
    proc = subprocess.run(
        ["/bin/sh", "-lc", ssh_cmd],
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    json_line = "{}"
    for line in reversed(proc.stdout.splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            json_line = line
            break
    payload = json.loads(json_line)
    if not isinstance(payload, dict):
        raise RuntimeError("remote_status_not_dict")
    return payload


def render_status_text(payload: dict) -> str:
    status = str(payload.get("status", "unknown"))
    archive_name = payload.get("archive_name")
    progress_done = payload.get("progress_done")
    progress_total = payload.get("progress_total")
    last_stage = payload.get("last_stage")

    if status == "download":
        return f"download ({archive_name})"
    if status == "process":
        if isinstance(progress_done, int) and isinstance(progress_total, int) and progress_total > 0:
            return f"process ({archive_name}) ({progress_done}/{progress_total})"
        return f"process ({archive_name})"
    if status == "upload":
        return f"upload ({archive_name})"
    if status == "failed":
        return f"failed ({archive_name})"
    if status == "idle":
        if last_stage and archive_name:
            return f"idle, last {last_stage} ({archive_name})"
        return "idle"
    return f"unknown ({archive_name or '<none>'})"


def render_line(entry: dict, payload: dict | None, cache_entry: dict | None) -> str:
    label = entry_label(entry)
    if payload is not None:
        status_text = render_status_text(payload)
        return f"{label} | {status_text}"

    if isinstance(cache_entry, dict) and cache_entry.get("status_text"):
        last_seen = cache_entry.get("last_seen_ts")
        suffix = f" @ {last_seen}" if last_seen else ""
        return f"{label} | no response, last status: {cache_entry['status_text']}{suffix}"
    return f"{label} | no response"


def do_register(args) -> int:
    registry_path = Path(args.registry)
    registry = load_registry(registry_path)
    entry = {
        "id": entry_id(args.remote, args.port),
        "name": args.name,
        "remote": args.remote,
        "port": int(args.port),
        "remote_root": args.remote_root,
        "remote_python": args.remote_python,
        "manifest_rel": args.manifest_rel,
        "cycle_log_rel": args.cycle_log_rel,
        "cycle_pid_rel": args.cycle_pid_rel,
        "registered_ts": timestamp(),
    }
    registry = upsert_registry_entry(registry, entry)
    write_json(registry_path, registry)
    print(f"registered {entry_label(entry)}")
    return 0


def do_unregister(args) -> int:
    registry_path = Path(args.registry)
    registry = load_registry(registry_path)
    registry = remove_registry_entry(registry, entry_id(args.remote, args.port))
    write_json(registry_path, registry)
    print(f"unregistered {args.remote}:{int(args.port)}")
    return 0


def do_refresh(args) -> int:
    registry_path = Path(args.registry)
    cache_path = Path(args.cache)
    output_path = Path(args.output)

    registry = load_registry(registry_path)
    cache = load_json(cache_path, {})
    if not isinstance(cache, dict):
        cache = {}

    lines = []
    next_cache = dict(cache)

    for entry in registry:
        cache_key = str(entry["id"])
        payload = None
        try:
            payload = fetch_remote_status(entry, args.ssh_timeout)
            next_cache[cache_key] = {
                "status_text": render_status_text(payload),
                "last_seen_ts": timestamp(),
                "payload": payload,
            }
        except Exception:
            payload = None
        lines.append(render_line(entry, payload, next_cache.get(cache_key)))

    if not lines:
        lines = ["no registered faceclip remotes"]

    write_json(cache_path, next_cache)
    write_text(output_path, "\n".join(lines) + "\n")

    if args.print_output:
        print("\n".join(lines))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    register = subparsers.add_parser("register")
    register.add_argument("--registry", required=True)
    register.add_argument("--name", default="remote")
    register.add_argument("--remote", required=True)
    register.add_argument("--port", required=True, type=int)
    register.add_argument("--remote-root", required=True)
    register.add_argument("--remote-python", default="python3")
    register.add_argument("--manifest-rel", default=DEFAULT_MANIFEST_REL)
    register.add_argument("--cycle-log-rel", default=DEFAULT_CYCLE_LOG_REL)
    register.add_argument("--cycle-pid-rel", default=DEFAULT_CYCLE_PID_REL)

    unregister = subparsers.add_parser("unregister")
    unregister.add_argument("--registry", required=True)
    unregister.add_argument("--remote", required=True)
    unregister.add_argument("--port", required=True, type=int)

    refresh = subparsers.add_parser("refresh")
    refresh.add_argument("--registry", required=True)
    refresh.add_argument("--cache", required=True)
    refresh.add_argument("--output", required=True)
    refresh.add_argument("--ssh-timeout", type=int, default=8)
    refresh.add_argument("--print-output", action="store_true")

    args = parser.parse_args()
    if args.cmd == "register":
        return do_register(args)
    if args.cmd == "unregister":
        return do_unregister(args)
    if args.cmd == "refresh":
        return do_refresh(args)
    raise ValueError(f"unsupported cmd: {args.cmd}")


if __name__ == "__main__":
    raise SystemExit(main())
