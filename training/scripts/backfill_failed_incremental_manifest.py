#!/usr/bin/env python3
"""
Backfill incremental processing failure manifests from pipeline logs.
"""

import argparse
import json
import os
import re
import time


FAIL_RE = re.compile(
    r"^\[(?P<prefix>[^\]]+)\]\s+"
    r"(?P<result>transcode_fail|process_fail|worker_fail):\s+"
    r"(?P<name>[^:]+):\s*(?P<message>.*)$"
)


def failed_manifest_path(processed_dir: str) -> str:
    return os.path.join(processed_dir, "_failed_samples.jsonl")


def load_existing_names(manifest_path: str) -> set[str]:
    names: set[str] = set()
    if not os.path.exists(manifest_path):
        return names
    with open(manifest_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            name = obj.get("name")
            if name:
                names.add(str(name))
    return names


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", required=True)
    parser.add_argument("--log-file", action="append", default=[], help="Pipeline log(s) to scan")
    parser.add_argument("--log-prefix", action="append", default=[], help="Only accept these prefixes")
    args = parser.parse_args()

    manifest_path = failed_manifest_path(args.processed_dir)
    os.makedirs(args.processed_dir, exist_ok=True)
    existing = load_existing_names(manifest_path)
    allowed_prefixes = set(args.log_prefix)
    added = 0
    matched = 0

    with open(manifest_path, "a") as out:
        for log_file in args.log_file:
            if not os.path.exists(log_file):
                print(f"[backfill_failures] missing log_file={log_file}", flush=True)
                continue
            with open(log_file) as f:
                for raw_line in f:
                    line = raw_line.rstrip("\n")
                    match = FAIL_RE.match(line)
                    if not match:
                        continue
                    prefix = match.group("prefix")
                    if allowed_prefixes and prefix not in allowed_prefixes:
                        continue
                    matched += 1
                    name = match.group("name")
                    if name in existing:
                        continue
                    payload = {
                        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "name": name,
                        "result": match.group("result"),
                        "message": match.group("message"),
                        "raw_path": None,
                        "source_log": log_file,
                        "source_prefix": prefix,
                    }
                    out.write(json.dumps(payload, ensure_ascii=False) + "\n")
                    existing.add(name)
                    added += 1

    print(
        f"[backfill_failures] processed_dir={args.processed_dir} "
        f"matched={matched} added={added} total_failed={len(existing)} "
        f"manifest={manifest_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
