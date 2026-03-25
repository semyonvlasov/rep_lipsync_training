#!/usr/bin/env python3
"""
Build a clean train/holdout split for SyncNet training from processed and/or
lazy faceclip roots.

Supported layouts:

1. Materialized processed samples:
   root/
     sample_name/
       frames.npy
       mel.npy
       bbox.json

2. Canonical lazy faceclips:
   root/
     confident/sample_name.mp4
     confident/sample_name.json
     medium/sample_name.mp4
     medium/sample_name.json
     ...

Holdout is chosen as the newest clean clips according to:
  1. raw video mtime when a matching --raw-dir is provided
  2. lazy mp4 mtime
  3. bbox/json mtime

Optional frame capping lets us keep only the newest or oldest train clips until
the cumulative frame count reaches the target budget.
"""

import argparse
import json
import os
from collections import Counter


def load_allowlist(path):
    if not path:
        return None
    with open(path) as f:
        return {line.strip() for line in f if line.strip()}


def align_raw_dirs(processed_roots, raw_dirs):
    if not raw_dirs:
        return [None] * len(processed_roots)
    if len(raw_dirs) != len(processed_roots):
        raise RuntimeError(
            f"Expected either 0 raw dirs or one raw dir per processed root, "
            f"got processed_roots={len(processed_roots)} raw_dirs={len(raw_dirs)}"
        )
    return raw_dirs


def load_meta(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def entry_priority(entry):
    return 1 if entry["type"] == "processed" else 0


def collect_clean_items(processed_roots, raw_dirs=None, speaker_allowlist=None):
    items_by_name = {}
    invalid_meta = []
    duplicates_skipped = []

    for processed_root, raw_dir in zip(processed_roots, align_raw_dirs(processed_roots, raw_dirs or [])):
        for name in sorted(os.listdir(processed_root)):
            speaker_dir = os.path.join(processed_root, name)
            if not os.path.isdir(speaker_dir):
                continue
            frames_path = os.path.join(speaker_dir, "frames.npy")
            mel_path = os.path.join(speaker_dir, "mel.npy")
            meta_path = os.path.join(speaker_dir, "bbox.json")
            if not (os.path.exists(frames_path) and os.path.exists(mel_path) and os.path.exists(meta_path)):
                continue
            if speaker_allowlist is not None and name not in speaker_allowlist:
                continue

            meta = load_meta(meta_path)
            if meta is None:
                invalid_meta.append((meta_path, "meta_decode_failed"))
                continue
            if meta.get("bad_sample", False):
                continue

            frame_count = int(meta.get("n_frames") or meta.get("frames") or 0)
            if frame_count <= 0:
                continue

            raw_path = os.path.join(raw_dir, f"{name}.mp4") if raw_dir else None
            raw_mtime = os.path.getmtime(raw_path) if raw_path and os.path.exists(raw_path) else None
            entry = {
                "name": name,
                "type": "processed",
                "speaker_dir": speaker_dir,
                "processed_root": processed_root,
                "raw_dir": raw_dir,
                "raw_mtime": raw_mtime,
                "source_mtime": raw_mtime if raw_mtime is not None else os.path.getmtime(meta_path),
                "frames": frame_count,
                "fps": float(meta.get("fps", 0.0) or 0.0),
                "meta_path": meta_path,
            }
            existing = items_by_name.get(name)
            if existing is None or entry_priority(entry) > entry_priority(existing):
                items_by_name[name] = entry
            else:
                duplicates_skipped.append((name, existing["type"], entry["type"]))

        for dirpath, dirnames, filenames in os.walk(processed_root):
            dirnames[:] = [d for d in dirnames if d not in {"__pycache__", "_lazy_cache"}]
            for filename in sorted(filenames):
                if not filename.endswith(".mp4"):
                    continue
                mp4_path = os.path.join(dirpath, filename)
                json_path = os.path.splitext(mp4_path)[0] + ".json"
                if not os.path.exists(json_path):
                    continue

                name = os.path.splitext(filename)[0]
                if speaker_allowlist is not None and name not in speaker_allowlist:
                    continue

                meta = load_meta(json_path)
                if meta is None:
                    invalid_meta.append((json_path, "meta_decode_failed"))
                    continue
                if meta.get("bad_sample", False):
                    continue

                frame_count = int(meta.get("n_frames") or meta.get("frames") or 0)
                if frame_count <= 0:
                    continue

                entry = {
                    "name": name,
                    "type": "lazy",
                    "speaker_dir": mp4_path,
                    "processed_root": processed_root,
                    "raw_dir": raw_dir,
                    "raw_mtime": None,
                    "source_mtime": os.path.getmtime(mp4_path),
                    "frames": frame_count,
                    "fps": float(meta.get("fps", 0.0) or 0.0),
                    "meta_path": json_path,
                }
                existing = items_by_name.get(name)
                if existing is None or entry_priority(entry) > entry_priority(existing):
                    items_by_name[name] = entry
                else:
                    duplicates_skipped.append((name, existing["type"], entry["type"]))

    if invalid_meta:
        print(f"skipped_invalid_meta={len(invalid_meta)}")
        for path, reason in invalid_meta[:20]:
            print(f"invalid_meta: {path} ({reason})")
    if duplicates_skipped:
        counts = Counter((old_t, new_t) for _, old_t, new_t in duplicates_skipped)
        print(f"duplicates_skipped={len(duplicates_skipped)} details={dict(counts)}")

    return sorted(items_by_name.values(), key=lambda item: item["name"])


def cap_train_items(train_items, max_train_frames, train_selection):
    if max_train_frames <= 0:
        return train_items

    if train_selection == "oldest":
        ordered = list(train_items)
        reverse_back = False
    else:
        ordered = list(reversed(train_items))
        reverse_back = True

    selected = []
    total_frames = 0
    for item in ordered:
        if selected and total_frames >= max_train_frames:
            break
        selected.append(item)
        total_frames += int(item["frames"])

    if reverse_back:
        selected.reverse()
    return selected


def write_list(path, names):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for name in names:
            f.write(name + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-root", action="append", required=True)
    parser.add_argument("--raw-dir", action="append", default=[])
    parser.add_argument("--holdout-count", type=int, default=10)
    parser.add_argument("--max-train-frames", type=int, default=0, help="Cap the train split by cumulative frames (0=disabled)")
    parser.add_argument("--train-selection", choices=["newest", "oldest"], default="newest")
    parser.add_argument("--speaker-list", default=None)
    parser.add_argument("--train-out", required=True)
    parser.add_argument("--holdout-out", required=True)
    parser.add_argument("--summary-out", required=True)
    args = parser.parse_args()

    speaker_allowlist = load_allowlist(args.speaker_list)
    items = collect_clean_items(args.processed_root, raw_dirs=args.raw_dir, speaker_allowlist=speaker_allowlist)
    if len(items) <= args.holdout_count:
        raise RuntimeError(
            f"Need more than holdout_count={args.holdout_count} clean clips, found {len(items)}"
        )

    items.sort(key=lambda item: (item["source_mtime"], item["name"]))

    holdout_items = items[-args.holdout_count :]
    train_candidates = items[: -args.holdout_count]
    train_items = cap_train_items(train_candidates, args.max_train_frames, args.train_selection)

    train_names = [item["name"] for item in train_items]
    holdout_names = [item["name"] for item in holdout_items]

    write_list(args.train_out, train_names)
    write_list(args.holdout_out, holdout_names)

    train_total_frames = sum(int(item["frames"]) for item in train_items)
    holdout_total_frames = sum(int(item["frames"]) for item in holdout_items)

    summary = {
        "processed_roots": args.processed_root,
        "raw_dirs": args.raw_dir,
        "speaker_list": args.speaker_list,
        "clean_total": len(items),
        "clean_total_frames": sum(int(item["frames"]) for item in items),
        "train_count": len(train_names),
        "train_total_frames": train_total_frames,
        "train_selection": args.train_selection,
        "max_train_frames": args.max_train_frames,
        "holdout_count": len(holdout_names),
        "holdout_total_frames": holdout_total_frames,
        "holdout_names": holdout_names,
        "train_tail_names": train_names[-10:],
        "type_breakdown": dict(Counter(item["type"] for item in items)),
    }
    os.makedirs(os.path.dirname(args.summary_out), exist_ok=True)
    with open(args.summary_out, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"clean_total={len(items)}")
    print(f"clean_total_frames={summary['clean_total_frames']}")
    print(f"train_count={len(train_names)}")
    print(f"train_total_frames={train_total_frames}")
    print(f"holdout_count={len(holdout_names)}")
    print(f"holdout_total_frames={holdout_total_frames}")
    print("holdout_names=" + ", ".join(holdout_names))
    print(f"saved_train={args.train_out}")
    print(f"saved_holdout={args.holdout_out}")
    print(f"saved_summary={args.summary_out}")


if __name__ == "__main__":
    main()
