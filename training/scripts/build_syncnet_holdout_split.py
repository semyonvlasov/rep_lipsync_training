#!/usr/bin/env python3
"""
Build a clean train/holdout split for SyncNet training from processed HDTF clips.

The split keeps only valid processed speaker dirs:
  - bbox.json exists
  - frames.npy exists
  - mel.npy exists
  - bad_sample is not set

Holdout is chosen as the newest clean clips according to raw video mtime when
available, otherwise bbox.json mtime.
"""

import argparse
import json
import os


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


def collect_clean_items(processed_roots, raw_dirs=None, speaker_allowlist=None):
    items = []
    name_to_root = {}
    duplicates = []
    invalid_meta = []
    for processed_root, raw_dir in zip(processed_roots, align_raw_dirs(processed_roots, raw_dirs or [])):
        for name in sorted(os.listdir(processed_root)):
            speaker_dir = os.path.join(processed_root, name)
            if not os.path.isdir(speaker_dir):
                continue
            if speaker_allowlist is not None and name not in speaker_allowlist:
                continue

            meta_path = os.path.join(speaker_dir, "bbox.json")
            frames_path = os.path.join(speaker_dir, "frames.npy")
            mel_path = os.path.join(speaker_dir, "mel.npy")
            if not (os.path.exists(meta_path) and os.path.exists(frames_path) and os.path.exists(mel_path)):
                continue

            try:
                with open(meta_path) as f:
                    meta = json.load(f)
            except Exception as exc:
                invalid_meta.append((meta_path, type(exc).__name__, str(exc)))
                continue
            if meta.get("bad_sample", False):
                continue

            if name in name_to_root:
                duplicates.append((name, name_to_root[name], processed_root))
                continue
            name_to_root[name] = processed_root

            raw_mtime = None
            raw_path = os.path.join(raw_dir, f"{name}.mp4") if raw_dir else None
            if raw_path and os.path.exists(raw_path):
                raw_mtime = os.path.getmtime(raw_path)

            items.append(
                {
                    "name": name,
                    "speaker_dir": speaker_dir,
                    "processed_root": processed_root,
                    "raw_dir": raw_dir,
                    "raw_mtime": raw_mtime,
                    "bbox_mtime": os.path.getmtime(meta_path),
                    "frames": int(meta.get("frames", -1)),
                    "fps": float(meta.get("fps", 0.0)),
                }
            )

    if duplicates:
        details = "\n".join(
            f"  - {name}: {root_a} vs {root_b}"
            for name, root_a, root_b in duplicates[:20]
        )
        raise RuntimeError(
            "Duplicate speaker names across processed roots would make the speaker allowlist ambiguous:\n"
            f"{details}"
        )
    if invalid_meta:
        print(f"skipped_invalid_meta={len(invalid_meta)}")
        for path, exc_name, detail in invalid_meta[:20]:
            print(f"invalid_meta: {path} ({exc_name}: {detail})")
    return items


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
    parser.add_argument("--speaker-list", default=None)
    parser.add_argument("--train-out", required=True)
    parser.add_argument("--holdout-out", required=True)
    parser.add_argument("--summary-out", required=True)
    args = parser.parse_args()

    speaker_allowlist = load_allowlist(args.speaker_list)
    items = collect_clean_items(args.processed_root, raw_dirs=args.raw_dir, speaker_allowlist=speaker_allowlist)
    if len(items) <= args.holdout_count:
        raise RuntimeError(
            f"Need more than holdout_count={args.holdout_count} clean processed clips, found {len(items)}"
        )

    items.sort(
        key=lambda item: (
            item["raw_mtime"] if item["raw_mtime"] is not None else item["bbox_mtime"],
            item["name"],
        )
    )

    holdout_items = items[-args.holdout_count :]
    train_items = items[: -args.holdout_count]

    train_names = [item["name"] for item in train_items]
    holdout_names = [item["name"] for item in holdout_items]

    write_list(args.train_out, train_names)
    write_list(args.holdout_out, holdout_names)

    summary = {
        "processed_roots": args.processed_root,
        "raw_dirs": args.raw_dir,
        "speaker_list": args.speaker_list,
        "clean_total": len(items),
        "train_count": len(train_names),
        "holdout_count": len(holdout_names),
        "holdout_names": holdout_names,
        "train_tail_names": train_names[-10:],
    }
    os.makedirs(os.path.dirname(args.summary_out), exist_ok=True)
    with open(args.summary_out, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"clean_total={len(items)}")
    print(f"train_count={len(train_names)}")
    print(f"holdout_count={len(holdout_names)}")
    print("holdout_names=" + ", ".join(holdout_names))
    print(f"saved_train={args.train_out}")
    print(f"saved_holdout={args.holdout_out}")
    print(f"saved_summary={args.summary_out}")


if __name__ == "__main__":
    main()
