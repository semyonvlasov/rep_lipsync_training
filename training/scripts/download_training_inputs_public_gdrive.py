#!/usr/bin/env python3

"""
Download training inputs from a public Google Drive folder and stage them into
the local training/data layout.
"""

import argparse
import fnmatch
import os
import shutil
import sys
from pathlib import Path


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from scripts.download_preprocessed_public_gdrive import (  # noqa: E402
    download_folder,
    extract_archive,
    find_processed_roots,
    infer_dataset_key,
    looks_like_speaker_dir,
    resolve_training_path,
)


DEFAULT_TARGETS = {
    "hdtf_processed": "data/hdtf/processed",
    "hdtf_clips": "data/hdtf/clips",
    "hdtf_raw": "data/hdtf/raw",
    "talkvid_raw": "data/talkvid_remote/raw",
    "talkvid_processed": "data/talkvid/processed",
    "talkvid_processed_soft": "data/talkvid/processed_soft",
    "talkvid_processed_medium": "data/talkvid/processed_medium",
    "talkvid_processed_strict": "data/talkvid/processed_strict",
}

LEGACY_HDTF_MONOLITH = "hdtf_clips_20260321.tar"


def parse_target_overrides(values):
    overrides = {}
    for value in values or []:
        if "=" not in value:
            raise SystemExit(f"Invalid --target override '{value}', expected key=path")
        key, path = value.split("=", 1)
        key = key.strip()
        path = path.strip()
        if key not in DEFAULT_TARGETS:
            raise SystemExit(f"Unknown dataset key for --target: {key}")
        if not path:
            raise SystemExit(f"Empty path for --target: {value}")
        overrides[key] = path
    return overrides


def gather_source_roots(source_root, extract_root, include_globs=None, exclude_globs=None):
    source_entries = []
    batch_hdtf_archives_present = any(source_root.rglob("hdtf_clips_batch_*.tar"))
    for child in sorted(source_root.rglob("*")):
        if child.is_dir():
            continue
        if child.is_file():
            lower = child.name.lower()
            if lower.endswith(".tar") or lower.endswith(".tar.gz") or lower.endswith(".tgz") or lower.endswith(".zip"):
                if not archive_selected(child, include_globs, exclude_globs):
                    print(f"[stage] skipping archive by glob filter: {child.name}", flush=True)
                    continue
                if batch_hdtf_archives_present and child.name == LEGACY_HDTF_MONOLITH:
                    print(
                        f"[stage] skipping legacy HDTF monolith because batch archives are present: {child}",
                        flush=True,
                    )
                    continue
                rel_name = child.relative_to(source_root).as_posix().replace("/", "__")
                dest = extract_root / rel_name
                if dest.exists():
                    shutil.rmtree(dest)
                extract_archive(child, dest)
                source_entries.append((child, dest))
    source_entries.append((source_root, source_root))
    return source_entries


def map_processed_stage_key(processed_key):
    if processed_key == "hdtf":
        return "hdtf_processed"
    if processed_key == "talkvid":
        return "talkvid_processed"
    if processed_key == "talkvid_soft":
        return "talkvid_processed_soft"
    if processed_key == "talkvid_medium":
        return "talkvid_processed_medium"
    if processed_key == "talkvid_strict":
        return "talkvid_processed_strict"
    return None


def count_speakers(root):
    if not root.is_dir():
        return 0
    return sum(1 for child in root.iterdir() if looks_like_speaker_dir(child))


def move_speaker_dirs(src_root, dst_root, overwrite=False):
    dst_root.mkdir(parents=True, exist_ok=True)
    moved = 0
    skipped = 0
    for child in sorted(src_root.iterdir()):
        if not looks_like_speaker_dir(child):
            continue
        dst = dst_root / child.name
        if dst.exists():
            if overwrite:
                shutil.rmtree(dst)
            else:
                skipped += 1
                continue
        shutil.move(str(child), str(dst))
        moved += 1
    return moved, skipped


def find_media_roots(search_root):
    candidates = []
    seen = set()
    for dirpath, dirnames, filenames in os.walk(search_root):
        current = Path(dirpath)
        mp4s = [fname for fname in filenames if fname.endswith(".mp4")]
        if mp4s:
            resolved = current.resolve()
            if resolved not in seen:
                candidates.append(current)
                seen.add(resolved)
            dirnames[:] = []
    return candidates


def classify_media_stage_key(origin_path, media_root):
    probes = " ".join((origin_path.name, str(origin_path), media_root.name, str(media_root))).lower()
    mp4s = sorted(media_root.glob("*.mp4"))
    sidecar_count = sum(1 for path in mp4s if path.with_suffix(".json").is_file())

    if sidecar_count > 0:
        return "talkvid_raw"
    if "talkvid" in probes:
        return "talkvid_raw"
    if "hdtf" in probes and "raw" in probes:
        return "hdtf_raw"
    if "hdtf" in probes:
        return "hdtf_clips"
    return None


def move_media_files(src_root, dst_root, stage_key, overwrite=False):
    dst_root.mkdir(parents=True, exist_ok=True)
    moved = 0
    skipped = 0
    for mp4_path in sorted(src_root.glob("*.mp4")):
        dst_mp4 = dst_root / mp4_path.name
        if dst_mp4.exists():
            if overwrite:
                dst_mp4.unlink()
            else:
                skipped += 1
                continue
        shutil.move(str(mp4_path), str(dst_mp4))
        moved += 1

        if stage_key == "talkvid_raw":
            json_path = mp4_path.with_suffix(".json")
            if json_path.exists():
                dst_json = dst_root / json_path.name
                if dst_json.exists():
                    if overwrite:
                        dst_json.unlink()
                    else:
                        skipped += 1
                        continue
                shutil.move(str(json_path), str(dst_json))
                moved += 1
    return moved, skipped


def count_media_files(root):
    if not root.is_dir():
        return 0
    return sum(1 for _ in root.glob("*.mp4"))


def archive_selected(path, include_globs, exclude_globs):
    name = path.name
    if include_globs and not any(fnmatch.fnmatch(name, pattern) for pattern in include_globs):
        return False
    if exclude_globs and any(fnmatch.fnmatch(name, pattern) for pattern in exclude_globs):
        return False
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder-url", default=None, help="Public Google Drive folder URL")
    parser.add_argument("--source-dir", default=None, help="Local directory with already downloaded archives or folders")
    parser.add_argument("--download-dir", default="data/_imports/gdrive_downloads", help="Where to place downloaded files")
    parser.add_argument("--extract-dir", default="data/_imports/gdrive_extracted", help="Where to unpack archives")
    parser.add_argument("--python", default=sys.executable, help="Python executable used to invoke gdown")
    parser.add_argument("--default-talkvid-processed", default="talkvid_medium", choices=["talkvid", "talkvid_soft", "talkvid_medium", "talkvid_strict"])
    parser.add_argument("--target", action="append", default=[], help="Override target root mapping, e.g. talkvid_raw=data/talkvid_remote/raw")
    parser.add_argument(
        "--include-archive-glob",
        action="append",
        default=[],
        help="Only extract/stage matching archive basenames (supports shell globs). Repeatable.",
    )
    parser.add_argument(
        "--exclude-archive-glob",
        action="append",
        default=[],
        help="Skip matching archive basenames even if present (supports shell globs). Repeatable.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files in target roots")
    parser.add_argument("--keep-downloads", action="store_true", help="Keep downloaded archives after staging")
    parser.add_argument("--keep-extracted", action="store_true", help="Keep extracted temp directories after staging")
    args = parser.parse_args()

    if not args.folder_url and not args.source_dir:
        raise SystemExit("Pass either --folder-url or --source-dir")

    target_paths = dict(DEFAULT_TARGETS)
    target_paths.update(parse_target_overrides(args.target))
    targets = {key: resolve_training_path(path) for key, path in target_paths.items()}
    download_dir = resolve_training_path(args.download_dir)
    extract_dir = resolve_training_path(args.extract_dir)
    source_root = resolve_training_path(args.source_dir) if args.source_dir else download_dir

    if source_root.exists() and not args.source_dir and not args.keep_downloads:
        shutil.rmtree(source_root)
    if extract_dir.exists() and not args.keep_extracted:
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)

    include_globs = [item.strip() for item in args.include_archive_glob if item.strip()]
    exclude_globs = [item.strip() for item in args.exclude_archive_glob if item.strip()]

    if args.folder_url:
        download_folder(
            args.folder_url,
            download_dir,
            args.python,
            include_globs=include_globs,
            exclude_globs=exclude_globs,
        )

    if not source_root.is_dir():
        raise SystemExit(f"Source directory does not exist: {source_root}")

    seen_processed = set()
    seen_media = set()
    touched = set()
    moved_totals = {key: 0 for key in targets}
    skipped_totals = {key: 0 for key in targets}

    for origin_path, search_root in gather_source_roots(source_root, extract_dir, include_globs, exclude_globs):
        for processed_root in find_processed_roots(search_root):
            resolved = processed_root.resolve()
            if resolved in seen_processed:
                continue
            seen_processed.add(resolved)
            processed_key = infer_dataset_key(origin_path, processed_root, args.default_talkvid_processed)
            stage_key = map_processed_stage_key(processed_key)
            if not stage_key:
                continue
            moved, skipped = move_speaker_dirs(processed_root, targets[stage_key], overwrite=args.overwrite)
            moved_totals[stage_key] += moved
            skipped_totals[stage_key] += skipped
            touched.add(stage_key)
            print(f"[stage] {stage_key}: moved={moved} skipped={skipped} target={targets[stage_key]}", flush=True)

        for media_root in find_media_roots(search_root):
            resolved = media_root.resolve()
            if resolved in seen_media:
                continue
            seen_media.add(resolved)
            stage_key = classify_media_stage_key(origin_path, media_root)
            if not stage_key:
                continue
            moved, skipped = move_media_files(media_root, targets[stage_key], stage_key, overwrite=args.overwrite)
            moved_totals[stage_key] += moved
            skipped_totals[stage_key] += skipped
            touched.add(stage_key)
            print(f"[stage] {stage_key}: moved={moved} skipped={skipped} target={targets[stage_key]}", flush=True)

    for stage_key, target_root in targets.items():
        if stage_key not in touched:
            continue
        if stage_key.endswith("processed") or "_processed_" in stage_key:
            count = count_speakers(target_root)
        else:
            count = count_media_files(target_root)
        print(
            f"[stage] ready {stage_key}: count={count} moved={moved_totals[stage_key]} skipped={skipped_totals[stage_key]} root={target_root}",
            flush=True,
        )

    if args.folder_url and not args.keep_downloads and download_dir.exists():
        shutil.rmtree(download_dir)
    if not args.keep_extracted and extract_dir.exists():
        shutil.rmtree(extract_dir)


if __name__ == "__main__":
    main()
