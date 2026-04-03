#!/usr/bin/env python3
"""
Merge processed faceclip archives into the current dataset roots with dedupe.

The script imports canonical faceclip samples (`mp4 + json`) from processed
archives into the existing dataset roots under:

  <dataset_root>/_lazy_imports/<quality_tier>/<sample>.mp4
  <dataset_root>/_lazy_imports/<quality_tier>/<sample>.json

Dedupe is based on sample name. Existing clean materialized samples
(`frames.npy + mel.npy + bbox.json`) always win over new lazy imports.
Existing lazy imports also block re-imports of the same sample name.

This means we can incrementally widen the dataset without deleting the current
remote processed roots, while keeping one effective dataset per root.
"""

import argparse
import fnmatch
import json
import os
import shutil
import subprocess
import tarfile
import time
from pathlib import Path


TRAINING_ROOT = Path(__file__).resolve().parents[1]


def timestamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S %Z")


def log(message: str) -> None:
    print(f"{timestamp()} {message}", flush=True)


def resolve_training_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return TRAINING_ROOT / path


def append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def load_latest_state(path: Path) -> dict[str, dict]:
    latest = {}
    if not path.exists():
        return latest
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            archive_name = obj.get("source_archive")
            if archive_name:
                latest[str(archive_name)] = obj
    return latest


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


def rclone_lsf(remote: str, folder_id: str) -> list[str]:
    cmd = [
        "rclone",
        "lsf",
        "--files-only",
        "--drive-root-folder-id",
        folder_id,
        remote,
    ]
    proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return sorted([line.strip() for line in proc.stdout.splitlines() if line.strip()])


def run_logged(cmd: list[str], prefix: str) -> None:
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        line = line.rstrip()
        if line:
            log(f"{prefix} {line}")
    rc = process.wait()
    if rc != 0:
        raise subprocess.CalledProcessError(rc, cmd)


def archive_dataset_kind(name: str) -> str:
    lower = name.lower()
    if "hdtf" in lower:
        return "hdtf"
    if "talkvid" in lower:
        return "talkvid"
    return "talkvid"


def list_source_archives(args) -> list[str]:
    if args.source_dir:
        source_dir = resolve_training_path(args.source_dir)
        return sorted(
            [
                path.name
                for path in source_dir.iterdir()
                if path.is_file()
                and path.name.endswith(".tar")
                and fnmatch.fnmatch(path.name, args.archive_glob)
            ]
        )
    if not args.folder_id:
        raise SystemExit("Pass either --source-dir or --folder-id")
    return [
        name
        for name in rclone_lsf(args.remote, args.folder_id)
        if name.endswith(".tar") and fnmatch.fnmatch(name, args.archive_glob)
    ]


def ensure_archive_local(args, archive_name: str, download_dir: Path) -> tuple[Path, bool]:
    if args.source_dir:
        source_path = resolve_training_path(args.source_dir) / archive_name
        if not source_path.exists():
            raise FileNotFoundError(source_path)
        return source_path, False

    local_path = download_dir / archive_name
    if local_path.exists():
        return local_path, True

    log(f"[FaceclipMerge] downloading {archive_name}")
    run_logged(
        [
            "rclone",
            "copyto",
            "--drive-root-folder-id",
            args.folder_id,
            f"{args.remote}{archive_name}",
            str(local_path),
        ],
        prefix="[FaceclipMerge:rclone-download]",
    )
    return local_path, True


def extract_tar(tar_path: Path, extract_dir: Path) -> None:
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "r") as tar:
        try:
            tar.extractall(extract_dir, filter="data")
        except TypeError:
            tar.extractall(extract_dir)


def iter_faceclip_samples(extract_root: Path):
    for json_path in sorted(extract_root.rglob("*.json")):
        if json_path.name == "summary.json" or json_path.name.endswith(".detections.json"):
            continue
        mp4_path = json_path.with_suffix(".mp4")
        if not mp4_path.exists():
            continue
        detections_path = json_path.with_name(json_path.stem + ".detections.json")
        meta = load_json(json_path)
        name = str(meta.get("name") or json_path.stem)
        tier = str(meta.get("quality_tier") or json_path.parent.name or "unclassified")
        dataset_kind = str(meta.get("source_dataset") or "")
        if not dataset_kind:
            dataset_kind = archive_dataset_kind(extract_root.name)
        yield {
            "name": name,
            "tier": tier,
            "dataset_kind": dataset_kind,
            "meta": meta,
            "mp4_path": mp4_path,
            "json_path": json_path,
            "detections_path": detections_path if detections_path.exists() else None,
        }


def build_existing_name_index(dataset_root: Path, import_subdir: str) -> dict[str, int]:
    stats = {
        "processed": 0,
        "lazy": 0,
        "bad_processed": 0,
        "bad_lazy": 0,
    }
    names = set()

    if not dataset_root.exists():
        return {"names": names, "stats": stats}

    import_root = dataset_root / import_subdir

    for child in sorted(dataset_root.iterdir()):
        if not child.is_dir():
            continue
        if child.name in {import_subdir, "_lazy_cache"}:
            continue
        frames_path = child / "frames.npy"
        mel_path = child / "mel.npy"
        meta_path = child / "bbox.json"
        if not (frames_path.exists() and mel_path.exists() and meta_path.exists()):
            continue
        meta = load_json(meta_path)
        if meta.get("bad_sample", False):
            stats["bad_processed"] += 1
            continue
        names.add(child.name)
        stats["processed"] += 1

    if import_root.exists():
        for json_path in sorted(import_root.rglob("*.json")):
            if json_path.name == "summary.json" or json_path.name.endswith(".detections.json"):
                continue
            meta = load_json(json_path)
            if meta.get("bad_sample", False):
                stats["bad_lazy"] += 1
                continue
            names.add(str(meta.get("name") or json_path.stem))
            stats["lazy"] += 1

    return {"names": names, "stats": stats}


def cleanup_paths(paths: list[Path]) -> None:
    for path in paths:
        if not path.exists():
            continue
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        else:
            try:
                path.unlink()
            except OSError:
                pass


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder-id", default=None, help="Google Drive folder id with processed faceclip archives")
    parser.add_argument("--source-dir", default=None, help="Optional local directory with already downloaded faceclip archives")
    parser.add_argument("--remote", default="gdrive:")
    parser.add_argument("--archive-glob", default="*faceclips*.tar")
    parser.add_argument("--max-archives", type=int, default=0, help="0=all pending")
    parser.add_argument("--manifest-path", default="output/faceclip_merge/merge_manifest.jsonl")
    parser.add_argument("--download-dir", default="data/_imports/faceclip_merge_downloads")
    parser.add_argument("--extract-dir", default="data/_imports/faceclip_merge_extracted")
    parser.add_argument("--hdtf-root", default="data/hdtf/processed")
    parser.add_argument("--talkvid-root", default="data/talkvid/processed_medium")
    parser.add_argument("--import-subdir", default="_lazy_imports")
    parser.add_argument(
        "--include-tier",
        action="append",
        default=[],
        help="Only import matching quality tiers (repeatable). Default: import all tiers.",
    )
    parser.add_argument("--keep-downloads", action="store_true")
    parser.add_argument("--keep-extracted", action="store_true")
    args = parser.parse_args()

    manifest_path = resolve_training_path(args.manifest_path)
    download_dir = resolve_training_path(args.download_dir)
    extract_dir = resolve_training_path(args.extract_dir)
    hdtf_root = resolve_training_path(args.hdtf_root)
    talkvid_root = resolve_training_path(args.talkvid_root)
    include_tiers = {item.strip() for item in args.include_tier if item.strip()}

    download_dir.mkdir(parents=True, exist_ok=True)
    extract_dir.mkdir(parents=True, exist_ok=True)
    hdtf_root.mkdir(parents=True, exist_ok=True)
    talkvid_root.mkdir(parents=True, exist_ok=True)

    latest_state = load_latest_state(manifest_path)
    source_archives = list_source_archives(args)
    log(f"[FaceclipMerge] source_archives={len(source_archives)}")
    if include_tiers:
        log(f"[FaceclipMerge] include_tiers={sorted(include_tiers)}")

    existing_by_dataset = {
        "hdtf": build_existing_name_index(hdtf_root, args.import_subdir),
        "talkvid": build_existing_name_index(talkvid_root, args.import_subdir),
    }
    for dataset_kind, payload in existing_by_dataset.items():
        stats = payload["stats"]
        log(
            f"[FaceclipMerge] existing {dataset_kind}: usable={len(payload['names'])} "
            f"processed={stats['processed']} lazy={stats['lazy']} "
            f"bad_processed={stats['bad_processed']} bad_lazy={stats['bad_lazy']}"
        )

    merged_archives = 0
    for archive_name in source_archives:
        state = latest_state.get(archive_name, {})
        if str(state.get("stage") or "") == "merged_cleaned":
            continue
        if args.max_archives and merged_archives >= args.max_archives:
            break

        archive_stem = archive_name[:-4] if archive_name.endswith(".tar") else archive_name
        local_tar = None
        downloaded_here = False
        extract_root = extract_dir / archive_stem

        append_jsonl(
            manifest_path,
            {
                "ts": timestamp(),
                "source_archive": archive_name,
                "stage": "start",
            },
        )

        try:
            local_tar, downloaded_here = ensure_archive_local(args, archive_name, download_dir)
            append_jsonl(
                manifest_path,
                {
                    "ts": timestamp(),
                    "source_archive": archive_name,
                    "stage": "downloaded",
                    "local_tar": str(local_tar),
                },
            )

            log(f"[FaceclipMerge] extracting {archive_name}")
            extract_tar(local_tar, extract_root)

            counts = {
                "imported": 0,
                "skipped_duplicate": 0,
                "skipped_bad": 0,
                "skipped_tier": 0,
                "samples_total": 0,
            }
            datasets_seen = set()
            tiers_seen = set()

            for sample in iter_faceclip_samples(extract_root):
                counts["samples_total"] += 1
                dataset_kind = sample["dataset_kind"]
                datasets_seen.add(dataset_kind)
                tiers_seen.add(sample["tier"])

                if sample["meta"].get("bad_sample", False):
                    counts["skipped_bad"] += 1
                    continue
                if include_tiers and sample["tier"] not in include_tiers:
                    counts["skipped_tier"] += 1
                    continue

                if dataset_kind == "hdtf":
                    target_root = hdtf_root
                else:
                    target_root = talkvid_root

                existing_names = existing_by_dataset[dataset_kind]["names"]
                if sample["name"] in existing_names:
                    counts["skipped_duplicate"] += 1
                    continue

                dest_dir = target_root / args.import_subdir / sample["tier"]
                dest_dir.mkdir(parents=True, exist_ok=True)
                dest_mp4 = dest_dir / f"{sample['name']}.mp4"
                dest_json = dest_dir / f"{sample['name']}.json"
                dest_detections = dest_dir / f"{sample['name']}.detections.json"
                if dest_mp4.exists() or dest_json.exists():
                    counts["skipped_duplicate"] += 1
                    existing_names.add(sample["name"])
                    continue

                shutil.move(str(sample["mp4_path"]), str(dest_mp4))
                shutil.move(str(sample["json_path"]), str(dest_json))
                if sample["detections_path"] is not None and sample["detections_path"].exists():
                    shutil.move(str(sample["detections_path"]), str(dest_detections))
                existing_names.add(sample["name"])
                counts["imported"] += 1

            append_jsonl(
                manifest_path,
                {
                    "ts": timestamp(),
                    "source_archive": archive_name,
                    "stage": "merged",
                    "datasets_seen": sorted(datasets_seen),
                    "tiers_seen": sorted(tiers_seen),
                    **counts,
                },
            )

            log(
                f"[FaceclipMerge] merged {archive_name}: "
                f"imported={counts['imported']} dup={counts['skipped_duplicate']} "
                f"bad={counts['skipped_bad']} tier_skip={counts['skipped_tier']}"
            )

            cleanup = [extract_root]
            if downloaded_here and not args.keep_downloads and local_tar is not None:
                cleanup.append(local_tar)
            if not args.keep_extracted:
                cleanup_paths(cleanup)

            append_jsonl(
                manifest_path,
                {
                    "ts": timestamp(),
                    "source_archive": archive_name,
                    "stage": "merged_cleaned",
                    "download_kept": bool(args.keep_downloads or not downloaded_here),
                    "extract_kept": bool(args.keep_extracted),
                    **counts,
                },
            )
            merged_archives += 1
        except Exception as exc:
            append_jsonl(
                manifest_path,
                {
                    "ts": timestamp(),
                    "source_archive": archive_name,
                    "stage": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )
            raise

    log(f"[FaceclipMerge] merged_archives={merged_archives}")
    log(f"[FaceclipMerge] hdtf_root={hdtf_root}")
    log(f"[FaceclipMerge] talkvid_root={talkvid_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
