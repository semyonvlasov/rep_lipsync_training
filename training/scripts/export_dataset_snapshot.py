#!/usr/bin/env python3
"""Export a portable lazy dataset snapshot with cache and split metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import yaml

TRAINING_ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "lipsync.dataset_snapshot.v1"
DEFAULT_CONFIG = "configs/generator_mirror_gan_tiltaware_dataset_adaptive.yaml"
DEFAULT_PREPARED_DIR = "output/generator_mirror_gan_tiltaware_dataset_adaptive_prepared"
DEFAULT_SPLIT_DIR = "output/generator_mirror_gan_tiltaware_dataset_adaptive_split"


def timestamp_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def resolve_training_path(value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return TRAINING_ROOT / path


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise SystemExit(f"Config is not a mapping: {path}")
    return payload


def format_cache_value(value) -> str:
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return str(value).replace(".", "p")
    return str(value).replace(".", "p")


def frames_cache_name(cfg: dict) -> str:
    data_cfg = cfg.get("data", {})
    size = data_cfg.get("materialize_frames_size", cfg.get("model", {}).get("img_size"))
    if size is None:
        return "frames.npy"
    if isinstance(size, (list, tuple)):
        width, height = int(size[0]), int(size[1])
    else:
        width = height = int(size)
    if width == height:
        return f"frames_s{width}.npy"
    return f"frames_{width}x{height}.npy"


def mel_cache_name(cfg: dict) -> str:
    audio = cfg.get("audio") or {}
    if not audio:
        return "mel.npy"
    return (
        "mel"
        f"_sr{format_cache_value(audio['sample_rate'])}"
        f"_hop{format_cache_value(audio['hop_size'])}"
        f"_n{format_cache_value(audio['n_mels'])}"
        f"_fft{format_cache_value(audio['n_fft'])}"
        f"_win{format_cache_value(audio['win_size'])}"
        f"_fmin{format_cache_value(audio['fmin'])}"
        f"_fmax{format_cache_value(audio['fmax'])}"
        f"_pre{format_cache_value(audio['preemphasis'])}.npy"
    )


def relpath(path: Path) -> str:
    return str(path.resolve().relative_to(TRAINING_ROOT.resolve()))


def infer_dataset_kind(root: Path, meta: dict) -> str:
    value = str(meta.get("source_dataset") or "").strip().lower()
    if value:
        return value
    text = str(root).lower()
    if "hdtf" in text:
        return "hdtf"
    return "talkvid"


def lazy_cache_dir(root: Path, mp4_path: Path, name: str, cfg: dict) -> Path:
    cache_root = cfg.get("data", {}).get("lazy_cache_root")
    if cache_root:
        base_root = resolve_training_path(cache_root)
    else:
        base_root = root / "_lazy_cache"
    digest = hashlib.sha1(str(mp4_path.resolve()).encode("utf-8")).hexdigest()[:12]
    return base_root / f"{name}--{digest}"


def iter_samples(root: Path, import_subdir: str):
    import_root = root / import_subdir
    if not import_root.exists():
        return
    for json_path in sorted(import_root.rglob("*.json")):
        if json_path.name == "summary.json" or json_path.name.endswith(".detections.json"):
            continue
        mp4_path = json_path.with_suffix(".mp4")
        if not mp4_path.exists():
            continue
        meta = load_json(json_path)
        name = str(meta.get("name") or json_path.stem)
        tier = str(meta.get("quality_tier") or json_path.parent.name or "unclassified")
        yield {
            "name": name,
            "tier": tier,
            "dataset_kind": infer_dataset_kind(root, meta),
            "root": root,
            "mp4_path": mp4_path,
            "json_path": json_path,
            "detections_path": json_path.with_name(json_path.stem + ".detections.json"),
            "meta": meta,
        }


def archive_dataset_kind(name: str) -> str:
    lower = name.lower()
    if "hdtf" in lower:
        return "hdtf"
    if "talkvid" in lower:
        return "talkvid"
    return ""


def load_completed_archive_names(merge_manifest: Path) -> list[str]:
    if not merge_manifest.exists():
        return []
    latest: dict[str, str] = {}
    with merge_manifest.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            archive = item.get("source_archive")
            stage = item.get("stage")
            if archive and stage:
                latest[str(archive)] = str(stage)
    return sorted(
        archive
        for archive, stage in latest.items()
        if stage in {"merged", "merged_cleaned"}
    )


def recover_source_archive_index(archive_dirs: list[Path], archive_names: list[str]) -> dict[str, str]:
    wanted = set(archive_names)
    index: dict[str, str] = {}
    for archive_dir in archive_dirs:
        if not archive_dir or not archive_dir.exists():
            continue
        for tar_path in sorted(archive_dir.glob("*.tar")):
            if wanted and tar_path.name not in wanted:
                continue
            dataset_kind_from_archive = archive_dataset_kind(tar_path.name)
            try:
                with tarfile.open(tar_path, "r") as tar:
                    for member in tar:
                        if not member.isfile() or not member.name.endswith(".json"):
                            continue
                        name = Path(member.name).name
                        if name == "summary.json" or name.endswith(".detections.json"):
                            continue
                        handle = tar.extractfile(member)
                        if handle is None:
                            continue
                        try:
                            meta = json.loads(handle.read().decode("utf-8"))
                        except Exception:
                            continue
                        sample_name = str(meta.get("name") or Path(member.name).stem)
                        dataset_kind = str(meta.get("source_dataset") or dataset_kind_from_archive or "").lower()
                        if dataset_kind:
                            index[f"{dataset_kind}/{sample_name}"] = tar_path.name
                        index.setdefault(sample_name, tar_path.name)
            except tarfile.TarError:
                continue
    return index


def sync_status(meta: dict) -> str:
    if meta.get("bad_sample"):
        return "bad_sample"
    sync = meta.get("sync_alignment")
    if not isinstance(sync, dict):
        return "missing"
    status = str(sync.get("status") or "missing")
    if status == "aligned" and "audio_shift_mel_ticks" not in sync:
        return "missing"
    return status


def read_names(path: Path | None) -> set[str]:
    if not path or not path.exists():
        return set()
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def add_file(files: dict[Path, str], path: Path) -> None:
    if path.exists() and path.is_file():
        files[path.resolve()] = relpath(path)


def copy_optional_file(source: Path | None, target: Path) -> bool:
    if not source or not source.exists():
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return True


def copy_optional_dir(source: Path | None, target: Path) -> bool:
    if not source or not source.exists():
        return False
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(source, target)
    return True


def run_cmd(cmd: list[str], *, dry_run: bool = False) -> None:
    printable = " ".join(str(part) for part in cmd)
    if dry_run:
        print(f"[dataset-snapshot] dry-run: {printable}", flush=True)
        return
    print(f"[dataset-snapshot] {printable}", flush=True)
    subprocess.run(cmd, check=True)


def create_hf_repo(repo_id: str, *, private: bool, dry_run: bool = False) -> None:
    cmd = ["hf", "repos", "create", repo_id, "--type", "dataset", "--exist-ok"]
    if private:
        cmd.append("--private")
    run_cmd(cmd, dry_run=dry_run)


def create_hf_branch(repo_id: str, revision: str, *, dry_run: bool = False) -> None:
    run_cmd(
        ["hf", "repos", "branch", "create", repo_id, revision, "--type", "dataset", "--exist-ok"],
        dry_run=dry_run,
    )


def check_hf_auth(*, dry_run: bool = False) -> None:
    run_cmd(["hf", "auth", "whoami"], dry_run=dry_run)


def upload_hf_file(
    repo_id: str,
    local_path: Path,
    repo_path: str,
    *,
    revision: str | None = None,
    dry_run: bool = False,
) -> None:
    cmd = [
        "hf",
        "upload",
        repo_id,
        str(local_path),
        repo_path,
        "--type",
        "dataset",
    ]
    if revision:
        cmd.extend(["--revision", revision])
    run_cmd(cmd, dry_run=dry_run)


def upload_snapshot_metadata(
    repo_id: str,
    snapshot_dir: Path,
    *,
    revision: str | None = None,
    dry_run: bool = False,
) -> list[str]:
    uploaded = []
    for path in sorted(snapshot_dir.rglob("*")):
        if not path.is_file():
            continue
        repo_path = path.relative_to(snapshot_dir)
        if "shards" in repo_path.parts:
            continue
        upload_hf_file(repo_id, path, str(repo_path), revision=revision, dry_run=dry_run)
        uploaded.append(str(repo_path))
    return uploaded


def write_shards(
    files: dict[Path, str],
    snapshot_dir: Path,
    shard_size_bytes: int,
    *,
    hf_repo_id: str | None = None,
    hf_revision: str | None = None,
    delete_uploaded_shards: bool = False,
) -> list[dict]:
    shard_dir = snapshot_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    shards = []
    shard_idx = 0
    current_size = 0
    current_count = 0
    tar = None
    tar_path = None

    def close_current():
        nonlocal tar, tar_path, current_size, current_count
        if tar is None or tar_path is None:
            return
        tar.close()
        repo_path = str(tar_path.relative_to(snapshot_dir))
        size_bytes = tar_path.stat().st_size
        uploaded = False
        local_deleted = False
        if hf_repo_id:
            upload_hf_file(hf_repo_id, tar_path, repo_path, revision=hf_revision)
            uploaded = True
            if delete_uploaded_shards:
                tar_path.unlink()
                local_deleted = True
        shards.append(
            {
                "path": repo_path,
                "repo_path": repo_path,
                "size_bytes": size_bytes,
                "file_count": current_count,
                "payload_bytes": current_size,
                "uploaded": uploaded,
                "local_deleted": local_deleted,
            }
        )
        tar = None
        tar_path = None
        current_size = 0
        current_count = 0

    for source, arcname in sorted(files.items(), key=lambda item: item[1]):
        size = source.stat().st_size
        if tar is not None and current_count > 0 and current_size + size > shard_size_bytes:
            close_current()
        if tar is None:
            tar_path = shard_dir / f"dataset_{shard_idx:05d}.tar"
            shard_idx += 1
            tar = tarfile.open(tar_path, "w")
        tar.add(source, arcname=arcname)
        current_size += size
        current_count += 1
    close_current()
    return shards


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--snapshot-dir", default=None)
    parser.add_argument("--hdtf-root", default=None)
    parser.add_argument("--talkvid-root", default=None)
    parser.add_argument("--import-subdir", default="_lazy_imports")
    parser.add_argument("--merge-manifest", default="output/faceclip_merge/merge_manifest.jsonl")
    parser.add_argument("--sync-alignment-registry", default="output/sync_alignment/sync_alignment_manifest.jsonl")
    parser.add_argument("--prepared-dir", default=DEFAULT_PREPARED_DIR)
    parser.add_argument("--split-dir", default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--source-archive-dir", action="append", default=[])
    parser.add_argument("--include-cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-lazy-imports", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--shard-size-gb", type=float, default=20.0)
    parser.add_argument("--hf-repo-id", default=None, help="Optional Hugging Face dataset repo id for streaming shard upload")
    parser.add_argument("--hf-create-repo", action="store_true")
    parser.add_argument("--hf-revision", default=None, help="Optional HF branch/revision to upload this snapshot into")
    parser.add_argument("--hf-create-branch", action="store_true")
    parser.add_argument("--hf-private", action="store_true")
    parser.add_argument(
        "--delete-uploaded-shards",
        action="store_true",
        help="Delete each local shard after successful HF upload. Requires --hf-repo-id.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.delete_uploaded_shards and not args.hf_repo_id:
        raise SystemExit("--delete-uploaded-shards requires --hf-repo-id")
    if args.hf_create_repo and not args.hf_repo_id:
        raise SystemExit("--hf-create-repo requires --hf-repo-id")
    if args.hf_revision and not args.hf_repo_id:
        raise SystemExit("--hf-revision requires --hf-repo-id")
    if args.hf_create_branch and not args.hf_repo_id:
        raise SystemExit("--hf-create-branch requires --hf-repo-id")
    if args.hf_create_branch and not args.hf_revision:
        raise SystemExit("--hf-create-branch requires --hf-revision")

    cfg = load_config(resolve_training_path(args.config))
    data_cfg = cfg.get("data", {})
    roots = {
        "hdtf": resolve_training_path(args.hdtf_root or data_cfg.get("hdtf_root")),
        "talkvid": resolve_training_path(args.talkvid_root or data_cfg.get("talkvid_root")),
    }
    snapshot_dir = resolve_training_path(args.snapshot_dir) if args.snapshot_dir else (
        TRAINING_ROOT / "output" / "dataset_snapshots" / f"dataset_snapshot_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    merge_manifest = resolve_training_path(args.merge_manifest)
    sync_registry = resolve_training_path(args.sync_alignment_registry)
    prepared_dir = resolve_training_path(args.prepared_dir)
    split_dir = resolve_training_path(args.split_dir)
    train_names = read_names(split_dir / "train_snapshot.txt" if split_dir else None)
    val_names = read_names(split_dir / "val_snapshot.txt" if split_dir else None)
    archive_dirs = [resolve_training_path(item) for item in args.source_archive_dir]
    default_download_dir = TRAINING_ROOT / "data" / "_imports" / "faceclip_merge_downloads"
    if default_download_dir.exists():
        archive_dirs.append(default_download_dir)
    archive_index = recover_source_archive_index(
        [path for path in archive_dirs if path],
        load_completed_archive_names(merge_manifest),
    )

    files: dict[Path, str] = {}
    counts = {
        "samples": 0,
        "aligned": 0,
        "failed": 0,
        "missing": 0,
        "bad_sample": 0,
        "cache_ready": 0,
        "source_archive_known": 0,
        "source_archive_missing": 0,
    }
    sample_index_path = snapshot_dir / "sample_index.jsonl"
    cache_index_path = snapshot_dir / "cache_index.jsonl"
    if not args.dry_run:
        if snapshot_dir.exists():
            shutil.rmtree(snapshot_dir)
        snapshot_dir.mkdir(parents=True, exist_ok=True)
    if args.hf_repo_id:
        check_hf_auth(dry_run=args.dry_run)
        if args.hf_create_repo:
            create_hf_repo(args.hf_repo_id, private=args.hf_private, dry_run=args.dry_run)
        if args.hf_create_branch:
            create_hf_branch(args.hf_repo_id, args.hf_revision, dry_run=args.dry_run)

    frame_name = frames_cache_name(cfg)
    mel_name = mel_cache_name(cfg)

    for dataset_kind, root in roots.items():
        if root is None or not root.exists():
            continue
        for sample in iter_samples(root, args.import_subdir):
            counts["samples"] += 1
            meta = sample["meta"]
            status = sync_status(meta)
            counts[status if status in counts else "missing"] += 1
            key = f"{sample['dataset_kind']}/{sample['name']}"
            source_archive = meta.get("source_archive") or archive_index.get(key) or archive_index.get(sample["name"])
            if source_archive:
                counts["source_archive_known"] += 1
            else:
                counts["source_archive_missing"] += 1

            cache_dir = lazy_cache_dir(root, sample["mp4_path"], sample["name"], cfg)
            frames_path = cache_dir / frame_name
            mel_path = cache_dir / mel_name
            cache_ready = frames_path.exists() and mel_path.exists()
            if cache_ready:
                counts["cache_ready"] += 1

            if args.include_lazy_imports:
                add_file(files, sample["mp4_path"])
                add_file(files, sample["json_path"])
                add_file(files, sample["detections_path"])
            if args.include_cache and cache_dir.exists():
                for path in sorted(cache_dir.rglob("*")):
                    add_file(files, path)

            sample_record = {
                "name": sample["name"],
                "dataset_kind": sample["dataset_kind"],
                "tier": sample["tier"],
                "source_archive": source_archive,
                "mp4_path": relpath(sample["mp4_path"]),
                "json_path": relpath(sample["json_path"]),
                "detections_path": relpath(sample["detections_path"]) if sample["detections_path"].exists() else None,
                "sync_status": status,
                "bad_sample": bool(meta.get("bad_sample")),
                "cache_ready": cache_ready,
                "in_train": sample["name"] in train_names,
                "in_val": sample["name"] in val_names,
            }
            cache_record = {
                "name": sample["name"],
                "dataset_kind": sample["dataset_kind"],
                "cache_dir": relpath(cache_dir),
                "frames_path": relpath(frames_path),
                "mel_path": relpath(mel_path),
                "frames_exists": frames_path.exists(),
                "mel_exists": mel_path.exists(),
                "cache_key_source_path": str(sample["mp4_path"].resolve()),
            }
            if not args.dry_run:
                append_jsonl(sample_index_path, sample_record)
                append_jsonl(cache_index_path, cache_record)

    copied = {}
    if not args.dry_run:
        copied["merge_manifest"] = copy_optional_file(merge_manifest, snapshot_dir / "merge_manifest.jsonl")
        copied["sync_alignment_registry"] = copy_optional_file(sync_registry, snapshot_dir / "sync_alignment_manifest.jsonl")
        copied["prepared_dir"] = copy_optional_dir(prepared_dir, snapshot_dir / "prepared")
        copied["split_dir"] = copy_optional_dir(split_dir, snapshot_dir / "split")
        shards = write_shards(
            files,
            snapshot_dir,
            shard_size_bytes=max(1, int(args.shard_size_gb * 1024 * 1024 * 1024)),
            hf_repo_id=args.hf_repo_id,
            hf_revision=args.hf_revision,
            delete_uploaded_shards=args.delete_uploaded_shards,
        )
    else:
        copied = {}
        shards = []

    manifest = {
        "schema": SCHEMA,
        "created_at": timestamp_utc(),
        "config": args.config,
        "paths": {
            "merge_manifest": args.merge_manifest,
            "sync_alignment_registry": args.sync_alignment_registry,
            "prepared_dir": args.prepared_dir,
            "split_dir": args.split_dir,
            "hdtf_root": str(roots["hdtf"].relative_to(TRAINING_ROOT)) if roots["hdtf"] else None,
            "talkvid_root": str(roots["talkvid"].relative_to(TRAINING_ROOT)) if roots["talkvid"] else None,
            "import_subdir": args.import_subdir,
        },
        "cache": {
            "frames_name": frame_name,
            "mel_name": mel_name,
            "path_sensitive": True,
            "expected_training_root": str(TRAINING_ROOT),
        },
        "counts": counts,
        "copied": copied,
        "shards": shards,
        "hf_upload": {
            "repo_id": args.hf_repo_id,
            "revision": args.hf_revision,
            "streamed_shards": bool(args.hf_repo_id),
            "delete_uploaded_shards": bool(args.delete_uploaded_shards),
        },
    }
    if not args.dry_run:
        write_json(snapshot_dir / "dataset_snapshot_manifest.json", manifest)
        if args.hf_repo_id:
            manifest["hf_upload"]["metadata_files"] = upload_snapshot_metadata(
                args.hf_repo_id,
                snapshot_dir,
                revision=args.hf_revision,
            )
            write_json(snapshot_dir / "dataset_snapshot_manifest.json", manifest)
            upload_hf_file(
                args.hf_repo_id,
                snapshot_dir / "dataset_snapshot_manifest.json",
                "dataset_snapshot_manifest.json",
                revision=args.hf_revision,
            )
    print(json.dumps({"snapshot_dir": str(snapshot_dir), **manifest}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
