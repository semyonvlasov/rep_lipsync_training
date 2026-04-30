#!/usr/bin/env python3
"""Runtime CLI for Docker/Vast lip-sync training deployments."""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as exc:  # pragma: no cover - exercised only on incomplete hosts.
    raise SystemExit("PyYAML is required. Install training/requirements-server.txt first.") from exc


REPO_ROOT = Path(__file__).resolve().parents[2]
TRAINING_ROOT = REPO_ROOT / "training"
DEFAULT_ARTIFACT_CONFIG = TRAINING_ROOT / "configs" / "runtime_artifacts.yaml"
DEFAULT_RCLONE_CONFIG = Path("/run/secrets/rclone.conf")
DEFAULT_DATASET_FOLDER_ID = "1D6vtNpRmZabqnlW4598X6lqgFETZ9RvO"
DEFAULT_SYNC_ALIGNMENT_REGISTRY_PATH = "output/sync_alignment/sync_alignment_manifest.jsonl"
DEFAULT_SYNC_ALIGNMENT_REMOTE_NAME = "sync_alignment_manifest.jsonl"


def log(message: str) -> None:
    print(f"[lipsyncctl] {message}", flush=True)


def run(cmd: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None, dry_run: bool = False) -> None:
    printable = " ".join(str(part) for part in cmd)
    if cwd is not None:
        printable = f"(cd {cwd} && {printable})"
    if dry_run:
        log(f"dry-run: {printable}")
        return
    log(printable)
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, env=env, check=True)


def capture(cmd: list[str]) -> str:
    try:
        proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        return f"unavailable: {exc}"
    return proc.stdout.strip()


def load_artifact_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict):
        raise ValueError(f"artifact config is not a mapping: {path}")
    return cfg


def repo_path(repo_root: Path, rel_or_abs: str) -> Path:
    path = Path(rel_or_abs)
    if path.is_absolute():
        return path
    return repo_root / path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file(path: Path, expected_sha256: str | None, expected_size: int | None) -> bool:
    if not path.exists():
        log(f"missing: {path}")
        return False
    ok = True
    if expected_size is not None:
        actual_size = path.stat().st_size
        if actual_size != int(expected_size):
            log(f"size mismatch: {path} actual={actual_size} expected={expected_size}")
            ok = False
    if expected_sha256:
        actual_sha256 = sha256_file(path)
        if actual_sha256 != expected_sha256:
            log(f"sha256 mismatch: {path} actual={actual_sha256} expected={expected_sha256}")
            ok = False
    if ok:
        log(f"verified: {path}")
    return ok


def iter_artifacts(cfg: dict[str, Any], *, include_optional: bool) -> list[tuple[str, str, dict[str, Any]]]:
    result: list[tuple[str, str, dict[str, Any]]] = []
    for group_name in ("official_checkpoints", "current_best_checkpoints"):
        group = cfg.get(group_name) or {}
        if not isinstance(group, dict):
            continue
        for name, item in group.items():
            if not isinstance(item, dict):
                continue
            if item.get("prepare_default") is False and not include_optional:
                continue
            result.append((group_name, str(name), item))
    return result


def selected_artifacts(
    cfg: dict[str, Any],
    *,
    include_optional: bool,
    only: list[str] | None,
) -> list[tuple[str, str, dict[str, Any]]]:
    artifacts = iter_artifacts(cfg, include_optional=include_optional)
    if not only:
        return artifacts
    wanted = set(only)
    selected = []
    for group_name, name, item in artifacts:
        if name in wanted or f"{group_name}.{name}" in wanted:
            selected.append((group_name, name, item))
    missing = wanted - {name for _, name, _ in selected} - {f"{group}.{name}" for group, name, _ in selected}
    if missing:
        raise SystemExit(f"unknown artifact(s): {', '.join(sorted(missing))}")
    return selected


def runtime_env(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    rclone_config = Path(args.rclone_config)
    if rclone_config.exists():
        env["RCLONE_CONFIG"] = str(rclone_config)
    return env


def install_bundled_assets(cfg: dict[str, Any], repo_root: Path, *, dry_run: bool = False) -> None:
    bundled = cfg.get("bundled_assets") or {}
    copies: list[tuple[str, str]] = []
    audio = bundled.get("benchmark_audio") or {}
    if audio.get("install_path"):
        copies.append((audio["path"], audio["install_path"]))
    for face in bundled.get("benchmark_faces") or []:
        if face.get("install_path"):
            copies.append((face["path"], face["install_path"]))

    for source_rel, target_rel in copies:
        source = repo_path(repo_root, source_rel)
        target = repo_path(repo_root, target_rel)
        if not source.exists():
            raise FileNotFoundError(source)
        if source.resolve() == target.resolve():
            continue
        if dry_run:
            log(f"dry-run: copy {source} -> {target}")
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        log(f"installed asset: {target}")


def command_prepare(args: argparse.Namespace) -> int:
    repo_root = Path(args.repo_root).resolve()
    cfg = load_artifact_config(Path(args.artifact_config))
    env = runtime_env(args)
    remote = args.remote or str(cfg.get("remote") or "gdrive:")
    if not remote.endswith(":"):
        remote = f"{remote}:"

    if args.install_assets:
        install_bundled_assets(cfg, repo_root, dry_run=args.dry_run)

    artifacts = selected_artifacts(
        cfg,
        include_optional=args.include_optional,
        only=args.only,
    )
    failures = 0
    for group_name, name, item in artifacts:
        target = repo_path(repo_root, item["target_path"])
        expected_sha = item.get("sha256")
        expected_size = item.get("size_bytes")
        if target.exists() and args.skip_existing:
            log(f"skip existing: {group_name}.{name} -> {target}")
            if args.verify:
                failures += 0 if verify_file(target, expected_sha, expected_size) else 1
            continue

        file_id = item.get("drive_file_id")
        if not file_id:
            raise SystemExit(f"artifact {group_name}.{name} has no drive_file_id")
        target.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            "rclone",
            "--drive-acknowledge-abuse",
            "backend",
            "copyid",
            remote,
            str(file_id),
            str(target),
        ]
        run(cmd, env=env, dry_run=args.dry_run)
        if args.verify and not args.dry_run:
            failures += 0 if verify_file(target, expected_sha, expected_size) else 1

    return 1 if failures else 0


def command_list_artifacts(args: argparse.Namespace) -> int:
    cfg = load_artifact_config(Path(args.artifact_config))
    for group_name, name, item in iter_artifacts(cfg, include_optional=True):
        default = item.get("prepare_default", True)
        print(
            f"{group_name}.{name}\tdefault={default}\t"
            f"file_id={item.get('drive_file_id', '')}\t"
            f"target={item.get('target_path', '')}"
        )
    return 0


def command_doctor(args: argparse.Namespace) -> int:
    repo_root = Path(args.repo_root).resolve()
    cfg = load_artifact_config(Path(args.artifact_config))
    failures = 0
    print(f"repo_root={repo_root}")
    print(f"python={sys.executable}")
    if shutil.which("ffmpeg"):
        print(f"ffmpeg={capture(['ffmpeg', '-version']).splitlines()[0]}")
    else:
        print("ffmpeg=missing")
        failures += 1
    if shutil.which("rclone"):
        print(f"rclone={capture(['rclone', 'version']).splitlines()[0]}")
    else:
        print("rclone=missing")
        failures += 1
    torch_probe = (
        "import torch; "
        "print('torch=' + torch.__version__); "
        "print('cuda_available=' + str(torch.cuda.is_available())); "
        "print('cuda_device_count=' + str(torch.cuda.device_count()))"
    )
    torch_output = capture([sys.executable, "-c", torch_probe])
    print(torch_output)
    if torch_output.startswith("unavailable:"):
        failures += 1

    bundled = cfg.get("bundled_assets") or {}
    for item in [bundled.get("benchmark_audio") or {}, bundled.get("face_landmarker") or {}]:
        if item.get("path"):
            failures += 0 if verify_file(repo_path(repo_root, item["path"]), item.get("sha256"), item.get("size_bytes")) else 1
    for face in bundled.get("benchmark_faces") or []:
        failures += 0 if verify_file(repo_path(repo_root, face["path"]), face.get("sha256"), face.get("size_bytes")) else 1

    if args.require_prepared:
        for _, _, item in iter_artifacts(cfg, include_optional=args.include_optional):
            failures += 0 if verify_file(repo_path(repo_root, item["target_path"]), item.get("sha256"), item.get("size_bytes")) else 1
    return 1 if failures else 0


def command_merge_dataset(args: argparse.Namespace) -> int:
    cmd = [
        sys.executable,
        "scripts/merge_faceclip_archives_from_gdrive.py",
        "--remote",
        args.remote,
        "--folder-id",
        args.folder_id,
        "--archive-glob",
        args.archive_glob,
        "--max-archives",
        str(args.max_archives),
        "--manifest-path",
        args.manifest_path,
        "--download-dir",
        args.download_dir,
        "--extract-dir",
        args.extract_dir,
        "--hdtf-root",
        args.hdtf_root,
        "--talkvid-root",
        args.talkvid_root,
        "--import-subdir",
        args.import_subdir,
    ]
    if args.reload_all:
        cmd.append("--reload-all")
    for tier in args.include_tier or []:
        cmd.extend(["--include-tier", tier])
    run(cmd, cwd=TRAINING_ROOT, env=runtime_env(args), dry_run=args.dry_run)
    return 0


def command_prewarm_cache(args: argparse.Namespace) -> int:
    cmd = [
        sys.executable,
        "scripts/prewarm_lazy_cache.py",
        "--config",
        args.config,
        "--log-every",
        str(args.log_every),
        "--sync-alignment-registry-path",
        args.sync_alignment_registry_path,
        "--sync-alignment-remote",
        args.sync_alignment_remote,
        "--sync-alignment-remote-name",
        args.sync_alignment_remote_name,
    ]
    if args.sync_alignment_folder_id:
        cmd.extend(["--sync-alignment-folder-id", args.sync_alignment_folder_id])
    if args.no_sync_alignment_download:
        cmd.append("--no-sync-alignment-download")
    if args.no_sync_alignment_upload:
        cmd.append("--no-sync-alignment-upload")
    if args.speaker_list:
        cmd.extend(["--speaker-list", args.speaker_list])
    if args.max_items is not None:
        cmd.extend(["--max-items", str(args.max_items)])
    run(cmd, cwd=TRAINING_ROOT, env=runtime_env(args), dry_run=args.dry_run)
    return 0


def command_train_syncnet(args: argparse.Namespace) -> int:
    cmd = [sys.executable, "scripts/train_syncnet.py", "--config", args.config]
    if args.resume:
        cmd.extend(["--resume", args.resume])
    if args.speaker_list:
        cmd.extend(["--speaker-list", args.speaker_list])
    if args.val_speaker_list:
        cmd.extend(["--val-speaker-list", args.val_speaker_list])
    run(cmd, cwd=TRAINING_ROOT, env=runtime_env(args), dry_run=args.dry_run)
    return 0


def command_train_generator(args: argparse.Namespace) -> int:
    cmd = [
        sys.executable,
        "scripts/train_generator.py",
        "--config",
        args.config,
        "--syncnet",
        args.syncnet,
    ]
    if args.resume:
        cmd.extend(["--resume", args.resume])
    if args.speaker_list:
        cmd.extend(["--speaker-list", args.speaker_list])
    run(cmd, cwd=TRAINING_ROOT, env=runtime_env(args), dry_run=args.dry_run)
    return 0


def command_train_generator_gan(args: argparse.Namespace) -> int:
    cmd = [
        sys.executable,
        "scripts/train_generator_mirror_gan.py",
        "--config",
        args.config,
        "--syncnet",
        args.syncnet,
    ]
    if args.resume:
        cmd.extend(["--checkpoint-path", args.resume])
    if args.disc_resume:
        cmd.extend(["--disc-checkpoint-path", args.disc_resume])
    if args.speaker_list:
        cmd.extend(["--speaker-list", args.speaker_list])
    if args.val_speaker_list:
        cmd.extend(["--val-speaker-list", args.val_speaker_list])
    cmd.extend(["--eval-seed", str(args.eval_seed)])
    run(cmd, cwd=TRAINING_ROOT, env=runtime_env(args), dry_run=args.dry_run)
    return 0


def default_benchmark_checkpoint(repo_root: Path) -> Path:
    current_best = repo_root / "training/output/current_best_gan_20260414/generator_step000066000.pth"
    if current_best.exists():
        return current_best
    return repo_root / "models/wav2lip/checkpoints/wav2lip_gan.pth"


def command_benchmark(args: argparse.Namespace) -> int:
    repo_root = Path(args.repo_root).resolve()
    cfg = load_artifact_config(Path(args.artifact_config))
    install_bundled_assets(cfg, repo_root, dry_run=args.dry_run)
    checkpoint = Path(args.checkpoint).resolve() if args.checkpoint else default_benchmark_checkpoint(repo_root)
    if not args.dry_run and not checkpoint.exists():
        raise FileNotFoundError(f"benchmark checkpoint not found: {checkpoint}; run lipsyncctl prepare first")

    outfile = Path(args.outfile) if args.outfile else repo_root / "training/output/benchmarks/lipsync_benchmark/default.mp4"
    cmd = [
        sys.executable,
        "training/scripts/run_lipsync_benchmark.py",
        "--face",
        str(repo_root / args.face),
        "--audio",
        str(repo_root / args.audio),
        "--checkpoint",
        str(checkpoint),
        "--outfile",
        str(outfile),
        "--device",
        args.device,
        "--landmarker_device",
        args.landmarker_device,
        "--batch_size",
        str(args.batch_size),
        "--face_landmarker_path",
        str(repo_root / args.face_landmarker_path),
    ]
    if args.keep_intermediates:
        cmd.append("--keep_intermediates")
    run(cmd, cwd=repo_root, env=runtime_env(args), dry_run=args.dry_run)
    return 0


def add_common_runtime_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--repo-root", default=os.environ.get("LIPSYNC_REPO_ROOT", str(REPO_ROOT)))
    parser.add_argument("--artifact-config", default=str(DEFAULT_ARTIFACT_CONFIG))
    parser.add_argument("--rclone-config", default=os.environ.get("RCLONE_CONFIG", str(DEFAULT_RCLONE_CONFIG)))
    parser.add_argument("--dry-run", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Docker/Vast runtime CLI for lip-sync training")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="Download runtime checkpoints and install bundled assets")
    add_common_runtime_args(prepare)
    prepare.add_argument("--remote", default=None)
    prepare.add_argument("--only", action="append", help="Artifact name or group.name to download; repeatable")
    prepare.add_argument("--include-optional", action="store_true", help="Also download optional artifacts such as noGAN")
    prepare.add_argument("--skip-existing", action="store_true")
    prepare.add_argument("--no-verify", dest="verify", action="store_false")
    prepare.add_argument("--no-install-assets", dest="install_assets", action="store_false")
    prepare.set_defaults(func=command_prepare, verify=True, install_assets=True)

    list_artifacts = subparsers.add_parser("list-artifacts", help="List runtime artifact registry entries")
    list_artifacts.add_argument("--artifact-config", default=str(DEFAULT_ARTIFACT_CONFIG))
    list_artifacts.set_defaults(func=command_list_artifacts)

    doctor = subparsers.add_parser("doctor", help="Check runtime dependencies, assets, and optional checkpoints")
    add_common_runtime_args(doctor)
    doctor.add_argument("--require-prepared", action="store_true")
    doctor.add_argument("--include-optional", action="store_true")
    doctor.set_defaults(func=command_doctor)

    merge = subparsers.add_parser("merge-dataset", help="Merge processed faceclip archives from Drive")
    add_common_runtime_args(merge)
    merge.add_argument("--remote", default="gdrive:")
    merge.add_argument("--folder-id", default=DEFAULT_DATASET_FOLDER_ID)
    merge.add_argument("--archive-glob", default="*faceclips*.tar")
    merge.add_argument("--max-archives", type=int, default=0)
    merge.add_argument("--manifest-path", default="output/faceclip_merge/merge_manifest.jsonl")
    merge.add_argument("--download-dir", default="data/_imports/faceclip_merge_downloads")
    merge.add_argument("--extract-dir", default="data/_imports/faceclip_merge_extracted")
    merge.add_argument("--hdtf-root", default="data/hdtf/processed")
    merge.add_argument("--talkvid-root", default="data/talkvid/processed")
    merge.add_argument("--import-subdir", default="_lazy_imports")
    merge.add_argument("--include-tier", action="append")
    merge.add_argument("--reload-all", action="store_true")
    merge.set_defaults(func=command_merge_dataset)

    prewarm = subparsers.add_parser("prewarm-cache", help="Materialize lazy frames/mels and sync-alignment cache")
    add_common_runtime_args(prewarm)
    prewarm.add_argument("--config", default="configs/syncnet_mirror_batch32.yaml")
    prewarm.add_argument("--speaker-list", default=None)
    prewarm.add_argument("--max-items", type=int, default=None)
    prewarm.add_argument("--log-every", type=int, default=100)
    prewarm.add_argument("--sync-alignment-registry-path", default=DEFAULT_SYNC_ALIGNMENT_REGISTRY_PATH)
    prewarm.add_argument("--sync-alignment-remote", default="gdrive:")
    prewarm.add_argument("--sync-alignment-folder-id", default=DEFAULT_DATASET_FOLDER_ID)
    prewarm.add_argument("--sync-alignment-remote-name", default=DEFAULT_SYNC_ALIGNMENT_REMOTE_NAME)
    prewarm.add_argument("--no-sync-alignment-download", action="store_true")
    prewarm.add_argument("--no-sync-alignment-upload", action="store_true")
    prewarm.set_defaults(func=command_prewarm_cache)

    sync_align = subparsers.add_parser("sync-align", help="Compute/load dataset sync alignment and update the Drive registry")
    add_common_runtime_args(sync_align)
    sync_align.add_argument("--config", default="configs/syncnet_mirror_batch32.yaml")
    sync_align.add_argument("--speaker-list", default=None)
    sync_align.add_argument("--max-items", type=int, default=None)
    sync_align.add_argument("--log-every", type=int, default=100)
    sync_align.add_argument("--sync-alignment-registry-path", default=DEFAULT_SYNC_ALIGNMENT_REGISTRY_PATH)
    sync_align.add_argument("--sync-alignment-remote", default="gdrive:")
    sync_align.add_argument("--sync-alignment-folder-id", default=DEFAULT_DATASET_FOLDER_ID)
    sync_align.add_argument("--sync-alignment-remote-name", default=DEFAULT_SYNC_ALIGNMENT_REMOTE_NAME)
    sync_align.add_argument("--no-sync-alignment-download", action="store_true")
    sync_align.add_argument("--no-sync-alignment-upload", action="store_true")
    sync_align.set_defaults(func=command_prewarm_cache)

    train_syncnet = subparsers.add_parser("train-syncnet", help="Run SyncNet training")
    add_common_runtime_args(train_syncnet)
    train_syncnet.add_argument("--config", default="configs/syncnet_mirror_batch32.yaml")
    train_syncnet.add_argument("--resume", default=None)
    train_syncnet.add_argument("--speaker-list", default=None)
    train_syncnet.add_argument("--val-speaker-list", default=None)
    train_syncnet.set_defaults(func=command_train_syncnet)

    train_generator = subparsers.add_parser("train-generator", help="Run non-GAN generator training")
    add_common_runtime_args(train_generator)
    train_generator.add_argument("--config", default="configs/lipsync_cuda3090_hdtf_talkvid.yaml")
    train_generator.add_argument("--syncnet", default="../models/official_syncnet/checkpoints/lipsync_expert.pth")
    train_generator.add_argument("--resume", default=None)
    train_generator.add_argument("--speaker-list", default=None)
    train_generator.set_defaults(func=command_train_generator)

    train_gan = subparsers.add_parser("train-generator-gan", help="Run mirror official-HQ GAN generator training")
    add_common_runtime_args(train_gan)
    train_gan.add_argument("--config", default="configs/generator_mirror_gan_tiltaware_dataset_adaptive_20260414.yaml")
    train_gan.add_argument("--syncnet", default="output/syncnet_current_best_20260428/syncnet_best_our_eval.pth")
    train_gan.add_argument("--resume", default=None)
    train_gan.add_argument("--disc-resume", default=None)
    train_gan.add_argument("--speaker-list", default=None)
    train_gan.add_argument("--val-speaker-list", default=None)
    train_gan.add_argument("--eval-seed", type=int, default=20260408)
    train_gan.set_defaults(func=command_train_generator_gan)

    benchmark = subparsers.add_parser("benchmark", help="Run the tilt-aware x96 benchmark")
    add_common_runtime_args(benchmark)
    benchmark.add_argument("--face", default="assets/benchmark/portrait_avatar.mp4")
    benchmark.add_argument("--audio", default="assets/benchmark/short_4s.mp3")
    benchmark.add_argument("--checkpoint", default=None)
    benchmark.add_argument("--outfile", default=None)
    benchmark.add_argument("--device", default="auto", choices=("auto", "cpu", "mps", "cuda"))
    benchmark.add_argument("--landmarker-device", default="auto", choices=("auto", "cpu", "gpu"))
    benchmark.add_argument("--batch-size", type=int, default=16)
    benchmark.add_argument("--face-landmarker-path", default="models/face_processing/face_landmarker_v2_with_blendshapes.task")
    benchmark.add_argument("--keep-intermediates", action="store_true")
    benchmark.set_defaults(func=command_benchmark)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
