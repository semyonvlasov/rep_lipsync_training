#!/usr/bin/env python3
"""
Upload training artifacts to the shared Google Drive folder, keep a local
manifest, and append a git-tracked markdown entry to the training run log.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from pathlib import Path

try:
    import yaml
except ImportError:  # pragma: no cover - optional on bare local machines
    yaml = None

DEFAULT_DRIVE_ROOT_ID = "1y1P-LI3YTPV65zpHXMSrNYsykN2-s5Pv"
DEFAULT_DRIVE_ROOT_LINK = "https://drive.google.com/drive/folders/1y1P-LI3YTPV65zpHXMSrNYsykN2-s5Pv?usp=sharing"

REPORT_SUFFIXES = {
    ".json",
    ".jsonl",
    ".log",
    ".md",
    ".txt",
    ".yaml",
    ".yml",
    ".csv",
}
MEDIA_SUFFIXES = {".jpg", ".jpeg", ".png", ".mp4", ".wav", ".mp3"}
SKIP_FILENAMES = {".DS_Store", "._.DS_Store"}
SKIP_DIRS = {"__pycache__", ".tmp", "tmp"}
SUMMARY_LINE_RE = re.compile(r"^\[(?P<status>PASS|FAIL)\]\s+(?P<name>[^:]+):\s+(?P<metrics>.*)$")


def timestamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S %Z")


def log(message: str) -> None:
    print(f"{timestamp()} {message}", flush=True)


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


def run_capture(cmd: list[str]) -> str:
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return proc.stdout.strip()


def detect_run_kind(source_dir: Path, explicit: str | None) -> str:
    if explicit and explicit != "auto":
        return explicit
    name = source_dir.name.lower()
    has_syncnet = (source_dir / "syncnet").exists()
    has_generator = (source_dir / "generator").exists()
    if "smoke" in name:
        return "smoke"
    if has_syncnet and has_generator:
        return "pipeline"
    if has_generator:
        return "generator"
    if has_syncnet:
        return "syncnet"
    return "run"


def should_skip(rel_path: Path) -> bool:
    if rel_path.name in SKIP_FILENAMES:
        return True
    if any(part in SKIP_DIRS for part in rel_path.parts):
        return True
    return False


def classify_artifact(rel_path: Path, include_media: bool) -> str | None:
    suffix = rel_path.suffix.lower()
    if suffix == ".pth":
        return "checkpoints"
    if suffix in REPORT_SUFFIXES:
        return "reports"
    if include_media and suffix in MEDIA_SUFFIXES:
        return "media"
    return None


def collect_artifacts(source_dir: Path, include_media: bool) -> list[dict]:
    items: list[dict] = []
    for path in sorted(source_dir.rglob("*")):
        if not path.is_file():
            continue
        rel_path = path.relative_to(source_dir)
        if should_skip(rel_path):
            continue
        category = classify_artifact(rel_path, include_media)
        if category is None:
            continue
        items.append(
            {
                "local_path": str(path),
                "relative_path": rel_path.as_posix(),
                "category": category,
                "size_bytes": path.stat().st_size,
            }
        )
    return items


def build_remote_path(run_kind: str, run_name: str, artifact: dict) -> str:
    rel = artifact["relative_path"]
    category = artifact["category"]
    return f"{run_kind}/{run_name}/{category}/{rel}"


def rclone_copy_file(local_path: str, remote: str, drive_root_folder_id: str, remote_path: str, transfers: int) -> None:
    cmd = [
        "rclone",
        "copyto",
        "--drive-root-folder-id",
        drive_root_folder_id,
        "--transfers",
        str(transfers),
        local_path,
        f"{remote}{remote_path}",
    ]
    run_logged(cmd, prefix="[TrainArtifacts:rclone-copy]")


def rclone_link(remote: str, drive_root_folder_id: str, remote_path: str) -> str | None:
    cmd = [
        "rclone",
        "link",
        "--drive-root-folder-id",
        drive_root_folder_id,
        f"{remote}{remote_path}",
    ]
    try:
        return run_capture(cmd)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def load_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    if not path.exists():
        return records
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def maybe_load_torch_config_from_checkpoint(checkpoint_path: Path) -> tuple[dict | None, str | None]:
    try:
        import torch
    except ImportError:
        return None, None
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except Exception:
        return None, None
    cfg = checkpoint.get("config") if isinstance(checkpoint, dict) else None
    if isinstance(cfg, dict):
        return cfg, str(checkpoint_path)
    return None, None


def load_training_config(source_dir: Path, config_path: Path | None, artifacts: list[dict]) -> tuple[dict | None, str | None]:
    if config_path is not None and config_path.exists():
        if yaml is None:
            log(f"[TrainArtifacts] config parsing skipped because PyYAML is not installed: {config_path}")
            return None, str(config_path)
        with config_path.open(encoding="utf-8") as f:
            return yaml.safe_load(f), str(config_path)

    checkpoint_candidates = [
        Path(item["local_path"])
        for item in artifacts
        if item["category"] == "checkpoints" and item["relative_path"].endswith(".pth")
    ]
    for ckpt in checkpoint_candidates:
        cfg, source = maybe_load_torch_config_from_checkpoint(ckpt)
        if cfg:
            return cfg, source
    return None, None


def summarise_config(cfg: dict | None, config_source: str | None) -> dict | None:
    if not cfg:
        return None

    summary: dict[str, object] = {
        "config_source": config_source,
        "model": {
            "img_size": cfg.get("model", {}).get("img_size"),
            "mel_steps": cfg.get("model", {}).get("mel_steps"),
            "predict_alpha": cfg.get("model", {}).get("predict_alpha"),
            "base_channels": cfg.get("model", {}).get("base_channels"),
        },
        "audio": {
            "sample_rate": cfg.get("audio", {}).get("sample_rate"),
            "hop_size": cfg.get("audio", {}).get("hop_size"),
            "n_mels": cfg.get("audio", {}).get("n_mels"),
            "n_fft": cfg.get("audio", {}).get("n_fft"),
        },
        "data": {
            "fps": cfg.get("data", {}).get("fps"),
            "crop_size": cfg.get("data", {}).get("crop_size"),
            "num_workers": cfg.get("data", {}).get("num_workers"),
            "materialize_frames_size": cfg.get("data", {}).get("materialize_frames_size"),
            "hdtf_root": cfg.get("data", {}).get("hdtf_root"),
            "talkvid_root": cfg.get("data", {}).get("talkvid_root"),
            "lazy_cache_root": cfg.get("data", {}).get("lazy_cache_root"),
        },
        "training": {
            "device": cfg.get("training", {}).get("device"),
            "mixed_precision": cfg.get("training", {}).get("mixed_precision"),
            "save_every": cfg.get("training", {}).get("save_every"),
            "sample_every": cfg.get("training", {}).get("sample_every"),
            "output_dir": cfg.get("training", {}).get("output_dir"),
        },
    }

    if cfg.get("syncnet"):
        summary["syncnet"] = {
            "epochs": cfg.get("syncnet", {}).get("epochs"),
            "batch_size": cfg.get("syncnet", {}).get("batch_size"),
            "lr": cfg.get("syncnet", {}).get("lr"),
            "T": cfg.get("syncnet", {}).get("T"),
        }
    if cfg.get("generator"):
        summary["generator"] = {
            "epochs": cfg.get("generator", {}).get("epochs"),
            "batch_size": cfg.get("generator", {}).get("batch_size"),
            "lr": cfg.get("generator", {}).get("lr"),
            "lr_scheduler": cfg.get("generator", {}).get("lr_scheduler"),
            "loss": cfg.get("generator", {}).get("loss", {}),
        }

    return summary


def parse_sanity_summary(summary_path: Path) -> list[dict]:
    rows: list[dict] = []
    if not summary_path.exists():
        return rows
    with summary_path.open(encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            match = SUMMARY_LINE_RE.match(line)
            if not match:
                continue
            metrics: dict[str, float] = {}
            metric_blob = match.group("metrics")
            for piece in metric_blob.split(","):
                piece = piece.strip()
                if "=" not in piece:
                    continue
                key, value = piece.split("=", 1)
                try:
                    metrics[key] = float(value)
                except ValueError:
                    continue
            rows.append(
                {
                    "name": match.group("name"),
                    "pass": match.group("status") == "PASS",
                    "metrics": metrics,
                }
            )
    return rows


def extract_benchmark_summary(source_dir: Path) -> dict:
    summary: dict[str, object] = {}

    compare_candidates = sorted(source_dir.glob("syncnet_teacher_compare*.json"))
    if compare_candidates:
        compare_path = compare_candidates[-1]
        compare = load_json(compare_path)
        if compare:
            teachers = compare.get("teachers", {})
            summary["syncnet_compare"] = {
                "path": compare_path.name,
                "holdout_total": compare.get("holdout_total"),
                "samples": compare.get("samples"),
                "teachers": {
                    name: {
                        "pairwise_acc_mean": metrics.get("pairwise_acc_mean"),
                        "margin_mean": metrics.get("margin_mean"),
                        "shifted_pairwise_acc": metrics.get("shifted_pairwise_acc"),
                        "foreign_pairwise_acc": metrics.get("foreign_pairwise_acc"),
                    }
                    for name, metrics in teachers.items()
                },
            }

    selected_candidates = sorted(source_dir.glob("syncnet_selected_teacher*.json"))
    if selected_candidates:
        selected_path = selected_candidates[-1]
        selected = load_json(selected_path)
        if selected:
            summary["selected_teacher"] = {
                "path": selected_path.name,
                "winner_name": selected.get("winner_name"),
                "winner_kind": selected.get("winner_kind"),
                "pairwise_acc_mean": selected.get("winner_metrics", {}).get("pairwise_acc_mean"),
                "margin_mean": selected.get("winner_metrics", {}).get("margin_mean"),
            }

    watchdog_history_path = source_dir / "watchdog_history.jsonl"
    if watchdog_history_path.exists():
        records = load_jsonl(watchdog_history_path)
        if records:
            best = max(records, key=lambda item: float(item.get("watchdog", {}).get("score", float("-inf"))))
            last = records[-1]
            summary["watchdog"] = {
                "path": watchdog_history_path.name,
                "epochs_logged": len(records),
                "best_epoch": best.get("epoch"),
                "best_score": best.get("watchdog", {}).get("score"),
                "last_epoch": last.get("epoch"),
                "last_score": last.get("watchdog", {}).get("score"),
                "last_pass_count": last.get("watchdog", {}).get("pass_count"),
            }

    status_path = source_dir / "watchdog_status.txt"
    if status_path.exists():
        status: dict[str, str] = {}
        for line in status_path.read_text(encoding="utf-8").splitlines():
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            status[key] = value
        if status:
            summary["watchdog_status"] = status

    sanity_candidates = sorted(source_dir.glob("sanity_epoch*/summary.txt"))
    if sanity_candidates:
        sanity_path = sanity_candidates[-1]
        sanity_rows = parse_sanity_summary(sanity_path)
        if sanity_rows:
            summary["sanity"] = {
                "path": str(sanity_path.relative_to(source_dir)),
                "checks": sanity_rows,
            }

    return summary


def ensure_log_header(path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "# Training Run Log\n\n"
        "Git-tracked registry of training runs and uploaded Drive artifacts.\n\n"
        f"Artifacts root: {DEFAULT_DRIVE_ROOT_LINK}\n",
        encoding="utf-8",
    )


def format_link(label: str, url: str | None, fallback: str) -> str:
    if url:
        return f"[{label}]({url})"
    return f"`{fallback}`"


def append_markdown_log(
    log_path: Path,
    uploaded_at: str,
    run_kind: str,
    run_name: str,
    source_dir: Path,
    drive_root_link: str,
    remote_base_path: str,
    manifest_remote_path: str,
    manifest_link: str | None,
    artifacts: list[dict],
    config_summary: dict | None,
    benchmark_summary: dict,
) -> None:
    ensure_log_header(log_path)

    checkpoint_lines = []
    for item in artifacts:
        if item["category"] != "checkpoints":
            continue
        checkpoint_lines.append(
            f"- {format_link(Path(item['relative_path']).name, item.get('link'), item['remote_path'])}"
        )

    report_lines = []
    for item in artifacts:
        if item["category"] != "reports":
            continue
        if "/" in item["relative_path"]:
            continue
        report_lines.append(
            f"- {format_link(Path(item['relative_path']).name, item.get('link'), item['remote_path'])}"
        )

    config_lines: list[str] = []
    if config_summary:
        config_lines.append(
            "- model:"
            f" `img_size={config_summary.get('model', {}).get('img_size')}`"
            f" `mel_steps={config_summary.get('model', {}).get('mel_steps')}`"
            f" `base_channels={config_summary.get('model', {}).get('base_channels')}`"
            f" `predict_alpha={config_summary.get('model', {}).get('predict_alpha')}`"
        )
        config_lines.append(
            "- audio:"
            f" `sr={config_summary.get('audio', {}).get('sample_rate')}`"
            f" `hop={config_summary.get('audio', {}).get('hop_size')}`"
            f" `n_mels={config_summary.get('audio', {}).get('n_mels')}`"
            f" `n_fft={config_summary.get('audio', {}).get('n_fft')}`"
        )
        data_cfg = config_summary.get("data", {})
        config_lines.append(
            "- data:"
            f" `fps={data_cfg.get('fps')}`"
            f" `crop={data_cfg.get('crop_size')}`"
            f" `workers={data_cfg.get('num_workers')}`"
            f" `materialize={data_cfg.get('materialize_frames_size')}`"
            f" `hdtf_root={data_cfg.get('hdtf_root')}`"
            f" `talkvid_root={data_cfg.get('talkvid_root')}`"
        )
        if config_summary.get("syncnet"):
            sync_cfg = config_summary["syncnet"]
            config_lines.append(
                "- syncnet:"
                f" `epochs={sync_cfg.get('epochs')}`"
                f" `batch={sync_cfg.get('batch_size')}`"
                f" `lr={sync_cfg.get('lr')}`"
                f" `T={sync_cfg.get('T')}`"
            )
        if config_summary.get("generator"):
            gen_cfg = config_summary["generator"]
            config_lines.append(
                "- generator:"
                f" `epochs={gen_cfg.get('epochs')}`"
                f" `batch={gen_cfg.get('batch_size')}`"
                f" `lr={gen_cfg.get('lr')}`"
                f" `sched={gen_cfg.get('lr_scheduler')}`"
            )
            loss_cfg = gen_cfg.get("loss", {})
            if loss_cfg:
                config_lines.append(
                    "- generator_loss:"
                    f" `l1={loss_cfg.get('l1')}`"
                    f" `sync={loss_cfg.get('sync')}`"
                    f" `perc={loss_cfg.get('perceptual')}`"
                    f" `gan={loss_cfg.get('gan')}`"
                    f" `alpha_reg={loss_cfg.get('alpha_reg')}`"
                )

    benchmark_lines: list[str] = []
    selected = benchmark_summary.get("selected_teacher")
    if isinstance(selected, dict):
        benchmark_lines.append(
            "- selected_teacher:"
            f" `winner={selected.get('winner_name')}`"
            f" `kind={selected.get('winner_kind')}`"
            f" `pairwise_acc={selected.get('pairwise_acc_mean')}`"
            f" `margin={selected.get('margin_mean')}`"
        )
    sync_compare = benchmark_summary.get("syncnet_compare")
    if isinstance(sync_compare, dict):
        benchmark_lines.append(
            "- syncnet_compare:"
            f" `holdout_total={sync_compare.get('holdout_total')}`"
            f" `samples={sync_compare.get('samples')}`"
        )
        teachers = sync_compare.get("teachers", {})
        if isinstance(teachers, dict):
            for name, metrics in teachers.items():
                benchmark_lines.append(
                    f"- teacher `{name}`:"
                    f" `pairwise_acc={metrics.get('pairwise_acc_mean')}`"
                    f" `margin={metrics.get('margin_mean')}`"
                    f" `shifted_acc={metrics.get('shifted_pairwise_acc')}`"
                    f" `foreign_acc={metrics.get('foreign_pairwise_acc')}`"
                )
    watchdog = benchmark_summary.get("watchdog")
    if isinstance(watchdog, dict):
        benchmark_lines.append(
            "- watchdog:"
            f" `best_epoch={watchdog.get('best_epoch')}`"
            f" `best_score={watchdog.get('best_score')}`"
            f" `last_epoch={watchdog.get('last_epoch')}`"
            f" `last_score={watchdog.get('last_score')}`"
            f" `pass_count={watchdog.get('last_pass_count')}`"
        )
    sanity = benchmark_summary.get("sanity")
    if isinstance(sanity, dict):
        for row in sanity.get("checks", []):
            benchmark_lines.append(
                f"- sanity `{row.get('name')}`:"
                f" `pass={row.get('pass')}`"
                + "".join(f" `{k}={v}`" for k, v in row.get("metrics", {}).items())
            )

    lines = [
        "",
        f"## {uploaded_at} `{run_name}`",
        f"- kind: `{run_kind}`",
        f"- local_output: `{source_dir.as_posix()}`",
        f"- artifacts_root: [Google Drive]({drive_root_link})",
        f"- drive_subdir: `{remote_base_path}`",
        f"- manifest: {format_link('artifacts_upload_manifest.json', manifest_link, manifest_remote_path)}",
    ]
    if config_lines:
        lines.append("- training_params:")
        lines.extend(config_lines)
    if benchmark_lines:
        lines.append("- benchmarks:")
        lines.extend(benchmark_lines)
    if checkpoint_lines:
        lines.append("- checkpoints:")
        lines.extend(checkpoint_lines)
    if report_lines:
        lines.append("- top_reports:")
        lines.extend(report_lines)

    with log_path.open("a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", required=True, help="Training output directory to upload")
    parser.add_argument("--run-kind", default="auto", help="syncnet|generator|pipeline|smoke|run|auto")
    parser.add_argument("--run-name", default=None, help="Override output basename used on Drive")
    parser.add_argument("--remote", default="gdrive:")
    parser.add_argument("--drive-root-folder-id", default=DEFAULT_DRIVE_ROOT_ID)
    parser.add_argument("--drive-root-link", default=DEFAULT_DRIVE_ROOT_LINK)
    parser.add_argument("--config-path", default=None, help="Optional training config used for this run")
    parser.add_argument("--manifest-path", default=None, help="Local JSON manifest path")
    parser.add_argument("--log-path", default=None, help="Git-tracked markdown log file")
    parser.add_argument("--rclone-transfers", type=int, default=1)
    parser.add_argument("--include-media", action="store_true", help="Also upload previews/samples/media")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    source_dir = Path(args.source_dir).resolve()
    if not source_dir.exists():
        raise FileNotFoundError(f"source_dir does not exist: {source_dir}")

    repo_root = Path(__file__).resolve().parents[2]
    log_path = Path(args.log_path) if args.log_path else (repo_root / "docs" / "training_runs" / "log.md")
    manifest_path = Path(args.manifest_path) if args.manifest_path else (source_dir / "artifacts_upload_manifest.json")
    config_path = Path(args.config_path).resolve() if args.config_path else None

    run_kind = detect_run_kind(source_dir, args.run_kind)
    run_name = args.run_name or source_dir.name
    remote_base_path = f"{run_kind}/{run_name}"
    uploaded_at = timestamp()

    artifacts = collect_artifacts(source_dir, include_media=args.include_media)
    for item in artifacts:
        item["remote_path"] = build_remote_path(run_kind, run_name, item)

    log(f"[TrainArtifacts] source_dir={source_dir}")
    log(f"[TrainArtifacts] run_kind={run_kind}")
    log(f"[TrainArtifacts] run_name={run_name}")
    log(f"[TrainArtifacts] artifacts={len(artifacts)}")
    log(f"[TrainArtifacts] drive_subdir={remote_base_path}")

    cfg, config_source = load_training_config(source_dir, config_path, artifacts)
    config_summary = summarise_config(cfg, config_source)
    benchmark_summary = extract_benchmark_summary(source_dir)

    if not args.dry_run:
        for item in artifacts:
            log(f"[TrainArtifacts] uploading {item['relative_path']} -> {item['remote_path']}")
            rclone_copy_file(
                item["local_path"],
                remote=args.remote,
                drive_root_folder_id=args.drive_root_folder_id,
                remote_path=item["remote_path"],
                transfers=args.rclone_transfers,
            )
            item["link"] = rclone_link(args.remote, args.drive_root_folder_id, item["remote_path"])
    else:
        for item in artifacts:
            log(f"[TrainArtifacts] dry-run {item['relative_path']} -> {item['remote_path']}")
            item["link"] = None

    manifest_payload = {
        "uploaded_at": uploaded_at,
        "source_dir": str(source_dir),
        "run_kind": run_kind,
        "run_name": run_name,
        "drive_root_folder_id": args.drive_root_folder_id,
        "drive_root_link": args.drive_root_link,
        "remote_base_path": remote_base_path,
        "config_summary": config_summary,
        "benchmark_summary": benchmark_summary,
        "artifacts": artifacts,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest_payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    manifest_remote_path = f"{remote_base_path}/reports/{manifest_path.name}"
    manifest_link = None
    if not args.dry_run:
        log(f"[TrainArtifacts] uploading manifest -> {manifest_remote_path}")
        rclone_copy_file(
            str(manifest_path),
            remote=args.remote,
            drive_root_folder_id=args.drive_root_folder_id,
            remote_path=manifest_remote_path,
            transfers=args.rclone_transfers,
        )
        manifest_link = rclone_link(args.remote, args.drive_root_folder_id, manifest_remote_path)

    append_markdown_log(
        log_path=log_path,
        uploaded_at=uploaded_at,
        run_kind=run_kind,
        run_name=run_name,
        source_dir=source_dir,
        drive_root_link=args.drive_root_link,
        remote_base_path=remote_base_path,
        manifest_remote_path=manifest_remote_path,
        manifest_link=manifest_link,
        artifacts=artifacts,
        config_summary=config_summary,
        benchmark_summary=benchmark_summary,
    )
    log(f"[TrainArtifacts] wrote manifest={manifest_path}")
    log(f"[TrainArtifacts] updated_log={log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
