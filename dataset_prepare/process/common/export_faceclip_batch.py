#!/usr/bin/env python3
"""
Batch-export raw videos into ranked face segments using the vendored
`face_processing` pipeline.

Output layout:
  output_dir/
    confident/
      sample.mp4
      sample.json
    medium/
      ...
    unconfident/
      ...
    reports/
      source_report.json
      source_frame_log.csv  # optional
    export_manifest.jsonl
    export_resume_state.json
    summary.json

The inner per-video processor is source-agnostic. The outer batch exporter keeps
the existing archive-level orchestration contract:
  - batch-local JSONL manifest
  - resume from the next unfinished source video
  - tiered output folders for downstream packing
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import fields
from pathlib import Path
from typing import Any

THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset_prepare.common.config import (
    ConfigError,
    get_bool,
    get_mapping,
    get_str,
    load_yaml_config,
    resolve_repo_path,
)
from dataset_prepare.process.common.transcode_video import (
    resolve_ffmpeg_bin,
    select_video_encoder,
)


QUALITY_TIERS = ("confident", "medium", "unconfident")


def timestamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S %Z")


def log(message: str) -> None:
    print(f"{timestamp()} {message}", flush=True)


def append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def load_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return None


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def iter_videos(input_dir: Path):
    for path in sorted(input_dir.glob("*.mp4")):
        if path.is_file():
            yield path


def resolve_dataset_kind(requested: str, source_archive: str, input_dir: Path) -> str:
    if requested != "auto":
        return requested
    lower = (source_archive or "").lower()
    if "talkvid" in lower:
        return "talkvid"
    if "hdtf" in lower:
        return "hdtf"
    if any(path.with_suffix(".json").exists() for path in input_dir.glob("*.mp4")):
        return "talkvid"
    return "hdtf"


def load_resume_progress(manifest_path: Path, resume_state_path: Path) -> tuple[dict, int, int]:
    counters = {
        "ok": 0,
        "skip": 0,
        "discard": 0,
        "fail": 0,
        "confident": 0,
        "medium": 0,
        "unconfident": 0,
    }
    total_segments = 0
    latest_manifest_index = 0

    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue

                status = payload.get("status")
                tier = payload.get("tier")
                if status in counters:
                    counters[status] += 1
                if status == "ok" and tier in QUALITY_TIERS:
                    counters[tier] += 1
                total_segments += 1
                try:
                    latest_manifest_index = max(latest_manifest_index, int(payload.get("index") or 0))
                except (TypeError, ValueError):
                    pass

    resume_state = load_json(resume_state_path)
    next_video_index = 0
    if isinstance(resume_state, dict):
        try:
            next_video_index = int(resume_state.get("next_video_index") or 0)
        except (TypeError, ValueError):
            next_video_index = 0

    next_video_index = max(next_video_index, latest_manifest_index)
    return counters, total_segments, next_video_index


def _apply_mapping(target: object, mapping: dict[str, Any], section_name: str) -> None:
    allowed = {field.name for field in fields(target)}
    for key, value in mapping.items():
        if key not in allowed:
            raise ConfigError(f"unsupported key in {section_name}: {key}")
        setattr(target, key, value)


def build_face_processing_config(config_path: Path, config: dict[str, Any], output_dir: Path):
    from face_processing.config import PipelineConfig

    pipeline_cfg = PipelineConfig()

    root_mapping = get_mapping(config, "face_processing")
    normalization_mapping = dict(get_mapping(config, "face_processing", "normalization"))
    detection_mapping = dict(get_mapping(config, "face_processing", "detection"))
    bad_frame_mapping = dict(get_mapping(config, "face_processing", "bad_frame"))
    ranking_mapping = dict(get_mapping(config, "face_processing", "ranking"))
    export_mapping = dict(get_mapping(config, "face_processing", "export"))

    model_path = resolve_repo_path(REPO_ROOT, get_str(config, "face_processing", "detection", "model_path"))
    assert model_path is not None
    if not model_path.is_file():
        raise ConfigError(
            f"missing MediaPipe face landmarker model: {model_path}. "
            "Place the .task file at the configured path before launching processing."
        )
    detection_mapping["model_path"] = str(model_path)

    ffmpeg_bin = resolve_ffmpeg_bin(get_str(config, "runtime", "ffmpeg_bin", allow_empty=True) or None)
    ffmpeg_threads = int(get_mapping(config, "runtime").get("ffmpeg_threads", 0))
    ffmpeg_timeout = int(get_mapping(config, "runtime").get("ffmpeg_timeout", 180))

    normalization_codec = select_video_encoder(str(normalization_mapping.get("codec", "auto")), ffmpeg_bin)
    export_codec = select_video_encoder(str(export_mapping.get("codec", "auto")), ffmpeg_bin)
    normalization_mapping["codec"] = normalization_codec
    export_mapping["codec"] = export_codec
    normalization_mapping["ffmpeg_bin"] = ffmpeg_bin
    normalization_mapping["ffmpeg_threads"] = ffmpeg_threads
    normalization_mapping["ffmpeg_timeout"] = ffmpeg_timeout
    export_mapping["ffmpeg_bin"] = ffmpeg_bin
    export_mapping["ffmpeg_threads"] = ffmpeg_threads
    export_mapping["ffmpeg_timeout"] = ffmpeg_timeout

    _apply_mapping(pipeline_cfg.normalization, normalization_mapping, "face_processing.normalization")
    _apply_mapping(pipeline_cfg.detection, detection_mapping, "face_processing.detection")
    _apply_mapping(pipeline_cfg.bad_frame, bad_frame_mapping, "face_processing.bad_frame")
    _apply_mapping(pipeline_cfg.ranking, ranking_mapping, "face_processing.ranking")
    _apply_mapping(pipeline_cfg.export, export_mapping, "face_processing.export")

    pipeline_cfg.save_frame_log = get_bool(root_mapping, "save_frame_log")
    pipeline_cfg.keep_normalized = get_bool(root_mapping, "keep_normalized")
    pipeline_cfg.output_dir = str(output_dir)
    return pipeline_cfg, model_path


def copy_report_artifacts(video_work_dir: Path, reports_dir: Path, source_name: str) -> None:
    reports_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("_report.json", "_frame_log.csv"):
        src = video_work_dir / f"{source_name}{suffix}"
        if src.exists():
            os.replace(src, reports_dir / src.name)


def promote_exported_segments(
    *,
    video_path: Path,
    source_archive: str,
    dataset_kind: str,
    video_work_dir: Path,
    batch_output_dir: Path,
) -> list[dict]:
    source_name = video_path.stem
    promoted: list[dict] = []
    raw_sidecar = load_json(video_path.with_suffix(".json"))

    for meta_path in sorted(video_work_dir.glob(f"{source_name}_seg_*.json")):
        payload = load_json(meta_path)
        if not isinstance(payload, dict):
            raise RuntimeError(f"invalid_segment_json: {meta_path}")

        rank = str(payload.get("rank") or "")
        if rank not in QUALITY_TIERS:
            raise RuntimeError(f"missing_or_invalid_rank in {meta_path}: {rank!r}")

        mp4_path = meta_path.with_suffix(".mp4")
        if not mp4_path.exists():
            raise RuntimeError(f"missing_segment_video: {mp4_path}")

        payload["source_archive"] = source_archive
        payload["source_dataset"] = dataset_kind
        payload["raw_sidecar_present"] = raw_sidecar is not None

        tier_dir = batch_output_dir / rank
        tier_dir.mkdir(parents=True, exist_ok=True)

        final_mp4 = tier_dir / mp4_path.name
        final_json = tier_dir / meta_path.name
        tmp_meta = meta_path.with_suffix(".json.tmp")
        with tmp_meta.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        os.replace(tmp_meta, meta_path)
        os.replace(mp4_path, final_mp4)
        os.replace(meta_path, final_json)
        promoted.append(payload)

    return promoted


def build_segment_name(source_name: str, segment_id: int) -> str:
    return f"{source_name}_seg_{int(segment_id):03d}"


def result_to_manifest_entries(result, source_name: str, promoted_payloads: list[dict]) -> list[dict]:
    promoted_by_id = {
        int(payload["segment_id"]): payload
        for payload in promoted_payloads
        if "segment_id" in payload
    }
    entries: list[dict] = []
    for segment in result.segments:
        segment_name = build_segment_name(source_name, segment.segment_id)
        if segment.status == "exported":
            payload = promoted_by_id.get(int(segment.segment_id), {})
            entries.append(
                {
                    "name": segment_name,
                    "status": "ok",
                    "tier": segment.rank,
                    "message": (
                        f"{segment_name}: ok frames={segment.length} "
                        f"source_range={segment.start_frame}:{segment.end_frame - 1} "
                        f"size={segment.output_size} rank={segment.rank}"
                    ),
                    "source_segment_start_frame": int(segment.start_frame),
                    "source_segment_end_frame": int(segment.end_frame - 1),
                    "segment_index": int(segment.segment_id) + 1,
                    "segment_count": len(result.segments),
                    "segment_meta": payload,
                }
            )
        elif segment.status == "dropped":
            reason = segment.drop_reason or "dropped"
            entries.append(
                {
                    "name": segment_name,
                    "status": "discard",
                    "tier": None,
                    "message": (
                        f"{segment_name}: dropped reason={reason} "
                        f"source_range={segment.start_frame}:{segment.end_frame - 1}"
                    ),
                    "source_segment_start_frame": int(segment.start_frame),
                    "source_segment_end_frame": int(segment.end_frame - 1),
                    "segment_index": int(segment.segment_id) + 1,
                    "segment_count": len(result.segments),
                }
            )

    if not entries and result.status == "dropped":
        entries.append(
            {
                "name": source_name,
                "status": "discard",
                "tier": None,
                "message": f"{source_name}: video_dropped reason={result.drop_reason or 'unknown'}",
                "source_segment_start_frame": 0,
                "source_segment_end_frame": -1,
                "segment_index": 1,
                "segment_count": 1,
            }
        )
    return entries


def process_one_video(
    *,
    video_path: Path,
    dataset_kind: str,
    source_archive: str,
    batch_output_dir: Path,
    work_root: Path,
    pipeline_cfg,
):
    from face_processing.pipeline import process_video

    source_name = video_path.stem
    pipeline_cfg.output_dir = str(work_root)
    result = process_video(str(video_path), pipeline_cfg)
    video_work_dir = work_root / source_name
    reports_dir = batch_output_dir / "reports"
    promoted_payloads = promote_exported_segments(
        video_path=video_path,
        source_archive=source_archive,
        dataset_kind=dataset_kind,
        video_work_dir=video_work_dir,
        batch_output_dir=batch_output_dir,
    )
    copy_report_artifacts(video_work_dir, reports_dir, source_name)
    shutil.rmtree(video_work_dir, ignore_errors=True)
    return result, result_to_manifest_entries(result, source_name, promoted_payloads)


def cleanup_video_artifacts(*, video_path: Path, batch_output_dir: Path, work_root: Path) -> None:
    source_name = video_path.stem
    shutil.rmtree(work_root / source_name, ignore_errors=True)
    reports_dir = batch_output_dir / "reports"
    for path in reports_dir.glob(f"{source_name}_*"):
        try:
            path.unlink()
        except OSError:
            pass
    for tier in QUALITY_TIERS:
        tier_dir = batch_output_dir / tier
        for path in tier_dir.glob(f"{source_name}_seg_*"):
            try:
                path.unlink()
            except OSError:
                pass


def write_gpu_override_config(config_path: Path, *, use_gpu: bool) -> Path:
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=".yaml",
        prefix="faceclip_export_",
        delete=False,
    )
    with handle:
        handle.write(f"extends: {config_path}\n\n")
        handle.write("face_processing:\n")
        handle.write("  detection:\n")
        handle.write(f"    use_gpu: {'true' if use_gpu else 'false'}\n")
    return Path(handle.name)


def run_video_worker(
    *,
    config_path: Path,
    video_path: Path,
    source_archive: str,
    dataset_kind: str,
    batch_output_dir: Path,
    work_root: Path,
    use_gpu: bool,
) -> tuple[int, list[dict] | None]:
    cleanup_video_artifacts(video_path=video_path, batch_output_dir=batch_output_dir, work_root=work_root)

    worker_result = work_root / f"{video_path.stem}.worker_result.json"
    try:
        worker_result.unlink()
    except OSError:
        pass

    worker_config_path = write_gpu_override_config(config_path, use_gpu=use_gpu)
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--config",
        str(worker_config_path),
        "--input-dir",
        str(video_path.parent),
        "--output-dir",
        str(batch_output_dir),
        "--normalized-dir",
        str(work_root),
        "--source-archive",
        source_archive,
        "--dataset-kind",
        dataset_kind,
        "--worker-video-path",
        str(video_path),
        "--worker-result-path",
        str(worker_result),
    ]
    try:
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
                print(line, flush=True)
        rc = process.wait()
        payload = load_json(worker_result)
        if isinstance(payload, dict):
            entries = payload.get("segment_entries")
            if isinstance(entries, list):
                return rc, entries
        return rc, None
    finally:
        try:
            worker_result.unlink()
        except OSError:
            pass
        try:
            worker_config_path.unlink()
        except OSError:
            pass


def worker_main(args: argparse.Namespace) -> int:
    if not args.worker_video_path or not args.worker_result_path:
        raise SystemExit("worker mode requires --worker-video-path and --worker-result-path")

    try:
        config_path, config = load_yaml_config(args.config)
    except ConfigError as exc:
        log(f"[FaceclipExport] config_error={exc}")
        return 2

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    work_root = Path(args.normalized_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    work_root.mkdir(parents=True, exist_ok=True)

    pipeline_cfg, _ = build_face_processing_config(config_path, config, work_root)
    dataset_kind = resolve_dataset_kind(args.dataset_kind, args.source_archive, input_dir)
    video_path = Path(args.worker_video_path)

    try:
        _, segment_entries = process_one_video(
            video_path=video_path,
            dataset_kind=dataset_kind,
            source_archive=args.source_archive,
            batch_output_dir=output_dir,
            work_root=work_root,
            pipeline_cfg=pipeline_cfg,
        )
    except Exception as exc:
        segment_entries = [
            {
                "name": video_path.stem,
                "status": "fail",
                "tier": None,
                "message": f"{video_path.stem}: unexpected_fail ({type(exc).__name__}: {exc})",
                "source_segment_start_frame": 0,
                "source_segment_end_frame": -1,
                "segment_index": 1,
                "segment_count": 1,
            }
        ]

    write_json(Path(args.worker_result_path), {"segment_entries": segment_entries})
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to the process-stage YAML config")
    parser.add_argument("--input-dir", required=True, help="Directory with source .mp4 files")
    parser.add_argument("--output-dir", required=True, help="Directory for tiered exported segments")
    parser.add_argument(
        "--normalized-dir",
        required=True,
        help="Per-video temporary work root used by the inner face_processing pipeline",
    )
    parser.add_argument("--source-archive", default="", help="Source archive name for provenance")
    parser.add_argument("--dataset-kind", choices=["auto", "talkvid", "hdtf"], default="auto")
    parser.add_argument("--worker-video-path", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker-result-path", default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.worker_video_path or args.worker_result_path:
        return worker_main(args)

    try:
        config_path, config = load_yaml_config(args.config)
    except ConfigError as exc:
        log(f"[FaceclipExport] config_error={exc}")
        return 2

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    work_root = Path(args.normalized_dir)
    manifest_path = output_dir / "export_manifest.jsonl"
    resume_state_path = output_dir / "export_resume_state.json"
    summary_path = output_dir / "summary.json"

    output_dir.mkdir(parents=True, exist_ok=True)
    work_root.mkdir(parents=True, exist_ok=True)

    pipeline_cfg, model_path = build_face_processing_config(config_path, config, work_root)
    dataset_kind = resolve_dataset_kind(args.dataset_kind, args.source_archive, input_dir)

    counters, total_segments, start_video_index = load_resume_progress(manifest_path, resume_state_path)
    videos = list(iter_videos(input_dir))
    start_video_index = max(0, min(start_video_index, len(videos)))

    log(f"[FaceclipExport] input_dir={input_dir}")
    log(f"[FaceclipExport] output_dir={output_dir}")
    log(f"[FaceclipExport] work_root={work_root}")
    log(f"[FaceclipExport] source_archive={args.source_archive or '<none>'}")
    log(f"[FaceclipExport] dataset_kind={dataset_kind}")
    log(f"[FaceclipExport] model_path={model_path}")
    log(
        "[FaceclipExport] normalization="
        f"fps={pipeline_cfg.normalization.fps} "
        f"bitrate={pipeline_cfg.normalization.bitrate} "
        f"codec={pipeline_cfg.normalization.codec}"
    )
    log(
        "[FaceclipExport] export="
        f"fps={pipeline_cfg.export.fps} "
        f"bitrate={pipeline_cfg.export.bitrate} "
        f"codec={pipeline_cfg.export.codec}"
    )
    log(
        "[FaceclipExport] detection="
        f"num_faces={pipeline_cfg.detection.num_faces} "
        f"min_detection_confidence={pipeline_cfg.detection.min_detection_confidence} "
        f"min_presence_confidence={pipeline_cfg.detection.min_presence_confidence} "
        f"use_gpu={pipeline_cfg.detection.use_gpu}"
    )
    log(f"[FaceclipExport] videos={len(videos)}")

    if start_video_index > 0 and start_video_index < len(videos):
        log(
            f"[FaceclipExport] resume_from_video_index={start_video_index + 1} "
            f"source_video={videos[start_video_index].name}"
        )
    elif start_video_index >= len(videos) and len(videos) > 0:
        log("[FaceclipExport] resume_at_end=true; no source videos left to export")

    for idx, video_path in enumerate(videos[start_video_index:], start=start_video_index + 1):
        rc, segment_entries = run_video_worker(
            config_path=config_path,
            video_path=video_path,
            source_archive=args.source_archive,
            dataset_kind=dataset_kind,
            batch_output_dir=output_dir,
            work_root=work_root,
            use_gpu=bool(pipeline_cfg.detection.use_gpu),
        )
        if rc != 0 and pipeline_cfg.detection.use_gpu:
            log(
                f"[FaceclipExport] gpu_worker_failed source_video={video_path.name} rc={rc}; "
                "retrying on CPU"
            )
            rc, segment_entries = run_video_worker(
                config_path=config_path,
                video_path=video_path,
                source_archive=args.source_archive,
                dataset_kind=dataset_kind,
                batch_output_dir=output_dir,
                work_root=work_root,
                use_gpu=False,
            )

        if segment_entries is None:
            segment_entries = [
                {
                    "name": video_path.stem,
                    "status": "fail",
                    "tier": None,
                    "message": f"{video_path.stem}: worker_failed (rc={rc})",
                    "source_segment_start_frame": 0,
                    "source_segment_end_frame": -1,
                    "segment_index": 1,
                    "segment_count": 1,
                }
            ]

        total_segments += len(segment_entries)
        for entry in segment_entries:
            status = entry["status"]
            tier = entry.get("tier")
            counters[status] += 1
            if status == "ok" and tier in QUALITY_TIERS:
                counters[tier] += 1
            log(
                f"[FaceclipExport] [{idx}/{len(videos)}]"
                f"[{entry['segment_index']}/{entry['segment_count']}] {entry['message']}"
            )
            append_jsonl(
                manifest_path,
                {
                    "ts": timestamp(),
                    "index": idx,
                    "total": len(videos),
                    "segment_index": entry["segment_index"],
                    "segment_count": entry["segment_count"],
                    "name": entry["name"],
                    "status": status,
                    "tier": tier,
                    "message": entry["message"],
                    "source_video": video_path.name,
                    "source_segment_start_frame": entry["source_segment_start_frame"],
                    "source_segment_end_frame": entry["source_segment_end_frame"],
                },
            )

        write_json(
            resume_state_path,
            {
                "ts": timestamp(),
                "next_video_index": idx,
                "last_completed_index": idx,
                "last_completed_source_video": video_path.name,
            },
        )

    summary = {
        "ts": timestamp(),
        "processor": "face_processing",
        "config_path": str(config_path),
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "work_root": str(work_root),
        "source_archive": args.source_archive,
        "dataset_kind": dataset_kind,
        "model_path": str(model_path),
        "normalization_codec": pipeline_cfg.normalization.codec,
        "normalization_bitrate": pipeline_cfg.normalization.bitrate,
        "export_codec": pipeline_cfg.export.codec,
        "export_bitrate": pipeline_cfg.export.bitrate,
        "total_videos": len(videos),
        "total_segments": total_segments,
        **counters,
    }
    write_json(summary_path, summary)
    log(f"[FaceclipExport] summary={summary}")
    try:
        resume_state_path.unlink()
    except OSError:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
