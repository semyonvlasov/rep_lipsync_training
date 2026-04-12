from __future__ import annotations

import json
import logging
import os
import tempfile

import cv2

from face_processing.config import PipelineConfig
from face_processing.crop_export import compute_output_size, export_segment
from face_processing.face_analysis import analyze_frames
from face_processing.frame_quality import (
    DROP_REASON_PRIORITY,
    classify_frames,
    compute_deltas,
    pick_primary_reason,
    smooth_pose,
)
from face_processing.logging_utils import save_frame_log
from face_processing.models import Segment, VideoResult
from face_processing.normalize import normalize_video
from face_processing.ranking import compute_segment_metrics, rank_segment
from face_processing.segmentation import split_into_segments

logger = logging.getLogger(__name__)


def process_video(input_path: str, config: PipelineConfig | None = None) -> VideoResult:
    if config is None:
        config = PipelineConfig()

    source_name = os.path.splitext(os.path.basename(input_path))[0]
    out_dir = os.path.join(config.output_dir, source_name)
    os.makedirs(out_dir, exist_ok=True)

    result = VideoResult(source_video=os.path.basename(input_path))

    # --- Stage 1: Normalize ---
    logger.info("=== Stage 1: Normalizing video ===")
    normalized_path = os.path.join(out_dir, "normalized.mp4")
    normalize_video(input_path, normalized_path, config.normalization)

    # Get frame dimensions
    cap = cv2.VideoCapture(normalized_path)
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    result.total_frames = total_frames

    # --- Stage 2: Face analysis ---
    logger.info("=== Stage 2: Analyzing frames ===")
    frame_data = analyze_frames(normalized_path, config.detection)
    result.frame_data = frame_data

    # --- Stage 3: Smooth pose, compute deltas, classify ---
    logger.info("=== Stage 3: Smoothing pose + classifying frames ===")
    smooth_pose(frame_data, window=5)
    compute_deltas(frame_data)
    classify_frames(frame_data, config.bad_frame, frame_w, frame_h)

    # --- Stage 4: Segment ---
    logger.info("=== Stage 4: Segmenting ===")
    exportable, dropped = split_into_segments(
        frame_data,
        config.bad_frame.min_segment_length,
        config.bad_frame.max_segment_length,
    )
    logger.info(
        "Found %d exportable segments, %d dropped (too short)",
        len(exportable), len(dropped),
    )

    all_segments = exportable + dropped

    # Check if video should be dropped entirely
    if not exportable:
        reason = _determine_video_drop_reason(frame_data)
        result.status = "dropped"
        result.drop_reason = reason
        result.segments = all_segments
        _write_drop_report(out_dir, result)
        if config.save_frame_log:
            save_frame_log(frame_data, os.path.join(out_dir, f"{source_name}_frame_log.csv"))
        if not config.keep_normalized:
            _cleanup(normalized_path)
        logger.info("Video DROPPED: %s", reason)
        return result

    # --- Stage 5: Ranking ---
    logger.info("=== Stage 5: Ranking segments ===")
    for seg in exportable:
        S = compute_output_size(seg, frame_w, frame_h)
        seg.output_size = S
        metrics = compute_segment_metrics(seg)
        seg.metrics = metrics
        seg.rank = rank_segment(metrics, config.ranking, S)
        logger.info(
            "  Segment %d: frames %d-%d (%d frames), size=%dx%d, rank=%s",
            seg.segment_id, seg.start_frame, seg.end_frame,
            seg.length, S, S, seg.rank,
        )

    # --- Stage 6: Exporting ---
    logger.info("=== Stage 6: Exporting segments ===")
    for seg in exportable:
        seg_filename = f"{source_name}_seg_{seg.segment_id:03d}.mp4"
        seg_path = os.path.join(out_dir, seg_filename)
        export_segment(
            seg, normalized_path, seg_path,
            frame_w, frame_h, seg.output_size, config.export,
            source_video_path=input_path,
        )
        seg.status = "exported"

        meta = seg.to_dict(
            source_video=os.path.basename(input_path),
            export_mode=config.export.mode,
        )
        json_path = os.path.join(out_dir, f"{source_name}_seg_{seg.segment_id:03d}.json")
        with open(json_path, "w") as f:
            json.dump(meta, f, indent=2)

        logger.info("  Exported segment %d -> %s", seg.segment_id, seg_filename)

    result.segments = all_segments
    result.status = "processed"

    # Write summary report
    _write_summary(out_dir, source_name, result, config)

    # Optional frame log
    if config.save_frame_log:
        save_frame_log(
            frame_data,
            os.path.join(out_dir, f"{source_name}_frame_log.csv"),
            segments=result.segments,
        )

    if not config.keep_normalized:
        _cleanup(normalized_path)
    logger.info("=== Done: %d segments exported ===", len(exportable))
    return result


def _determine_video_drop_reason(frame_data: list) -> str:
    """Determine the single primary drop reason for the whole video."""
    all_reasons: list[str] = []
    for fd in frame_data:
        all_reasons.extend(fd.bad_reasons)

    if not all_reasons:
        return "segment_too_short"

    # Count each reason
    reason_counts: dict[str, int] = {}
    for r in all_reasons:
        reason_counts[r] = reason_counts.get(r, 0) + 1

    # Pick by priority (most impactful)
    return pick_primary_reason(list(reason_counts.keys()))


def _write_drop_report(out_dir: str, result: VideoResult) -> None:
    report = {
        "source_video": result.source_video,
        "status": "dropped",
        "reason": result.drop_reason,
    }
    name = os.path.splitext(result.source_video)[0]
    path = os.path.join(out_dir, f"{name}_report.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=2)


def _write_summary(
    out_dir: str, source_name: str, result: VideoResult, config: PipelineConfig,
) -> None:
    exported = [s for s in result.segments if s.status == "exported"]
    dropped = [s for s in result.segments if s.status == "dropped"]
    report = {
        "source_video": result.source_video,
        "status": result.status,
        "total_frames": result.total_frames,
        "exported_segments": len(exported),
        "dropped_segments": len(dropped),
        "segments": [
            s.to_dict(result.source_video, config.export.mode)
            for s in result.segments
        ],
    }
    path = os.path.join(out_dir, f"{source_name}_report.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=2)


def _cleanup(normalized_path: str) -> None:
    """Remove the intermediate normalized video."""
    try:
        os.remove(normalized_path)
        logger.info("Cleaned up normalized video: %s", normalized_path)
    except OSError:
        pass
