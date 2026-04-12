from __future__ import annotations

import csv
import logging

from face_processing.models import FrameData, Segment

logger = logging.getLogger(__name__)


def save_frame_log(
    frame_data: list[FrameData],
    output_path: str,
    segments: list[Segment] | None = None,
) -> None:
    """Save per-frame metrics to CSV for debugging and restore."""
    # Build frame_idx -> (segment_id, output_size) map
    seg_map: dict[int, tuple[int, int]] = {}
    if segments:
        for seg in segments:
            if seg.status == "exported" and seg.output_size is not None:
                for fd in seg.frame_data:
                    seg_map[fd.frame_idx] = (seg.segment_id, seg.output_size)

    fieldnames = [
        "frame_idx",
        "segment_id", "output_size",
        "num_faces",
        "face_detected",
        "confidence",
        "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
        "face_w", "face_h",
        "cx", "cy",
        "face_area_ratio",
        "yaw", "pitch", "roll",
        "pose_valid",
        "crop_cx_rot", "crop_cy_rot", "crop_w_rot",
        "delta_cx", "delta_cy",
        "delta_face_w", "delta_face_h",
        "delta_yaw", "delta_pitch", "delta_roll",
        "face_h_ratio", "face_w_ratio",
        "is_bad",
        "bad_reasons",
        "primary_reason",
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for fd in frame_data:
            seg_id, out_size = seg_map.get(fd.frame_idx, ("", ""))
            row = {
                "frame_idx": fd.frame_idx,
                "segment_id": seg_id,
                "output_size": out_size,
                "num_faces": fd.num_faces,
                "face_detected": fd.face_detected,
                "confidence": round(fd.confidence, 4),
                "bbox_x1": fd.bbox[0] if fd.bbox else "",
                "bbox_y1": fd.bbox[1] if fd.bbox else "",
                "bbox_x2": fd.bbox[2] if fd.bbox else "",
                "bbox_y2": fd.bbox[3] if fd.bbox else "",
                "face_w": round(fd.face_w, 2),
                "face_h": round(fd.face_h, 2),
                "cx": round(fd.cx, 2),
                "cy": round(fd.cy, 2),
                "face_area_ratio": round(fd.face_area_ratio, 6),
                "yaw": round(fd.yaw, 2),
                "pitch": round(fd.pitch, 2),
                "roll": round(fd.roll, 2),
                "pose_valid": fd.pose_valid,
                "crop_cx_rot": round(fd.crop_cx_rot, 2) if fd.crop_cx_rot is not None else "",
                "crop_cy_rot": round(fd.crop_cy_rot, 2) if fd.crop_cy_rot is not None else "",
                "crop_w_rot": round(fd.crop_w_rot, 2) if fd.crop_w_rot is not None else "",
                "delta_cx": round(fd.delta_cx, 2) if fd.delta_cx is not None else "",
                "delta_cy": round(fd.delta_cy, 2) if fd.delta_cy is not None else "",
                "delta_face_w": round(fd.delta_face_w, 2) if fd.delta_face_w is not None else "",
                "delta_face_h": round(fd.delta_face_h, 2) if fd.delta_face_h is not None else "",
                "delta_yaw": round(fd.delta_yaw, 2) if fd.delta_yaw is not None else "",
                "delta_pitch": round(fd.delta_pitch, 2) if fd.delta_pitch is not None else "",
                "delta_roll": round(fd.delta_roll, 2) if fd.delta_roll is not None else "",
                "face_h_ratio": round(fd.face_h_ratio, 4) if fd.face_h_ratio is not None else "",
                "face_w_ratio": round(fd.face_w_ratio, 4) if fd.face_w_ratio is not None else "",
                "is_bad": fd.is_bad,
                "bad_reasons": "|".join(fd.bad_reasons) if fd.bad_reasons else "",
                "primary_reason": fd.primary_reason or "",
            }
            writer.writerow(row)

    logger.info("Frame log saved to %s (%d rows)", output_path, len(frame_data))
