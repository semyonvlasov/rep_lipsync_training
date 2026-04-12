from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class FrameData:
    frame_idx: int
    num_faces: int = 0
    face_detected: bool = False
    confidence: float = 0.0

    # Bounding box in original frame coords
    bbox: tuple[int, int, int, int] | None = None  # (x1, y1, x2, y2)
    face_w: float = 0.0
    face_h: float = 0.0
    cx: float = 0.0
    cy: float = 0.0
    face_area_ratio: float = 0.0

    # 478 landmarks as (N, 3) normalized 0..1
    landmarks: np.ndarray | None = None

    # Head pose
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    pose_valid: bool = False

    # Optional MediaPipe transformation matrix
    transform_matrix: np.ndarray | None = None

    # Rotated crop geometry (filled during export)
    crop_cx_rot: float | None = None
    crop_cy_rot: float | None = None
    crop_w_rot: float | None = None

    # Inter-frame deltas (None for first frame or when previous had no face)
    delta_cx: float | None = None
    delta_cy: float | None = None
    delta_face_w: float | None = None
    delta_face_h: float | None = None
    delta_yaw: float | None = None
    delta_pitch: float | None = None
    delta_roll: float | None = None
    face_h_ratio: float | None = None
    face_w_ratio: float | None = None

    # Quality classification
    is_bad: bool = False
    bad_reasons: list[str] = field(default_factory=list)
    primary_reason: str | None = None


@dataclass
class SegmentMetrics:
    mean_abs_yaw: float = 0.0
    max_abs_yaw: float = 0.0
    mean_abs_pitch: float = 0.0
    max_abs_pitch: float = 0.0
    mean_abs_roll: float = 0.0
    max_abs_roll: float = 0.0
    mean_face_h: float = 0.0
    min_face_h: float = 0.0
    face_size_std_ratio: float = 0.0
    std_cx: float = 0.0
    std_cy: float = 0.0
    jump_ratio: float = 0.0
    missing_ratio: float = 0.0
    low_conf_ratio: float = 0.0

    def to_dict(self) -> dict:
        return {
            "mean_abs_yaw": round(self.mean_abs_yaw, 2),
            "mean_abs_pitch": round(self.mean_abs_pitch, 2),
            "mean_abs_roll": round(self.mean_abs_roll, 2),
            "max_abs_yaw": round(self.max_abs_yaw, 2),
            "max_abs_pitch": round(self.max_abs_pitch, 2),
            "max_abs_roll": round(self.max_abs_roll, 2),
            "mean_face_h": round(self.mean_face_h, 1),
            "min_face_h": round(self.min_face_h, 1),
            "face_size_std_ratio": round(self.face_size_std_ratio, 4),
            "std_cx": round(self.std_cx, 2),
            "std_cy": round(self.std_cy, 2),
            "jump_ratio": round(self.jump_ratio, 4),
            "missing_ratio": round(self.missing_ratio, 4),
            "low_conf_ratio": round(self.low_conf_ratio, 4),
        }


@dataclass
class Segment:
    segment_id: int
    start_frame: int  # inclusive
    end_frame: int  # exclusive
    length: int
    frame_data: list[FrameData] = field(default_factory=list)

    metrics: SegmentMetrics | None = None
    rank: str | None = None

    output_size: int | None = None
    status: str = "pending"  # "exported" / "dropped"
    drop_reason: str | None = None

    def to_dict(self, source_video: str = "", export_mode: str = "stretch_to_square") -> dict:
        d: dict[str, Any] = {
            "source_video": source_video,
            "segment_id": self.segment_id,
            "status": self.status,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "length_frames": self.length,
            "rank": self.rank,
            "drop_reason": self.drop_reason,
            "output_size": self.output_size,
            "export_mode": export_mode,
        }
        if self.metrics is not None:
            d["metrics"] = self.metrics.to_dict()
        return d


@dataclass
class VideoResult:
    source_video: str
    status: str = "processed"  # "processed" / "dropped"
    drop_reason: str | None = None
    total_frames: int = 0
    segments: list[Segment] = field(default_factory=list)
    frame_data: list[FrameData] = field(default_factory=list)
