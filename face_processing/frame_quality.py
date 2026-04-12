from __future__ import annotations

import math
from collections import deque

from face_processing.config import BadFrameThresholds
from face_processing.models import FrameData

# Priority order for drop reasons (lowest index = highest priority)
DROP_REASON_PRIORITY = [
    "low_face_confidence",
    "head_pose_extreme",
    "face_missing_or_tracking_lost",
    "frame_jumps",
    "strong_face_zoom_in_out",
    "excessive_face_motion",
    "face_too_small",
    "multiple_faces",
    "segment_too_short",
]


def smooth_pose(frame_data: list[FrameData], window: int = 5) -> None:
    """Smooth yaw/pitch/roll with a centered moving average (mutates in place)."""
    half = window // 2
    n = len(frame_data)

    raw_yaw = [fd.yaw for fd in frame_data]
    raw_pitch = [fd.pitch for fd in frame_data]
    raw_roll = [fd.roll for fd in frame_data]
    valid = [fd.pose_valid and fd.face_detected and fd.num_faces == 1 for fd in frame_data]

    for i in range(n):
        if not valid[i]:
            continue
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        yaws, pitches, rolls = [], [], []
        for j in range(lo, hi):
            if valid[j]:
                yaws.append(raw_yaw[j])
                pitches.append(raw_pitch[j])
                rolls.append(raw_roll[j])
        if yaws:
            frame_data[i].yaw = sum(yaws) / len(yaws)
            frame_data[i].pitch = sum(pitches) / len(pitches)
            frame_data[i].roll = sum(rolls) / len(rolls)


def compute_deltas(frame_data: list[FrameData]) -> None:
    """Fill inter-frame delta fields on each FrameData (mutates in place)."""
    for i in range(1, len(frame_data)):
        cur = frame_data[i]
        prev = frame_data[i - 1]

        if not prev.face_detected or not cur.face_detected:
            continue
        if prev.num_faces != 1 or cur.num_faces != 1:
            continue

        cur.delta_cx = cur.cx - prev.cx
        cur.delta_cy = cur.cy - prev.cy
        cur.delta_face_w = cur.face_w - prev.face_w
        cur.delta_face_h = cur.face_h - prev.face_h
        cur.delta_yaw = cur.yaw - prev.yaw
        cur.delta_pitch = cur.pitch - prev.pitch
        cur.delta_roll = cur.roll - prev.roll

        if prev.face_h > 0:
            cur.face_h_ratio = cur.face_h / prev.face_h
        if prev.face_w > 0:
            cur.face_w_ratio = cur.face_w / prev.face_w


def classify_frames(
    frame_data: list[FrameData],
    thresholds: BadFrameThresholds,
    frame_w: int,
    frame_h: int,
) -> None:
    """Classify each frame as good or bad (mutates in place)."""
    # Pre-compute excessive motion using rolling window
    _mark_excessive_motion(frame_data, thresholds)

    for fd in frame_data:
        reasons: list[str] = []

        # --- face missing / tracking lost ---
        if not fd.face_detected or (fd.num_faces == 1 and not fd.pose_valid):
            reasons.append("face_missing_or_tracking_lost")

        # --- multiple faces ---
        if fd.num_faces > 1:
            reasons.append("multiple_faces")

        # Only check further metrics if a single face was detected
        if fd.face_detected and fd.num_faces == 1:
            # --- low confidence ---
            if fd.confidence < thresholds.min_confidence:
                reasons.append("low_face_confidence")

            # --- face too small ---
            if fd.face_h < thresholds.min_face_h:
                reasons.append("face_too_small")

            # --- extreme head pose ---
            if fd.pose_valid:
                if (
                    abs(fd.yaw) > thresholds.max_abs_yaw
                    or abs(fd.pitch) > thresholds.max_abs_pitch
                    or abs(fd.roll) > thresholds.max_abs_roll
                ):
                    reasons.append("head_pose_extreme")

            # --- frame jumps ---
            if fd.delta_cx is not None:
                jump = False
                if abs(fd.delta_cx) > thresholds.max_delta_cx_ratio * frame_w:
                    jump = True
                if abs(fd.delta_cy) > thresholds.max_delta_cy_ratio * frame_h:
                    jump = True
                if fd.delta_yaw is not None and abs(fd.delta_yaw) > thresholds.max_delta_yaw:
                    jump = True
                if fd.delta_pitch is not None and abs(fd.delta_pitch) > thresholds.max_delta_pitch:
                    jump = True
                if fd.delta_roll is not None and abs(fd.delta_roll) > thresholds.max_delta_roll:
                    jump = True
                if jump:
                    reasons.append("frame_jumps")

            # --- strong zoom in/out ---
            if fd.face_h_ratio is not None:
                if abs(fd.face_h_ratio - 1.0) > thresholds.max_face_h_ratio_deviation:
                    reasons.append("strong_face_zoom_in_out")

        # --- excessive face motion (already marked) ---
        if getattr(fd, "_excessive_motion", False):
            reasons.append("excessive_face_motion")

        # Deduplicate
        reasons = list(dict.fromkeys(reasons))

        fd.bad_reasons = reasons
        fd.is_bad = len(reasons) > 0
        fd.primary_reason = _pick_primary(reasons) if reasons else None


def _mark_excessive_motion(
    frame_data: list[FrameData],
    thresholds: BadFrameThresholds,
) -> None:
    """Mark frames with excessive cumulative face motion in a rolling window."""
    window_size = thresholds.motion_window_frames
    max_ratio = thresholds.max_cumulative_motion_ratio

    # Compute per-frame displacement
    displacements: list[float] = [0.0] * len(frame_data)
    for i in range(1, len(frame_data)):
        cur = frame_data[i]
        prev = frame_data[i - 1]
        if (
            cur.face_detected
            and prev.face_detected
            and cur.num_faces == 1
            and prev.num_faces == 1
        ):
            dx = cur.cx - prev.cx
            dy = cur.cy - prev.cy
            displacements[i] = math.sqrt(dx * dx + dy * dy)

    # Rolling sum over window
    window: deque[float] = deque()
    running_sum = 0.0

    for i, fd in enumerate(frame_data):
        window.append(displacements[i])
        running_sum += displacements[i]
        if len(window) > window_size:
            running_sum -= window.popleft()

        # Compare cumulative motion to face size
        if fd.face_detected and fd.face_h > 0:
            ratio = running_sum / fd.face_h
            if ratio > max_ratio:
                fd._excessive_motion = True  # type: ignore[attr-defined]


def _pick_primary(reasons: list[str]) -> str:
    """Pick the highest-priority reason from the list."""
    best_idx = len(DROP_REASON_PRIORITY)
    best = reasons[0]
    for r in reasons:
        try:
            idx = DROP_REASON_PRIORITY.index(r)
        except ValueError:
            idx = len(DROP_REASON_PRIORITY)
        if idx < best_idx:
            best_idx = idx
            best = r
    return best


def pick_primary_reason(reasons: list[str]) -> str:
    """Public helper — pick highest-priority drop reason."""
    return _pick_primary(reasons)
