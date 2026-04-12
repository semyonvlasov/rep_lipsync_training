from __future__ import annotations

import logging
import subprocess

import cv2
import numpy as np

from face_processing.config import ExportConfig
from face_processing.models import FrameData, Segment

logger = logging.getLogger(__name__)


def compute_output_size(
    segment: Segment,
    frame_w: int,
    frame_h: int,
) -> int:
    """Compute S = min(face_h) measured on roll-corrected landmarks.

    No video I/O — rotates landmarks mathematically using existing pose data.
    """
    min_face_h = float("inf")

    for fd in segment.frame_data:
        if fd.landmarks is None:
            continue

        lmks_px = fd.landmarks.copy()
        lmks_px[:, 0] *= frame_w
        lmks_px[:, 1] *= frame_h

        roll = fd.roll if fd.pose_valid else 0.0
        rotated_lmks = _rotate_landmarks(lmks_px[:, :2], -roll, frame_w, frame_h)

        ys = rotated_lmks[:, 1]
        face_h_rot = float(np.max(ys) - np.min(ys))
        if face_h_rot < min_face_h:
            min_face_h = face_h_rot

    S = max(1, int(min_face_h))
    if S % 2 != 0:
        S -= 1
    return S


def export_segment(
    segment: Segment,
    video_path: str,
    output_path: str,
    frame_w: int,
    frame_h: int,
    output_size: int,
    config: ExportConfig | None = None,
    source_video_path: str | None = None,
) -> str:
    """Export a segment as a square face video.

    For each frame:
    1. Rotate by -roll
    2. Compute crop around rotated face
    3. Resize to output_size x output_size (stretch_to_square)
    4. Pipe to ffmpeg
    """
    if config is None:
        config = ExportConfig()

    S = output_size
    logger.info(
        "Exporting segment %d: frames %d-%d, size %dx%d -> %s",
        segment.segment_id, segment.start_frame, segment.end_frame, S, S, output_path,
    )

    # Audio timing from segment boundaries
    start_sec = segment.start_frame / config.fps
    duration_sec = segment.length / config.fps

    cmd = [
        config.ffmpeg_bin, "-y",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{S}x{S}",
        "-r", str(config.fps),
        "-i", "pipe:0",
    ]
    if config.ffmpeg_threads and config.ffmpeg_threads > 0:
        cmd[1:1] = ["-threads", str(int(config.ffmpeg_threads))]
    if source_video_path:
        cmd += [
            "-ss", f"{start_sec:.4f}",
            "-t", f"{duration_sec:.4f}",
            "-i", source_video_path,
            "-c:v", config.codec,
            "-b:v", config.bitrate,
            "-pix_fmt", config.pixel_format,
            "-c:a", "aac", "-b:a", "128k",
            "-map", "0:v:0", "-map", "1:a:0?",
            "-shortest",
        ]
    else:
        cmd += [
            "-c:v", config.codec,
            "-b:v", config.bitrate,
            "-pix_fmt", config.pixel_format,
            "-an",
        ]
    cmd.append(output_path)
    proc = subprocess.Popen(
        cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, segment.start_frame)

    try:
        for fd in segment.frame_data:
            ret, frame_bgr = cap.read()
            if not ret or fd.landmarks is None:
                black = np.zeros((S, S, 3), dtype=np.uint8)
                proc.stdin.write(black.tobytes())
                continue

            roll = fd.roll if fd.pose_valid else 0.0
            cropped = _crop_face_rotated(
                frame_bgr, fd, roll, frame_w, frame_h, S, config.mode,
            )
            assert cropped.shape == (S, S, 3), f"Bad crop shape {cropped.shape}, expected ({S},{S},3)"
            proc.stdin.write(cropped.tobytes())
    except BrokenPipeError:
        pass
    finally:
        cap.release()
        try:
            proc.stdin.close()
        except OSError:
            pass

    try:
        proc.wait(timeout=config.ffmpeg_timeout)
    except subprocess.TimeoutExpired as exc:
        proc.kill()
        raise RuntimeError(
            f"ffmpeg export timed out after {config.ffmpeg_timeout}s for {output_path}"
        ) from exc
    stderr = proc.stderr.read() if proc.stderr else b""
    if proc.returncode != 0:
        logger.error("ffmpeg stderr: %s", stderr.decode())
        raise RuntimeError(
            f"ffmpeg export failed (code {proc.returncode}):\n{stderr.decode()}"
        )

    logger.info("Exported segment %d to %s", segment.segment_id, output_path)
    return output_path


def _crop_face_rotated(
    frame_bgr: np.ndarray,
    fd: FrameData,
    roll: float,
    frame_w: int,
    frame_h: int,
    S: int,
    mode: str,
) -> np.ndarray:
    """Rotate frame by -roll, crop face region, resize to SxS."""
    # Rotation center at frame center
    center = (frame_w / 2.0, frame_h / 2.0)
    M = cv2.getRotationMatrix2D(center, -roll, 1.0)
    rotated = cv2.warpAffine(
        frame_bgr, M, (frame_w, frame_h),
        borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0),
    )

    # Rotate landmarks
    lmks_px = fd.landmarks.copy()
    lmks_px[:, 0] *= frame_w
    lmks_px[:, 1] *= frame_h
    rotated_lmks = _rotate_landmarks(lmks_px[:, :2], -roll, frame_w, frame_h)

    # Face bounding box in rotated frame
    xs = rotated_lmks[:, 0]
    ys = rotated_lmks[:, 1]
    cx_rot = float(np.mean(xs))
    cy_rot = float(np.mean(ys))
    face_w_rot = float(np.max(xs) - np.min(xs))
    face_h_rot = float(np.max(ys) - np.min(ys))

    # Save crop geometry for restore
    fd.crop_cx_rot = cx_rot
    fd.crop_cy_rot = cy_rot
    fd.crop_w_rot = face_w_rot

    if mode == "stretch_to_square":
        crop = _extract_crop_stretch(rotated, cx_rot, cy_rot, face_w_rot, S, frame_w, frame_h)
    else:
        # pad_to_square
        crop = _extract_crop_pad(rotated, cx_rot, cy_rot, face_w_rot, face_h_rot, S, frame_w, frame_h)

    return crop


def _extract_crop_stretch(
    frame: np.ndarray,
    cx: float,
    cy: float,
    face_w: float,
    S: int,
    frame_w: int,
    frame_h: int,
) -> np.ndarray:
    """Crop a rectangle of (face_w x S) centered on face, then stretch to SxS."""
    half_w = face_w / 2.0
    half_h = S / 2.0

    x1 = int(round(cx - half_w))
    x2 = int(round(cx + half_w))
    y1 = int(round(cy - half_h))
    y2 = int(round(cy + half_h))

    # Clamp and pad
    crop = _safe_crop(frame, x1, y1, x2, y2, frame_w, frame_h)

    # Stretch to square
    if crop.shape[0] > 0 and crop.shape[1] > 0:
        crop = cv2.resize(crop, (S, S), interpolation=cv2.INTER_LINEAR)
    else:
        crop = np.zeros((S, S, 3), dtype=np.uint8)

    return crop


def _extract_crop_pad(
    frame: np.ndarray,
    cx: float,
    cy: float,
    face_w: float,
    face_h: float,
    S: int,
    frame_w: int,
    frame_h: int,
) -> np.ndarray:
    """Crop preserving aspect ratio, pad to SxS."""
    half_s = S / 2.0
    x1 = int(round(cx - half_s))
    x2 = int(round(cx + half_s))
    y1 = int(round(cy - half_s))
    y2 = int(round(cy + half_s))

    crop = _safe_crop(frame, x1, y1, x2, y2, frame_w, frame_h)

    if crop.shape[0] != S or crop.shape[1] != S:
        crop = cv2.resize(crop, (S, S), interpolation=cv2.INTER_LINEAR)

    return crop


def _safe_crop(
    frame: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    frame_w: int, frame_h: int,
) -> np.ndarray:
    """Crop with zero-padding for out-of-bounds regions."""
    h_crop = y2 - y1
    w_crop = x2 - x1
    if h_crop <= 0 or w_crop <= 0:
        return np.zeros((max(h_crop, 1), max(w_crop, 1), 3), dtype=np.uint8)

    result = np.zeros((h_crop, w_crop, 3), dtype=np.uint8)

    # Source region (clamped to frame bounds)
    sx1 = max(0, x1)
    sy1 = max(0, y1)
    sx2 = min(frame_w, x2)
    sy2 = min(frame_h, y2)

    if sx1 >= sx2 or sy1 >= sy2:
        return result

    # Destination offsets
    dx1 = sx1 - x1
    dy1 = sy1 - y1
    dx2 = dx1 + (sx2 - sx1)
    dy2 = dy1 + (sy2 - sy1)

    result[dy1:dy2, dx1:dx2] = frame[sy1:sy2, sx1:sx2]
    return result


def _rotate_landmarks(
    lmks_2d: np.ndarray,
    angle_deg: float,
    frame_w: int,
    frame_h: int,
) -> np.ndarray:
    """Rotate 2D landmark points by angle_deg around frame center."""
    cx = frame_w / 2.0
    cy = frame_h / 2.0
    M = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)

    ones = np.ones((lmks_2d.shape[0], 1), dtype=np.float64)
    pts = np.hstack([lmks_2d, ones])  # (N, 3)
    rotated = (M @ pts.T).T  # (N, 2)
    return rotated
