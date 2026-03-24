#!/usr/bin/env python3
"""
Preprocess HDTF videos into training-ready face crops + mel spectrograms.

For each video:
  1. Detect face bbox (one per video, assumed static speaker)
  2. Extract face crops at target resolution
  3. Extract audio → compute mel spectrogram
  4. Save as numpy arrays for fast loading during training

Output structure:
  output_dir/
    speaker_001/
      frames.npy      # (N, H, W, 3) uint8 BGR face crops
      mel.npy          # (80, T) float32 mel spectrogram
      bbox.json        # face bbox in original video coords
    speaker_002/
      ...

Usage:
    python scripts/preprocess_dataset.py --input data/hdtf/clips --output data/hdtf/processed --size 256

Detector guidance:
  - Local / CPU-only iteration: prefer the default OpenCV detector. It is much
    lighter and works well with multi-worker preprocessing.
  - Remote CUDA box: SFD is the higher-quality detector and is worth trying on
    GPU, but it should usually run in a single worker unless we build a shared
    GPU detector service.
"""

import argparse
import json
import os
import shutil
import sys
import threading
import time

import cv2
import numpy as np

TRAINING_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(TRAINING_ROOT)
sys.path.insert(0, TRAINING_ROOT)
sys.path.insert(0, REPO_ROOT)
from data.audio import AudioProcessor

_DETECTOR_LOCAL = threading.local()


def resolve_detector_device(detector_backend, detector_device):
    # For SFD we prefer an accelerator when it is really available; otherwise
    # we fall back to CPU. On local development machines this commonly resolves
    # to CPU, while on remote CUDA boxes it should resolve to cuda.
    if detector_device != "auto":
        return detector_device
    if detector_backend != "sfd":
        return "cpu"
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is not None and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def resolve_resize_device(resize_device):
    if resize_device != "auto":
        return resize_device
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is not None and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def sample_is_complete(speaker_dir):
    frames_path = os.path.join(speaker_dir, "frames.npy")
    mel_path = os.path.join(speaker_dir, "mel.npy")
    meta_path = os.path.join(speaker_dir, "bbox.json")
    if not (
        os.path.exists(frames_path)
        and os.path.exists(mel_path)
        and os.path.exists(meta_path)
    ):
        return False
    try:
        with open(meta_path) as f:
            json.load(f)
    except Exception:
        return False
    return True


class BaseFaceDetector:
    def detect_batch(self, frames, min_size=60):
        return [self.detect(frame, min_size=min_size) for frame in frames]


class OpenCVHaarFaceDetector(BaseFaceDetector):
    def __init__(self):
        self.cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

    def detect(self, frame, min_size=60):
        return detect_face_on_frame(self.cascade, frame, min_size=min_size)


class SFDFaceDetector(BaseFaceDetector):
    def __init__(self, device="cpu", batch_size=4):
        official_syncnet_root = os.path.join(REPO_ROOT, "models", "official_syncnet")
        if official_syncnet_root not in sys.path:
            sys.path.insert(0, official_syncnet_root)
        from face_detection import FaceAlignment, LandmarksType

        self.device = device
        self.batch_size = max(int(batch_size), 1)
        self.face_alignment = FaceAlignment(
            LandmarksType._2D,
            device=device,
            flip_input=False,
            face_detector="sfd",
        )

    def _detect_chunk(self, frames):
        batch = np.stack(frames, axis=0)
        try:
            return self.face_alignment.get_detections_for_batch(batch)
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower() or len(frames) <= 1:
                raise
            try:
                import torch

                if self.device == "cuda":
                    torch.cuda.empty_cache()
            except Exception:
                pass
            mid = max(len(frames) // 2, 1)
            left = self._detect_chunk(frames[:mid])
            right = self._detect_chunk(frames[mid:])
            return left + right
        finally:
            del batch
            try:
                import torch

                if self.device == "cuda":
                    torch.cuda.empty_cache()
            except Exception:
                pass

    def detect_batch(self, frames, min_size=60):
        results = [(None, "no_face")] * len(frames)
        valid_indices = []
        valid_frames = []

        for idx, frame in enumerate(frames):
            if frame.mean() < 20:
                results[idx] = (None, "dark_frame")
                continue
            valid_indices.append(idx)
            valid_frames.append(frame)

        if not valid_frames:
            return results

        for start in range(0, len(valid_frames), self.batch_size):
            chunk_frames = valid_frames[start : start + self.batch_size]
            chunk_indices = valid_indices[start : start + self.batch_size]
            detections = self._detect_chunk(chunk_frames)
            for idx, det in zip(chunk_indices, detections):
                if det is None:
                    results[idx] = (None, "no_face")
                    continue
                x1, y1, x2, y2 = [int(v) for v in det]
                w = max(x2 - x1, 0)
                h = max(y2 - y1, 0)
                if max(w, h) < min_size:
                    results[idx] = (None, "small_face")
                    continue
                bbox = expand_face_bbox((x1, y1, w, h), frames[idx].shape)
                results[idx] = (bbox, "ok")
        return results


def get_face_detector(detector_backend="opencv", detector_device="auto", detector_batch_size=4):
    resolved_device = resolve_detector_device(detector_backend, detector_device)
    key = (detector_backend, resolved_device, int(detector_batch_size))
    cache = getattr(_DETECTOR_LOCAL, "cache", None)
    if cache is None:
        cache = {}
        _DETECTOR_LOCAL.cache = cache
    if key in cache:
        return cache[key]

    if detector_backend == "opencv":
        detector = OpenCVHaarFaceDetector()
    elif detector_backend == "sfd":
        detector = SFDFaceDetector(device=resolved_device, batch_size=detector_batch_size)
    else:
        raise ValueError(f"Unsupported detector backend: {detector_backend}")

    cache[key] = detector
    return detector


def expand_face_bbox(face, frame_shape, scale=0.7):
    x, y, w, h = face
    cx = x + w / 2.0
    cy = y + h / 2.0
    half = max(w, h) * scale
    y1 = max(0, int(round(cy - half)))
    x1 = max(0, int(round(cx - half)))
    y2 = min(frame_shape[0], int(round(cy + half)))
    x2 = min(frame_shape[1], int(round(cx + half)))
    return (y1, y2, x1, x2)


def bbox_center_and_size(bbox):
    y1, y2, x1, x2 = bbox
    center = np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=np.float32)
    size = float(max(y2 - y1, x2 - x1))
    return center, max(size, 1.0)


def bbox_edge_margin_ratio(bbox, frame_shape):
    y1, y2, x1, x2 = bbox
    face_size = max(float(max(y2 - y1, x2 - x1)), 1.0)
    margin = min(x1, y1, frame_shape[1] - x2, frame_shape[0] - y2)
    return float(margin) / face_size


def moving_average(values, window):
    if window <= 1 or len(values) <= 2:
        return values
    if window % 2 == 0:
        window += 1
    pad = window // 2
    kernel = np.ones(window, dtype=np.float32) / float(window)
    padded = np.pad(values, (pad, pad), mode="edge")
    return np.convolve(padded, kernel, mode="valid")[: len(values)]


def smooth_bbox_track(track, window):
    if window <= 1 or len(track) <= 2:
        return track
    smoothed = np.zeros_like(track, dtype=np.float32)
    for col in range(track.shape[1]):
        smoothed[:, col] = moving_average(track[:, col], window)
    return smoothed


def detect_face_on_frame(cascade, frame, min_size=60):
    if frame.mean() < 20:
        return None, "dark_frame"
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, 1.05, 3, minSize=(min_size, min_size))
    if len(faces) == 0:
        return None, "no_face"
    face = max(faces, key=lambda f: f[2] * f[3])
    return expand_face_bbox(face, frame.shape), "ok"


def resize_face_crops(frames, bboxes, img_size, resize_device="cpu", batch_size=32):
    if not frames:
        return np.empty((0, img_size, img_size, 3), dtype=np.uint8)

    resolved_device = resolve_resize_device(resize_device)
    if resolved_device == "cpu":
        crops = []
        for frame, bbox in zip(frames, bboxes):
            y1, y2, x1, x2 = [int(v) for v in bbox]
            crop = frame[y1:y2, x1:x2]
            crop = cv2.resize(crop, (img_size, img_size))
            crops.append(crop)
        return np.array(crops, dtype=np.uint8)

    try:
        import torch
        import torch.nn.functional as F

        resized_chunks = []
        out_h = int(img_size)
        out_w = int(img_size)
        batch_size = max(int(batch_size), 1)
        with torch.no_grad():
            for start in range(0, len(frames), batch_size):
                frame_chunk = frames[start : start + batch_size]
                bbox_chunk = bboxes[start : start + batch_size]
                frame_tensor = torch.from_numpy(
                    np.stack(frame_chunk, axis=0).astype(np.float32, copy=False)
                )
                frame_tensor = frame_tensor.permute(0, 3, 1, 2).to(
                    device=resolved_device,
                    dtype=torch.float32,
                )

                chunk_boxes = np.asarray(bbox_chunk, dtype=np.float32)
                y1 = torch.from_numpy(chunk_boxes[:, 0]).to(resolved_device)
                y2 = torch.from_numpy(chunk_boxes[:, 1]).to(resolved_device)
                x1 = torch.from_numpy(chunk_boxes[:, 2]).to(resolved_device)
                x2 = torch.from_numpy(chunk_boxes[:, 3]).to(resolved_device)
                crop_h = torch.clamp(y2 - y1, min=1.0)
                crop_w = torch.clamp(x2 - x1, min=1.0)

                grid_x_base = torch.arange(out_w, device=resolved_device, dtype=torch.float32)
                grid_y_base = torch.arange(out_h, device=resolved_device, dtype=torch.float32)
                src_x = x1[:, None] + ((grid_x_base[None, :] + 0.5) * crop_w[:, None] / out_w) - 0.5
                src_y = y1[:, None] + ((grid_y_base[None, :] + 0.5) * crop_h[:, None] / out_h) - 0.5

                width = float(frame_tensor.shape[3])
                height = float(frame_tensor.shape[2])
                grid_x = (2.0 * (src_x + 0.5) / width) - 1.0
                grid_y = (2.0 * (src_y + 0.5) / height) - 1.0
                grid_y = grid_y[:, :, None].expand(-1, -1, out_w)
                grid_x = grid_x[:, None, :].expand(-1, out_h, -1)
                grid = torch.stack((grid_x, grid_y), dim=-1)

                out = F.grid_sample(
                    frame_tensor,
                    grid,
                    mode="bilinear",
                    padding_mode="zeros",
                    align_corners=False,
                )
                out = (
                    out.permute(0, 2, 3, 1)
                    .round()
                    .clamp(0, 255)
                    .to(dtype=torch.uint8)
                    .cpu()
                    .numpy()
                )
                resized_chunks.append(out)

                del frame_tensor, grid, out
                try:
                    if resolved_device == "cuda":
                        torch.cuda.empty_cache()
                except Exception:
                    pass
        return np.concatenate(resized_chunks, axis=0).astype(np.uint8, copy=False)
    except Exception:
        crops = []
        for frame, bbox in zip(frames, bboxes):
            y1, y2, x1, x2 = [int(v) for v in bbox]
            crop = frame[y1:y2, x1:x2]
            crop = cv2.resize(crop, (img_size, img_size))
            crops.append(crop)
        return np.array(crops, dtype=np.uint8)


def choose_detection_span(valid_records, max_gap_frames):
    if not valid_records:
        return []
    spans = [[valid_records[0]]]
    for record in valid_records[1:]:
        if record["frame_idx"] - spans[-1][-1]["frame_idx"] > max_gap_frames:
            spans.append([record])
        else:
            spans[-1].append(record)
    return max(
        spans,
        key=lambda span: (
            span[-1]["frame_idx"] - span[0]["frame_idx"],
            len(span),
        ),
    )


def build_face_track(
    frames,
    detect_every=10,
    min_size=60,
    smooth_window=9,
    detector_backend="opencv",
    detector_device="auto",
    detector_batch_size=4,
    min_valid_detections=3,
    min_detection_coverage=0.45,
    min_edge_margin_ratio=0.06,
    max_center_jump_ratio=0.55,
    max_size_jump_ratio=0.40,
    span_pad_frames=None,
    boundary_trim_frames=None,
    boundary_trim_min_kept_frames=180,
    min_clean_frames=100,
    min_kept_ratio=0.18,
):
    detector = get_face_detector(detector_backend, detector_device, detector_batch_size)
    if len(frames) == 0:
        return {"ok": False, "status": "empty_video"}

    sample_indices = list(range(0, len(frames), detect_every))
    if sample_indices[-1] != len(frames) - 1:
        sample_indices.append(len(frames) - 1)

    sampled_frames = [frames[fi] for fi in sample_indices]
    detection_results = detector.detect_batch(sampled_frames, min_size=min_size)

    detections = []
    valid_records = []
    for fi, (bbox, reason) in zip(sample_indices, detection_results):
        edge_ratio = bbox_edge_margin_ratio(bbox, frames[fi].shape) if bbox is not None else 0.0
        record = {
            "frame_idx": fi,
            "bbox": bbox,
            "reason": reason,
            "edge_margin_ratio": edge_ratio,
        }
        detections.append(record)
        if bbox is not None and edge_ratio >= min_edge_margin_ratio:
            valid_records.append(record)

    if len(valid_records) < min_valid_detections:
        return {
            "ok": False,
            "status": f"no_stable_face ({len(valid_records)} valid detections)",
            "detections": detections,
        }

    max_gap_frames = max(detect_every * 3, 1)
    span_records = choose_detection_span(valid_records, max_gap_frames=max_gap_frames)
    span_start = span_records[0]["frame_idx"]
    span_end = span_records[-1]["frame_idx"]
    if span_pad_frames is None:
        span_pad_frames = detect_every
    if boundary_trim_frames is None:
        boundary_trim_frames = detect_every * 2

    raw_trim_start = max(0, span_start - max(span_pad_frames, 0))
    raw_trim_end = min(len(frames) - 1, span_end + max(span_pad_frames - 1, 0))
    raw_kept_frames = raw_trim_end - raw_trim_start + 1
    extra_boundary_trim = 0
    if raw_kept_frames >= boundary_trim_min_kept_frames:
        max_safe_trim = max((raw_kept_frames - min_clean_frames) // 2, 0)
        extra_boundary_trim = min(boundary_trim_frames, max_safe_trim)

    trim_start = min(raw_trim_start + extra_boundary_trim, raw_trim_end)
    trim_end = max(trim_start, raw_trim_end - extra_boundary_trim)

    span_sample_count = sum(1 for fi in sample_indices if trim_start <= fi <= trim_end)
    coverage = len(span_records) / max(span_sample_count, 1)

    valid_indices = np.array([r["frame_idx"] for r in span_records], dtype=np.int32)
    valid_boxes = np.array([r["bbox"] for r in span_records], dtype=np.float32)

    frame_indices = np.arange(trim_start, trim_end + 1, dtype=np.int32)
    track = np.stack(
        [
            np.interp(frame_indices, valid_indices, valid_boxes[:, 0]),
            np.interp(frame_indices, valid_indices, valid_boxes[:, 1]),
            np.interp(frame_indices, valid_indices, valid_boxes[:, 2]),
            np.interp(frame_indices, valid_indices, valid_boxes[:, 3]),
        ],
        axis=1,
    )
    track = smooth_bbox_track(track, smooth_window)

    clipped_track = np.zeros_like(track, dtype=np.int32)
    edge_ratios = []
    for i, bbox in enumerate(track):
        y1, y2, x1, x2 = bbox.tolist()
        y1 = max(0, int(round(y1)))
        x1 = max(0, int(round(x1)))
        y2 = min(frames[frame_indices[i]].shape[0], int(round(y2)))
        x2 = min(frames[frame_indices[i]].shape[1], int(round(x2)))
        if y2 <= y1:
            y2 = min(frames[frame_indices[i]].shape[0], y1 + 1)
        if x2 <= x1:
            x2 = min(frames[frame_indices[i]].shape[1], x1 + 1)
        clipped_track[i] = (y1, y2, x1, x2)
        edge_ratios.append(bbox_edge_margin_ratio(clipped_track[i], frames[frame_indices[i]].shape))

    centers = []
    sizes = []
    for record in span_records:
        center, size = bbox_center_and_size(record["bbox"])
        centers.append(center)
        sizes.append(size)
    centers = np.stack(centers, axis=0)
    sizes = np.array(sizes, dtype=np.float32)

    if len(centers) > 1:
        center_jumps = np.linalg.norm(np.diff(centers, axis=0), axis=1) / np.maximum(sizes[:-1], 1.0)
        size_jumps = np.abs(np.diff(sizes)) / np.maximum(sizes[:-1], 1.0)
        max_center_jump = float(center_jumps.max(initial=0.0))
        max_size_jump = float(size_jumps.max(initial=0.0))
    else:
        max_center_jump = 0.0
        max_size_jump = 0.0

    bad_reasons = []
    if coverage < min_detection_coverage:
        bad_reasons.append("low_detection_coverage")
    if np.mean(np.array(edge_ratios) < min_edge_margin_ratio) > 0.10:
        bad_reasons.append("face_near_edge")
    if max_center_jump > max_center_jump_ratio:
        bad_reasons.append("bbox_center_jump")
    if max_size_jump > max_size_jump_ratio:
        bad_reasons.append("bbox_scale_jump")
    if len(frame_indices) < 25:
        bad_reasons.append("trimmed_too_short")
    if len(frame_indices) < min_clean_frames:
        bad_reasons.append("short_segment")
    kept_ratio = len(frame_indices) / max(len(frames), 1)
    if kept_ratio < min_kept_ratio:
        bad_reasons.append("heavily_trimmed")

    quality = {
        "sampled_frames": len(sample_indices),
        "valid_detections": len(span_records),
        "detection_coverage": float(coverage),
        "raw_trim_start": int(raw_trim_start),
        "raw_trim_end": int(raw_trim_end),
        "trim_start": int(trim_start),
        "trim_end": int(trim_end),
        "kept_frames": int(len(frame_indices)),
        "kept_ratio": float(kept_ratio),
        "boundary_trim_frames": int(extra_boundary_trim),
        "min_edge_margin_ratio": float(min(edge_ratios) if edge_ratios else 0.0),
        "max_center_jump_ratio": float(max_center_jump),
        "max_size_jump_ratio": float(max_size_jump),
    }

    return {
        "ok": len(frame_indices) >= 25,
        "status": "ok" if len(frame_indices) >= 25 else "trimmed_too_short",
        "trim_start": int(trim_start),
        "trim_end": int(trim_end),
        "frame_indices": frame_indices,
        "bboxes": clipped_track,
        "detections": detections,
        "quality": quality,
        "bad_sample": bool(bad_reasons),
        "bad_reasons": bad_reasons,
    }


def save_preview_contact(
    name,
    frames,
    bboxes,
    crops,
    frame_indices,
    output_path,
    bad_sample,
    bad_reasons,
    quality=None,
):
    if len(crops) == 0:
        return

    sample_positions = np.linspace(0, len(crops) - 1, num=min(5, len(crops)), dtype=int)
    source_tiles = []
    crop_tiles = []
    border_color = (0, 0, 255) if bad_sample else (0, 200, 0)

    for pos in sample_positions:
        src = frames[pos].copy()
        y1, y2, x1, x2 = [int(v) for v in bboxes[pos]]
        cv2.rectangle(src, (x1, y1), (x2, y2), border_color, 2)
        src = cv2.resize(src, (256, 256))
        crop = cv2.resize(crops[pos], (256, 256))
        label = f"{name} #{int(frame_indices[pos])}"
        for tile in (src, crop):
            cv2.putText(tile, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(tile, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
        source_tiles.append(src)
        crop_tiles.append(crop)

    top = cv2.hconcat(source_tiles)
    bottom = cv2.hconcat(crop_tiles)
    header = np.full((66, top.shape[1], 3), 245, dtype=np.uint8)
    status = "BAD" if bad_sample else "OK"
    reason_text = ", ".join(bad_reasons[:3]) if bad_reasons else "clean_track"
    text = f"{name} | {status} | {reason_text}"
    cv2.putText(header, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2, cv2.LINE_AA)
    if quality:
        meta = (
            f"trim={quality.get('trim_start', 0)}:{quality.get('trim_end', 0)} | "
            f"kept={quality.get('kept_frames', 0)} | "
            f"ratio={quality.get('kept_ratio', 0.0):.3f} | "
            f"cov={quality.get('detection_coverage', 0.0):.3f}"
        )
        cv2.putText(header, meta, (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40, 40, 40), 2, cv2.LINE_AA)
    contact = cv2.vconcat([header, top, bottom])
    cv2.imwrite(output_path, contact, [int(cv2.IMWRITE_JPEG_QUALITY), 92])


def resolve_ffmpeg_bin(cli_value=None):
    """Prefer a real system ffmpeg over minimal conda builds on some images."""
    candidates = [
        cli_value,
        os.environ.get("FFMPEG_BIN"),
        "/usr/bin/ffmpeg",
        shutil.which("ffmpeg"),
    ]
    for candidate in candidates:
        if candidate and os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return "ffmpeg"


def extract_audio_wav(
    video_path,
    output_wav,
    sample_rate=16000,
    start_time=0.0,
    duration=None,
    ffmpeg_bin=None,
    ffmpeg_threads=0,
):
    """Extract mono 16kHz audio from video via ffmpeg."""
    import subprocess
    ffmpeg_bin = resolve_ffmpeg_bin(ffmpeg_bin)
    try:
        cmd = [ffmpeg_bin, "-y"]
        if start_time > 0:
            cmd.extend(["-ss", f"{start_time:.3f}"])
        cmd.extend(["-i", video_path])
        if duration is not None and duration > 0:
            cmd.extend(["-t", f"{duration:.3f}"])
        if ffmpeg_threads and ffmpeg_threads > 0:
            cmd.extend(["-threads", str(int(ffmpeg_threads))])
        cmd.extend(
            ["-ar", str(sample_rate), "-ac", "1", "-f", "wav", output_wav]
        )
        subprocess.run(
            cmd,
            check=True, capture_output=True, timeout=60,
        )
        return True
    except Exception:
        return False


def process_video(
    video_path,
    output_dir,
    img_size,
    audio_proc,
    max_frames=750,
    detect_every=10,
    smooth_window=9,
    save_preview=True,
    overwrite=False,
    ffmpeg_bin=None,
    ffmpeg_threads=0,
    detector_backend="opencv",
    detector_device="auto",
    detector_batch_size=4,
    resize_device="cpu",
):
    """Process a single video into face crops + mel."""
    t0 = time.time()
    name = os.path.splitext(os.path.basename(video_path))[0]
    speaker_dir = os.path.join(output_dir, name)

    # Skip if already processed
    if not overwrite and sample_is_complete(speaker_dir):
        return name, "skip", 0

    # Read video frames
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    frames = []
    while len(frames) < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()

    if len(frames) < 25:
        return name, f"too_short ({len(frames)} frames)", time.time() - t0

    track = build_face_track(
        frames,
        detect_every=detect_every,
        smooth_window=smooth_window,
        detector_backend=detector_backend,
        detector_device=detector_device,
        detector_batch_size=detector_batch_size,
    )
    if not track["ok"]:
        return name, track["status"], time.time() - t0

    trim_start = track["trim_start"]
    trim_end = track["trim_end"]
    trimmed_frames = frames[trim_start : trim_end + 1]
    trimmed_bboxes = track["bboxes"]
    trimmed_indices = track["frame_indices"]

    if trim_start > 0:
        print(f"    Trimmed start: {trim_start} frames")
    if trim_end < len(frames) - 1:
        print(f"    Trimmed end: {len(frames) - trim_end - 1} frames")

    crops_arr = resize_face_crops(
        trimmed_frames,
        trimmed_bboxes,
        img_size,
        resize_device=resize_device,
    )

    # Extract audio and compute mel
    os.makedirs(speaker_dir, exist_ok=True)
    wav_path = os.path.join(speaker_dir, "audio.wav")
    trim_duration = len(trimmed_frames) / float(fps)
    if not extract_audio_wav(
        video_path,
        wav_path,
        start_time=trim_start / float(fps),
        duration=trim_duration,
        ffmpeg_bin=ffmpeg_bin,
        ffmpeg_threads=ffmpeg_threads,
    ):
        return name, "audio_fail", time.time() - t0

    wav = audio_proc.load_wav(wav_path)
    mel = audio_proc.melspectrogram(wav)

    # Save
    np.save(os.path.join(speaker_dir, "frames.npy"), crops_arr)
    np.save(os.path.join(speaker_dir, "mel.npy"), mel)
    preview_path = os.path.join(speaker_dir, "preview_contact.jpg")
    if save_preview:
        save_preview_contact(
            name=name,
            frames=trimmed_frames,
            bboxes=trimmed_bboxes,
            crops=crops_arr,
            frame_indices=trimmed_indices,
            output_path=preview_path,
            bad_sample=track["bad_sample"],
            bad_reasons=track["bad_reasons"],
            quality=track["quality"],
        )
    meta_path = os.path.join(speaker_dir, "bbox.json")
    meta_tmp_path = meta_path + ".tmp"
    with open(meta_tmp_path, "w") as f:
        json.dump(
            {
                "bbox": [int(v) for v in trimmed_bboxes[len(trimmed_bboxes) // 2]],
                "fps": float(fps),
                "n_frames": int(crops_arr.shape[0]),
                "mel_frames": mel.shape[1],
                "img_size": img_size,
                "detector_backend": detector_backend,
                "detector_device": resolve_detector_device(detector_backend, detector_device),
                "resize_device": resolve_resize_device(resize_device),
                "trim_start_frame": int(trim_start),
                "trim_end_frame": int(trim_end),
                "bad_sample": bool(track["bad_sample"]),
                "bad_reasons": track["bad_reasons"],
                "quality": track["quality"],
                "preview_contact": os.path.basename(preview_path) if save_preview else None,
            },
            f,
        )
    os.replace(meta_tmp_path, meta_path)

    # Remove temp wav
    os.remove(wav_path)

    elapsed = time.time() - t0
    prefix = "ok_bad" if track["bad_sample"] else "ok"
    details = f"{int(crops_arr.shape[0])} frames, mel={mel.shape[1]}, trim={trim_start}:{trim_end}"
    if track["bad_sample"]:
        details += f", reasons={','.join(track['bad_reasons'])}"
    return name, f"{prefix} ({details})", elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Directory with video files")
    parser.add_argument("--output", required=True, help="Output directory for processed data")
    parser.add_argument("--size", type=int, default=256, help="Face crop size")
    parser.add_argument("--max-frames", type=int, default=750, help="Max frames per video (30s@25fps)")
    parser.add_argument("--max-videos", type=int, default=0, help="Max videos to process (0=all)")
    parser.add_argument("--detect-every", type=int, default=10, help="Run face detection every N frames")
    parser.add_argument("--smooth-window", type=int, default=9, help="Temporal smoothing window for bbox track")
    parser.add_argument("--no-preview", action="store_true", help="Do not save preview contact sheets")
    parser.add_argument("--overwrite", action="store_true", help="Reprocess videos even if frames.npy exists")
    parser.add_argument(
        "--detector-backend",
        choices=["opencv", "sfd"],
        default="opencv",
        help="Face detector backend",
    )
    parser.add_argument(
        "--detector-device",
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
        help="Device for detector backend",
    )
    parser.add_argument(
        "--detector-batch-size",
        type=int,
        default=4,
        help="Batch size for SFD face detection on sampled frames",
    )
    parser.add_argument(
        "--ffmpeg-bin",
        default=None,
        help="Path to ffmpeg binary for audio extraction; defaults to system ffmpeg when available",
    )
    parser.add_argument(
        "--ffmpeg-threads",
        type=int,
        default=0,
        help="Threads per ffmpeg invocation (0=ffmpeg default)",
    )
    parser.add_argument(
        "--resize-device",
        choices=["auto", "cpu", "cuda", "mps"],
        default="cpu",
        help="Device for face crop resize stage",
    )
    args = parser.parse_args()

    audio_cfg = {
        "sample_rate": 16000, "n_fft": 800, "hop_size": 200, "win_size": 800,
        "n_mels": 80, "fmin": 55, "fmax": 7600, "preemphasis": 0.97,
    }
    audio_proc = AudioProcessor(audio_cfg)

    # Find all videos
    videos = sorted(
        os.path.join(args.input, f)
        for f in os.listdir(args.input)
        if f.endswith((".mp4", ".avi", ".mov"))
    )
    print(f"[Preprocess] Found {len(videos)} videos in {args.input}")
    print(f"[Preprocess] Using ffmpeg: {resolve_ffmpeg_bin(args.ffmpeg_bin)}")
    print(
        f"[Preprocess] Detector: {args.detector_backend}"
        f" ({resolve_detector_device(args.detector_backend, args.detector_device)})"
    )
    print(f"[Preprocess] detector_batch_size={args.detector_batch_size}")
    print(f"[Preprocess] resize_device={resolve_resize_device(args.resize_device)}")
    print(f"[Preprocess] ffmpeg_threads={args.ffmpeg_threads}")

    if args.max_videos > 0:
        videos = videos[:args.max_videos]
        print(f"[Preprocess] Limited to {len(videos)}")

    os.makedirs(args.output, exist_ok=True)

    ok = 0
    bad = 0
    fail = 0
    skip = 0
    for i, vpath in enumerate(videos):
        name, status, elapsed = process_video(
            vpath,
            args.output,
            args.size,
            audio_proc,
            args.max_frames,
            detect_every=args.detect_every,
            smooth_window=args.smooth_window,
            save_preview=not args.no_preview,
            overwrite=args.overwrite,
            ffmpeg_bin=args.ffmpeg_bin,
            ffmpeg_threads=args.ffmpeg_threads,
            detector_backend=args.detector_backend,
            detector_device=args.detector_device,
            detector_batch_size=args.detector_batch_size,
            resize_device=args.resize_device,
        )
        if status.startswith("ok_bad"):
            ok += 1
            bad += 1
            print(f"  [{i+1}/{len(videos)}] {name}: {status} ({elapsed:.1f}s)")
        elif status.startswith("ok"):
            ok += 1
            print(f"  [{i+1}/{len(videos)}] {name}: {status} ({elapsed:.1f}s)")
        elif status == "skip":
            skip += 1
            print(f"  [{i+1}/{len(videos)}] {name}: already processed")
        else:
            fail += 1
            print(f"  [{i+1}/{len(videos)}] {name}: FAIL — {status} ({elapsed:.1f}s)")

    print(f"\n[Preprocess] Done: {ok} ok ({bad} flagged bad), {skip} skipped, {fail} failed")
    print(f"[Preprocess] Output: {args.output}")

    # Summary
    total_frames = 0
    for d in os.listdir(args.output):
        frames_path = os.path.join(args.output, d, "frames.npy")
        if os.path.exists(frames_path):
            total_frames += np.load(frames_path, mmap_mode="r").shape[0]
    print(f"[Preprocess] Total frames: {total_frames}")


if __name__ == "__main__":
    main()
