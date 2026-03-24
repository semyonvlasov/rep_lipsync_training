#!/usr/bin/env python3
"""
Helpers for normalizing short talking-head clips to a standard training format.
"""

import os
import shutil
import subprocess
import sys
import tempfile
from functools import lru_cache


VIDEO_ENCODER_CHOICES = [
    "auto",
    "libx264",
    "h264_videotoolbox",
    "h264_nvenc",
]


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


@lru_cache(maxsize=16)
def _available_encoders(ffmpeg_bin):
    try:
        proc = subprocess.run(
            [ffmpeg_bin, "-hide_banner", "-encoders"],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except Exception:
        return set()

    encoders = set()
    for line in proc.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0].startswith("V"):
            encoders.add(parts[1])
    return encoders


@lru_cache(maxsize=32)
def _probe_encoder(ffmpeg_bin, video_encoder):
    if video_encoder == "libx264":
        return True, "software"

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp_path = tmp.name
        cmd = [
            ffmpeg_bin,
            "-hide_banner",
            "-y",
            "-f",
            "lavfi",
            "-i",
            # NVENC rejects tiny synthetic frames like 64x64, so use a
            # conservative talking-head-sized probe frame here.
            "color=c=black:s=256x256:r=25:d=0.2",
            "-f",
            "lavfi",
            "-i",
            "anullsrc=r=16000:cl=mono",
            "-shortest",
            *build_video_codec_args(video_encoder, "600k"),
            "-c:a",
            "aac",
            "-b:a",
            "64k",
            "-ar",
            "16000",
            "-ac",
            "1",
            tmp_path,
        ]
        subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            timeout=20,
        )
        return True, "ok"
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "").strip()
        if detail:
            detail = " ".join(detail.split())
        return False, detail[-240:] if detail else "probe_failed"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def select_video_encoder(requested, ffmpeg_bin):
    requested = (requested or "auto").strip()
    if requested not in VIDEO_ENCODER_CHOICES:
        raise ValueError(
            f"Unsupported video encoder '{requested}'. "
            f"Expected one of: {', '.join(VIDEO_ENCODER_CHOICES)}"
        )

    encoders = _available_encoders(ffmpeg_bin)
    if requested == "auto":
        if sys.platform == "darwin" and "h264_videotoolbox" in encoders:
            ok, _ = _probe_encoder(ffmpeg_bin, "h264_videotoolbox")
            if ok:
                return "h264_videotoolbox"
        if sys.platform.startswith("linux") and "h264_nvenc" in encoders:
            ok, _ = _probe_encoder(ffmpeg_bin, "h264_nvenc")
            if ok:
                return "h264_nvenc"
        return "libx264"

    if requested != "libx264" and requested not in encoders:
        raise RuntimeError(
            f"ffmpeg encoder '{requested}' is not available in {ffmpeg_bin}"
        )
    if requested != "libx264":
        ok, detail = _probe_encoder(ffmpeg_bin, requested)
        if not ok:
            raise RuntimeError(
                f"ffmpeg encoder '{requested}' is present but unusable: {detail}"
            )
    return requested


def build_video_codec_args(video_encoder, video_bitrate):
    if video_encoder == "libx264":
        return ["-c:v", "libx264", "-preset", "fast", "-crf", "23"]
    if video_encoder == "h264_videotoolbox":
        return [
            "-c:v",
            "h264_videotoolbox",
            "-allow_sw",
            "1",
            "-b:v",
            str(video_bitrate),
        ]
    if video_encoder == "h264_nvenc":
        return [
            "-c:v",
            "h264_nvenc",
            "-preset",
            "p4",
            "-b:v",
            str(video_bitrate),
        ]
    raise ValueError(f"Unsupported video encoder '{video_encoder}'")


def _candidate_ffprobe_bins(ffmpeg_bin):
    candidates = []
    if ffmpeg_bin:
        ffmpeg_dir = os.path.dirname(ffmpeg_bin)
        sibling = os.path.join(ffmpeg_dir, "ffprobe")
        candidates.append(sibling)
    candidates.extend(
        [
            os.environ.get("FFPROBE_BIN"),
            shutil.which("ffprobe"),
        ]
    )
    seen = set()
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            yield candidate


def media_file_is_valid(path, ffmpeg_bin):
    if not os.path.exists(path):
        return False
    try:
        if os.path.getsize(path) <= 0:
            return False
    except OSError:
        return False

    for ffprobe_bin in _candidate_ffprobe_bins(ffmpeg_bin):
        try:
            proc = subprocess.run(
                [
                    ffprobe_bin,
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=codec_name",
                    "-of",
                    "default=noprint_wrappers=1:nokey=1",
                    path,
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            )
            return bool(proc.stdout.strip())
        except Exception:
            continue
    return True


def normalize_video_clip(
    input_path,
    output_path,
    fps,
    *,
    start_time=0.0,
    duration=None,
    ffmpeg_bin=None,
    ffmpeg_threads=0,
    video_encoder="auto",
    video_bitrate="2200k",
    audio_bitrate="128k",
    sample_rate=16000,
    timeout=120,
):
    """
    Normalize a clip to CFR video + mono 16kHz audio.

    Returns `(ok, detail, selected_encoder)`.
    """
    ffmpeg_bin = resolve_ffmpeg_bin(ffmpeg_bin)
    selected_encoder = select_video_encoder(video_encoder, ffmpeg_bin)
    if os.path.exists(output_path):
        if media_file_is_valid(output_path, ffmpeg_bin):
            return True, "exists", selected_encoder
        try:
            os.remove(output_path)
        except OSError:
            return False, "stale_output_not_removable", selected_encoder
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    cmd = [ffmpeg_bin, "-y"]
    if start_time and float(start_time) > 0:
        cmd.extend(["-ss", str(start_time)])
    cmd.extend(["-i", input_path])
    if duration is not None and float(duration) > 0:
        cmd.extend(["-t", str(duration)])
    cmd.extend(
        [
            "-vf",
            f"fps={fps}",
            "-vsync",
            "cfr",
        ]
    )
    if ffmpeg_threads and int(ffmpeg_threads) > 0:
        cmd.extend(["-threads", str(int(ffmpeg_threads))])
    cmd.extend(build_video_codec_args(selected_encoder, video_bitrate))
    cmd.extend(
        [
            "-c:a",
            "aac",
            "-b:a",
            str(audio_bitrate),
            "-ar",
            str(int(sample_rate)),
            "-ac",
            "1",
            output_path,
        ]
    )

    try:
        subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return True, "ok", selected_encoder
    except subprocess.TimeoutExpired:
        return False, "timeout", selected_encoder
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "").strip()
        if detail:
            detail = " ".join(detail.split())
        return False, detail[-240:] if detail else "ffmpeg_failed", selected_encoder
