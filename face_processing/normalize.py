from __future__ import annotations

import logging
import subprocess

from face_processing.config import NormalizationConfig

logger = logging.getLogger(__name__)


def normalize_video(
    input_path: str,
    output_path: str,
    config: NormalizationConfig | None = None,
) -> str:
    if config is None:
        config = NormalizationConfig()

    cmd = [
        config.ffmpeg_bin,
        "-y",
        "-i", input_path,
        "-r", str(config.fps),
        "-c:v", config.codec,
        "-b:v", config.bitrate,
        "-pix_fmt", config.pixel_format,
        "-an",
        output_path,
    ]
    if config.ffmpeg_threads and config.ffmpeg_threads > 0:
        cmd[1:1] = ["-threads", str(int(config.ffmpeg_threads))]
    logger.info("Normalizing: %s", " ".join(cmd))
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=config.ffmpeg_timeout,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg normalization failed (code {result.returncode}):\n{result.stderr}"
        )
    logger.info("Normalized video saved to %s", output_path)
    return output_path
