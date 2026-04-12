from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, fields, asdict


@dataclass
class NormalizationConfig:
    fps: int = 25
    bitrate: str = "20M"
    codec: str = "libx264"
    pixel_format: str = "yuv420p"
    ffmpeg_bin: str = "ffmpeg"
    ffmpeg_threads: int = 0
    ffmpeg_timeout: int = 180


@dataclass
class DetectionConfig:
    model_path: str = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "models",
        "face_processing",
        "face_landmarker_v2_with_blendshapes.task",
    )
    num_faces: int = 2
    min_detection_confidence: float = 0.85
    min_presence_confidence: float = 0.85
    use_gpu: bool = False


@dataclass
class PoseConfig:
    landmark_indices: tuple[int, ...] = (1, 152, 33, 263, 61, 291)


@dataclass
class BadFrameThresholds:
    min_face_h: int = 180
    max_abs_yaw: float = 30.0
    max_abs_pitch: float = 25.0
    max_abs_roll: float = 20.0
    max_delta_cx_ratio: float = 0.08
    max_delta_cy_ratio: float = 0.08
    max_delta_yaw: float = 10.0
    max_delta_pitch: float = 10.0
    max_delta_roll: float = 12.0
    max_face_h_ratio_deviation: float = 0.12
    min_confidence: float = 0.5
    min_segment_length: int = 50
    max_segment_length: int = 200  # 8 sec at 25fps
    # Excessive face motion: rolling window
    motion_window_frames: int = 25
    max_cumulative_motion_ratio: float = 0.25


@dataclass
class RankingThresholds:
    # confident
    conf_mean_abs_yaw: float = 12.0
    conf_mean_abs_pitch: float = 10.0
    conf_mean_abs_roll: float = 8.0
    conf_max_abs_yaw: float = 20.0
    conf_max_abs_pitch: float = 16.0
    conf_max_abs_roll: float = 12.0
    conf_face_size_std_ratio: float = 0.08
    conf_std_cx_ratio: float = 0.04
    conf_std_cy_ratio: float = 0.04
    conf_jump_ratio: float = 0.02
    conf_low_conf_ratio: float = 0.02
    # medium
    med_mean_abs_yaw: float = 18.0
    med_mean_abs_pitch: float = 14.0
    med_mean_abs_roll: float = 12.0
    med_max_abs_yaw: float = 28.0
    med_max_abs_pitch: float = 20.0
    med_max_abs_roll: float = 18.0
    med_face_size_std_ratio: float = 0.15
    med_std_cx_ratio: float = 0.08
    med_std_cy_ratio: float = 0.08
    med_jump_ratio: float = 0.08
    med_low_conf_ratio: float = 0.08


@dataclass
class ExportConfig:
    mode: str = "stretch_to_square"
    fps: int = 25
    bitrate: str = "1M"
    codec: str = "libx264"
    pixel_format: str = "yuv420p"
    ffmpeg_bin: str = "ffmpeg"
    ffmpeg_threads: int = 0
    ffmpeg_timeout: int = 180


@dataclass
class PipelineConfig:
    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    pose: PoseConfig = field(default_factory=PoseConfig)
    bad_frame: BadFrameThresholds = field(default_factory=BadFrameThresholds)
    ranking: RankingThresholds = field(default_factory=RankingThresholds)
    export: ExportConfig = field(default_factory=ExportConfig)
    save_frame_log: bool = False
    keep_normalized: bool = False
    output_dir: str = "output"

    @staticmethod
    def from_json(path: str) -> PipelineConfig:
        with open(path) as f:
            data = json.load(f)
        cfg = PipelineConfig()
        _nested = {
            "normalization": NormalizationConfig,
            "detection": DetectionConfig,
            "pose": PoseConfig,
            "bad_frame": BadFrameThresholds,
            "ranking": RankingThresholds,
            "export": ExportConfig,
        }
        for key, cls in _nested.items():
            if key in data:
                sub = getattr(cfg, key)
                for k, v in data[key].items():
                    if hasattr(sub, k):
                        setattr(sub, k, v)
        for k in ("save_frame_log", "keep_normalized", "output_dir"):
            if k in data:
                setattr(cfg, k, data[k])
        return cfg

    def to_dict(self) -> dict:
        return asdict(self)
