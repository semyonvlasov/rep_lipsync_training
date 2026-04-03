#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

TRAINING_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(TRAINING_ROOT)
if TRAINING_ROOT not in sys.path:
    sys.path.insert(0, TRAINING_ROOT)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from data.sync_alignment import (  # noqa: E402
    DEFAULT_SYNC_ALIGNMENT_GUARD_MEL_TICKS,
    write_sync_alignment_to_meta_path,
)
from evaluate_official_syncnet_video_shifts import (  # noqa: E402
    AudioProcessor,
    DEFAULT_AUDIO_CFG,
    extract_wav,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args():
    parser = argparse.ArgumentParser(description="Backfill sync_alignment into faceclip manifests from probe JSON")
    parser.add_argument("--probe-json", action="append", required=True)
    parser.add_argument(
        "--guard-mel-ticks",
        type=int,
        default=DEFAULT_SYNC_ALIGNMENT_GUARD_MEL_TICKS,
    )
    return parser.parse_args()


def resolve_meta_path(meta_json: str) -> Path:
    path = Path(meta_json)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def resolve_export_video_path(export_video: str) -> Path:
    path = Path(export_video)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def main():
    args = parse_args()
    mel_frames_per_second = float(DEFAULT_AUDIO_CFG["sample_rate"]) / float(DEFAULT_AUDIO_CFG["hop_size"])
    audio_proc = AudioProcessor(DEFAULT_AUDIO_CFG)
    written = 0

    for probe_path in args.probe_json:
        payload = json.loads(Path(probe_path).read_text())
        fps = float(payload.get("fps", 25.0))
        offsets = [int(v) for v in payload.get("offsets", [])]
        samples = int(payload.get("samples", 0))
        seed = int(payload.get("seed", 0))
        min_start_gap_ratio = float(payload.get("min_start_gap_ratio", 0.0))
        start_gap_multiple = int(payload.get("start_gap_multiple", 0))

        for detail in payload.get("details", []):
            if detail.get("sample_mode") != "export_video":
                continue

            meta_path = resolve_meta_path(detail["meta_json"])
            export_video_path = resolve_export_video_path(detail["export_video"])
            meta = json.loads(meta_path.read_text())
            sample_result = detail["result"]
            summary = detail["summary"]
            best_offset_frames = int(summary["best_offset"])
            audio_shift_mel_ticks = int(round((-best_offset_frames) * mel_frames_per_second / fps))
            wav_path = extract_wav(str(export_video_path), "ffmpeg")
            try:
                wav = audio_proc.load_wav(wav_path)
                mel = audio_proc.melspectrogram(wav)
            finally:
                try:
                    os.remove(wav_path)
                except OSError:
                    pass
            extra = {
                "best_visual_offset_frames": best_offset_frames,
                "probe_offsets_frames": offsets,
                "probe_sample_mode": "export_video_frame_offset_probe",
            }
            write_sync_alignment_to_meta_path(
                str(meta_path),
                meta,
                audio_shift_mel_ticks=audio_shift_mel_ticks,
                n_frames=int(sample_result["n_frames"]),
                mel_total_steps=int(mel.shape[1]),
                fps=fps,
                mel_frames_per_second=mel_frames_per_second,
                mel_step_size=int(payload.get("mel_step_size", 16)),
                search_guard_mel_ticks=int(args.guard_mel_ticks),
                source="frame_offset_probe_export_video",
                search_samples=samples,
                search_seed=seed,
                min_start_gap_ratio=min_start_gap_ratio,
                start_gap_multiple=start_gap_multiple,
                best_mean_loss=float(summary["best_mean_loss"]),
                zero_mean_loss=float(summary["zero_mean_loss"]),
                extra=extra,
            )
            written += 1

    print(json.dumps({"written": written}, ensure_ascii=False))


if __name__ == "__main__":
    main()
