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
    DEFAULT_SYNCNET_CHECKPOINT,
    compute_sync_alignment_from_faceclip,
    write_sync_alignment_to_meta_path,
)
from evaluate_official_syncnet_video_shifts import (  # noqa: E402
    AudioProcessor,
    DEFAULT_AUDIO_CFG,
    extract_wav,
    read_frames,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Recompute sync_alignment for existing exported faceclip manifests"
    )
    parser.add_argument("--probe-json", action="append", default=[])
    parser.add_argument("--export-dir", action="append", default=[])
    parser.add_argument("--syncnet", default=DEFAULT_SYNCNET_CHECKPOINT)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--ffmpeg-bin", default="ffmpeg")
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--mel-step-size", type=int, default=16)
    parser.add_argument("--syncnet-T", type=int, default=5)
    parser.add_argument("--search-mel-ticks", type=int, default=10)
    parser.add_argument("--guard-mel-ticks", type=int, default=10)
    parser.add_argument("--samples", type=int, default=0)
    parser.add_argument("--sample-density-per-5s", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=20260403)
    parser.add_argument("--min-start-gap-ratio", type=float, default=0.0)
    parser.add_argument("--start-gap-multiple", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=640)
    parser.add_argument("--outlier-trim-ratio", type=float, default=0.2)
    parser.add_argument("--output-json", default="")
    return parser.parse_args()


def resolve_repo_path(path: str) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (PROJECT_ROOT / candidate).resolve()


def iter_meta_pairs_from_probe_json(path: Path):
    payload = json.loads(path.read_text())
    for detail in payload.get("details", []):
        meta_json = detail.get("meta_json")
        export_video = detail.get("export_video")
        if not meta_json or not export_video:
            continue
        yield resolve_repo_path(meta_json), resolve_repo_path(export_video)


def iter_meta_pairs_from_export_dir(path: Path):
    for meta_path in sorted(path.rglob("*.json")):
        if meta_path.name.endswith(".detections.json"):
            continue
        video_path = meta_path.with_suffix(".mp4")
        if video_path.exists():
            yield meta_path.resolve(), video_path.resolve()


def collect_pairs(args):
    seen = set()
    pairs = []
    for probe_json in args.probe_json:
        for meta_path, video_path in iter_meta_pairs_from_probe_json(resolve_repo_path(probe_json)):
            key = (str(meta_path), str(video_path))
            if key not in seen:
                seen.add(key)
                pairs.append((meta_path, video_path))
    for export_dir in args.export_dir:
        for meta_path, video_path in iter_meta_pairs_from_export_dir(resolve_repo_path(export_dir)):
            key = (str(meta_path), str(video_path))
            if key not in seen:
                seen.add(key)
                pairs.append((meta_path, video_path))
    return pairs


def main():
    args = parse_args()
    pairs = collect_pairs(args)
    if not pairs:
        raise RuntimeError("No export manifests found; provide --probe-json and/or --export-dir")

    audio_proc = AudioProcessor(DEFAULT_AUDIO_CFG)
    mel_frames_per_second = float(audio_proc.sample_rate) / float(audio_proc.hop_size)

    rows = []
    for idx, (meta_path, video_path) in enumerate(pairs, start=1):
        meta = json.loads(meta_path.read_text())
        frames = read_frames(str(video_path))
        wav_path = extract_wav(str(video_path), args.ffmpeg_bin)
        try:
            wav = audio_proc.load_wav(wav_path)
            mel = audio_proc.melspectrogram(wav)
        finally:
            try:
                os.remove(wav_path)
            except OSError:
                pass

        result = compute_sync_alignment_from_faceclip(
            frames=frames,
            mel=mel,
            fps=args.fps,
            mel_frames_per_second=mel_frames_per_second,
            mel_step_size=args.mel_step_size,
            syncnet_T=args.syncnet_T,
            checkpoint_path=args.syncnet,
            device=args.device,
            search_mel_ticks=args.search_mel_ticks,
            search_guard_mel_ticks=args.guard_mel_ticks,
            samples=args.samples,
            sample_density_per_5s=args.sample_density_per_5s,
            seed=args.seed,
            min_start_gap_ratio=args.min_start_gap_ratio,
            start_gap_multiple=args.start_gap_multiple,
            batch_size=args.batch_size,
            outlier_trim_ratio=args.outlier_trim_ratio,
        )

        updated = write_sync_alignment_to_meta_path(
            str(meta_path),
            meta,
            audio_shift_mel_ticks=result["audio_shift_mel_ticks"],
            n_frames=len(frames),
            mel_total_steps=int(mel.shape[1]),
            fps=args.fps,
            mel_frames_per_second=mel_frames_per_second,
            mel_step_size=args.mel_step_size,
            search_guard_mel_ticks=args.guard_mel_ticks,
            source="recomputed_from_export_faceclip",
            search_range_mel_ticks=args.search_mel_ticks,
            search_samples=result["samples"],
            search_seed=args.seed,
            min_start_gap_ratio=args.min_start_gap_ratio,
            start_gap_multiple=args.start_gap_multiple,
            best_mean_loss=result["best_mean_loss"],
            zero_mean_loss=result["zero_mean_loss"],
            extra={
                "compute_device": result["device"],
                "starts": result["starts"],
                "kept_starts": result.get("kept_starts", []),
                "dropped_starts": result.get("dropped_starts", []),
                "local_best_shifts": result.get("local_best_shifts", []),
                "local_best_shift_center": result.get("local_best_shift_center"),
                "num_points_before_trim": result.get("num_points_before_trim"),
                "num_points_after_trim": result.get("num_points_after_trim"),
                "outlier_trim_ratio": result.get("outlier_trim_ratio"),
                "sample_density_per_5s": result.get("sample_density_per_5s"),
            },
        )

        row = {
            "index": idx,
            "total": len(pairs),
            "clip": meta.get("name", meta_path.stem),
            "meta_path": str(meta_path),
            "video_path": str(video_path),
            "audio_shift_mel_ticks": int(updated["sync_alignment"]["audio_shift_mel_ticks"]),
            "valid_frame_count": int(updated["sync_alignment"]["valid_frame_count"]),
            "num_points_before_trim": int(updated["sync_alignment"].get("num_points_before_trim") or 0),
            "num_points_after_trim": int(updated["sync_alignment"].get("num_points_after_trim") or 0),
        }
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

    if args.output_json:
        out_path = resolve_repo_path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({"rows": rows}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
