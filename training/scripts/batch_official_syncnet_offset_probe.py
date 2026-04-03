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
from evaluate_official_syncnet_video_shifts import (
    AudioProcessor,
    DEFAULT_AUDIO_CFG,
    choose_shared_starts,
    evaluate_video,
    load_syncnet,
    prepare_source_crop_sample,
    resolve_device,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_OFFSETS = [0, 1, -1, 2, -2, 3, -3, 4, -4, 6, -6, 8, -8]


def parse_args():
    parser = argparse.ArgumentParser(description="Batch official SyncNet offset probe over prepared detections/meta")
    parser.add_argument("--syncnet", required=True)
    parser.add_argument("--normalized-input-dir", default="")
    parser.add_argument("--export-dir", required=True)
    parser.add_argument("--dataset-label", required=True)
    parser.add_argument(
        "--sample-mode",
        choices=("source_crop", "export_video"),
        default="source_crop",
        help="Evaluate either direct source-crop from normalized source or exported lazy faceclip mp4",
    )
    parser.add_argument("--samples", type=int, required=True)
    parser.add_argument(
        "--min-start-gap-ratio",
        type=float,
        default=0.0,
        help="Minimum spacing between sampled starts as a fraction of clip length",
    )
    parser.add_argument(
        "--start-gap-multiple",
        type=int,
        default=0,
        help=(
            "If > 0, require pairwise differences between sampled starts to be multiples "
            "of this value"
        ),
    )
    parser.add_argument("--seed", type=int, default=20260402)
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--mel-step-size", type=int, default=16)
    parser.add_argument("--syncnet-T", type=int, default=5)
    parser.add_argument("--offsets", default="0,1,-1,2,-2,3,-3,4,-4,6,-6,8,-8")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--ffmpeg-bin", default="ffmpeg")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-markdown", default="")
    parser.add_argument(
        "--respect-sync-alignment",
        action="store_true",
        help="For export_video samples, apply sync_alignment from sample manifest before probing offsets",
    )
    parser.add_argument(
        "--write-sync-alignment",
        action="store_true",
        help="Write sync_alignment into export manifest json when sample-mode=export_video",
    )
    parser.add_argument(
        "--sync-alignment-guard-mel-ticks",
        type=int,
        default=DEFAULT_SYNC_ALIGNMENT_GUARD_MEL_TICKS,
    )
    return parser.parse_args()


def find_clip_stems(export_dir: Path):
    stems = []
    for tier in ("confident", "medium", "unconfident"):
        tier_dir = export_dir / tier
        if not tier_dir.exists():
            continue
        for meta_path in sorted(tier_dir.glob("*.json")):
            if meta_path.name.endswith(".detections.json"):
                continue
            stem = meta_path.stem
            det_path = tier_dir / f"{stem}.detections.json"
            if det_path.exists():
                stems.append((stem, str(meta_path), str(det_path), tier))
    return stems


def compare_best_vs_zero(result):
    best_offset = int(result["best_offset_by_mean_loss"])
    best_row = result["metrics_by_offset"][str(best_offset)]
    zero_row = result["metrics_by_offset"]["0"]
    best_samples = {int(row["start"]): row for row in best_row.get("sample_values", [])}
    zero_samples = {int(row["start"]): row for row in zero_row.get("sample_values", [])}
    starts = sorted(set(best_samples) & set(zero_samples))
    wins = sum(1 for start in starts if best_samples[start]["loss"] < zero_samples[start]["loss"])
    return {
        "best_offset": best_offset,
        "best_mean_loss": float(best_row["mean_loss"]),
        "zero_mean_loss": float(zero_row["mean_loss"]),
        "best_wins_over_zero": int(wins),
        "num_samples": int(len(starts)),
    }


def write_markdown(path: Path, rows):
    lines = [
        "| dataset | clip | tier | winner_offset | winner_mean_loss | zero_mean_loss | best_vs_zero_wins | n |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {dataset} | {clip} | {tier} | {best_offset} | {best_mean_loss:.6f} | {zero_mean_loss:.6f} | {best_wins_over_zero} | {num_samples} |".format(
                **row
            )
        )
    path.write_text("\n".join(lines) + "\n")


def resolve_meta_path(meta_json: str) -> Path:
    path = Path(meta_json)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def write_sync_alignment_for_detail(
    *,
    detail_meta_json: str,
    summary: dict,
    sample: dict,
    offsets: list[int],
    fps: float,
    mel_frames_per_second: float,
    mel_step_size: int,
    guard_mel_ticks: int,
    samples: int,
    seed: int,
    min_start_gap_ratio: float,
    start_gap_multiple: int,
):
    meta_path = resolve_meta_path(detail_meta_json)
    meta = json.loads(meta_path.read_text())
    best_offset_frames = int(summary["best_offset"])
    audio_shift_mel_ticks = int(round((-best_offset_frames) * float(mel_frames_per_second) / float(fps)))
    extra = {
        "best_visual_offset_frames": best_offset_frames,
        "probe_offsets_frames": [int(v) for v in offsets],
        "probe_sample_mode": "export_video_frame_offset_probe",
    }
    write_sync_alignment_to_meta_path(
        str(meta_path),
        meta,
        audio_shift_mel_ticks=audio_shift_mel_ticks,
        n_frames=len(sample["frames"]),
        mel_total_steps=int(sample["mel_total_steps"]),
        fps=float(fps),
        mel_frames_per_second=float(mel_frames_per_second),
        mel_step_size=int(mel_step_size),
        search_guard_mel_ticks=int(guard_mel_ticks),
        source="frame_offset_probe_export_video",
        search_samples=int(samples),
        search_seed=int(seed),
        min_start_gap_ratio=float(min_start_gap_ratio),
        start_gap_multiple=int(start_gap_multiple),
        best_mean_loss=float(summary["best_mean_loss"]),
        zero_mean_loss=float(summary["zero_mean_loss"]),
        extra=extra,
    )


def main():
    args = parse_args()
    device = resolve_device(args.device)
    offsets = [int(x.strip()) for x in args.offsets.split(",") if x.strip()]

    model = load_syncnet(args.syncnet, device)
    audio_proc = AudioProcessor(DEFAULT_AUDIO_CFG)
    mel_frames_per_second = float(audio_proc.sample_rate) / float(audio_proc.hop_size)

    normalized_input_dir = Path(args.normalized_input_dir)
    export_dir = Path(args.export_dir)
    clips = find_clip_stems(export_dir)

    rows = []
    detailed = []
    for stem, meta_json, detections_json, tier in clips:
        export_video = export_dir / tier / f"{stem}.mp4"
        if args.sample_mode == "source_crop":
            source_video = normalized_input_dir / f"{stem}.mp4"
            if not source_video.exists():
                raise FileNotFoundError(f"Missing normalized source video for {stem}: {source_video}")
            source_spec = {
                "label": stem,
                "source_video": str(source_video),
                "detections_json": detections_json,
                "trimmed_meta_json": meta_json,
            }
            sample = prepare_source_crop_sample(
                source_spec,
                audio_proc=audio_proc,
                ffmpeg_bin=args.ffmpeg_bin,
                fps=args.fps,
                mel_step_size=args.mel_step_size,
            )
            sample_video_path = str(source_video)
        else:
            if not export_video.exists():
                raise FileNotFoundError(f"Missing exported video for {stem}: {export_video}")
            from evaluate_official_syncnet_video_shifts import prepare_video_sample

            sample = prepare_video_sample(
                {"label": stem, "path": str(export_video), "meta_json": meta_json},
                audio_proc=audio_proc,
                ffmpeg_bin=args.ffmpeg_bin,
                fps=args.fps,
                mel_step_size=args.mel_step_size,
                respect_sync_alignment=args.respect_sync_alignment,
            )
            sample_video_path = str(export_video)
        starts = choose_shared_starts(
            [sample],
            offsets,
            args.syncnet_T,
            args.samples,
            args.seed,
            min_gap_ratio=args.min_start_gap_ratio,
            start_gap_multiple=args.start_gap_multiple,
        )
        result = evaluate_video(
            model,
            device,
            sample,
            starts,
            offsets,
            args.syncnet_T,
            store_sample_values=True,
        )
        summary = compare_best_vs_zero(result)
        row = {
            "dataset": args.dataset_label,
            "clip": stem,
            "tier": tier,
            **summary,
        }
        rows.append(row)
        detailed.append(
            {
                "dataset": args.dataset_label,
                "clip": stem,
                "tier": tier,
                "sample_mode": args.sample_mode,
                "respect_sync_alignment": bool(args.respect_sync_alignment),
                "source_video": sample_video_path,
                "meta_json": meta_json,
                "detections_json": detections_json,
                "export_video": str(export_video),
                "result": result,
                "summary": row,
            }
        )
        if args.write_sync_alignment and args.sample_mode == "export_video":
            write_sync_alignment_for_detail(
                detail_meta_json=meta_json,
                summary=row,
                sample=sample,
                offsets=offsets,
                fps=args.fps,
                mel_frames_per_second=mel_frames_per_second,
                mel_step_size=args.mel_step_size,
                guard_mel_ticks=args.sync_alignment_guard_mel_ticks,
                samples=args.samples,
                seed=args.seed,
                min_start_gap_ratio=args.min_start_gap_ratio,
                start_gap_multiple=args.start_gap_multiple,
            )
        print(json.dumps(row, ensure_ascii=False), flush=True)

    payload = {
        "dataset": args.dataset_label,
        "syncnet_checkpoint": args.syncnet,
        "device": device,
        "samples": int(args.samples),
        "seed": int(args.seed),
        "fps": float(args.fps),
        "mel_step_size": int(args.mel_step_size),
        "offsets": offsets,
        "sample_mode": args.sample_mode,
        "respect_sync_alignment": bool(args.respect_sync_alignment),
        "min_start_gap_ratio": float(args.min_start_gap_ratio),
        "start_gap_multiple": int(args.start_gap_multiple),
        "rows": rows,
        "details": detailed,
    }
    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))

    if args.output_markdown:
        md_path = Path(args.output_markdown)
        md_path.parent.mkdir(parents=True, exist_ok=True)
        write_markdown(md_path, rows)


if __name__ == "__main__":
    main()
