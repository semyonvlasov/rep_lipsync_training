#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Create sidecar AAC files next to existing faceclip MP4s using "
            "best_offset values from SyncNet lazy-mp4 probe JSON"
        )
    )
    parser.add_argument(
        "--probe-json",
        action="append",
        required=True,
        help="Path to *_syncnet_probe_lazy_mp4.json (repeatable)",
    )
    parser.add_argument(
        "--repo-root",
        default=".",
        help="Repository root used to resolve relative export_video paths",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=25.0,
        help="FPS used to convert frame offsets to seconds",
    )
    parser.add_argument(
        "--audio-bitrate",
        default="128k",
        help="AAC bitrate for re-encoded sidecar audio",
    )
    parser.add_argument(
        "--ffmpeg-bin",
        default="ffmpeg",
    )
    parser.add_argument(
        "--ffprobe-bin",
        default="ffprobe",
    )
    parser.add_argument(
        "--output-manifest",
        default="",
        help="Optional output manifest path; defaults next to each probe JSON",
    )
    return parser.parse_args()


def load_json(path: Path):
    return json.loads(path.read_text())


def resolve_path(repo_root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (repo_root / path).resolve()


def ffprobe_duration(ffprobe_bin: str, path: Path) -> float:
    cmd = [
        ffprobe_bin,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    out = subprocess.check_output(cmd, text=True).strip()
    return float(out)


def ffprobe_has_audio(ffprobe_bin: str, path: Path) -> bool:
    cmd = [
        ffprobe_bin,
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=index",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    out = subprocess.check_output(cmd, text=True).strip()
    return bool(out)


def build_filter_chain(gap_seconds: float, duration_seconds: float) -> str:
    if gap_seconds > 0:
        delay_ms = int(round(gap_seconds * 1000.0))
        return (
            f"adelay={delay_ms}:all=1,"
            f"apad,"
            f"atrim=duration={duration_seconds:.6f}"
        )
    if gap_seconds < 0:
        trim_start = -gap_seconds
        return (
            f"atrim=start={trim_start:.6f},"
            f"asetpts=PTS-STARTPTS,"
            f"apad,"
            f"atrim=duration={duration_seconds:.6f}"
        )
    return f"atrim=duration={duration_seconds:.6f}"


def create_sidecar(
    ffmpeg_bin: str,
    ffprobe_bin: str,
    video_path: Path,
    output_audio_path: Path,
    offset_frames: int,
    fps: float,
    audio_bitrate: str,
):
    if not ffprobe_has_audio(ffprobe_bin, video_path):
        return {
            "status": "skipped_no_audio",
            "video_path": str(video_path),
            "audio_path": str(output_audio_path),
            "offset_frames": int(offset_frames),
            "gap_seconds": float(offset_frames / fps),
        }

    duration_seconds = ffprobe_duration(ffprobe_bin, video_path)
    gap_seconds = float(offset_frames / fps)
    filter_chain = build_filter_chain(gap_seconds, duration_seconds)

    output_audio_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg_bin,
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-af",
        filter_chain,
        "-c:a",
        "aac",
        "-b:a",
        audio_bitrate,
        str(output_audio_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return {
        "status": "ok",
        "video_path": str(video_path),
        "audio_path": str(output_audio_path),
        "offset_frames": int(offset_frames),
        "gap_seconds": gap_seconds,
        "duration_seconds": duration_seconds,
        "filter_chain": filter_chain,
    }


def main():
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    all_manifests = []

    for probe_json_value in args.probe_json:
        probe_json_path = Path(probe_json_value).resolve()
        probe = load_json(probe_json_path)
        rows = []

        for detail in probe.get("details", []):
            export_video_value = detail.get("export_video") or detail.get("source_video")
            if not export_video_value:
                continue
            video_path = resolve_path(repo_root, export_video_value)
            output_audio_path = video_path.with_suffix(".aac")
            result = create_sidecar(
                ffmpeg_bin=args.ffmpeg_bin,
                ffprobe_bin=args.ffprobe_bin,
                video_path=video_path,
                output_audio_path=output_audio_path,
                offset_frames=int(detail["summary"]["best_offset"]),
                fps=float(args.fps),
                audio_bitrate=args.audio_bitrate,
            )
            result.update(
                {
                    "clip": detail["clip"],
                    "tier": detail.get("tier"),
                    "probe_json": str(probe_json_path),
                }
            )
            rows.append(result)
            print(
                json.dumps(
                    {
                        "clip": detail["clip"],
                        "offset_frames": int(detail["summary"]["best_offset"]),
                        "audio_path": str(output_audio_path),
                        "status": result["status"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

        manifest = {
            "probe_json": str(probe_json_path),
            "repo_root": str(repo_root),
            "fps": float(args.fps),
            "audio_bitrate": args.audio_bitrate,
            "rows": rows,
        }
        if args.output_manifest:
            manifest_path = Path(args.output_manifest).resolve()
        else:
            manifest_path = probe_json_path.with_name(
                probe_json_path.stem + "_sidecar_aac_manifest.json"
            )
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
        all_manifests.append(str(manifest_path))

    print(
        json.dumps(
            {
                "status": "done",
                "manifests": all_manifests,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
