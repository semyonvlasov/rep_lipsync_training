#!/usr/bin/env python3
"""
Evaluate how the official SyncNet scores fixed video/audio offsets on faceclip videos.

Protocol:
  - choose the same random 5-frame window starts for every input video
  - keep audio anchored at the base start
  - shift only the visual window by the requested frame offsets
  - compute official-style sync loss against positive label=1
  - average over N random starts
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import random
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F


TRAINING_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(TRAINING_ROOT)
if TRAINING_ROOT not in sys.path:
    sys.path.insert(0, TRAINING_ROOT)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from data.audio import AudioProcessor
from data.sync_alignment import build_shifted_frame_aligned_mels, load_sync_alignment


DEFAULT_AUDIO_CFG = {
    "sample_rate": 16000,
    "n_fft": 800,
    "hop_size": 200,
    "win_size": 800,
    "n_mels": 80,
    "fmin": 55,
    "fmax": 7600,
    "preemphasis": 0.97,
}


def log(msg: str) -> None:
    print(msg, flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate official SyncNet video-shift curves")
    parser.add_argument("--syncnet", required=True, help="Official SyncNet checkpoint")
    parser.add_argument(
        "--video",
        action="append",
        default=[],
        help="Video spec in the form label=/abs/or/rel/path.mp4 (repeatable)",
    )
    parser.add_argument(
        "--source-crop",
        action="append",
        default=[],
        help=(
            "Source-crop spec in the form "
            "label=/path/source.mp4|/path/detections.json|/path/trimmed_meta.json"
        ),
    )
    parser.add_argument(
        "--offsets",
        default="0,1,-1,2,-2,3,-3,4,-4,6,-6,8,-8",
        help="Comma-separated visual frame offsets relative to fixed audio",
    )
    parser.add_argument("--samples", type=int, default=50, help="Random starts per video")
    parser.add_argument("--seed", type=int, default=20260402)
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--mel-step-size", type=int, default=16)
    parser.add_argument("--syncnet-T", type=int, default=5)
    parser.add_argument(
        "--min-start-gap-ratio",
        type=float,
        default=0.0,
        help="Minimum spacing between sampled start points as a fraction of clip length",
    )
    parser.add_argument(
        "--start-gap-multiple",
        type=int,
        default=0,
        help=(
            "If > 0, require pairwise differences between sampled starts to be multiples "
            "of this value by selecting all starts from one residue class"
        ),
    )
    parser.add_argument("--ffmpeg-bin", default="ffmpeg")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--store-sample-values",
        action="store_true",
        help="Store per-start loss/cosine values for each evaluated offset",
    )
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def parse_video_specs(values):
    specs = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"Video spec must be label=path, got: {value}")
        label, path = value.split("=", 1)
        label = label.strip()
        path = path.strip()
        if not label or not path:
            raise ValueError(f"Bad video spec: {value}")
        specs.append({"label": label, "path": path})
    return specs


def parse_source_crop_specs(values):
    specs = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"Source-crop spec must be label=source|detections|meta, got: {value}")
        label, payload = value.split("=", 1)
        label = label.strip()
        parts = [part.strip() for part in payload.split("|")]
        if len(parts) != 3 or not label or any(not part for part in parts):
            raise ValueError(f"Bad source-crop spec: {value}")
        specs.append(
            {
                "label": label,
                "source_video": parts[0],
                "detections_json": parts[1],
                "trimmed_meta_json": parts[2],
            }
        )
    return specs


def resolve_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    return device_arg


def load_official_syncnet_class():
    official_models_dir = os.path.join(REPO_ROOT, "models", "official_syncnet", "models")

    def load_module(name, filepath):
        spec = importlib.util.spec_from_file_location(name, filepath)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod

    conv_mod = load_module(
        "official_syncnet_models.conv",
        os.path.join(official_models_dir, "conv.py"),
    )
    with open(os.path.join(official_models_dir, "syncnet.py")) as f:
        syncnet_src = f.read()
    syncnet_src = syncnet_src.replace("from .conv import Conv2d", "")
    syncnet_ns = {"__builtins__": __builtins__}
    exec("import torch\nfrom torch import nn\nfrom torch.nn import functional as F\n", syncnet_ns)
    syncnet_ns["Conv2d"] = conv_mod.Conv2d
    exec(syncnet_src, syncnet_ns)
    return syncnet_ns["SyncNet_color"]


def load_syncnet(checkpoint_path: str, device: str):
    SyncNet = load_official_syncnet_class()
    model = SyncNet().to(device)
    ck = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ck["state_dict"] if isinstance(ck, dict) and "state_dict" in ck else ck
    model.load_state_dict(state_dict)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def extract_wav(video_path: str, ffmpeg_bin: str, start_sec: float | None = None, duration_sec: float | None = None) -> str:
    fd, wav_path = tempfile.mkstemp(suffix=".wav", prefix="syncnet_eval_")
    os.close(fd)
    cmd = [
        ffmpeg_bin,
        "-y",
    ]
    if start_sec is not None:
        cmd.extend(["-ss", f"{start_sec:.6f}"])
    if duration_sec is not None:
        cmd.extend(["-t", f"{duration_sec:.6f}"])
    cmd.extend([
        "-i",
        video_path,
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        wav_path,
    ])
    subprocess.run(cmd, check=True, capture_output=True)
    return wav_path


def build_mel_chunks(mel: np.ndarray, fps: float, mel_step_size: int):
    mel_idx_mult = 80.0 / float(fps)
    chunks = []
    n_frames = max(1, int(np.ceil(mel.shape[1] / mel_idx_mult)))
    for i in range(n_frames):
        start = int(i * mel_idx_mult)
        end = start + mel_step_size
        if mel.shape[1] < mel_step_size:
            chunk = np.pad(
                mel,
                ((0, 0), (0, mel_step_size - mel.shape[1])),
                mode="edge",
            )
        elif end > mel.shape[1]:
            chunk = mel[:, -mel_step_size:]
        else:
            chunk = mel[:, start:end]
        chunks.append(chunk.astype(np.float32, copy=False))
    return chunks


def read_frames(video_path: str):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames decoded from {video_path}")
    return frames


def load_json(path: str):
    return json.loads(Path(path).read_text())


def crop_resize_face(frame: np.ndarray, bbox, size: int):
    y1, y2, x1, x2 = [int(round(v)) for v in bbox]
    y1 = max(0, min(y1, frame.shape[0] - 1))
    y2 = max(y1 + 1, min(y2, frame.shape[0]))
    x1 = max(0, min(x1, frame.shape[1] - 1))
    x2 = max(x1 + 1, min(x2, frame.shape[1]))
    crop = frame[y1:y2, x1:x2]
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_LINEAR)


def read_source_face_crops(source_video: str, frame_indices, detections_by_frame, crop_size: int):
    cap = cv2.VideoCapture(source_video)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open source video: {source_video}")

    frames = []
    frame_idx = 0
    target_iter = iter(frame_indices)
    current_target = next(target_iter, None)

    while current_target is not None:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx == current_target:
            det = detections_by_frame.get(frame_idx)
            bbox = det.get("bbox") if det else None
            if bbox is None:
                raise RuntimeError(f"Missing bbox for frame_idx={frame_idx} in {source_video}")
            frames.append(crop_resize_face(frame, bbox, crop_size))
            current_target = next(target_iter, None)
        frame_idx += 1

    cap.release()
    if len(frames) != len(frame_indices):
        raise RuntimeError(
            f"Decoded {len(frames)} cropped frames, expected {len(frame_indices)} from {source_video}"
        )
    return frames


def build_visual_tensor(frames, start: int, T: int, device: str):
    window = []
    for t in range(T):
        face = frames[start + t].astype(np.float32) / 255.0
        lower = face[face.shape[0] // 2 :, :, :]
        chw = np.transpose(lower, (2, 0, 1))
        window.append(chw)
    stacked = np.concatenate(window, axis=0)
    visual = torch.from_numpy(np.ascontiguousarray(stacked)).unsqueeze(0).to(device)
    visual = F.interpolate(visual, size=(48, 96), mode="bilinear", align_corners=False)
    return visual


def build_audio_tensor(mel_chunks, start: int, device: str):
    audio = torch.from_numpy(mel_chunks[start][np.newaxis, np.newaxis]).to(device)
    return audio


def official_sync_loss_from_cosine(cos_sim: torch.Tensor) -> torch.Tensor:
    targets = torch.ones((cos_sim.size(0), 1), device=cos_sim.device, dtype=torch.float32)
    probs = cos_sim.float().clamp_(1.0e-6, 1.0 - 1.0e-6).unsqueeze(1)
    if cos_sim.device.type == "cuda":
        with torch.amp.autocast("cuda", enabled=False):
            return F.binary_cross_entropy(probs, targets)
    return F.binary_cross_entropy(probs, targets)


def mean(values):
    return float(sum(values) / max(len(values), 1))


def std(values):
    if not values:
        return 0.0
    m = mean(values)
    return float((sum((x - m) ** 2 for x in values) / len(values)) ** 0.5)


def _apply_sync_alignment_to_export_video_sample(
    *,
    frames,
    mel: np.ndarray,
    meta: dict | None,
    fps: float,
    mel_step_size: int,
    mel_frames_per_second: float,
):
    sync_alignment = load_sync_alignment(meta)
    if sync_alignment is None:
        mel_chunks = build_mel_chunks(mel, fps=fps, mel_step_size=mel_step_size)
        return frames, mel_chunks, False

    audio_shift_mel_ticks = int(sync_alignment.get("audio_shift_mel_ticks") or 0)
    mel_chunks, valid_indices = build_shifted_frame_aligned_mels(
        mel,
        n_frames=len(frames),
        fps=fps,
        mel_frames_per_second=mel_frames_per_second,
        mel_step_size=mel_step_size,
        audio_shift_mel_ticks=audio_shift_mel_ticks,
    )
    if valid_indices:
        if valid_indices == list(range(valid_indices[0], valid_indices[-1] + 1)):
            frames = frames[valid_indices[0] : valid_indices[-1] + 1]
        else:
            frames = [frames[idx] for idx in valid_indices]
    else:
        frames = []
    return frames, mel_chunks, True


def prepare_video_sample(
    video_spec,
    audio_proc: AudioProcessor,
    ffmpeg_bin: str,
    fps: float,
    mel_step_size: int,
    respect_sync_alignment: bool = False,
):
    path = video_spec["path"]
    frames = read_frames(path)
    wav_path = extract_wav(path, ffmpeg_bin)
    try:
        wav = audio_proc.load_wav(wav_path)
        mel = audio_proc.melspectrogram(wav)
    finally:
        try:
            os.remove(wav_path)
        except OSError:
            pass

    meta = None
    sync_applied = False
    if respect_sync_alignment and video_spec.get("meta_json"):
        meta = load_json(video_spec["meta_json"])
        frames, mel_chunks, sync_applied = _apply_sync_alignment_to_export_video_sample(
            frames=frames,
            mel=mel,
            meta=meta,
            fps=fps,
            mel_step_size=mel_step_size,
            mel_frames_per_second=float(audio_proc.sample_rate) / float(audio_proc.hop_size),
        )
    else:
        mel_chunks = build_mel_chunks(mel, fps=fps, mel_step_size=mel_step_size)

    n_positions = min(len(frames), len(mel_chunks))
    if n_positions <= 0:
        raise RuntimeError(f"No sync positions available for {path}")
    return {
        "label": video_spec["label"],
        "path": path,
        "frames": frames,
        "mel_chunks": mel_chunks,
        "mel_total_steps": int(mel.shape[1]),
        "n_positions": n_positions,
        "input_kind": "video",
        "sync_alignment_applied": bool(sync_applied),
        "meta_json": video_spec.get("meta_json"),
        "meta": meta,
    }


def prepare_source_crop_sample(source_crop_spec, audio_proc: AudioProcessor, ffmpeg_bin: str, fps: float, mel_step_size: int):
    source_video = source_crop_spec["source_video"]
    detections_payload = load_json(source_crop_spec["detections_json"])
    trimmed_meta = load_json(source_crop_spec["trimmed_meta_json"])
    frame_indices = trimmed_meta["trimmed_frame_indices"]
    crop_size = int(trimmed_meta.get("img_size", 256))
    detections_by_frame = {
        int(row["frame_idx"]): row
        for row in detections_payload["detections"]
        if row.get("bbox") is not None
    }
    frames = read_source_face_crops(
        source_video=source_video,
        frame_indices=frame_indices,
        detections_by_frame=detections_by_frame,
        crop_size=crop_size,
    )
    start_sec = frame_indices[0] / float(fps)
    duration_sec = len(frame_indices) / float(fps)
    wav_path = extract_wav(
        source_video,
        ffmpeg_bin,
        start_sec=start_sec,
        duration_sec=duration_sec,
    )
    try:
        wav = audio_proc.load_wav(wav_path)
        mel = audio_proc.melspectrogram(wav)
    finally:
        try:
            os.remove(wav_path)
        except OSError:
            pass
    mel_chunks = build_mel_chunks(mel, fps=fps, mel_step_size=mel_step_size)
    n_positions = min(len(frames), len(mel_chunks))
    if n_positions <= 0:
        raise RuntimeError(f"No sync positions available for source-crop sample {source_crop_spec['label']}")
    return {
        "label": source_crop_spec["label"],
        "path": source_video,
        "frames": frames,
        "mel_chunks": mel_chunks,
        "mel_total_steps": int(mel.shape[1]),
        "n_positions": n_positions,
        "input_kind": "source_crop",
        "source_video": source_video,
        "detections_json": source_crop_spec["detections_json"],
        "trimmed_meta_json": source_crop_spec["trimmed_meta_json"],
    }


def choose_spaced_starts(starts, samples: int, min_gap_frames: int, seed: int):
    if len(starts) < samples:
        return sorted(random.Random(seed).choices(starts, k=samples))
    if min_gap_frames <= 0:
        rng = random.Random(seed)
        if len(starts) >= samples:
            return sorted(rng.sample(starts, samples))
        return sorted(rng.choice(starts) for _ in range(samples))

    best = []
    for trial in range(256):
        rng = random.Random(seed + trial)
        shuffled = starts[:]
        rng.shuffle(shuffled)
        chosen = []
        for start in shuffled:
            if all(abs(start - prev) >= min_gap_frames for prev in chosen):
                chosen.append(start)
                if len(chosen) == samples:
                    return sorted(chosen)
        if len(chosen) > len(best):
            best = chosen
    raise RuntimeError(
        f"Unable to choose {samples} starts with min_gap_frames={min_gap_frames}; "
        f"best_found={len(best)}"
    )


def choose_spaced_starts_with_gap_multiple(
    starts,
    samples: int,
    min_gap_frames: int,
    seed: int,
    gap_multiple: int,
):
    if gap_multiple <= 1:
        return choose_spaced_starts(starts, samples, min_gap_frames, seed)

    residue_buckets = []
    for residue in range(gap_multiple):
        bucket = [start for start in starts if start % gap_multiple == residue]
        if bucket:
            residue_buckets.append((residue, bucket))

    if not residue_buckets:
        raise RuntimeError(f"No candidate starts available for gap_multiple={gap_multiple}")

    feasible = []
    best_residue = None
    best_bucket_size = -1
    best_error = None
    for residue, bucket in residue_buckets:
        if len(bucket) > best_bucket_size:
            best_bucket_size = len(bucket)
            best_residue = residue
        try:
            chosen = choose_spaced_starts(
                bucket,
                samples,
                min_gap_frames,
                seed + residue * 9973,
            )
            feasible.append((residue, chosen))
        except RuntimeError as exc:
            best_error = exc

    if feasible:
        feasible.sort(key=lambda item: item[0])
        rng = random.Random(seed)
        return list(rng.choice(feasible)[1])

    raise RuntimeError(
        "Unable to choose starts satisfying both spacing constraints; "
        f"gap_multiple={gap_multiple}, best_residue={best_residue}, "
        f"best_bucket_size={best_bucket_size}, last_error={best_error}"
    )


def choose_shared_starts(
    video_samples,
    offsets,
    T,
    samples,
    seed,
    min_gap_ratio: float = 0.0,
    start_gap_multiple: int = 0,
):
    max_abs_shift = max(abs(v) for v in offsets)
    common_positions = min(sample["n_positions"] for sample in video_samples)
    min_start = max_abs_shift
    max_start = common_positions - T - max_abs_shift
    if max_start < min_start:
        raise RuntimeError(
            f"Not enough shared frames for offsets up to {max_abs_shift}: "
            f"common_positions={common_positions}, T={T}"
        )
    starts = list(range(min_start, max_start + 1))
    min_gap_frames = 0
    if min_gap_ratio > 0:
        min_gap_frames = max(1, int(math.ceil(common_positions * float(min_gap_ratio))))
    return choose_spaced_starts_with_gap_multiple(
        starts,
        samples,
        min_gap_frames,
        seed,
        start_gap_multiple,
    )


def evaluate_video(model, device, sample, starts, offsets, T, store_sample_values: bool = False):
    results = {}
    with torch.no_grad():
        for offset in offsets:
            losses = []
            cosines = []
            sample_rows = []
            for start in starts:
                audio_start = start
                visual_start = start + offset
                visual = build_visual_tensor(sample["frames"], visual_start, T, device)
                audio = build_audio_tensor(sample["mel_chunks"], audio_start, device)
                audio_emb, video_emb = model(audio, visual)
                cos = F.cosine_similarity(audio_emb, video_emb)
                loss = official_sync_loss_from_cosine(cos)
                loss_value = float(loss.item())
                cosine_value = float(cos.mean().item())
                losses.append(loss_value)
                cosines.append(cosine_value)
                if store_sample_values:
                    sample_rows.append(
                        {
                            "start": int(start),
                            "audio_start": int(audio_start),
                            "visual_start": int(visual_start),
                            "loss": loss_value,
                            "cosine": cosine_value,
                        }
                    )
            row = {
                "offset": int(offset),
                "mean_loss": mean(losses),
                "std_loss": std(losses),
                "mean_cosine": mean(cosines),
                "std_cosine": std(cosines),
                "num_samples": len(losses),
            }
            if store_sample_values:
                row["sample_values"] = sample_rows
            results[str(offset)] = row
    ranked = sorted(results.values(), key=lambda row: row["mean_loss"])
    best_offset = ranked[0]["offset"] if ranked else None
    zero_rank = None
    for idx, row in enumerate(ranked, start=1):
        if row["offset"] == 0:
            zero_rank = idx
            break
    return {
        "video": sample["path"],
        "label": sample["label"],
        "n_frames": len(sample["frames"]),
        "n_mel_chunks": len(sample["mel_chunks"]),
        "n_positions": sample["n_positions"],
        "metrics_by_offset": results,
        "ranked_by_mean_loss": ranked,
        "best_offset_by_mean_loss": best_offset,
        "zero_offset_rank_by_mean_loss": zero_rank,
    }


def main():
    args = parse_args()
    video_specs = parse_video_specs(args.video)
    source_crop_specs = parse_source_crop_specs(args.source_crop)
    if not video_specs and not source_crop_specs:
        raise RuntimeError("Provide at least one --video or --source-crop input")
    offsets = [int(x.strip()) for x in args.offsets.split(",") if x.strip()]
    device = resolve_device(args.device)

    log(f"Loading official SyncNet from {args.syncnet} on {device}")
    model = load_syncnet(args.syncnet, device)
    audio_proc = AudioProcessor(DEFAULT_AUDIO_CFG)

    log("Decoding videos + audio...")
    video_samples = [
        prepare_video_sample(spec, audio_proc, args.ffmpeg_bin, args.fps, args.mel_step_size)
        for spec in video_specs
    ] + [
        prepare_source_crop_sample(spec, audio_proc, args.ffmpeg_bin, args.fps, args.mel_step_size)
        for spec in source_crop_specs
    ]

    starts = choose_shared_starts(
        video_samples,
        offsets,
        args.syncnet_T,
        args.samples,
        args.seed,
        min_gap_ratio=args.min_start_gap_ratio,
        start_gap_multiple=args.start_gap_multiple,
    )
    log(
        f"Using {len(starts)} shared random starts "
        f"(seed={args.seed}, min={min(starts)}, max={max(starts)})"
    )

    payload = {
        "syncnet_checkpoint": args.syncnet,
        "device": device,
        "seed": int(args.seed),
        "samples": int(args.samples),
        "fps": float(args.fps),
        "mel_step_size": int(args.mel_step_size),
        "syncnet_T": int(args.syncnet_T),
        "offsets": offsets,
        "min_start_gap_ratio": float(args.min_start_gap_ratio),
        "start_gap_multiple": int(args.start_gap_multiple),
        "shared_starts": starts,
        "videos": [],
    }

    for sample in video_samples:
        log(f"Evaluating {sample['label']}...")
        result = evaluate_video(
            model,
            device,
            sample,
            starts,
            offsets,
            args.syncnet_T,
            store_sample_values=args.store_sample_values,
        )
        payload["videos"].append(result)
        best = result["best_offset_by_mean_loss"]
        zero_rank = result["zero_offset_rank_by_mean_loss"]
        zero_loss = result["metrics_by_offset"]["0"]["mean_loss"]
        log(
            f"  best_offset={best} zero_rank={zero_rank} "
            f"zero_mean_loss={zero_loss:.6f}"
        )

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    log(f"Saved results to {out_path}")


if __name__ == "__main__":
    main()
