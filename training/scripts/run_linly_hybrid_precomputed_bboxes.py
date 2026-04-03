#!/usr/bin/env python3
"""
Hybrid x96-preprocess + Linly 256 benchmark runner using precomputed bbox manifest.

Expected manifest format:
{
  "video_width": ...,
  "video_height": ...,
  "frames": [{"x1": ..., "y1": ..., "x2": ..., "y2": ...}, ...]
}
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from run_linly_hybrid_x96_benchmark import (
    DEFAULT_AUDIO_CFG,
    IMG_SIZE,
    MEL_STEP_SIZE,
    datagen,
    get_mel_chunks,
    load_linly_model,
    read_frames,
    resolve_device,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Hybrid x96-preprocess + Linly 256 benchmark runner using precomputed bbox manifest"
    )
    parser.add_argument("--face", required=True, help="Video or image file with a face")
    parser.add_argument("--audio", required=True, help="Audio file (wav/mp3/...)")
    parser.add_argument("--checkpoint", required=True, help="Linly Wav2Lipv2 checkpoint")
    parser.add_argument("--manifest-in", required=True, help="Input bbox manifest path")
    parser.add_argument("--outfile", required=True, help="Output video path")
    parser.add_argument("--fps", type=float, default=25.0, help="FPS for static image input")
    parser.add_argument("--batch_size", type=int, default=16, help="Inference batch size")
    parser.add_argument("--static", action="store_true", help="Use only first frame")
    parser.add_argument("--resize_factor", type=int, default=1, help="Downscale input by this factor")
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "mps", "cuda"),
        default="auto",
        help="Inference device for Linly model",
    )
    return parser.parse_args()


def load_face_det_results_from_manifest(manifest_path: str, frames):
    payload = json.loads(Path(manifest_path).read_text())
    frame_entries = payload["frames"]
    if not frame_entries:
        raise ValueError("Manifest contains no frames")

    expected_w = int(payload.get("video_width") or frames[0].shape[1])
    expected_h = int(payload.get("video_height") or frames[0].shape[0])
    actual_h, actual_w = frames[0].shape[:2]
    if expected_w != actual_w or expected_h != actual_h:
        raise ValueError(
            f"Manifest size {expected_w}x{expected_h} does not match video size {actual_w}x{actual_h}"
        )

    out = []
    total = len(frames)
    for idx in range(total):
        entry = frame_entries[0 if len(frame_entries) == 1 else idx]
        x1 = int(round(entry["x1"]))
        y1 = int(round(entry["y1"]))
        x2 = int(round(entry["x2"]))
        y2 = int(round(entry["y2"]))
        x1 = max(0, min(x1, actual_w - 1))
        y1 = max(0, min(y1, actual_h - 1))
        x2 = max(x1 + 1, min(x2, actual_w))
        y2 = max(y1 + 1, min(y2, actual_h))
        out.append((frames[idx][y1:y2, x1:x2], (y1, y2, x1, x2)))
    return out


def main():
    args = parse_args()
    t_start = time.time()

    device = resolve_device(args.device)

    print("=" * 60)
    print("Hybrid Linly Benchmark + Precomputed BBoxes")
    print(f"Device: {device}")
    print("=" * 60)

    is_image = Path(args.face).suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    if is_image:
        args.static = True

    frames, video_fps = read_frames(args.face, args.resize_factor)
    fps = video_fps or args.fps

    model = load_linly_model(args.checkpoint, device=device)
    mel_chunks = get_mel_chunks(
        args.audio,
        fps,
        audio_cfg=DEFAULT_AUDIO_CFG,
        mel_step_size=MEL_STEP_SIZE,
    )
    if not args.static:
        frames = frames[: len(mel_chunks)]
        print(f"[sync] Trimmed to {len(frames)} frames for {len(mel_chunks)} mel chunks")

    face_det_results = load_face_det_results_from_manifest(args.manifest_in, frames)
    print(f"[manifest] Loaded {len(face_det_results)} bbox entries from {args.manifest_in}")

    Path(args.outfile).parent.mkdir(parents=True, exist_ok=True)

    frame_h, frame_w = frames[0].shape[:2]
    fd, temp_avi = tempfile.mkstemp(prefix="linly_hybrid_bench_", suffix=".avi")
    os.close(fd)
    writer = cv2.VideoWriter(temp_avi, cv2.VideoWriter_fourcc(*"DIVX"), fps, (frame_w, frame_h))

    n_frames = 0
    t_inf = time.time()

    try:
        total_batches = int(np.ceil(len(mel_chunks) / args.batch_size))
        generator = datagen(frames, mel_chunks, face_det_results, args.batch_size, args.static)

        for img_batch, mel_batch, frame_batch, coords_batch in tqdm(
            generator, total=total_batches, desc="[infer] LinlyHybrid"
        ):
            img_tensor = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(device)
            mel_tensor = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(device)

            with torch.inference_mode():
                pred = model(mel_tensor, img_tensor)

            if isinstance(pred, tuple):
                pred = pred[0]
            pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.0

            for patch, frame, coords in zip(pred, frame_batch, coords_batch):
                y1, y2, x1, x2 = coords
                patch = cv2.resize(patch.astype(np.uint8), (x2 - x1, y2 - y1))
                frame[y1:y2, x1:x2] = patch
                writer.write(frame)
                n_frames += 1
    finally:
        writer.release()

    inf_time = time.time() - t_inf
    print(f"[infer] {n_frames} frames in {inf_time:.1f}s ({n_frames / inf_time:.1f} FPS)")

    print("[mux] Adding audio track...")
    subprocess.run(
        ["ffmpeg", "-y", "-i", temp_avi, "-i", args.audio, "-strict", "-2", "-q:v", "1", args.outfile],
        check=True,
        capture_output=True,
    )
    os.unlink(temp_avi)

    total_time = time.time() - t_start
    print("\n" + "=" * 60)
    print(f"Done! Total: {total_time:.1f}s")
    print(f"Output: {args.outfile}")
    print("=" * 60)


if __name__ == "__main__":
    main()
