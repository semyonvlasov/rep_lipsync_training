#!/usr/bin/env python3
"""
Official Wav2Lip benchmark runner with bbox manifest export.

Runs the same SFD -> Wav2Lip -> paste-back path as
`run_official_wav2lip_benchmark.py`, but also writes a face-data manifest:

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

from run_official_wav2lip_benchmark import (
    datagen,
    default_output_path,
    face_detect_sfd,
    get_mel_chunks,
    load_model,
    prepare_custom_batch,
    read_frames,
    resolve_device,
    resolve_s3fd_path,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Official Wav2Lip benchmark runner with bbox manifest export"
    )
    parser.add_argument("--face", required=True, help="Video or image file with a face")
    parser.add_argument("--audio", required=True, help="Audio file (wav/mp3/...)")
    parser.add_argument("--checkpoint", required=True, help="Official Wav2Lip checkpoint")
    parser.add_argument("--outfile", default=None, help="Output video path")
    parser.add_argument("--manifest-out", required=True, help="Output bbox manifest path")
    parser.add_argument("--fps", type=float, default=25.0, help="FPS for static image input")
    parser.add_argument(
        "--pads",
        nargs=4,
        type=int,
        default=[0, 10, 0, 0],
        metavar=("TOP", "BOTTOM", "LEFT", "RIGHT"),
        help="Face detection padding: top bottom left right",
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Inference batch size")
    parser.add_argument(
        "--face_det_batch_size",
        type=int,
        default=4,
        help="SFD face detection batch size",
    )
    parser.add_argument("--nosmooth", action="store_true", help="Disable bbox temporal smoothing")
    parser.add_argument("--static", action="store_true", help="Use only first frame")
    parser.add_argument("--resize_factor", type=int, default=1, help="Downscale input by this factor")
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "mps", "cuda"),
        default="auto",
        help="Inference device for Wav2Lip",
    )
    parser.add_argument(
        "--detector_device",
        choices=("auto", "cpu", "mps", "cuda"),
        default="auto",
        help="Device for SFD face detection",
    )
    parser.add_argument(
        "--s3fd_path",
        default=None,
        help="Optional path to s3fd.pth. If missing, bundled detector will download it.",
    )
    return parser.parse_args()


def write_manifest(manifest_path: str, frames, face_det_results) -> None:
    frame_entries = []
    total = len(frames)
    for idx in range(total):
        det_idx = 0 if len(face_det_results) == 1 else idx
        _face, coords = face_det_results[det_idx]
        y1, y2, x1, x2 = coords
        frame_entries.append(
            {
                "x1": float(x1),
                "y1": float(y1),
                "x2": float(x2),
                "y2": float(y2),
            }
        )

    payload = {
        "video_width": int(frames[0].shape[1]),
        "video_height": int(frames[0].shape[0]),
        "frames": frame_entries,
    }
    Path(manifest_path).parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2)


def main():
    args = parse_args()
    t_start = time.time()

    device = resolve_device(args.device)
    detector_device = resolve_device(args.detector_device)

    print("=" * 50)
    print("Official Wav2Lip Benchmark + Manifest")
    print(f"Device: {device}")
    print(f"Detector: SFD on {detector_device}")
    print("=" * 50)

    is_image = Path(args.face).suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    if is_image:
        args.static = True

    frames, video_fps = read_frames(args.face, args.resize_factor)
    fps = video_fps or args.fps

    model, model_meta = load_model(args.checkpoint, device=device)
    mel_chunks = get_mel_chunks(
        args.audio,
        fps,
        audio_cfg=model_meta["audio_cfg"],
        mel_step_size=model_meta["mel_step_size"],
    )
    if not args.static:
        frames = frames[: len(mel_chunks)]
        print(f"[sync] Trimmed to {len(frames)} frames for {len(mel_chunks)} mel chunks")

    t_fd = time.time()
    s3fd_path = resolve_s3fd_path(args.s3fd_path)
    detection_frames = [frames[0]] if args.static else frames
    face_det_results = face_detect_sfd(
        detection_frames,
        pads=args.pads,
        nosmooth=args.nosmooth,
        batch_size=args.face_det_batch_size,
        detector_device=detector_device,
        s3fd_path=s3fd_path,
    )
    print(f"[face_det] Total: {time.time() - t_fd:.1f}s")
    write_manifest(args.manifest_out, frames, face_det_results)
    print(f"[manifest] Wrote {args.manifest_out}")

    outfile = args.outfile or default_output_path(args)
    Path(outfile).parent.mkdir(parents=True, exist_ok=True)

    frame_h, frame_w = frames[0].shape[:2]
    fd, temp_avi = tempfile.mkstemp(prefix="wav2lip_bench_", suffix=".avi")
    os.close(fd)
    writer = cv2.VideoWriter(temp_avi, cv2.VideoWriter_fourcc(*"DIVX"), fps, (frame_w, frame_h))

    n_frames = 0
    t_inf = time.time()

    try:
        total_batches = int(np.ceil(len(mel_chunks) / args.batch_size))
        if model_meta["kind"] == "official":
            generator = datagen(frames, mel_chunks, face_det_results, args.batch_size, args.static)

            for img_batch, mel_batch, frame_batch, coords_batch in tqdm(
                generator, total=total_batches, desc="[infer] Wav2Lip"
            ):
                img_tensor = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(device)
                mel_tensor = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(device)

                with torch.inference_mode():
                    pred = model(mel_tensor, img_tensor)

                pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.0

                for patch, frame, coords in zip(pred, frame_batch, coords_batch):
                    y1, y2, x1, x2 = coords
                    patch = cv2.resize(patch.astype(np.uint8), (x2 - x1, y2 - y1))
                    frame[y1:y2, x1:x2] = patch
                    writer.write(frame)
                    n_frames += 1
        else:
            img_size = model_meta["img_size"]
            syncnet_T = model_meta["syncnet_T"]
            face_crops = [cv2.resize(face.copy(), (img_size, img_size)) for face, _coords in face_det_results]

            for start in tqdm(range(0, len(mel_chunks), args.batch_size), total=total_batches, desc="[infer] LocalGen"):
                indices = list(range(start, min(start + args.batch_size, len(mel_chunks))))
                source_indices = [0 if args.static else idx % len(face_crops) for idx in indices]
                face_tensor, mel_tensor = prepare_custom_batch(
                    face_crops,
                    mel_chunks,
                    source_indices,
                    img_size=img_size,
                    syncnet_T=syncnet_T,
                    device=device,
                )

                with torch.inference_mode():
                    pred = model(mel_tensor, face_tensor)

                if isinstance(pred, tuple):
                    pred = pred[0]
                if pred.dim() == 5:
                    pred = pred[:, :, pred.shape[2] // 2]
                pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.0

                for local_idx, patch in enumerate(pred):
                    frame_idx = source_indices[local_idx]
                    frame = frames[frame_idx].copy()
                    _face, coords = face_det_results[frame_idx]
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
        ["ffmpeg", "-y", "-i", temp_avi, "-i", args.audio, "-strict", "-2", "-q:v", "1", outfile],
        check=True,
        capture_output=True,
    )
    os.unlink(temp_avi)

    total_time = time.time() - t_start
    print("\n" + "=" * 50)
    print(f"Done! Total: {total_time:.1f}s")
    print(f"Output: {outfile}")
    print(f"Manifest: {args.manifest_out}")
    print("=" * 50)


if __name__ == "__main__":
    main()
