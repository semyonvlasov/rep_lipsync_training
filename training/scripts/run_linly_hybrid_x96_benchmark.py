#!/usr/bin/env python3
"""
Hybrid benchmark:
official x96 Wav2Lip preprocessing/inference path, but with Linly Wav2Lipv2.

What stays from the x96 benchmark path:
- SFD face detection
- bbox smoothing
- face crop taken directly from the x96 bbox
- paste-back into the same bbox
- mel chunking / 1 mel chunk -> 1 output frame

What changes:
- face crop is resized to 256x256 instead of 96x96
- inference runs through the local Linly Wav2Lipv2 checkpoint
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

TRAINING_REPO_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
OFFICIAL_SYNCNET_ROOT = TRAINING_REPO_ROOT / "models" / "official_syncnet"
LINLY_ROOT = WORKSPACE_ROOT / "third_party" / "linly_wav2lipv2"
LINLY_SRC_ROOT = LINLY_ROOT / "src"

for path in (
    str(TRAINING_REPO_ROOT),
    str(OFFICIAL_SYNCNET_ROOT),
    str(LINLY_SRC_ROOT),
):
    if path not in sys.path:
        sys.path.insert(0, path)

from training.data.audio import AudioProcessor
from modelsv2 import Wav2Lip as LinlyWav2Lip
import face_detection

IMG_SIZE = 256
MEL_STEP_SIZE = 16
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


def parse_args():
    parser = argparse.ArgumentParser(description="Hybrid x96-preprocess + Linly 256 benchmark runner")
    parser.add_argument("--face", required=True, help="Video or image file with a face")
    parser.add_argument("--audio", required=True, help="Audio file (wav/mp3/...)")
    parser.add_argument(
        "--checkpoint",
        default=str(LINLY_ROOT / "checkpoints" / "wav2lipv2.pth"),
        help="Linly Wav2Lipv2 checkpoint",
    )
    parser.add_argument("--outfile", default=None, help="Output video path")
    parser.add_argument("--fps", type=float, default=25.0, help="FPS for static image input")
    parser.add_argument(
        "--pads",
        nargs=4,
        type=int,
        default=[0, 10, 0, 0],
        metavar=("TOP", "BOTTOM", "LEFT", "RIGHT"),
        help="Face detection padding: top bottom left right",
    )
    parser.add_argument("--batch_size", type=int, default=16, help="Inference batch size")
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
        help="Inference device for Linly model",
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


def resolve_device(requested: str) -> str:
    if requested == "cpu":
        return "cpu"
    if requested == "mps":
        if torch.backends.mps.is_available():
            return "mps"
        raise RuntimeError("MPS requested but not available")
    if requested == "cuda":
        if torch.cuda.is_available():
            return "cuda"
        raise RuntimeError("CUDA requested but not available")
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def patch_face_detector_for_mps():
    from face_detection.detection.core import FaceDetector as _FaceDetector

    original_init = _FaceDetector.__init__

    def _patched_init(self, device, verbose):
        self.device = device
        self.verbose = verbose
        if verbose and "cpu" in device:
            import logging

            logger = logging.getLogger(__name__)
            logger.warning("Detection running on CPU, this may be potentially slow.")

    _FaceDetector.__init__ = _patched_init
    return _FaceDetector, original_init


def load_linly_model(path: str, device: str):
    print(f"[model] Loading Linly checkpoint from {path} ...")
    t0 = time.time()
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    clean_state = {key.replace("module.", ""): value for key, value in state_dict.items()}
    model = LinlyWav2Lip()
    model.load_state_dict(clean_state)
    model = model.to(device)
    model.eval()
    print(f"[model] Loaded in {time.time() - t0:.1f}s")
    return model


def read_frames(face_path: str, resize_factor: int):
    ext = os.path.splitext(face_path)[1].lower()
    if ext in (".jpg", ".jpeg", ".png", ".bmp"):
        image = cv2.imread(face_path)
        if image is None:
            raise FileNotFoundError(face_path)
        print(f"[read] Loaded image {image.shape[1]}x{image.shape[0]}")
        return [image], None

    cap = cv2.VideoCapture(face_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for _ in tqdm(range(total), desc="[read] Loading frames"):
        ok, frame = cap.read()
        if not ok:
            break
        if resize_factor > 1:
            frame = cv2.resize(frame, (frame.shape[1] // resize_factor, frame.shape[0] // resize_factor))
        frames.append(frame)
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames read from {face_path}")
    height, width = frames[0].shape[:2]
    print(f"[read] {len(frames)} frames @ {fps:.1f} FPS, {width}x{height}")
    return frames, fps


def get_smoothened_boxes(boxes: np.ndarray, window: int = 5) -> np.ndarray:
    smoothed = boxes.copy()
    for index in range(len(smoothed)):
        start = max(0, index - window // 2)
        end = min(len(smoothed), index + window // 2 + 1)
        smoothed[index] = np.mean(boxes[start:end], axis=0)
    return smoothed


def resolve_s3fd_path(user_path: str | None) -> str | None:
    if user_path:
        return user_path
    candidate = WORKSPACE_ROOT / "models" / "official_syncnet" / "checkpoints" / "s3fd.pth"
    if candidate.exists():
        return str(candidate)
    return None


def face_detect_sfd(images, pads, nosmooth: bool, batch_size: int, detector_device: str, s3fd_path: str | None):
    print(f"[face_det] Initializing SFD detector on {detector_device}...")
    face_detector_cls, original_init = patch_face_detector_for_mps()
    try:
        from face_detection.detection.sfd import FaceDetector as SFDDetector

        detector_kwargs = {"device": detector_device, "verbose": False}
        if s3fd_path:
            detector_kwargs["path_to_detector"] = s3fd_path
        detector = SFDDetector(**detector_kwargs)
    finally:
        face_detector_cls.__init__ = original_init

    predictions = []
    for idx in tqdm(range(0, len(images), batch_size), desc="[face_det] Detecting faces"):
        batch = np.array(images[idx:idx + batch_size])[..., ::-1].copy()
        predictions.extend(detector.detect_from_batch(batch))

    pady1, pady2, padx1, padx2 = pads
    boxes = []
    for detected, image in zip(predictions, images):
        if len(detected) == 0:
            raise ValueError("Face not detected in a frame")
        rect = np.asarray(detected[0], dtype=np.float32)
        x1, y1, x2, y2 = map(int, np.clip(rect[:4], 0, None))
        y1 = max(0, y1 - pady1)
        y2 = min(image.shape[0], y2 + pady2)
        x1 = max(0, x1 - padx1)
        x2 = min(image.shape[1], x2 + padx2)
        boxes.append([x1, y1, x2, y2])

    boxes = np.asarray(boxes, dtype=np.float32)
    if not nosmooth:
        boxes = get_smoothened_boxes(boxes, window=5)

    out = []
    for image, (x1, y1, x2, y2) in zip(images, boxes):
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        out.append((image[y1:y2, x1:x2], (y1, y2, x1, x2)))

    del detector
    print(f"[face_det] Done - {len(out)} detections")
    return out


def get_mel_chunks(audio_path: str, fps: float, audio_cfg: dict, mel_step_size: int):
    print(f"[audio] Processing {audio_path} ...")
    t0 = time.time()
    audio_processor = AudioProcessor(audio_cfg)

    wav_path = audio_path
    temp_wav = None
    if not audio_path.endswith(".wav"):
        fd, temp_wav = tempfile.mkstemp(prefix="wav2lip_audio_", suffix=".wav")
        os.close(fd)
        wav_path = temp_wav
        subprocess.run(
            ["ffmpeg", "-y", "-i", audio_path, "-ar", str(audio_cfg["sample_rate"]), "-ac", "1", wav_path],
            check=True,
            capture_output=True,
        )
        print("[audio] Converted to wav")

    try:
        wav = audio_processor.load_wav(wav_path)
        mel = audio_processor.melspectrogram(wav)
    finally:
        if temp_wav and os.path.exists(temp_wav):
            os.unlink(temp_wav)

    print(f"[audio] Mel spectrogram: {mel.shape}")
    if np.isnan(mel).any():
        raise ValueError("Mel contains NaN")

    chunks = audio_processor.mel_chunks(mel, fps, mel_step_size=mel_step_size)
    print(f"[audio] {len(chunks)} mel chunks @ {fps} FPS ({time.time() - t0:.1f}s)")
    return chunks


def make_batch(img_batch, mel_batch, frame_batch, coords_batch):
    img_batch = np.asarray(img_batch)
    mel_batch = np.asarray(mel_batch)

    img_masked = img_batch.copy()
    img_masked[:, IMG_SIZE // 2 :] = 0

    img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.0
    mel_batch = mel_batch.reshape(len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1)

    return img_batch, mel_batch, frame_batch, coords_batch


def datagen(frames, mels, face_det_results, batch_size: int, static: bool):
    img_batch = []
    mel_batch = []
    frame_batch = []
    coords_batch = []

    for index, mel in enumerate(mels):
        frame_index = 0 if static else index % len(frames)
        frame_to_save = frames[frame_index].copy()
        face, coords = face_det_results[frame_index]
        face = cv2.resize(face.copy(), (IMG_SIZE, IMG_SIZE))

        img_batch.append(face)
        mel_batch.append(mel)
        frame_batch.append(frame_to_save)
        coords_batch.append(coords)

        if len(img_batch) >= batch_size:
            yield make_batch(img_batch, mel_batch, frame_batch, coords_batch)
            img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

    if img_batch:
        yield make_batch(img_batch, mel_batch, frame_batch, coords_batch)


def default_output_path(args) -> str:
    out_dir = TRAINING_REPO_ROOT / "training" / "output" / "benchmarks" / "linly_hybrid_x96_to_256"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_name = Path(args.checkpoint).stem
    audio_name = Path(args.audio).stem
    face_name = Path(args.face).stem
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return str(out_dir / f"{face_name}_{audio_name}_{ckpt_name}_{timestamp}.mp4")


def main():
    args = parse_args()
    t_start = time.time()

    device = resolve_device(args.device)
    detector_device = resolve_device(args.detector_device)

    print("=" * 60)
    print("Hybrid Linly Benchmark")
    print("Preprocess: official x96 benchmark path (SFD + bbox paste-back)")
    print("Model: Linly Wav2Lipv2 on 256x256 resized face crop")
    print(f"Device: {device}")
    print(f"Detector: SFD on {detector_device}")
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

    outfile = args.outfile or default_output_path(args)
    Path(outfile).parent.mkdir(parents=True, exist_ok=True)

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
        ["ffmpeg", "-y", "-i", temp_avi, "-i", args.audio, "-strict", "-2", "-q:v", "1", outfile],
        check=True,
        capture_output=True,
    )
    os.unlink(temp_avi)

    total_time = time.time() - t_start
    print("\n" + "=" * 60)
    print(f"Done! Total: {total_time:.1f}s")
    print(f"Output: {outfile}")
    print("=" * 60)


if __name__ == "__main__":
    main()
