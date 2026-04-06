#!/usr/bin/env python3
"""
Official Wav2Lip-style benchmark runner.

This is the report/reference benchmark path for lip-sync inference:
video/image -> SFD face detection -> 96x96 Wav2Lip -> paste back -> mux audio.

Weights are intentionally not stored in git. The runner supports:
- TorchScript SD checkpoints (.pt)
- regular Wav2Lip checkpoints (.pth)
- local generator checkpoints with `generator + config`

Examples:
    python training/scripts/run_official_wav2lip_benchmark.py \
        --face /abs/path/portrait_avatar.mp4 \
        --audio /abs/path/short_4s.mp3 \
        --checkpoint /abs/path/Wav2Lip-SD-GAN.pt
"""

from __future__ import annotations

import argparse
import hashlib
import json
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

REPO_ROOT = Path(__file__).resolve().parents[2]
TRAINING_ROOT = REPO_ROOT / "training"
OFFICIAL_SYNCNET_ROOT = REPO_ROOT / "models" / "official_syncnet"

for path in (str(REPO_ROOT), str(OFFICIAL_SYNCNET_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from training.data.audio import AudioProcessor
from training.models.generator import LipSyncGenerator
from models import Wav2Lip
import face_detection

IMG_SIZE = 96
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
CACHE_VERSION = 1


def parse_args():
    parser = argparse.ArgumentParser(description="Official Wav2Lip benchmark runner")
    parser.add_argument("--face", required=True, help="Video or image file with a face")
    parser.add_argument("--audio", required=True, help="Audio file (wav/mp3/...)")
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Official Wav2Lip or local generator checkpoint",
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
    parser.add_argument(
        "--cache_root",
        default=None,
        help="Optional cache root for decoded frames, face detections, and audio features",
    )
    parser.add_argument("--no_cache", action="store_true", help="Disable sample cache reuse")
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
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
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


def load_model(path: str, device: str):
    print(f"[model] Loading from {path} ...")
    t0 = time.time()
    if path.endswith(".pt"):
        model = torch.jit.load(path, map_location="cpu")
        model = model.to(device)
        model.eval()
        print(f"[model] Loaded TorchScript model in {time.time() - t0:.1f}s")
        return model, {
            "kind": "official",
            "img_size": IMG_SIZE,
            "mel_step_size": MEL_STEP_SIZE,
            "audio_cfg": DEFAULT_AUDIO_CFG,
            "syncnet_T": 1,
        }

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "generator" in ckpt and "config" in ckpt:
        cfg = ckpt["config"]
        model = LipSyncGenerator(
            img_size=cfg["model"]["img_size"],
            base_channels=cfg["model"]["base_channels"],
            predict_alpha=cfg["model"]["predict_alpha"],
        )
        model.load_state_dict(ckpt["generator"])
        model = model.to(device)
        model.eval()
        print(f"[model] Loaded local generator checkpoint in {time.time() - t0:.1f}s")
        return model, {
            "kind": "custom",
            "img_size": int(cfg["model"]["img_size"]),
            "mel_step_size": int(cfg["model"].get("mel_steps", MEL_STEP_SIZE)),
            "audio_cfg": cfg["audio"],
            "syncnet_T": int(cfg.get("syncnet", {}).get("T", 5)),
        }

    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    state = {k.replace("module.", ""): v for k, v in state.items()}
    model = Wav2Lip()
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()
    print(f"[model] Loaded checkpoint in {time.time() - t0:.1f}s")
    return model, {
        "kind": "official",
        "img_size": IMG_SIZE,
        "mel_step_size": MEL_STEP_SIZE,
        "audio_cfg": DEFAULT_AUDIO_CFG,
        "syncnet_T": 1,
    }


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
    candidate = REPO_ROOT / "models" / "official_syncnet" / "checkpoints" / "s3fd.pth"
    if candidate.exists():
        return str(candidate)
    return None


def default_cache_root() -> Path:
    return TRAINING_ROOT / "output" / "checkpoint_benchmark_cache"


def file_sha256(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(payload: dict) -> str:
    data = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_frames_cached(face_path: str, resize_factor: int, cache_root: Path | None):
    if cache_root is None:
        return read_frames(face_path, resize_factor), None

    face_hash = file_sha256(face_path)
    cache_key = stable_hash(
        {
            "kind": "frames",
            "version": CACHE_VERSION,
            "face_hash": face_hash,
            "resize_factor": int(resize_factor),
        }
    )
    cache_dir = cache_root / "frames" / cache_key
    frames_path = cache_dir / "frames.npy"
    meta_path = cache_dir / "meta.json"

    if frames_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        frames = np.load(frames_path, allow_pickle=False)
        print(f"[cache] frames hit: {cache_dir}")
        return (list(frames), meta.get("fps")), face_hash

    frames, fps = read_frames(face_path, resize_factor)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(frames_path, np.asarray(frames, dtype=np.uint8), allow_pickle=False)
    save_json(
        meta_path,
        {
            "kind": "frames",
            "version": CACHE_VERSION,
            "face_path": str(face_path),
            "face_hash": face_hash,
            "resize_factor": int(resize_factor),
            "fps": fps,
            "frame_count": int(len(frames)),
        },
    )
    print(f"[cache] frames store: {cache_dir}")
    return (frames, fps), face_hash


def rebuild_face_det_results(images, coords: np.ndarray):
    out = []
    for image, rect in zip(images, coords):
        y1, y2, x1, x2 = [int(v) for v in rect.tolist()]
        out.append((image[y1:y2, x1:x2], (y1, y2, x1, x2)))
    return out


def face_detect_sfd_cached(
    images,
    pads,
    nosmooth: bool,
    batch_size: int,
    detector_device: str,
    s3fd_path: str | None,
    cache_root: Path | None,
    face_hash: str,
    static: bool,
):
    if cache_root is None:
        return face_detect_sfd(images, pads, nosmooth, batch_size, detector_device, s3fd_path)

    frame_shape = list(images[0].shape[:2]) if images else None
    cache_key = stable_hash(
        {
            "kind": "face_det",
            "version": CACHE_VERSION,
            "face_hash": face_hash,
            "static": bool(static),
            "pads": [int(v) for v in pads],
            "nosmooth": bool(nosmooth),
            "frame_count": int(len(images)),
            "frame_shape": frame_shape,
            "detector": "sfd",
        }
    )
    cache_dir = cache_root / "face_det" / cache_key
    coords_path = cache_dir / "coords.npy"
    meta_path = cache_dir / "meta.json"

    if coords_path.exists() and meta_path.exists():
        coords = np.load(coords_path, allow_pickle=False)
        print(f"[cache] face_det hit: {cache_dir}")
        return rebuild_face_det_results(images, coords)

    face_det_results = face_detect_sfd(images, pads, nosmooth, batch_size, detector_device, s3fd_path)
    coords = np.asarray([coords for _face, coords in face_det_results], dtype=np.int32)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(coords_path, coords, allow_pickle=False)
    save_json(
        meta_path,
        {
            "kind": "face_det",
            "version": CACHE_VERSION,
            "face_hash": face_hash,
            "static": bool(static),
            "pads": [int(v) for v in pads],
            "nosmooth": bool(nosmooth),
            "frame_count": int(len(images)),
            "frame_shape": frame_shape,
            "detector": "sfd",
        },
    )
    print(f"[cache] face_det store: {cache_dir}")
    return face_det_results


def get_mel_chunks_cached(
    audio_path: str,
    fps: float,
    audio_cfg: dict,
    mel_step_size: int,
    cache_root: Path | None,
):
    if cache_root is None:
        return get_mel_chunks(audio_path, fps, audio_cfg, mel_step_size), None

    audio_hash = file_sha256(audio_path)
    cache_key = stable_hash(
        {
            "kind": "audio",
            "version": CACHE_VERSION,
            "audio_hash": audio_hash,
            "fps": float(fps),
            "mel_step_size": int(mel_step_size),
            "audio_cfg": audio_cfg,
        }
    )
    cache_dir = cache_root / "audio" / cache_key
    mel_chunks_path = cache_dir / "mel_chunks.npy"
    meta_path = cache_dir / "meta.json"

    if mel_chunks_path.exists() and meta_path.exists():
        mel_chunks = np.load(mel_chunks_path, allow_pickle=False)
        print(f"[cache] audio hit: {cache_dir}")
        return mel_chunks, audio_hash

    mel_chunks = get_mel_chunks(audio_path, fps, audio_cfg, mel_step_size)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(mel_chunks_path, np.asarray(mel_chunks, dtype=np.float32), allow_pickle=False)
    save_json(
        meta_path,
        {
            "kind": "audio",
            "version": CACHE_VERSION,
            "audio_path": str(audio_path),
            "audio_hash": audio_hash,
            "fps": float(fps),
            "mel_step_size": int(mel_step_size),
            "audio_cfg": audio_cfg,
            "mel_chunk_count": int(len(mel_chunks)),
        },
    )
    print(f"[cache] audio store: {cache_dir}")
    return mel_chunks, audio_hash


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
    img_masked[:, IMG_SIZE // 2:] = 0

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


def window_indices(center_idx: int, total: int, window: int):
    half = window // 2
    return [min(max(center_idx - half + offset, 0), total - 1) for offset in range(window)]


def prepare_custom_batch(face_crops, mel_chunks, indices, img_size: int, syncnet_T: int, device: str):
    face_batch = []
    mel_batch = []
    half = syncnet_T // 2

    for center_idx in indices:
        frame_indices = window_indices(center_idx, len(face_crops), syncnet_T)
        start = center_idx - half
        mel_indices = [
            min(max(start - 1 + offset, 0), len(mel_chunks) - 1)
            for offset in range(syncnet_T)
        ]

        target_window = np.stack(
            [face_crops[frame_idx].astype(np.float32) / 255.0 for frame_idx in frame_indices],
            axis=0,
        )
        masked_target = target_window.copy()
        masked_target[:, img_size // 2 :, :, :] = 0

        face_input = np.concatenate(
            [masked_target.transpose(3, 0, 1, 2), target_window.transpose(3, 0, 1, 2)],
            axis=0,
        )
        indiv_mels = np.stack([mel_chunks[idx] for idx in mel_indices], axis=0)[:, np.newaxis]

        face_batch.append(face_input)
        mel_batch.append(indiv_mels)

    face_tensor = torch.from_numpy(np.ascontiguousarray(np.array(face_batch, dtype=np.float32))).to(device)
    mel_tensor = torch.from_numpy(np.ascontiguousarray(np.array(mel_batch, dtype=np.float32))).to(device)
    return face_tensor, mel_tensor


def default_output_path(args) -> str:
    out_dir = REPO_ROOT / "training" / "output" / "benchmarks" / "wav2lip"
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

    print("=" * 50)
    print("Official Wav2Lip Benchmark")
    print(f"Device: {device}")
    print(f"Detector: SFD on {detector_device}")
    print("=" * 50)

    is_image = Path(args.face).suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    if is_image:
        args.static = True

    cache_root = None if args.no_cache else (Path(args.cache_root).resolve() if args.cache_root else default_cache_root())
    if cache_root is not None:
        cache_root.mkdir(parents=True, exist_ok=True)
        print(f"[cache] root: {cache_root}")

    (frames, video_fps), face_hash = load_frames_cached(args.face, args.resize_factor, cache_root)
    fps = video_fps or args.fps

    model, model_meta = load_model(args.checkpoint, device=device)
    mel_chunks, _audio_hash = get_mel_chunks_cached(
        args.audio,
        fps,
        audio_cfg=model_meta["audio_cfg"],
        mel_step_size=model_meta["mel_step_size"],
        cache_root=cache_root,
    )
    if not args.static:
        frames = frames[: len(mel_chunks)]
        print(f"[sync] Trimmed to {len(frames)} frames for {len(mel_chunks)} mel chunks")

    t_fd = time.time()
    s3fd_path = resolve_s3fd_path(args.s3fd_path)
    detection_frames = [frames[0]] if args.static else frames
    face_det_results = face_detect_sfd_cached(
        detection_frames,
        pads=args.pads,
        nosmooth=args.nosmooth,
        batch_size=args.face_det_batch_size,
        detector_device=detector_device,
        s3fd_path=s3fd_path,
        cache_root=cache_root,
        face_hash=face_hash or file_sha256(args.face),
        static=args.static,
    )
    print(f"[face_det] Total: {time.time() - t_fd:.1f}s")

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
    print("=" * 50)


if __name__ == "__main__":
    main()
