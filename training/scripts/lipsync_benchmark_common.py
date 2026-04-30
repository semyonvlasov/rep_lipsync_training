from __future__ import annotations

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

from models import Wav2Lip
from training.data.audio import AudioProcessor
from training.models.generator import LipSyncGenerator

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


def load_model(path: str, device: str):
    print(f"[model] Loading from {path} ...", flush=True)
    t0 = time.time()
    if path.endswith(".pt"):
        model = torch.jit.load(path, map_location="cpu")
        model = model.to(device)
        model.eval()
        print(f"[model] Loaded TorchScript model in {time.time() - t0:.1f}s", flush=True)
        return model, {
            "kind": "official",
            "img_size": IMG_SIZE,
            "mel_step_size": MEL_STEP_SIZE,
            "audio_cfg": DEFAULT_AUDIO_CFG,
            "syncnet_T": 1,
        }

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, torch.jit.RecursiveScriptModule):
        model = ckpt.to(device)
        model.eval()
        print(f"[model] Loaded TorchScript model in {time.time() - t0:.1f}s", flush=True)
        return model, {
            "kind": "official",
            "img_size": IMG_SIZE,
            "mel_step_size": MEL_STEP_SIZE,
            "audio_cfg": DEFAULT_AUDIO_CFG,
            "syncnet_T": 1,
        }
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
        print(f"[model] Loaded local generator checkpoint in {time.time() - t0:.1f}s", flush=True)
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
    print(f"[model] Loaded checkpoint in {time.time() - t0:.1f}s", flush=True)
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
        print(f"[read] Loaded image {image.shape[1]}x{image.shape[0]}", flush=True)
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
    print(f"[read] {len(frames)} frames @ {fps:.1f} FPS, {width}x{height}", flush=True)
    return frames, fps


def default_cache_root() -> Path:
    return TRAINING_ROOT / "output" / "benchmark_cache" / "lipsync"


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
        print(f"[cache] frames hit: {cache_dir}", flush=True)
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
    print(f"[cache] frames store: {cache_dir}", flush=True)
    return (frames, fps), face_hash


def get_mel_chunks(audio_path: str, fps: float, audio_cfg: dict, mel_step_size: int):
    print(f"[audio] Processing {audio_path} ...", flush=True)
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
        print("[audio] Converted to wav", flush=True)

    try:
        wav = audio_processor.load_wav(wav_path)
        mel = audio_processor.melspectrogram(wav)
    finally:
        if temp_wav and os.path.exists(temp_wav):
            os.unlink(temp_wav)

    print(f"[audio] Mel spectrogram: {mel.shape}", flush=True)
    if np.isnan(mel).any():
        raise ValueError("Mel contains NaN")

    chunks = audio_processor.mel_chunks(mel, fps, mel_step_size=mel_step_size)
    print(f"[audio] {len(chunks)} mel chunks @ {fps} FPS ({time.time() - t0:.1f}s)", flush=True)
    return chunks


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
        print(f"[cache] audio hit: {cache_dir}", flush=True)
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
    print(f"[cache] audio store: {cache_dir}", flush=True)
    return mel_chunks, audio_hash


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
