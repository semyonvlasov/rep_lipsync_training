"""
Helpers for storing and applying lazy faceclip audio/video sync alignment.

Manifest convention:
  - `audio_shift_mel_ticks` is applied directly to the nominal mel start for a
    video frame: `shifted_start = base_start + audio_shift_mel_ticks`.
  - Positive shift moves audio later, negative shift moves audio earlier.
  - Frames whose full mel chunk falls outside the available audio after the
    shift are considered invalid and excluded from training/eval sampling.
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import random
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F


TRAINING_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(TRAINING_ROOT)
WORKSPACE_ROOT = os.path.dirname(PROJECT_ROOT)


def _resolve_existing_path(*candidates: str) -> str:
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return candidates[0]


OFFICIAL_SYNCNET_MODELS_DIR = _resolve_existing_path(
    os.path.join(PROJECT_ROOT, "models", "official_syncnet", "models"),
    os.path.join(WORKSPACE_ROOT, "models", "official_syncnet", "models"),
)
DEFAULT_SYNCNET_CHECKPOINT = _resolve_existing_path(
    os.path.join(PROJECT_ROOT, "models", "official_syncnet", "checkpoints", "lipsync_expert.pth"),
    os.path.join(WORKSPACE_ROOT, "models", "official_syncnet", "checkpoints", "lipsync_expert.pth"),
    os.path.join(PROJECT_ROOT, "models", "wav2lip", "checkpoints", "lipsync_expert.pth"),
    os.path.join(WORKSPACE_ROOT, "models", "wav2lip", "checkpoints", "lipsync_expert.pth"),
)

SYNC_ALIGNMENT_VERSION = 1
DEFAULT_SYNC_ALIGNMENT_GUARD_MEL_TICKS = 10
DEFAULT_SYNC_ALIGNMENT_SEARCH_MEL_TICKS = 10
DEFAULT_SYNC_ALIGNMENT_SAMPLES = 0
DEFAULT_SYNC_ALIGNMENT_SEED = 20260403
DEFAULT_SYNC_ALIGNMENT_MIN_START_GAP_RATIO = 0.0
DEFAULT_SYNC_ALIGNMENT_START_GAP_MULTIPLE = 0
DEFAULT_SYNC_ALIGNMENT_BATCH_SIZE = 640
DEFAULT_SYNC_ALIGNMENT_SAMPLE_DENSITY_PER_5S = 10.0
DEFAULT_SYNC_ALIGNMENT_OUTLIER_TRIM_RATIO = 0.2
DEFAULT_SYNC_ALIGNMENT_MIN_CONSENSUS_RATIO = None
DEFAULT_SYNC_ALIGNMENT_MAX_SHIFT_MAD = None

_SYNCNET_MODEL_CACHE = {}


def _timestamp_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _atomic_write_json(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        prefix=os.path.basename(path) + ".",
        suffix=".tmp",
        dir=os.path.dirname(path),
    )
    os.close(fd)
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def resolve_sync_alignment_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested for sync alignment but not available")
    if device_arg not in {"cpu", "cuda"}:
        return "cpu"
    return device_arg


def default_sync_alignment_block(guard_mel_ticks: int = DEFAULT_SYNC_ALIGNMENT_GUARD_MEL_TICKS) -> dict:
    return {
        "version": SYNC_ALIGNMENT_VERSION,
        "status": "missing",
        "search_guard_mel_ticks": int(guard_mel_ticks),
    }


def _base_mel_start_for_frame(
    frame_idx: int,
    fps: float,
    mel_frames_per_second: float,
) -> int:
    return int(frame_idx * (float(mel_frames_per_second) / float(fps)))


def compute_valid_frame_range(
    *,
    n_frames: int,
    mel_total_steps: int,
    fps: float,
    mel_frames_per_second: float,
    mel_step_size: int,
    audio_shift_mel_ticks: int,
) -> tuple[int, int, int]:
    valid_start = None
    valid_end = None
    valid_count = 0

    for frame_idx in range(max(0, int(n_frames))):
        start = (
            _base_mel_start_for_frame(frame_idx, fps=fps, mel_frames_per_second=mel_frames_per_second)
            + int(audio_shift_mel_ticks)
        )
        end = start + int(mel_step_size)
        if start < 0 or end > int(mel_total_steps):
            continue
        if valid_start is None:
            valid_start = frame_idx
        valid_end = frame_idx
        valid_count += 1

    if valid_start is None:
        return 0, -1, 0
    return int(valid_start), int(valid_end), int(valid_count)


def build_shifted_frame_aligned_mels(
    mel: np.ndarray,
    *,
    n_frames: int,
    fps: float,
    mel_frames_per_second: float,
    mel_step_size: int,
    audio_shift_mel_ticks: int,
):
    chunks = []
    valid_indices = []
    mel_total_steps = int(mel.shape[1])

    for frame_idx in range(max(0, int(n_frames))):
        start = (
            _base_mel_start_for_frame(frame_idx, fps=fps, mel_frames_per_second=mel_frames_per_second)
            + int(audio_shift_mel_ticks)
        )
        end = start + int(mel_step_size)
        if start < 0 or end > mel_total_steps:
            continue
        chunks.append(mel[:, start:end].astype(np.float32, copy=False))
        valid_indices.append(frame_idx)

    return chunks, valid_indices


def build_sync_alignment_record(
    *,
    audio_shift_mel_ticks: int,
    n_frames: int,
    mel_total_steps: int,
    fps: float,
    mel_frames_per_second: float,
    mel_step_size: int,
    search_guard_mel_ticks: int,
    source: str,
    search_range_mel_ticks: int | None = None,
    search_samples: int | None = None,
    search_seed: int | None = None,
    min_start_gap_ratio: float | None = None,
    start_gap_multiple: int | None = None,
    best_mean_loss: float | None = None,
    zero_mean_loss: float | None = None,
    extra: dict | None = None,
) -> dict:
    valid_frame_start, valid_frame_end, valid_frame_count = compute_valid_frame_range(
        n_frames=n_frames,
        mel_total_steps=mel_total_steps,
        fps=fps,
        mel_frames_per_second=mel_frames_per_second,
        mel_step_size=mel_step_size,
        audio_shift_mel_ticks=audio_shift_mel_ticks,
    )

    payload = {
        "version": SYNC_ALIGNMENT_VERSION,
        "status": "aligned",
        "source": str(source),
        "audio_shift_mel_ticks": int(audio_shift_mel_ticks),
        "fps": float(fps),
        "mel_frames_per_second": float(mel_frames_per_second),
        "mel_step_size": int(mel_step_size),
        "mel_total_steps": int(mel_total_steps),
        "n_frames_total": int(n_frames),
        "valid_frame_start": int(valid_frame_start),
        "valid_frame_end": int(valid_frame_end),
        "valid_frame_count": int(valid_frame_count),
        "search_guard_mel_ticks": int(search_guard_mel_ticks),
        "computed_at": _timestamp_utc(),
    }
    if search_range_mel_ticks is not None:
        payload["search_range_mel_ticks"] = int(search_range_mel_ticks)
    if search_samples is not None:
        payload["search_samples"] = int(search_samples)
    if search_seed is not None:
        payload["search_seed"] = int(search_seed)
    if min_start_gap_ratio is not None:
        payload["min_start_gap_ratio"] = float(min_start_gap_ratio)
    if start_gap_multiple is not None:
        payload["start_gap_multiple"] = int(start_gap_multiple)
    if best_mean_loss is not None:
        payload["best_mean_loss"] = float(best_mean_loss)
    if zero_mean_loss is not None:
        payload["zero_mean_loss"] = float(zero_mean_loss)
    if extra:
        payload.update(extra)
    return payload


def build_failed_sync_alignment_record(
    *,
    n_frames: int | None = None,
    mel_total_steps: int | None = None,
    fps: float | None = None,
    mel_frames_per_second: float | None = None,
    mel_step_size: int | None = None,
    search_guard_mel_ticks: int | None = None,
    source: str,
    reason: str,
    error: str | None = None,
    extra: dict | None = None,
) -> dict:
    payload = {
        "version": SYNC_ALIGNMENT_VERSION,
        "status": "failed",
        "source": str(source),
        "reason": str(reason),
        "computed_at": _timestamp_utc(),
    }
    if error:
        payload["error"] = str(error)
    if n_frames is not None:
        payload["n_frames_total"] = int(n_frames)
    if mel_total_steps is not None:
        payload["mel_total_steps"] = int(mel_total_steps)
    if fps is not None:
        payload["fps"] = float(fps)
    if mel_frames_per_second is not None:
        payload["mel_frames_per_second"] = float(mel_frames_per_second)
    if mel_step_size is not None:
        payload["mel_step_size"] = int(mel_step_size)
    if search_guard_mel_ticks is not None:
        payload["search_guard_mel_ticks"] = int(search_guard_mel_ticks)
    if extra:
        payload.update(extra)
    return payload


def upsert_sync_alignment(
    meta: dict,
    *,
    audio_shift_mel_ticks: int,
    n_frames: int,
    mel_total_steps: int,
    fps: float,
    mel_frames_per_second: float,
    mel_step_size: int,
    search_guard_mel_ticks: int,
    source: str,
    search_range_mel_ticks: int | None = None,
    search_samples: int | None = None,
    search_seed: int | None = None,
    min_start_gap_ratio: float | None = None,
    start_gap_multiple: int | None = None,
    best_mean_loss: float | None = None,
    zero_mean_loss: float | None = None,
    extra: dict | None = None,
) -> dict:
    updated = dict(meta or {})
    updated["sync_alignment"] = build_sync_alignment_record(
        audio_shift_mel_ticks=audio_shift_mel_ticks,
        n_frames=n_frames,
        mel_total_steps=mel_total_steps,
        fps=fps,
        mel_frames_per_second=mel_frames_per_second,
        mel_step_size=mel_step_size,
        search_guard_mel_ticks=search_guard_mel_ticks,
        source=source,
        search_range_mel_ticks=search_range_mel_ticks,
        search_samples=search_samples,
        search_seed=search_seed,
        min_start_gap_ratio=min_start_gap_ratio,
        start_gap_multiple=start_gap_multiple,
        best_mean_loss=best_mean_loss,
        zero_mean_loss=zero_mean_loss,
        extra=extra,
    )
    return updated


def upsert_failed_sync_alignment(
    meta: dict,
    *,
    n_frames: int | None = None,
    mel_total_steps: int | None = None,
    fps: float | None = None,
    mel_frames_per_second: float | None = None,
    mel_step_size: int | None = None,
    search_guard_mel_ticks: int | None = None,
    source: str,
    reason: str,
    error: str | None = None,
    extra: dict | None = None,
) -> dict:
    updated = dict(meta or {})
    updated["sync_alignment"] = build_failed_sync_alignment_record(
        n_frames=n_frames,
        mel_total_steps=mel_total_steps,
        fps=fps,
        mel_frames_per_second=mel_frames_per_second,
        mel_step_size=mel_step_size,
        search_guard_mel_ticks=search_guard_mel_ticks,
        source=source,
        reason=reason,
        error=error,
        extra=extra,
    )
    return updated


def load_sync_alignment(meta: dict | None) -> dict | None:
    if not isinstance(meta, dict):
        return None
    payload = meta.get("sync_alignment")
    if not isinstance(payload, dict):
        return None
    if payload.get("status") != "aligned":
        return None
    if "audio_shift_mel_ticks" not in payload:
        return None
    return payload


def _load_official_syncnet_class():
    def load_module(name, filepath):
        spec = importlib.util.spec_from_file_location(name, filepath)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod

    conv_mod = load_module(
        "official_syncnet_models.conv",
        os.path.join(OFFICIAL_SYNCNET_MODELS_DIR, "conv.py"),
    )
    with open(os.path.join(OFFICIAL_SYNCNET_MODELS_DIR, "syncnet.py")) as f:
        syncnet_src = f.read()
    syncnet_src = syncnet_src.replace("from .conv import Conv2d", "")
    syncnet_ns = {"__builtins__": __builtins__}
    exec("import torch\nfrom torch import nn\nfrom torch.nn import functional as F\n", syncnet_ns)
    syncnet_ns["Conv2d"] = conv_mod.Conv2d
    exec(syncnet_src, syncnet_ns)
    return syncnet_ns["SyncNet_color"]


def load_official_syncnet(checkpoint_path: str, device: str):
    cache_key = (os.path.abspath(checkpoint_path), device)
    cached = _SYNCNET_MODEL_CACHE.get(cache_key)
    if cached is not None:
        return cached

    SyncNet = _load_official_syncnet_class()
    model = SyncNet().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    _SYNCNET_MODEL_CACHE[cache_key] = model
    return model


def _official_sync_loss_from_cosine(cos_sim: torch.Tensor) -> torch.Tensor:
    targets = torch.ones((cos_sim.size(0), 1), device=cos_sim.device, dtype=torch.float32)
    probs = cos_sim.float().clamp_(1.0e-6, 1.0 - 1.0e-6).unsqueeze(1)
    if cos_sim.device.type == "cuda":
        with torch.amp.autocast("cuda", enabled=False):
            return F.binary_cross_entropy(probs, targets, reduction="none").squeeze(1)
    return F.binary_cross_entropy(probs, targets, reduction="none").squeeze(1)


def _choose_spaced_starts(starts, samples: int, min_gap_frames: int, seed: int):
    if not starts:
        raise RuntimeError("No candidate starts available for sync alignment")

    if len(starts) < samples:
        return sorted(random.Random(seed).choices(starts, k=samples))
    if min_gap_frames <= 0:
        rng = random.Random(seed)
        return sorted(rng.sample(starts, samples))

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

    if best:
        return sorted(best)
    raise RuntimeError(
        f"Unable to choose {samples} starts with min_gap_frames={min_gap_frames}"
    )


def _choose_uniform_random_starts(starts, samples: int, seed: int):
    starts = sorted(int(v) for v in starts)
    if not starts:
        raise RuntimeError("No candidate starts available for sync alignment")
    if samples <= 0:
        return []
    if len(starts) <= samples:
        return starts

    rng = random.Random(seed)
    chosen = []
    total = len(starts)
    for idx in range(samples):
        left = int(math.floor(idx * total / float(samples)))
        right = int(math.floor((idx + 1) * total / float(samples)))
        bucket = starts[left:max(left + 1, right)]
        chosen.append(int(rng.choice(bucket)))
    return sorted(chosen)


def _choose_spaced_starts_with_gap_multiple(
    starts,
    samples: int,
    min_gap_frames: int,
    seed: int,
    gap_multiple: int,
):
    if gap_multiple <= 1:
        return _choose_uniform_random_starts(starts, samples, seed)

    residue_buckets = []
    for residue in range(gap_multiple):
        bucket = [start for start in starts if start % gap_multiple == residue]
        if bucket:
            residue_buckets.append((residue, bucket))

    if not residue_buckets:
        raise RuntimeError(f"No candidate starts available for gap_multiple={gap_multiple}")

    feasible = []
    for residue, bucket in residue_buckets:
        try:
            chosen = _choose_uniform_random_starts(
                bucket,
                samples,
                seed + residue * 9973,
            )
            if min_gap_frames > 0:
                ok = all(
                    abs(chosen[idx] - chosen[idx - 1]) >= min_gap_frames
                    for idx in range(1, len(chosen))
                )
                if not ok:
                    continue
            feasible.append((residue, chosen))
        except RuntimeError:
            continue

    if feasible:
        feasible.sort(key=lambda item: item[0])
        rng = random.Random(seed)
        return list(rng.choice(feasible)[1])

    raise RuntimeError(
        "Unable to choose starts satisfying both spacing constraints; "
        f"gap_multiple={gap_multiple}"
    )


def _resolve_sync_alignment_sample_count(
    *,
    n_frames: int,
    fps: float,
    requested_samples: int,
    density_per_5s: float,
    candidate_count: int,
    min_gap_frames: int = 0,
    candidate_starts=None,
):
    if requested_samples and requested_samples > 0:
        sample_count = min(int(requested_samples), int(candidate_count))
    else:
        duration_sec = max(0.0, float(n_frames) / float(fps))
        inferred = int(round((duration_sec / 5.0) * float(density_per_5s)))
        inferred = max(1, inferred)
        sample_count = min(inferred, int(candidate_count))

    if min_gap_frames > 0 and candidate_starts:
        max_by_gap = 1 + max(0, int(candidate_starts[-1]) - int(candidate_starts[0])) // int(min_gap_frames)
        sample_count = min(sample_count, max_by_gap)
    return max(1, int(sample_count))


def _trim_outlier_points_by_local_shift(
    losses: np.ndarray,
    audio_shifts,
    trim_ratio: float,
):
    n_points = int(losses.shape[0])
    if n_points <= 1 or trim_ratio <= 0.0:
        keep_mask = np.ones(n_points, dtype=bool)
        return keep_mask, np.zeros(n_points, dtype=np.int64), 0.0

    local_best_indices = np.argmin(losses, axis=1)
    local_best_shifts = np.asarray([int(audio_shifts[idx]) for idx in local_best_indices], dtype=np.int64)
    center_shift = float(np.median(local_best_shifts))
    deviations = np.abs(local_best_shifts.astype(np.float32) - center_shift)

    if deviations.size == 0 or float(np.max(deviations)) <= 0.0:
        keep_mask = np.ones(n_points, dtype=bool)
        return keep_mask, local_best_shifts, center_shift

    trim_count = int(math.floor(n_points * float(trim_ratio)))
    positive_dev_indices = [idx for idx, dev in enumerate(deviations.tolist()) if dev > 0.0]
    trim_count = min(trim_count, len(positive_dev_indices))
    if trim_count <= 0:
        keep_mask = np.ones(n_points, dtype=bool)
        return keep_mask, local_best_shifts, center_shift

    local_best_losses = losses[np.arange(n_points), local_best_indices]
    ranked = sorted(
        range(n_points),
        key=lambda idx: (deviations[idx], local_best_losses[idx]),
        reverse=True,
    )
    ranked = [idx for idx in ranked if deviations[idx] > 0.0]
    drop_indices = set(ranked[:trim_count])

    keep_mask = np.ones(n_points, dtype=bool)
    for idx in drop_indices:
        keep_mask[idx] = False
    return keep_mask, local_best_shifts, center_shift


def _compute_post_trim_shift_stats(
    kept_local_best_shifts: np.ndarray,
) -> tuple[float | None, float | None, float | None]:
    if kept_local_best_shifts.size == 0:
        return None, None, None

    center_shift = float(np.median(kept_local_best_shifts))
    abs_dev = np.abs(kept_local_best_shifts.astype(np.float32) - center_shift)
    shift_mad = float(np.median(abs_dev))
    consensus_ratio = float(np.mean(abs_dev <= 1.0))
    return center_shift, shift_mad, consensus_ratio


def _build_visual_batch(frames, starts, T: int, device: str) -> torch.Tensor:
    windows = []
    for start in starts:
        window = []
        for t in range(T):
            face = frames[start + t].astype(np.float32) / 255.0
            lower = face[face.shape[0] // 2 :, :, :]
            window.append(np.transpose(lower, (2, 0, 1)))
        windows.append(np.concatenate(window, axis=0))

    visual = torch.from_numpy(np.ascontiguousarray(np.stack(windows, axis=0))).to(device)
    visual = F.interpolate(visual, size=(48, 96), mode="bilinear", align_corners=False)
    return visual


def compute_sync_alignment_from_faceclip(
    *,
    frames,
    mel: np.ndarray,
    fps: float,
    mel_frames_per_second: float,
    mel_step_size: int,
    syncnet_T: int,
    checkpoint_path: str,
    device: str = "auto",
    search_mel_ticks: int = DEFAULT_SYNC_ALIGNMENT_SEARCH_MEL_TICKS,
    search_guard_mel_ticks: int = DEFAULT_SYNC_ALIGNMENT_GUARD_MEL_TICKS,
    samples: int = DEFAULT_SYNC_ALIGNMENT_SAMPLES,
    sample_density_per_5s: float = DEFAULT_SYNC_ALIGNMENT_SAMPLE_DENSITY_PER_5S,
    seed: int = DEFAULT_SYNC_ALIGNMENT_SEED,
    min_start_gap_ratio: float = DEFAULT_SYNC_ALIGNMENT_MIN_START_GAP_RATIO,
    start_gap_multiple: int = DEFAULT_SYNC_ALIGNMENT_START_GAP_MULTIPLE,
    batch_size: int = DEFAULT_SYNC_ALIGNMENT_BATCH_SIZE,
    outlier_trim_ratio: float = DEFAULT_SYNC_ALIGNMENT_OUTLIER_TRIM_RATIO,
    min_consensus_ratio: float | None = DEFAULT_SYNC_ALIGNMENT_MIN_CONSENSUS_RATIO,
    max_shift_mad: float | None = DEFAULT_SYNC_ALIGNMENT_MAX_SHIFT_MAD,
):
    resolved_device = resolve_sync_alignment_device(device)
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Sync alignment checkpoint not found: {checkpoint_path}")

    n_frames = int(len(frames))
    mel_total_steps = int(mel.shape[1])
    if n_frames < int(syncnet_T):
        raise RuntimeError(f"Not enough frames for sync alignment: n_frames={n_frames}, T={syncnet_T}")
    if mel_total_steps < int(mel_step_size):
        raise RuntimeError(
            f"Not enough mel steps for sync alignment: mel_total_steps={mel_total_steps}, "
            f"mel_step_size={mel_step_size}"
        )

    candidate_starts = []
    for start in range(0, n_frames - int(syncnet_T) + 1):
        base_start = _base_mel_start_for_frame(start, fps=fps, mel_frames_per_second=mel_frames_per_second)
        earliest = base_start - int(search_mel_ticks)
        latest = base_start + int(search_mel_ticks) + int(mel_step_size)
        if earliest < int(search_guard_mel_ticks):
            continue
        if latest > (mel_total_steps - int(search_guard_mel_ticks)):
            continue
        candidate_starts.append(start)

    if not candidate_starts:
        raise RuntimeError(
            "No valid candidate starts for sync alignment "
            f"(n_frames={n_frames}, mel_total_steps={mel_total_steps})"
        )

    min_gap_frames = 0
    if min_start_gap_ratio > 0:
        min_gap_frames = max(1, int(math.ceil(n_frames * float(min_start_gap_ratio))))

    chosen_sample_count = _resolve_sync_alignment_sample_count(
        n_frames=n_frames,
        fps=fps,
        requested_samples=int(samples),
        density_per_5s=float(sample_density_per_5s),
        candidate_count=len(candidate_starts),
        min_gap_frames=min_gap_frames,
        candidate_starts=candidate_starts,
    )

    chosen_starts = _choose_spaced_starts_with_gap_multiple(
        candidate_starts,
        max(1, int(chosen_sample_count)),
        min_gap_frames,
        int(seed),
        int(start_gap_multiple),
    )

    audio_shifts = list(range(-int(search_mel_ticks), int(search_mel_ticks) + 1))
    visual_batch = _build_visual_batch(frames, chosen_starts, int(syncnet_T), resolved_device)
    visual_batch = visual_batch.repeat_interleave(len(audio_shifts), dim=0)

    audio_chunks = []
    for start in chosen_starts:
        base_start = _base_mel_start_for_frame(start, fps=fps, mel_frames_per_second=mel_frames_per_second)
        for audio_shift in audio_shifts:
            mel_start = base_start + int(audio_shift)
            mel_end = mel_start + int(mel_step_size)
            audio_chunks.append(mel[:, mel_start:mel_end])

    audio_batch = torch.from_numpy(
        np.ascontiguousarray(np.stack(audio_chunks, axis=0)[:, np.newaxis])
    ).to(resolved_device)

    model = load_official_syncnet(checkpoint_path, resolved_device)

    losses = []
    with torch.no_grad():
        for start_idx in range(0, audio_batch.shape[0], int(batch_size)):
            end_idx = min(start_idx + int(batch_size), audio_batch.shape[0])
            batch_audio = audio_batch[start_idx:end_idx]
            batch_visual = visual_batch[start_idx:end_idx]
            audio_emb, video_emb = model(batch_audio, batch_visual)
            cos = F.cosine_similarity(audio_emb, video_emb)
            losses.append(_official_sync_loss_from_cosine(cos).detach().cpu().numpy())

    losses = np.concatenate(losses, axis=0).reshape(len(chosen_starts), len(audio_shifts))
    keep_mask, local_best_shifts, center_shift = _trim_outlier_points_by_local_shift(
        losses,
        audio_shifts,
        float(outlier_trim_ratio),
    )
    kept_losses = losses[keep_mask]
    kept_local_best_shifts = local_best_shifts[keep_mask]
    post_trim_center_shift, shift_mad, consensus_ratio = _compute_post_trim_shift_stats(
        kept_local_best_shifts
    )
    mean_losses = kept_losses.mean(axis=0)
    best_idx = int(np.argmin(mean_losses))
    best_audio_shift = int(audio_shifts[best_idx])
    zero_idx = audio_shifts.index(0)
    consensus_failed = False
    shift_mad_failed = False
    weak_sync_signal_reasons = []
    if min_consensus_ratio is not None and consensus_ratio is not None:
        consensus_failed = float(consensus_ratio) <= float(min_consensus_ratio)
        if consensus_failed:
            weak_sync_signal_reasons.append(
                f"consensus_ratio<={float(min_consensus_ratio):.4f}"
            )
    if max_shift_mad is not None and shift_mad is not None:
        shift_mad_failed = float(shift_mad) >= float(max_shift_mad)
        if shift_mad_failed:
            weak_sync_signal_reasons.append(
                f"shift_mad>={float(max_shift_mad):.4f}"
            )

    if min_consensus_ratio is not None and max_shift_mad is not None:
        weak_sync_signal = bool(consensus_failed or shift_mad_failed)
    elif min_consensus_ratio is not None:
        weak_sync_signal = bool(consensus_failed)
    elif max_shift_mad is not None:
        weak_sync_signal = bool(shift_mad_failed)
    else:
        weak_sync_signal = False

    metrics_by_shift = {
        str(int(shift)): {
            "audio_shift_mel_ticks": int(shift),
            "mean_loss": float(mean_losses[idx]),
            "std_loss": float(kept_losses[:, idx].std()),
            "num_samples": int(kept_losses.shape[0]),
        }
        for idx, shift in enumerate(audio_shifts)
    }

    return {
        "audio_shift_mel_ticks": best_audio_shift,
        "best_mean_loss": float(mean_losses[best_idx]),
        "zero_mean_loss": float(mean_losses[zero_idx]),
        "metrics_by_shift": metrics_by_shift,
        "starts": [int(v) for v in chosen_starts],
        "kept_starts": [int(chosen_starts[idx]) for idx, keep in enumerate(keep_mask.tolist()) if keep],
        "dropped_starts": [int(chosen_starts[idx]) for idx, keep in enumerate(keep_mask.tolist()) if not keep],
        "local_best_shifts": [int(v) for v in local_best_shifts.tolist()],
        "kept_local_best_shifts": [int(v) for v in kept_local_best_shifts.tolist()],
        "local_best_shift_center": float(center_shift),
        "consensus_ratio": None if consensus_ratio is None else float(consensus_ratio),
        "shift_mad": None if shift_mad is None else float(shift_mad),
        "post_trim_shift_center": None if post_trim_center_shift is None else float(post_trim_center_shift),
        "weak_sync_signal": bool(weak_sync_signal),
        "weak_sync_signal_reasons": weak_sync_signal_reasons,
        "min_consensus_ratio": None if min_consensus_ratio is None else float(min_consensus_ratio),
        "max_shift_mad": None if max_shift_mad is None else float(max_shift_mad),
        "outlier_trim_ratio": float(outlier_trim_ratio),
        "num_points_before_trim": int(losses.shape[0]),
        "num_points_after_trim": int(kept_losses.shape[0]),
        "search_mel_ticks": int(search_mel_ticks),
        "search_guard_mel_ticks": int(search_guard_mel_ticks),
        "samples": int(len(chosen_starts)),
        "sample_density_per_5s": float(sample_density_per_5s),
        "seed": int(seed),
        "min_start_gap_ratio": float(min_start_gap_ratio),
        "start_gap_multiple": int(start_gap_multiple),
        "device": resolved_device,
    }


def write_sync_alignment_to_meta_path(
    meta_path: str,
    meta: dict,
    *,
    audio_shift_mel_ticks: int,
    n_frames: int,
    mel_total_steps: int,
    fps: float,
    mel_frames_per_second: float,
    mel_step_size: int,
    search_guard_mel_ticks: int,
    source: str,
    search_range_mel_ticks: int | None = None,
    search_samples: int | None = None,
    search_seed: int | None = None,
    min_start_gap_ratio: float | None = None,
    start_gap_multiple: int | None = None,
    best_mean_loss: float | None = None,
    zero_mean_loss: float | None = None,
    extra: dict | None = None,
) -> dict:
    updated = upsert_sync_alignment(
        meta,
        audio_shift_mel_ticks=audio_shift_mel_ticks,
        n_frames=n_frames,
        mel_total_steps=mel_total_steps,
        fps=fps,
        mel_frames_per_second=mel_frames_per_second,
        mel_step_size=mel_step_size,
        search_guard_mel_ticks=search_guard_mel_ticks,
        source=source,
        search_range_mel_ticks=search_range_mel_ticks,
        search_samples=search_samples,
        search_seed=search_seed,
        min_start_gap_ratio=min_start_gap_ratio,
        start_gap_multiple=start_gap_multiple,
        best_mean_loss=best_mean_loss,
        zero_mean_loss=zero_mean_loss,
        extra=extra,
    )
    _atomic_write_json(meta_path, updated)
    return updated


def write_failed_sync_alignment_to_meta_path(
    meta_path: str,
    meta: dict,
    *,
    n_frames: int | None = None,
    mel_total_steps: int | None = None,
    fps: float | None = None,
    mel_frames_per_second: float | None = None,
    mel_step_size: int | None = None,
    search_guard_mel_ticks: int | None = None,
    source: str,
    reason: str,
    error: str | None = None,
    extra: dict | None = None,
) -> dict:
    updated = upsert_failed_sync_alignment(
        meta,
        n_frames=n_frames,
        mel_total_steps=mel_total_steps,
        fps=fps,
        mel_frames_per_second=mel_frames_per_second,
        mel_step_size=mel_step_size,
        search_guard_mel_ticks=search_guard_mel_ticks,
        source=source,
        reason=reason,
        error=error,
        extra=extra,
    )
    _atomic_write_json(meta_path, updated)
    return updated
