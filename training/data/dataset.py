"""
LipSync training dataset — loads preprocessed face crops + mel.

Expects preprocessed directory structure:
  root/
    speaker_001/
      frames.npy   # (N, H, W, 3) uint8 BGR
      mel.npy      # (80, T) float32
      bbox.json    # contains original fps
    speaker_002/
      ...

Generator samples follow the temporal LipSync training setup:
  - face_input: (6, T, H, W) = [masked_target_window | full_reference_window]
  - indiv_mels: (T, 1, 80, 16) per-frame mel windows for the generator
  - mel: (1, 80, 16) single mel chunk for SyncNet supervision
  - gt_face: (3, T, H, W) target face window

At load time, legacy frame sequences can still be normalized to the configured
target fps so that mel/frame alignment stays consistent even when the original
clip fps differs. New datasets should ideally already be normalized upstream.
"""

import os
import random
import json
from collections import OrderedDict
import numpy as np
import torch
from torch.utils.data import Dataset


class LipSyncDataset(Dataset):
    def __init__(self, roots, img_size=256, mel_step_size=16, fps=25,
                 audio_cfg=None, syncnet_T=5, mode="generator", cache_size=8,
                 skip_bad_samples=True, speaker_allowlist=None):
        self.img_size = img_size
        self.mel_step_size = mel_step_size
        self.fps = fps
        self.mode = mode
        self.syncnet_T = syncnet_T
        self.cache_size = max(int(cache_size), 0)
        self.skip_bad_samples = skip_bad_samples
        self.speaker_allowlist = (
            {name.strip() for name in speaker_allowlist if name.strip()}
            if speaker_allowlist else None
        )

        # Collect all preprocessed speakers
        self.speakers = []
        for root in roots:
            if root is None or not os.path.isdir(root):
                continue
            for name in sorted(os.listdir(root)):
                speaker_dir = os.path.join(root, name)
                frames_path = os.path.join(speaker_dir, "frames.npy")
                mel_path = os.path.join(speaker_dir, "mel.npy")
                if os.path.exists(frames_path) and os.path.exists(mel_path):
                    if self.speaker_allowlist is not None and name not in self.speaker_allowlist:
                        continue
                    meta = self._load_meta(speaker_dir)
                    if self.skip_bad_samples and meta.get("bad_sample", False):
                        print(f"  [Dataset] Skipping bad sample: {name} ({','.join(meta.get('bad_reasons', []))})")
                        continue
                    self.speakers.append(speaker_dir)

        print(f"[Dataset] Found {len(self.speakers)} speakers from {len(roots)} roots")

        # Preload metadata (frame counts) for weighting
        self._frame_counts = []
        for sp in self.speakers:
            frames = np.load(os.path.join(sp, "frames.npy"), mmap_mode="r")
            self._frame_counts.append(frames.shape[0])
            print(f"  {os.path.basename(sp)}: {frames.shape[0]} frames, "
                  f"{frames.shape[1]}x{frames.shape[2]}")

        self._total_frames = sum(self._frame_counts)
        print(f"[Dataset] Total: {self._total_frames} frames across {len(self.speakers)} speakers")

        # Cache loaded speaker tensors with an LRU cap. Unbounded caching is fine
        # for tiny smoke sets, but it will eventually load the whole HDTF corpus
        # into RAM and get the process SIGKILLed on MPS runs.
        self._cache = OrderedDict()

    @staticmethod
    def _load_meta(speaker_dir):
        meta_path = os.path.join(speaker_dir, "bbox.json")
        if not os.path.exists(meta_path):
            return {}
        try:
            with open(meta_path) as f:
                return json.load(f)
        except Exception:
            return {}

    def _normalize_frames_to_target_fps(self, frames, source_fps):
        if source_fps <= 0:
            source_fps = self.fps
        if abs(source_fps - self.fps) < 1e-3 or len(frames) <= 1:
            return frames

        duration_sec = len(frames) / float(source_fps)
        target_frames = max(1, int(round(duration_sec * self.fps)))
        indices = np.round(np.arange(target_frames) * (source_fps / self.fps)).astype(np.int64)
        indices = np.clip(indices, 0, len(frames) - 1)
        return frames[indices]

    def _build_frame_aligned_mels(self, mel, n_frames):
        mel_idx_mult = 80.0 / self.fps
        chunks = []
        for i in range(max(1, n_frames)):
            start = int(i * mel_idx_mult)
            end = start + self.mel_step_size
            if mel.shape[1] < self.mel_step_size:
                chunk = np.pad(
                    mel,
                    ((0, 0), (0, self.mel_step_size - mel.shape[1])),
                    mode="edge",
                )
            elif end > mel.shape[1]:
                chunk = mel[:, -self.mel_step_size:]
            else:
                chunk = mel[:, start:end]
            chunks.append(chunk.astype(np.float32, copy=False))
        return chunks

    def _resize_window_if_needed(self, window):
        if not window:
            return window
        if window[0].shape[0] == self.img_size and window[0].shape[1] == self.img_size:
            return window

        import cv2

        return [cv2.resize(face, (self.img_size, self.img_size)) for face in window]

    @staticmethod
    def _window_to_float(window):
        return np.stack([face.astype(np.float32) / 255.0 for face in window], axis=0)

    @staticmethod
    def _window_to_chw(window):
        return np.ascontiguousarray(window.transpose(3, 0, 1, 2))

    def _pick_reference_start(self, n_frames, target_start, T):
        max_start = n_frames - T
        if max_start <= 0:
            return 0

        candidates = [
            idx for idx in range(max_start + 1)
            if abs(idx - target_start) >= T
        ]
        if not candidates:
            return min(max_start, (target_start + T) % (max_start + 1))
        return random.choice(candidates)

    def make_generator_sample(self, speaker_idx, start, ref_start):
        frames, mel_chunks, _ = self._load_speaker(self.speakers[speaker_idx])
        return self._build_generator_sample(frames, mel_chunks, start, ref_start)

    def __len__(self):
        return max(self._total_frames, len(self.speakers) * 100)

    def _load_speaker(self, speaker_dir):
        if speaker_dir in self._cache:
            data = self._cache.pop(speaker_dir)
            self._cache[speaker_dir] = data
            return data

        # Keep frames as memmaps whenever possible instead of loading full clips
        # into RAM. This matters a lot once we scale from ~65 to ~350 speakers.
        frames = np.load(os.path.join(speaker_dir, "frames.npy"), mmap_mode="r")  # (N, H, W, 3)
        mel = np.load(os.path.join(speaker_dir, "mel.npy"))  # (80, T)
        meta = self._load_meta(speaker_dir)

        # Keep a legacy fallback here for older processed sets that were built
        # directly from raw videos without an explicit 25fps normalization stage.
        source_fps = float(meta.get("fps", self.fps) or self.fps)
        frames = self._normalize_frames_to_target_fps(frames, source_fps)
        chunks = self._build_frame_aligned_mels(mel, len(frames))

        data = (frames, chunks, meta)
        self._cache[speaker_dir] = data
        while self.cache_size and len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return data

    def __getitem__(self, idx):
        # Pick a random speaker (weighted by frame count)
        sp_idx = random.choices(range(len(self.speakers)),
                                weights=self._frame_counts, k=1)[0]
        speaker_dir = self.speakers[sp_idx]

        try:
            frames, mel_chunks, _ = self._load_speaker(speaker_dir)
        except Exception as e:
            print(f"[Dataset] ERROR loading {speaker_dir}: {e}")
            S = self.img_size
            T = self.syncnet_T
            if self.mode == "syncnet":
                return (
                    torch.zeros(T * 3, S // 2, S),
                    torch.zeros(1, 80, T * self.mel_step_size),
                    torch.tensor(1.0),
                )
            return (
                torch.zeros(6, T, S, S),
                torch.zeros(T, 1, 80, self.mel_step_size),
                torch.zeros(1, 80, self.mel_step_size),
                torch.zeros(3, T, S, S),
            )

        if self.mode == "syncnet":
            return self._get_syncnet_sample(frames, mel_chunks)
        else:
            return self._get_generator_sample(frames, mel_chunks)

    def _build_generator_sample(self, frames, mel_chunks, start, ref_start):
        S = self.img_size
        T = self.syncnet_T
        target_window = [frames[start + t] for t in range(T)]
        ref_window = [frames[ref_start + t] for t in range(T)]
        target_window = self._resize_window_if_needed(target_window)
        ref_window = self._resize_window_if_needed(ref_window)

        target_f = self._window_to_float(target_window)   # (T, H, W, 3)
        ref_f = self._window_to_float(ref_window)         # (T, H, W, 3)

        gt = self._window_to_chw(target_f)                # (3, T, H, W)

        masked_target = target_f.copy()
        masked_target[:, S // 2:, :, :] = 0

        face_input = np.concatenate([
            self._window_to_chw(masked_target),           # (3, T, H, W)
            self._window_to_chw(ref_f),                   # (3, T, H, W)
        ], axis=0)                                        # (6, T, H, W)

        # The generator gets per-frame mel chunks. For SyncNet supervision we use
        # the center-of-window chunk, which is the most stable alignment once the
        # clip has been normalized to the target fps.
        mel_indices = [min(max(start - 1 + t, 0), len(mel_chunks) - 1) for t in range(T)]
        indiv_mels = np.stack([mel_chunks[idx] for idx in mel_indices], axis=0)[:, np.newaxis]
        mel_input = mel_chunks[min(start + T // 2, len(mel_chunks) - 1)][np.newaxis]

        return (
            torch.from_numpy(np.ascontiguousarray(face_input)),
            torch.from_numpy(np.ascontiguousarray(indiv_mels)),
            torch.from_numpy(np.ascontiguousarray(mel_input)),
            torch.from_numpy(np.ascontiguousarray(gt)),
        )

    def _get_generator_sample(self, frames, mel_chunks):
        S = self.img_size
        T = self.syncnet_T
        n_frames = min(len(frames), len(mel_chunks))
        if n_frames < 3 * T:
            return (
                torch.zeros(6, T, S, S),
                torch.zeros(T, 1, 80, self.mel_step_size),
                torch.zeros(1, 80, self.mel_step_size),
                torch.zeros(3, T, S, S),
            )

        start = random.randint(0, n_frames - T)
        ref_start = self._pick_reference_start(n_frames, start, T)
        return self._build_generator_sample(frames, mel_chunks, start, ref_start)

    def _get_syncnet_sample(self, frames, mel_chunks):
        S = self.img_size
        T = self.syncnet_T
        n_frames = min(len(frames), len(mel_chunks))

        if n_frames < T + 5:
            return (torch.zeros(T * 3, S // 2, S), torch.zeros(1, 80, T * 16),
                    torch.tensor(1.0))

        start = random.randint(0, n_frames - T - 1)

        # Visual: T consecutive lower-face crops
        visual_frames = []
        for t in range(T):
            face = frames[start + t]
            if face.shape[0] != S or face.shape[1] != S:
                import cv2
                face = cv2.resize(face, (S, S))
            lower = face[S // 2:, :, :]
            visual_frames.append(lower.astype(np.float32) / 255.0)

        visual = np.concatenate([f.transpose(2, 0, 1) for f in visual_frames], axis=0)

        # Audio
        is_sync = random.random() > 0.5
        if is_sync:
            audio_start = start
        else:
            offset = random.choice([-1, 1]) * random.randint(5, max(5, n_frames // 2))
            audio_start = (start + offset) % max(1, n_frames - T)

        mel_concat = np.concatenate(
            [mel_chunks[min(audio_start + t, len(mel_chunks) - 1)] for t in range(T)],
            axis=1,
        )
        audio = mel_concat[np.newaxis]

        return (
            torch.from_numpy(visual),
            torch.from_numpy(audio),
            torch.tensor(1.0 if is_sync else 0.0),
        )
