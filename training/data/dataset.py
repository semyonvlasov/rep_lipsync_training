"""
LipSync training dataset.

Supports two on-disk layouts:

1. Fully materialized processed samples:
   root/
     sample_name/
       frames.npy
       mel.npy
       bbox.json

2. Canonical faceclip samples for lazy materialization:
   root/
     confident/sample_name.mp4
     confident/sample_name.json
     medium/sample_name.mp4
     medium/sample_name.json
     ...

For canonical faceclips, frames are decoded on first use into a cache dir as
`frames*.npy`, while audio features are generated lazily into a parameterized
`mel_*.npy`. Frame caches may be native, square, or rectangular. The canonical
`mp4 + json` pair remains the source of truth.
"""

import hashlib
import json
import os
import random
import subprocess
import tempfile
import time
from collections import OrderedDict

import numpy as np
import torch
from torch.utils.data import Dataset

from .audio import AudioProcessor


def _format_cache_value(value):
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return str(value).replace(".", "p")
    return str(value).replace(".", "p")


def _load_json(path):
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


def _atomic_save_npy(path, array):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        prefix=os.path.basename(path) + ".",
        suffix=".tmp",
        dir=os.path.dirname(path),
    )
    os.close(fd)
    try:
        with open(tmp_path, "wb") as f:
            np.save(f, array)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


class LipSyncDataset(Dataset):
    def __init__(
        self,
        roots,
        img_size=256,
        mel_step_size=16,
        fps=25,
        audio_cfg=None,
        syncnet_T=5,
        mode="generator",
        cache_size=8,
        skip_bad_samples=True,
        speaker_allowlist=None,
        lazy_cache_root=None,
        ffmpeg_bin="ffmpeg",
        materialize_timeout=600,
        materialize_frames_size=None,
        min_samples_per_speaker=100,
    ):
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
        self.lazy_cache_root = lazy_cache_root
        self.ffmpeg_bin = ffmpeg_bin
        self.materialize_timeout = max(int(materialize_timeout), 30)
        self.materialize_frames_size = self._normalize_materialize_frames_size(
            materialize_frames_size
        )
        self.min_samples_per_speaker = max(int(min_samples_per_speaker), 0)
        self.audio_cfg = dict(audio_cfg or {})
        self._audio_proc = AudioProcessor(self.audio_cfg) if self.audio_cfg else None

        if self.audio_cfg:
            self.mel_frames_per_second = (
                float(self.audio_cfg["sample_rate"]) / float(self.audio_cfg["hop_size"])
            )
            self._mel_cache_name = (
                "mel"
                f"_sr{_format_cache_value(self.audio_cfg['sample_rate'])}"
                f"_hop{_format_cache_value(self.audio_cfg['hop_size'])}"
                f"_n{_format_cache_value(self.audio_cfg['n_mels'])}"
                f"_fft{_format_cache_value(self.audio_cfg['n_fft'])}"
                f"_win{_format_cache_value(self.audio_cfg['win_size'])}"
                f"_fmin{_format_cache_value(self.audio_cfg['fmin'])}"
                f"_fmax{_format_cache_value(self.audio_cfg['fmax'])}"
                f"_pre{_format_cache_value(self.audio_cfg['preemphasis'])}.npy"
            )
        else:
            self.mel_frames_per_second = 80.0
            self._mel_cache_name = "mel.npy"

        self.speakers = []
        self._entries = {}
        self._root_stats = []
        self._entry_name_to_key = {}

        for root in roots:
            if root is None or not os.path.isdir(root):
                continue
            entries, stats = self._collect_entries_from_root(root)
            for entry in entries:
                existing_key = self._entry_name_to_key.get(entry["name"])
                if existing_key is None:
                    self.speakers.append(entry["key"])
                    self._entries[entry["key"]] = entry
                    self._entry_name_to_key[entry["name"]] = entry["key"]
                    continue

                existing_entry = self._entries[existing_key]
                if self._entry_priority(entry) > self._entry_priority(existing_entry):
                    self._entries.pop(existing_key, None)
                    self._entries[entry["key"]] = entry
                    self._entry_name_to_key[entry["name"]] = entry["key"]
                    self.speakers = [
                        entry["key"] if key == existing_key else key
                        for key in self.speakers
                    ]
                    stats["duplicates_replaced"] += 1
                else:
                    stats["duplicates_skipped"] += 1
            self._root_stats.append(stats)

        print(f"[Dataset] Found {len(self.speakers)} samples from {len(roots)} roots")
        for stats in self._root_stats:
            print(
                "  "
                f"{stats['root']}: processed={stats['processed']} lazy={stats['lazy']} "
                f"skipped_bad={stats['skipped_bad']} skipped_allowlist={stats['skipped_allowlist']} "
                f"duplicates_skipped={stats['duplicates_skipped']} duplicates_replaced={stats['duplicates_replaced']}"
            )

        self._frame_counts = []
        for key in self.speakers:
            entry = self._entries[key]
            frame_count = self._entry_frame_count(entry)
            self._frame_counts.append(frame_count)
            print(
                f"  {entry['name']}: {frame_count} frames "
                f"[{entry['type']}]"
            )

        self._total_frames = sum(self._frame_counts)
        print(f"[Dataset] Total: {self._total_frames} frames across {len(self.speakers)} samples")

        self._cache = OrderedDict()

    @staticmethod
    def _normalize_materialize_frames_size(value):
        if value in (None, "", "native", "source", "original"):
            return None
        if isinstance(value, (list, tuple)):
            if len(value) != 2:
                raise ValueError("materialize_frames_size list/tuple must have exactly 2 items")
            width, height = value
        elif isinstance(value, str):
            text = value.strip().lower()
            if "x" in text:
                width_str, height_str = text.split("x", 1)
                width, height = width_str, height_str
            else:
                size = max(1, int(text))
                return (size, size)
        else:
            size = max(1, int(value))
            return (size, size)

        width = max(1, int(width))
        height = max(1, int(height))
        return (width, height)

    def _frames_cache_name(self):
        if self.materialize_frames_size is None:
            return "frames.npy"
        width, height = self.materialize_frames_size
        if width == height:
            return f"frames_s{width}.npy"
        return f"frames_{width}x{height}.npy"

    @staticmethod
    def _resize_and_center_crop(frame, target_width, target_height, cv2_mod):
        src_height, src_width = frame.shape[:2]
        if src_width == target_width and src_height == target_height:
            return frame

        scale = max(target_width / float(src_width), target_height / float(src_height))
        resized_width = max(target_width, int(round(src_width * scale)))
        resized_height = max(target_height, int(round(src_height * scale)))

        interpolation = cv2_mod.INTER_AREA if scale <= 1.0 else cv2_mod.INTER_LINEAR
        resized = cv2_mod.resize(frame, (resized_width, resized_height), interpolation=interpolation)

        x0 = max(0, (resized_width - target_width) // 2)
        y0 = max(0, (resized_height - target_height) // 2)
        x1 = x0 + target_width
        y1 = y0 + target_height
        return resized[y0:y1, x0:x1]

    def _collect_entries_from_root(self, root):
        entries = []
        stats = {
            "root": root,
            "processed": 0,
            "lazy": 0,
            "skipped_bad": 0,
            "skipped_allowlist": 0,
            "duplicates_skipped": 0,
            "duplicates_replaced": 0,
        }

        for name in sorted(os.listdir(root)):
            speaker_dir = os.path.join(root, name)
            if not os.path.isdir(speaker_dir):
                continue
            frames_path = os.path.join(speaker_dir, "frames.npy")
            mel_path = os.path.join(speaker_dir, "mel.npy")
            if not (os.path.exists(frames_path) and os.path.exists(mel_path)):
                continue

            if self.speaker_allowlist is not None and name not in self.speaker_allowlist:
                stats["skipped_allowlist"] += 1
                continue

            meta_path = os.path.join(speaker_dir, "bbox.json")
            meta = _load_json(meta_path)
            if self.skip_bad_samples and meta.get("bad_sample", False):
                stats["skipped_bad"] += 1
                continue

            entry = {
                "key": speaker_dir,
                "type": "processed",
                "name": name,
                "root": root,
                "frames_path": frames_path,
                "mel_path": mel_path,
                "meta_path": meta_path,
                "meta": meta,
            }
            entries.append(entry)
            stats["processed"] += 1

        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [
                d for d in dirnames
                if d not in {"__pycache__", "_lazy_cache"}
            ]
            for filename in sorted(filenames):
                if not filename.endswith(".mp4"):
                    continue
                mp4_path = os.path.join(dirpath, filename)
                json_path = os.path.splitext(mp4_path)[0] + ".json"
                if not os.path.exists(json_path):
                    continue

                name = os.path.splitext(filename)[0]
                if self.speaker_allowlist is not None and name not in self.speaker_allowlist:
                    stats["skipped_allowlist"] += 1
                    continue

                meta = _load_json(json_path)
                if self.skip_bad_samples and meta.get("bad_sample", False):
                    stats["skipped_bad"] += 1
                    continue

                cache_dir = self._lazy_cache_dir(root, mp4_path, name)
                entry = {
                    "key": mp4_path,
                    "type": "lazy",
                    "name": name,
                    "root": root,
                    "video_path": mp4_path,
                    "meta_path": json_path,
                    "meta": meta,
                    "cache_dir": cache_dir,
                    "frames_path": os.path.join(cache_dir, self._frames_cache_name()),
                    "mel_path": os.path.join(cache_dir, self._mel_cache_name),
                }
                entries.append(entry)
                stats["lazy"] += 1

        return entries, stats

    @staticmethod
    def _entry_priority(entry):
        return 1 if entry["type"] == "processed" else 0

    def _lazy_cache_dir(self, root, video_path, name):
        if self.lazy_cache_root:
            base_root = self.lazy_cache_root
        else:
            base_root = os.path.join(root, "_lazy_cache")
        digest = hashlib.sha1(os.path.abspath(video_path).encode("utf-8")).hexdigest()[:12]
        return os.path.join(base_root, f"{name}--{digest}")

    def _entry_frame_count(self, entry):
        meta = entry.get("meta") or {}
        count = int(meta.get("n_frames") or meta.get("frames") or 0)
        if count > 0:
            return count
        frames = np.load(entry["frames_path"], mmap_mode="r")
        return int(frames.shape[0])

    @staticmethod
    def _load_meta(entry):
        meta = entry.get("meta")
        if meta is not None:
            return meta
        meta = _load_json(entry["meta_path"])
        entry["meta"] = meta
        return meta

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
        mel_idx_mult = float(self.mel_frames_per_second) / float(self.fps)
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
        return max(self._total_frames, len(self.speakers) * self.min_samples_per_speaker)

    def _lock_path(self, target_path):
        return target_path + ".lock"

    def _acquire_lock(self, lock_path):
        started = time.time()
        while True:
            try:
                fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.close(fd)
                return
            except FileExistsError:
                if time.time() - started > self.materialize_timeout:
                    raise TimeoutError(f"Timeout acquiring lock for {lock_path}")
                time.sleep(0.1)

    @staticmethod
    def _release_lock(lock_path):
        if os.path.exists(lock_path):
            os.remove(lock_path)

    def _materialize_frames(self, entry):
        frames_path = entry["frames_path"]
        if os.path.exists(frames_path):
            return frames_path

        os.makedirs(entry["cache_dir"], exist_ok=True)
        lock_path = self._lock_path(frames_path)
        self._acquire_lock(lock_path)
        try:
            if os.path.exists(frames_path):
                return frames_path

            import cv2

            cap = cv2.VideoCapture(entry["video_path"])
            if not cap.isOpened():
                raise RuntimeError(f"Could not open video: {entry['video_path']}")

            target_shape = self.materialize_frames_size
            frames = []
            try:
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    if target_shape is not None:
                        target_width, target_height = target_shape
                        if frame.shape[1] != target_width or frame.shape[0] != target_height:
                            frame = self._resize_and_center_crop(
                                frame,
                                target_width=target_width,
                                target_height=target_height,
                                cv2_mod=cv2,
                            )
                    frames.append(frame)
            finally:
                cap.release()

            if not frames:
                raise RuntimeError(f"No frames decoded from {entry['video_path']}")

            frames_arr = np.stack(frames, axis=0)
            _atomic_save_npy(frames_path, frames_arr)
            return frames_path
        finally:
            self._release_lock(lock_path)

    def _materialize_mel(self, entry):
        mel_path = entry["mel_path"]
        if os.path.exists(mel_path):
            return mel_path
        if self._audio_proc is None:
            raise RuntimeError("audio_cfg is required for lazy mel generation")

        os.makedirs(entry["cache_dir"], exist_ok=True)
        lock_path = self._lock_path(mel_path)
        self._acquire_lock(lock_path)
        try:
            if os.path.exists(mel_path):
                return mel_path

            fd, wav_path = tempfile.mkstemp(
                prefix=f"{entry['name']}.",
                suffix=".wav",
                dir=entry["cache_dir"],
            )
            os.close(fd)
            try:
                cmd = [
                    self.ffmpeg_bin,
                    "-y",
                    "-i",
                    entry["video_path"],
                    "-ar",
                    str(int(self.audio_cfg["sample_rate"])),
                    "-ac",
                    "1",
                    "-f",
                    "wav",
                    wav_path,
                ]
                subprocess.run(cmd, check=True, capture_output=True, text=True)
                wav = self._audio_proc.load_wav(wav_path)
                mel = self._audio_proc.melspectrogram(wav)
                _atomic_save_npy(mel_path, mel)
            finally:
                if os.path.exists(wav_path):
                    os.remove(wav_path)
            return mel_path
        finally:
            self._release_lock(lock_path)

    def _materialize_lazy_entry(self, entry):
        os.makedirs(entry["cache_dir"], exist_ok=True)
        self._materialize_frames(entry)
        self._materialize_mel(entry)

    def _load_speaker(self, speaker_ref):
        entry = self._entries.get(speaker_ref)
        if entry is None:
            entry = {
                "key": speaker_ref,
                "type": "processed",
                "name": os.path.basename(speaker_ref),
                "frames_path": os.path.join(speaker_ref, "frames.npy"),
                "mel_path": os.path.join(speaker_ref, "mel.npy"),
                "meta_path": os.path.join(speaker_ref, "bbox.json"),
                "meta": None,
            }

        cache_key = entry["key"]
        if cache_key in self._cache:
            data = self._cache.pop(cache_key)
            self._cache[cache_key] = data
            return data

        if entry["type"] == "lazy":
            self._materialize_lazy_entry(entry)

        frames = np.load(entry["frames_path"], mmap_mode="r")
        mel = np.load(entry["mel_path"])
        meta = self._load_meta(entry)

        source_fps = float(meta.get("fps", self.fps) or self.fps)
        frames = self._normalize_frames_to_target_fps(frames, source_fps)
        chunks = self._build_frame_aligned_mels(mel, len(frames))

        data = (frames, chunks, meta)
        self._cache[cache_key] = data
        while self.cache_size and len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return data

    def __getitem__(self, idx):
        sp_idx = random.choices(range(len(self.speakers)), weights=self._frame_counts, k=1)[0]
        speaker_key = self.speakers[sp_idx]

        try:
            frames, mel_chunks, _ = self._load_speaker(speaker_key)
        except Exception as e:
            name = self._entries.get(speaker_key, {}).get("name", str(speaker_key))
            print(f"[Dataset] ERROR loading {name}: {e}")
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
        return self._get_generator_sample(frames, mel_chunks)

    def _build_generator_sample(self, frames, mel_chunks, start, ref_start):
        S = self.img_size
        T = self.syncnet_T
        target_window = [frames[start + t] for t in range(T)]
        ref_window = [frames[ref_start + t] for t in range(T)]
        target_window = self._resize_window_if_needed(target_window)
        ref_window = self._resize_window_if_needed(ref_window)

        target_f = self._window_to_float(target_window)
        ref_f = self._window_to_float(ref_window)

        gt = self._window_to_chw(target_f)

        masked_target = target_f.copy()
        masked_target[:, S // 2:, :, :] = 0

        face_input = np.concatenate(
            [
                self._window_to_chw(masked_target),
                self._window_to_chw(ref_f),
            ],
            axis=0,
        )

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
            return (
                torch.zeros(T * 3, S // 2, S),
                torch.zeros(1, 80, T * self.mel_step_size),
                torch.tensor(1.0),
            )

        start = random.randint(0, n_frames - T - 1)

        visual_frames = []
        for t in range(T):
            face = frames[start + t]
            if face.shape[0] != S or face.shape[1] != S:
                import cv2
                face = cv2.resize(face, (S, S))
            lower = face[S // 2:, :, :]
            visual_frames.append(lower.astype(np.float32) / 255.0)

        visual = np.concatenate([f.transpose(2, 0, 1) for f in visual_frames], axis=0)

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
