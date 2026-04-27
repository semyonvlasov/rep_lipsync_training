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
import shutil
import subprocess
import tempfile
import time
from collections import OrderedDict

import numpy as np
import torch
from torch.utils.data import Dataset

from .audio import AudioProcessor
from .sync_alignment import (
    DEFAULT_SYNCNET_CHECKPOINT,
    DEFAULT_SYNC_ALIGNMENT_BATCH_SIZE,
    DEFAULT_SYNC_ALIGNMENT_GUARD_MEL_TICKS,
    DEFAULT_SYNC_ALIGNMENT_MAX_SHIFT_MAD,
    DEFAULT_SYNC_ALIGNMENT_MIN_START_GAP_RATIO,
    DEFAULT_SYNC_ALIGNMENT_MIN_CONSENSUS_RATIO,
    DEFAULT_SYNC_ALIGNMENT_OUTLIER_TRIM_RATIO,
    DEFAULT_SYNC_ALIGNMENT_SAMPLE_DENSITY_PER_5S,
    DEFAULT_SYNC_ALIGNMENT_SAMPLES,
    DEFAULT_SYNC_ALIGNMENT_SEARCH_MEL_TICKS,
    DEFAULT_SYNC_ALIGNMENT_SEED,
    DEFAULT_SYNC_ALIGNMENT_START_GAP_MULTIPLE,
    build_shifted_frame_aligned_mels,
    compute_sync_alignment_from_faceclip,
    append_sync_alignment_registry_record,
    build_sync_alignment_registry_record,
    find_sync_alignment_registry_record,
    is_failed_sync_alignment,
    load_sync_alignment_registry,
    load_sync_alignment,
    upsert_sync_alignment,
    write_raw_sync_alignment_to_meta_path,
    write_sync_alignment_to_meta_path,
    write_failed_sync_alignment_to_meta_path,
)


TRAINING_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(TRAINING_ROOT)


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


def _resolve_repo_relative_path(path):
    if not path:
        return path
    path = os.fspath(path)
    if os.path.isabs(path):
        return path

    project_relative_prefixes = (
        "assets" + os.sep,
        "models" + os.sep,
        "training" + os.sep,
    )
    if path.startswith(project_relative_prefixes):
        candidates = [
            os.path.join(PROJECT_ROOT, path),
            os.path.join(TRAINING_ROOT, path),
            path,
        ]
    else:
        candidates = [
            os.path.join(TRAINING_ROOT, path),
            os.path.join(PROJECT_ROOT, path),
            path,
        ]

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return candidates[0]


def _format_sync_alignment_progress(meta):
    sync_alignment = meta.get("sync_alignment") if isinstance(meta, dict) else None
    if not isinstance(sync_alignment, dict):
        return ""
    status = sync_alignment.get("status")
    if status == "aligned":
        parts = []
        shift = sync_alignment.get("audio_shift_mel_ticks")
        if shift is not None:
            parts.append(f"shift={int(shift):+d}")
        consensus_ratio = sync_alignment.get("consensus_ratio")
        if consensus_ratio is not None:
            parts.append(f"consensus={float(consensus_ratio):.3f}")
        shift_mad = sync_alignment.get("shift_mad")
        if shift_mad is not None:
            parts.append(f"mad={float(shift_mad):.3f}")
        valid_frame_count = sync_alignment.get("valid_frame_count")
        if valid_frame_count is not None:
            parts.append(f"valid_frames={int(valid_frame_count)}")
        return f" {' '.join(parts)}" if parts else ""
    if status == "failed":
        parts = ["drop"]
        reason = sync_alignment.get("reason")
        if reason:
            parts.append(f"reason={reason}")
        error = sync_alignment.get("error")
        if error:
            parts.append(f"detail={error}")
        candidate_shift = sync_alignment.get("candidate_audio_shift_mel_ticks")
        if candidate_shift is not None:
            parts.append(f"candidate_shift={int(candidate_shift):+d}")
        consensus_ratio = sync_alignment.get("consensus_ratio")
        if consensus_ratio is not None:
            parts.append(f"consensus={float(consensus_ratio):.3f}")
        shift_mad = sync_alignment.get("shift_mad")
        if shift_mad is not None:
            parts.append(f"mad={float(shift_mad):.3f}")
        return f" {' '.join(parts)}"
    return ""


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
        syncnet_style="local",
        cache_size=8,
        skip_bad_samples=True,
        speaker_allowlist=None,
        lazy_cache_root=None,
        ffmpeg_bin="ffmpeg",
        materialize_timeout=600,
        materialize_frames_size=None,
        materialize_min_face_height_ratio=0.95,
        allow_empty=False,
        sync_alignment_enabled=True,
        sync_alignment_compute_if_missing=True,
        sync_alignment_guard_mel_ticks=DEFAULT_SYNC_ALIGNMENT_GUARD_MEL_TICKS,
        sync_alignment_search_mel_ticks=DEFAULT_SYNC_ALIGNMENT_SEARCH_MEL_TICKS,
        sync_alignment_samples=DEFAULT_SYNC_ALIGNMENT_SAMPLES,
        sync_alignment_sample_density_per_5s=DEFAULT_SYNC_ALIGNMENT_SAMPLE_DENSITY_PER_5S,
        sync_alignment_seed=DEFAULT_SYNC_ALIGNMENT_SEED,
        sync_alignment_min_start_gap_ratio=DEFAULT_SYNC_ALIGNMENT_MIN_START_GAP_RATIO,
        sync_alignment_start_gap_multiple=DEFAULT_SYNC_ALIGNMENT_START_GAP_MULTIPLE,
        sync_alignment_device="auto",
        sync_alignment_batch_size=DEFAULT_SYNC_ALIGNMENT_BATCH_SIZE,
        sync_alignment_outlier_trim_ratio=DEFAULT_SYNC_ALIGNMENT_OUTLIER_TRIM_RATIO,
        sync_alignment_min_consensus_ratio=DEFAULT_SYNC_ALIGNMENT_MIN_CONSENSUS_RATIO,
        sync_alignment_max_shift_mad=DEFAULT_SYNC_ALIGNMENT_MAX_SHIFT_MAD,
        sync_alignment_syncnet_checkpoint=None,
        sync_alignment_write_manifest=True,
        sync_alignment_registry_path=None,
    ):
        self.img_size = img_size
        self.mel_step_size = mel_step_size
        self.fps = fps
        self.mode = mode
        self.syncnet_T = syncnet_T
        self.syncnet_style = syncnet_style
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
        self.materialize_min_face_height_ratio = min(
            1.0,
            max(0.0, float(materialize_min_face_height_ratio)),
        )
        self.allow_empty = bool(allow_empty)
        self.audio_cfg = dict(audio_cfg or {})
        self._audio_proc = AudioProcessor(self.audio_cfg) if self.audio_cfg else None
        self.sync_alignment_enabled = bool(sync_alignment_enabled and self.audio_cfg)
        self.sync_alignment_compute_if_missing = bool(sync_alignment_compute_if_missing)
        self.sync_alignment_guard_mel_ticks = max(0, int(sync_alignment_guard_mel_ticks))
        self.sync_alignment_search_mel_ticks = max(0, int(sync_alignment_search_mel_ticks))
        self.sync_alignment_samples = int(sync_alignment_samples or 0)
        self.sync_alignment_sample_density_per_5s = max(0.1, float(sync_alignment_sample_density_per_5s))
        self.sync_alignment_seed = int(sync_alignment_seed)
        self.sync_alignment_min_start_gap_ratio = max(0.0, float(sync_alignment_min_start_gap_ratio))
        self.sync_alignment_start_gap_multiple = max(0, int(sync_alignment_start_gap_multiple))
        self.sync_alignment_device = sync_alignment_device
        self.sync_alignment_batch_size = max(1, int(sync_alignment_batch_size))
        self.sync_alignment_outlier_trim_ratio = min(
            0.49,
            max(0.0, float(sync_alignment_outlier_trim_ratio)),
        )
        self.sync_alignment_min_consensus_ratio = (
            None
            if sync_alignment_min_consensus_ratio is None
            else float(sync_alignment_min_consensus_ratio)
        )
        self.sync_alignment_max_shift_mad = (
            None
            if sync_alignment_max_shift_mad is None
            else float(sync_alignment_max_shift_mad)
        )
        self.sync_alignment_syncnet_checkpoint = _resolve_repo_relative_path(
            sync_alignment_syncnet_checkpoint or DEFAULT_SYNCNET_CHECKPOINT
        )
        self.sync_alignment_write_manifest = bool(sync_alignment_write_manifest)
        self.sync_alignment_registry_path = sync_alignment_registry_path
        self.sync_alignment_registry = load_sync_alignment_registry(sync_alignment_registry_path)
        if self.sync_alignment_registry:
            print(
                f"[Dataset] Sync alignment registry: loaded "
                f"{len(self.sync_alignment_registry)} records from {sync_alignment_registry_path}",
                flush=True,
            )

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

        if self.sync_alignment_enabled and self.sync_alignment_compute_if_missing:
            self._ensure_sync_alignment_for_lazy_entries()

        if self.mode == "syncnet" and self.syncnet_style == "mirror":
            min_frames = (3 * self.syncnet_T) + 1
            filtered_speakers = []
            skipped_short = 0
            for key in self.speakers:
                entry = self._entries[key]
                try:
                    frame_count = self._entry_frame_count(entry)
                except Exception:
                    frame_count = 0
                if frame_count < min_frames:
                    skipped_short += 1
                    continue
                filtered_speakers.append(key)
            self.speakers = filtered_speakers
            print(
                f"[Dataset] SyncNetMirror prefilter: skipped {skipped_short} "
                f"samples with < {min_frames} frames"
            )

        print(f"[Dataset] Found {len(self.speakers)} samples from {len(roots)} roots")
        for stats in self._root_stats:
            print(
                "  "
                f"{stats['root']}: processed={stats['processed']} lazy={stats['lazy']} "
                f"skipped_bad={stats['skipped_bad']} skipped_allowlist={stats['skipped_allowlist']} "
                f"skipped_materialize_size={stats['skipped_materialize_size']} "
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
        if not self.speakers and not self.allow_empty:
            raise RuntimeError("Dataset is empty after filtering")

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

    def _required_materialize_face_height(self):
        if self.materialize_frames_size is not None:
            return int(self.materialize_frames_size[1])
        return int(self.img_size)

    @staticmethod
    def _detection_face_height(record):
        raw_bbox = record.get("raw_bbox")
        if isinstance(raw_bbox, (list, tuple)) and len(raw_bbox) >= 4:
            return max(int(raw_bbox[3]) - int(raw_bbox[1]), 0)
        bbox = record.get("bbox")
        if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
            return max(int(bbox[1]) - int(bbox[0]), 0)
        return 0

    def _lazy_entry_materialize_height_stats(self, entry, meta):
        cached = entry.get("_materialize_height_stats")
        if cached is not None:
            return cached

        detections_path = entry.get("detections_path")
        if not detections_path or not os.path.exists(detections_path):
            return None

        trimmed_indices = meta.get("trimmed_frame_indices")
        if not isinstance(trimmed_indices, list) or not trimmed_indices:
            return None

        detections_payload = _load_json(detections_path)
        detections = detections_payload.get("detections")
        if not isinstance(detections, list) or not detections:
            return None

        height_by_frame = {}
        for record in detections:
            try:
                frame_idx = int(record.get("frame_idx", -1))
            except Exception:
                continue
            if frame_idx < 0:
                continue
            height_by_frame[frame_idx] = self._detection_face_height(record)

        total = len(trimmed_indices)
        if total <= 0:
            return None

        required_height = self._required_materialize_face_height()
        valid = 0
        for frame_idx in trimmed_indices:
            try:
                lookup_idx = int(frame_idx)
            except Exception:
                continue
            if int(height_by_frame.get(lookup_idx, 0)) >= required_height:
                valid += 1
        ratio = float(valid) / float(total)
        stats = {
            "required_height": int(required_height),
            "valid_frames": int(valid),
            "total_frames": int(total),
            "valid_ratio": float(ratio),
            "passes": bool(ratio >= self.materialize_min_face_height_ratio),
        }
        entry["_materialize_height_stats"] = stats
        return stats

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
            "skipped_materialize_size": 0,
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
            meta = self._apply_sync_alignment_registry_to_meta(
                meta,
                meta_path=meta_path,
                name=name,
                root=root,
            )
            if self.skip_bad_samples and meta.get("bad_sample", False):
                stats["skipped_bad"] += 1
                continue
            if self.skip_bad_samples and is_failed_sync_alignment(meta):
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
                meta = self._apply_sync_alignment_registry_to_meta(
                    meta,
                    meta_path=json_path,
                    name=name,
                    root=root,
                )
                if self.skip_bad_samples and meta.get("bad_sample", False):
                    stats["skipped_bad"] += 1
                    continue
                if self.skip_bad_samples and is_failed_sync_alignment(meta):
                    stats["skipped_bad"] += 1
                    cache_dir = self._lazy_cache_dir(root, mp4_path, name)
                    self._cleanup_lazy_materialization({"cache_dir": cache_dir})
                    continue

                detections_path = os.path.splitext(mp4_path)[0] + ".detections.json"
                entry = {
                    "key": mp4_path,
                    "type": "lazy",
                    "name": name,
                    "root": root,
                    "video_path": mp4_path,
                    "meta_path": json_path,
                    "detections_path": detections_path if os.path.exists(detections_path) else None,
                    "meta": meta,
                }
                height_stats = self._lazy_entry_materialize_height_stats(entry, meta)
                if height_stats is not None and not height_stats["passes"]:
                    stats["skipped_materialize_size"] += 1
                    continue

                cache_dir = self._lazy_cache_dir(root, mp4_path, name)
                entry["cache_dir"] = cache_dir
                entry["frames_path"] = os.path.join(cache_dir, self._frames_cache_name())
                entry["mel_path"] = os.path.join(cache_dir, self._mel_cache_name)
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

    @staticmethod
    def _infer_dataset_kind(root, meta):
        if isinstance(meta, dict):
            dataset_kind = str(meta.get("source_dataset") or "").strip().lower()
            if dataset_kind:
                return dataset_kind
        root_text = str(root).lower()
        if "hdtf" in root_text:
            return "hdtf"
        if "talkvid" in root_text:
            return "talkvid"
        return ""

    def _apply_sync_alignment_registry_to_meta(self, meta, *, meta_path, name, root):
        if not self.sync_alignment_registry:
            return meta
        dataset_kind = self._infer_dataset_kind(root, meta)
        record = find_sync_alignment_registry_record(
            self.sync_alignment_registry,
            name=name,
            dataset_kind=dataset_kind,
        )
        if not record:
            return meta

        record_alignment = record.get("sync_alignment")
        if not isinstance(record_alignment, dict):
            return meta
        record_status = record_alignment.get("status")
        if meta.get("sync_alignment") == record_alignment:
            return meta

        updated_meta = write_raw_sync_alignment_to_meta_path(
            meta_path,
            meta,
            record_alignment,
        )
        print(
            f"[Dataset] Sync alignment registry: applied {record_status} to {name}",
            flush=True,
        )
        return updated_meta

    def _sync_alignment_registry_record_for_entry(self, entry, meta):
        if not self.sync_alignment_registry_path:
            return None
        if not isinstance(meta, dict):
            return None
        sync_alignment = meta.get("sync_alignment")
        if not isinstance(sync_alignment, dict):
            return None
        if sync_alignment.get("status") not in {"aligned", "failed"}:
            return None
        return build_sync_alignment_registry_record(
            name=entry.get("name") or meta.get("name") or os.path.splitext(os.path.basename(entry.get("meta_path", "")))[0],
            dataset_kind=self._infer_dataset_kind(entry.get("root", ""), meta),
            quality_tier=meta.get("quality_tier"),
            sync_alignment=sync_alignment,
            meta_path=entry.get("meta_path"),
        )

    def _append_sync_alignment_registry_for_entry(self, entry, meta):
        record = self._sync_alignment_registry_record_for_entry(entry, meta)
        if not record:
            return
        append_sync_alignment_registry_record(self.sync_alignment_registry_path, record)
        self.sync_alignment_registry[record["key"]] = record

    @staticmethod
    def _cleanup_lazy_materialization(entry):
        cache_dir = entry.get("cache_dir")
        if cache_dir and os.path.isdir(cache_dir):
            shutil.rmtree(cache_dir, ignore_errors=True)
            return
        for key in ("frames_path", "mel_path"):
            path = entry.get(key)
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass

    def _entry_frame_count(self, entry):
        cached = entry.get("_frame_count")
        if cached is not None:
            return cached
        meta = entry.get("meta") or {}
        if self.skip_bad_samples and is_failed_sync_alignment(meta):
            entry["_frame_count"] = 0
            return 0
        sync_alignment = load_sync_alignment(meta)
        if sync_alignment is not None:
            count = int(sync_alignment.get("valid_frame_count") or 0)
        else:
            count = int(meta.get("n_frames") or meta.get("frames") or 0)
        if count > 0:
            entry["_frame_count"] = count
            return count
        frames = np.load(entry["frames_path"], mmap_mode="r")
        count = int(frames.shape[0])
        entry["_frame_count"] = count
        return count

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
        return max(self._total_frames, len(self.speakers) * 100)

    def _lock_path(self, target_path):
        return target_path + ".lock"

    @staticmethod
    def _lock_owner_alive(lock_path):
        try:
            with open(lock_path, "r", encoding="utf-8") as f:
                first_line = f.readline().strip()
        except OSError:
            return None
        if not first_line:
            return None
        pid_text = first_line.split()[0]
        try:
            pid = int(pid_text)
        except ValueError:
            return None
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        except OSError:
            return None

    def _clear_stale_lock(self, lock_path):
        try:
            age = time.time() - os.path.getmtime(lock_path)
        except FileNotFoundError:
            return False
        if age <= self.materialize_timeout:
            return False
        owner_alive = self._lock_owner_alive(lock_path)
        if owner_alive is True:
            return False
        try:
            os.remove(lock_path)
            print(
                f"[Dataset] Removed stale lock after {age:.1f}s: {lock_path}",
                flush=True,
            )
            return True
        except FileNotFoundError:
            return False

    def _acquire_lock(self, lock_path):
        started = time.time()
        while True:
            try:
                fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                try:
                    payload = f"{os.getpid()} {time.time():.6f}\n".encode("utf-8")
                    os.write(fd, payload)
                except OSError:
                    pass
                os.close(fd)
                return
            except FileExistsError:
                if self._clear_stale_lock(lock_path):
                    continue
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

    def _ensure_sync_alignment_for_lazy_entries(self):
        missing = []
        for key in self.speakers:
            entry = self._entries.get(key)
            if not entry or entry.get("type") != "lazy":
                continue
            meta = self._load_meta(entry)
            sync_alignment = meta.get("sync_alignment") if isinstance(meta, dict) else None
            if load_sync_alignment(meta) is None:
                if (
                    self.skip_bad_samples
                    and is_failed_sync_alignment(meta)
                ):
                    entry["_frame_count"] = 0
                    self._cleanup_lazy_materialization(entry)
                    continue
                missing.append(entry)

        if not missing:
            return

        if not self.sync_alignment_syncnet_checkpoint or not os.path.exists(self.sync_alignment_syncnet_checkpoint):
            raise FileNotFoundError(
                "Sync alignment is enabled but the official SyncNet checkpoint is missing: "
                f"{self.sync_alignment_syncnet_checkpoint}"
            )

        print(
            f"[Dataset] Sync alignment: computing {len(missing)} missing lazy manifests "
            f"using {self.sync_alignment_syncnet_checkpoint}",
            flush=True,
        )
        for idx, entry in enumerate(missing, start=1):
            meta = self._ensure_entry_sync_alignment(entry)
            print(
                f"[Dataset] Sync alignment [{idx}/{len(missing)}] {entry['name']}"
                f"{_format_sync_alignment_progress(meta)}",
                flush=True,
            )

    def _ensure_entry_sync_alignment(self, entry):
        meta = self._load_meta(entry)
        if load_sync_alignment(meta) is not None:
            return meta
        sync_alignment = meta.get("sync_alignment") if isinstance(meta, dict) else None
        if (
            self.skip_bad_samples
            and is_failed_sync_alignment(meta)
        ):
            entry["_frame_count"] = 0
            self._cleanup_lazy_materialization(entry)
            return meta

        if entry.get("type") != "lazy":
            return meta

        if (
            not self.sync_alignment_syncnet_checkpoint
            or not os.path.exists(self.sync_alignment_syncnet_checkpoint)
        ):
            raise FileNotFoundError(
                "Sync alignment checkpoint not found: "
                f"{self.sync_alignment_syncnet_checkpoint}"
            )

        self._materialize_lazy_entry(entry)

        lock_path = self._lock_path(entry["meta_path"] + ".sync")
        self._acquire_lock(lock_path)
        try:
            meta = _load_json(entry["meta_path"])
            entry["meta"] = meta
            if load_sync_alignment(meta) is not None:
                return meta
            if self.skip_bad_samples and is_failed_sync_alignment(meta):
                entry["_frame_count"] = 0
                self._cleanup_lazy_materialization(entry)
                return meta

            frames = np.load(entry["frames_path"], mmap_mode="r")
            mel = np.load(entry["mel_path"])
            source_fps = float(meta.get("fps", self.fps) or self.fps)
            frames = self._normalize_frames_to_target_fps(frames, source_fps)
            try:
                result = compute_sync_alignment_from_faceclip(
                    frames=frames,
                    mel=mel,
                    fps=self.fps,
                    mel_frames_per_second=self.mel_frames_per_second,
                    mel_step_size=self.mel_step_size,
                    syncnet_T=self.syncnet_T,
                    checkpoint_path=self.sync_alignment_syncnet_checkpoint,
                    device=self.sync_alignment_device,
                    search_mel_ticks=self.sync_alignment_search_mel_ticks,
                    search_guard_mel_ticks=self.sync_alignment_guard_mel_ticks,
                    samples=self.sync_alignment_samples,
                    sample_density_per_5s=self.sync_alignment_sample_density_per_5s,
                    seed=self.sync_alignment_seed,
                    min_start_gap_ratio=self.sync_alignment_min_start_gap_ratio,
                    start_gap_multiple=self.sync_alignment_start_gap_multiple,
                    batch_size=self.sync_alignment_batch_size,
                    outlier_trim_ratio=self.sync_alignment_outlier_trim_ratio,
                    min_consensus_ratio=self.sync_alignment_min_consensus_ratio,
                    max_shift_mad=self.sync_alignment_max_shift_mad,
                )
            except FileNotFoundError:
                raise
            except Exception as exc:
                if not self.skip_bad_samples:
                    raise
                failed_meta = write_failed_sync_alignment_to_meta_path(
                    entry["meta_path"],
                    meta,
                    n_frames=len(frames),
                    mel_total_steps=int(mel.shape[1]),
                    fps=self.fps,
                    mel_frames_per_second=self.mel_frames_per_second,
                    mel_step_size=self.mel_step_size,
                    search_guard_mel_ticks=self.sync_alignment_guard_mel_ticks,
                    source="computed_on_demand",
                    reason="compute_failed",
                    error=str(exc),
                )
                entry["meta"] = failed_meta
                entry["_frame_count"] = 0
                self._append_sync_alignment_registry_for_entry(entry, failed_meta)
                self._cleanup_lazy_materialization(entry)
                return failed_meta

            extra = {
                "candidate_audio_shift_mel_ticks": result.get("audio_shift_mel_ticks"),
                "candidate_best_mean_loss": result.get("best_mean_loss"),
                "candidate_zero_mean_loss": result.get("zero_mean_loss"),
                "compute_device": result["device"],
                "starts": result["starts"],
                "kept_starts": result.get("kept_starts", []),
                "dropped_starts": result.get("dropped_starts", []),
                "local_best_shifts": result.get("local_best_shifts", []),
                "kept_local_best_shifts": result.get("kept_local_best_shifts", []),
                "local_best_shift_center": result.get("local_best_shift_center"),
                "post_trim_shift_center": result.get("post_trim_shift_center"),
                "consensus_ratio": result.get("consensus_ratio"),
                "shift_mad": result.get("shift_mad"),
                "weak_sync_signal": result.get("weak_sync_signal"),
                "weak_sync_signal_reasons": result.get("weak_sync_signal_reasons", []),
                "min_consensus_ratio": result.get("min_consensus_ratio"),
                "max_shift_mad": result.get("max_shift_mad"),
                "num_points_before_trim": result.get("num_points_before_trim"),
                "num_points_after_trim": result.get("num_points_after_trim"),
                "outlier_trim_ratio": result.get("outlier_trim_ratio"),
                "sample_density_per_5s": result.get("sample_density_per_5s"),
            }
            if self.skip_bad_samples and bool(result.get("weak_sync_signal")):
                failed_meta = write_failed_sync_alignment_to_meta_path(
                    entry["meta_path"],
                    meta,
                    n_frames=len(frames),
                    mel_total_steps=int(mel.shape[1]),
                    fps=self.fps,
                    mel_frames_per_second=self.mel_frames_per_second,
                    mel_step_size=self.mel_step_size,
                    search_guard_mel_ticks=self.sync_alignment_guard_mel_ticks,
                    source="computed_on_demand",
                    reason="weak_sync_signal",
                    error=";".join(result.get("weak_sync_signal_reasons", [])) or None,
                    extra=extra,
                )
                entry["meta"] = failed_meta
                entry["_frame_count"] = 0
                self._append_sync_alignment_registry_for_entry(entry, failed_meta)
                self._cleanup_lazy_materialization(entry)
                return failed_meta
            updated_meta = upsert_sync_alignment(
                meta,
                audio_shift_mel_ticks=result["audio_shift_mel_ticks"],
                n_frames=len(frames),
                mel_total_steps=int(mel.shape[1]),
                fps=self.fps,
                mel_frames_per_second=self.mel_frames_per_second,
                mel_step_size=self.mel_step_size,
                search_guard_mel_ticks=self.sync_alignment_guard_mel_ticks,
                source="computed_on_demand",
                search_range_mel_ticks=self.sync_alignment_search_mel_ticks,
                search_samples=result["samples"],
                search_seed=self.sync_alignment_seed,
                min_start_gap_ratio=self.sync_alignment_min_start_gap_ratio,
                start_gap_multiple=self.sync_alignment_start_gap_multiple,
                best_mean_loss=result["best_mean_loss"],
                zero_mean_loss=result["zero_mean_loss"],
                extra=extra,
            )
            if self.sync_alignment_write_manifest:
                updated_meta = write_sync_alignment_to_meta_path(
                    entry["meta_path"],
                    meta,
                    audio_shift_mel_ticks=result["audio_shift_mel_ticks"],
                    n_frames=len(frames),
                    mel_total_steps=int(mel.shape[1]),
                    fps=self.fps,
                    mel_frames_per_second=self.mel_frames_per_second,
                    mel_step_size=self.mel_step_size,
                    search_guard_mel_ticks=self.sync_alignment_guard_mel_ticks,
                    source="computed_on_demand",
                    search_range_mel_ticks=self.sync_alignment_search_mel_ticks,
                    search_samples=result["samples"],
                    search_seed=self.sync_alignment_seed,
                    min_start_gap_ratio=self.sync_alignment_min_start_gap_ratio,
                    start_gap_multiple=self.sync_alignment_start_gap_multiple,
                    best_mean_loss=result["best_mean_loss"],
                    zero_mean_loss=result["zero_mean_loss"],
                    extra=extra,
                )
            entry["meta"] = updated_meta
            entry["_frame_count"] = int(
                updated_meta["sync_alignment"].get("valid_frame_count") or 0
            )
            self._append_sync_alignment_registry_for_entry(entry, updated_meta)
            return updated_meta
        finally:
            self._release_lock(lock_path)

    def _apply_sync_alignment(self, entry, frames, mel, meta):
        sync_alignment = load_sync_alignment(meta)
        if sync_alignment is None:
            chunks = self._build_frame_aligned_mels(mel, len(frames))
            return frames, chunks, meta

        audio_shift_mel_ticks = int(sync_alignment.get("audio_shift_mel_ticks") or 0)
        chunks, valid_indices = build_shifted_frame_aligned_mels(
            mel,
            n_frames=len(frames),
            fps=self.fps,
            mel_frames_per_second=self.mel_frames_per_second,
            mel_step_size=self.mel_step_size,
            audio_shift_mel_ticks=audio_shift_mel_ticks,
        )
        if not valid_indices:
            return frames[:0], [], meta

        valid_start = valid_indices[0]
        valid_end = valid_indices[-1] + 1
        if len(valid_indices) == (valid_end - valid_start):
            aligned_frames = frames[valid_start:valid_end]
        else:
            aligned_frames = frames[np.asarray(valid_indices, dtype=np.int64)]

        if self.sync_alignment_write_manifest:
            recorded_start = int(sync_alignment.get("valid_frame_start", valid_start))
            recorded_end = int(sync_alignment.get("valid_frame_end", valid_end - 1))
            recorded_count = int(sync_alignment.get("valid_frame_count", len(valid_indices)))
            if (
                recorded_start != valid_start
                or recorded_end != (valid_end - 1)
                or recorded_count != len(valid_indices)
            ):
                updated_meta = write_sync_alignment_to_meta_path(
                    entry["meta_path"],
                    meta,
                    audio_shift_mel_ticks=audio_shift_mel_ticks,
                    n_frames=len(frames),
                    mel_total_steps=int(mel.shape[1]),
                    fps=self.fps,
                    mel_frames_per_second=self.mel_frames_per_second,
                    mel_step_size=self.mel_step_size,
                    search_guard_mel_ticks=int(
                        sync_alignment.get(
                            "search_guard_mel_ticks",
                            self.sync_alignment_guard_mel_ticks,
                        )
                    ),
                    source=str(sync_alignment.get("source", "manifest_refresh")),
                    search_range_mel_ticks=sync_alignment.get("search_range_mel_ticks"),
                    search_samples=sync_alignment.get("search_samples"),
                    search_seed=sync_alignment.get("search_seed"),
                    min_start_gap_ratio=sync_alignment.get("min_start_gap_ratio"),
                    start_gap_multiple=sync_alignment.get("start_gap_multiple"),
                    best_mean_loss=sync_alignment.get("best_mean_loss"),
                    zero_mean_loss=sync_alignment.get("zero_mean_loss"),
                    extra={
                        key: value
                        for key, value in sync_alignment.items()
                        if key
                        not in {
                            "version",
                            "status",
                            "source",
                            "audio_shift_mel_ticks",
                            "fps",
                            "mel_frames_per_second",
                            "mel_step_size",
                            "mel_total_steps",
                            "n_frames_total",
                            "valid_frame_start",
                            "valid_frame_end",
                            "valid_frame_count",
                            "search_guard_mel_ticks",
                            "computed_at",
                            "search_range_mel_ticks",
                            "search_samples",
                            "search_seed",
                            "min_start_gap_ratio",
                            "start_gap_multiple",
                            "best_mean_loss",
                            "zero_mean_loss",
                        }
                    },
                )
                entry["meta"] = updated_meta
                meta = updated_meta

        return aligned_frames, chunks, meta

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

        if self.sync_alignment_enabled and entry["type"] == "lazy":
            self._ensure_entry_sync_alignment(entry)

        meta = self._load_meta(entry)
        if (
            self.sync_alignment_enabled
            and self.skip_bad_samples
            and is_failed_sync_alignment(meta)
        ):
            self._cleanup_lazy_materialization(entry)
            reason = meta.get("sync_alignment", {}).get("reason", "failed")
            error = meta.get("sync_alignment", {}).get("error", "")
            detail = f"{reason}: {error}".strip(": ")
            raise RuntimeError(f"sync alignment failed for {entry['name']} ({detail})")

        if entry["type"] == "lazy":
            self._materialize_lazy_entry(entry)

        frames = np.load(entry["frames_path"], mmap_mode="r")
        mel = np.load(entry["mel_path"])
        meta = self._load_meta(entry)
        sync_alignment = meta.get("sync_alignment") if isinstance(meta, dict) else None
        if (
            self.sync_alignment_enabled
            and self.skip_bad_samples
            and isinstance(sync_alignment, dict)
            and sync_alignment.get("status") == "failed"
        ):
            reason = sync_alignment.get("reason", "failed")
            error = sync_alignment.get("error", "")
            detail = f"{reason}: {error}".strip(": ")
            raise RuntimeError(f"sync alignment failed for {entry['name']} ({detail})")

        source_fps = float(meta.get("fps", self.fps) or self.fps)
        frames = self._normalize_frames_to_target_fps(frames, source_fps)
        frames, chunks, meta = self._apply_sync_alignment(entry, frames, mel, meta)

        data = (frames, chunks, meta)
        self._cache[cache_key] = data
        while self.cache_size and len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return data

    def __getitem__(self, idx):
        if self.mode == "syncnet" and self.syncnet_style == "mirror":
            max_attempts = max(16, min(len(self.speakers), 128))
            for _ in range(max_attempts):
                sp_idx = random.choices(range(len(self.speakers)), weights=self._frame_counts, k=1)[0]
                speaker_key = self.speakers[sp_idx]
                try:
                    frames, mel_chunks, _ = self._load_speaker(speaker_key)
                    sample = self._get_syncnet_sample(frames, mel_chunks)
                    if sample is not None:
                        return sample
                except Exception as e:
                    name = self._entries.get(speaker_key, {}).get("name", str(speaker_key))
                    print(f"[Dataset] ERROR loading {name}: {e}")
                    continue
            raise RuntimeError("SyncNetMirror could not produce a valid sample after retries")

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
                return self._empty_syncnet_sample()
            return (
                torch.zeros(6, T, S, S),
                torch.zeros(T, 1, 80, self.mel_step_size),
                torch.zeros(1, 80, self.mel_step_size),
                torch.zeros(3, T, S, S),
            )

        if self.mode == "syncnet":
            return self._get_syncnet_sample(frames, mel_chunks)
        return self._get_generator_sample(frames, mel_chunks)

    def _empty_syncnet_sample(self):
        S = self.img_size
        T = self.syncnet_T
        if self.syncnet_style == "mirror":
            audio = torch.zeros(1, 80, self.mel_step_size)
        else:
            audio = torch.zeros(1, 80, T * self.mel_step_size)
        return (
            torch.zeros(T * 3, S // 2, S),
            audio,
            torch.tensor(1.0),
        )

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

        # Match the reference Wav2Lip generator path:
        # - indiv_mels are offset by -1 frame across the 5-frame window
        # - mel_input for the sync loss is anchored at the window start
        mel_indices = [min(max(start - 1 + t, 0), len(mel_chunks) - 1) for t in range(T)]
        indiv_mels = np.stack([mel_chunks[idx] for idx in mel_indices], axis=0)[:, np.newaxis]
        mel_input = mel_chunks[min(start, len(mel_chunks) - 1)][np.newaxis]

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
        if self.syncnet_style == "mirror":
            return self._get_syncnet_mirror_sample(frames, mel_chunks)

        S = self.img_size
        T = self.syncnet_T
        n_frames = min(len(frames), len(mel_chunks))

        if n_frames < T + 5:
            return self._empty_syncnet_sample()

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

    def _get_syncnet_mirror_sample(self, frames, mel_chunks):
        S = self.img_size
        T = self.syncnet_T
        n_frames = min(len(frames), len(mel_chunks))

        if n_frames <= 3 * T:
            return None

        start = random.randint(0, n_frames - T)
        wrong_start = random.randint(0, n_frames - T)
        while wrong_start == start:
            wrong_start = random.randint(0, n_frames - T)

        is_sync = random.random() > 0.5
        chosen_start = start if is_sync else wrong_start

        visual_frames = []
        for t in range(T):
            face = frames[chosen_start + t]
            if face.shape[0] != S or face.shape[1] != S:
                import cv2
                face = cv2.resize(face, (S, S))
            lower = face[S // 2:, :, :]
            visual_frames.append(lower.astype(np.float32) / 255.0)

        visual = np.concatenate([f.transpose(2, 0, 1) for f in visual_frames], axis=0)
        audio = mel_chunks[min(start, len(mel_chunks) - 1)][np.newaxis]
        return (
            torch.from_numpy(np.ascontiguousarray(visual)),
            torch.from_numpy(np.ascontiguousarray(audio)),
            torch.tensor(1.0 if is_sync else 0.0),
        )
