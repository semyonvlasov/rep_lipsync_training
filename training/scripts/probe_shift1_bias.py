#!/usr/bin/env python3
"""
Probe whether a +1 audio-frame shift consistently scores better than "good_sync".

This script builds repeated small random batches separately for HDTF and TalkVid,
then compares `good_sync` vs `audio_shift_1` on both the official and local
SyncNet teachers. It is intended to answer whether the observed shift+1 advantage
is systematic or just sampling noise.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import random
import sys
from collections import defaultdict
from statistics import mean, median

import numpy as np
import torch
import yaml

TRAINING_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(TRAINING_ROOT)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
from config_loader import load_config

from training.data.dataset import LipSyncDataset
from training.scripts.dataset_roots import get_dataset_roots
from training.scripts.train_generator import (
    load_syncnet_teacher,
    official_sync_loss_from_cosine,
    sync_cosine_score,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Probe shift+1 bias across providers")
    parser.add_argument("--config", required=True)
    parser.add_argument("--speaker-list", required=True)
    parser.add_argument("--local-syncnet", required=True)
    parser.add_argument("--official-syncnet", required=True)
    parser.add_argument("--device", default="cuda", choices=("cpu", "cuda"))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--batches-per-provider", type=int, default=10)
    parser.add_argument("--shift-frame", type=int, default=1)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--cached-only", action="store_true")
    return parser.parse_args()


def load_allowlist(path: str):
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def build_dataset(cfg: dict, speaker_allowlist):
    roots = get_dataset_roots(cfg)
    with contextlib.redirect_stdout(io.StringIO()):
        dataset = LipSyncDataset(
            roots=roots,
            img_size=cfg["model"]["img_size"],
            mel_step_size=cfg["model"]["mel_steps"],
            fps=cfg["data"]["fps"],
            audio_cfg=cfg["audio"],
            syncnet_T=cfg["syncnet"]["T"],
            mode="generator",
            cache_size=0,
            skip_bad_samples=cfg["data"].get("skip_bad_samples", True),
            speaker_allowlist=speaker_allowlist,
            lazy_cache_root=cfg["data"].get("lazy_cache_root"),
            ffmpeg_bin=cfg["data"].get("ffmpeg_bin", "ffmpeg"),
            materialize_timeout=cfg["data"].get("materialize_timeout", 600),
            materialize_frames_size=cfg["data"].get(
                "materialize_frames_size", cfg["model"]["img_size"]
            ),
        )
    return dataset


def entry_has_materialized_cache(dataset: LipSyncDataset, speaker_key) -> bool:
    entry = dataset._entries.get(speaker_key)
    if entry is None:
        return False
    return os.path.exists(entry["frames_path"]) and os.path.exists(entry["mel_path"])


def classify_provider(entry: dict) -> str | None:
    root = os.path.abspath(entry.get("root", "")).lower()
    if "/hdtf/" in root:
        return "hdtf"
    if "/talkvid/" in root:
        return "talkvid"
    return None


def build_visual_window(dataset: LipSyncDataset, frames, start: int) -> torch.Tensor:
    T = dataset.syncnet_T
    window = [frames[start + t] for t in range(T)]
    window = dataset._resize_window_if_needed(window)
    window_f = dataset._window_to_float(window)
    chw = dataset._window_to_chw(window_f)
    return torch.from_numpy(np.ascontiguousarray(chw)).unsqueeze(0)


def build_audio_inputs(dataset: LipSyncDataset, mel_chunks, start: int):
    T = dataset.syncnet_T
    mel_input = mel_chunks[min(start, len(mel_chunks) - 1)][np.newaxis]
    mel_indices = [min(max(start - 1 + t, 0), len(mel_chunks) - 1) for t in range(T)]
    indiv_mels = np.stack([mel_chunks[idx] for idx in mel_indices], axis=0)[:, np.newaxis]
    return (
        torch.from_numpy(np.ascontiguousarray(mel_input)).unsqueeze(0),
        torch.from_numpy(np.ascontiguousarray(indiv_mels)).unsqueeze(0),
    )


def build_candidate_specs(dataset: LipSyncDataset, shift_frame: int, cached_only: bool):
    candidates = defaultdict(list)
    for speaker_key in dataset.speakers:
        if cached_only and not entry_has_materialized_cache(dataset, speaker_key):
            continue
        entry = dataset._entries[speaker_key]
        provider = classify_provider(entry)
        if provider is None:
            continue

        meta = entry.get("meta") or {}
        n_frames = int(meta.get("n_frames") or meta.get("frames") or 0)
        if n_frames < (dataset.syncnet_T + shift_frame + 2):
            continue

        max_start = n_frames - dataset.syncnet_T - shift_frame - 1
        if max_start < 0:
            continue

        candidates[provider].append(
            {
                "speaker_key": speaker_key,
                "speaker_name": entry["name"],
                "start": max_start // 2,
            }
        )
    return candidates


def sample_batch(rng: random.Random, specs, batch_size: int):
    if len(specs) >= batch_size:
        return rng.sample(specs, batch_size)
    return [rng.choice(specs) for _ in range(batch_size)]


def summarize_batch_metrics(batch_metrics):
    deltas = [row["delta_sync_loss"] for row in batch_metrics]
    shift_better = sum(1 for row in batch_metrics if row["shift1_better"])
    good_better = sum(1 for row in batch_metrics if not row["shift1_better"])
    return {
        "num_batches": len(batch_metrics),
        "shift1_better_batches": shift_better,
        "good_better_batches": good_better,
        "shift1_better_rate": (shift_better / len(batch_metrics)) if batch_metrics else None,
        "mean_delta_sync_loss": mean(deltas) if deltas else None,
        "median_delta_sync_loss": median(deltas) if deltas else None,
        "mean_good_sync_loss": mean([row["good_sync_loss"] for row in batch_metrics]) if batch_metrics else None,
        "mean_shift1_sync_loss": mean([row["shift1_sync_loss"] for row in batch_metrics]) if batch_metrics else None,
        "mean_good_cos": mean([row["good_cos"] for row in batch_metrics]) if batch_metrics else None,
        "mean_shift1_cos": mean([row["shift1_cos"] for row in batch_metrics]) if batch_metrics else None,
    }


def main():
    args = parse_args()
    rng = random.Random(args.seed)
    cfg = load_config(args.config)
    allowlist = load_allowlist(args.speaker_list)
    dataset = build_dataset(cfg, allowlist)

    candidates = build_candidate_specs(dataset, args.shift_frame, args.cached_only)

    teachers = {}
    for name, path in (
        ("official", args.official_syncnet),
        ("local", args.local_syncnet),
    ):
        teacher, kind, _ = load_syncnet_teacher(path, args.device, cfg["syncnet"]["T"])
        teachers[name] = (teacher, kind)

    all_results = {
        "config": args.config,
        "speaker_list": args.speaker_list,
        "device": args.device,
        "cached_only": bool(args.cached_only),
        "batch_size": args.batch_size,
        "batches_per_provider": args.batches_per_provider,
        "shift_frame": args.shift_frame,
        "candidate_counts": {provider: len(specs) for provider, specs in candidates.items()},
        "providers": {},
    }

    with torch.no_grad():
        for provider in ("hdtf", "talkvid"):
            specs = candidates.get(provider, [])
            provider_result = {
                "candidate_count": len(specs),
                "batches": {},
                "teachers": {},
            }
            if not specs:
                all_results["providers"][provider] = provider_result
                continue

            batch_specs_list = []
            for batch_idx in range(args.batches_per_provider):
                batch_specs = sample_batch(rng, specs, args.batch_size)
                batch_specs_list.append(batch_specs)
                provider_result["batches"][str(batch_idx)] = {
                    "speaker_names": [spec["speaker_name"] for spec in batch_specs],
                }

            teacher_metrics = defaultdict(list)

            for batch_idx, batch_specs in enumerate(batch_specs_list):
                batch_rows = defaultdict(list)
                for spec in batch_specs:
                    frames, mel_chunks, _ = dataset._load_speaker(spec["speaker_key"])
                    face = build_visual_window(dataset, frames, spec["start"]).to(args.device)
                    good_mel, good_indiv_mels = build_audio_inputs(dataset, mel_chunks, spec["start"])
                    shift_mel, shift_indiv_mels = build_audio_inputs(
                        dataset, mel_chunks, spec["start"] + args.shift_frame
                    )
                    good_mel = good_mel.to(args.device)
                    good_indiv_mels = good_indiv_mels.to(args.device)
                    shift_mel = shift_mel.to(args.device)
                    shift_indiv_mels = shift_indiv_mels.to(args.device)

                    for teacher_name, (teacher, teacher_kind) in teachers.items():
                        good_cos = sync_cosine_score(
                            teacher, teacher_kind, good_mel, good_indiv_mels, face
                        )
                        shift_cos = sync_cosine_score(
                            teacher, teacher_kind, shift_mel, shift_indiv_mels, face
                        )
                        good_loss = official_sync_loss_from_cosine(good_cos)
                        shift_loss = official_sync_loss_from_cosine(shift_cos)
                        batch_rows[teacher_name].append(
                            {
                                "speaker_name": spec["speaker_name"],
                                "good_sync_loss": float(good_loss.item()),
                                "shift1_sync_loss": float(shift_loss.item()),
                                "good_cos": float(good_cos.mean().item()),
                                "shift1_cos": float(shift_cos.mean().item()),
                            }
                        )

                for teacher_name, rows in batch_rows.items():
                    batch_metric = {
                        "batch_idx": batch_idx,
                        "num_items": len(rows),
                        "good_sync_loss": mean([row["good_sync_loss"] for row in rows]),
                        "shift1_sync_loss": mean([row["shift1_sync_loss"] for row in rows]),
                        "good_cos": mean([row["good_cos"] for row in rows]),
                        "shift1_cos": mean([row["shift1_cos"] for row in rows]),
                    }
                    batch_metric["delta_sync_loss"] = (
                        batch_metric["shift1_sync_loss"] - batch_metric["good_sync_loss"]
                    )
                    batch_metric["shift1_better"] = (
                        batch_metric["shift1_sync_loss"] < batch_metric["good_sync_loss"]
                    )
                    teacher_metrics[teacher_name].append(batch_metric)

            for teacher_name, batch_metrics in teacher_metrics.items():
                provider_result["teachers"][teacher_name] = {
                    "summary": summarize_batch_metrics(batch_metrics),
                    "batch_metrics": batch_metrics,
                }

            all_results["providers"][provider] = provider_result

    print(json.dumps(all_results, indent=2))


if __name__ == "__main__":
    main()
