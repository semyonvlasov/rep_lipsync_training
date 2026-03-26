#!/usr/bin/env python3
"""
Calibrate sync-loss scales between two SyncNet teachers on the same GT visuals.

This script measures average sync loss / cosine similarity across a small held-out
subset for several controlled audio desync severities:
  - exact match
  - same-video mel shifted by N frames
  - foreign audio

It is useful when an "official" heuristic threshold (for example 0.75) is
transplanted onto a different teacher whose loss scale may differ.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import torch
import yaml

TRAINING_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(TRAINING_ROOT)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from training.data.dataset import LipSyncDataset
from training.scripts.dataset_roots import get_dataset_roots
from training.scripts.train_generator import (
    load_syncnet_teacher,
    official_sync_loss_from_cosine,
    sync_cosine_score,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Calibrate SyncNet loss scales")
    parser.add_argument("--config", required=True)
    parser.add_argument("--speaker-list", required=True)
    parser.add_argument("--local-syncnet", required=True)
    parser.add_argument("--official-syncnet", required=True)
    parser.add_argument(
        "--shift-frames",
        default="0,1,2,4,8,12,16",
        help="Comma-separated same-video audio shifts in frame units",
    )
    parser.add_argument("--max-items", type=int, default=64)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    return parser.parse_args()


def load_allowlist(path: str) -> List[str]:
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def build_dataset(cfg: dict, speaker_allowlist: List[str]) -> LipSyncDataset:
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


def choose_sample_specs(dataset: LipSyncDataset, max_items: int, max_shift: int) -> List[dict]:
    specs: List[dict] = []
    total = min(len(dataset.speakers), max_items)
    for idx in range(total):
        speaker_key = dataset.speakers[idx]
        frames, mel_chunks, _ = dataset._load_speaker(speaker_key)
        n_frames = min(len(frames), len(mel_chunks))
        if n_frames < (dataset.syncnet_T + max_shift + 2):
            continue

        # Pick a start that leaves room for the largest tested shift.
        max_start = n_frames - dataset.syncnet_T - max_shift - 1
        if max_start < 0:
            continue
        start = max_start // 2

        foreign_idx = (idx + 1) % len(dataset.speakers)
        if foreign_idx == idx:
            continue
        foreign_key = dataset.speakers[foreign_idx]
        foreign_frames, foreign_mel_chunks, _ = dataset._load_speaker(foreign_key)
        foreign_n_frames = min(len(foreign_frames), len(foreign_mel_chunks))
        if foreign_n_frames < (dataset.syncnet_T + 2):
            continue
        foreign_start = max(0, (foreign_n_frames - dataset.syncnet_T - 1) // 2)

        specs.append(
            {
                "speaker_key": speaker_key,
                "start": start,
                "foreign_key": foreign_key,
                "foreign_start": foreign_start,
            }
        )
    return specs


def build_visual_window(dataset: LipSyncDataset, frames: List[np.ndarray], start: int) -> torch.Tensor:
    T = dataset.syncnet_T
    window = [frames[start + t] for t in range(T)]
    window = dataset._resize_window_if_needed(window)
    window_f = dataset._window_to_float(window)
    chw = dataset._window_to_chw(window_f)
    return torch.from_numpy(np.ascontiguousarray(chw)).unsqueeze(0)


def build_audio_inputs(dataset: LipSyncDataset, mel_chunks: np.ndarray, start: int) -> Tuple[torch.Tensor, torch.Tensor]:
    T = dataset.syncnet_T
    mel_input = mel_chunks[min(start, len(mel_chunks) - 1)][np.newaxis]
    mel_indices = [min(max(start - 1 + t, 0), len(mel_chunks) - 1) for t in range(T)]
    indiv_mels = np.stack([mel_chunks[idx] for idx in mel_indices], axis=0)[:, np.newaxis]
    return (
        torch.from_numpy(np.ascontiguousarray(mel_input)).unsqueeze(0),
        torch.from_numpy(np.ascontiguousarray(indiv_mels)).unsqueeze(0),
    )


def scenario_label(shift: int) -> str:
    return "good_sync" if shift == 0 else f"audio_shift_{shift}"


def main():
    args = parse_args()
    cfg = yaml.safe_load(open(args.config))
    allowlist = load_allowlist(args.speaker_list)
    shift_frames = sorted({int(x.strip()) for x in args.shift_frames.split(",") if x.strip()})
    max_shift = max(shift_frames)

    dataset = build_dataset(cfg, allowlist)
    specs = choose_sample_specs(dataset, args.max_items, max_shift)

    teachers = {}
    for name, path in (
        ("official", args.official_syncnet),
        ("local", args.local_syncnet),
    ):
        teacher, kind, epoch = load_syncnet_teacher(path, args.device, cfg["syncnet"]["T"])
        teachers[name] = (teacher, kind)

    metrics: Dict[str, Dict[str, Dict[str, float]]] = {
        name: {
            scenario_label(shift): {"losses": [], "cos": []}
            for shift in shift_frames
        }
        for name in teachers
    }
    for name in teachers:
        metrics[name]["foreign_audio"] = {"losses": [], "cos": []}

    with torch.no_grad():
        for spec in specs:
            frames, mel_chunks, _ = dataset._load_speaker(spec["speaker_key"])
            foreign_frames, foreign_mel_chunks, _ = dataset._load_speaker(spec["foreign_key"])

            face = build_visual_window(dataset, frames, spec["start"]).to(args.device)

            for shift in shift_frames:
                mel, indiv_mels = build_audio_inputs(dataset, mel_chunks, spec["start"] + shift)
                mel = mel.to(args.device)
                indiv_mels = indiv_mels.to(args.device)
                label = scenario_label(shift)
                for teacher_name, (teacher, teacher_kind) in teachers.items():
                    cos = sync_cosine_score(teacher, teacher_kind, mel, indiv_mels, face)
                    loss = official_sync_loss_from_cosine(cos)
                    metrics[teacher_name][label]["losses"].append(float(loss.item()))
                    metrics[teacher_name][label]["cos"].append(float(cos.mean().item()))

            foreign_mel, foreign_indiv_mels = build_audio_inputs(
                dataset, foreign_mel_chunks, spec["foreign_start"]
            )
            foreign_mel = foreign_mel.to(args.device)
            foreign_indiv_mels = foreign_indiv_mels.to(args.device)
            for teacher_name, (teacher, teacher_kind) in teachers.items():
                cos = sync_cosine_score(teacher, teacher_kind, foreign_mel, foreign_indiv_mels, face)
                loss = official_sync_loss_from_cosine(cos)
                metrics[teacher_name]["foreign_audio"]["losses"].append(float(loss.item()))
                metrics[teacher_name]["foreign_audio"]["cos"].append(float(cos.mean().item()))

    summary = {
        "subset_items": len(specs),
        "shift_frames": shift_frames,
        "teachers": {},
    }
    for teacher_name, teacher_metrics in metrics.items():
        summary["teachers"][teacher_name] = {}
        for scenario_name, values in teacher_metrics.items():
            summary["teachers"][teacher_name][scenario_name] = {
                "avg_sync_loss": (
                    sum(values["losses"]) / len(values["losses"]) if values["losses"] else None
                ),
                "avg_cos": sum(values["cos"]) / len(values["cos"]) if values["cos"] else None,
                "count": len(values["losses"]),
            }

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
